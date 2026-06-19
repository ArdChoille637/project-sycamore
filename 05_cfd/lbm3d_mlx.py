"""
Project Sycamore — 3-D D3Q19 Lattice-Boltzmann on Apple-Silicon (MLX/Metal).

The keystone for "4-D" (3-D space + time) simulation: a fully GPU-resident,
time-stepping 3-D CFD core. State lives in unified memory and never returns to
the host during a run; timesteps sync adaptively (small grids batch to hide sync
latency, large grids sync per step to bound memory).

Memory: a *fused per-population collision* keeps peak near ~3× the stored state
(no full (19,N,N,N) cu/feq temporaries), so 256³ fits in 16 GB at ~8 GB peak —
the practical ceiling on this machine. Note the collision works in fp32, so fp16
*storage* halves the resident state but not the transient working set; it doesn't
raise the grid ceiling here. (A custom fused Metal kernel would.)

Scheme: D3Q19, BGK collision, periodic streaming (extends the validated 2-D
D2Q9 solver to 3-D). Lattice units (Δx = Δt = 1, c_s² = 1/3).

Validation: the Taylor-Green vortex, whose kinetic energy decays at the analytic
rate 4·ν·k² — the 3-D analogue of the Ghia cavity benchmark (see validate_tgv).

    ~/ds/bin/python 05_cfd/lbm3d_mlx.py            # TGV validation (3 orientations)
    ~/ds/bin/python 05_cfd/lbm3d_mlx.py --bench    # throughput (MLUPS) + memory
"""

import sys
import time
import numpy as np
import mlx.core as mx

# ── D3Q19 lattice ────────────────────────────────────────────────────────────
CX = np.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0], 'i4')
CY = np.array([0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1], 'i4')
CZ = np.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1], 'i4')
W  = np.array([1/3] + [1/18]*6 + [1/36]*12, 'f4')
OPP = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], 'i4')


def _roll3(a, sz, sy, sx):
    """Periodic stream of a (Nz,Ny,Nx) field by integer (sz,sy,sx)."""
    if sz: a = mx.roll(a, sz, axis=0)
    if sy: a = mx.roll(a, sy, axis=1)
    if sx: a = mx.roll(a, sx, axis=2)
    return a


class LBM3D:
    """D3Q19 BGK lattice-Boltzmann, GPU-resident, periodic."""

    def __init__(self, N, nu, store_dtype=mx.float32):
        self.N = N
        self.nu = nu
        self.tau = 3.0*nu + 0.5
        self.omega = 1.0/self.tau
        self.sd = store_dtype
        self.cx = mx.array(CX.astype('f4')).reshape(19, 1, 1, 1)
        self.cy = mx.array(CY.astype('f4')).reshape(19, 1, 1, 1)
        self.cz = mx.array(CZ.astype('f4')).reshape(19, 1, 1, 1)
        self.w  = mx.array(W).reshape(19, 1, 1, 1)
        # nonzero velocity components per axis — for low-memory momentum sums
        self._nzx = [(i, int(CX[i])) for i in range(19) if CX[i]]
        self._nzy = [(i, int(CY[i])) for i in range(19) if CY[i]]
        self._nzz = [(i, int(CZ[i])) for i in range(19) if CZ[i]]
        self.f = None

    def equilibrium(self, rho, ux, uy, uz):
        cu = 3.0*(self.cx*ux + self.cy*uy + self.cz*uz)
        usqr = 1.5*(ux*ux + uy*uy + uz*uz)
        return self.w * rho * (1.0 + cu + 0.5*cu*cu - usqr)

    def macroscopic(self):
        f = self.f.astype(mx.float32)
        rho = mx.sum(f, axis=0)
        ux = mx.sum(f*self.cx, axis=0)/rho
        uy = mx.sum(f*self.cy, axis=0)/rho
        uz = mx.sum(f*self.cz, axis=0)/rho
        return rho, ux, uy, uz

    def init_tgv(self, U=0.04, plane='xy'):
        """Taylor-Green vortex (one wavelength) in a chosen plane; z-uniform along
        the third axis so the analytic 2-D TGV decay applies exactly in 3-D."""
        N = self.N; k = 2*np.pi/N
        idx = np.arange(N)
        Z, Y, X = np.meshgrid(idx, idx, idx, indexing='ij')
        zero = np.zeros((N, N, N), 'f4')
        if plane == 'xy':
            ux = -U*np.cos(k*X)*np.sin(k*Y); uy = U*np.sin(k*X)*np.cos(k*Y); uz = zero
        elif plane == 'xz':
            ux = -U*np.cos(k*X)*np.sin(k*Z); uz = U*np.sin(k*X)*np.cos(k*Z); uy = zero
        elif plane == 'yz':
            uy = -U*np.cos(k*Y)*np.sin(k*Z); uz = U*np.sin(k*Y)*np.cos(k*Z); ux = zero
        else:
            raise ValueError(plane)
        rho = mx.ones((N, N, N))
        self.f = self.equilibrium(rho, mx.array(ux.astype('f4')),
                                  mx.array(uy.astype('f4')), mx.array(uz.astype('f4'))).astype(self.sd)
        mx.eval(self.f)
        self.k = k; self.U = U

    @staticmethod
    def _moment(f, nz):
        """Σ_i c_i f[i] over the nonzero c_i (±1) — avoids a full (19,N,N,N) product."""
        s = None
        for i, c in nz:
            t = f[i] if c == 1 else -f[i]
            s = t if s is None else s + t
        return s

    def step(self):
        """Fused collide+stream: only one (N,N,N) population materialises at a time,
        so peak memory is ~2–3× storage instead of ~5× (no (19,N,N,N) cu/feq arrays)."""
        f = self.f.astype(mx.float32)
        rho = mx.sum(f, axis=0)
        inv = 1.0/rho
        ux = self._moment(f, self._nzx)*inv
        uy = self._moment(f, self._nzy)*inv
        uz = self._moment(f, self._nzz)*inv
        usqr = 1.5*(ux*ux + uy*uy + uz*uz)
        om = self.omega
        new = []
        for i in range(19):
            cu = 3.0*(int(CX[i])*ux + int(CY[i])*uy + int(CZ[i])*uz)
            feq = float(W[i])*rho*(1.0 + cu + 0.5*cu*cu - usqr)
            fcol = f[i] - om*(f[i] - feq)
            new.append(_roll3(fcol, int(CZ[i]), int(CY[i]), int(CX[i])))
        self.f = mx.stack(new, axis=0).astype(self.sd)

    def run(self, n, batch=None):
        """Advance n steps, syncing once per `batch`. Default batch is adaptive:
        small grids batch (hide sync latency), large grids sync every step (bound
        the lazy-graph memory so big grids don't OOM)."""
        if batch is None:
            batch = max(1, min(16, 4_000_000 // self.N**3))
        done = 0
        while done < n:
            for _ in range(min(batch, n - done)):
                self.step()
            mx.eval(self.f)
            done += batch

    def energy(self):
        _, ux, uy, uz = self.macroscopic()
        return float(mx.mean(0.5*(ux*ux + uy*uy + uz*uz)))

    def vorticity_mag(self):
        """|∇×u| as a NumPy field (for slice visualisation)."""
        _, ux, uy, uz = self.macroscopic()
        ux, uy, uz = np.array(ux), np.array(uy), np.array(uz)
        wz = np.gradient(uy, axis=2) - np.gradient(ux, axis=1)
        wy = np.gradient(ux, axis=0) - np.gradient(uz, axis=2)
        wx = np.gradient(uz, axis=1) - np.gradient(uy, axis=0)
        return np.sqrt(wx*wx + wy*wy + wz*wz)


# ── Taylor-Green validation ──────────────────────────────────────────────────

def validate_tgv(N=128, tau=0.6, U=0.04, plane='xy', steps=3000, store=mx.float32,
                 verbose=True):
    nu = (tau - 0.5)/3.0
    sim = LBM3D(N, nu, store)
    sim.init_tgv(U, plane)
    k = 2*np.pi/N
    rate_exact = 4.0*nu*k*k                              # analytic energy decay rate
    ts, Es = [], []
    for s in range(0, steps + 1, 250):
        Es.append(sim.energy()); ts.append(s)
        if s < steps:
            sim.run(250)
    ts = np.array(ts, 'f8'); Es = np.array(Es, 'f8')
    slope = np.polyfit(ts, np.log(Es), 1)[0]
    rate_meas = -slope
    err = abs(rate_meas - rate_exact)/rate_exact*100
    if verbose:
        sdn = 'fp16' if store == mx.float16 else 'fp32'
        print(f"  TGV {plane}  N={N} τ={tau} [{sdn}]: decay rate {rate_meas:.3e} vs "
              f"analytic {rate_exact:.3e}  ({err:.1f}% err)  E:{Es[0]:.2e}→{Es[-1]:.2e}")
    return err


def benchmark():
    """Memory-safe throughput sweep (≤256³ — the 16 GB ceiling)."""
    print(f"{'grid':>7} {'store':>6} {'steps/s':>9} {'MLUPS':>7} {'storage':>8} {'peak':>7}")
    for N, store in [(64, mx.float32), (128, mx.float32), (192, mx.float32),
                     (256, mx.float32), (256, mx.float16)]:
        sim = LBM3D(N, (0.6 - 0.5)/3, store)
        sim.init_tgv(0.04, 'xy')
        sim.run(12)                                     # warmup
        timed = 12 if N >= 256 else (25 if N >= 128 else 40)
        t0 = time.perf_counter(); sim.run(timed); dt = (time.perf_counter()-t0)/timed
        bpc = 2 if store == mx.float16 else 4
        stor = 19*N**3*bpc*2/1e9
        peak = mx.get_peak_memory()/1e9 if hasattr(mx, 'get_peak_memory') else float('nan')
        sdn = 'fp16' if store == mx.float16 else 'fp32'
        print(f"{N}³{'':>2} {sdn:>6} {1.0/dt:9.1f} {N**3/dt/1e6:7.0f} {stor:6.1f}GB {peak:5.1f}GB")
        del sim
        try:
            mx.clear_cache(); mx.reset_peak_memory()
        except Exception:
            pass


if __name__ == '__main__':
    if '--bench' in sys.argv:
        benchmark()
    else:
        print("3-D D3Q19 Taylor-Green vortex validation (analytic decay rate 4νk²):")
        for plane in ('xy', 'xz', 'yz'):
            validate_tgv(N=64, plane=plane)
        print("fp16 storage:")
        validate_tgv(N=64, plane='xy', store=mx.float16)
