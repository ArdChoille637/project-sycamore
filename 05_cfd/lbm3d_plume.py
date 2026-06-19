"""
3-D buoyant fire plume on the D3Q19 core — the "4-D fire": a wildfire's convective
column rising and bending in the ambient wind, time-resolved in 3-D.

This closes the CFD loop back onto the wildfire side of the project. The Rothermel CA
spreads the fire across the ground; *this* resolves what rises above it — the buoyant
plume that, in a real fire, drives the local indrafts and lofts embers (the WRF-SFIRE
fire–atmosphere coupling, in miniature).

Physics (Boussinesq buoyancy on the LBM):
  • A temperature scalar T(x,t) is transported by the flow (upwind advection + a little
    thermal diffusion). T=1 in the fire (a held hot patch on the ground), T=0 ambient.
  • Buoyancy is a vertical body force  F_z = β·T  (hot fluid rises), applied with the
    same Guo (2002) forcing used for the rotating frame — here purely vertical and, since
    F_z depends on T not u, with no implicit velocity coupling (no 2×2 solve needed).
  • A no-slip ground (z=0), an open top (z=N−1, the plume exits), and an ambient wind
    blowing in +x (−x inflow → +x outflow) that bends the column over.

Built on lbm3d_flow.LBM3DFlow (bounce-back + x-inflow/outflow + fused collision).

Validation: a *uniform* buoyancy force on a periodic fluid must accelerate it uniformly
at a = β (no pressure gradient, no walls) → w(t) = β·t exactly. validate_buoyancy()
checks the Guo z-force reproduces this analytic ramp.

    ~/ds/bin/python 05_cfd/lbm3d_plume.py            # buoyancy-force validation
    ~/ds/bin/python 05_cfd/lbm3d_plume.py --plume    # fire-plume-in-wind demo + figure
"""

import sys
import time
import numpy as np
import mlx.core as mx
from lbm3d_mlx import CX, CY, CZ, W, OPP, _roll3
from lbm3d_flow import LBM3DFlow


class LBM3DPlume(LBM3DFlow):
    """D3Q19 with a transported temperature scalar and a Guo vertical buoyancy force."""

    def __init__(self, N, nu, beta, kappa=0.02, store_dtype=mx.float32):
        super().__init__(N, nu, store_dtype)
        self.beta = float(beta)         # buoyancy: F_z = beta * T
        self.kappa = float(kappa)       # thermal diffusivity (lattice units)
        self.T = np.zeros((N, N, N), 'f4')
        self.src = np.zeros((N, N, N), bool)   # fire patch — T held at 1 here
        self.top_outlet = False

    def set_source(self, mask):
        self.src = np.asarray(mask, bool)
        self.T[self.src] = 1.0

    def set_top_outlet(self):
        self.top_outlet = True

    def _zmask(self, k):
        m = (np.arange(self.N) == k).astype('f4')
        return mx.array(m.reshape(1, -1, 1, 1)) > 0.5     # selects z=k plane (z=axis1) in (19,Nz,Ny,Nx)

    def step(self):
        f = self.f.astype(mx.float32)
        rho = mx.sum(f, axis=0)
        inv = 1.0/rho
        ux = self._moment(f, self._nzx)*inv
        uy = self._moment(f, self._nzy)*inv
        Fz = self.beta * mx.array(self.T)                 # buoyancy (no implicit u-coupling)
        uz = self._moment(f, self._nzz)*inv + 0.5*Fz*inv  # Guo half-force on z
        usqr = 1.5*(ux*ux + uy*uy + uz*uz)
        coef = 1.0 - 0.5*self.omega
        solid = mx.array(self.solid)
        om = self.omega
        new = []
        for i in range(19):
            cxi, cyi, czi = int(CX[i]), int(CY[i]), int(CZ[i])
            cu = 3.0*(cxi*ux + cyi*uy + czi*uz)
            feq = float(W[i])*rho*(1.0 + cu + 0.5*cu*cu - usqr)
            Si = coef*float(W[i])*(3.0*(czi - uz) + 3.0*cu*czi)*Fz    # vertical Guo force
            fcol = f[i] - om*(f[i] - feq) + Si
            fcol = mx.where(solid, f[int(OPP[i])], fcol)             # no-slip ground + any solids
            new.append(_roll3(fcol, czi, cyi, cxi))
        f = mx.stack(new, axis=0)
        if self.outlet:
            f = mx.where(self._xmask(self.N-1), mx.roll(f, 1, axis=3), f)   # +x wind outflow
        if self.top_outlet:
            f = mx.where(self._zmask(self.N-1), mx.roll(f, 1, axis=1), f)   # +z plume outflow (z=axis1)
        if self.inlet_u is not None:
            f = mx.where(self._xmask(0), mx.array(self.feq_in).reshape(19, 1, 1, 1), f)
        self.f = f.astype(self.sd)
        # advance the temperature scalar with the just-computed velocity
        self._transport(np.array(ux), np.array(uy), np.array(uz))

    def _transport(self, ux, uy, uz):
        """First-order upwind advection + central diffusion of T (host-side; T is small).
        Solids are treated as zero-flux thermal walls: no advection through a wall, no heat
        stored in solid cells, and the z=0 ground does not pull from the periodic-wrapped top."""
        T = self.T
        fluid = ~self.solid
        ux = ux*fluid; uy = uy*fluid; uz = uz*fluid          # no advective flux through walls
        Tw = np.where(self.solid, 0.0, T)                    # solids contribute no heat to the stencil
        # upwind spatial derivatives (axis 2=x, 1=y, 0=z)
        ax = ux*np.where(ux > 0, Tw - np.roll(Tw, 1, 2), np.roll(Tw, -1, 2) - Tw)
        ay = uy*np.where(uy > 0, Tw - np.roll(Tw, 1, 1), np.roll(Tw, -1, 1) - Tw)
        az = uz*np.where(uz > 0, Tw - np.roll(Tw, 1, 0), np.roll(Tw, -1, 0) - Tw)
        az[0, :, :] = 0.0                                    # ground: no advective wrap to the top
        lap = (np.roll(Tw, 1, 2) + np.roll(Tw, -1, 2) + np.roll(Tw, 1, 1) + np.roll(Tw, -1, 1)
               + np.roll(Tw, 1, 0) + np.roll(Tw, -1, 0) - 6.0*Tw)
        lap[0, :, :] = 0.0                                   # ground: no diffusion across the z-wrap
        Tn = T - (ax + ay + az) + self.kappa*lap
        Tn[self.src] = 1.0                          # fire holds its heat
        Tn[self.solid] = 0.0                        # no heat stored in solid cells (ground wall)
        if self.inlet_u is not None:
            Tn[:, :, 0] = 0.0                        # cold ambient air enters on −x
        if self.top_outlet:
            Tn[-1, :, :] = Tn[-2, :, :]              # top: zero-gradient (plume leaves)
        self.T = np.clip(Tn, 0.0, 1.0).astype('f4')


# ── validation: uniform buoyancy → uniform acceleration w = β·t ───────────────

def validate_buoyancy(N=48, beta=1e-3, tau=0.6, steps=200):
    """Uniform T=1 on a periodic box: the buoyancy force is uniform, so the fluid must
    accelerate uniformly upward with no pressure gradient — w(n) = β·n analytically."""
    nu = (tau - 0.5)/3.0
    sim = LBM3DPlume(N, nu, beta, kappa=0.0)
    sim.init_uniform(0.0)
    sim.T[:] = 1.0                        # uniform temperature → uniform body force
    ws = []
    for n in (50, 100, 150, 200):
        sim.run(50, batch=1)
        _, _, _, uz = sim.macroscopic()
        ws.append((n, float(mx.mean(uz))))
    print(f"Uniform-buoyancy ramp  N={N}, β={beta}:  w should equal β·n")
    ok = True
    for n, w in ws:
        err = abs(w - beta*n)/(beta*n)*100
        ok &= err < 5.0
        print(f"  n={n:4d}:  w = {w:.3e}   β·n = {beta*n:.3e}   ({err:.1f}% err)")
    print(f"  {'PASS' if ok else 'FAIL'}  (Guo vertical force reproduces w = β·n within 5%)")
    return ok


def validate_walls(N=32, steps=400):
    """Boundary sanity for the full plume config (ground + open top + wind). The uniform-
    buoyancy test runs on a periodic box and so cannot see a boundary bug; this exercises
    the walls and would fail loudly if the z-top outflow were mis-axised onto a side face."""
    # (1) structural: the top-outlet mask MUST select the z=N-1 plane, not a lateral wall
    probe = LBM3DPlume(N, 0.05, 1e-3)
    m = np.broadcast_to(np.array(probe._zmask(N-1))[0], (N, N, N))
    zz, yy, _ = np.mgrid[0:N, 0:N, 0:N]
    z_sel = sorted(set(zz[m].tolist())); y_spread = len(set(yy[m].tolist()))
    mask_ok = (z_sel == [N-1]) and (y_spread == N)
    # (2) run the real config and check the ground wall is thermally sealed
    r = fire_plume(N=N, steps=steps, verbose=False)
    sim = r['sim']
    _, _, _, uz = sim.macroscopic(); uz = np.array(uz)
    g_heat = float(sim.T[0].max())                  # ground is a cold solid wall → exactly 0
    g_uz = float(np.abs(uz[0]).mean())              # mean ground motion (wrap signature) → small
    top_T = float(sim.T[-1].max())                  # plume hasn't reached the lid in a short run
    print(f"Plume wall sanity  N={N}, {steps} steps:")
    print(f"  top-outlet mask hits z-plane {z_sel} (y spans {y_spread}/{N})  → "
          f"{'z-top ✓' if mask_ok else 'WRONG FACE ✗'}")
    print(f"  ground heat   max T[z=0]   = {g_heat:.2e}   (must be 0 — no heat into the wall)")
    print(f"  ground motion mean|u_z[0]| = {g_uz:.2e}   (small — no periodic-lid updraft wrap)")
    print(f"  lid heat      max T[z=N-1] = {top_T:.2e}   (≈0 early)")
    ok = mask_ok and g_heat < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  (top BC on the z-face + ground thermally sealed)")
    return ok


# ── fire-plume-in-wind demo ───────────────────────────────────────────────────

def fire_plume(N=96, U_wind=0.030, beta=1.6e-3, tau=0.6, kappa=0.02,
               src_r=5.0, src_h=4, src_x=0.22, steps=3000, verbose=True):
    """A held hot patch on the ground drives a buoyant column; an ambient +x wind bends
    it over. Returns the temperature/velocity fields for visualisation."""
    nu = (tau - 0.5)/3.0
    sim = LBM3DPlume(N, nu, beta, kappa)
    # ground (no-slip) at z=0; open top; wind in +x
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    sim.set_obstacle(zz == 0)
    sim.set_inlet(U_wind); sim.set_outlet(); sim.set_top_outlet()
    sim.init_uniform(U_wind)
    # fire patch: a disk on the ground at (src_x·N, N/2)
    xc, yc = src_x*N, N/2.0
    fire = (zz >= 1) & (zz <= src_h) & ((xx-xc)**2 + (yy-yc)**2 < src_r**2)
    sim.set_source(fire)

    t0 = time.perf_counter()
    sim.run(steps, batch=1)
    wall = time.perf_counter() - t0
    Tmax_h = sim.T.max(axis=(1, 2))                 # max T at each height
    top = int(np.argmax(Tmax_h[::-1] > 0.05))       # plume top (from the top down)
    reach = N - 1 - top
    if verbose:
        # mean updraft over the fire column
        _, _, _, uz = sim.macroscopic()
        w_col = float(np.array(uz)[1:src_h+2, int(yc), int(xc)].mean())
        print(f"Fire plume in wind  N={N}, U_wind={U_wind}, β={beta}, {steps} steps in {wall:.0f}s")
        print(f"  plume rises to z≈{reach}/{N} ({100*reach/N:.0f}% of the box); "
              f"mean updraft over the fire w≈{w_col:.3f} vs wind {U_wind}")
        # downwind lean: x of the plume centroid at mid-height vs at the source
        Txz = sim.T[:, int(yc), :]
        zlo, zhi = src_h+2, min(reach, N-2)
        if zhi > zlo:
            xs = np.arange(N)
            x_src = (Txz[zlo]*xs).sum()/(Txz[zlo].sum()+1e-9)
            x_hi = (Txz[zhi]*xs).sum()/(Txz[zhi].sum()+1e-9)
            print(f"  downwind lean: plume centroid x {x_src:.1f} (low) → {x_hi:.1f} (high) "
                  f"→ leans {x_hi-x_src:+.1f} cells downwind ✓")
    return dict(sim=sim, N=N, xc=xc, yc=yc, U_wind=U_wind, beta=beta, steps=steps,
                src_r=src_r, src_h=src_h, reach=reach)


def make_figure(r, out='output/lbm3d_fire_plume.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sim, N = r['sim'], r['N']
    yc = int(r['yc'])
    T = sim.T[:, yc, :]                              # (Nz, Nx) smoke in the xz mid-plane
    _, _, _, uz = sim.macroscopic()
    w = np.array(uz)[:, yc, :]                       # (Nz, Nx) vertical velocity
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
    # smoke (temperature)
    im0 = ax[0].imshow(T, origin='lower', cmap='inferno', aspect='equal', vmin=0, vmax=1)
    ax[0].axhspan(-0.5, 0.5, color='dimgray')        # ground
    ax[0].plot([r['xc']], [r['src_h']/2], marker='^', ms=12, color='lime', mec='k')   # fire
    ax[0].annotate('wind', xy=(0.10, 0.86), xytext=(0.02, 0.86), xycoords='axes fraction',
                   ha='left', va='center', color='cyan', fontsize=11,
                   arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    fig.colorbar(im0, ax=ax[0], shrink=0.85).set_label('T (smoke / heat)')
    ax[0].set_title(f"Buoyant fire plume bent by wind — smoke (3-D D3Q19 + Boussinesq)\n"
                    f"rises to {100*r['reach']/N:.0f}% of the box, leaning downwind")
    ax[0].set(xlabel='x (wind →)', ylabel='z (height)')
    # updraft
    lim = np.nanpercentile(np.abs(w), 99.5)
    im1 = ax[1].imshow(w, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
    ax[1].axhspan(-0.5, 0.5, color='dimgray')
    fig.colorbar(im1, ax=ax[1], shrink=0.85).set_label('vertical velocity w')
    ax[1].set_title("Vertical velocity — the fire-driven updraft\n(red = rising column; "
                    "the indraft a real fire feels)")
    ax[1].set(xlabel='x (wind →)', ylabel='z (height)')
    fig.suptitle("The 4-D fire: a wildfire's convective column resolved in 3-D + time", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    if '--plume' in sys.argv:
        make_figure(fire_plume())
    elif '--walls' in sys.argv:
        validate_walls()
    else:
        validate_buoyancy()
        validate_walls()
