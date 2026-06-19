"""
Rotating-reference-frame 3-D LBM — toward the autorotating samara's leading-edge vortex.

A samara autorotates: the wing spins about a vertical axis while the seed descends.
In the co-rotating frame the wing is *stationary* (a solid obstacle), and two inertial
body forces appear:

    Coriolis     a_cor = −2 Ω × u
    centrifugal  a_cen = −Ω × (Ω × r) = Ω² r_perp   (outward)

Rotation drives a spanwise (radially outward) flow along the wing that drains
leading-edge-vortex vorticity and keeps the LEV stable and attached — the mechanism
(Lentink & Dickinson 2009) that a 2-D static section cannot reproduce.

Forces are applied with the Guo (2002) scheme. The Coriolis term depends on u while u
depends on the force (Guo's half-force correction), so the 2×2 in-plane coupling is
solved analytically each step. Built on lbm3d_flow.LBM3DFlow (bounce-back + fused
collision); rotation axis = z, through (x0,y0).

Validation: solid-body rotation — fluid initialised at rest in the frame must stay at
rest (Coriolis does no work; centrifugal is balanced by pressure), developing the
analytic density profile ρ(r) = ρ₀·exp(1.5·Ω²·r²).

    ~/ds/bin/python 05_cfd/lbm3d_rotor.py            # solid-body forcing validation
    ~/ds/bin/python 05_cfd/lbm3d_rotor.py --wing     # revolving-wing LEV demo
"""

import sys
import time
import numpy as np
import mlx.core as mx
from lbm3d_mlx import CX, CY, CZ, W, OPP, _roll3
from lbm3d_flow import LBM3DFlow


class LBM3DRotating(LBM3DFlow):
    """D3Q19 in a frame rotating about z at Ω, with Guo Coriolis+centrifugal forcing."""

    def __init__(self, N, nu, Omega, axis=None, store_dtype=mx.float32):
        super().__init__(N, nu, store_dtype)
        self.Om = float(Omega)
        x0, y0 = axis or (N/2.0, N/2.0)
        zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
        self.rx = mx.array((xx - x0).astype('f4'))         # (N,N,N) distance from axis
        self.ry = mx.array((yy - y0).astype('f4'))
        self._x0, self._y0 = x0, y0

    def step(self):
        Om = self.Om
        f = self.f.astype(mx.float32)
        rho = mx.sum(f, axis=0)
        inv = 1.0/rho
        mxr = self._moment(f, self._nzx)*inv               # bare momentum/ρ
        myr = self._moment(f, self._nzy)*inv
        uz  = self._moment(f, self._nzz)*inv
        # Guo half-force: u = (Σfc + F/2)/ρ, F=ρ(−2Ω×u + Ω²r). In-plane 2×2 closed form:
        mux = mxr + 0.5*Om*Om*self.rx
        muy = myr + 0.5*Om*Om*self.ry
        ux = (mux + Om*muy)/(1.0 + Om*Om)
        uy = muy - Om*ux
        Fx = rho*(2.0*Om*uy + Om*Om*self.rx)
        Fy = rho*(-2.0*Om*ux + Om*Om*self.ry)
        usqr = 1.5*(ux*ux + uy*uy + uz*uz)
        coef = 1.0 - 0.5*self.omega
        solid = mx.array(self.solid)
        om = self.omega
        new = []
        for i in range(19):
            cxi, cyi, czi = int(CX[i]), int(CY[i]), int(CZ[i])
            cu = 3.0*(cxi*ux + cyi*uy + czi*uz)
            feq = float(W[i])*rho*(1.0 + cu + 0.5*cu*cu - usqr)
            Si = coef*float(W[i])*((3.0*(cxi - ux) + 3.0*cu*cxi)*Fx
                                   + (3.0*(cyi - uy) + 3.0*cu*cyi)*Fy)
            fcol = f[i] - om*(f[i] - feq) + Si
            fcol = mx.where(solid, f[int(OPP[i])], fcol)
            new.append(_roll3(fcol, czi, cyi, cxi))
        self.f = mx.stack(new, axis=0).astype(self.sd)

    def frame_velocity(self):
        """Velocity in the rotating frame (includes the Guo half-force correction)."""
        Om = self.Om
        f = self.f.astype(mx.float32)
        rho = mx.sum(f, axis=0); inv = 1.0/rho
        mux = self._moment(f, self._nzx)*inv + 0.5*Om*Om*self.rx
        muy = self._moment(f, self._nzy)*inv + 0.5*Om*Om*self.ry
        ux = (mux + Om*muy)/(1.0 + Om*Om)
        uy = muy - Om*ux
        uz = self._moment(f, self._nzz)*inv
        return np.array(ux), np.array(uy), np.array(uz), np.array(rho)


# ── validation: solid-body rotation ──────────────────────────────────────────

def validate_solidbody(N=64, Omega=0.004, tau=0.8, steps=3000):
    nu = (tau - 0.5)/3.0
    sim = LBM3DRotating(N, nu, Omega)
    sim.init_uniform(0.0)                                  # rest in the rotating frame
    for s in range(0, steps, 200):
        sim.run(200)
    ux, uy, uz, rho = sim.frame_velocity()
    interior = (slice(2, N-2),)*3
    umax = float(np.max(np.hypot(np.hypot(ux, uy), uz)[interior]))
    # centrifugal density profile ρ(r)/ρ(0) = exp(1.5 Ω² r²) along a midplane radial line
    c = N//2
    rline = np.arange(0, N//2 - 2)
    rho_meas = rho[c, c, c:c + len(rline)] / rho[c, c, c]
    rho_pred = np.exp(1.5 * Omega**2 * rline**2)
    rerr = float(np.max(np.abs(rho_meas - rho_pred)) / (rho_pred.max() - 1 + 1e-9)) * 100
    print(f"Solid-body rotation  N={N}, Ω={Omega}, {steps} steps:")
    print(f"  max |u_frame| interior = {umax:.2e}  (must stay ≈0 — fluid at rest in frame)")
    print(f"  ρ(r) centrifugal profile vs exp(1.5Ω²r²): max dev {rerr:.1f}% of the profile span")
    ok = umax < 5e-3 and rerr < 20
    print(f"  {'PASS' if ok else 'FAIL'}  (|u|<5e-3 and ρ-profile within 20%)")
    return ok


# ── revolving-wing LEV demo ──────────────────────────────────────────────────

def revolving_wing(N=64, Omega=0.003, R=0.30, chord=12.0, span=16.0, aoa=28.0,
                   thick=3.0, revolutions=0.30, verbose=True):
    """Thin plate at radius R·N from the z-axis, span radial, at angle of attack, in a
    frame rotating at Ω. Fluid starts at rest in the LAB frame (−Ω×r in the frame), so
    the wing sees a relative inflow at the local rotational speed. Short run = early-time
    LEV formation, before the closed box spins up. Returns fields for visualisation."""
    nu = (0.52 - 0.5)/3.0 + 0.006                          # τ≈0.55, stable
    sim = LBM3DRotating(N, nu, Omega)
    x0, y0, z0 = N/2.0, N/2.0, N/2.0
    Rc = R*N
    a = np.deg2rad(aoa)
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype('f4')
    # plate: centred at (x0+Rc, y0), spanwise along +x (radial), chord along the
    # rotation tangent (y), pitched by aoa about the spanwise axis (tilts into z)
    xs = xx - (x0 + Rc); ys = yy - y0; zs = zz - z0
    chord_dir_y = np.cos(a); chord_dir_z = np.sin(a)
    c_coord = ys*chord_dir_y + zs*chord_dir_z              # along chord
    n_coord = -ys*chord_dir_z + zs*chord_dir_y             # plate normal
    plate = (np.abs(xs) <= span/2) & (np.abs(c_coord) <= chord/2) & (np.abs(n_coord) <= thick/2)
    sim.set_obstacle(plate)
    # IC: fluid at rest in lab frame → u_frame = −Ω×r = (Ω·ry, −Ω·rx, 0)
    rx = xx - x0; ry = yy - y0
    ux0 = Omega*ry; uy0 = -Omega*rx; uz0 = np.zeros_like(ux0)
    rho = mx.ones((N, N, N))
    sim.f = sim.equilibrium(rho, mx.array(ux0), mx.array(uy0), mx.array(uz0)).astype(sim.sd)
    mx.eval(sim.f)

    steps = int(revolutions * 2*np.pi/Omega)
    t0 = time.perf_counter()
    sim.run(steps, batch=1)
    wall = time.perf_counter() - t0
    if verbose:
        tip = Omega*(Rc + span/2)
        print(f"Revolving wing  N={N}, Ω={Omega}, R={Rc:.0f}, chord={chord:.0f}, AoA={aoa:.0f}°")
        print(f"  {steps} steps ({revolutions:.2f} rev) in {wall:.0f}s; tip speed {tip:.3f} (Ma {tip/0.577:.2f})")
    return dict(sim=sim, plate=plate, N=N, x0=x0, y0=y0, z0=z0, Rc=Rc, chord=chord,
                span=span, aoa=aoa, Omega=Omega, steps=steps)


def make_wing_figure(r, out='output/lbm3d_revolving_wing.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sim, N = r['sim'], r['N']
    ux, uy, uz, rho = sim.frame_velocity()
    # spanwise-station chord-plane slice (x = x0+Rc): show the LEV in the chordwise plane
    xc = int(r['x0'] + r['Rc'])
    wy = np.gradient(ux, axis=2) - np.gradient(uz, axis=0)          # ω_y (spanwise vorticity) approx
    vort = np.gradient(uy, axis=0) - np.gradient(ux, axis=1)        # use ω_z as LEV proxy in plane
    # vorticity magnitude in the chord plane (y,z) at this span station
    sl = vort[:, :, xc]                                             # (Nz, Ny)
    pl = r['plate'][:, :, xc]
    sl = np.ma.masked_where(pl, sl)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    lim = np.nanpercentile(np.abs(sl), 99)
    im0 = ax[0].imshow(sl, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
    ax[0].contour(pl, levels=[0.5], colors='k', linewidths=1.5)
    ax[0].set_title(f"Chord-plane vorticity at the wing span station\n(LEV roll-up; AoA={r['aoa']:.0f}°, {r['steps']} steps)")
    ax[0].set(xlabel='y (chord/tangent)', ylabel='z (axial)')
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    # spanwise flow (u along +x = radial) in the wing's suction-side plane — the LEV stabiliser
    zc = int(r['z0'])
    span_u = ux[zc]                                                 # (Ny, Nx): radial velocity
    span_u = np.ma.masked_where(r['plate'][zc], span_u)
    lim2 = np.nanpercentile(np.abs(span_u), 98)
    im1 = ax[1].imshow(span_u, origin='lower', cmap='PuOr', vmin=-lim2, vmax=lim2, aspect='equal')
    ax[1].contour(r['plate'][zc], levels=[0.5], colors='k', linewidths=1.2)
    ax[1].set_title("Spanwise (radial) velocity over the wing\n(outward flow drains the LEV — the 3-D stabiliser)")
    ax[1].set(xlabel='x (radial →)', ylabel='y (tangent)')
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    fig.suptitle("Revolving samara wing in a rotating frame (3-D D3Q19 LBM) — "
                 "the leading-edge vortex the 2-D static section can't sustain", y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    if '--wing' in sys.argv:
        make_wing_figure(revolving_wing())
    else:
        validate_solidbody()
