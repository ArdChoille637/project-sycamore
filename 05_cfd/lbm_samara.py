"""
Samara airfoil section in cross-flow — the leading-edge vortex (LEV) that the
blade-element-momentum model (01_aero/samara_bem.py) cannot represent.

A samara autorotates at high local angle of attack; its lift is dominated by a
leading-edge vortex on the suction side (Lentink et al., Science 2009). BEM
assumes attached strip-theory flow and misses this entirely. Here a thin section
is placed at high AoA in the D2Q9 LBM flow and the vorticity field reveals the LEV.

Honest scope: this is a 2-D, STATIC section. It captures LEV *formation* and
separation, but not the 3-D spanwise-flow stabilisation that keeps a real samara's
LEV attached — that needs a rotating 3-D simulation. Forces are indicative.

Run: ~/ds/bin/python 05_cfd/lbm_samara.py
Output: output/lbm_samara.png
"""

import time
import numpy as np
from lbm_mlx import LBM, HAVE_MLX


def _plate_mask(ny, nx, xLE, yLE, chord, thick, aoa_deg):
    """Thin flat plate (chord `chord`, thickness `thick`) at angle of attack."""
    a = np.deg2rad(aoa_deg)
    tx, ty = np.cos(a), -np.sin(a)                       # chord direction (TE below LE → +AoA)
    yy, xx = np.mgrid[0:ny, 0:nx].astype('f4')
    dx, dy = xx - xLE, yy - yLE
    s = dx*tx + dy*ty                                    # along chord
    n = -dx*ty + dy*tx                                   # perpendicular
    return (s >= 0) & (s <= chord) & (np.abs(n) <= thick/2)


def samara(Re=400.0, chord=44.0, aoa=30.0, U=0.1, nx=460, ny=200, thick=3.0,
           steps=18000, use_mlx=True, verbose=True):
    nu = U*chord/Re
    sim = LBM(ny, nx, nu, use_mlx=use_mlx)
    xLE, yLE = nx*0.28, ny*0.5
    plate = _plate_mask(ny, nx, xLE, yLE, chord, thick, aoa)
    sim.set_obstacle(plate)
    sim.add_walls(top=True, bottom=True)
    sim.set_inlet(U)
    sim.set_outlet()

    fx_t, fy_t = [], []
    t0 = time.perf_counter()
    for it in range(steps):
        sim.step()
        if it >= steps//2:
            Fx, Fy = sim.force_on_solid()
            fx_t.append(Fx); fy_t.append(Fy)
    wall = time.perf_counter() - t0

    q = 0.5*U*U*chord
    Cd = np.mean(fx_t)/q
    Cl = np.mean(fy_t)/q                                 # lift ⟂ flow (here +y); sign per orientation
    print(f"\nSamara section — Re={Re:.0f}, chord={chord:.0f}, AoA={aoa:.0f}°, "
          f"{sim.be.mlx and 'MLX' or 'NumPy'}, {steps} steps in {wall:.1f}s")
    print(f"  indicative  Cl ≈ {Cl:+.2f},  Cd ≈ {Cd:.2f},  L/D ≈ {abs(Cl/Cd):.2f}")
    print("  (2-D static section; absolute coefficients indicative — see module docstring)")
    return dict(sim=sim, plate=plate, Cl=Cl, Cd=Cd, Re=Re, aoa=aoa, chord=chord,
                xLE=xLE, yLE=yLE, U=U, nx=nx, ny=ny)


def make_figure(r, out='output/lbm_samara.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sim = r['sim']
    vort = np.ma.masked_where(r['plate'], sim.vorticity())
    ux, uy = sim.velocity()
    speed = np.ma.masked_where(r['plate'], np.hypot(ux, uy))
    lim = np.nanpercentile(np.abs(vort), 99)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')
    ax[0].imshow(vort, cmap='RdBu_r', origin='lower', vmin=-lim, vmax=lim, aspect='equal')
    ax[0].contour(r['plate'], levels=[0.5], colors='k', linewidths=1.5)
    ax[0].set_title(f"Samara section, Re={r['Re']:.0f}, AoA={r['aoa']:.0f}° — vorticity "
                    f"(leading-edge vortex on the suction side)")
    ax[0].set(xlabel='x [lattice]', ylabel='y')

    ax[1].imshow(speed, cmap='viridis', origin='lower', aspect='equal')
    sk = 7
    yy, xx = np.mgrid[0:r['ny']:sk, 0:r['nx']:sk]
    ax[1].quiver(xx, yy, ux[::sk, ::sk], uy[::sk, ::sk], color='white', scale=3.0,
                 width=0.0018, alpha=0.6)
    ax[1].contour(r['plate'], levels=[0.5], colors='r', linewidths=1.5)
    ax[1].set_title("speed + velocity vectors (separation & recirculation over the section)")
    ax[1].set(xlabel='x [lattice]', ylabel='y')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    make_figure(samara(use_mlx=HAVE_MLX))
