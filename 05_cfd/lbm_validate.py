"""
Validate the D2Q9 LBM against the Ghia, Ghia & Shin (1982) lid-driven cavity
benchmark — the standard reference for incompressible Navier-Stokes solvers.

Re = 100. Compares the u-velocity along the vertical centreline and the
v-velocity along the horizontal centreline to the tabulated Ghia values.

Run: ~/ds/bin/python 05_cfd/lbm_validate.py
"""

import time
import numpy as np
from lbm_mlx import LBM, HAVE_MLX

# Ghia et al. (1982), Re=100 — u on the vertical centreline (x=0.5)
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                   0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                   -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                   0.78871, 0.84123, 1.00000])
# v on the horizontal centreline (y=0.5)
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                   0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
GHIA_V = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                   0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
                   -0.07391, -0.05906, 0.00000])


def cavity(Re=100.0, N=128, U=0.1, steps=40000, tol=1e-6, use_mlx=True, verbose=True):
    nu = U*N/Re
    sim = LBM(N, N, nu, use_mlx=use_mlx)
    lid = np.zeros((N, N), bool); lid[-1, :] = True       # top row = moving lid
    sim.add_walls(bottom=True, left=True, right=True)
    sim.set_moving_wall(lid, ux=U)

    t0 = time.perf_counter()
    ux_prev = None
    for it in range(1, steps+1):
        sim.step()
        if it % 1000 == 0:
            ux, _ = sim.velocity()
            if ux_prev is not None:
                d = np.nanmax(np.abs(ux - ux_prev))/U
                if verbose:
                    print(f"  step {it:6d}  Δu/U = {d:.2e}")
                if d < tol:
                    break
            ux_prev = ux.copy()
    wall = time.perf_counter() - t0

    ux, uy = sim.velocity()
    # interior fluid spans rows/cols 1..N-2; normalise coordinates 0..1
    yc = (np.arange(N) - 0.5) / (N - 2)
    xc = (np.arange(N) - 0.5) / (N - 2)
    j_mid = N // 2
    u_centre = ux[:, j_mid] / U
    v_centre = uy[j_mid, :] / U

    u_interp = np.interp(GHIA_Y, yc, u_centre)
    v_interp = np.interp(GHIA_X, xc, v_centre)
    rmse_u = float(np.sqrt(np.mean((u_interp - GHIA_U)**2)))
    rmse_v = float(np.sqrt(np.mean((v_interp - GHIA_V)**2)))

    print(f"\nLid-driven cavity Re={Re:.0f}, {N}×{N}, {sim.be.mlx and 'MLX' or 'NumPy'}, "
          f"{it} steps in {wall:.1f}s")
    print(f"  u-centreline RMSE vs Ghia : {rmse_u:.4f}")
    print(f"  v-centreline RMSE vs Ghia : {rmse_v:.4f}")
    ok = rmse_u < 0.03 and rmse_v < 0.03
    print(f"  {'PASS' if ok else 'FAIL'}  (RMSE < 0.03 of lid speed)")
    return dict(sim=sim, ux=ux, uy=uy, rmse_u=rmse_u, rmse_v=rmse_v,
                u_centre=u_centre, v_centre=v_centre, yc=yc, xc=xc, U=U, N=N, ok=ok)


def make_figure(r, out='output/lbm_cavity.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N = r['N']
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
    spd = np.hypot(r['ux'], r['uy'])/r['U']
    ax[0].imshow(spd, cmap='viridis', origin='lower')
    yy, xx = np.mgrid[0:N:5, 0:N:5]
    ax[0].streamplot(np.arange(N), np.arange(N), r['ux'], r['uy'], density=1.1,
                     color='white', linewidth=0.5, arrowsize=0.6)
    ax[0].set_title(f"Lid-driven cavity Re=100 ({N}×{N})\nspeed + streamlines")
    ax[0].set(xlabel='x', ylabel='y')
    ax[1].plot(r['u_centre'], r['yc'], '-', color='#185FA5', label='LBM')
    ax[1].plot(GHIA_U, GHIA_Y, 'o', color='#A32D2D', ms=5, label='Ghia 1982')
    ax[1].set(xlabel='u / U', ylabel='y', title=f"vertical centreline u (RMSE {r['rmse_u']:.4f})")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(r['xc'], r['v_centre'], '-', color='#185FA5', label='LBM')
    ax[2].plot(GHIA_X, GHIA_V, 'o', color='#A32D2D', ms=5, label='Ghia 1982')
    ax[2].set(xlabel='x', ylabel='v / U', title=f"horizontal centreline v (RMSE {r['rmse_v']:.4f})")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    make_figure(cavity(N=96, steps=30000, use_mlx=HAVE_MLX, verbose=False))
