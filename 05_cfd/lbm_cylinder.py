"""
Second LBM validation — flow past a circular cylinder at Re = 100.

Tests the external-flow path (velocity inlet, open outlet, wall + obstacle
bounce-back) and the unsteady Kármán vortex street. The shedding Strouhal number
St = f·D/U should be ≈ 0.16–0.17 at Re = 100 (Williamson 1996).

Run: ~/ds/bin/python 05_cfd/lbm_cylinder.py
Output: output/lbm_cylinder.png
"""

import time
import numpy as np
from lbm_mlx import LBM, HAVE_MLX


def cylinder(Re=100.0, D=20.0, U=0.1, nx=440, ny=120, steps=32000,
             warmup=12000, offset=1.5, use_mlx=True, verbose=True):
    nu = U*D/Re
    sim = LBM(ny, nx, nu, use_mlx=use_mlx)
    cx, cy = nx*0.22, ny*0.5
    yy, xx = np.mgrid[0:ny, 0:nx]
    disk = ((xx - cx)**2 + (yy - (cy+offset))**2) < (D/2)**2   # vertical offset breaks symmetry
    sim.set_obstacle(disk)
    sim.add_walls(top=True, bottom=True)
    sim.set_inlet(U)
    sim.set_outlet()

    fx_t, fy_t = [], []                                     # drag/lift force time series
    t0 = time.perf_counter()
    for it in range(steps):
        sim.step()
        if it >= warmup:
            Fx, Fy = sim.force_on_solid()
            fx_t.append(Fx); fy_t.append(Fy)
    wall = time.perf_counter() - t0
    fx_t = np.array(fx_t); fy_t = np.array(fy_t)

    # St from the lift force (clean fundamental, unlike a centreline velocity probe)
    sig = fy_t - fy_t.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    f_shed = freqs[1 + int(np.argmax(spec[1:]))]
    St = f_shed * D / U
    q = 0.5 * U*U * D                                       # dynamic pressure × projected length
    Cd = fx_t.mean() / q
    Cl_amp = (fy_t.max() - fy_t.min()) / 2 / q

    be = sim.be
    print(f"\nFlow past cylinder Re={Re:.0f}, D={D:.0f}, {nx}×{ny}, "
          f"{be.mlx and 'MLX' or 'NumPy'}, {steps} steps in {wall:.1f}s")
    print(f"  Strouhal  St = {St:.4f}   (literature ≈ 0.164 at Re=100)")
    print(f"  drag      Cd = {Cd:.3f}    (literature ≈ 1.3–1.4)")
    print(f"  lift amp  Cl = {Cl_amp:.3f}    (literature ≈ 0.3)")
    ok = (0.15 <= St <= 0.18) and (1.2 <= Cd <= 1.5)
    print(f"  {'PASS' if ok else 'FAIL'}  (St 0.15–0.18 and Cd 1.2–1.5)")
    return dict(sim=sim, St=St, f=f_shed, sig=sig, Cd=Cd, Cl=Cl_amp, ok=ok, D=D, U=U,
                nx=nx, ny=ny, cx=cx, cy=cy, disk=disk)


def make_figure(r, out='output/lbm_cylinder.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vort = r['sim'].vorticity()
    vort = np.ma.masked_where(r['disk'], vort)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), facecolor='white',
                           gridspec_kw={'height_ratios': [2, 1]})
    lim = np.nanpercentile(np.abs(vort), 99)
    ax[0].imshow(vort, cmap='RdBu_r', origin='lower', vmin=-lim, vmax=lim, aspect='equal')
    th = np.linspace(0, 2*np.pi, 60)
    ax[0].fill(r['cx'] + r['D']/2*np.cos(th), r['cy']+0.5 + r['D']/2*np.sin(th), color='k')
    ax[0].set_title(f"Kármán vortex street — Re=100, St={r['St']:.3f}, Cd={r['Cd']:.2f} "
                    f"({r['sim'].be.mlx and 'MLX' or 'NumPy'} D2Q9 LBM)")
    ax[0].set(xlabel='x [lattice]', ylabel='y')
    ax[1].plot(r['sig'], lw=0.7, color='#185FA5')
    ax[1].set(xlabel='step (after warm-up)', ylabel='lift force', title='lift force (shedding fundamental)')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    make_figure(cylinder(use_mlx=HAVE_MLX))
