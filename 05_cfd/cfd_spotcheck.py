"""
CFD spot-check of the Wave-3 geometry optimum — Project Sycamore Phase-1.

The geometry optimiser chose a low-aspect-ratio wing (Rossby R/c̄ ≈ 3) to enter
the stable-leading-edge-vortex (LEV) window — an assumption the low-order
blade-element model cannot test. This script probes it with the validated 3-D
rotating-frame LBM (lbm3d_rotor), but its MAIN honest finding is what the
available tooling CANNOT show.

HONEST OUTCOME (after adversarial review):
  - The revolving-wing setup is a fully PERIODIC closed box with an IMPULSIVE
    start. The off-centre wing progressively co-rotates the whole fluid mass, so
    the relative inflow collapses within ~0.15 rev and any vortex is an
    EARLY-TIME transient — there is NO sustained, settled, attached LEV to
    measure (a single-snapshot "drainage" peaks at the snapshot time then decays).
  - The effective lattice Reynolds number here is O(50–100), orders below the
    LEV-stabilisation regime (Lentink & Dickinson 2009), and tip Mach ≈ 0.2.
  - Therefore this LBM CANNOT validate the LEV assumption at the optimum. It can
    only give a weak, EARLY-TIME, DIRECTIONAL hint.

What it does show (with the metric cleaned up — wing-INDUCED radial velocity,
i.e. ux minus the imposed −Ω×r background): the lower-Rossby OPTIMUM produces
directionally MORE wing-induced spanwise drainage than the baseline, throughout
the early-time window, consistent with the rotation-stabilised-LEV mechanism.
That is a hint, not a validation.

Conclusion: a resolved-Reynolds, open-domain CFD or a wind-tunnel test is
REQUIRED to actually validate the optimum's LEV — i.e. this is exactly the
SBIR Phase-1 → Phase-2 transition.

Tier: Indicative / qualitative / early-time only.

Run:  ~/ds/bin/python 05_cfd/cfd_spotcheck.py
Output: output/cfd_spotcheck.png
"""

import os
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from lbm3d_rotor import LBM3DRotating, validate_solidbody

os.makedirs('output', exist_ok=True)

N      = 96               # Rc=0.30·96=28.8 → Rc/chord = 2.88 (opt) / 5.43 (base)
OMEGA  = 0.003
R_FRAC = 0.30
SPAN   = 18.0
AOA    = 25.0
THICK  = 3.0
REVS   = 0.8
NCHK   = 8
CASES  = {'optimum (Ro≈3.0)': dict(chord=10.0, c='C0'),
          'baseline (Ro≈5.5)': dict(chord=5.3, c='C3')}


def build(chord):
    """Replicates lbm3d_rotor.revolving_wing's setup so we can checkpoint in time."""
    nu = (0.52 - 0.5)/3.0 + 0.006
    sim = LBM3DRotating(N, nu, OMEGA)
    x0 = y0 = z0 = N/2.0
    Rc = R_FRAC*N
    a = np.deg2rad(AOA)
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype('f4')
    xs = xx - (x0 + Rc); ys = yy - y0; zs = zz - z0
    c_coord = ys*np.cos(a) + zs*np.sin(a)
    n_coord = -ys*np.sin(a) + zs*np.cos(a)
    plate = (np.abs(xs) <= SPAN/2) & (np.abs(c_coord) <= chord/2) & (np.abs(n_coord) <= THICK/2)
    sim.set_obstacle(plate)
    rx = xx - x0; ry = yy - y0
    rho = mx.ones((N, N, N))
    sim.f = sim.equilibrium(rho, mx.array(OMEGA*ry), mx.array(-OMEGA*rx),
                            mx.array(np.zeros_like(rx))).astype(sim.sd)
    mx.eval(sim.f)
    g = dict(N=N, x0=x0, y0=y0, z0=z0, Rc=Rc, chord=chord, span=SPAN, Omega=OMEGA,
             tip=OMEGA*(Rc + SPAN/2), ry=ry)
    return sim, plate, g


def induced_drain(sim, plate, g):
    """Wing-INDUCED spanwise drainage: max(ux − Ω·ry) in a shell round the wing /
    tip speed. Subtracting the imposed −Ω×r background removes the ~40–60% pure
    kinematic floor the raw metric carried."""
    ux, uy, uz, rho = sim.frame_velocity()
    ux_ind = ux - g['Omega']*g['ry']           # subtract the imposed −Ω×r background
    shell = binary_dilation(plate, iterations=3) & ~plate
    return float(np.nanmax(ux_ind[shell])) / g['tip']


def omega_x_plane(sim, plate, g):
    """Correct spanwise (x-aligned) LEV vorticity ω_x=∂u_z/∂y−∂u_y/∂z in the
    mid-span chord plane (the previous code used a mislabeled mixed component)."""
    ux, uy, uz, rho = sim.frame_velocity()
    wx = np.gradient(uz, axis=1) - np.gradient(uy, axis=0)     # [z,y,x]: ∂uz/∂y − ∂uy/∂z
    xc = int(g['x0'] + g['Rc'])
    return np.ma.masked_where(plate[:, :, xc], wx[:, :, xc])


if __name__ == '__main__':
    print("[note] solid-body validation below covers ONLY the empty rotating frame "
          "(τ=0.8, no obstacle);\n       it does NOT validate the wing/bounce-back path "
          "(τ≈0.54) the comparison uses.")
    validate_solidbody(N=64)
    print()

    steps_total = int(REVS * 2*np.pi/OMEGA)
    checkpoints = np.unique(np.linspace(steps_total/NCHK, steps_total, NCHK).astype(int))
    hist = {}
    finals = {}
    for name, cfg in CASES.items():
        sim, plate, g = build(cfg['chord'])
        revs, drain = [], []
        prev = 0
        for s in checkpoints:
            sim.run(int(s - prev)); prev = int(s)
            revs.append(s*OMEGA/(2*np.pi)); drain.append(induced_drain(sim, plate, g))
        hist[name] = (np.array(revs), np.array(drain))
        finals[name] = (sim, plate, g)
        print(f"  {name:20s} Ro≈{g['Rc']/g['chord']:.2f}  induced-drainage(t): "
              f"peak {max(drain):.2f} @ {revs[int(np.argmax(drain))]:.2f} rev "
              f"→ {drain[-1]:.2f} @ {revs[-1]:.2f} rev  (decays = transient, not steady)")

    ro, do = hist['optimum (Ro≈3.0)']; rb, db = hist['baseline (Ro≈5.5)']
    print(f"\n{'='*70}")
    print("  VERDICT (early-time, qualitative — NOT a validation):")
    print(f"  • The induced drainage is a TRANSIENT (rises then decays as the closed box")
    print(f"    spins up) — no sustained attached LEV exists to measure in this setup.")
    print(f"  • Throughout the window the lower-Rossby OPTIMUM induces MORE drainage than")
    print(f"    the baseline (mean {do.mean():.2f} vs {db.mean():.2f}), a directional hint")
    print(f"    consistent with rotation-stabilised-LEV theory — but at Re~O(50–100), far")
    print(f"    below the LEV regime, so it is a hint, not evidence of a stable LEV.")
    print(f"  • A resolved-Re, open-domain CFD or wind-tunnel test is REQUIRED to validate")
    print(f"    the optimum (the SBIR Phase-1 → Phase-2 transition).")
    print(f"{'='*70}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='white')
    fig.suptitle("CFD spot-check of the geometry optimum — what the closed-box LBM can "
                 "(and cannot) show", fontsize=11)
    for name, cfg in CASES.items():
        rv, dr = hist[name]
        ax[0].plot(rv, dr, '-o', color=cfg['c'], lw=2, ms=4, label=name)
    ax[0].set(xlabel='revolutions', ylabel='wing-induced drainage / tip speed',
              title='Induced drainage is a TRANSIENT (peaks then decays)\n— no steady attached LEV')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.4)
    sim, plate, g = finals['optimum (Ro≈3.0)']
    wx = omega_x_plane(sim, plate, g)
    xc = int(g['x0'] + g['Rc'])
    lim = np.nanpercentile(np.abs(wx.filled(np.nan)), 99)
    ax[1].imshow(wx, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='equal')
    ax[1].contour(plate[:, :, xc], levels=[0.5], colors='k', linewidths=1.3)
    ax[1].set(xlabel='y (chord)', ylabel='z (axial)',
              title=f'Optimum: early-time spanwise vorticity ω_x at {REVS:.1f} rev\n'
                    '(impulsive-start vortex, not a settled LEV)')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('output/cfd_spotcheck.png', dpi=130, bbox_inches='tight', facecolor='white')
    print('\nSaved: output/cfd_spotcheck.png')
