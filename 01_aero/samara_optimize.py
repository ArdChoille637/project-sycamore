"""
Wing-geometry optimization for stable autorotation — Project Sycamore Phase-1
Wave 3.

Delivers the "optimize wing geometry ... for stable autorotation" clause of the
§6 Phase-1 mandate, on the validated aero solver (samara_bem) + its Config.

The governing geometry knob is the aspect ratio, expressed as the Rossby number
Ro = R/c̄ (≈ blade aspect ratio for this planform):
  - Ro must be LOW enough for a stable leading-edge vortex — the lift mechanism
    the whole aero model assumes (Lentink & Dickinson 2009: Ro ≲ 3–4).
  - Ro must be HIGH enough that the blade stays slender, so the strip-theory /
    per-annulus-momentum aero stays in its VALIDATED regime (chord ≪ local
    radius). Pushing Ro well below 3 makes the root chord approach the radius
    (c_root/R → 0.5), where the per-annulus model is geometrically dubious.
These squeeze the design into the **samara aspect-ratio band Ro ∈ [3, 4]** — the
window real samaras occupy, and which brackets the model's validated anchor
(Sycamore A, Ro = 3.9). The baseline device sits OUTSIDE it at Ro ≈ 5.5 (too
slender → LEV not guaranteed), which is the problem this wave fixes.

Design map over the two knobs:
  - chord scale s_chord → Ro = R/(s_chord·c̄₀)
  - uniform pitch θ     → descent rate & RPM (θ<0 required to autorotate; the
                          autorotation floor is near θ≈0, NOT a fixed −3°)
Feasibility = autorotates AND the equilibrium is well-posed AND tip Mach < 0.30
AND Ro ∈ [3, 4]. The optimum is the slowest-descending feasible geometry (loiter
time for the wildfire-monitoring fail-safe); the V_d–RPM Pareto exposes the
trade with spin rate (higher Ω = more regenerative-induction harvest).

Honest scope (tier Indicative): descent is pitch-DOMINANT (~3×) but the pitch and
chord axes are COUPLED, not orthogonal. The feasible optimum is pinned to the
wide-chord (low-Ro) edge of the band, i.e. the slender-blade validity limit —
descent would keep falling with wider chord but only by exiting the validated
aero regime, so a **CFD spot-check of the optimum is essential, not optional**.
Ro is a validity flag on the LEV assumption, not a force the model resolves. This
wave does not optimise mass/stability (Wave 2's axis).

Run:  ~/ds/bin/python samara_optimize.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import samara_bem as sb

os.makedirs('output', exist_ok=True)

C_ROOT0, C_TIP0 = 0.08, 0.03      # baseline chords [m]
CBAR0 = 0.5*(C_ROOT0 + C_TIP0)
ROSSBY_MIN = 3.0                  # strip-theory / slender-blade floor
ROSSBY_MAX = 4.0                  # stable-LEV ceiling (Sycamore A = 3.9)
MACH_MAX   = 0.30


def evaluate(R=0.30, s_chord=1.0, theta_deg=-2.6, taper=None, m=0.075,
             cl_alpha=5.3, cd0=0.025):
    """Build a geometry and return its autorotation metrics + feasibility."""
    if taper is None:                                  # scale both chords (hold taper)
        c_root, c_tip = C_ROOT0*s_chord, C_TIP0*s_chord
    else:                                              # taper = c_tip/c_root at scaled mean
        if taper <= 0:
            raise ValueError(f"taper must be > 0 (got {taper})")
        cbar = CBAR0*s_chord
        c_root = 2*cbar/(1+taper); c_tip = taper*c_root
    cfg = sb.Config(name='opt', R=R, c_root=c_root, c_tip=c_tip,
                    theta_root_deg=theta_deg, theta_tip_deg=theta_deg,
                    alpha_sign=1.0, cl_alpha=cl_alpha, cd0=cd0)
    Vd, Om, info = sb.solve(cfg, m, return_info=True)
    cbar = 0.5*(c_root + c_tip)
    auto = not np.isnan(Vd)
    rpm  = Om*60/(2*np.pi) if auto else np.nan
    mach = Om*R/340 if auto else np.nan
    rossby = R/cbar
    wp = bool(info.get('well_posed', False))           # stricter than self_correcting alone
    # NOTE: self_correcting (dQ/dΩ<0) is near-tautological given how solve() picks
    # the equilibrium, and tip Mach never binds here (≪0.30); in practice feasibility
    # is set by the Ro∈[3,4] window. We gate on the solver's full well_posed flag.
    feasible = bool(auto and wp and mach < MACH_MAX
                    and ROSSBY_MIN <= rossby <= ROSSBY_MAX)
    return dict(R=R, s_chord=s_chord, theta=theta_deg, c_root=c_root, c_tip=c_tip,
                cbar=cbar, AR=rossby, Vd=Vd, rpm=rpm, rossby=rossby, tip_mach=mach,
                autorotates=auto, self_correcting=bool(info.get('self_correcting', False)),
                well_posed=wp, feasible=feasible)


def sweep(s_chords, thetas, **kw):
    """2-D design sweep; returns a dict of (len(thetas), len(s_chords)) arrays."""
    nT, nS = len(thetas), len(s_chords)
    keys = ['Vd', 'rpm', 'rossby', 'tip_mach', 'feasible']
    G = {k: np.full((nT, nS), np.nan) for k in keys}
    recs = []
    for i, th in enumerate(thetas):
        for j, sc in enumerate(s_chords):
            r = evaluate(s_chord=sc, theta_deg=th, **kw)
            for k in keys:
                G[k][i, j] = r[k]
            recs.append(r)
    return G, recs


def optimum(recs):
    feas = [r for r in recs if r['feasible']]
    return min(feas, key=lambda r: r['Vd']) if feas else None


def pareto(recs):
    """Non-dominated set on (minimise V_d, maximise RPM) among feasible points."""
    feas = [r for r in recs if r['feasible']]
    nd = []
    for r in feas:
        if not any((o['Vd'] <= r['Vd'] and o['rpm'] >= r['rpm'] and o is not r)
                   and (o['Vd'] < r['Vd'] or o['rpm'] > r['rpm']) for o in feas):
            nd.append(r)
    return sorted(nd, key=lambda r: r['Vd'])


def rossby_sensitivity(recs):
    """How the feasible-set size depends on the Ro window edges (the threshold is
    a soft literature value, so report its effect)."""
    out = {}
    for lo, hi in [(3.0, 4.0), (3.0, 3.5), (3.5, 4.0)]:
        n = sum(1 for r in recs if r['autorotates'] and r['well_posed']
                and lo <= r['rossby'] <= hi and r['tip_mach'] < MACH_MAX)
        out[(lo, hi)] = n
    return out


if __name__ == '__main__':
    base = evaluate(s_chord=1.0, theta_deg=-2.6)
    print(f"Baseline geometry: V_d={base['Vd']:.2f} m/s, {base['rpm']:.0f} RPM, "
          f"Ro={base['rossby']:.2f} (window [{ROSSBY_MIN},{ROSSBY_MAX}]) → feasible={base['feasible']}")

    s_chords = np.linspace(1.0, 2.0, 11)         # Ro 5.45 → 2.73 (window 1.36 → 1.82)
    thetas   = np.linspace(-1.5, -8.0, 14)       # 0.5° steps; autorotation floor ~θ≈0
    print(f"\nSweeping {len(s_chords)}×{len(thetas)} geometries (chord scale × pitch)…")
    G, recs = sweep(s_chords, thetas)

    opt = optimum(recs)
    if opt:
        # local pitch refine at the optimum's chord (grid θ step is coarse)
        fine = [evaluate(s_chord=opt['s_chord'], theta_deg=th) for th in np.arange(-1.0, -6.0, 0.25)]
        fine = [r for r in fine if r['feasible']]
        opt = min(fine, key=lambda r: r['Vd']) if fine else opt
        print(f"\nOPTIMUM (slowest feasible descent within the Ro window):")
        print(f"  chord ×{opt['s_chord']:.2f} (c_root={opt['c_root']*100:.1f}cm, "
              f"c_tip={opt['c_tip']*100:.1f}cm, AR≈{opt['AR']:.1f}), pitch θ={opt['theta']:.2f}°")
        print(f"  V_d={opt['Vd']:.2f} m/s ({base['Vd']:.2f} baseline, "
              f"{(opt['Vd']/base['Vd']-1)*100:+.0f}%), {opt['rpm']:.0f} RPM, "
              f"Ro={opt['rossby']:.2f}, tip Ma={opt['tip_mach']:.2f}")
        print(f"  ⚠ pinned to the wide-chord (low-Ro) edge of the validity band — "
              f"CFD spot-check needed before trusting the magnitude.")
    else:
        print("\nNo feasible geometry found in the swept box.")

    pf = pareto(recs)
    print(f"\nPareto front (V_d vs RPM, feasible): {len(pf)} points")
    for r in pf[:8]:
        print(f"  V_d={r['Vd']:.2f} m/s  {r['rpm']:.0f} RPM  Ro={r['rossby']:.2f}  θ={r['theta']:.1f}°")

    sens = rossby_sensitivity(recs)
    print(f"\nRo-window sensitivity (feasible-set size): "
          f"[3,4]→{sens[(3.0,4.0)]}, [3,3.5]→{sens[(3.0,3.5)]}, [3.5,4]→{sens[(3.5,4.0)]} "
          f"(threshold is soft; optimum sits at the low-Ro edge regardless)")

    # ── figure: V_d design map (Ro × pitch) + feasible band + Pareto ──
    rossby_axis = G['rossby'][0, :]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle('Sycamore wing-geometry optimisation (Ro∈[3,4] stable-LEV / valid-strip-theory band)', fontsize=10.5)

    Vd = np.ma.masked_invalid(G['Vd'])
    cf = ax[0].contourf(rossby_axis, thetas, Vd, levels=14, cmap='viridis')
    fig.colorbar(cf, ax=ax[0], label='descent V_d [m/s]')
    # shade infeasible cells (autorotation NaN or Ro outside [3,4]) in translucent grey
    infeas = np.ma.masked_where(G['feasible'] > 0.5, np.ones_like(G['feasible']))
    ax[0].contourf(rossby_axis, thetas, infeas, levels=[0, 2], colors=['gray'], alpha=0.40)
    for rb in (ROSSBY_MIN, ROSSBY_MAX):
        ax[0].axvline(rb, color='red', lw=1.8, ls='--')
    ax[0].plot(base['rossby'], base['theta'], 'wo', ms=9, mec='k', label='baseline (Ro 5.5, infeasible)')
    if opt:
        ax[0].plot(opt['rossby'], opt['theta'], 'r*', ms=20, mec='k', label='optimum')
    ax[0].set(xlabel='Rossby number R/c̄ ≈ AR  (← wider chord)', ylabel='pitch θ [deg]',
              title='Descent rate; grey=infeasible, red=Ro band [3,4]')
    ax[0].invert_xaxis(); ax[0].legend(fontsize=8, loc='lower left')

    feas_recs = [r for r in recs if r['feasible']]
    if feas_recs:
        sc = ax[1].scatter([r['rpm'] for r in feas_recs], [r['Vd'] for r in feas_recs],
                           c=[r['rossby'] for r in feas_recs], cmap='plasma', s=30)
        fig.colorbar(sc, ax=ax[1], label='Rossby')
        if pf:
            ax[1].plot([r['rpm'] for r in pf], [r['Vd'] for r in pf], 'k-o', lw=1.5, ms=5, label='Pareto front')
        if opt:
            ax[1].plot(opt['rpm'], opt['Vd'], 'r*', ms=20, mec='k', label='optimum')
        ax[1].set(xlabel='autorotation Ω [RPM]', ylabel='descent V_d [m/s]',
                  title='V_d–RPM trade (feasible geometries)')
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/samara_optimize.png', dpi=140, bbox_inches='tight')
    print('\nSaved: output/samara_optimize.png')
