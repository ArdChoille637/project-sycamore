"""
Wing-geometry optimization for stable autorotation — Project Sycamore Phase-1
Wave 3.

Delivers the "optimize wing geometry ... for stable autorotation" clause of the
§6 Phase-1 mandate, building on the validated aero solver (samara_bem) and its
Config dataclass. The flagged Wave-3 target is the Rossby number: the baseline
device sits at R/c̄ ≈ 5.5, ABOVE the stable-LEV window (Ro ≲ 3–4, Lentink &
Dickinson 2009), so the leading-edge vortex the whole aero model assumes is not
guaranteed to stay attached. Lowering the aspect ratio (wider chord) brings Ro
into range — at the cost of changing the descent/RPM operating point.

Design map over the two governing geometry knobs:
  - chord scale s_chord  → Rossby R/c̄ = R / (s_chord · c̄₀)   (the LEV constraint)
  - uniform pitch θ      → descent rate & RPM (must be θ<0 to autorotate, per the
                           Wave-1 negative-pitch result)
For each geometry the steady BEM gives V_d, Ω; feasibility = autorotates AND
self-correcting (dQ/dΩ<0) AND Rossby ≤ 3.5 AND tip Mach < 0.30. The optimum is
the slowest-descending feasible geometry (slow descent = loiter time for the
wildfire-monitoring fail-safe); the V_d–RPM Pareto front exposes the trade with
spin rate (higher Ω = more regenerative-induction harvest).

Tier: Indicative (aero-tier forces; Rossby is a validity flag on the LEV
assumption, not a force the model resolves). A CFD spot-check of the optimum is
the next step. Does not optimise mass/stability (that is Wave 2's axis).

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
ROSSBY_MAX = 3.5                  # stable-LEV window ceiling
MACH_MAX   = 0.30


def evaluate(R=0.30, s_chord=1.0, theta_deg=-2.6, taper=None, m=0.075,
             cl_alpha=5.3, cd0=0.025):
    """Build a geometry and return its autorotation metrics + feasibility."""
    if taper is None:                                  # scale both chords (hold taper)
        c_root, c_tip = C_ROOT0*s_chord, C_TIP0*s_chord
    else:                                              # taper = c_tip/c_root at scaled mean
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
    sc = bool(info.get('self_correcting', False))
    feasible = bool(auto and sc and rossby <= ROSSBY_MAX and mach < MACH_MAX)
    return dict(R=R, s_chord=s_chord, theta=theta_deg, c_root=c_root, c_tip=c_tip,
                cbar=cbar, Vd=Vd, rpm=rpm, rossby=rossby, tip_mach=mach,
                autorotates=auto, self_correcting=sc, feasible=feasible)


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


if __name__ == '__main__':
    base = evaluate(s_chord=1.0, theta_deg=-2.6)
    print(f"Baseline geometry: V_d={base['Vd']:.2f} m/s, {base['rpm']:.0f} RPM, "
          f"Rossby={base['rossby']:.2f} (target ≤{ROSSBY_MAX}), tip Ma={base['tip_mach']:.2f} "
          f"→ feasible={base['feasible']}")

    s_chords = np.linspace(1.0, 1.9, 10)         # Rossby 5.45 → 2.87
    thetas   = np.linspace(-2.0, -12.0, 9)
    print(f"\nSweeping {len(s_chords)}×{len(thetas)} geometries (chord scale × pitch)…")
    G, recs = sweep(s_chords, thetas)

    opt = optimum(recs)
    if opt:
        print(f"\nOPTIMUM (slowest feasible descent, Rossby ≤ {ROSSBY_MAX}):")
        print(f"  chord ×{opt['s_chord']:.2f} (c_root={opt['c_root']*100:.1f}cm, "
              f"c_tip={opt['c_tip']*100:.1f}cm), pitch θ={opt['theta']:.1f}°")
        print(f"  V_d={opt['Vd']:.2f} m/s ({base['Vd']:.2f} baseline, "
              f"{(opt['Vd']/base['Vd']-1)*100:+.0f}%), {opt['rpm']:.0f} RPM, "
              f"Rossby={opt['rossby']:.2f}, tip Ma={opt['tip_mach']:.2f}")
    else:
        print("\nNo feasible geometry found in the swept box.")

    pf = pareto(recs)
    print(f"\nPareto front (V_d vs RPM, feasible): {len(pf)} points")
    for r in pf:
        print(f"  V_d={r['Vd']:.2f} m/s  {r['rpm']:.0f} RPM  Ro={r['rossby']:.2f}  θ={r['theta']:.0f}°")

    # ── figure: V_d design map (Rossby × pitch) + feasible region + Pareto ──
    rossby_axis = G['rossby'][0, :]              # rossby depends only on s_chord
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle('Sycamore wing-geometry optimisation (stable-autorotation feasible region)', fontsize=11)

    Vd = np.ma.masked_invalid(G['Vd'])
    cf = ax[0].contourf(rossby_axis, thetas, Vd, levels=14, cmap='viridis')
    fig.colorbar(cf, ax=ax[0], label='descent V_d [m/s]')
    # feasibility: shade where NOT feasible (Rossby>3.5 or no autorotation)
    feas = G['feasible'].astype(float)
    ax[0].contourf(rossby_axis, thetas, np.ma.masked_where(feas > 0.5, np.ones_like(feas)),
                   colors='none', hatches=['xx'], alpha=0)
    ax[0].axvline(ROSSBY_MAX, color='red', lw=2, ls='--', label=f'Rossby={ROSSBY_MAX} (LEV limit)')
    ax[0].plot(base['rossby'], base['theta'], 'wo', ms=9, mec='k', label='baseline (Ro 5.5, infeasible)')
    if opt:
        ax[0].plot(opt['rossby'], opt['theta'], 'r*', ms=20, mec='k', label='optimum')
    ax[0].set(xlabel='Rossby number R/c̄  (← wider chord)', ylabel='pitch θ [deg]',
              title='Descent rate over the design space')
    ax[0].invert_xaxis(); ax[0].legend(fontsize=8, loc='lower left')

    if pf:
        allf = [r for r in recs if r['feasible']]
        sccol = ax[1].scatter([r['rpm'] for r in allf], [r['Vd'] for r in allf],
                              c=[r['rossby'] for r in allf], cmap='plasma', s=30)
        fig.colorbar(sccol, ax=ax[1], label='Rossby')
        ax[1].plot([r['rpm'] for r in pf], [r['Vd'] for r in pf], 'k-o', lw=1.5, ms=5, label='Pareto front')
        if opt:
            ax[1].plot(opt['rpm'], opt['Vd'], 'r*', ms=20, mec='k')
        ax[1].set(xlabel='autorotation Ω [RPM]', ylabel='descent V_d [m/s]',
                  title='V_d–RPM trade (feasible geometries)')
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/samara_optimize.png', dpi=140, bbox_inches='tight')
    print('\nSaved: output/samara_optimize.png')
