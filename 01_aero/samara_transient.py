"""
Transient active→passive mode-transition solver — Project Sycamore Phase-1
Wave 2b.

The steady BEM finds the autorotation equilibrium; it cannot say whether the
craft actually REACHES it after the propeller is cut, how long that takes, how
much altitude is lost, or whether the equilibrium is a stable attractor. Those
are exactly the questions an SBIR reviewer asks of a "fail-safe", and they are
the "seamless active-to-passive mode transition" half of the §6 Phase-1 mandate.

Model — 2-DOF rigid-body flight dynamics (descent rate V_z, spin rate Ω):
    m   · dV_z/dt = m·g − T(V_z, Ω)        (vertical force balance)
    I_zz· dΩ /dt = Q(V_z, Ω)               (spin: net aero torque, no engine)
T and Q are the validated blade-element thrust and net torque (samara_bem); the
equilibrium they define (T=W, Q=0) is the steady solution the transient must
asymptote to. I_zz comes from the Wave-2a mass model. The propeller is cut at
t=0 (T_prop=Q_prop=0); the powered state enters only as the initial condition
(V_z0, Ω0), so this does not require a powered-mode aero model — the IC
ambiguity is handled by mapping the whole basin of attraction.

Deliverables: transition time, altitude lost, peak AoA / root-load overshoot,
Jacobian eigenvalues (stable-attractor check), and the basin of attraction
(does the powered state converge to autorotation).

Tier: Indicative (inherits the aero forces' tier; the EOM and the asymptote-to-
equilibrium self-consistency are exact).

Run:  ~/ds/bin/python samara_transient.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import samara_bem as sb
import samara_mass as sm

G = 9.81
os.makedirs('output', exist_ok=True)


def _TQ(Vz, Om, cfg, grid):
    Tarr, Qarr = sb.forces(max(float(Vz), 1e-2), max(float(Om), 1e-2), cfg, grid)
    return Tarr[0], Qarr[0]


def transition(cfg, mass, Vz0, Om0, eq=None, t_max=40.0, settle_tol=0.02, metrics=True):
    """Integrate engine-cut → autorotation from (V_z0, Ω0). `eq`=(Vd*,Ω*) may be
    passed to avoid re-solving the equilibrium (basin sweeps)."""
    grid = cfg.grid(); m = mass.m_total; Izz = mass.properties()['I_zz']
    Vd_eq, Om_eq = eq if eq is not None else sb.solve(cfg, m)

    def rhs(t, y):
        Vz, Om = y
        T, Q = _TQ(Vz, Om, cfg, grid)
        return [G - T/m, Q/Izz]

    sol = solve_ivp(rhs, [0, t_max], [Vz0, Om0], max_step=0.1, rtol=1e-7, atol=1e-9)
    t, Vz, Om = sol.t, sol.y[0], sol.y[1]
    converged = (abs(Vz[-1]-Vd_eq) < 0.05*Vd_eq) and (abs(Om[-1]-Om_eq) < 0.05*Om_eq)

    out = dict(t=t, Vz=Vz, Om=Om, Vd_eq=Vd_eq, Om_eq=Om_eq, converged=converged)
    if not metrics:
        return out

    # time to settle within ±settle_tol of BOTH states; altitude descended to then
    band = (np.abs(Vz-Vd_eq) > settle_tol*Vd_eq) | (np.abs(Om-Om_eq) > settle_tol*Om_eq)
    t_settle = float(t[np.where(band)[0][-1]]) if band.any() else 0.0
    in_settle = t <= t_settle
    alt_to_settle = float(np.trapezoid(np.maximum(Vz[in_settle], 0), t[in_settle])) if t_settle > 0 else 0.0
    # peak spanwise AoA and root-bending load during the transient
    r, c, theta = grid; dr = r[1]-r[0]; peak_aoa = 0.0; peak_load = 0.0
    Vd_load = abs(_root_load(Vd_eq, Om_eq, cfg, grid))
    for i in range(0, len(t), max(1, len(t)//250)):
        _, _, phi, al, U2, cl, cd, _ = sb._state(max(Vz[i], 1e-2), max(Om[i], 1e-2), cfg, r, c, theta)
        peak_aoa = max(peak_aoa, float(np.max(np.abs(np.rad2deg(al[0])))))
        peak_load = max(peak_load, abs(_root_load(Vz[i], Om[i], cfg, grid)))
    out.update(t_settle=t_settle, alt_lost=alt_to_settle, peak_aoa_deg=peak_aoa,
               peak_load=peak_load, load_overshoot=peak_load/Vd_load if Vd_load > 0 else np.nan)
    return out


def _root_load(Vz, Om, cfg, grid):
    """Root-bending moment proxy ∫ dT·r dr (axial load moment about the hub)."""
    r, c, theta = grid; dr = r[1]-r[0]
    _, _, phi, al, U2, cl, cd, _ = sb._state(max(float(Vz), 1e-2), max(float(Om), 1e-2), cfg, r, c, theta)
    q = 0.5*cfg.rho*U2[0]*c
    dT = (q*cl[0]*np.cos(phi[0]) + q*cd[0]*np.sin(phi[0])) * cfg.n_blades
    return float(np.sum(dT*r*dr))


def jacobian(cfg, mass):
    """2×2 Jacobian of [dV_z/dt, dΩ/dt] at the equilibrium → eigenvalues."""
    grid = cfg.grid(); m = mass.m_total; Izz = mass.properties()['I_zz']
    Vd, Om = sb.solve(cfg, m)
    def f(vz, om):
        T, Q = _TQ(vz, om, cfg, grid)
        return np.array([G - T/m, Q/Izz])
    ev, eo = 2e-3, 2e-2
    J = np.zeros((2, 2))
    J[:, 0] = (f(Vd+ev, Om) - f(Vd-ev, Om)) / (2*ev)
    J[:, 1] = (f(Vd, Om+eo) - f(Vd, Om-eo)) / (2*eo)
    eig = np.linalg.eigvals(J)
    return J, eig, Vd, Om


def basin(cfg, mass, vz_grid=None, om_grid=None, t_max=30.0):
    """Sweep initial conditions; mark which converge to autorotation."""
    if vz_grid is None: vz_grid = np.linspace(0.0, 9.0, 8)
    if om_grid is None: om_grid = np.linspace(5.0, 220.0, 8)
    eq = sb.solve(cfg, mass.m_total)
    M = np.zeros((len(om_grid), len(vz_grid)), bool)
    for i, om0 in enumerate(om_grid):
        for j, vz0 in enumerate(vz_grid):
            M[i, j] = transition(cfg, mass, vz0, om0, eq=eq, t_max=t_max, metrics=False)['converged']
    return vz_grid, om_grid, M, eq


if __name__ == '__main__':
    cfg = sb.SAMARA
    mass = sm.MassModel(cfg=cfg, m_total=0.075)
    Izz = mass.properties()['I_zz']
    Vd_eq, Om_eq = sb.solve(cfg, mass.m_total)
    print(f"Equilibrium (steady BEM): V_d*={Vd_eq:.2f} m/s, Ω*={Om_eq*60/2/np.pi:.0f} RPM, "
          f"I_zz={Izz*1e4:.2f}e-4 kg·m²")

    # stable-attractor check
    J, eig, _, _ = jacobian(cfg, mass)
    re = eig.real
    print(f"\n[stability] Jacobian eigenvalues = {eig[0]:.3f}, {eig[1]:.3f}  "
          f"→ {'STABLE attractor ✓' if np.all(re < 0) else 'UNSTABLE ✗'} "
          f"(τ ≈ {1/abs(re).min():.1f} s slow mode)")

    # nominal transition: powered hold-altitude (V_z0=0) at autorotation spin → cut
    nom = transition(cfg, mass, Vz0=0.0, Om0=Om_eq)
    print(f"\n[transition · nominal]  IC V_z0=0 (hover), Ω0=Ω*")
    print(f"  converged to autorotation: {nom['converged']}  (final V_z={nom['Vz'][-1]:.2f}, "
          f"final Ω={nom['Om'][-1]*60/2/np.pi:.0f} RPM)")
    print(f"  settling time (±2%): {nom['t_settle']:.1f} s   altitude descended to settle: "
          f"{nom['alt_lost']:.1f} m")
    print(f"  peak AoA during transient: {nom['peak_aoa_deg']:.0f}°   "
          f"root-load overshoot: {nom['load_overshoot']:.2f}× steady")

    # worst case: spin up from near-rest
    cold = transition(cfg, mass, Vz0=0.0, Om0=0.3*Om_eq)
    print(f"\n[transition · cold spin-up]  IC V_z0=0, Ω0=0.3·Ω*")
    print(f"  converged: {cold['converged']}   settle {cold['t_settle']:.1f} s   "
          f"altitude {cold['alt_lost']:.1f} m   peak AoA {cold['peak_aoa_deg']:.0f}°")

    # validation: the transient asymptote must match the steady equilibrium
    err = abs(nom['Vz'][-1]-Vd_eq)/Vd_eq
    print(f"\n[validation] transient asymptote vs steady BEM: ΔV_d={err*100:.2f}%  "
          f"→ {'PASS ✓' if err < 0.02 and nom['converged'] else 'FAIL ✗'}")

    # basin of attraction
    vz_g, om_g, M, _ = basin(cfg, mass)
    frac = M.mean()
    print(f"\n[basin] {M.sum()}/{M.size} sampled powered states converge to autorotation "
          f"({frac*100:.0f}%); powered hover (low V_z) {'IS' if M[:,0].any() else 'is NOT'} in the basin")

    # figure: time histories + basin
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle('Sycamore active→passive transition (engine cut at t=0)', fontsize=11)
    ax[0].plot(nom['t'], nom['Vz'], 'steelblue', lw=2, label='nominal (Ω0=Ω*)')
    ax[0].plot(cold['t'], cold['Vz'], 'darkorange', lw=2, ls='--', label='cold (Ω0=0.3Ω*)')
    ax[0].axhline(Vd_eq, color='k', ls=':', lw=1, label='V_d* equilibrium')
    ax[0].set(xlabel='time [s]', ylabel='descent V_z [m/s]', title='Descent rate'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.4)
    ax[1].plot(nom['t'], nom['Om']*60/2/np.pi, 'steelblue', lw=2)
    ax[1].plot(cold['t'], cold['Om']*60/2/np.pi, 'darkorange', lw=2, ls='--')
    ax[1].axhline(Om_eq*60/2/np.pi, color='k', ls=':', lw=1)
    ax[1].set(xlabel='time [s]', ylabel='spin Ω [RPM]', title='Spin rate'); ax[1].grid(alpha=0.4)
    ax[2].imshow(M, origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                 extent=[vz_g[0], vz_g[-1], om_g[0]*60/2/np.pi, om_g[-1]*60/2/np.pi])
    ax[2].scatter([0],[Om_eq*60/2/np.pi], c='blue', s=40, marker='*', label='powered hover IC', zorder=5)
    ax[2].set(xlabel='initial V_z0 [m/s]', ylabel='initial Ω0 [RPM]', title='Basin of attraction (green=converges)')
    ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig('output/samara_transition.png', dpi=140, bbox_inches='tight')
    print('\nSaved: output/samara_transition.png')
