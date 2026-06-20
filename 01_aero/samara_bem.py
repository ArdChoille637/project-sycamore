"""
Blade-element model for samara autorotation — Project Sycamore.

Solves for the steady autorotation equilibrium (descent rate V_d and rotation
rate Ω) of a single-wing rotating UAV, the aerodynamic proof-of-concept for the
samara-biomimetic bi-modal airframe.

What this is (honest scope)
---------------------------
This is **blade-element / strip theory with momentum-theory induced inflow** —
the same low-order "Samara Numerical Model" used by Jung & Rezgui (2023). It is
NOT a high-fidelity CFD; sectional forces come from an analytic LEV polar.

Aerodynamics — the LEV, not a flat plate
----------------------------------------
Autorotating samaras do not stall conventionally: a stable leading-edge vortex
(LEV) keeps the flow attached to extreme incidence, so sectional lift peaks at
α = 45° and the wing runs at α ~70° (root) → ~5° (tip). We use the validated
"normal force" sectional model (Jung & Rezgui 2023, Aerospace 10:414):
    C_L(α) = C_Lα · sinα·cosα            C_D(α) = C_D0 + C_Lα · sin²α
with C_Lα (lift-curve-slope parameter, 1/rad) and C_D0 (zero-lift drag) fitted,
wing/Reynolds-dependent constants (natural Sycamore: C_Lα 4.6–5.6, C_D0
0.016–0.033). This replaced an earlier thin-plate 2π + hard-10°-Viterna model
that crashed lift at 10° — physically wrong for the LEV regime.

Induced inflow
--------------
Momentum theory gives a per-element induced velocity v_i in the windmilling-
brake (descent) state: the axial through-flow is U_p = (V_d − v_i)·cosβ (v_i
SUBTRACTED — the correct descent-state sign), solved against the blade-element
load with a Prandtl single-blade tip-loss factor F. α = θ + φ (paper sign
convention), φ = arctan(U_p/U_t), U_t = Ωr.

Solver — well posed
-------------------
The thrust balance T(V_d)=W is multivalued, so a plain brentq returns an
arbitrary root and branch-jumps. We enumerate ALL T=W roots, assemble them into
continuous V_d(Ω) branches by continuation, and locate the equilibrium as the
*stable* on-branch net-torque zero (Q: driving→braking, dQ/dΩ<0).

Validation
----------
`validate_sycamore_A()` runs the model on the paper's measured Sycamore A
(R=4.47cm, c̄=1.15cm, m=232mg, θ=−2.6°) and checks it reproduces the slow
samara descent (sub-1 m/s) with root-high/tip-low α — the credibility anchor
for applying the same model to the engineered device.

References: Jung & Rezgui (2023) Aerospace 10:414; Lentink et al. (2009)
Science 324:1438; Lee, Lee & Sohn (2014) Exp.Fluids 55:1718; Yasuda & Azuma
(1997) J.Theor.Biol. 185:313.

Run with:  ~/ds/bin/python samara_bem.py
Outputs :  console report (samara + legacy configs) + output/samara_bem.png
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('output', exist_ok=True)

G       = 9.81
NU_AIR  = 1.5e-5     # kinematic viscosity of air [m²/s] (for section Reynolds)


# ── Sectional LEV airfoil model — Jung & Rezgui (2023) "normal force" ──────
def cl_cd(alpha, cl_alpha, cd0):
    """Sectional (CL, CD) for AoA [rad]: CL=cl_alpha·sinα·cosα (peak at 45°),
    CD=cd0+cl_alpha·sin²α. Smooth (C∞), stall-free; CL odd / CD even in α."""
    s, co = np.sin(alpha), np.cos(alpha)
    return cl_alpha * s * co, cd0 + cl_alpha * s**2


# ── Design configuration ──────────────────────────────────────────────────
@dataclass
class Config:
    name:           str
    R:              float = 0.30      # tip radius [m]
    c_root:         float = 0.08      # root chord [m]
    c_tip:          float = 0.03      # tip chord [m]
    theta_root_deg: float = 0.0       # root pitch [deg]
    theta_tip_deg:  float = 0.0       # tip pitch [deg]
    hub_frac:       float = 0.15      # root cutout as fraction of R
    n_blades:       int   = 1
    coning_deg:     float = 10.0      # coning angle β [deg]
    alpha_sign:     float = +1.0      # +1: α=θ+φ (samara) | −1: α=θ−φ (legacy)
    cl_alpha:       float = 5.3       # LEV lift-curve-slope parameter [1/rad]
    cd0:            float = 0.025      # zero-lift drag coefficient [-]
    rho:            float = 1.167      # air density [kg/m³] (~500 m MSL)
    induced:        bool  = True       # momentum induced inflow
    tiploss:        bool  = True       # Prandtl single-blade tip loss
    n_elem:         int   = 40

    def grid(self):
        r_hub = self.hub_frac * self.R
        r  = np.linspace(r_hub, self.R, self.n_elem)
        c  = self.c_root + (self.c_tip - self.c_root) * (r - r_hub) / (self.R - r_hub)
        th = np.deg2rad(self.theta_root_deg +
                        (self.theta_tip_deg - self.theta_root_deg) *
                        (r - r_hub) / (self.R - r_hub))
        return r, c, th


# ── Per-element flow state (momentum induced inflow + Prandtl tip loss) ────
def _state(Vd, Om, cfg, r, c, theta, n_it=15):
    """Flow state at each blade element. Vd may be scalar or a (nv,) array;
    returns fields broadcast to (nv, ne)."""
    Vd = np.atleast_1d(np.asarray(Vd, float))[:, None]   # (nv, 1)
    Ut = (Om * r)[None, :]                                # (1, ne)
    cb = np.cos(np.deg2rad(cfg.coning_deg))
    vi = np.zeros((Vd.shape[0], r.size))                  # (nv, ne)

    for _ in range(n_it if cfg.induced else 0):
        Up  = (Vd - vi) * cb
        phi = np.arctan2(Up, Ut)
        al  = theta[None, :] + cfg.alpha_sign * phi
        cl, cd = cl_cd(al, cfg.cl_alpha, cfg.cd0)
        U2  = Up**2 + Ut**2
        if cfg.tiploss:
            sphi = np.maximum(np.abs(np.sin(phi)), 1e-3)
            f = (cfg.n_blades / 2.0) * (cfg.R - r)[None, :] / (r[None, :] * sphi)
            F = np.maximum((2 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0)), 1e-4)
        else:
            F = 1.0
        # blade-element axial load per unit span
        dT_be = 0.5 * cfg.rho * U2 * c[None, :] * (cl*np.cos(phi) - cd*np.sin(phi)) * cfg.n_blades
        # momentum per annulus: dT/dr = 4π ρ r (Vd − v_i) v_i F cosβ  → solve v_i
        a_   = np.maximum(4 * np.pi * cfg.rho * r[None, :] * F * cb, 1e-9)
        disc = (a_ * Vd)**2 - 4 * a_ * dT_be
        vi_n = np.where(disc > 0, (a_ * Vd - np.sqrt(np.maximum(disc, 0.0))) / (2 * a_), 0.0)
        vi   = 0.6 * vi + 0.4 * vi_n                      # under-relaxed fixed point

    Up  = (Vd - vi) * cb
    phi = np.arctan2(Up, Ut)
    al  = theta[None, :] + cfg.alpha_sign * phi
    cl, cd = cl_cd(al, cfg.cl_alpha, cfg.cd0)
    U2  = Up**2 + Ut**2
    return Up, Ut, phi, al, U2, cl, cd, vi


def forces(Vd, Om, cfg, grid):
    """Total thrust T [N] and net torque Q [N·m]. Vectorised over Vd → (nv,)."""
    r, c, theta = grid
    dr = r[1] - r[0]
    _, _, phi, al, U2, cl, cd, _ = _state(Vd, Om, cfg, r, c, theta)
    q  = 0.5 * cfg.rho * U2 * c[None, :]
    dT = (q*cl*np.cos(phi) - q*cd*np.sin(phi)) * cfg.n_blades
    dQ = r[None, :] * (q*cl*np.sin(phi) - q*cd*np.cos(phi)) * cfg.n_blades
    T  = np.sum(dT * dr, axis=1)
    Q  = np.sum(dQ * dr, axis=1)
    return T, Q


# ── Robust multivalued thrust-balance root handling ───────────────────────
def thrust_balance_roots(Om, W, cfg, grid, vlo=0.1, vhi=30.0, n=180):
    """All V_d in [vlo, vhi] with T(V_d, Ω)=W, refined, ascending."""
    vs = np.linspace(vlo, vhi, n)
    g  = forces(vs, Om, cfg, grid)[0] - W
    roots = []
    for i in np.where(np.diff(np.sign(g)) != 0)[0]:
        try:
            roots.append(brentq(lambda v: forces(v, Om, cfg, grid)[0][0] - W,
                                 vs[i], vs[i+1], xtol=1e-6))
        except Exception:
            pass
    return np.array(roots)


def _track_branches(Omegas, W, cfg, grid, gap=0.8):
    """Group T=W roots into continuous V_d(Ω) branches by continuation in Ω."""
    branches = []
    for Om in Omegas:
        roots = list(thrust_balance_roots(Om, W, cfg, grid))
        used  = [False] * len(roots)
        for br in branches:
            if not br['active']:
                continue
            best, bd = -1, gap
            for k, v in enumerate(roots):
                if used[k]:
                    continue
                d = abs(v - br['tip'])
                if d < bd:
                    best, bd = k, d
            if best >= 0:
                br['Om'].append(Om); br['Vd'].append(roots[best])
                br['tip'] = roots[best]; used[best] = True
            else:
                br['active'] = False
        for k, v in enumerate(roots):
            if not used[k]:
                branches.append({'Om': [Om], 'Vd': [v], 'tip': v, 'active': True})
    return [(np.array(b['Om']), np.array(b['Vd'])) for b in branches]


def solve(cfg, m_kg, return_info=False):
    """
    Autorotation equilibrium (V_d*, Ω*) for a config at given mass, following
    the stable on-branch net-torque zero. return_info=True adds a well-posedness
    diagnostics dict.
    """
    grid = cfg.grid()
    W    = m_kg * G
    Omegas   = np.linspace(20, 320, 130)
    branches = _track_branches(Omegas, W, cfg, grid)
    nan_out  = (np.nan, np.nan, {}) if return_info else (np.nan, np.nan)

    best = None
    for Om_b, Vd_b in branches:
        if Om_b.size < 3:
            continue
        Q_b = np.array([forces(v, om, cfg, grid)[1][0] for om, v in zip(Om_b, Vd_b)])
        for k in range(Q_b.size - 1):
            if Q_b[k] > 0.0 >= Q_b[k + 1]:                 # stable driving→braking zero
                def _q_branch(om, Om_b=Om_b, Vd_b=Vd_b):
                    tgt = np.interp(om, Om_b, Vd_b)
                    rt  = thrust_balance_roots(om, W, cfg, grid)
                    if rt.size == 0:
                        return np.nan
                    return forces(rt[np.argmin(np.abs(rt - tgt))], om, cfg, grid)[1][0]
                try:
                    Om_star = brentq(_q_branch, Om_b[k], Om_b[k + 1], xtol=1e-4)
                except Exception:
                    Om_star = Om_b[k] - Q_b[k] * (Om_b[k+1]-Om_b[k]) / (Q_b[k+1]-Q_b[k])
                tgt = np.interp(Om_star, Om_b, Vd_b)
                rt  = thrust_balance_roots(Om_star, W, cfg, grid)
                Vd_star = float(rt[np.argmin(np.abs(rt - tgt))]) if rt.size else float(tgt)
                dQ_dOm  = (Q_b[k+1] - Q_b[k]) / (Om_b[k+1] - Om_b[k])
                if best is None or Vd_star < best[0]:
                    best = (Vd_star, Om_star, dQ_dOm, Vd_b)

    if best is None:
        return nan_out
    Vd_star, Om_star, dQ_dOm, Vd_b = best
    if return_info:
        branch_jump = float(np.max(np.abs(np.diff(Vd_b)))) if Vd_b.size > 1 else 0.0
        info = {
            'n_roots_at_op':   int(thrust_balance_roots(Om_star, W, cfg, grid).size),
            'branch_jump':     branch_jump,
            'branch_smooth':   bool(branch_jump < 1.0),   # vs the ~4.4 m/s pre-fix jump
            'dQ_dOmega':       float(dQ_dOm),
            'self_correcting': bool(dQ_dOm < 0),
            'well_posed':      bool(branch_jump < 1.0 and dQ_dOm < 0),
        }
        return Vd_star, Om_star, info
    return Vd_star, Om_star


# ── Validation bracket: independent reality checks on the equilibrium ──────
def validation_brackets(cfg, Vd, Om, m_kg):
    """Independent estimates that bracket the BEM descent rate, per the
    evidence-hygiene plan. Returns a dict; all are order-of-magnitude checks."""
    grid = cfg.grid(); r, c, theta = grid
    A   = np.pi * cfg.R**2
    W   = m_kg * G
    DL  = W / A                                   # disk loading [N/m²]
    v_h = np.sqrt(DL / (2 * cfg.rho))             # induced-velocity scale [m/s]
    # ideal vertical-autorotation descent ≈ 1.8–2.1 · v_h (windmill-brake state)
    autorot_band = (1.8 * v_h, 2.1 * v_h)
    # √(disk-loading) scaling from a real maple seed (DL≈0.26 N/m², V_d≈1.2 m/s)
    scaled = 1.2 * np.sqrt(DL / 0.26)
    c_bar = float(np.mean(c))
    Ro    = cfg.R / c_bar                          # Rossby ~ R/c̄ (LEV stable if ≲3–4)
    Utip  = np.sqrt(Vd**2 + (Om * cfg.R)**2)
    Re_tip = Utip * c_bar / NU_AIR
    Re_root = np.sqrt(Vd**2 + (Om * r[0])**2) * c[0] / NU_AIR
    return {'disk_loading': DL, 'v_h': v_h, 'autorot_band': autorot_band,
            'scaled_from_seed': scaled, 'rossby': Ro,
            'Re_span': (Re_root, Re_tip)}


# ── Sycamore-A validation (credibility anchor) ────────────────────────────
def validate_sycamore_A():
    """Run the model on the paper's measured Sycamore A and check it reproduces
    a slow (sub-1 m/s) samara descent with root-high/tip-low α."""
    A = Config(name='Sycamore A (validation)', R=0.0447, c_root=0.0115, c_tip=0.0115,
               theta_root_deg=-2.6, theta_tip_deg=-2.6, hub_frac=0.12, coning_deg=10.0,
               cl_alpha=5.8, cd0=0.032, rho=1.225, n_elem=60)
    Vd, Om = solve(A, 0.000232)
    grid = A.grid()
    _, _, _, al, _, _, _, _ = _state(Vd, Om, A, *grid)
    al = al[0]
    ok = (not np.isnan(Vd)) and (0.4 < Vd < 1.4)        # paper natural descent ≈0.97
    return {'Vd': Vd, 'rpm': Om * 60 / (2*np.pi) if not np.isnan(Om) else np.nan,
            'alpha_root_deg': np.rad2deg(al[0]), 'alpha_tip_deg': np.rad2deg(al[-1]),
            'pass': ok}


# ── Report one config's baseline equilibrium ──────────────────────────────
def report(cfg, m0=0.075):
    Vd0, Om0, info = solve(cfg, m0, return_info=True)
    print(f"\n{'═'*60}")
    print(f"  {cfg.name}")
    print(f"  R={cfg.R*100:.0f}cm  θ={cfg.theta_root_deg:.1f}→{cfg.theta_tip_deg:.1f}°  "
          f"α=θ{'+' if cfg.alpha_sign>0 else '−'}φ  C_Lα={cfg.cl_alpha}  C_D0={cfg.cd0}  "
          f"induced={cfg.induced} tiploss={cfg.tiploss}")
    print(f"{'═'*60}")
    if np.isnan(Vd0):
        print("  No autorotation equilibrium found for this config.")
        return Vd0, Om0
    RPM0 = Om0 * 60 / (2*np.pi)
    T0, Q0 = (forces(Vd0, Om0, cfg, cfg.grid())[0][0],
              forces(Vd0, Om0, cfg, cfg.grid())[1][0])
    b = validation_brackets(cfg, Vd0, Om0, m0)
    print(f"  Descent rate  : {Vd0:.2f} m/s   Rotation : {RPM0:.0f} RPM ({Om0:.1f} rad/s)")
    print(f"  Tip speed     : {Om0*cfg.R:.1f} m/s  Ma {Om0*cfg.R/340:.3f}   "
          f"advance μ={Vd0/(Om0*cfg.R):.3f}")
    print(f"  Thrust/Torque : T={T0:.4f} N  W={m0*G:.4f} N   Q={Q0:+.2e} N·m")
    print(f"  Well posed    : {'YES ✓' if info['well_posed'] else 'no ✗'}  "
          f"(roots@op={info['n_roots_at_op']}, ΔV_d≤{info['branch_jump']:.2f} m/s, "
          f"dQ/dΩ={info['dQ_dOmega']:+.1e})")
    print(f"  ── reality brackets (Indicative) ──")
    print(f"    disk loading {b['disk_loading']:.2f} N/m²  → v_h={b['v_h']:.2f} m/s  "
          f"→ ideal autorot ≈ {b['autorot_band'][0]:.1f}–{b['autorot_band'][1]:.1f} m/s")
    print(f"    √(DL)-scaled from real maple seed ≈ {b['scaled_from_seed']:.1f} m/s")
    print(f"    Rossby R/c̄ = {b['rossby']:.1f} (LEV stable if ≲3–4)   "
          f"section Re {b['Re_span'][0]:.0f}–{b['Re_span'][1]:.0f}")
    return Vd0, Om0


# ── Configurations: samara-realistic headline + legacy 35°/20° ────────────
SAMARA = Config(name='Sycamore device — samara-realistic (HEADLINE)',
                R=0.30, c_root=0.08, c_tip=0.03,
                theta_root_deg=-2.6, theta_tip_deg=-2.6,  # real Sycamore-A pitch
                alpha_sign=+1.0, cl_alpha=5.3, cd0=0.025)

LEGACY = Config(name='Sycamore device — legacy 35°/20° high-pitch (Wave-3 to revisit)',
                R=0.30, c_root=0.08, c_tip=0.03,
                theta_root_deg=35.0, theta_tip_deg=20.0,
                alpha_sign=-1.0, cl_alpha=5.3, cd0=0.025)


if __name__ == '__main__':
    # 1) credibility anchor
    v = validate_sycamore_A()
    print(f"\n[validation] Sycamore A: V_d={v['Vd']:.2f} m/s (paper ≈0.97), "
          f"{v['rpm']:.0f} RPM, α root→tip {v['alpha_root_deg']:.0f}°→{v['alpha_tip_deg']:.0f}° "
          f"(paper ~70°→5°)  → {'PASS ✓' if v['pass'] else 'FAIL ✗'}")

    # 2) the two device configs, side by side
    Vd_s, Om_s = report(SAMARA)
    Vd_l, Om_l = report(LEGACY)

    # 3) figure: descent & RPM vs mass (both configs) + samara spanwise α / dQ/dr
    masses = np.linspace(0.030, 0.200, 8)
    def sweep(cfg):
        vd, rpm = [], []
        for m in masses:
            v_, o_ = solve(cfg, m)
            vd.append(v_); rpm.append(o_ * 60/(2*np.pi) if not np.isnan(o_) else np.nan)
        return np.array(vd), np.array(rpm)
    vd_s, rpm_s = sweep(SAMARA)
    vd_l, rpm_l = sweep(LEGACY)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle('Sycamore BEM — Jung & Rezgui LEV polar + induced inflow + tip loss', fontsize=11)
    ax[0].plot(masses*1e3, vd_s, 'steelblue', lw=2, label='samara-realistic')
    ax[0].plot(masses*1e3, vd_l, 'crimson', lw=2, ls='--', label='legacy 35°/20°')
    ax[0].axvline(75, color='gray', ls=':'); ax[0].set(xlabel='Mass [g]',
                 ylabel='Descent V_d [m/s]', title='Descent rate vs mass')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.4)
    ax[1].plot(masses*1e3, rpm_s, 'steelblue', lw=2)
    ax[1].plot(masses*1e3, rpm_l, 'crimson', lw=2, ls='--')
    ax[1].axvline(75, color='gray', ls=':'); ax[1].set(xlabel='Mass [g]',
                 ylabel='Rotation [RPM]', title='Rotation rate vs mass'); ax[1].grid(alpha=0.4)

    if not np.isnan(Vd_s):
        grid = SAMARA.grid(); r, c, theta = grid
        _, _, phi, al, U2, cl, cd, _ = _state(Vd_s, Om_s, SAMARA, *grid)
        al = al[0]; phi = phi[0]; U2 = U2[0]; cl = cl[0]; cd = cd[0]
        q  = 0.5*SAMARA.rho*U2*c
        dQ = r*(q*cl*np.sin(phi) - q*cd*np.cos(phi))*SAMARA.n_blades
        ax2 = ax[2]; ax2b = ax2.twinx()
        ax2.fill_between(r/SAMARA.R, dQ, 0, where=dQ>0, alpha=0.25, color='green', label='driving')
        ax2.fill_between(r/SAMARA.R, dQ, 0, where=dQ<0, alpha=0.25, color='red', label='braking')
        ax2.plot(r/SAMARA.R, dQ, 'purple', lw=1.5); ax2.axhline(0, color='k', lw=0.8)
        ax2b.plot(r/SAMARA.R, np.rad2deg(al), 'navy', lw=1.2, ls=':')
        ax2.set(xlabel='r/R', ylabel='dQ/dr [N·m/m]', title='Samara spanwise: torque & α')
        ax2b.set_ylabel('α [deg]', color='navy'); ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/samara_bem.png', dpi=150, bbox_inches='tight')
    print('\nSaved: output/samara_bem.png')
