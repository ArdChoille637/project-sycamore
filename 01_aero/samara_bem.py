"""
Reduced-order Blade-Element Momentum (BEM) model for samara autorotation.

Solves for steady-state descent velocity and rotation rate of a single-wing
rotating UAV. This is the aerodynamic proof-of-concept for Project Sycamore's
morphing-wing bi-modal design.

Physics
-------
Blade-element theory resolves aerodynamic forces at each spanwise station:
  - Thrust dT/dr = N(L'cosφ − D'sinφ)
  - Torque dQ/dr = Nr(L'sinφ − D'cosφ)
where φ = arctan(V_d / Ω r) is the local inflow angle.

At steady autorotation two conditions must hold simultaneously:
  (1) Thrust T = Weight W       (vertical force balance)
  (2) Net torque Q = 0          (no engine, rotor self-sustains)

Solver
------
The thrust balance T(V_d,Ω)=W is multivalued in V_d, so a plain brentq over
the whole interval returns an arbitrary root and silently jumps branches as Ω
varies. Instead we: enumerate ALL T=W roots on a grid, assemble them into
continuous V_d(Ω) branches by continuation, and locate the autorotation
equilibrium as the *stable* net-torque zero (Q: driving→braking, dQ/dΩ<0) on a
single branch. The post-stall polar is C¹-blended so T(V_d) is smooth: a hard
stall switch produced 5–13 spurious roots and a V_d 4.4→8.8 m/s branch jump at
the operating Ω, making the old reported equilibrium partly a numerical
artifact. `solve(m, return_info=True)` returns well-posedness diagnostics.

Pitch geometry note
-------------------
Samara-inspired UAVs require higher collective pitch than a helicopter —
typically 30–40° at root, 15–25° at tip — so that the inner blade stations
(large φ) operate in positive AoA and drive the rotation while the tip brakes.
The pitch angles below are representative of Stahl et al. (2019) style designs.

Reference: Lentink et al. (2009) Science 324:1438; Yasuda & Azuma (1997)
           J.Theor.Biol. 185:313.

Run with:  ~/ds/bin/python samara_bem.py
Outputs :  output/samara_bem.png
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('output', exist_ok=True)

# ── UAV design parameters (edit here) ─────────────────────────────────────
R         = 0.30            # tip radius [m]
c_root    = 0.08            # root chord [m]
c_tip     = 0.03            # tip chord [m]
theta_root = np.deg2rad(35) # root collective pitch [rad]  (samara-scale requires ~35°)
theta_tip  = np.deg2rad(20) # tip pitch [rad]
N_BLADES  = 1               # samara = single wing
RHO       = 1.167           # air density at ~500 m MSL [kg/m³]
N_ELEM    = 50              # spanwise integration elements

# ── Spanwise discretisation ───────────────────────────────────────────────
r_hub = 0.15 * R            # 15% root cutout (seed body region)
r     = np.linspace(r_hub, R, N_ELEM)
dr    = r[1] - r[0]
c     = c_root + (c_tip - c_root) * (r - r_hub) / (R - r_hub)
theta = theta_root + (theta_tip - theta_root) * (r - r_hub) / (R - r_hub)

# ── Thin-plate airfoil model with Viterna post-stall (C¹-blended) ────────
# Pre-stall: thin airfoil (2π slope).
# Post-stall: Viterna (1982) flat-plate extension — CL drops, CD rises sharply.
# This is essential for the tip-braking region that sustains autorotation.
#
# The pre/post branches are joined by a smooth logistic weight rather than a
# hard `where` switch at ALPHA_STALL. The hard switch put a kink in CL(α) at
# every spanwise element's stall crossing, which made the summed thrust
# T(V_d) non-monotonic — at the operating Ω, T(V_d)=W had 5–13 spurious roots
# and the root-finder branch-jumped (V_d 4.4→8.8 m/s across Ω≈107), so the
# reported equilibrium was partly a numerical artifact. The blend makes
# cl_cd C¹ (in fact C∞), so T(V_d) is smooth and the equilibrium is well
# posed. BLEND→0 recovers the original hard switch.
ALPHA_STALL = np.deg2rad(10)
CD_MAX      = 1.2             # flat plate maximum drag coefficient
BLEND       = np.deg2rad(2.0) # stall-transition half-width for C¹ blending

def cl_cd(alpha):
    """Returns (CL, CD) for AoA [rad], C¹-continuous across stall."""
    sign = np.sign(alpha)
    a    = np.abs(alpha)
    # smooth pre→post weight: ~1 below stall, ~0 above, continuous derivative
    w    = 1.0 / (1.0 + np.exp((a - ALPHA_STALL) / BLEND))

    cl_pre  = 2 * np.pi * alpha
    cl_post = sign * (CD_MAX / 2) * np.sin(2 * a)

    cd_pre  = 0.01 + 0.05 * alpha**2
    cd_post = CD_MAX * np.sin(a)**2 + 0.01 * np.cos(a)**2

    return w * cl_pre + (1 - w) * cl_post, w * cd_pre + (1 - w) * cd_post

# ── Force integrals at a given operating point ────────────────────────────
def forces(Vd, Omega):
    Ut   = Omega * r
    Veff = np.sqrt(Ut**2 + Vd**2)
    phi  = np.arctan2(Vd, Ut)        # inflow angle from rotor plane [rad]
    alpha = theta - phi
    cl, cd = cl_cd(alpha)
    q    = 0.5 * RHO * Veff**2 * c   # dynamic pressure per unit span [N/m]
    T    = N_BLADES * np.sum((q*cl*np.cos(phi) - q*cd*np.sin(phi)) * dr)
    Q    = N_BLADES * np.sum(r * (q*cl*np.sin(phi) - q*cd*np.cos(phi)) * dr)
    return T, Q

# ── Thrust-balance root handling (robust, branch-aware) ──────────────────
# The old solver called brentq on the whole [0.1, 30] interval, which returns
# *an* arbitrary root and silently jumps branches as Ω varies (the source of
# the V_d 4.4→8.8 m/s discontinuity). Instead we enumerate ALL T(V_d)=W roots
# on a grid and select the physical branch = lowest descent rate (the
# windmill-brake autorotation state); higher-V_d roots are deep-stall states.

def _thrust_curve(Vd_arr, Om, W):
    """Vectorised T(V_d) − W over a V_d array at fixed Ω (one shot)."""
    Vd    = np.asarray(Vd_arr)[:, None]          # (nv, 1)
    Ut    = Om * r[None, :]                       # (1, ne)
    Veff  = np.sqrt(Ut**2 + Vd**2)
    phi   = np.arctan2(Vd, Ut)
    alpha = theta[None, :] - phi
    cl, cd = cl_cd(alpha)
    q     = 0.5 * RHO * Veff**2 * c[None, :]
    T     = N_BLADES * np.sum((q*cl*np.cos(phi) - q*cd*np.sin(phi)) * dr, axis=1)
    return T - W

def thrust_balance_roots(Om, W, vlo=0.1, vhi=30.0, n=400):
    """All V_d in [vlo, vhi] with T(V_d, Ω)=W, refined and ascending."""
    vs = np.linspace(vlo, vhi, n)
    g  = _thrust_curve(vs, Om, W)
    roots = []
    for i in np.where(np.diff(np.sign(g)) != 0)[0]:
        try:
            roots.append(brentq(lambda v: forces(v, Om)[0] - W,
                                 vs[i], vs[i+1], xtol=1e-6))
        except Exception:
            pass
    return np.array(roots)

def vd_physical(Om, W):
    """Lowest-descent (physical autorotation) V_d root at this Ω, or nan."""
    roots = thrust_balance_roots(Om, W)
    return roots[0] if roots.size else np.nan

# ── Branch continuation: assemble continuous V_d(Ω) autorotation branches ─
def _track_branches(Omegas, W, gap=0.8):
    """
    Group the T(V_d)=W roots into continuous V_d(Ω) branches by continuation
    in Ω (match each root to the nearest active branch tip; unmatched roots
    seed new branches). Returns a list of (Om_array, Vd_array).

    This is the heart of the well-posedness fix: the thrust balance is
    multivalued (a slow-descent attached branch, a fast-descent stalled
    branch, and a spurious near-zero branch that appears at higher Ω), and the
    physical autorotation branch is the MIDDLE one — not the lowest root. The
    old single-brentq search could not see this and jumped branches.
    """
    branches = []   # each: {'Om':[], 'Vd':[], 'tip':float, 'active':bool}
    for Om in Omegas:
        roots = list(thrust_balance_roots(Om, W))
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

# ── Branch-aware autorotation solver ──────────────────────────────────────
def solve(m_kg, return_info=False):
    """
    Find (V_d*, Omega*) for autorotation at given UAV mass.

    The thrust balance T(V_d)=W is multivalued, so its roots are first
    assembled into continuous branches by continuation (`_track_branches`).
    The physical autorotation equilibrium is the *stable* net-torque zero on a
    branch — Q passing from driving (Q>0) to braking (Q<0), i.e. dQ/dΩ<0, a
    self-correcting RPM. Among such crossings the slowest-descent one is
    returned. This replaces the old single-brentq search, which returned an
    arbitrary root and branch-jumped (V_d 4.4→8.8 m/s) at the operating Ω.

    return_info=True also returns a diagnostics dict:
      n_roots_at_op  — T=W multiplicity at Ω* (context, not a defect)
      branch_jump    — largest V_d step along the SELECTED branch [m/s]
      branch_smooth  — bool(branch_jump < 0.5): the tracked branch is continuous
      dQ_dOmega      — on-branch torque slope at the zero (<0 ⇒ self-correcting)
      self_correcting, well_posed
    """
    W = m_kg * 9.81
    Omegas   = np.linspace(20, 300, 280)
    branches = _track_branches(Omegas, W)
    nan_out  = (np.nan, np.nan, {}) if return_info else (np.nan, np.nan)

    best = None   # (Vd_star, Om_star, dQ_dOm, Om_b, Vd_b)
    for Om_b, Vd_b in branches:
        if Om_b.size < 3:
            continue
        Q_b = np.array([forces(v, om)[1] for om, v in zip(Om_b, Vd_b)])
        for k in range(Q_b.size - 1):
            if Q_b[k] > 0.0 >= Q_b[k + 1]:                 # stable driving→braking zero
                # refine Ω* staying on THIS branch (root nearest the interpolated tip)
                def _q_branch(om, Om_b=Om_b, Vd_b=Vd_b):
                    tgt   = np.interp(om, Om_b, Vd_b)
                    roots = thrust_balance_roots(om, W)
                    if roots.size == 0:
                        return np.nan
                    return forces(roots[np.argmin(np.abs(roots - tgt))], om)[1]
                try:
                    Om_star = brentq(_q_branch, Om_b[k], Om_b[k + 1], xtol=1e-4)
                except Exception:
                    Om_star = (Om_b[k] - Q_b[k] *
                               (Om_b[k+1] - Om_b[k]) / (Q_b[k+1] - Q_b[k]))
                tgt     = np.interp(Om_star, Om_b, Vd_b)
                roots   = thrust_balance_roots(Om_star, W)
                Vd_star = float(roots[np.argmin(np.abs(roots - tgt))]) if roots.size else tgt
                dQ_dOm  = (Q_b[k+1] - Q_b[k]) / (Om_b[k+1] - Om_b[k])
                if best is None or Vd_star < best[0]:      # slowest-descent stable eq.
                    best = (Vd_star, Om_star, dQ_dOm, Om_b, Vd_b)

    if best is None:
        return nan_out
    Vd_star, Om_star, dQ_dOm, Om_b, Vd_b = best

    if return_info:
        branch_jump = float(np.max(np.abs(np.diff(Vd_b)))) if Vd_b.size > 1 else 0.0
        info = {
            'n_roots_at_op':   int(thrust_balance_roots(Om_star, W).size),
            'branch_jump':     branch_jump,
            'branch_smooth':   bool(branch_jump < 0.5),
            'dQ_dOmega':       float(dQ_dOm),
            'self_correcting': bool(dQ_dOm < 0),
            'well_posed':      bool(branch_jump < 0.5 and dQ_dOm < 0),
        }
        return Vd_star, Om_star, info
    return Vd_star, Om_star

# ── Baseline design ───────────────────────────────────────────────────────
m0 = 0.075   # 75 g baseline UAV
Vd0, Om0, info0 = solve(m0, return_info=True)

if np.isnan(Vd0):
    print("WARNING: BEM solver did not find an autorotation solution.")
    print("  Check blade pitch angles and mass range.")
else:
    RPM0 = Om0 * 60 / (2 * np.pi)
    mu0  = Vd0 / (Om0 * R)    # advance ratio
    T0, Q0 = forces(Vd0, Om0)

    print(f"\n{'─'*52}")
    print(f"  Sycamore BEM  |  m={m0*1000:.0f} g  R={R*100:.0f} cm  single blade")
    print(f"{'─'*52}")
    print(f"  Descent rate  : {Vd0:.2f} m/s  ({Vd0*196.85:.0f} ft/min)")
    print(f"  Rotation rate : {RPM0:.0f} RPM  ({Om0:.1f} rad/s)")
    print(f"  Tip speed     : {Om0*R:.1f} m/s   Ma {Om0*R/340:.3f}")
    print(f"  Advance ratio : μ = {mu0:.3f}")
    print(f"  Disk loading  : {m0*9.81/(np.pi*R**2):.2f} N/m²")
    print(f"  Thrust check  : T = {T0:.4f} N  W = {m0*9.81:.4f} N")
    print(f"  Torque check  : Q = {Q0:.6f} N·m  (target: 0)")
    print(f"{'─'*52}")
    print(f"  Well-posedness (Wave-0 fix):")
    print(f"    T=W roots at op Ω : {info0['n_roots_at_op']}  "
          f"(equilibrium tracked on the stable branch)")
    print(f"    branch continuity : ΔV_d ≤ {info0['branch_jump']:.3f} m/s  "
          f"({'smooth ✓' if info0['branch_smooth'] else 'JUMP ✗'}; was ~4.4 pre-fix)")
    print(f"    dQ/dΩ at crossing : {info0['dQ_dOmega']:+.2e} N·m/(rad/s)  "
          f"({'self-correcting ✓' if info0['self_correcting'] else 'divergent ✗'})")
    print(f"    → equilibrium {'WELL POSED ✓' if info0['well_posed'] else 'still ill-posed ✗'}")
    print(f"{'─'*52}\n")

# ── Parametric sweep: mass 20 g → 200 g ──────────────────────────────────
masses = np.linspace(0.020, 0.200, 25)
Vds, Oms = [], []
for m in masses:
    Vd_i, Om_i = solve(m)
    Vds.append(Vd_i); Oms.append(Om_i)
Vds  = np.array(Vds)
Oms  = np.array(Oms)
RPMs = Oms * 60 / (2 * np.pi)

# ── Blade-spanwise distributions at baseline ─────────────────────────────
if not np.isnan(Vd0):
    Ut_bl  = Om0 * r
    Veff_bl = np.sqrt(Ut_bl**2 + Vd0**2)
    phi_bl  = np.arctan2(Vd0, Ut_bl)
    alpha_bl = theta - phi_bl
    cl_bl, cd_bl = cl_cd(alpha_bl)
    q_bl    = 0.5 * RHO * Veff_bl**2 * c
    dQ_dr   = N_BLADES * r * (q_bl*cl_bl*np.sin(phi_bl) - q_bl*cd_bl*np.cos(phi_bl))
    dT_dr   = N_BLADES * (q_bl*cl_bl*np.cos(phi_bl) - q_bl*cd_bl*np.sin(phi_bl))

# ── Plots ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f'Sycamore BEM  —  R={R*100:.0f} cm  θ_root={np.rad2deg(theta_root):.0f}°  '
             f'θ_tip={np.rad2deg(theta_tip):.0f}°  single blade', fontsize=10)

valid = ~np.isnan(Vds)
ax[0].plot(masses[valid]*1000, Vds[valid], 'steelblue', lw=2)
if not np.isnan(Vd0):
    ax[0].axvline(m0*1000, ls='--', color='crimson', label=f'{m0*1000:.0f} g baseline')
ax[0].set(xlabel='Mass [g]', ylabel='Descent velocity [m/s]', title='Descent Rate vs. Mass')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.4)

ax[1].plot(masses[valid]*1000, RPMs[valid], 'darkorange', lw=2)
if not np.isnan(Om0):
    ax[1].axvline(m0*1000, ls='--', color='crimson')
ax[1].set(xlabel='Mass [g]', ylabel='Rotation rate [RPM]', title='Rotation Rate vs. Mass')
ax[1].grid(alpha=0.4)

if not np.isnan(Vd0):
    ax[2].fill_between(r/R, dQ_dr, 0, where=dQ_dr>0, alpha=0.30, color='green', label='Driving')
    ax[2].fill_between(r/R, dQ_dr, 0, where=dQ_dr<0, alpha=0.30, color='red',   label='Braking')
    ax[2].plot(r/R, dQ_dr, 'purple', lw=1.5)
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set(xlabel='r/R', ylabel='dQ/dr [N·m/m]',
              title=f'Torque distribution  ({m0*1000:.0f} g baseline)')
    ax[2].legend(); ax[2].grid(alpha=0.4)

plt.tight_layout()
out = 'output/samara_bem.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
