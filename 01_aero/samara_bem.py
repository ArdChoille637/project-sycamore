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
Two-step sequential approach (more robust than direct 2D system solve):
  For each Ω in a sweep, find V_d(Ω) via brentq s.t. T(V_d,Ω) = W.
  Evaluate Q at that operating point.  Find Ω* where Q changes sign.

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

# ── Thin-plate airfoil model with Viterna post-stall ─────────────────────
# Pre-stall: thin airfoil (2π slope).
# Post-stall: Viterna (1982) flat-plate extension — CL drops, CD rises sharply.
# This is essential for the tip-braking region that sustains autorotation.
ALPHA_STALL = np.deg2rad(10)
CD_MAX      = 1.2            # flat plate maximum drag coefficient

def cl_cd(alpha):
    """Returns (CL, CD) arrays for the given AoA array [rad]."""
    sign = np.sign(alpha)
    a    = np.abs(alpha)
    pre  = a <= ALPHA_STALL

    cl_pre  = 2 * np.pi * alpha
    cl_post = sign * (CD_MAX / 2) * np.sin(2 * a)

    cd_pre  = 0.01 + 0.05 * alpha**2
    cd_post = CD_MAX * np.sin(a)**2 + 0.01 * np.cos(a)**2

    return np.where(pre, cl_pre, cl_post), np.where(pre, cd_pre, cd_post)

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

# ── Two-step sequential solver ────────────────────────────────────────────
def solve(m_kg):
    """
    Find (V_d*, Omega*) for autorotation at given UAV mass.
    Returns (V_d, Omega) or (nan, nan) if no solution found.
    """
    W = m_kg * 9.81

    Omegas = np.linspace(20, 300, 200)
    Vd_curve = np.full(len(Omegas), np.nan)
    Q_curve  = np.full(len(Omegas), np.nan)

    for i, Om in enumerate(Omegas):
        # At this Om, does T(Vd) = W have a solution in [0.1, 30] m/s?
        try:
            f_lo = forces(0.1,  Om)[0] - W
            f_hi = forces(30.0, Om)[0] - W
            if f_lo * f_hi >= 0:
                continue                    # no sign change → no root
            Vd_i = brentq(lambda v: forces(v, Om)[0] - W, 0.1, 30.0, xtol=1e-4)
            Vd_curve[i] = Vd_i
            Q_curve[i]  = forces(Vd_i, Om)[1]
        except Exception:
            continue

    # Find first sign change in Q
    valid = ~np.isnan(Q_curve)
    if valid.sum() < 2:
        return np.nan, np.nan

    idx = np.where(valid)[0]
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k+1]
        if Q_curve[i] * Q_curve[j] < 0:
            # Linear interpolate Omega* across the sign change
            Om_lo, Om_hi = Omegas[i], Omegas[j]
            Q_lo,  Q_hi  = Q_curve[i], Q_curve[j]
            Om_star = Om_lo - Q_lo * (Om_hi - Om_lo) / (Q_hi - Q_lo)
            Vd_star = brentq(lambda v: forces(v, Om_star)[0] - W, 0.1, 30.0, xtol=1e-5)
            return Vd_star, Om_star

    return np.nan, np.nan

# ── Baseline design ───────────────────────────────────────────────────────
m0 = 0.075   # 75 g baseline UAV
Vd0, Om0 = solve(m0)

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
