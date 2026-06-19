"""
Rothermel (1972) surface fire spread model.

Implements the semi-empirical rate-of-spread equations from:
  Rothermel, R.C. (1972). USFS Research Paper INT-115.

Inputs are in SI; all internal calculations use the original US customary
units (ft, lb, BTU, min) to match the published coefficients, then convert
outputs back to SI.

Includes:
  - Core Rothermel equations
  - Five standard Anderson (1982) fuel models
  - Sweeps: ROS vs. wind speed, ROS vs. fuel moisture
  - Simple Huygens elliptical fire perimeter growth simulation

Run with: ~/ds/bin/python rothermel.py
Outputs : output/rothermel_ros.png, output/fire_perimeter.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('output', exist_ok=True)

# ── Unit conversion factors ───────────────────────────────────────────────
MPH_TO_FTMIN   = 88.0       # 1 mph = 88 ft/min
MS_TO_FTMIN    = 196.85
FTMIN_TO_MS    = 1 / MS_TO_FTMIN
FTMIN_TO_MMIN  = 0.3048

# ── Anderson (1982) fuel models ───────────────────────────────────────────
# Simplified single-fuel-class representation.
# w0 [lb/ft²], delta [ft], sigma [1/ft], h [BTU/lb], Mx [fraction]
FUEL_MODELS = {
    1: {'name': 'Short Grass',          'w0': 0.034, 'delta': 1.0,  'sigma': 3500, 'h': 8000, 'Mx': 0.12},
    2: {'name': 'Timber/Grass Mix',     'w0': 0.092, 'delta': 1.0,  'sigma': 3000, 'h': 8000, 'Mx': 0.15},
    5: {'name': 'Brush',                'w0': 0.046, 'delta': 2.0,  'sigma': 2000, 'h': 8000, 'Mx': 0.20},
    7: {'name': 'Southern Rough',       'w0': 0.115, 'delta': 2.5,  'sigma': 1750, 'h': 8000, 'Mx': 0.40},
    8: {'name': 'Closed Timber Litter', 'w0': 0.069, 'delta': 0.2,  'sigma': 2000, 'h': 8000, 'Mx': 0.30},
}

# Mineral content constants (standard defaults)
S_T = 0.055   # total mineral fraction
S_e = 0.010   # effective mineral fraction
rho_p = 32.0  # oven-dry particle density [lb/ft³]

def rothermel_ros(fuel, M, U_ftmin, slope_deg=0.0):
    """
    Rate of spread from Rothermel (1972).

    Parameters
    ----------
    fuel      : dict  from FUEL_MODELS
    M         : float  moisture content [fraction, e.g. 0.08 for 8%]
    U_ftmin   : float  wind speed at mid-flame height [ft/min]
    slope_deg : float  terrain slope [degrees, upslope positive]

    Returns
    -------
    ros : float  rate of spread [ft/min]; NaN if M >= Mx
    """
    w0    = fuel['w0']
    delta = fuel['delta']
    sigma = fuel['sigma']
    h     = fuel['h']
    Mx    = fuel['Mx']

    if M >= Mx:
        return np.nan

    # Moisture damping
    r_M   = M / Mx
    eta_M = 1.0 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3
    eta_M = max(eta_M, 0.0)

    # Mineral damping
    eta_s = 0.174 / (S_e ** 0.19)
    eta_s = min(eta_s, 1.0)

    # Packing ratio
    rho_b = w0 / delta          # bulk density [lb/ft³]
    beta  = rho_b / rho_p       # packing ratio

    # Optimum packing ratio and reaction velocity
    beta_op  = 3.348 / (sigma ** 0.8189)
    Gamma_max = (sigma**1.5) / (495 + 0.0594 * sigma**1.5)
    A         = 133.0 / (sigma ** 0.7913)
    ratio     = beta / beta_op
    Gamma     = Gamma_max * (ratio**A) * np.exp(A * (1.0 - ratio))

    # Net fuel loading (mineral-free), canonical Rothermel/BehavePlus form (Andrews 2018)
    w_n = w0 / (1.0 + S_T)

    # Reaction intensity [BTU/ft²/min]
    I_R = Gamma * w_n * h * eta_M * eta_s

    # Propagating flux ratio (Rothermel eq. 42)
    xi = np.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)

    # Effective heating number and heat of preignition
    eps   = np.exp(-138.0 / sigma)
    Q_ig  = 250.0 + 1116.0 * M

    # Wind factor
    C     = 7.47 * np.exp(-0.133 * sigma**0.55)
    B     = 0.02526 * sigma**0.54
    E     = 0.715  * np.exp(-3.59e-4 * sigma)
    Phi_W = C * (max(U_ftmin, 0.0)**B) * (ratio**(-E))

    # Slope factor
    tan_phi  = np.tan(np.deg2rad(slope_deg))
    Phi_S    = 5.275 * (beta**(-0.3)) * (tan_phi**2)

    # Rate of spread [ft/min]
    ros = I_R * xi * (1.0 + Phi_W + Phi_S) / (rho_b * eps * Q_ig)
    return max(ros, 0.0)

# ── Plot 1: ROS vs wind speed for five fuel models ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Rothermel (1972) Fire Spread Model  —  Project Sycamore', fontsize=11)

wind_ms   = np.linspace(0, 15, 60)           # m/s
wind_ftmin = wind_ms * MS_TO_FTMIN
M_fixed   = 0.08                              # 8% moisture

colors = ['firebrick', 'darkorange', 'goldenrod', 'steelblue', 'mediumseagreen']
for (fid, fuel), col in zip(FUEL_MODELS.items(), colors):
    ros_ft = np.array([rothermel_ros(fuel, M_fixed, U) for U in wind_ftmin])
    ros_m  = ros_ft * FTMIN_TO_MMIN
    axes[0].plot(wind_ms, ros_m, lw=2, color=col, label=f"FM{fid} {fuel['name']}")

axes[0].set(xlabel='Wind speed [m/s]', ylabel='Rate of spread [m/min]',
            title=f'ROS vs Wind Speed  (M={M_fixed*100:.0f}%, slope=0°)')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.4)

# ── Plot 2: ROS vs fuel moisture (FM1 short grass, 5 m/s wind) ─────────────
fuel_ref = FUEL_MODELS[1]
U_ref    = 5.0 * MS_TO_FTMIN   # 5 m/s
slopes   = [0, 15, 30]

for slope in slopes:
    M_range = np.linspace(0.02, fuel_ref['Mx'] * 0.95, 50)
    ros_m   = np.array([rothermel_ros(fuel_ref, M, U_ref, slope) * FTMIN_TO_MMIN
                        for M in M_range])
    axes[1].plot(M_range * 100, ros_m, lw=2, label=f'slope={slope}°')

axes[1].set(xlabel='Fuel moisture content [%]', ylabel='Rate of spread [m/min]',
            title=f'ROS vs Moisture  (FM1 Short Grass, U=5 m/s)')
axes[1].axvline(fuel_ref['Mx']*100, ls='--', color='k', lw=0.8, label='Mx (extinction)')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('output/rothermel_ros.png', dpi=150, bbox_inches='tight')
print('Saved: output/rothermel_ros.png')

# ── Fire perimeter growth (Huygens elliptical model) ──────────────────────
# Wind-driven fire perimeter is approximated as an expanding ellipse where
# head-fire ROS sets the major axis growth rate and backing-fire ROS sets
# the minor axis. Length-to-breadth ratio is empirical (Anderson 1983):
#   LB = 0.936 * exp(0.2566 * U_mph) + 0.461 * exp(-0.1548 * U_mph) - 0.397

from shapely.geometry import Point
from shapely.affinity  import scale, rotate

U_sim   = 5.0                          # wind speed [m/s]
U_mph   = U_sim * 2.237
fuel_sim = FUEL_MODELS[2]              # FM2 Timber/Grass Mix
M_sim   = 0.10                         # 10% moisture

ros_head = rothermel_ros(fuel_sim, M_sim, U_sim * MS_TO_FTMIN) * FTMIN_TO_MMIN  # m/min
LB   = 0.936 * np.exp(0.2566*U_mph) + 0.461 * np.exp(-0.1548*U_mph) - 0.397
ros_back = ros_head / (LB**2)   # backing rate [m/min]

print(f"\nFire spread simulation: FM{2}, U={U_sim:.0f} m/s, M={M_sim*100:.0f}%")
print(f"  Head fire ROS : {ros_head:.2f} m/min  ({ros_head*60:.0f} m/h)")
print(f"  Length/Breadth: {LB:.2f}")
print(f"  Backing ROS   : {ros_back:.2f} m/min")

t_max = 60     # minutes
dt    = 1
times = np.arange(0, t_max + dt, dt)

fig2, ax2 = plt.subplots(figsize=(7, 6))
cmap = plt.cm.plasma
for i, t in enumerate(times[::10]):
    # Semi-major axis (head direction), semi-minor axis
    a = ros_head * t / 2 + 1e-3
    b = a / LB
    ellipse = scale(Point(0, 0).buffer(1.0), xfact=b, yfact=a)
    x, y = ellipse.exterior.xy
    color = cmap(i / len(times[::10]))
    ax2.fill(x, y, alpha=0.15, color=color)
    ax2.plot(x, y, color=color, lw=1.2, label=f't={t:.0f} min' if i % 2 == 0 else None)

ax2.axhline(0, color='k', lw=0.5); ax2.axvline(0, color='k', lw=0.5)
arrow_len = ros_head * t_max * 0.55
ax2.annotate('', xy=(0, arrow_len), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
ax2.text(0.5, arrow_len * 0.5, f'Wind {U_sim:.0f} m/s', color='navy', fontsize=9, ha='left')
ax2.set_aspect('equal')
ax2.set(xlabel='Cross-wind [m]', ylabel='Along-wind [m]',
        title=f'Fire perimeter growth  (FM2, U={U_sim:.0f} m/s, M={M_sim*100:.0f}%)\n'
              f't = 0 → {t_max} min, every 10 min')
ax2.legend(fontsize=7, loc='lower right', ncol=2)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/fire_perimeter.png', dpi=150, bbox_inches='tight')
print('Saved: output/fire_perimeter.png')
