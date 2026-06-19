#!/usr/bin/env ~/ds/bin/python
"""
Project Sycamore — Live Simulation Dashboard

Animated window showing three simulation tracks running simultaneously:
  Left  : BEM autorotation — mass sweep results revealed as they compute
  Center: Rothermel fire spread — perimeter growing in real time
  Right : Swarm deployment — agents appearing with coverage tracking live

Run with:  ~/ds/bin/python dashboard.py
"""

import sys, os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from shapely.geometry import Point
from shapely.affinity import scale as geo_scale
import networkx as nx

# ═══════════════════════════════════════════════════════════════════════════
# BEM PHYSICS
# ═══════════════════════════════════════════════════════════════════════════
RHO  = 1.167
R, c_root, c_tip = 0.30, 0.08, 0.03
θ_root, θ_tip    = np.deg2rad(35), np.deg2rad(20)
N_BL, N_EL       = 1, 50

r_hub = 0.15 * R
r_e = np.linspace(r_hub, R, N_EL); dr = r_e[1] - r_e[0]
c_e = c_root + (c_tip - c_root)*(r_e - r_hub)/(R - r_hub)
θ_e = θ_root  + (θ_tip  - θ_root) *(r_e - r_hub)/(R - r_hub)
CD_MAX = 1.2; A_STALL = np.deg2rad(10)

def cl_cd(α):
    s = np.sign(α); a = np.abs(α); pre = a <= A_STALL
    cl = np.where(pre, 2*np.pi*α,  s*(CD_MAX/2)*np.sin(2*a))
    cd = np.where(pre, 0.01+0.05*α**2, CD_MAX*np.sin(a)**2+0.01*np.cos(a)**2)
    return cl, cd

def bem_forces(Vd, Om):
    Ut = Om*r_e; Veff = np.sqrt(Ut**2+Vd**2)
    φ = np.arctan2(Vd, Ut); α = θ_e - φ
    cl, cd = cl_cd(α); q = 0.5*RHO*Veff**2*c_e
    T = N_BL*np.sum((q*cl*np.cos(φ)-q*cd*np.sin(φ))*dr)
    Q = N_BL*np.sum(r_e*(q*cl*np.sin(φ)-q*cd*np.cos(φ))*dr)
    return T, Q

def solve_bem(m):
    W = m*9.81; Oms = np.linspace(20, 300, 200)
    Vd_c = np.full(200, np.nan); Q_c = np.full(200, np.nan)
    for i, Om in enumerate(Oms):
        try:
            fl = bem_forces(0.1, Om)[0]-W; fh = bem_forces(30., Om)[0]-W
            if fl*fh >= 0: continue
            Vi = brentq(lambda v: bem_forces(v, Om)[0]-W, 0.1, 30., xtol=1e-4)
            Vd_c[i] = Vi; Q_c[i] = bem_forces(Vi, Om)[1]
        except: continue
    idx = np.where(~np.isnan(Q_c))[0]
    for k in range(len(idx)-1):
        i, j = idx[k], idx[k+1]
        if Q_c[i]*Q_c[j] < 0:
            Om_s = Oms[i] - Q_c[i]*(Oms[j]-Oms[i])/(Q_c[j]-Q_c[i])
            Vs = brentq(lambda v: bem_forces(v, Om_s)[0]-W, 0.1, 30., xtol=1e-5)
            return Vs, Om_s
    return np.nan, np.nan

# ═══════════════════════════════════════════════════════════════════════════
# FIRE SPREAD PHYSICS
# ═══════════════════════════════════════════════════════════════════════════
FM2 = {'w0':0.092,'delta':1.0,'sigma':3000,'h':8000,'Mx':0.15}
S_T, S_e, ρ_p = 0.055, 0.010, 32.0

def rothermel(fuel, M, U_ftmin):
    if M >= fuel['Mx']: return 0.0
    w0,δ,σ,h,Mx = fuel['w0'],fuel['delta'],fuel['sigma'],fuel['h'],fuel['Mx']
    rM = M/Mx; ηM = max(0, 1-2.59*rM+5.11*rM**2-3.52*rM**3)
    ηs = min(1, 0.174/S_e**0.19); ρb = w0/δ; β = ρb/ρ_p
    β_op = 3.348/σ**0.8189; Γmax = σ**1.5/(495+0.0594*σ**1.5)
    A = 133/σ**0.7913; r = β/β_op
    Γ = Γmax*(r**A)*np.exp(A*(1-r))
    IR = Γ*w0*(1-S_T)*h*ηM*ηs
    ξ = np.exp((0.792+0.681*σ**0.5)*(β+0.1))/(192+0.2595*σ)
    ε = np.exp(-138/σ); Qig = 250+1116*M
    C=7.47*np.exp(-0.133*σ**0.55); B=0.02526*σ**0.54; E=0.715*np.exp(-3.59e-4*σ)
    ΦW = C*(max(U_ftmin,0)**B)*(r**(-E))
    return max(0, IR*ξ*(1+ΦW)/(ρb*ε*Qig))

U_FIRE = 3.0; M_FIRE = 0.10                   # 3 m/s wind, 10% moisture
ros_ft  = rothermel(FM2, M_FIRE, U_FIRE*196.85)
ros_mmin = ros_ft * 0.3048                     # ft/min → m/min
U_mph = U_FIRE*2.237
LB = 0.936*np.exp(0.2566*U_mph)+0.461*np.exp(-0.1548*U_mph)-0.397
FIRE_DUR = 90                                  # simulate 90 minutes

# ═══════════════════════════════════════════════════════════════════════════
# SWARM SETUP
# ═══════════════════════════════════════════════════════════════════════════
RNG = np.random.default_rng(42)
FA, FB = 800., 400.; R_SENSE, R_COMM = 80., 500.
NP = 300
φp = np.linspace(0, 2*np.pi, NP, endpoint=False)
px_c, py_c = FA*np.cos(φp), FB*np.sin(φp)
perim = np.column_stack([px_c, py_c])
MAX_N = 30

def make_agents(n):
    idx = np.linspace(0, NP, n, endpoint=False).astype(int)
    pts = perim[idx].copy()
    return pts + RNG.uniform(-R_SENSE*.35, R_SENSE*.35, pts.shape)

def coverage(agents):
    if not len(agents): return 0.0
    d = np.min(np.linalg.norm(perim[:,None,:]-agents[None,:,:], axis=2), axis=1)
    return np.mean(d < R_SENSE)

ALL_AG = make_agents(MAX_N)

# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE BEM SWEEP (terminal progress shown here)
# ═══════════════════════════════════════════════════════════════════════════
BEM_MASSES = np.linspace(0.025, 0.200, 22)
BEM_VD  = np.full(len(BEM_MASSES), np.nan)
BEM_RPM = np.full(len(BEM_MASSES), np.nan)

print('\n  Computing BEM mass sweep…')
for i, m in enumerate(BEM_MASSES):
    Vd_i, Om_i = solve_bem(m)
    BEM_VD[i]  = Vd_i
    BEM_RPM[i] = Om_i*60/(2*np.pi) if not np.isnan(Om_i) else np.nan
    bar = '█'*int((i+1)/len(BEM_MASSES)*30)
    sys.stdout.write(f'\r  [{bar:<30}] {i+1}/{len(BEM_MASSES)}  m={m*1000:.0f}g'
                     f'  Vd={Vd_i:.2f} m/s  RPM={BEM_RPM[i]:.0f}    ')
    sys.stdout.flush()

print(f'\n  Done — {np.sum(~np.isnan(BEM_VD))}/{len(BEM_MASSES)} converged\n'
      f'  Launching dashboard…\n')

# Baseline torque distribution
Vd0, Om0 = solve_bem(0.075)
if not np.isnan(Vd0):
    Ut_b = Om0*r_e; Vf_b = np.sqrt(Ut_b**2+Vd0**2)
    φ_b = np.arctan2(Vd0, Ut_b); α_b = θ_e - φ_b
    cl_b, cd_b = cl_cd(α_b); q_b = 0.5*RHO*Vf_b**2*c_e
    dQ = N_BL*r_e*(q_b*cl_b*np.sin(φ_b)-q_b*cd_b*np.cos(φ_b))
else:
    dQ = np.zeros(N_EL)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
DARK, PANEL, TC = '#0d1117', '#161b22', '#c9d1d9'
fig = plt.figure(figsize=(17, 9), facecolor=DARK)
fig.suptitle('  Project Sycamore — Live Simulation Dashboard',
             color='white', fontsize=13, fontweight='bold',
             x=0.02, ha='left', y=0.98)

gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.35,
                      left=0.05, right=0.97, top=0.93, bottom=0.07)

ax_vd  = fig.add_subplot(gs[0, 0])   # BEM descent rate
ax_rpm = fig.add_subplot(gs[1, 0])   # BEM RPM
ax_tq  = fig.add_subplot(gs[2, 0])   # BEM torque distribution
ax_fi  = fig.add_subplot(gs[:, 1])   # fire (tall center)
ax_sw  = fig.add_subplot(gs[0:2, 2]) # swarm map
ax_cv  = fig.add_subplot(gs[2, 2])   # coverage vs N

for ax in [ax_vd, ax_rpm, ax_tq, ax_fi, ax_sw, ax_cv]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TC, labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

def sa(ax, t, xl, yl):
    ax.set_title(t, color=TC, fontsize=8.5, pad=3, loc='left')
    ax.set_xlabel(xl, color=TC, fontsize=7.5)
    ax.set_ylabel(yl, color=TC, fontsize=7.5)
    ax.grid(True, color='#21262d', linewidth=0.6)

sa(ax_vd,  'BEM   descent rate vs mass',   'Mass [g]',  'V_d  [m/s]')
sa(ax_rpm, 'BEM   rotation rate vs mass',  'Mass [g]',  'RPM')
sa(ax_tq,  'BEM   torque distribution  75 g',  'r / R', 'dQ/dr  [N·m/m]')
sa(ax_fi,  f'Rothermel fire spread   FM2, U={U_FIRE:.0f} m/s, M={M_FIRE*100:.0f}%',
           'East  [m]', 'North  [m]')
sa(ax_sw,  'Swarm deployment',             'East  [m]', 'North  [m]')
sa(ax_cv,  'Swarm perimeter coverage',     'Fleet size N', 'Coverage  [%]')

# ── BEM axis limits ────────────────────────────────────────────────────────
mg = BEM_MASSES*1000
ax_vd.set_xlim(mg[0], mg[-1]); ax_vd.set_ylim(0, np.nanmax(BEM_VD)*1.25)
ax_rpm.set_xlim(mg[0], mg[-1]); ax_rpm.set_ylim(0, np.nanmax(BEM_RPM)*1.25)
ax_tq.set_xlim(0, 1)

# BEM static: torque distribution
ax_tq.fill_between(r_e/R, dQ, 0, where=dQ>0, color='#3fb950', alpha=0.45, label='Driving')
ax_tq.fill_between(r_e/R, dQ, 0, where=dQ<0, color='#f85149', alpha=0.45, label='Braking')
ax_tq.plot(r_e/R, dQ, color='#bc8cff', lw=1.4)
ax_tq.axhline(0, color='#8b949e', lw=0.6)
ax_tq.legend(fontsize=7, labelcolor=TC, facecolor=PANEL, edgecolor='none')

# ── Fire axis ─────────────────────────────────────────────────────────────
half = ros_mmin*FIRE_DUR/2
ax_fi.set_aspect('equal')
ax_fi.set_xlim(-half*0.85, half*0.85)
ax_fi.set_ylim(-half*0.55, half*1.10)

clock_txt = ax_fi.text(0.03, 0.97, '', transform=ax_fi.transAxes,
                        color='#f0c040', fontsize=9, va='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='#000000bb', edgecolor='none'))

# Wind arrow
arr_len = half*0.22
ax_fi.annotate('', xy=(0, arr_len), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='#79c0ff', lw=1.8))
ax_fi.text(arr_len*0.06, arr_len*0.48, f'wind\n{U_FIRE:.0f} m/s',
           color='#79c0ff', fontsize=7.5, ha='left')

fire_fill_art  = [None]
fire_edge_art  = [None]

# ── Swarm axis ────────────────────────────────────────────────────────────
lim_s = max(FA, FB)*1.3
ax_sw.set_aspect('equal')
ax_sw.set_xlim(-lim_s, lim_s); ax_sw.set_ylim(-lim_s*.65, lim_s*.65)
ax_sw.plot(px_c, py_c, color='#f85149', lw=1, alpha=0.5, zorder=1)
ax_sw.fill(px_c, py_c, color='#f85149', alpha=0.08, zorder=0)

cov_txt = ax_sw.text(0.03, 0.97, 'N = 0\nCoverage: 0.0%',
                     transform=ax_sw.transAxes, color='#3fb950',
                     fontsize=8.5, va='top', fontweight='bold')

agent_sc = ax_sw.scatter([], [], s=45, color='#58a6ff',
                          edgecolors='#c9d1d9', linewidths=0.5, zorder=5)
comm_lc  = mc.LineCollection([], color='#58a6ff', alpha=0.25, lw=0.6, zorder=2)
ax_sw.add_collection(comm_lc)
sense_patches = []

# ── Coverage vs N axis ────────────────────────────────────────────────────
ax_cv.set_xlim(0, MAX_N+1); ax_cv.set_ylim(0, 102)
ax_cv.axhline(80, ls='--', color='#f0883e', lw=1, alpha=0.8, label='80% target')
ax_cv.legend(fontsize=7.5, labelcolor=TC, facecolor=PANEL, edgecolor='none')

line_cv, = ax_cv.plot([], [], color='#58a6ff', lw=2)
dot_cv,  = ax_cv.plot([], [], 'o', color='#58a6ff', ms=5, zorder=5)

# ── BEM line artists ──────────────────────────────────────────────────────
line_vd,    = ax_vd.plot([],  [], color='#58a6ff',  lw=2)
dot_vd,     = ax_vd.plot([],  [], 'o', color='#58a6ff',  ms=5)
line_rpm_p, = ax_rpm.plot([], [], color='#d2a8ff', lw=2)
dot_rpm_p,  = ax_rpm.plot([], [], 'o', color='#d2a8ff', ms=5)

# Metric callout boxes on BEM plots
vd_box = ax_vd.text(0.97, 0.95, '', transform=ax_vd.transAxes,
                    color='#58a6ff', fontsize=8, va='top', ha='right',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000aa', edgecolor='none'))
rpm_box = ax_rpm.text(0.97, 0.95, '', transform=ax_rpm.transAxes,
                      color='#d2a8ff', fontsize=8, va='top', ha='right',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000aa', edgecolor='none'))

# State for coverage accumulation
cv_xs, cv_ys = [], []

# ═══════════════════════════════════════════════════════════════════════════
# ANIMATION UPDATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_FRAMES = max(len(BEM_MASSES), FIRE_DUR, MAX_N)

def update(frame):
    # ── BEM reveal (one mass point per frame) ─────────────────────────────
    n = min(frame+1, len(BEM_MASSES))
    v  = ~np.isnan(BEM_VD[:n])
    xs = mg[:n][v]
    line_vd.set_data(xs, BEM_VD[:n][v])
    line_rpm_p.set_data(xs, BEM_RPM[:n][v])
    if len(xs):
        dot_vd.set_data([xs[-1]], [BEM_VD[:n][v][-1]])
        dot_rpm_p.set_data([xs[-1]], [BEM_RPM[:n][v][-1]])
        vd_box.set_text(f'{BEM_VD[:n][v][-1]:.2f} m/s')
        rpm_box.set_text(f'{BEM_RPM[:n][v][-1]:.0f} RPM')

    # ── Fire perimeter growth (one minute per frame) ───────────────────────
    t = min(frame, FIRE_DUR-1)
    a = ros_mmin*t/2 + 0.5   # semi-major (along-wind)
    b = a/LB                  # semi-minor
    ell = geo_scale(Point(0,0).buffer(1.0), xfact=b, yfact=a)
    xe, ye = ell.exterior.xy

    if fire_fill_art[0] is not None:
        fire_fill_art[0].remove()
    if fire_edge_art[0] is not None:
        fire_edge_art[0].remove()

    frac = t/FIRE_DUR
    fc = plt.cm.YlOrRd(0.25+0.65*frac)
    fire_fill_art[0] = ax_fi.fill(xe, ye, color=fc, alpha=0.18+0.42*frac, zorder=1)[0]
    fire_edge_art[0], = ax_fi.plot(xe, ye, color='#ff7b72', lw=1.0, alpha=0.8, zorder=2)

    area_ha = np.pi*a*b/1e4
    clock_txt.set_text(
        f't = {t:3d} min\n'
        f'Head  {ros_mmin:.1f} m/min\n'
        f'Area  {area_ha:.1f} ha\n'
        f'Head  {2*a:.0f} m\n'
        f'Cross {2*b:.0f} m'
    )

    # ── Swarm deployment (one agent per frame) ────────────────────────────
    n_ag = min(frame+1, MAX_N)
    ag   = ALL_AG[:n_ag]
    agent_sc.set_offsets(ag if n_ag else np.empty((0,2)))

    # Sensor footprints
    for p in sense_patches: p.remove()
    sense_patches.clear()
    for pt in ag:
        c = Circle(pt, R_SENSE, color='#58a6ff', alpha=0.10, zorder=3)
        ax_sw.add_patch(c); sense_patches.append(c)

    # Comm links
    segs = []
    for i in range(n_ag):
        for j in range(i+1, n_ag):
            if np.linalg.norm(ag[i]-ag[j]) <= R_COMM:
                segs.append([ag[i], ag[j]])
    comm_lc.set_segments(segs)

    cov = coverage(ag)
    cov_txt.set_text(f'N = {n_ag}\nCoverage: {cov*100:.1f}%')

    # Coverage vs N
    if not cv_xs or cv_xs[-1] != n_ag:
        cv_xs.append(n_ag); cv_ys.append(cov*100)
    line_cv.set_data(cv_xs, cv_ys)
    dot_cv.set_data([cv_xs[-1]], [cv_ys[-1]])

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                     interval=200, blit=False, repeat=False)

plt.show()
