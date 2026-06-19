"""
Project Sycamore — Rothermel fire-spread cellular automaton (MLX / Apple-Silicon GPU).

Turns the point Rothermel (1972) rate-of-spread model into a 2-D, spatially
explicit fire over real terrain. Each grid cell carries fuel, moisture, and an
elevation; wind and slope set a per-cell *directional* spread ellipse; the fire
front is propagated as a minimum-arrival-time field by iterative relaxation —
a computation that is pure broadcasted array math and runs entirely on the GPU
via MLX (with a NumPy fallback).

Pipeline
  1. terrain z(x,y)  → slope magnitude + aspect (gradient)
  2. per-cell Rothermel  → no-wind/no-slope ROS_0  and  wind/slope factor coeffs
  3. wind vector ⊕ slope vector  → head-spread direction θ_max, head ROS, ellipse ε
  4. directional ROS(φ) for the 8 grid neighbours  → travel times
  5. Bellman-style min-relaxation of arrival time T  (GPU, ~H+W sweeps)
  6. isochrones / burned-area / ROS-field figures

Validation (run with --validate): the field ROS_0 and head ROS reproduce a
straight scalar transcription of the Rothermel equations to <1e-4 relative.

Run:  ~/ds/bin/python 02_fire/rothermel_ca.py            # simulate + figures
      ~/ds/bin/python 02_fire/rothermel_ca.py --validate # numerical checks
      ~/ds/bin/python 02_fire/rothermel_ca.py --bench    # GPU vs CPU timing
"""

import sys
import time
import numpy as np

try:
    import mlx.core as mx
    HAVE_MLX = True
except Exception:                                        # pragma: no cover
    mx = None
    HAVE_MLX = False

# ── Fuel models (Anderson 1982, single-class simplification) ───────────────────
#   w0 [lb/ft²], delta [ft], sigma [1/ft], h [BTU/lb], Mx [fraction]
FUEL_MODELS = {
    1: {'name': 'Short Grass',          'w0': 0.034, 'delta': 1.0, 'sigma': 3500, 'h': 8000, 'Mx': 0.12},
    2: {'name': 'Timber/Grass Mix',     'w0': 0.092, 'delta': 1.0, 'sigma': 3000, 'h': 8000, 'Mx': 0.15},
    5: {'name': 'Brush',                'w0': 0.046, 'delta': 2.0, 'sigma': 2000, 'h': 8000, 'Mx': 0.20},
    7: {'name': 'Southern Rough',       'w0': 0.115, 'delta': 2.5, 'sigma': 1750, 'h': 8000, 'Mx': 0.40},
    8: {'name': 'Closed Timber Litter', 'w0': 0.069, 'delta': 0.2, 'sigma': 2000, 'h': 8000, 'Mx': 0.30},
}

S_T   = 0.055      # total mineral fraction
S_e   = 0.010      # effective mineral fraction
rho_p = 32.0       # oven-dry particle density [lb/ft³]

MS_TO_FTMIN   = 196.85
FTMIN_TO_MMIN = 0.3048
MS_TO_MPH     = 2.23694


# ══════════════════════════════════════════════════════════════════════════════
#  Scalar Rothermel reference (used only to validate the vectorised field)
# ══════════════════════════════════════════════════════════════════════════════

def rothermel_ros_scalar(fuel, M, U_ftmin, slope_deg=0.0):
    """Rate of spread [ft/min] in the direction of max spread (Rothermel 1972)."""
    w0, delta, sigma = fuel['w0'], fuel['delta'], fuel['sigma']
    h, Mx = fuel['h'], fuel['Mx']
    if M >= Mx:
        return 0.0
    r_M   = M / Mx
    eta_M = max(1.0 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3, 0.0)
    eta_s = min(0.174 / (S_e**0.19), 1.0)
    rho_b = w0 / delta
    beta  = rho_b / rho_p
    beta_op   = 3.348 / (sigma**0.8189)
    Gamma_max = (sigma**1.5) / (495 + 0.0594*sigma**1.5)
    A     = 133.0 / (sigma**0.7913)
    ratio = beta / beta_op
    Gamma = Gamma_max * (ratio**A) * np.exp(A*(1.0 - ratio))
    w_n   = w0 / (1.0 + S_T)             # net (mineral-free) fuel load, BehavePlus/Andrews 2018
    I_R   = Gamma * w_n * h * eta_M * eta_s
    xi    = np.exp((0.792 + 0.681*sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595*sigma)
    eps   = np.exp(-138.0 / sigma)
    Q_ig  = 250.0 + 1116.0*M
    C = 7.47 * np.exp(-0.133*sigma**0.55)
    B = 0.02526 * sigma**0.54
    E = 0.715 * np.exp(-3.59e-4*sigma)
    Phi_W = C * (max(U_ftmin, 0.0)**B) * (ratio**(-E))
    Phi_S = 5.275 * (beta**(-0.3)) * (np.tan(np.deg2rad(slope_deg))**2)
    ros = I_R * xi * (1.0 + Phi_W + Phi_S) / (rho_b * eps * Q_ig)
    return max(ros, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorised Rothermel field (fuel/moisture constant; wind+slope vary in space)
# ══════════════════════════════════════════════════════════════════════════════

def rothermel_coeffs(fuel, M):
    """Scalar pieces that depend only on fuel + moisture (no wind, no slope).
    Returns ros0 [ft/min], and the wind/slope coefficients used per cell."""
    w0, delta, sigma = fuel['w0'], fuel['delta'], fuel['sigma']
    h, Mx = fuel['h'], fuel['Mx']
    r_M   = M / Mx
    eta_M = max(1.0 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3, 0.0)
    eta_s = min(0.174 / (S_e**0.19), 1.0)
    rho_b = w0 / delta
    beta  = rho_b / rho_p
    beta_op   = 3.348 / (sigma**0.8189)
    Gamma_max = (sigma**1.5) / (495 + 0.0594*sigma**1.5)
    A     = 133.0 / (sigma**0.7913)
    ratio = beta / beta_op
    Gamma = Gamma_max * (ratio**A) * np.exp(A*(1.0 - ratio))
    w_n   = w0 / (1.0 + S_T)             # net (mineral-free) fuel load, BehavePlus/Andrews 2018
    I_R   = Gamma * w_n * h * eta_M * eta_s
    xi    = np.exp((0.792 + 0.681*sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595*sigma)
    eps   = np.exp(-138.0 / sigma)
    Q_ig  = 250.0 + 1116.0*M
    ros0  = I_R * xi / (rho_b * eps * Q_ig)             # ft/min, no wind/slope
    C = 7.47 * np.exp(-0.133*sigma**0.55)
    B = 0.02526 * sigma**0.54
    E = 0.715 * np.exp(-3.59e-4*sigma)
    return dict(ros0=float(ros0), C=float(C), B=float(B), E=float(E),
                ratio=float(ratio), beta=float(beta), slope_coeff=5.275*beta**(-0.3))


LB_MAX = 8.0     # FARSITE-style cap: Anderson's formula diverges at high wind and
                 # is only validated for low–moderate winds; real fire ellipses
                 # plateau near LB≈8 even in extreme conditions.

def length_to_breadth(U_mph):
    """Anderson (1983) fire ellipse length-to-breadth ratio from effective wind,
    clamped to a physically realistic maximum."""
    lb = 0.936*np.exp(0.2566*U_mph) + 0.461*np.exp(-0.1548*U_mph) - 0.397
    return np.clip(lb, 1.0, LB_MAX)


def rothermel_coeffs_field(w0, delta, sigma, h, Mx, M):
    """Vectorised rothermel_coeffs — each fuel parameter is an (H,W) array, giving
    per-cell coefficients. Identical equations to the scalar version (net load
    w0/(1+S_T)). Lets a real LANDFIRE fuel map make every Rothermel input spatial."""
    w0 = np.asarray(w0, 'f8'); delta = np.asarray(delta, 'f8')
    sigma = np.asarray(sigma, 'f8'); h = np.asarray(h, 'f8'); Mx = np.asarray(Mx, 'f8')
    r_M   = M / Mx
    eta_M = np.clip(1.0 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3, 0.0, None)
    eta_s = min(0.174 / (S_e**0.19), 1.0)
    rho_b = w0 / delta
    beta  = rho_b / rho_p
    beta_op   = 3.348 / (sigma**0.8189)
    Gamma_max = (sigma**1.5) / (495 + 0.0594*sigma**1.5)
    A     = 133.0 / (sigma**0.7913)
    ratio = beta / beta_op
    Gamma = Gamma_max * (ratio**A) * np.exp(A*(1.0 - ratio))
    w_n   = w0 / (1.0 + S_T)
    I_R   = Gamma * w_n * h * eta_M * eta_s
    xi    = np.exp((0.792 + 0.681*sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595*sigma)
    eps   = np.exp(-138.0 / sigma)
    Q_ig  = 250.0 + 1116.0*M
    ros0  = I_R * xi / (rho_b * eps * Q_ig)
    C = 7.47 * np.exp(-0.133*sigma**0.55)
    B = 0.02526 * sigma**0.54
    E = 0.715 * np.exp(-3.59e-4*sigma)
    return dict(ros0=ros0, C=C, B=B, E=E, ratio=ratio, beta=beta,
                slope_coeff=5.275*beta**(-0.3))


# ══════════════════════════════════════════════════════════════════════════════
#  Array backend shim — same code path for NumPy and MLX
# ══════════════════════════════════════════════════════════════════════════════

class Backend:
    def __init__(self, use_mlx):
        self.mlx = use_mlx and HAVE_MLX
        self.xp  = mx if self.mlx else np
    def arr(self, a):    return mx.array(np.asarray(a, 'f4')) if self.mlx else np.asarray(a, 'f4')
    def to_np(self, a):  return np.array(a) if self.mlx else np.asarray(a)
    def eval(self, *a):
        if self.mlx: mx.eval(*a)
    # element-wise helpers that differ in name/signature across the two libs
    def hypot(self, x, y):          return self.xp.sqrt(x*x + y*y)
    def atan2(self, y, x):          return self.xp.arctan2(y, x) if not self.mlx else mx.arctan2(y, x)
    def clip(self, a, lo, hi):      return self.xp.clip(a, lo, hi)
    def where(self, c, a, b):       return self.xp.where(c, a, b)
    def maximum(self, a, b):        return self.xp.maximum(a, b)
    def minimum(self, a, b):        return self.xp.minimum(a, b)


# ══════════════════════════════════════════════════════════════════════════════
#  The simulation
# ══════════════════════════════════════════════════════════════════════════════

# 8-neighbour offsets (di, dj) and their cell-centre distances (×DX)
NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)]


def make_terrain(H, W, DX):
    """Synthetic but fire-relevant terrain: a steep ridge, two peaks and a
    drainage, with flank slopes in the 15–35° band where slope strongly drives
    spread. Returns elevation z [m] and the East/North gradient (∂z/∂x, ∂z/∂y)."""
    y, x = np.mgrid[0:H, 0:W].astype('f4')
    xm, ym = x*DX, y*DX
    Lx, Ly = W*DX, H*DX
    z  = 330.0*np.exp(-(((xm - 0.42*Lx)/(0.10*Lx))**2))              # steep N–S ridge
    z += 270.0*np.exp(-(((xm - 0.70*Lx)**2 + (ym - 0.32*Ly)**2)/(0.09*Lx)**2))  # peak
    z += 230.0*np.exp(-(((xm - 0.30*Lx)**2 + (ym - 0.76*Ly)**2)/(0.10*Lx)**2))  # peak
    z -= 130.0*np.exp(-(((ym - 0.55*Ly)/(0.06*Ly))**2))             # E–W drainage
    z += 0.03*ym                                                     # regional tilt
    dzdy, dzdx = np.gradient(z, DX)                                  # [m/m]
    return z.astype('f4'), dzdx.astype('f4'), dzdy.astype('f4')


def simulate(H=220, W=320, DX=25.0,
             fuel=FUEL_MODELS[2], M=0.08,
             wind_ms=2.5, wind_from_deg=270.0,     # MIDFLAME wind FROM the west → blows east
             ignition=(0.30, 0.55), use_mlx=True, max_sweeps=None, verbose=False,
             terrain_fn=None, fuel_fn=None, wind_fn=None, moisture_fn=None):
    """Run the fire-spread CA. Returns a dict of fields (NumPy) for plotting.
    `fuel_fn(H,W,DX)` → dict(w0,delta,sigma,h,Mx, burnable[, code]) makes fuel
    spatial (e.g. a real LANDFIRE map); else the uniform `fuel` model is used.
    `wind_fn(z,dzdx,dzdy,DX)` → (speed_field, from_deg_field) makes wind spatial
    (e.g. mass-conserving terrain downscaling); else uniform wind_ms/wind_from_deg."""
    be = Backend(use_mlx)
    xp = be.xp

    # ── terrain & static fields ──
    z, dzdx, dzdy = (terrain_fn or make_terrain)(H, W, DX)

    # moisture: scalar, or a spatial field (HRRR weather × terrain microclimate)
    M_field = None
    M_used = M
    if moisture_fn is not None:
        M_used = np.asarray(moisture_fn(z, dzdx, dzdy, DX), 'f8')
        M_field = M_used.astype('f4')

    burnable = None;  fuel_code = None
    if fuel_fn is not None:
        ff = fuel_fn(H, W, DX)
        co = rothermel_coeffs_field(ff['w0'], ff['delta'], ff['sigma'], ff['h'], ff['Mx'], M_used)
        burnable = np.asarray(ff['burnable']);  fuel_code = ff.get('code')
    elif moisture_fn is not None:                         # spatial moisture, uniform fuel
        g = lambda v: np.full((H, W), v, 'f8')
        co = rothermel_coeffs_field(g(fuel['w0']), g(fuel['delta']), g(fuel['sigma']),
                                    g(fuel['h']), g(fuel['Mx']), M_used)
    else:
        co = rothermel_coeffs(fuel, M)

    # slope magnitude + upslope direction (steepest ascent = +grad)
    slope_tan = np.hypot(dzdx, dzdy)                      # rise/run

    # wind: uniform scalar, or a spatial field from wind_fn (terrain downscaling)
    wind_speed_field = None
    if wind_fn is not None:
        spd_f, fromdeg_f = wind_fn(z, dzdx, dzdy, DX)     # (H,W) each
        U_ms = np.asarray(spd_f, 'f8');  from_deg = np.asarray(fromdeg_f, 'f8')
        wind_speed_field = np.asarray(spd_f, 'f4')
        wind_ms_disp = float(np.mean(spd_f))              # scalar summaries for the figure
        wind_deg_disp = float(wind_from_deg)
    else:
        U_ms = wind_ms;  from_deg = wind_from_deg
        wind_ms_disp = wind_ms;  wind_deg_disp = wind_from_deg

    wind_to = np.deg2rad(from_deg + 180.0)                # direction wind blows toward
    wdx, wdy = np.sin(wind_to), np.cos(wind_to)          # East, North components (unit; may be fields)
    U_ftmin = np.asarray(U_ms) * MS_TO_FTMIN

    # per-cell factors
    Phi_W = co['C'] * (np.maximum(U_ftmin, 0.0)**co['B']) * (co['ratio']**(-co['E']))
    Phi_S = co['slope_coeff'] * slope_tan**2                                      # field

    # combine wind + slope as vectors of magnitude Φ along their directions
    upx = np.where(slope_tan > 1e-9, dzdx/np.maximum(slope_tan, 1e-9), 0.0)
    upy = np.where(slope_tan > 1e-9, dzdy/np.maximum(slope_tan, 1e-9), 0.0)
    vx = Phi_W*wdx + Phi_S*upx
    vy = Phi_W*wdy + Phi_S*upy
    Phi_eff = np.hypot(vx, vy)                            # combined factor magnitude
    theta_max = np.arctan2(vy, vx)                        # head direction (math angle, East=0)
    head_ros_ms = co['ros0'] * (1.0 + Phi_eff) * FTMIN_TO_MMIN / 60.0    # m/s
    if burnable is not None:
        head_ros_ms = np.where(burnable, head_ros_ms, 0.0)               # nonburnable = no spread

    # ellipse eccentricity from the *effective* equivalent wind (invert Φ_W)
    U_eff_ftmin = (np.maximum(Phi_eff, 1e-6) / (co['C'] * co['ratio']**(-co['E']))) ** (1.0/co['B'])
    U_eff_mph   = U_eff_ftmin / MS_TO_FTMIN * MS_TO_MPH
    LB  = length_to_breadth(np.clip(U_eff_mph, 0, 50))
    ecc = np.sqrt(np.clip(LB**2 - 1.0, 0, None)) / LB     # 0 (circle) … →1

    # ── push static fields to the chosen backend ──
    head = be.arr(head_ros_ms)
    th   = be.arr(theta_max)
    e    = be.arr(ecc)
    INF  = 1e9

    # directional ROS toward each neighbour, precomputed per cell (m/s)
    # ROS(φ) = head * (1-e) / (1 - e cos(φ - θ_max))
    # Frame (consistent with theta_max): East = +j (col), North = +i (row, origin
    # 'lower'). A neighbour offset (di,dj) travels at math-angle arctan2(di,dj).
    burn_be = (mx.array(burnable) if be.mlx else burnable) if burnable is not None else None
    travel = []                                           # per-neighbour travel time [s] from a source cell
    for di, dj, dist in NEIGH:
        phi = np.arctan2(di, dj)
        ros_dir = be.maximum(head * (1.0 - e) /
                             (1.0 - e*xp.cos(be.arr(np.float32(phi)) - th)), 1e-6)
        tv = be.arr(np.float32(dist*DX)) / ros_dir
        if burn_be is not None:
            tv = be.where(burn_be, tv, INF)               # fire cannot spread FROM nonburnable
        travel.append(tv)
    be.eval(*travel)

    # ── arrival-time relaxation (Bellman / Dijkstra-on-grid, vectorised) ──
    # The travel time from a source cell toward a neighbour is constant, so its
    # shifted copy is computed ONCE. Each sweep then only shifts T. MLX is lazy:
    # we run a batch of sweeps without forcing evaluation so the GPU pipelines
    # them into one fused kernel launch, syncing only to test convergence.
    tt_shift = [_shift(xp, travel[k], di, dj, INF) for k, (di, dj, _) in enumerate(NEIGH)]
    be.eval(*tt_shift)
    offs = [(di, dj) for di, dj, _ in NEIGH]

    T = np.full((H, W), INF, 'f4')
    gi = int(ignition[1]*H); gj = int(ignition[0]*W)
    if burnable is not None and not bool(burnable[gi, gj]):
        rows, cols = np.where(burnable)                   # snap ignition to nearest burnable cell
        k = np.argmin((rows - gi)**2 + (cols - gj)**2)
        gi, gj = int(rows[k]), int(cols[k])
    T[gi, gj] = 0.0
    T = be.arr(T)

    if max_sweeps is None:
        max_sweeps = int((H + W) * 1.5)
    BATCH = 16
    prev_score = None;  prev_n = None
    t0 = time.perf_counter()
    done = 0
    while done < max_sweeps:
        for _ in range(BATCH):
            cand = T
            for k, (di, dj) in enumerate(offs):
                cand = be.minimum(cand, _shift(xp, T, di, dj, INF) + tt_shift[k])
            T = cand
        be.eval(T)                                   # one sync per BATCH sweeps
        done += BATCH
        Tnp = be.to_np(T)
        fin = Tnp < INF*0.5
        n_burn = int(fin.sum())
        score = float(Tnp[fin].sum()) if fin.any() else 0.0
        if verbose:
            print(f"  sweep {done:4d}  burned {fin.mean()*100:5.1f}%")
        # Converged only when the front has STOPPED advancing (burned count stable)
        # AND the arrival field has settled. Requiring the count to plateau avoids a
        # resolution-dependent premature stop (the score threshold alone scales with
        # the accumulated sum, so it would trip mid-spread on large grids).
        if (prev_n is not None and n_burn == prev_n
                and abs(score - prev_score) <= 1e-5*max(prev_score, 1.0)):
            break
        prev_score = score;  prev_n = n_burn
    wall = time.perf_counter() - t0

    T_np = be.to_np(T).astype('f4')
    T_np[T_np >= INF*0.5] = np.nan                        # unburned → NaN
    if burnable is not None:
        T_np[~burnable] = np.nan                          # nonburnable never burns
    return dict(T=T_np, z=z, slope_tan=slope_tan, head_ros_ms=head_ros_ms,
                theta_max=theta_max, ecc=ecc, DX=DX, H=H, W=W,
                ignition=(gi, gj), wind_ms=wind_ms_disp, wind_from_deg=wind_deg_disp,
                wind_speed_field=wind_speed_field, M_field=M_field,
                fuel=fuel, M=M, wall=wall, backend='MLX' if be.mlx else 'NumPy',
                sweeps=done, burnable=burnable, fuel_code=fuel_code)


def _shift(xp, a, di, dj, fill):
    """Shift 2-D array by (di,dj), filling exposed border with `fill`.
    Result[i,j] = a[i-di, j-dj]  (so a source at i-di,j-dj lands at i,j)."""
    H, W = a.shape
    out = xp.full((H, W), fill, dtype=a.dtype) if xp is np else mx.full((H, W), fill, dtype=a.dtype)
    si0, si1 = max(0, di),  H + min(0, di)
    sj0, sj1 = max(0, dj),  W + min(0, dj)
    ti0, ti1 = max(0, -di), H + min(0, -di)
    tj0, tj1 = max(0, -dj), W + min(0, -dj)
    if xp is np:
        out[si0:si1, sj0:sj1] = a[ti0:ti1, tj0:tj1]
    else:
        # MLX has no in-place slice assignment → rebuild via padding/slicing
        rows = a[ti0:ti1, tj0:tj1]
        out  = _mlx_place(rows, (H, W), si0, sj0, fill)
    return out


def _mlx_place(block, shape, i0, j0, fill):
    """Place `block` into an otherwise-`fill` array of `shape` at (i0,j0)."""
    H, W = shape
    bh, bw = block.shape
    pad_top, pad_bot = i0, H - i0 - bh
    pad_lft, pad_rgt = j0, W - j0 - bw
    return mx.pad(block, ((pad_top, pad_bot), (pad_lft, pad_rgt)), constant_values=fill)


# ══════════════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    print("Rothermel field vs scalar reference")
    print(f"{'fuel':<22}{'M%':>5}{'U m/s':>7}{'slope°':>8}"
          f"{'scalar ft/min':>15}{'field ft/min':>14}{'rel err':>10}")
    ok = True
    for fid in (1, 2, 5, 7):
        fuel = FUEL_MODELS[fid]
        for M in (0.05, 0.10):
            for U in (0.0, 5.0, 10.0):
                for slope in (0.0, 20.0):
                    Uft = U*MS_TO_FTMIN
                    ref = rothermel_ros_scalar(fuel, M, Uft, slope)
                    # field head ROS for a single cell with this wind+slope:
                    co = rothermel_coeffs(fuel, M)
                    Phi_W = co['C']*(max(Uft,0.0)**co['B'])*(co['ratio']**(-co['E']))
                    Phi_S = co['slope_coeff']*np.tan(np.deg2rad(slope))**2
                    # when wind & upslope are colinear the head factor is the scalar sum
                    head = co['ros0']*(1.0 + Phi_W + Phi_S)
                    rel = abs(head - ref)/max(ref, 1e-9)
                    if rel > 1e-4: ok = False
                    if U in (0.0, 10.0) and slope in (0.0, 20.0) and M == 0.10:
                        print(f"FM{fid} {fuel['name']:<17}{M*100:>4.0f}{U:>7.1f}{slope:>8.0f}"
                              f"{ref:>15.3f}{head:>14.3f}{rel:>10.1e}")
    print(f"\n  {'PASS' if ok else 'FAIL'}: scalar coefficient transcription matches Rothermel (<1e-4 rel)")

    # vector wind+slope coupling, pulled from an actual simulate() field (not just
    # the colinear scalar sum) — non-colinear case: wind East, slope rising North.
    def _north_ramp25(H, W, DX):
        y, _ = np.mgrid[0:H, 0:W].astype('f4')
        z = (y*DX)*np.tan(np.deg2rad(25.0))
        dzdy, dzdx = np.gradient(z, DX)
        return z.astype('f4'), dzdx.astype('f4'), dzdy.astype('f4')
    rv = simulate(H=80, W=80, DX=25.0, wind_ms=2.5, wind_from_deg=270.0,
                  use_mlx=False, max_sweeps=10, terrain_fn=_north_ramp25)
    ci, cj = 40, 40
    co = rothermel_coeffs(FUEL_MODELS[2], 0.08)
    PhiW = co['C']*((2.5*MS_TO_FTMIN)**co['B'])*(co['ratio']**(-co['E']))
    PhiS = co['slope_coeff']*np.tan(np.deg2rad(25.0))**2
    head_expect = co['ros0']*(1.0 + np.hypot(PhiW, PhiS))*FTMIN_TO_MMIN/60.0   # ⊥ combine
    th_expect   = np.degrees(np.arctan2(PhiS, PhiW))                          # between E and N
    head_got = float(rv['head_ros_ms'][ci, cj]); th_got = float(np.degrees(rv['theta_max'][ci, cj]))
    rel = abs(head_got - head_expect)/head_expect
    print(f"Vector wind⊕slope coupling (wind E ⟂ slope N, from simulate() field):")
    print(f"  head ROS got {head_got*60:.3f} vs expect {head_expect*60:.3f} m/min (rel {rel:.1e}); "
          f"θ_max got {th_got:.1f}° expect {th_expect:.1f}°")
    print(f"  {'PASS' if rel < 1e-4 and abs(th_got-th_expect) < 0.5 else 'FAIL'}: "
          f"vector combine matches |Φ_W ŵ + Φ_S ŝ|\n")

    # physics sanity 1 — wind only (flat-ish terrain), downwind faster than up
    r = simulate(H=120, W=160, DX=25.0, wind_ms=2.5, use_mlx=False, max_sweeps=400)
    T = r['T']; gi, gj = r['ignition']; DX = r['DX']
    d = 40
    t_east = T[gi, gj+d]; t_west = T[gi, gj-d]
    print(f"Wind check (wind from west, ±{d*DX:.0f} m of ignition):")
    print(f"  arrival downwind(E) = {t_east/60:6.1f} min   upwind(W) = {t_west/60:6.1f} min")
    print(f"  {'PASS' if t_east < t_west else 'FAIL'}: fire runs faster downwind")
    print(f"  {'PASS' if T[gi,gj]==0 else 'FAIL'}: ignition cell arrival time = 0")

    # physics sanity 2 — slope only (NO wind), fire must run UPslope faster
    def _north_ramp(H, W, DX):
        y, _ = np.mgrid[0:H, 0:W].astype('f4')
        z = (y*DX) * np.tan(np.deg2rad(25.0))            # plane rising to +i (North)
        dzdy, dzdx = np.gradient(z, DX)
        return z.astype('f4'), dzdx.astype('f4'), dzdy.astype('f4')
    rs = simulate(H=120, W=160, DX=25.0, wind_ms=0.0, ignition=(0.5, 0.5),
                  use_mlx=False, max_sweeps=600, terrain_fn=_north_ramp)
    Ts = rs['T']; si, sj = rs['ignition']
    t_up = Ts[si+d, sj]; t_dn = Ts[si-d, sj]
    print(f"Slope check (no wind, 25° slope rising North, ±{d*DX:.0f} m):")
    print(f"  arrival upslope(N) = {t_up/60:6.1f} min   downslope(S) = {t_dn/60:6.1f} min")
    print(f"  {'PASS' if t_up < t_dn else 'FAIL'}: fire runs faster upslope")
    burned = np.isfinite(T).mean()*100
    print(f"  burned fraction (wind run, {r['sweeps']} sweeps): {burned:.0f}%")


def benchmark():
    print(f"\n{'grid':>12}{'NumPy s':>10}{'MLX s':>10}{'speedup':>9}")
    for H, W in [(120, 160), (220, 320), (400, 600), (600, 900)]:
        r_np = simulate(H=H, W=W, use_mlx=False)
        line = f"{H}x{W:<7}{r_np['wall']:>10.2f}"
        if HAVE_MLX:
            r_mx = simulate(H=H, W=W, use_mlx=True)
            line += f"{r_mx['wall']:>10.2f}{r_np['wall']/r_mx['wall']:>8.2f}x"
        print(line)


# ══════════════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(r, out_prefix='output/fire_ca', note=None):
    import os
    os.makedirs('output', exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = r['T']; DX = r['DX']; H = r['H']; W = r['W']
    gi, gj = r['ignition']
    extent = [0, W*DX/1000, 0, H*DX/1000]                # km
    T_min = T/60.0                                        # minutes
    z = r['z']
    if r.get('fuel_code') is not None:
        fuel_label = "LANDFIRE FBFM40 (spatial)"
    else:
        fid = [k for k, v in FUEL_MODELS.items() if v is r['fuel']][0]
        fuel_label = f"FM{fid} {r['fuel']['name']}"
    m_label = (f"M≈{np.nanmean(r['M_field'])*100:.0f}% spatial"
               if r.get('M_field') is not None else f"M={r['M']*100:.0f}%")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), facecolor='white')

    # ── Panel 1: terrain hillshade + fire-progression isochrones ──
    # Fixed operational time levels (the head crosses the domain in a few hours;
    # the slow backing flank lands in the top bin). This is the standard fire-
    # progression map and sidesteps the huge head-vs-back dynamic range.
    levels = np.array([15, 30, 45, 60, 90, 120, 180, 240], dtype='f4')
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    hill = ls.hillshade(z, vert_exag=2.0, dx=DX, dy=DX)
    ax[0].imshow(hill, cmap='gray', extent=extent, origin='lower', alpha=0.9)
    ax[0].contour(z, levels=12, colors='saddlebrown', linewidths=0.4,
                  extent=extent, origin='lower', alpha=0.4)
    cs = ax[0].contourf(T_min, levels=levels, cmap='inferno', alpha=0.55,
                        extent=extent, origin='lower', extend='max')
    ax[0].contour(T_min, levels=levels, colors='k', linewidths=0.5,
                  extent=extent, origin='lower', alpha=0.5)
    ax[0].plot(gj*DX/1000, gi*DX/1000, '*', color='cyan', ms=16, mec='k', label='ignition')
    wt = np.deg2rad(r['wind_from_deg'] + 180.0)
    ax[0].annotate('', xy=(0.13+0.08*np.sin(wt), 0.90+0.08*np.cos(wt)),
                   xytext=(0.13, 0.90), xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='-|>', color='deepskyblue', lw=2.5))
    ax[0].text(0.13, 0.82, f"wind {r['wind_ms']:.1f} m/s", transform=ax[0].transAxes,
               color='deepskyblue', fontsize=9, ha='left')
    cb = fig.colorbar(cs, ax=ax[0], shrink=0.82, ticks=levels)
    cb.set_label('fire arrival time [min]')
    ax[0].set(xlabel='East [km]', ylabel='North [km]',
              title=f"Fire-spread isochrones on terrain  ({r['backend']} CA)\n"
                    f"{fuel_label}, {m_label}, "
                    f"{H}×{W} cells @ {DX:.0f} m")
    ax[0].legend(loc='lower right', fontsize=8)

    # ── Panel 2: head-ROS field + spread-direction quiver ──
    hr = r['head_ros_ms']*60                              # m/min
    im = ax[1].imshow(hr, cmap='YlOrRd', extent=extent, origin='lower')
    ax[1].contour(z, levels=12, colors='k', linewidths=0.35,
                  extent=extent, origin='lower', alpha=0.3)
    s = max(H, W)//26                                     # quiver decimation
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    th = r['theta_max'][::s, ::s]
    ax[1].quiver(xx*DX/1000, yy*DX/1000, np.cos(th), np.sin(th),
                 color='navy', scale=32, width=0.0025, alpha=0.7)
    cb2 = fig.colorbar(im, ax=ax[1], shrink=0.82); cb2.set_label('head ROS [m/min]')
    ax[1].plot(gj*DX/1000, gi*DX/1000, '*', color='cyan', ms=14, mec='k')
    ax[1].set(xlabel='East [km]', ylabel='North [km]',
              title='Per-cell head rate-of-spread + spread direction\n'
                    '(wind ⊕ slope vector coupling)')

    if note:
        fig.text(0.5, 0.005, note, ha='center', va='bottom', fontsize=8.5,
                 color='#333', style='italic')
    plt.tight_layout(rect=(0, 0.03, 1, 1) if note else None)
    path = f'{out_prefix}.png'
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    print(f"Saved {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate()
    elif '--bench' in sys.argv:
        benchmark()
    else:
        print(f"MLX available: {HAVE_MLX}")
        r = simulate(use_mlx=True, verbose=True)
        print(f"\nBackend {r['backend']}  —  {r['H']}×{r['W']} grid, "
              f"{r['sweeps']} sweeps in {r['wall']:.2f}s")
        print(f"burned area: {np.isfinite(r['T']).mean()*100:.0f}% of domain, "
              f"max arrival {np.nanmax(r['T'])/60:.0f} min")
        make_figures(r)
