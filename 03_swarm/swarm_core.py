"""
Project Sycamore — 3D swarm physics core (backend-free).

Two interchangeable implementations of the same 3-D boids step:
  * `step_np`  — NumPy reference (CPU), the ground-truth definition of the model.
  * `step_mx`  — MLX port (Apple-Silicon GPU via Metal), numerically equivalent
                 to `step_np` when Vicsek noise is disabled.

No matplotlib / no I/O here so the physics can be imported headless, validated,
and benchmarked. The renderer (`boids_swarm_3d.py`) and the validation harness
(`validate_mlx.py`) both import from this module.

Model (unchanged from the 2-D/3-D boids design):
  XZ plane : separation · alignment · fire tracking · arc-index spread · tangential patrol · speed reg
  Y axis   : PD altitude controller holding ALT_NOM
"""

import numpy as np

try:
    import mlx.core as mx
    HAVE_MLX = True
except Exception:                                    # pragma: no cover
    mx = None
    HAVE_MLX = False

# ── Fire perimeter (target ring the swarm patrols) ─────────────────────────────
FIRE_A  = 800.0
FIRE_B  = 400.0
N_PERIM = 400
_phi    = np.linspace(0, 2*np.pi, N_PERIM, endpoint=False)
PERIM   = np.column_stack([FIRE_A*np.cos(_phi), FIRE_B*np.sin(_phi)]).astype('f4')  # (N_P,2) XZ

# ── Swarm parameters ───────────────────────────────────────────────────────────
N_DRONES  = 150
ALT_NOM   = 150.0
V_CRUISE  = 6.5
V_MAX     = 12.0
R_PERC    = 250.0
R_SEP     = 100.0
R_ARC_IDX = 40
R_PATROL  = 200.0
R_SENSE   = 80.0

W_SEP = 0.5;  W_ALI = 0.1;  W_FIRE = 4.0
W_TAN = 1.5;  W_ARC = 5.0;  W_SPD  = 0.5
W_ALT = 3.0;  K_ALT_P = 0.15;  K_ALT_D = 0.6
SIGMA_NOISE = 0.05

DT      = 0.5
T_MAX   = 650.0
N_STEPS = int(T_MAX / DT)        # 1300

GROUND_Y = 20.0
VY_LIMIT = 5.0

# ── Initial condition ──────────────────────────────────────────────────────────
# Deployment scenario: the whole fleet launches from a staging base west of the
# fire and must self-organise to blanket the perimeter. With a large fleet this
# is the interesting regime — coverage climbs 0→100% as the swarm fans out — and
# it exercises every boids force (fire-seek, tangential patrol, arc-index spread).

BASE_XZ = np.array([-FIRE_A*1.6, 0.0], dtype='f4')   # staging point (East, North)
BASE_R  = 120.0                                       # launch cluster radius [m]

def init_state(rng):
    r   = BASE_R * np.sqrt(rng.uniform(0, 1, N_DRONES))
    th  = rng.uniform(0, 2*np.pi, N_DRONES)
    px  = BASE_XZ[0] + r*np.cos(th)
    pz  = BASE_XZ[1] + r*np.sin(th)
    py  = np.full(N_DRONES, ALT_NOM) + rng.uniform(-15, 15, N_DRONES)
    pos = np.c_[px, py, pz].astype('f4')
    # launch heading: toward fire centre, at cruise speed
    to  = np.c_[-px, -pz];  to /= np.linalg.norm(to, axis=1, keepdims=True).clip(min=1e-6)
    vel = np.c_[to[:, 0]*V_CRUISE,
                rng.uniform(-0.5, 0.5, N_DRONES),
                to[:, 1]*V_CRUISE].astype('f4')
    return pos, vel

# ══════════════════════════════════════════════════════════════════════════════
#  NumPy reference step
# ══════════════════════════════════════════════════════════════════════════════

def step_np(pos, vel, rng, sigma=SIGMA_NOISE, perim=None):
    """One boids step. `perim` (M,2 XZ) is the patrol target; defaults to the
    static fire ellipse PERIM but may be a time-varying live fire front."""
    if perim is None:
        perim = PERIM
    n_p = len(perim)
    N = len(pos)
    disp = pos[None] - pos[:, None]                     # (N,N,3)
    d3   = np.linalg.norm(disp, axis=2);  np.fill_diagonal(d3, np.inf)
    dxz  = np.linalg.norm(disp[:, :, [0, 2]], axis=2);  np.fill_diagonal(dxz, np.inf)

    perc = dxz < R_PERC
    sep  = (d3 < R_SEP) & (d3 > 1e-6)
    nN   = perc.sum(1).clip(min=1)

    sw    = np.where(sep, 1.0/d3**2, 0.0)
    f_sep = -np.einsum('ij,ijk->ik', sw, disp)

    sxz  = np.linalg.norm(vel[:, [0, 2]], axis=1);  ok = sxz > 1e-6
    vhxz = np.zeros((N, 2));  vhxz[ok] = vel[ok][:, [0, 2]] / sxz[ok, None]
    ali  = np.einsum('ij,jk->ik', perc.astype('f4'), vhxz) / nN[:, None]
    f_ali = np.c_[ali[:, 0], np.zeros(N), ali[:, 1]]

    pxz = pos[:, [0, 2]]
    dp  = np.linalg.norm(perim[None] - pxz[:, None], axis=2)
    kn  = np.argmin(dp, axis=1)
    ps  = perim[kn]
    toP = ps - pxz;  dP = np.linalg.norm(toP, axis=1, keepdims=True).clip(min=1e-6)
    f_fire_xz = toP / dP

    kf = (kn + 1) % n_p;  kb = (kn - 1) % n_p
    tan_raw = perim[kf] - perim[kb]
    tan_xz  = tan_raw / np.linalg.norm(tan_raw, axis=1, keepdims=True).clip(min=1e-6)
    ramp    = np.clip(1.0 - dP[:, 0]/R_PATROL, 0, 1)        # [:,0] not squeeze(): robust at N==1
    f_tan_xz = tan_xz * ramp[:, None]

    dk  = kn[None].astype(int) - kn[:, None].astype(int)
    dk  = np.where(dk >  n_p//2, dk - n_p, dk)
    dk  = np.where(dk < -n_p//2, dk + n_p, dk)
    adk = np.abs(dk)
    aw  = np.where((adk > 0) & (adk < R_ARC_IDX), -np.sign(dk).astype(float)/adk.clip(min=1), 0.)
    f_arc_xz = aw.sum(1)[:, None] * tan_xz

    def _xz3(a): return np.c_[a[:, 0], np.zeros(N), a[:, 1]]
    f_fire = _xz3(f_fire_xz);  f_tan = _xz3(f_tan_xz);  f_arc = _xz3(f_arc_xz)

    f_alt = np.zeros((N, 3))
    f_alt[:, 1] = K_ALT_P*(ALT_NOM - pos[:, 1]) - K_ALT_D*vel[:, 1]

    f_spd = _xz3(vhxz * (V_CRUISE - sxz[:, None]))

    accel = (W_SEP*f_sep + W_ALI*f_ali + W_FIRE*f_fire
           + W_TAN*f_tan + W_ARC*f_arc + W_SPD*f_spd + W_ALT*f_alt)

    if sigma > 0:
        noise = rng.normal(0, sigma, N)
    else:
        noise = np.zeros(N)
    cn, sn = np.cos(noise), np.sin(noise)
    vn = np.c_[vel[:, 0]*cn - vel[:, 2]*sn, vel[:, 1], vel[:, 0]*sn + vel[:, 2]*cn]
    vel_new = DT * accel + vn

    sxzn = np.linalg.norm(vel_new[:, [0, 2]], axis=1, keepdims=True).clip(min=1e-6)
    scl  = np.where(sxzn > V_MAX, V_MAX/sxzn, 1.0)
    vel_new[:, [0, 2]] *= scl
    vel_new[:, 1] = np.clip(vel_new[:, 1], -VY_LIMIT, VY_LIMIT)

    pos_new = pos + DT * vel_new
    below = pos_new[:, 1] < GROUND_Y
    if below.any():
        pos_new[below, 1] = GROUND_Y;  vel_new[below, 1] = np.abs(vel_new[below, 1])

    return pos_new.astype('f4'), vel_new.astype('f4')

# ══════════════════════════════════════════════════════════════════════════════
#  MLX (Metal GPU) step — numerically equivalent to step_np with sigma=0
# ══════════════════════════════════════════════════════════════════════════════

if HAVE_MLX:
    PERIM_MX = mx.array(PERIM)
    _F32 = mx.float32
    _INF = mx.array(1e18, dtype=_F32)

    def _xz3_mx(axz, N):
        """(N,2) XZ vector → (N,3) with zero Y."""
        return mx.stack([axz[:, 0], mx.zeros(N, dtype=_F32), axz[:, 1]], axis=1)

    def step_mx(pos, vel, key=None, sigma=0.0):
        """One boids step on the GPU. pos,vel are mx.array (N,3) float32."""
        N = pos.shape[0]
        eye = mx.eye(N, dtype=mx.bool_)

        disp = pos[None] - pos[:, None]                          # (N,N,3)
        d3   = mx.sqrt(mx.sum(disp**2, axis=2))
        d3   = mx.where(eye, _INF, d3)
        dxz  = mx.sqrt(disp[:, :, 0]**2 + disp[:, :, 2]**2)
        dxz  = mx.where(eye, _INF, dxz)

        perc = dxz < R_PERC
        sep  = (d3 < R_SEP) & (d3 > 1e-6)
        nN   = mx.maximum(mx.sum(perc.astype(_F32), axis=1), 1.0)        # (N,)

        sw    = mx.where(sep, 1.0/d3**2, 0.0)                            # (N,N)
        f_sep = -mx.sum(sw[:, :, None] * disp, axis=1)                   # (N,3)

        sxz  = mx.sqrt(vel[:, 0]**2 + vel[:, 2]**2)                      # (N,)
        ok   = sxz > 1e-6
        sxz_safe = mx.where(ok, sxz, 1.0)
        vhx  = mx.where(ok, vel[:, 0]/sxz_safe, 0.0)
        vhz  = mx.where(ok, vel[:, 2]/sxz_safe, 0.0)
        vhxz = mx.stack([vhx, vhz], axis=1)                             # (N,2)
        ali  = mx.matmul(perc.astype(_F32), vhxz) / nN[:, None]          # (N,2)
        f_ali = _xz3_mx(ali, N)

        pxz  = mx.stack([pos[:, 0], pos[:, 2]], axis=1)                  # (N,2)
        diff = PERIM_MX[None, :, :] - pxz[:, None, :]                    # (N,N_P,2)
        dp   = mx.sqrt(mx.sum(diff**2, axis=2))                          # (N,N_P)
        kn   = mx.argmin(dp, axis=1)                                     # (N,)
        ps   = PERIM_MX[kn]                                             # (N,2)
        toP  = ps - pxz
        dP   = mx.maximum(mx.sqrt(mx.sum(toP**2, axis=1, keepdims=True)), 1e-6)
        f_fire_xz = toP / dP

        kf = (kn + 1) % N_PERIM
        kb = (kn - 1 + N_PERIM) % N_PERIM
        tan_raw = PERIM_MX[kf] - PERIM_MX[kb]
        tan_xz  = tan_raw / mx.maximum(
            mx.sqrt(mx.sum(tan_raw**2, axis=1, keepdims=True)), 1e-6)
        ramp     = mx.clip(1.0 - dP[:, 0]/R_PATROL, 0.0, 1.0)            # (N,)
        f_tan_xz = tan_xz * ramp[:, None]

        kni = kn.astype(mx.int32)
        dk  = kni[None, :] - kni[:, None]                                # (N,N)
        dk  = mx.where(dk >  N_PERIM//2, dk - N_PERIM, dk)
        dk  = mx.where(dk < -N_PERIM//2, dk + N_PERIM, dk)
        adk = mx.abs(dk)
        amask = (adk > 0) & (adk < R_ARC_IDX)
        adk_safe = mx.maximum(adk, 1).astype(_F32)
        aw  = mx.where(amask, -mx.sign(dk.astype(_F32))/adk_safe, 0.0)
        f_arc_xz = mx.sum(aw, axis=1)[:, None] * tan_xz

        f_alt_y = K_ALT_P*(ALT_NOM - pos[:, 1]) - K_ALT_D*vel[:, 1]      # (N,)
        f_spd_xz = vhxz * (V_CRUISE - sxz[:, None])

        accel = (W_SEP*f_sep + W_ALI*f_ali
               + W_FIRE*_xz3_mx(f_fire_xz, N) + W_TAN*_xz3_mx(f_tan_xz, N)
               + W_ARC*_xz3_mx(f_arc_xz, N) + W_SPD*_xz3_mx(f_spd_xz, N))
        accel = accel + mx.stack(
            [mx.zeros(N, dtype=_F32), W_ALT*f_alt_y, mx.zeros(N, dtype=_F32)], axis=1)

        if sigma > 0:
            noise = mx.random.normal((N,), key=key) * sigma
        else:
            noise = mx.zeros(N, dtype=_F32)
        cn, sn = mx.cos(noise), mx.sin(noise)
        vn = mx.stack([vel[:, 0]*cn - vel[:, 2]*sn, vel[:, 1],
                       vel[:, 0]*sn + vel[:, 2]*cn], axis=1)
        vel_new = DT * accel + vn

        sxzn = mx.maximum(mx.sqrt(vel_new[:, 0]**2 + vel_new[:, 2]**2), 1e-6)
        scl  = mx.where(sxzn > V_MAX, V_MAX/sxzn, 1.0)
        vnx  = vel_new[:, 0]*scl
        vnz  = vel_new[:, 2]*scl
        vny  = mx.clip(vel_new[:, 1], -VY_LIMIT, VY_LIMIT)

        pny  = pos[:, 1] + DT*vny
        pnx  = pos[:, 0] + DT*vnx
        pnz  = pos[:, 2] + DT*vnz
        below = pny < GROUND_Y
        pny   = mx.where(below, GROUND_Y, pny)
        vny   = mx.where(below, mx.abs(vny), vny)

        pos_new = mx.stack([pnx, pny, pnz], axis=1)
        vel_new = mx.stack([vnx, vny, vnz], axis=1)
        return pos_new, vel_new

# ── Coverage metric ────────────────────────────────────────────────────────────

def coverage(pxz, perim=None):
    """Fraction of perimeter points within R_SENSE of some drone (NumPy)."""
    if perim is None:
        perim = PERIM
    d = np.linalg.norm(perim[:, None, :] - pxz[None, :, :], axis=2).min(axis=1)
    return float((d < R_SENSE).mean())

if HAVE_MLX:
    def coverage_mx(pos):
        """Same metric on-device; keeps the presim hot loop off the CPU."""
        pxz = mx.stack([pos[:, 0], pos[:, 2]], axis=1)              # (N,2)
        d   = mx.sqrt(mx.sum((PERIM_MX[:, None, :] - pxz[None, :, :])**2, axis=2))
        return mx.mean((mx.min(d, axis=1) < R_SENSE).astype(_F32))

# ── Pre-simulation drivers ─────────────────────────────────────────────────────

def presim_np(seed=7, n_steps=N_STEPS, verbose=False):
    rng = np.random.default_rng(seed)
    pos, vel = init_state(rng)
    tp = np.empty((n_steps, N_DRONES, 3), 'f4')
    tv = np.empty((n_steps, N_DRONES, 3), 'f4')
    tc = np.empty(n_steps, 'f4')
    for t in range(n_steps):
        tp[t] = pos;  tv[t] = vel;  tc[t] = coverage(pos[:, [0, 2]])
        if verbose and t % 200 == 0:
            print(f"  [np] t={t*DT:.0f}s  cov={tc[t]*100:.1f}%  alt={pos[:,1].mean():.1f}m")
        pos, vel = step_np(pos, vel, rng)
    return tp, tv, tc

def presim_mx(seed=7, n_steps=N_STEPS, verbose=False):
    """GPU pre-sim. Physics + coverage stay on-device; only the (small) per-step
    state is copied to the host for the rendered trajectory. Returns NumPy."""
    if not HAVE_MLX:
        raise RuntimeError("MLX not available")
    rng = np.random.default_rng(seed)
    pos_np, vel_np = init_state(rng)
    pos = mx.array(pos_np);  vel = mx.array(vel_np)
    mx.eval(pos, vel)
    key = mx.random.key(seed)
    tp = np.empty((n_steps, N_DRONES, 3), 'f4')
    tv = np.empty((n_steps, N_DRONES, 3), 'f4')
    tc = np.empty(n_steps, 'f4')
    for t in range(n_steps):
        cov = coverage_mx(pos)                # on-device metric
        key, sub = mx.random.split(key)
        pos_n, vel_n = step_mx(pos, vel, key=sub, sigma=SIGMA_NOISE)
        mx.eval(pos_n, vel_n, cov)            # single sync per step
        tp[t] = np.array(pos);  tv[t] = np.array(vel);  tc[t] = float(cov)
        if verbose and t % 200 == 0:
            print(f"  [mx] t={t*DT:.0f}s  cov={tc[t]*100:.1f}%  alt={tp[t,:,1].mean():.1f}m")
        pos, vel = pos_n, vel_n
    return tp, tv, tc
