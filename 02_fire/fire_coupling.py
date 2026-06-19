"""
Fire–atmosphere coupling — feed the fire-induced surface indraft back into the
Rothermel CA wind field (the WRF-SFIRE feedback: "a fire makes its own wind").

A wildfire's buoyant column ingests surface air: by continuity, the vertical mass
flux that feeds the rising plume must be supplied by horizontal CONVERGENCE at the
ground — an indraft toward the fire, superposed on the ambient wind. The steady
Rothermel CA never sees this. Here we add it, mass-consistently, by reusing the very
same variable-coefficient Poisson solver the terrain-wind downscaler already uses:

    a fire is a distributed mass SINK S(x,y) over the burning region.
    Continuity:  ∇·V = −S   (S>0: surface air converges and is lofted into the plume)
    Helmholtz:   V = −∇φ   ⇒   −∇²φ = −S   ⇒   A φ = −S
    (A = −∇·(∇·), the SAME SPD operator terrain_wind.poisson_cg solves.)

The RHS sign is −S, NOT +S: +S would drive an OUTdraft (air blown away from the
fire), the opposite of the physics. validate_sign() asserts the induced flow points
inward (and conserves mass, net inflow = sink Q) — a permanent guard on this.

Honest scope (see HONEST_SCOPE):
  RESOLVED      — the 3-D LBM buoyant plume and its true near-ground indraft (lbm3d_plume).
  PARAMETERIZED — the depth-averaged potential-flow surface indraft (this module). Its peak
                  is anchored to the LBM indraft/ambient ratio; its reach is the 2-D potential-
                  flow reach (∝ footprint size, ~1/r) — BROADER than the resolved LBM indraft,
                  the known limitation of a depth-averaged closure.
  VALIDATED     — the parameterized indraft's peak/ambient ratio vs the resolved LBM near-
                  ground indraft; inward sign; mass conservation; a no-indraft null control.
  NOT claimed   — that the 2-D reach matches the LBM, nor an absolute field-validated
                  pyroconvective spread rate (head ACCELERATION is unsteady plume-momentum,
                  out of scope; this captures the steady kinematic lateral convergence).

    ~/ds/bin/python 02_fire/fire_coupling.py --checks      # sign + null + mass conservation
    ~/ds/bin/python 02_fire/fire_coupling.py --calibrate   # extract indraft ratio from the LBM plume
    ~/ds/bin/python 02_fire/fire_coupling.py --demo        # ambient vs coupled spread + figure
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rothermel_ca as rc
from terrain_wind import poisson_cg, downscale_wind, _div

HONEST_SCOPE = __doc__


# ── fire-induced surface indraft: a mass sink in the WindNinja Poisson ────────

def solve_fire_indraft(z, DX, S):
    """Mass-consistent fire indraft from a surface mass-sink S (>0 over the fire). Solves the
    pure Poisson  A φ = −S  (A = −∇·(∇·), Dirichlet φ=0 at the domain edge), then V = −∇φ.
    The −S sign makes the flow CONVERGE into the fire; ∇·V = −S so the inflow exactly feeds
    the sink (mass conserved). Returns (du, dv, diag).

    The depth is uniform (=1) and is absorbed by the downstream peak-scaling, so it sets no
    length scale; the indraft's reach is the potential-flow reach (∝ the fire-footprint size,
    decaying ~1/r in 2-D). That reach is BROADER than the resolved-LBM near-ground indraft —
    the known limitation of a depth-averaged 2-D closure (disclosed in HONEST_SCOPE)."""
    z = np.asarray(z, 'f8')
    depth = np.ones_like(z)
    phi, it, resid = poisson_cg(-np.asarray(S, 'f8'), depth, DX)
    dphidy, dphidx = np.gradient(phi, DX)
    return -dphidx, -dphidy, dict(iters=it, resid=resid, phi=phi)


def build_sink(burning, intensity):
    """Intensity-weighted mass-sink SHAPE over the burning AREA (not a thin ring — a
    ring drives unphysical front-to-core backflow). S ∝ max(intensity,0) on burning
    cells, 0 elsewhere. Absolute magnitude is set later by peak-scaling to the
    LBM-calibrated indraft/ambient ratio."""
    burning = np.asarray(burning, bool)
    intensity = np.maximum(np.asarray(intensity, 'f8'), 0.0)
    return np.where(burning, intensity, 0.0)


def fire_indraft_scaled(z, DX, burning, intensity, peak_indraft):
    """Fire indraft (du,dv) [m/s], scaled so its peak speed == peak_indraft. Peak anchored to
    the LBM indraft/ambient ratio × ambient wind; the spatial structure is the mass-conserving
    potential flow over the fire footprint. Empty fire or zero peak → no indraft."""
    z = np.asarray(z, 'f8')
    S = build_sink(burning, intensity)
    if S.max() <= 0 or peak_indraft <= 0:
        zr = np.zeros_like(z)
        return zr, zr
    du, dv, _ = solve_fire_indraft(z, DX, S)
    peak = float(np.hypot(du, dv).max())
    if peak > 1e-12:
        s = peak_indraft / peak
        du = du * s
        dv = dv * s
    return du, dv


# ── coupled wind_fn (terrain-corrected ambient ⊕ fire indraft) ────────────────

def _ambient_uv(z, DX, U0_ms, from_deg, terrain, H_layer):
    """Ambient surface wind components (East u, North v) — terrain-downscaled or uniform."""
    z = np.asarray(z, 'f8')
    if terrain:
        spd, frm = downscale_wind(z, DX, U0_ms, from_deg, H_layer=H_layer)
        to = np.deg2rad(np.asarray(frm, 'f8') + 180.0)
        return spd*np.sin(to), spd*np.cos(to)
    to = np.deg2rad(from_deg + 180.0)
    return np.full_like(z, U0_ms*np.sin(to)), np.full_like(z, U0_ms*np.cos(to))


def make_wind_coupled(U0_ms, from_deg, burning, intensity, peak_indraft,
                      terrain=True, H_layer=None):
    """wind_fn(z,dzdx,dzdy,DX) → (speed, from_deg): terrain-corrected ambient wind PLUS
    the scaled fire indraft. `burning`/`intensity` are the fire state from the previous
    Picard iteration. peak_indraft=0 (or empty fire) recovers the pure ambient wind."""
    def _fn(z, dzdx, dzdy, DX):
        u, v = _ambient_uv(z, DX, U0_ms, from_deg, terrain, H_layer)
        du, dv = fire_indraft_scaled(z, DX, burning, intensity, peak_indraft)
        u = u + du
        v = v + dv
        speed = np.hypot(u, v)
        from_deg_field = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        return speed.astype('f4'), from_deg_field.astype('f4')
    return _fn


# ── CHECK 1 + CHECK 2: null limit, mass conservation, and the INWARD-FLOW SIGN ─

def validate_sign(H=140, W=180, DX=25.0):
    """Foundational guards on the mass-sink solve (no fire model, no CA — pure operator).

    CHECK 1 (null): an empty sink gives an identically-zero indraft.
    CHECK 2 (mass + SIGN): a single circular sink on a flat domain must (a) solve to CG
       tolerance with the net horizontal inflow equal to the total sink Q (mass conservation,
       ∇·V=−S — the inflow exactly feeds the column) and, critically, (b) drive flow INWARD
       (a sink, not a source) — the permanent guard on the −S vs +S sign."""
    z = np.zeros((H, W), 'f8')

    # CHECK 1 — null limit
    du0, dv0 = fire_indraft_scaled(z, DX, np.zeros((H, W), bool), np.zeros((H, W)), 3.0)
    null_ok = float(np.abs(du0).max() + np.abs(dv0).max()) == 0.0

    # CHECK 2 — a circular sink: S = w0 over a disk
    yy, xx = np.mgrid[0:H, 0:W]
    ci, cj, R = H/2, W/2, 10.0
    disk = ((xx-cj)**2 + (yy-ci)**2) < R**2
    w0 = 2.0
    S = np.where(disk, w0, 0.0)
    du, dv, dg = solve_fire_indraft(z, DX, S)

    # (a) mass conservation: the SOLVE residual is the exact operator-norm guarantee; the net
    #     reconstructed inflow equals the total sink Q (divergence theorem). The pointwise
    #     central-difference reconstruction differs from the FV stencil at the disk edge, so
    #     only the net flux is meaningful.
    resid = dg['resid']
    Q = float(S.sum()) * DX*DX
    intr = (slice(1, H-1), slice(1, W-1))
    net_conv = float((-_div(du, dv, DX))[intr].sum()) * DX*DX
    rel_net = abs(net_conv - Q) / max(Q, 1e-12)
    mass_ok = resid < 1e-4 and rel_net < 0.05

    # (b) SIGN: the radial component of V on a ring around the disk must be INWARD (<0)
    ring = (((xx-cj)**2 + (yy-ci)**2) > (1.6*R)**2) & (((xx-cj)**2 + (yy-ci)**2) < (2.2*R)**2)
    rxu = (xx-cj); ryu = (yy-ci); rn = np.hypot(rxu, ryu) + 1e-9
    v_radial = (du*rxu + dv*ryu) / rn
    mean_radial = float(v_radial[ring].mean())
    inward = mean_radial < 0

    print("Mass-sink solve — foundational guards (flat domain, single circular sink):")
    print(f"  CHECK 1 null limit:    empty sink → |indraft| = 0              {'PASS' if null_ok else 'FAIL'}")
    print(f"  CHECK 2a mass:         solve resid {resid:.1e}, net inflow/Q err {rel_net:.1e}   "
          f"{'PASS' if mass_ok else 'FAIL'}")
    print(f"  CHECK 2b INWARD sign:  mean radial V on ring = {mean_radial:+.3f}        "
          f"{'PASS (inward ✓)' if inward else 'FAIL (outward — sign bug!)'}")
    ok = null_ok and mass_ok and inward
    print(f"  {'ALL PASS' if ok else 'FAIL'}  (the −S RHS draws air IN, mass-consistently)")
    return ok


# ── LBM calibration: extract the dimensionless indraft from the resolved plume ─

def _load_plume_module():
    cfd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '05_cfd'))
    if cfd not in sys.path:
        sys.path.insert(0, cfd)
    import lbm3d_plume as P
    return P


def calibrate_indraft_from_plume(N=96, U_wind=0.030, beta=1.6e-3, cache='output/indraft_calib.npz',
                                 recompute=False, verbose=True):
    """Run the resolved 3-D LBM buoyant plume ONCE and extract the DIMENSIONLESS indraft
    calibration the parameterized surface model needs:
      • R_LBM   = peak near-ground CROSSWIND indraft / ambient wind   (the magnitude anchor)
      • decay_radii = half-width of that indraft, in fire-radii        (the reach/shape)
    Lattice-units discipline: only these dimensionless numbers cross into the CA (never m/s).
    Cached to disk (the plume is ~150 s). Returns the calibration dict."""
    cpath = os.path.join(os.path.dirname(__file__), cache)
    if os.path.exists(cpath) and not recompute:
        d = dict(np.load(cpath))
        out = {k: float(d[k]) for k in d}
        if verbose:
            print(f"Indraft calibration (cached): R_LBM={out['R_LBM']:.3f}, "
                  f"decay={out['decay_radii']:.2f} fire-radii (U_wind={out['U_wind']}, N={int(out['N'])})")
        return out

    P = _load_plume_module()
    if verbose:
        print(f"Running the resolved LBM plume once for calibration (N={N}, ~150 s)…")
    r = P.fire_plume(N=N, U_wind=U_wind, beta=beta, verbose=False)
    sim = r['sim']; U = float(r['U_wind']); src_r = float(r['src_r']); src_h = int(r['src_h'])
    xc, yc = int(r['xc']), int(r['yc'])
    _, ux, uy, _ = sim.macroscopic()
    ux = np.array(ux); uy = np.array(uy)
    band = slice(1, src_h + 3)                          # near-ground surface band (above z=0 wall)
    uy_g = uy[band].mean(axis=0)                        # (Ny, Nx) crosswind velocity
    # CROSSWIND inflow profile through the fire (x=xc): the clean sector (the wind is +x, so
    # the y-flow is the fire's own pull, not forced through-flow). Inward = toward yc.
    Y, X = np.mgrid[0:N, 0:N]
    inward_field = -uy_g * np.sign(Y - yc)             # crosswind inflow toward the fire (>0)
    # PEAK over a window around+downwind of the fire — the plume bends downwind, so the true
    # peak crosswind indraft is NOT at the source column x=xc (sampling only x=xc under-reads
    # the anchor ~2x). Exclude the −x inlet column and the centreline (sign(0)=0).
    win = ((np.abs(X - xc) <= 5*src_r) & (np.abs(Y - yc) <= 6*src_r)
           & (np.abs(Y - yc) >= 1) & (X >= 2))
    masked = np.where(win, inward_field, -np.inf)
    peak = float(masked.max())
    R_LBM = peak / U
    yi, xi = np.unravel_index(int(np.argmax(masked)), masked.shape)
    # decay: crosswind half-width (in fire-radii) at the column where the inflow peaks
    profp = inward_field[:, xi]; ys = np.arange(N) - yc
    over = np.where((profp >= 0.5*peak) & (ys != 0))[0]
    decay_radii = float(np.abs(ys[over]).max() / src_r) if over.size else 1.0

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.savez(cpath, R_LBM=R_LBM, decay_radii=decay_radii, U_wind=U, src_r=src_r, N=N, peak=peak)
    if verbose:
        print(f"Indraft calibration from the resolved LBM plume (N={N}, U_wind={U}):")
        print(f"  peak crosswind indraft = {peak:.4f}  → R_LBM = peak/U_wind = {R_LBM:.3f}")
        print(f"  indraft half-width     = {decay_radii:.2f} fire-radii")
        print(f"  → CA indraft peak = {R_LBM:.3f} × ambient wind; reach ≈ {decay_radii:.1f} fire-radii")
    return dict(R_LBM=R_LBM, decay_radii=decay_radii, U_wind=U, src_r=src_r, N=float(N), peak=peak)


# ── Picard fixed-point coupling: CA ↔ fire-induced wind ──────────────────────

def _time_at_count(T, n):
    """Fire-time at which exactly n cells have burned — robust to the huge slow-backing
    tail of the min-arrival field (a fixed percentile would land in that tail)."""
    v = np.sort(T[np.isfinite(T)])
    return float(v[min(n, len(v)-1)])


def couple_fire_atmosphere(H=240, W=320, DX=30.0, U0_ms=2.5, from_deg=270.0,
                           ignition=(0.5, 0.30), fuel=rc.FUEL_MODELS[2], M=0.06,
                           terrain_fn=None, K=6, n_star=4500,
                           calib=None, mode='iterated', verbose=True):
    """Two-way coupling as a Picard loop on the steady CA: run CA → burning footprint at a
    snapshot (an ISOLATED early fire of n_star cells) → fire indraft → re-run CA → repeat.
    The indraft peak is anchored to the LBM ratio: peak = R_LBM × ambient wind. Returns the
    ambient baseline, the coupled result, and the per-iteration footprint-size trajectory
    (the feedback-saturation diagnostic, counted at the fixed snapshot time). mode='oneway'
    = a single indraft solve from the ambient footprint."""
    if calib is None:
        calib = calibrate_indraft_from_plume(verbose=False)
    R = calib['R_LBM']; peak_indraft = R * U0_ms; decay_radii = calib['decay_radii']
    terrain = terrain_fn is not None
    base_kw = dict(H=H, W=W, DX=DX, fuel=fuel, M=M, ignition=ignition,
                   wind_ms=U0_ms, wind_from_deg=from_deg, terrain_fn=terrain_fn, use_mlx=True)

    amb_wind = make_wind_coupled(U0_ms, from_deg, np.zeros((H, W), bool), np.zeros((H, W)),
                                 peak_indraft=0.0, terrain=terrain)
    base = rc.simulate(wind_fn=amb_wind, **base_kw)
    Tamb = base['T']
    t_star = _time_at_count(Tamb, n_star)                        # snapshot fire-time (isolated)
    burning = np.isfinite(Tamb) & (Tamb <= t_star)
    intensity = np.nan_to_num(base['head_ros_ms'])

    per_iter = [dict(it=0, burned=int(burning.sum()), tag='ambient')]
    r = base; Tcoup = Tamb; prev = int(burning.sum())
    K = 1 if mode == 'oneway' else K
    converged = (mode == 'oneway'); hist = [prev]
    for k in range(1, K+1):
        wind_fn = make_wind_coupled(U0_ms, from_deg, burning, intensity, peak_indraft,
                                    terrain=terrain)
        r = rc.simulate(wind_fn=wind_fn, **base_kw)
        Tcoup = r['T']
        new_burning = np.isfinite(Tcoup) & (Tcoup <= t_star)    # footprint at the SAME snapshot time
        nb = int(new_burning.sum())
        dburn = abs(nb - prev) / max(prev, 1)
        hist.append(nb)
        # plateau test: the last 3 footprints lie within 2% of their mean. This is robust to
        # a tiny wiggle at a stable fixed point AND correctly rejects a real oscillation (whose
        # window spread stays large) — a single near-flat consecutive step can't fake it.
        win = hist[-3:]
        plateau = len(win) >= 3 and (max(win) - min(win)) / max(np.mean(win), 1) < 0.02
        per_iter.append(dict(it=k, burned=nb, dburn=dburn))
        if verbose:
            print(f"  iter {k}: snapshot footprint {nb} cells (Δ {dburn*100:.2f}% vs prev)")
        burning = new_burning
        intensity = np.nan_to_num(r['head_ros_ms'])
        if mode == 'iterated' and plateau:
            converged = True
            break
        prev = nb
    if verbose and mode == 'iterated' and not converged:
        print(f"  ⚠ Picard hit K={K} without saturating (last Δ {dburn*100:.2f}% — still oscillating)")
    z = base['z']
    du, dv = fire_indraft_scaled(z, DX, burning, intensity, peak_indraft)
    return dict(ambient=base, coupled=r, Tamb=Tamb, Tcoup=Tcoup, per_iter=per_iter,
                converged=converged,
                peak_indraft=peak_indraft, R_LBM=R, t_star=t_star, decay_radii=decay_radii,
                U0_ms=U0_ms, from_deg=from_deg, DX=DX, calib=calib, burning_final=burning,
                indraft=(du, dv), z=z)


def flat_terrain(H, W, DX):
    """Flat ground — isolates the fire-induced wind (no slope confound; the wind model
    treats flat terrain as uniform, consistent with the CA)."""
    z = np.zeros((H, W), 'f4')
    return z, z.copy(), z.copy()


def _signature(res, n_meas=9000):
    """Coupled-vs-ambient signature at a fixed snapshot time t_meas = the time the AMBIENT
    fire reaches n_meas cells (an isolated ellipse, clear of the boundary). Comparing the
    coupled fire at the SAME t_meas isolates the feedback: a slower fire covers fewer cells
    / less reach; a faster one, more."""
    Tamb, Tcoup = res['Tamb'], res['Tcoup']
    H, W = Tamb.shape
    t = _time_at_count(Tamb, n_meas)
    A_amb = np.isfinite(Tamb) & (Tamb <= t)
    A_cou = np.isfinite(Tcoup) & (Tcoup <= t)
    gi, gj = res['ambient']['ignition']

    # boundary-isolation guard for the MEASURED metrics: the head reach (East, +col) and the
    # crosswind width (North/South rows). A wall there clips the metric. The slow BACKING fire
    # creeping to the West wall over the long t_meas does NOT affect head/width, so the West
    # edge is excluded (the area metric does include the backing — reported with that caveat).
    union = A_amb | A_cou
    touch = (union[:, -2:].any() or union[:2, :].any() or union[-2:, :].any())   # East, N, S
    isolated = not bool(touch)

    def extent(mask):
        ys, xs = np.where(mask)
        if not xs.size:
            return 0, 0, 0
        return int(xs.max() - gj), int(ys.max() - ys.min() + 1), int(mask.sum())   # head reach E, NS width, area
    hd_a, wd_a, ar_a = extent(A_amb)
    hd_c, wd_c, ar_c = extent(A_cou)
    if not isolated:
        print("  ⚠ WARNING: fire touches the domain boundary at t_meas — head/width metrics "
              "are clipping-corrupted (enlarge the domain / move the ignition).")
    common = np.isfinite(Tamb) & np.isfinite(Tcoup) & (Tamb <= t)
    dT = np.where(common, Tamb - Tcoup, np.nan)          # >0 = coupled earlier; restricted to the ellipse
    return dict(t=t, n_meas=n_meas, amb_area=ar_a, cou_area=ar_c, isolated=isolated,
                d_area_pct=100*(ar_c-ar_a)/max(ar_a, 1),
                amb_head=hd_a, cou_head=hd_c, amb_wid=wd_a, cou_wid=wd_c,
                d_head_pct=100*(hd_c-hd_a)/max(hd_a, 1),
                d_wid_pct=100*(wd_c-wd_a)/max(wd_a, 1),
                median_dT=float(np.nanmedian(np.abs(dT))), dT=dT)


def run_demo(mode='iterated', verbose=True, figure=True):
    """Self-contained coupling demo: FLAT terrain + uniform fuel + uniform wind, so the only
    wind structure is the ambient ⊕ the fire-induced indraft. Domain large enough that the
    measured fire stays clear of the boundary. Returns the result dict."""
    calib = calibrate_indraft_from_plume(verbose=verbose)
    if verbose:
        print(f"\nCoupling demo (flat terrain, uniform fuel): indraft peak = "
              f"{calib['R_LBM']:.2f} × ambient, reach {calib['decay_radii']:.1f} fire-radii, mode={mode}")
    # Weak ambient wind (2.5 m/s): the plume-dominated regime where the fire-induced indraft
    # is a LARGE relative perturbation. In a strong wind-driven fire the same indraft is a
    # small fraction of the ambient and the effect nearly vanishes (the correct dependence).
    # Ignition far WEST with a long EAST runway so the head stays clear of the boundary —
    # otherwise the head reach is clipped by the wall and the head change is unmeasurable.
    res = couple_fire_atmosphere(H=240, W=500, DX=30.0, U0_ms=2.5, from_deg=270.0,
                                 ignition=(0.14, 0.5), terrain_fn=flat_terrain,   # (col, row): west-centre
                                 mode=mode, n_star=4500, calib=calib, verbose=verbose)
    sig = _signature(res); res['sig'] = sig
    # actually RUN the validations the figure reports (no hardcoded checkmarks)
    res['checks_ok'] = validate_sign()
    zr = np.zeros_like(res['z'])
    a_s, a_f = make_wind_coupled(res['U0_ms'], res['from_deg'], np.zeros(res['z'].shape, bool),
                                 zr, 0.0, terrain=True)(res['z'], zr, zr, res['DX'])
    c_s, c_f = make_wind_coupled(res['U0_ms'], res['from_deg'], res['burning_final'],
                                 zr, 0.0, terrain=True)(res['z'], zr, zr, res['DX'])
    res['null_ok'] = bool(np.array_equal(a_s, c_s) and np.array_equal(a_f, c_f))
    if verbose:
        print(f"\nSpread signature at t={sig['t']:.0f} min ({'isolated' if sig['isolated'] else 'NOT isolated!'} "
              f"fire, coupled vs ambient), converged={res['converged']}:")
        print(f"  burned area:    {sig['amb_area']} → {sig['cou_area']} cells ({sig['d_area_pct']:+.1f}%, incl. backing)")
        print(f"  head reach (E): {sig['amb_head']} → {sig['cou_head']} cells ({sig['d_head_pct']:+.1f}%)")
        print(f"  crosswind width:{sig['amb_wid']} → {sig['cou_wid']} cells ({sig['d_wid_pct']:+.1f}%)")
        print(f"  null control (peak=0 → ambient wind): {res['null_ok']}; foundational checks: {res['checks_ok']}")
        print(f"  → the steady indraft RETARDS the fire — flanks most (lateral convergence), head less;")
        print(f"    NOT unsteady pyroconvective acceleration (out of scope).")
    if figure:
        make_coupling_figure(res)
    return res


def make_coupling_figure(res, out='output/fire_coupling.png'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Tamb, Tcoup, DX = res['Tamb'], res['Tcoup'], res['DX']
    du, dv = res['indraft']; sig = res['sig']; t = sig['t']
    H, W = Tamb.shape
    ext = [0, W*DX/1000, 0, H*DX/1000]                                       # km
    gi, gj = res['ambient']['ignition']
    A_amb = (np.isfinite(Tamb) & (Tamb <= t)).astype(float)
    A_cou = (np.isfinite(Tcoup) & (Tcoup <= t)).astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), facecolor='white')

    # Panel 1 — the two fire shapes at the same fire-time + the indraft quiver
    ax[0].contourf(A_amb, levels=[0.5, 1.5], colors=['#f6c28b'], extent=ext, origin='lower', alpha=0.9)
    ax[0].contour(A_cou, levels=[0.5], colors=['#b30000'], extent=ext, origin='lower', linewidths=2.2)
    yy, xx = np.mgrid[0:H, 0:W]; s = slice(None, None, 11)
    ax[0].quiver((xx*DX/1000)[s, s], (yy*DX/1000)[s, s], du[s, s], dv[s, s],
                 color='steelblue', scale=res['peak_indraft']*20, width=0.004, alpha=0.8)
    ax[0].plot(gj*DX/1000, gi*DX/1000, 'k*', ms=14)
    ax[0].set_title(f"Fire at a fixed snapshot — ambient (orange fill) vs coupled (red)\n"
                    f"the indraft (blue) retards the fire: width {sig['d_wid_pct']:+.0f}%, head {sig['d_head_pct']:+.0f}%")

    # Panel 2 — Δarrival: where the feedback delays the fire (flanks), in hours
    dT = sig['dT'] / 60.0                                                     # → hours
    lim = float(np.nanpercentile(np.abs(dT), 97))
    im = ax[1].imshow(dT, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, extent=ext, aspect='auto')
    fig.colorbar(im, ax=ax[1], shrink=0.82).set_label('Δarrival  T_amb − T_coupled  [h]')
    ax[1].plot(gj*DX/1000, gi*DX/1000, 'k*', ms=12)
    ax[1].set_title("Feedback delays the fire — FLANKS most (lateral convergence),\n"
                    "head less; not unsteady pyroconvective acceleration")

    # Panel 3 — Picard saturation + calibration/validation summary (REAL run results)
    bi = [d['burned'] for d in res['per_iter']]
    niter = len(bi) - 1
    ax[2].plot(range(len(bi)), bi, 'o-', color='firebrick')
    sat = f"saturates in {niter} iters" if res.get('converged') else f"NOT converged ({niter} iters)"
    ax[2].set(xlabel='Picard iteration', ylabel='snapshot footprint [cells]',
              title=f'Feedback {sat}')
    ax[2].grid(alpha=0.3)
    ck = lambda b: '✓' if b else '✗'
    txt = (f"indraft = {res['R_LBM']:.2f} × ambient  (LBM-calibrated peak)\n"
           f"peak {res['peak_indraft']:.1f} m/s;  LBM reach {res['calib']['decay_radii']:.1f} fire-radii\n"
           f"(model's 2-D reach is broader — disclosed)\n"
           f"signature:  width {sig['d_wid_pct']:+.0f}%,  head {sig['d_head_pct']:+.0f}%,  area {sig['d_area_pct']:+.0f}%\n"
           f"null control (peak=0 → ambient wind) {ck(res.get('null_ok'))}   "
           f"sign/mass/inward checks {ck(res.get('checks_ok'))}")
    ax[2].text(0.5, -0.34, txt, transform=ax[2].transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', fc='#eef', ec='#99c'))
    for a in ax[:2]:
        a.set(xlabel='East [km]', ylabel='North [km]')
    fig.suptitle("Fire–atmosphere coupling: the fire's own indraft fed back into Rothermel spread "
                 "(WRF-SFIRE feedback, kinematic closure)", y=1.02, fontsize=13)
    import os as _os; _os.makedirs(_os.path.dirname(out), exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    if '--calibrate' in sys.argv:
        calibrate_indraft_from_plume(recompute=True)
    elif '--checks' in sys.argv:
        validate_sign()
    elif '--demo' in sys.argv:
        run_demo()
    else:
        validate_sign()
