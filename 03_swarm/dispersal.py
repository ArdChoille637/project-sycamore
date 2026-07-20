"""
Project Sycamore — passive-autorotation DISPERSAL & coverage (swarm-scale).

This is the "step up" that connects the validated Phase-1 single-vehicle
autorotation aerodynamics to the swarm layer. The existing boids modules
(`swarm_core.py`, `fire_swarm.py`) model the ACTIVE, powered patrol — generic
holonomic agents flying at a placeholder cruise speed. They contain no samara
physics. This module models the other half of the bi-modal concept, the half
that IS the biomimicry: **passive dispersal**.

A samara is a wind-dispersed seed. A samara-UAV can be dispersed the same way —
released as a cloud at altitude upwind of a fire, each unit transitions to
autorotation and descends passively while drifting on the wind, its downward
sensor sweeping a swath of ground on the way down. No propulsion is spent until
(optionally) the unit transitions back to powered flight near the ground.

The one number that makes this concrete — the steady autorotation **descent
rate V_d** — is imported live from the Phase-1 solver (`01_aero/samara_bem.py`,
`SAMARA` config), so this model rests on a *Validated* result, not a guess.
Everything the descent rate then drives (drift, footprint, coverage) is
*Indicative*: it depends on the release geometry and a first-order wind-advection
transport model, and is honestly tiered as such.

Key modelling finding (why the metric is AREA, not perimeter): a passive cloud
drifts as one blob and can only seed an AREA — it cannot ring an 800 m fire
perimeter the way a powered patrol can. And the drift distance is bounded,
`drift = wind · H / V_d`, so one release sweeps a limited downwind band. Fleet
size, release altitude, and curtain size therefore trade off against how much of
a fire a single dispersal can blanket — which is exactly what this module maps.

Tiering (house style):
  * V_d (terminal autorotation sink)            — Validated  (Phase-1, Sycamore-A anchored)
  * spin-up transient (alt/time to terminal)    — Indicative (Phase-1 samara_transient brackets)
  * wind drift = ∫wind dt  (seed advection)     — Indicative (passive body ≈ moves with wind)
  * footprint / area coverage                   — Indicative (release-geometry + wind dependent)

Run:  ~/ds/bin/python 03_swarm/dispersal.py            # headless static summary → output/dispersal.png
      ~/ds/bin/python 03_swarm/dispersal.py --live     # live interactive descent window
      ~/ds/bin/python 03_swarm/dispersal.py --gif [p]  # headless animation → output/dispersal.gif
      ~/ds/bin/python 03_swarm/dispersal.py --mp4 [p]  # headless animation → output/dispersal.mp4
      ~/ds/bin/python 03_swarm/dispersal.py --validate # self-checks only (no matplotlib)

Live vs headless is chosen by the entry point, not hardcoded: --live selects the
native macosx backend (or $MPLBACKEND); every other path forces Agg. The same
precomputed dispersal drives all of them — only playback/target differs.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Phase-1 aero (the Validated descent rate) and the shared fire target.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '01_aero'))
import samara_bem as sb          # noqa: E402
import swarm_core as sc          # noqa: E402  (FIRE_A/FIRE_B/PERIM — same fire the patrol covers)

G = sb.G

# ── Phase-1 transient brackets (Indicative; from samara_transient, see STATUS.md) ──
# Altitude and time a released unit consumes reaching terminal autorotation.
#   nominal — released already spinning (dropped from powered-flight rotor state)
#   cold    — released not rotating (ejected cold); converges but eats more height
SPINUP = {
    'nominal': dict(alt=8.1,  t=3.0),    # no AoA overshoot
    'cold':    dict(alt=45.7, t=14.5),   # AoA overshoot to ~66°, basin still 100%
}


# ══════════════════════════════════════════════════════════════════════════════
#  Release configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Release:
    """A dispersal release of M samara-UAVs over the (swarm_core) fire ellipse.

    Frame: x=East, z=North, y=altitude; fire centred at the origin. Wind is
    meteorological `wind_from` (270 = from the west, blows east). The release
    patch is defined in the WIND-ALIGNED frame — `w_along` half-width along the
    wind, `w_cross` half-width across it — because a wind-transverse curtain is
    the natural dispersal geometry (it drifts across the fire as it descends)."""
    M:            int   = 150         # fleet size
    mass_kg:      float = 0.075       # per-unit mass (SAMARA design point)
    alt:          float = 600.0       # release altitude AGL [m]
    offset_up:    float = 250.0       # release-patch centroid, this far UPWIND of fire centre [m]
    w_along:      float = 350.0       # release half-width ALONG wind [m]
    w_cross:      float = 450.0       # release half-width ACROSS wind [m] (the curtain span)
    turb_i:       float = 0.15        # wind-variability intensity σ_gust/|wind| across the cloud
    wind_ms:      float = 2.2         # wind speed [m/s]  (matches fire_swarm F_WIND_MS)
    wind_from:    float = 270.0       # wind FROM this compass deg (270 = from west)
    spinup:       str   = 'cold'      # 'nominal' | 'cold' transient bracket
    fov_deg:      float = 60.0        # downward sensor full field-of-view [deg]
    r_det_max:    float = 150.0       # detection-radius cap (honesty bound on FOV footprint) [m]
    seed:         int   = 7
    cfg:          sb.Config = field(default_factory=lambda: sb.SAMARA)

    # derived (filled by solve_descent)
    Vd:           float = field(default=np.nan, init=False)
    rpm:          float = field(default=np.nan, init=False)


def wind_vector(speed, from_deg):
    """Meteorological from_deg → horizontal (East,North) velocity the air (and a
    passive body) travels. 270 (from west) → +East."""
    toward = np.deg2rad((from_deg + 180.0) % 360.0)      # compass: 0=N(+z), 90=E(+x)
    return np.array([speed * np.sin(toward), speed * np.cos(toward)], dtype=float)  # (East, North)


def wind_frame(from_deg):
    """Downwind unit vector d and cross-wind unit vector c (both East,North)."""
    d = wind_vector(1.0, from_deg)                        # unit downwind (dir only)
    c = np.array([-d[1], d[0]])                           # +90° rotation
    return d, c


def solve_descent(rel: Release):
    """Import the Validated terminal autorotation descent rate for this config/mass."""
    Vd, Om = sb.solve(rel.cfg, rel.mass_kg)
    rel.Vd  = float(Vd)
    rel.rpm = float(Om * 60 / (2 * np.pi))
    return rel.Vd


# ══════════════════════════════════════════════════════════════════════════════
#  Dispersal simulation
# ══════════════════════════════════════════════════════════════════════════════

def descent_time(rel: Release):
    """Total time from release to ground: spin-up transient + terminal descent
    over the remaining altitude. Requires alt > spin-up altitude."""
    su = SPINUP[rel.spinup]
    if rel.alt <= su['alt']:
        raise ValueError(f"release altitude {rel.alt} m below spin-up altitude "
                         f"{su['alt']} m ({rel.spinup}); unit never reaches terminal")
    return su['t'] + (rel.alt - su['alt']) / rel.Vd


def simulate(rel: Release, n_t=60):
    """Simulate the dispersal cloud from release to landing.

    Returns per-unit trajectory (n_t, M, 3) with x=East, y=alt, z=North.
    Descent is constant-V_d after a spin-up transient. Horizontal motion is
    **wind advection** — a passively autorotating body is a large drag disc with
    little lateral control authority, so it rides the local wind (the classic
    seed-dispersal model: terminal velocity + wind transport). Cloud SPREAD
    comes from wind variability across the cloud (each unit rides wind + a
    frozen per-unit gust, σ_gust = turb_i·|wind|), not a persistent ballistic
    ejection (which a drag disc would damp in seconds)."""
    if np.isnan(rel.Vd):
        solve_descent(rel)
    rng = np.random.default_rng(rel.seed)
    M = rel.M
    w2 = wind_vector(rel.wind_ms, rel.wind_from)
    d, c = wind_frame(rel.wind_from)

    # release patch: uniform rectangle in the wind-aligned frame, centred
    # offset_up UPWIND of the fire centre.
    centre = -d * rel.offset_up
    a = rng.uniform(-rel.w_along, rel.w_along, M)         # along-wind coord
    b = rng.uniform(-rel.w_cross, rel.w_cross, M)         # cross-wind coord
    p0 = centre[None, :] + a[:, None] * d[None, :] + b[:, None] * c[None, :]   # (M,2) EN
    x0, z0 = p0[:, 0], p0[:, 1]
    release_mean = p0.mean(axis=0)

    # per-unit frozen wind gust: spread from spatial wind variability (∝ wind)
    gust = rng.normal(0, rel.turb_i * rel.wind_ms, (M, 2))

    T = descent_time(rel)
    t = np.linspace(0, T, n_t)
    su = SPINUP[rel.spinup]
    alt = np.where(t <= su['t'],
                   rel.alt - (su['alt'] / su['t']) * t,
                   rel.alt - su['alt'] - rel.Vd * (t - su['t']))
    alt = np.clip(alt, 0.0, None)

    vx = w2[0] + gust[:, 0]
    vz = w2[1] + gust[:, 1]
    X = x0[None, :] + vx[None, :] * t[:, None]
    Z = z0[None, :] + vz[None, :] * t[:, None]
    Y = np.broadcast_to(alt[:, None], (n_t, M))

    traj = np.stack([X, Y, Z], axis=2).astype('f4')
    land = traj[-1][:, [0, 2]].copy()
    return dict(rel=rel, t=t, traj=traj, land=land, centre=centre,
                release_mean=release_mean, wind=w2, d=d, c=c, T=T)


# ══════════════════════════════════════════════════════════════════════════════
#  Footprint & coverage metrics
# ══════════════════════════════════════════════════════════════════════════════

def footprint_stats(res):
    """Landing-footprint centroid + spread (East/North), with a closed-form
    cross-check in the wind-aligned frame."""
    rel, land = res['rel'], res['land']
    cen = land.mean(axis=0)
    std = land.std(axis=0)

    cf_cen = res['release_mean'] + res['wind'] * res['T']          # advection
    drift  = float(np.linalg.norm(res['wind']) * res['T'])
    # closed-form spread in along/cross frame: uniform half-width w → σ=w/√3,
    # gust adds σ = turb_i·|wind|·T in quadrature (isotropic).
    sg = rel.turb_i * rel.wind_ms * res['T']
    cf_along = np.hypot(rel.w_along / np.sqrt(3), sg)
    cf_cross = np.hypot(rel.w_cross / np.sqrt(3), sg)
    return dict(centroid=cen, std=std, cf_centroid=cf_cen, drift=drift,
                cf_along=float(cf_along), cf_cross=float(cf_cross))


def _fov_radius(alt, rel: Release):
    """Downward-sensor ground footprint radius at altitude, capped for honesty."""
    r = alt * np.tan(np.deg2rad(rel.fov_deg) / 2.0)
    return min(float(r), rel.r_det_max)


def fire_area_points(spacing=40.0):
    """Sample points on the fire interior (inside the swarm_core ellipse)."""
    A, B = sc.FIRE_A, sc.FIRE_B
    xs = np.arange(-A, A + 1e-6, spacing)
    zs = np.arange(-B, B + 1e-6, spacing)
    X, Z = np.meshgrid(xs, zs)
    P = np.column_stack([X.ravel(), Z.ravel()])
    inside = (P[:, 0] / A) ** 2 + (P[:, 1] / B) ** 2 <= 1.0
    return P[inside].astype('f4')


def _swept_mask(traj, rel, pts):
    """Bool mask over `pts`: covered by some unit's capped FOV footprint at ANY
    frame of the descent. traj is (n_t, M, 3); footprint radius uses the shared
    per-frame altitude traj[k,0,1]. Taking traj[:, :M] gives a nested sub-fleet,
    so coverage is monotone in M by construction."""
    ever = np.zeros(len(pts), dtype=bool)
    for k in range(len(traj)):
        r_f = _fov_radius(traj[k, 0, 1], rel)              # shared altitude at step k
        pxz = traj[k][:, [0, 2]]                            # (M,2)
        dmin = np.linalg.norm(pts[:, None, :] - pxz[None, :, :], axis=2).min(axis=1)
        ever |= (dmin < r_f)
    return ever


def coverage(res, area_pts=None, perim=None):
    """Coverage of the fire during descent.

    PRIMARY (area): fraction of fire-interior sample points that pass within
    some unit's downward-sensor footprint at any time during the descent — the
    recon value of the drifting fly-over. SECONDARY (perimeter): same metric on
    the perimeter ring (what a powered patrol targets), reported for contrast."""
    rel = res['rel']
    if area_pts is None:
        area_pts = fire_area_points()
    if perim is None:
        perim = sc.PERIM
    area_mask = _swept_mask(res['traj'], rel, area_pts)
    perim_mask = _swept_mask(res['traj'], rel, perim)
    return dict(area=float(area_mask.mean()), perim=float(perim_mask.mean()),
                area_pts=area_pts, area_mask=area_mask, perim_mask=perim_mask)


# ══════════════════════════════════════════════════════════════════════════════
#  Design sweep — the operational question
# ══════════════════════════════════════════════════════════════════════════════

def sweep(fleet, alts, base: Release = None, n_t=30):
    """Fire-AREA coverage over a grid of (fleet size × release altitude).

    Answers the operational design question: how many samaras, released from how
    high, to blanket what fraction of the fire? Returns (len(fleet), len(alts)).

    To isolate the fleet-size effect from resampling noise, each altitude
    simulates the MAX fleet ONCE and every smaller fleet is the first-M NESTED
    subset of that same cloud (traj[:, :M]) — so coverage is monotone in M by
    construction, not up to RNG luck. (Also faster: len(alts) sims, not the
    product.)"""
    base = base or Release()
    solve_descent(base)
    area_pts = fire_area_points()
    fleet = [int(m) for m in fleet]
    m_max = max(fleet)
    cov = np.zeros((len(fleet), len(alts)), dtype=float)
    for j, a in enumerate(alts):
        rel = Release(M=m_max, mass_kg=base.mass_kg, alt=float(a),
                      offset_up=base.offset_up, w_along=base.w_along,
                      w_cross=base.w_cross, turb_i=base.turb_i,
                      wind_ms=base.wind_ms, wind_from=base.wind_from,
                      spinup=base.spinup, fov_deg=base.fov_deg,
                      r_det_max=base.r_det_max, seed=base.seed, cfg=base.cfg)
        rel.Vd = base.Vd; rel.rpm = base.rpm
        traj = simulate(rel, n_t=n_t)['traj']
        for i, m in enumerate(fleet):
            cov[i, j] = float(_swept_mask(traj[:, :m], rel, area_pts).mean())
    return cov


# ══════════════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    """Self-checks: V_d import parity, closed-form footprint parity (wind-aligned
    frame), null-wind symmetry, monotone drift-with-altitude, and a coverage
    sanity bound."""
    print(f"\n{'─'*64}\n  dispersal.py — validation\n{'─'*64}")
    ok = True

    # 1) V_d parity — the descent rate really is the Phase-1 equilibrium.
    rel = Release(); Vd = solve_descent(rel)
    Vd_ref, _ = sb.solve(sb.SAMARA, 0.075)
    p1 = abs(Vd - Vd_ref) < 1e-9; ok &= p1
    print(f"  [{'PASS' if p1 else 'FAIL'}] V_d import parity: {Vd:.4f} == {Vd_ref:.4f} m/s (Phase-1)")

    # 2) closed-form footprint parity — large fleet so gust sampling (∝1/√M)
    #    doesn't dominate. Check centroid + along/cross spread in the wind frame.
    relN = Release(M=6000); solve_descent(relN)
    res = simulate(relN, n_t=80); fs = footprint_stats(res)
    land, d, c = res['land'], res['d'], res['c']
    A = land @ d; C = land @ c                             # project to wind frame
    cf_A = res['release_mean'] @ d + np.linalg.norm(res['wind']) * res['T']
    cf_C = res['release_mean'] @ c
    dc = np.hypot(A.mean() - cf_A, C.mean() - cf_C)
    dsa = abs(A.std() - fs['cf_along']) / fs['cf_along']
    dsc = abs(C.std() - fs['cf_cross']) / fs['cf_cross']
    p2 = (dc < 5.0) and (dsa < 0.08) and (dsc < 0.08); ok &= p2
    print(f"  [{'PASS' if p2 else 'FAIL'}] footprint vs closed form: "
          f"Δcentroid={dc:.1f} m, Δσ_along={dsa*100:.0f}%, Δσ_cross={dsc*100:.0f}%")

    # 3) null-wind: no drift, no gust spread → land exactly on release centroid.
    relz = Release(wind_ms=0.0); solve_descent(relz)
    rz = simulate(relz, n_t=40); fz = footprint_stats(rz)
    p3 = (fz['drift'] < 1e-6) and (np.linalg.norm(fz['centroid'] - rz['release_mean']) < 1e-3)
    ok &= p3
    print(f"  [{'PASS' if p3 else 'FAIL'}] null-wind: drift={fz['drift']:.2e} m, "
          f"landed centroid == release centroid")

    # 4) drift grows with release altitude (longer fall → more advection).
    d_lo = footprint_stats(simulate(Release(alt=200.0), n_t=30))['drift']
    d_hi = footprint_stats(simulate(Release(alt=600.0), n_t=30))['drift']
    p4 = d_hi > d_lo + 1.0; ok &= p4
    print(f"  [{'PASS' if p4 else 'FAIL'}] drift monotone in altitude: "
          f"{d_lo:.0f} m @200 m < {d_hi:.0f} m @600 m")

    # 5) coverage sanity: a fraction in [0,1], and more fleet ⇒ ≥ coverage —
    #    tested on NESTED subsets of ONE cloud (the way sweep() does it), so it
    #    is a true invariant, not up to RNG luck (a subset can only cover less).
    relm = Release(M=300); solve_descent(relm)
    trajm = simulate(relm, n_t=30)['traj']; ap = fire_area_points()
    c_lo = float(_swept_mask(trajm[:, :40],  relm, ap).mean())
    c_hi = float(_swept_mask(trajm[:, :300], relm, ap).mean())
    p5 = (0.0 <= c_lo <= 1.0) and (0.0 <= c_hi <= 1.0) and (c_hi >= c_lo - 1e-12); ok &= p5
    print(f"  [{'PASS' if p5 else 'FAIL'}] area coverage monotone in fleet (nested): "
          f"{c_lo*100:.0f}% @M=40 ≤ {c_hi*100:.0f}% @M=300")

    print(f"{'─'*64}\n  {'ALL PASS ✓' if ok else 'FAILURES ✗'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Report + figure
# ══════════════════════════════════════════════════════════════════════════════

def report(rel: Release = None):
    rel = rel or Release()
    solve_descent(rel)
    res = simulate(rel, n_t=80)
    fs  = footprint_stats(res)
    cov = coverage(res)
    su  = SPINUP[rel.spinup]

    print(f"\n{'═'*64}")
    print(f"  DISPERSAL — {rel.M} samara-UAVs, {rel.mass_kg*1000:.0f} g each")
    print(f"{'═'*64}")
    print(f"  Descent rate V_d : {rel.Vd:.2f} m/s  ({rel.rpm:.0f} RPM)   [Validated — Phase-1]")
    print(f"  Release          : {rel.alt:.0f} m AGL, {rel.offset_up:.0f} m upwind, "
          f"patch {2*rel.w_along:.0f}(along)×{2*rel.w_cross:.0f}(cross) m")
    print(f"  Wind             : {rel.wind_ms:.1f} m/s from {rel.wind_from:.0f}°")
    print(f"  Spin-up ({rel.spinup:>7}): {su['alt']:.0f} m / {su['t']:.0f} s to terminal   [Indicative]")
    print(f"  Descent time     : {res['T']:.0f} s")
    print(f"  ── landing footprint (Indicative) ──")
    print(f"    downwind drift : {fs['drift']:.0f} m")
    print(f"    centroid       : ({fs['centroid'][0]:+.0f}, {fs['centroid'][1]:+.0f}) m   "
          f"[closed form ({fs['cf_centroid'][0]:+.0f}, {fs['cf_centroid'][1]:+.0f})]")
    print(f"    spread (1σ)    : {fs['std'][0]:.0f}(E) × {fs['std'][1]:.0f}(N) m")
    print(f"  ── fire coverage in one descent (Indicative) ──")
    print(f"    sensor         : {rel.fov_deg:.0f}° FOV, footprint capped at {rel.r_det_max:.0f} m "
          f"(coverage scales with this)")
    print(f"    AREA  covered  : {cov['area']*100:.0f}%   (fire interior, {len(cov['area_pts'])} sample pts)")
    print(f"    perimeter      : {cov['perim']*100:.0f}%   (contrast: what a powered patrol rings)")
    print(f"    (fire ellipse {sc.FIRE_A:.0f}×{sc.FIRE_B:.0f} m — same target as the powered patrol)")
    return res, fs, cov


def figure(out=None):
    import matplotlib
    matplotlib.use('Agg')                            # switch_backend; works before any figure
    assert matplotlib.get_backend().lower() == 'agg', "could not bind Agg for the static figure"
    import matplotlib.pyplot as plt
    if out is None:      # anchor to this module's dir so it lands in 03_swarm/output/
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'dispersal.png')

    rel = Release()
    res, fs, cov = report(rel)
    t, traj, land = res['t'], res['traj'], res['land']
    perim = sc.PERIM; ap = cov['area_pts']; am = cov['area_mask']

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 6.2), facecolor='#0d1117')

    # (1) plan view — fire area coverage, release patch, landing
    ax = fig.add_axes([0.045, 0.11, 0.42, 0.82]); ax.set_facecolor('#0d1117')
    ax.scatter(ap[~am, 0], ap[~am, 1], s=7, color='#30363d', label='fire area (uncovered)')
    ax.scatter(ap[am, 0], ap[am, 1], s=8, color='#39d353', label='fire area (covered)')
    ax.plot(np.r_[perim[:, 0], perim[0, 0]], np.r_[perim[:, 1], perim[0, 1]],
            color='#f0883e', lw=2.0, label='fire perimeter')
    ax.scatter(traj[0, :, 0], traj[0, :, 2], s=6, color='#8b949e', alpha=0.6, label='release')
    ax.scatter(land[:, 0], land[:, 1], s=10, color='#4fc3f7', label='landing')
    ax.annotate('', xy=res['wind']*40 + res['centre'], xytext=res['centre'],
                arrowprops=dict(color='#58a6ff', width=1.5, headwidth=7))
    ax.set_aspect('equal'); ax.set_xlabel('East [m]', color='#8b949e', fontsize=9)
    ax.set_ylabel('North [m]', color='#8b949e', fontsize=9)
    ax.set_title(f'Dispersal plan view — area coverage {cov["area"]*100:.0f}% '
                 f'(Indicative, footprint ≤{rel.r_det_max:.0f} m)',
                 color='#c9d1d9', fontsize=11)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.legend(loc='upper left', fontsize=6.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, color='#21262d', lw=0.4)

    # (2) side view — descent while drifting east
    ax2 = fig.add_axes([0.53, 0.60, 0.44, 0.33]); ax2.set_facecolor('#0d1117')
    step = max(rel.M // 40, 1)
    for i in range(0, rel.M, step):
        ax2.plot(traj[:, i, 0], traj[:, i, 1], color='#4fc3f7', alpha=0.35, lw=0.7)
    ax2.axhline(0, color='#2ea043', lw=1.2)
    ax2.set_xlabel('East [m]', color='#8b949e', fontsize=9)
    ax2.set_ylabel('Alt [m]', color='#8b949e', fontsize=9)
    ax2.set_title(f'Autorotation descent (V_d={rel.Vd:.2f} m/s, {res["T"]:.0f} s)  [Validated V_d]',
                  color='#c9d1d9', fontsize=10)
    ax2.tick_params(colors='#8b949e', labelsize=8); ax2.grid(True, color='#21262d', lw=0.4)

    # (3) design sweep — area coverage vs (fleet size × altitude)
    ax3 = fig.add_axes([0.53, 0.11, 0.44, 0.36]); ax3.set_facecolor('#0d1117')
    fleet = np.array([25, 50, 100, 150, 250, 400]); alts = np.linspace(200, 800, 9)
    cmap = sweep(fleet, alts, base=rel, n_t=26)
    im = ax3.imshow(cmap*100, origin='lower', aspect='auto', cmap='viridis',
                    extent=[alts[0], alts[-1], 0, len(fleet)-1], vmin=0, vmax=100)
    ax3.set_yticks(range(len(fleet))); ax3.set_yticklabels(fleet)
    ax3.set_xlabel('release altitude [m]', color='#8b949e', fontsize=9)
    ax3.set_ylabel('fleet size M', color='#8b949e', fontsize=9)
    ax3.set_title('Fire-AREA coverage [%]  (Indicative)', color='#c9d1d9', fontsize=10)
    ax3.tick_params(colors='#8b949e', labelsize=8)
    cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.02); cb.ax.tick_params(colors='#8b949e', labelsize=7)

    fig.text(0.53, 0.985, 'Project Sycamore — passive-autorotation dispersal',
             color='#c9d1d9', fontsize=12, va='top')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, facecolor='#0d1117')
    plt.close(fig)
    print(f"\nSaved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Live animation (interactive window) + headless gif/mp4 export
#
#  Backend discipline: importing this module must NOT force a matplotlib backend
#  (so --validate stays matplotlib-free). Each entry point selects its own:
#    * figure()  headless static PNG  → matplotlib.use('Agg')      (local)
#    * animate(out=...) headless gif/mp4 → matplotlib.use('Agg')   (local)
#    * animate()  live window → respect $MPLBACKEND, else 'macosx'  (local)
#  matplotlib.use() must run BEFORE the first pyplot import in the process; since
#  every import here is local to a function, whichever entry point runs first wins.
# ══════════════════════════════════════════════════════════════════════════════

def frame_coverage(res, area_pts=None, perim=None):
    """Per-frame CUMULATIVE coverage for the fill-in animation.

    coverage() gives only the end-of-descent 'ever covered' mask; the animation
    needs the covered set (and coverage fraction) AS OF each frame. Returns
    cumulative bool masks (n_t, N) and coverage(t) curves for both the fire area
    and the perimeter, reusing simulate()'s traj and the capped FOV footprint."""
    rel = res['rel']
    if area_pts is None:
        area_pts = fire_area_points()
    if perim is None:
        perim = sc.PERIM
    t, traj = res['t'], res['traj']

    def _cum(pts):
        step = np.zeros((len(t), len(pts)), dtype=bool)
        for k in range(len(t)):
            r_f = _fov_radius(traj[k, 0, 1], rel)
            pxz = traj[k][:, [0, 2]]
            dmin = np.linalg.norm(pts[:, None, :] - pxz[None, :, :], axis=2).min(axis=1)
            step[k] = dmin < r_f
        cum = np.logical_or.accumulate(step, axis=0)
        return cum, cum.mean(axis=1)

    a_cum, a_curve = _cum(area_pts)
    p_cum, p_curve = _cum(perim)
    return dict(area_pts=area_pts, perim=perim, a_cum=a_cum, a_curve=a_curve,
                p_cum=p_cum, p_curve=p_curve)


def build_animation(res, fc):
    """(fig, update, n_frames) for the live descent. 3D cloud descending over the
    fire + a 2-D plan inset whose fire-area fills in as it is covered + a live
    coverage-vs-time chart. Mirrors the boids_swarm_3d / fire_swarm contract."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401 (registers 3d)

    rel = res['rel']
    t, traj = res['t'], res['traj']
    n_t = len(t)
    ap, per = fc['area_pts'], fc['perim']
    A, B = sc.FIRE_A, sc.FIRE_B

    # fixed view bounds (fire + full trajectory)
    xs_all, zs_all = traj[:, :, 0], traj[:, :, 2]
    Xlim = [min(xs_all.min(), -A) * 1.05, max(xs_all.max(), A) * 1.05]
    Zlim = [min(zs_all.min(), -B) * 1.1,  max(zs_all.max(), B) * 1.1]
    ang = np.linspace(0, 2 * np.pi, 160); ex, ez = A * np.cos(ang), B * np.sin(ang)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    try:
        fig.canvas.manager.set_window_title(f"Project Sycamore — dispersal (N={rel.M})")
    except Exception:
        pass

    # ── 3D descent (left) ──
    ax3 = fig.add_axes([0.02, 0.06, 0.60, 0.90], projection='3d')
    ax3.set_facecolor('#0d1117')
    ax3.set_xlim(Xlim); ax3.set_ylim(0, rel.alt * 1.05); ax3.set_zlim(Zlim)
    ax3.set_box_aspect((3.0, 1.4, 2.0))
    ax3.set_xlabel('East [m]', color='#8b949e', fontsize=8, labelpad=8)
    ax3.set_ylabel('Alt [m]', color='#8b949e', fontsize=8, labelpad=4)
    ax3.set_zlabel('North [m]', color='#8b949e', fontsize=8, labelpad=8)
    ax3.tick_params(colors='#8b949e', labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#21262d')
    ax3.grid(True, color='#21262d', linewidth=0.4)
    ax3.view_init(elev=26, azim=-72)
    ax3.plot(ex, np.zeros_like(ex), ez, color='#f0883e', lw=2.0)   # fire outline (ground)
    cloud = ax3.scatter(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2],
                        s=14, c=traj[0, :, 1], cmap='cool', vmin=0, vmax=rel.alt,
                        depthshade=True, edgecolors='none')

    # ── plan inset (top-right): fire-area fill-in ──
    ax2 = fig.add_axes([0.65, 0.55, 0.33, 0.40]); ax2.set_facecolor('#0d1117')
    ax2.scatter(ap[:, 0], ap[:, 1], s=8, color='#30363d')      # uncovered (static backdrop)
    covsc = ax2.scatter([], [], s=9, color='#39d353')
    ax2.plot(np.r_[per[:, 0], per[0, 0]], np.r_[per[:, 1], per[0, 1]], color='#f0883e', lw=1.6)
    dots2, = ax2.plot(traj[0, :, 0], traj[0, :, 2], '.', color='#4fc3f7', ms=3)
    ax2.annotate('', xy=res['wind'] * 40 + res['centre'], xytext=res['centre'],
                 arrowprops=dict(color='#58a6ff', width=1.2, headwidth=6))
    ax2.set_aspect('equal'); ax2.set_xlim(Xlim); ax2.set_ylim(Zlim)
    ax2.set_title('fire-area coverage (plan) — Indicative', color='#8b949e', fontsize=9)
    ax2.tick_params(colors='#8b949e', labelsize=7)

    # ── coverage-vs-time chart (bottom-right) ──
    axc = fig.add_axes([0.65, 0.09, 0.33, 0.36]); axc.set_facecolor('#0d1117')
    axc.set_xlim(0, t[-1]); axc.set_ylim(0, 100)
    axc.plot(t, fc['a_curve'] * 100, color='#21262d', lw=0.7)
    a_line, = axc.plot([], [], color='#39d353', lw=1.6, label='area %')
    p_line, = axc.plot([], [], color='#f0883e', lw=1.2, ls='--', label='perimeter % (contrast)')
    a_dot,  = axc.plot([], [], 'o', color='#f0e68c', ms=4)
    axc.set_xlabel('t [s]', color='#8b949e', fontsize=8)
    axc.set_ylabel('coverage %  (Indicative)', color='#8b949e', fontsize=8)
    axc.tick_params(colors='#8b949e', labelsize=7); axc.grid(True, color='#21262d', lw=0.4)
    axc.legend(loc='lower right', fontsize=7, facecolor='#161b22',
               edgecolor='#30363d', labelcolor='#c9d1d9')

    time_txt = fig.text(0.32, 0.975, '', ha='center', va='top', color='#c9d1d9',
                        fontsize=10, fontfamily='monospace')
    fig.text(0.32, 0.945,
             f'V_d={rel.Vd:.2f} m/s [Validated]  ·  release {rel.alt:.0f} m  ·  wind {rel.wind_ms:.1f} m/s  ·  '
             f'sensor {rel.fov_deg:.0f}° FOV, footprint ≤{rel.r_det_max:.0f} m — coverage [Indicative], scales with this',
             ha='center', va='top', color='#8b949e', fontsize=8)

    def update(frame):
        k = min(frame, n_t - 1)
        p = traj[k]
        cloud._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        cloud.set_array(p[:, 1])
        am = fc['a_cum'][k]
        covsc.set_offsets(ap[am] if am.any() else np.empty((0, 2)))
        dots2.set_data(p[:, 0], p[:, 2])
        a_line.set_data(t[:k+1], fc['a_curve'][:k+1] * 100)
        p_line.set_data(t[:k+1], fc['p_curve'][:k+1] * 100)
        a_dot.set_data([t[k]], [fc['a_curve'][k] * 100])
        time_txt.set_text(f"t = {t[k]:5.0f} s   alt = {p[0,1]:4.0f} m   "
                          f"area = {fc['a_curve'][k]*100:4.0f}%   perim = {fc['p_curve'][k]*100:4.0f}%   "
                          f"N = {rel.M}")
        return [cloud, covsc, dots2, a_line, p_line, a_dot, time_txt]

    return fig, update, n_t


def animate(out=None, rel: Release = None, n_t=140, fps=30, verbose=True):
    """Drive the live descent animation.

    out=None  → open a live interactive window (respects $MPLBACKEND, else macosx).
    out=path  → headless: render to .gif (pillow) or .mp4 (ffmpeg). No window.
    Either way the physics is the same precomputed dispersal; only playback differs."""
    import matplotlib
    headless = out is not None
    if headless:
        matplotlib.use('Agg')                        # switch_backend; works before any figure
        assert matplotlib.get_backend().lower() == 'agg', "could not bind Agg for headless export"
    elif not os.environ.get('MPLBACKEND'):
        matplotlib.use('macosx')                     # native window on Apple Silicon
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    rel = rel or Release()
    solve_descent(rel)
    if verbose:
        print(f"Precomputing dispersal — N={rel.M}, {n_t} frames …")
    res = simulate(rel, n_t=n_t)
    fc  = frame_coverage(res)
    fig, update, n_frames = build_animation(res, fc)
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 / fps, blit=False)   # blit off: 3D axes
    if headless:
        d = os.path.dirname(out)                      # own the dir creation (guard bare filename)
        if d:
            os.makedirs(d, exist_ok=True)
        writer = 'pillow' if out.lower().endswith('.gif') else 'ffmpeg'
        ani.save(out, writer=writer, fps=fps, dpi=110,
                 savefig_kwargs=dict(facecolor='#0d1117'))
        plt.close(fig)
        print(f"Saved {out}  ({n_frames} frames, {writer})")
        return out
    globals()['_ani'] = ani                          # keep a ref so GC can't freeze it
    plt.show()
    return ani


def _default_export(ext):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'dispersal.{ext}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--validate' in args:
        sys.exit(0 if validate() else 1)
    elif '--live' in args:
        animate()                                    # interactive window
    elif '--gif' in args:
        i = args.index('--gif')
        out = args[i+1] if i+1 < len(args) and not args[i+1].startswith('-') else _default_export('gif')
        animate(out=out)                             # animate() owns dir creation (guards bare names)
    elif '--mp4' in args:
        i = args.index('--mp4')
        out = args[i+1] if i+1 < len(args) and not args[i+1].startswith('-') else _default_export('mp4')
        animate(out=out)
    else:
        figure()                                     # headless static summary PNG
