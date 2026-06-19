"""
Modern Boids swarm — Project Sycamore wildfire UAV patrol.

Each drone i has position x_i ∈ ℝ² and velocity v_i ∈ ℝ².
All state is local: a drone only observes neighbors within R_PERC.

Local rules (Reynolds 1987 + modern additions):
  1. Separation   f_sep  = −∑_{j: d<R_sep} (x_j−x_i) / d²
  2. Alignment    f_ali  =  (1/|N|) ∑_{j∈N}  v̂_j         (mean heading)
  3. Cohesion     f_coh  =  x̄_N − x_i                    (toward centroid)
  4. Fire track   f_fire =  (p*−x_i) / ||p*−x_i||        (nearest perimeter point)
  5. Speed reg    f_spd  =  v̂_i · (v_cruise − ||v_i||)   (match cruise speed)

"Modern" additions over Reynolds:
  — Euler-Maruyama noise (Vicsek-style heading perturbation each step)
  — Interior avoidance: triple fire weight if inside the fire polygon
  — Isolated-agent recovery: strong pull toward nearest if no neighbors

Update:
  a_i      = w₁f_sep + w₂f_ali + w₃f_coh + w₄f_fire + w₅f_spd
  v_i(t+1) = clip(v_i(t) + Δt·a_i + noise, v_max)
  x_i(t+1) = x_i(t) + Δt·v_i(t+1)

All operations are fully vectorised (numpy broadcasting). O(N²) per step
but runs in < 1 ms for N ≤ 100.

Run: ~/ds/bin/python 03_swarm/boids_swarm.py
"""

import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

# ── Samara silhouette from OBJ ─────────────────────────────────────────────
def _load_silhouette(path='/Users/home/Downloads/sycamore_active.obj'):
    """
    Project the 3-D OBJ onto the XZ plane (top-down view), take the 2-D
    convex hull, centre it and align the major axis with +x (nose forward).
    Returns normalised hull points shape (M, 2) with unit chord length.
    """
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[3])])   # x, z
    v   = np.array(verts)
    hull = ConvexHull(v)
    hp   = v[hull.vertices].copy()
    hp  -= hp.mean(axis=0)                                  # centre
    # PCA: align major axis → +x
    cov   = np.cov(hp.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    ang   = np.arctan2(major[1], major[0])
    c, s  = np.cos(-ang), np.sin(-ang)
    hp    = (np.array([[c, -s], [s, c]]) @ hp.T).T
    if hp[:, 0].max() < abs(hp[:, 0].min()):
        hp[:, 0] *= -1                                      # nose in +x
    hp /= (hp[:, 0].max() - hp[:, 0].min())                # unit chord
    return hp

SILHOUETTE  = _load_silhouette()    # (M, 2), nose at +x, chord = 1
DRONE_SCALE = 130.0                 # visual size [m] in simulation space

# ── Environment ────────────────────────────────────────────────────────────
FIRE_A, FIRE_B = 800.0, 400.0     # fire ellipse semi-axes [m]
N_PERIM        = 400               # perimeter sample points for nearest-pt lookup

φ_p   = np.linspace(0, 2*np.pi, N_PERIM, endpoint=False)
PERIM = np.column_stack([FIRE_A * np.cos(φ_p), FIRE_B * np.sin(φ_p)])
FIRE_POLY = Polygon(PERIM)         # shapely polygon for interior test

# ── Fleet parameters ───────────────────────────────────────────────────────
N_DRONES    = 20
# Perimeter ≈ 4 000 m → nominal drone spacing ≈ 200 m.
# Arc-index separation acts only along the tangent, spreading drones
# evenly around the ellipse regardless of 2-D curvature.
# Steady-state coverage: ~66% mean (57–74% range).
# Gap vs. optimal static (79%) is the price of decentralized local rules.
R_PERC      = 250.0    # perception radius [m]
R_SEP       = 100.0    # 2-D separation [m]  (emergency close-approach only)
R_ARC_IDX   = 40       # arc-index separation radius  (≈ 2× nominal spacing)
V_MAX       = 12.0     # speed cap [m/s]
V_CRUISE    = 6.5      # target cruise speed [m/s]

W_SEP, W_ALI, W_COH  = 0.5, 0.1, 0.0  # minimal social forces
W_FIRE                = 4.0              # approach fire perimeter
W_TAN                 = 1.5              # CCW tangential patrol
W_ARC                 = 5.0              # arc-index spreading force
W_SPD                 = 0.5
SIGMA_NOISE           = 0.05             # heading noise RMS [rad/step]
R_PATROL              = 180.0            # tangent-ramp distance [m]

DT   = 0.5    # simulation timestep [s]
T_MAX = 650   # ≈ 1 full orbit at V_CRUISE (4000m / 6.5 m/s ≈ 615 s)

RNG = np.random.default_rng(42)

# ── Initialise agents ──────────────────────────────────────────────────────
def init_agents(N):
    """Distribute drones uniformly around the perimeter, just outside."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += RNG.uniform(0, 2 * np.pi / N, N)
    pos = np.column_stack([
        1.25 * FIRE_A * np.cos(angles),
        1.25 * FIRE_B * np.sin(angles),
    ])
    # All drones start CCW; arc-separation spreads them, not direction alternation
    tangents = np.column_stack([-np.sin(angles), np.cos(angles)])
    vel = tangents * V_CRUISE * RNG.uniform(0.9, 1.1, (N, 1))
    return pos, vel

# ── Vectorised boids step ──────────────────────────────────────────────────
def boids_step(pos, vel):
    """
    Advance one timestep. Fully vectorised; no Python loops over drones.

    Broadcasting shapes used:
      disp[i,j] = pos[j] - pos[i]        (N, N, 2)
      dist[i,j] = ||pos[j] - pos[i]||    (N, N)
    """
    N = len(pos)

    # Pairwise displacement and distance
    disp = pos[None, :, :] - pos[:, None, :]          # (N, N, 2)
    dist = np.linalg.norm(disp, axis=2)               # (N, N)
    np.fill_diagonal(dist, np.inf)                    # exclude self

    perc_mask = dist < R_PERC                         # (N, N) bool
    sep_mask  = (dist < R_SEP) & (dist > 1e-6)
    n_neigh   = perc_mask.sum(axis=1).clip(min=1)     # (N,) avoid /0

    # 1. Separation — inverse-square repulsion
    sep_w = np.where(sep_mask, 1.0 / dist**2, 0.0)   # (N, N)
    f_sep = -np.einsum('ij,ijk->ik', sep_w, disp)    # (N, 2)

    # 2. Alignment — mean heading of neighbours
    speed = np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-6)
    v_hat = vel / speed                               # (N, 2) unit vel
    f_ali = np.einsum('ij,jk->ik', perc_mask.astype(float), v_hat) / n_neigh[:, None]

    # 3. Cohesion — toward centroid of neighbours
    centroid = np.einsum('ij,jk->ik', perc_mask.astype(float), pos) / n_neigh[:, None]
    coh_raw  = np.where(n_neigh[:, None] > 0, centroid - pos, 0.0)
    coh_norm = np.linalg.norm(coh_raw, axis=1, keepdims=True).clip(min=1e-6)
    f_coh    = coh_raw / coh_norm

    # 4. Fire perimeter — approach + tangential patrol
    d_perim  = np.linalg.norm(PERIM[None, :, :] - pos[:, None, :], axis=2)  # (N, Np)
    k_near   = np.argmin(d_perim, axis=1)            # (N,) nearest perim index
    p_star   = PERIM[k_near]                         # (N, 2)
    to_perim = p_star - pos
    d_to_p   = np.linalg.norm(to_perim, axis=1, keepdims=True).clip(min=1e-6)
    f_fire   = to_perim / d_to_p                     # normal: approach perimeter

    # Interior avoidance: triple weight if inside fire polygon
    inside = np.array([FIRE_POLY.contains(Point(p)) for p in pos])
    fire_w = np.where(inside[:, None], 3.0, 1.0)
    f_fire = f_fire * fire_w

    # Tangential patrol: when near perimeter, follow CCW ellipse tangent
    k_fwd  = (k_near + 1) % N_PERIM
    k_bwd  = (k_near - 1) % N_PERIM
    tan_raw = PERIM[k_fwd] - PERIM[k_bwd]           # (N, 2) CCW tangent
    tan_n   = np.linalg.norm(tan_raw, axis=1, keepdims=True).clip(min=1e-6)
    tangent  = tan_raw / tan_n                       # unit CCW tangent
    tan_w = np.clip(1.0 - d_to_p.squeeze() / R_PATROL, 0.0, 1.0)[:, None]
    f_tan = tangent * tan_w

    # Arc-index separation: spread drones along perimeter arc.
    # dk[i,j] = perimeter-index of j minus index of i (signed, wrapping).
    # Positive dk → j is ahead (CCW) → push i backward (−tangent).
    # Inverse-linear in |dk|; active within R_ARC_IDX index units.
    dk = k_near[None, :].astype(int) - k_near[:, None].astype(int)  # (N, N)
    dk = np.where(dk >  N_PERIM // 2, dk - N_PERIM, dk)
    dk = np.where(dk < -N_PERIM // 2, dk + N_PERIM, dk)
    adk = np.abs(dk)
    arc_mask = (adk > 0) & (adk < R_ARC_IDX)
    arc_w    = np.where(arc_mask, -np.sign(dk).astype(float) / adk.clip(min=1), 0.)
    # Project scalar arc-force onto each drone's own tangent direction
    arc_scalar = arc_w.sum(axis=1)                  # (N,) signed scalar
    f_arc = arc_scalar[:, None] * tangent            # (N, 2)

    # 5. Speed regulation — steer toward cruise speed
    speed_scalar = speed.squeeze()
    f_spd = v_hat * (V_CRUISE - speed_scalar)[:, None]

    # Isolated-agent recovery (no neighbours → sprint toward nearest)
    isolated = (n_neigh == 1)   # n_neigh was clipped from 0 to 1
    if isolated.any():
        np.fill_diagonal(dist, np.inf)
        nearest_j = np.argmin(dist, axis=1)
        to_near   = pos[nearest_j] - pos
        tn_norm   = np.linalg.norm(to_near, axis=1, keepdims=True).clip(min=1e-6)
        f_sep[isolated] = 4.0 * to_near[isolated] / tn_norm[isolated]

    # Combined acceleration
    accel = (W_SEP  * f_sep
           + W_ALI  * f_ali
           + W_COH  * f_coh
           + W_FIRE * f_fire
           + W_TAN  * f_tan
           + W_ARC  * f_arc
           + W_SPD  * f_spd)

    # Vicsek noise — random heading rotation
    noise_angle = RNG.normal(0, SIGMA_NOISE, N)
    cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
    vel_rot = np.column_stack([
        vel[:, 0] * cos_n - vel[:, 1] * sin_n,
        vel[:, 0] * sin_n + vel[:, 1] * cos_n,
    ])

    vel_new = vel + DT * accel + (vel_rot - vel)

    # Speed limits
    spd_new = np.linalg.norm(vel_new, axis=1, keepdims=True).clip(min=1e-6)
    vel_new = np.where(spd_new > V_MAX, vel_new / spd_new * V_MAX, vel_new)
    vel_new = np.where(spd_new < 1.0,  vel_new / spd_new * 1.0,   vel_new)

    return pos + DT * vel_new, vel_new

# ── Pre-simulate trajectory ────────────────────────────────────────────────
print("Simulating boids…")
pos, vel = init_agents(N_DRONES)
N_STEPS   = int(T_MAX / DT)
traj_pos  = np.zeros((N_STEPS, N_DRONES, 2))
traj_vel  = np.zeros((N_STEPS, N_DRONES, 2))

for t in range(N_STEPS):
    traj_pos[t] = pos
    traj_vel[t] = vel
    pos, vel = boids_step(pos, vel)
    if t % 100 == 0:
        cov_now = coverage_tmp = np.mean(
            np.min(np.linalg.norm(PERIM[:, None, :] - traj_pos[t][None, :, :], axis=2), axis=1)
            < 80.0
        )
        print(f"  t={t*DT:.0f}s  coverage={cov_now*100:.1f}%", end="\r")

print(f"\nDone. {N_STEPS} steps stored.\n")

# ── Compute coverage over time for the side panel ─────────────────────────
SAMPLE_EVERY = 5
times = np.arange(0, N_STEPS, SAMPLE_EVERY) * DT
covs  = []
for t in range(0, N_STEPS, SAMPLE_EVERY):
    d = np.min(np.linalg.norm(PERIM[:, None, :] - traj_pos[t][None, :, :], axis=2), axis=1)
    covs.append(np.mean(d < 80.0))
covs = np.array(covs) * 100

# ── Live animation ─────────────────────────────────────────────────────────
DARK, PANEL, TC = '#0d1117', '#161b22', '#c9d1d9'
fig, (ax_main, ax_cov) = plt.subplots(
    1, 2, figsize=(14, 7),
    gridspec_kw={'width_ratios': [2.5, 1]},
    facecolor=DARK
)

for ax in (ax_main, ax_cov):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TC, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.6)

lim = max(FIRE_A, FIRE_B) * 1.7
ax_main.set_aspect('equal')
ax_main.set_xlim(-lim, lim); ax_main.set_ylim(-lim * 0.65, lim * 0.65)
ax_main.set_xlabel('East [m]', color=TC, fontsize=9)
ax_main.set_ylabel('North [m]', color=TC, fontsize=9)
ax_main.set_title('Project Sycamore — Boids Swarm', color='white', fontsize=10)

ax_main.plot(PERIM[:, 0], PERIM[:, 1], color='#ef5350', lw=1.5, alpha=0.7, zorder=1)
ax_main.fill(PERIM[:, 0], PERIM[:, 1], color='#ef5350', alpha=0.08, zorder=0)

ax_cov.set_xlabel('Time [s]', color=TC, fontsize=9)
ax_cov.set_ylabel('Perimeter coverage [%]', color=TC, fontsize=9)
ax_cov.set_title('Coverage vs Time', color=TC, fontsize=9)
ax_cov.set_xlim(0, T_MAX); ax_cov.set_ylim(0, 102)
ax_cov.axhline(79, ls='--', color='#f0883e', lw=1.0, alpha=0.7,
               label='optimal static (79%)')

TRACE_LEN = 40
traces    = [ax_main.plot([], [], color='#4fc3f7', lw=0.6, alpha=0.30)[0]
             for _ in range(N_DRONES)]

# One Polygon patch per drone — vertices updated each frame to match heading
drone_patches = []
for _i in range(N_DRONES):
    poly = MplPolygon(SILHOUETTE * DRONE_SCALE,
                      closed=True,
                      facecolor='#4fc3f7', edgecolor='#c9d1d9',
                      linewidth=0.5, alpha=0.82, zorder=5)
    ax_main.add_patch(poly)
    drone_patches.append(poly)
comm_lc   = mc.LineCollection([], color='#58a6ff', alpha=0.20, lw=0.7, zorder=2)
ax_main.add_collection(comm_lc)
time_txt  = ax_main.text(0.02, 0.97, '', transform=ax_main.transAxes,
                          color='#f0c040', fontsize=9, va='top', fontfamily='monospace')
cov_line, = ax_cov.plot([], [], color='#4fc3f7', lw=1.8)
cov_dot,  = ax_cov.plot([], [], 'o', color='#4fc3f7', ms=5)

SPEED = 3   # display frames per sim step (fast-forward)

def _patch_verts(pos2d, vel2d):
    """Rotate SILHOUETTE to face vel2d, scale, translate to pos2d."""
    heading  = np.arctan2(vel2d[1], vel2d[0])
    c, s     = np.cos(heading), np.sin(heading)
    rot      = np.array([[c, -s], [s, c]])
    return (rot @ (SILHOUETTE * DRONE_SCALE).T).T + pos2d

def update(frame):
    t     = min(frame * SPEED, N_STEPS - 1)
    pos_t = traj_pos[t]
    vel_t = traj_vel[t]

    # Update each drone's silhouette polygon
    speeds = np.linalg.norm(vel_t, axis=1)
    for i in range(N_DRONES):
        drone_patches[i].set_xy(_patch_verts(pos_t[i], vel_t[i]))
        # Colour by normalised speed: blue (slow) → orange (fast)
        frac = np.clip((speeds[i] - 1.0) / (V_MAX - 1.0), 0, 1)
        r = 0x4f/255 + frac * (0xff/255 - 0x4f/255)
        g = 0xc3/255 + frac * (0xa7/255 - 0xc3/255)
        b = 0xf7/255 + frac * (0x26/255 - 0xf7/255)
        drone_patches[i].set_facecolor((r, g, b))

    t0 = max(0, t - TRACE_LEN)
    for i in range(N_DRONES):
        traces[i].set_data(traj_pos[t0:t+1, i, 0], traj_pos[t0:t+1, i, 1])

    # Comm links within R_PERC
    segs  = []
    d_mat = np.linalg.norm(pos_t[:, None, :] - pos_t[None, :, :], axis=2)
    ii, jj = np.where(
        (d_mat < R_PERC) & (np.triu(np.ones_like(d_mat), k=1).astype(bool))
    )
    for a, b in zip(ii, jj):
        segs.append([pos_t[a], pos_t[b]])
    comm_lc.set_segments(segs)

    elapsed = t * DT
    time_txt.set_text(f't = {elapsed:.0f} s\nN = {N_DRONES}')

    t_cov_idx = min(int(t / SAMPLE_EVERY), len(times) - 1)
    cov_line.set_data(times[:t_cov_idx], covs[:t_cov_idx])
    if t_cov_idx > 0:
        cov_dot.set_data([times[t_cov_idx-1]], [covs[t_cov_idx-1]])

    return [*drone_patches, *traces, comm_lc, time_txt, cov_line, cov_dot]

n_frames = N_STEPS // SPEED
anim = FuncAnimation(fig, update, frames=n_frames,
                     interval=40, blit=True, repeat=True)

plt.tight_layout()
plt.show()
