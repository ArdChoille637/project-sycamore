"""
Project Sycamore — 3D Swarm Simulation (matplotlib Axes3D)

Physics  : 3-D boids with altitude hold  (see swarm_core.py)
  XZ forces : separation · alignment · fire tracking · arc-index spread · tangential patrol
  Y  force  : PD altitude controller  (ALT_NOM = 150 m)
Backend  : MLX (Apple-Silicon GPU) when available, else NumPy — selected at runtime.
Fleet    : N_DRONES = 150  (see swarm_core.N_DRONES)
Rendering : matplotlib dark-theme Axes3D with OBJ disc silhouettes, ground plane,
            altitude pins, comm links, and live coverage + altitude charts.

Run: ~/ds/bin/python 03_swarm/boids_swarm_3d.py
"""

import os
import time
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib
if not os.environ.get('MPLBACKEND'):
    matplotlib.use('macosx')                 # native window; override via MPLBACKEND=Agg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import swarm_core as sc
from swarm_core import (N_DRONES, FIRE_A, FIRE_B, ALT_NOM, V_MAX,
                        R_PERC, R_SEP, DT, T_MAX, N_STEPS)

OBJ_PATH    = '/Users/home/Downloads/sycamore_active.obj'
SPEED_X     = 4                  # sim steps advanced per rendered frame
DRONE_SCALE = 90.0               # visual disc diameter [m]
R_LINK_VIS  = 140.0              # only draw comm links shorter than this (mesh clarity)

# ── OBJ silhouette (top-down, for 2-D disc rendering at altitude) ──────────────

def _load_silhouette():
    verts = []
    with open(OBJ_PATH) as fh:
        for ln in fh:
            if ln.startswith('v '):
                p = ln.split()
                verts.append([float(p[1]), float(p[3])])   # x, z
    v    = np.array(verts)
    hull = ConvexHull(v)
    hp   = v[hull.vertices].copy()
    hp  -= hp.mean(axis=0)
    cov  = np.cov(hp.T)
    evl, evc = np.linalg.eigh(cov)
    major = evc[:, np.argmax(evl)]
    ang   = np.arctan2(major[1], major[0])
    c, s  = np.cos(-ang), np.sin(-ang)
    hp    = (np.array([[c, -s], [s, c]]) @ hp.T).T
    if hp[:, 0].max() < abs(hp[:, 0].min()):
        hp[:, 0] *= -1
    hp /= (hp[:, 0].max() - hp[:, 0].min())   # unit chord length
    return hp.astype('f4')                      # (M, 2)

SILHOUETTE = _load_silhouette()

# ── Vectorised disc geometry (all N drones at once) ────────────────────────────

def _disc_polys_all(pos, vel, scale=DRONE_SCALE):
    """List of N (M,3) vertex arrays: OBJ hull rotated to each heading, at altitude."""
    hdg = np.arctan2(vel[:, 2], vel[:, 0])                 # (N,)
    c, s = np.cos(hdg)[:, None], np.sin(hdg)[:, None]      # (N,1)
    px, py, pz = pos[:, 0][:, None], pos[:, 1][:, None], pos[:, 2][:, None]
    ax, ay = SILHOUETTE[None, :, 0], SILHOUETTE[None, :, 1]  # (1,M)
    wx = px + (ax*scale)*c - (ay*scale)*s                  # (N,M)
    wz = pz + (ax*scale)*s + (ay*scale)*c
    wy = np.broadcast_to(py, wx.shape)
    return [np.column_stack([wx[i], wy[i], wz[i]]) for i in range(len(pos))]

# ── Speed → colour ─────────────────────────────────────────────────────────────

def _speed_color(spd_xz):
    f = np.clip((spd_xz - 1.0)/(V_MAX - 1.0), 0, 1)
    r = 0x4f/255 + f*(1.0      - 0x4f/255)
    g = 0xc3/255 + f*(0x98/255 - 0xc3/255)
    b = 0xf7/255 + f*(0.0      - 0xf7/255)
    return np.column_stack([r, g, b, np.full_like(f, 0.85)])

# ── Pre-simulation (GPU when available) ────────────────────────────────────────

def run_presim():
    if sc.HAVE_MLX:
        print(f"Pre-simulating {N_STEPS} steps on MLX (Apple-Silicon GPU), N={N_DRONES} …")
        t0 = time.perf_counter()
        tp, tv, tc = sc.presim_mx(verbose=True)
        print(f"Done in {time.perf_counter()-t0:.2f}s  —  final cov {tc[-1]*100:.1f}%")
    else:
        print(f"Pre-simulating {N_STEPS} steps on NumPy (CPU), N={N_DRONES} …")
        t0 = time.perf_counter()
        tp, tv, tc = sc.presim_np(verbose=True)
        print(f"Done in {time.perf_counter()-t0:.2f}s  —  final cov {tc[-1]*100:.1f}%")
    return tp, tv, tc

# ── Main ───────────────────────────────────────────────────────────────────────

def build_figure(tp, tv, tc):
    n_frames = N_STEPS // SPEED_X

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    try:
        fig.canvas.manager.set_window_title("Project Sycamore — 3D Swarm (N=%d)" % N_DRONES)
    except Exception:
        pass

    ax3 = fig.add_axes([0.02, 0.12, 0.68, 0.84], projection='3d')
    ax3.set_facecolor('#0d1117')
    ax3.set_xlim(-FIRE_A*1.95, FIRE_A*1.35)          # frame staging base + fire
    ax3.set_ylim(-50, 500)
    ax3.set_zlim(-FIRE_B*1.7, FIRE_B*1.7)
    ax3.set_xlabel('East [m]',  color='#8b949e', fontsize=8, labelpad=6)
    ax3.set_ylabel('Alt [m]',   color='#8b949e', fontsize=8, labelpad=6)
    ax3.set_zlabel('North [m]', color='#8b949e', fontsize=8, labelpad=6)
    ax3.tick_params(colors='#8b949e', labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#21262d')
    ax3.grid(True, color='#21262d', linewidth=0.4)
    ax3.view_init(elev=22, azim=35)

    # Coverage chart
    ax_cov = fig.add_axes([0.73, 0.55, 0.25, 0.38])
    ax_cov.set_facecolor('#0d1117')
    ax_cov.set_xlim(0, T_MAX);  ax_cov.set_ylim(0, 100)
    ax_cov.set_xlabel('t [s]', color='#8b949e', fontsize=8)
    ax_cov.set_ylabel('Coverage %', color='#8b949e', fontsize=8)
    ax_cov.tick_params(colors='#8b949e', labelsize=7)
    ax_cov.grid(True, color='#21262d', linewidth=0.4)
    ax_cov.axhline(80, color='#f85149', lw=0.7, ls='--')
    t_axis = np.arange(N_STEPS) * DT
    ax_cov.plot(t_axis, tc*100, color='#21262d', lw=0.7)
    cov_line, = ax_cov.plot([], [], color='#4fc3f7', lw=1.4)
    cov_dot,  = ax_cov.plot([], [], 'o', color='#f0e68c', ms=4)

    # Altitude chart
    ax_alt = fig.add_axes([0.73, 0.12, 0.25, 0.33])
    ax_alt.set_facecolor('#0d1117')
    ax_alt.set_xlim(0, T_MAX);  ax_alt.set_ylim(50, 300)
    ax_alt.set_xlabel('t [s]', color='#8b949e', fontsize=8)
    ax_alt.set_ylabel('Altitude [m]', color='#8b949e', fontsize=8)
    ax_alt.tick_params(colors='#8b949e', labelsize=7)
    ax_alt.grid(True, color='#21262d', linewidth=0.4)
    ax_alt.axhline(ALT_NOM, color='#f85149', lw=0.7, ls='--', label=f'nom {ALT_NOM:.0f}m')
    ax_alt.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#8b949e')
    alt_mean = tp[:, :, 1].mean(axis=1)
    alt_lo   = tp[:, :, 1].min(axis=1)
    alt_hi   = tp[:, :, 1].max(axis=1)
    ax_alt.fill_between(t_axis, alt_lo, alt_hi, color='#4fc3f7', alpha=0.12)
    ax_alt.plot(t_axis, alt_mean, color='#21262d', lw=0.7)
    alt_line, = ax_alt.plot([], [], color='#4fc3f7', lw=1.4)
    alt_dot,  = ax_alt.plot([], [], 'o', color='#f0e68c', ms=4)

    time_txt = fig.text(0.38, 0.965, '', ha='center', va='top',
                        color='#c9d1d9', fontsize=10, fontfamily='monospace')

    # Static scene: fire ground ellipse + ghost patrol ring
    ang_p = np.linspace(0, 2*np.pi, 120, endpoint=False)
    ex = FIRE_A*np.cos(ang_p);  ez = FIRE_B*np.sin(ang_p);  ey = np.zeros_like(ex)
    ax3.add_collection3d(Poly3DCollection(
        [list(zip(ex, ey, ez))], facecolor=(0.94, 0.33, 0.13, 0.18),
        edgecolor=(1.0, 0.42, 0.0, 0.7), linewidth=1.0, zorder=1))
    ax3.add_collection3d(Poly3DCollection(
        [list(zip(ex, np.full_like(ex, ALT_NOM), ez))], facecolor=(0, 0, 0, 0),
        edgecolor=(0.4, 0.4, 0.7, 0.25), linewidth=0.6, linestyle='--', zorder=1))

    # Staging base marker (launch point on the ground)
    bx, bz = float(sc.BASE_XZ[0]), float(sc.BASE_XZ[1])
    ba = np.linspace(0, 2*np.pi, 40)
    ax3.add_collection3d(Poly3DCollection(
        [list(zip(bx + sc.BASE_R*np.cos(ba), np.zeros(40), bz + sc.BASE_R*np.sin(ba)))],
        facecolor=(0.31, 0.76, 0.97, 0.12), edgecolor=(0.31, 0.76, 0.97, 0.6),
        linewidth=1.0, zorder=1))
    ax3.text(bx, 0, bz - sc.BASE_R*1.6, 'BASE', color='#4fc3f7', fontsize=7, ha='center')

    # Dynamic collections
    drone_polys = Poly3DCollection(
        _disc_polys_all(tp[0], tv[0]),
        facecolor=[(0.31, 0.76, 0.97, 0.82)]*N_DRONES,
        edgecolor=[(0.78, 0.82, 0.87, 0.4)]*N_DRONES, linewidth=0.3, zorder=5)
    ax3.add_collection3d(drone_polys)

    pin_segs = np.stack([np.column_stack([tp[0, :, 0], np.zeros(N_DRONES), tp[0, :, 2]]),
                         tp[0]], axis=1)
    drone_pins = Line3DCollection(pin_segs, colors=(0.31, 0.76, 0.97, 0.18),
                                  linewidths=0.4, zorder=2)
    ax3.add_collection3d(drone_pins)

    comm_lc = Line3DCollection([[[0, 0, 0], [0, 0, 0]]], colors=(0.31, 0.76, 0.97, 0.16),
                               linewidths=0.4, zorder=3)
    ax3.add_collection3d(comm_lc)

    def update(frame):
        t  = min(frame * SPEED_X, N_STEPS - 1)
        pt = tp[t];  vt = tv[t];  cov = tc[t];  sim_t = t * DT

        drone_polys.set_verts(_disc_polys_all(pt, vt))
        spds = np.linalg.norm(vt[:, [0, 2]], axis=1)
        drone_polys.set_facecolors(_speed_color(spds))
        drone_polys.set_edgecolors([(0.78, 0.82, 0.87, 0.35)]*N_DRONES)

        drone_pins.set_segments(np.stack(
            [np.column_stack([pt[:, 0], np.zeros(N_DRONES), pt[:, 2]]), pt], axis=1))

        pxz = pt[:, [0, 2]]
        D = np.linalg.norm(pxz[:, None] - pxz[None], axis=2)
        iu, ju = np.where(np.triu(D < R_LINK_VIS, k=1))
        comm_lc.set_segments(np.stack([pt[iu], pt[ju]], axis=1)
                             if len(iu) else [[[0, 0, 0], [0, 0, 0]]])

        cov_line.set_data(t_axis[:t+1], tc[:t+1]*100)
        cov_dot.set_data([sim_t], [cov*100])
        alt_line.set_data(t_axis[:t+1], alt_mean[:t+1])
        alt_dot.set_data([sim_t], [alt_mean[t]])
        time_txt.set_text(
            f"t = {sim_t:5.0f} s     coverage = {cov*100:.1f}%     "
            f"alt = {alt_mean[t]:.0f} m     N = {N_DRONES}     links = {len(iu)}")
        return [drone_polys, drone_pins, comm_lc, cov_line, cov_dot,
                alt_line, alt_dot, time_txt]

    return fig, update, n_frames


def main():
    tp, tv, tc = run_presim()
    fig, update, n_frames = build_figure(tp, tv, tc)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)
    globals()['_ani'] = ani          # keep a ref so GC doesn't kill the animation
    plt.show()
    return ani


def preview(frame_idx, out_path, tp=None, tv=None, tc=None):
    """Headless: render a single frame to a PNG (used for verification)."""
    if tp is None:
        tp, tv, tc = run_presim()
    fig, update, n_frames = build_figure(tp, tv, tc)
    update(min(frame_idx, n_frames - 1))
    fig.savefig(out_path, dpi=110, facecolor='#0d1117')
    print(f"Saved {out_path}")
    return tp, tv, tc


if __name__ == '__main__':
    main()
