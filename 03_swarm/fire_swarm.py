"""
Project Sycamore — coupled fire ↔ swarm simulation.

The Rothermel CA (02_fire/rothermel_ca.py) produces a fire arrival-time field
T(x,y) over terrain. Here the swarm's patrol target is no longer a static
ellipse: each step we extract the LIVE fire front — the closed contour of T at
the current fire time — and feed it to the boids physics as a time-varying
perimeter. The fleet deploys from a base, reaches the young fire, and must keep
re-wrapping the front as it expands and runs downwind.

Time: UAV dynamics run at the swarm DT (0.5 s). Fire evolves on a minutes
timescale, so fire time is COMPRESSED relative to the UAV clock (FIRE_T0→FIRE_T1
minutes mapped across the sim) so both evolve visibly in one view. The fire
physics and the drone physics are each real; only their relative playback rate
is a presentation choice (the apparent front speed stays below V_CRUISE so the
swarm can physically track it).

Run: ~/ds/bin/python 03_swarm/fire_swarm.py            # animated window
     headless preview via fire_swarm.preview(...)
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_fire'))
import contourpy
import rothermel_ca as rc
import swarm_core as sc
import boids_swarm_3d as b3            # reuse OBJ silhouette + disc/colour helpers

# ── Shared scenario (4×3 km, ignition at world origin, wind from west) ─────────
F_H, F_W, F_DX = 150, 200, 20.0
F_WIND_MS      = 2.2
F_IGNITION     = (0.28, 0.50)         # (col,row) fraction → placed at world origin
CX = F_IGNITION[0]*F_W*F_DX           # world-origin offset so ignition sits at (0,0)
CY = F_IGNITION[1]*F_H*F_DX
N_FRONT   = 400                       # resampled front points (matches swarm arc-index scale)
FIRE_T0   = 20.0                      # fire-minutes at sim start (young, ~0.5 km front)
FIRE_T1   = 120.0                     # fire-minutes at sim end  (~3 km front)


def grid_to_world(L):
    """contour points (col,row) → centred world XZ (East,North) in metres."""
    return np.column_stack([L[:, 0]*F_DX - CX, L[:, 1]*F_DX - CY]).astype('f4')


def build_fire(verbose=False, wind_ms=F_WIND_MS, wind_from_deg=270.0, wind_meta=None,
               terrain_fn=None, fuel_fn=None, wind_fn=None):
    r = rc.simulate(H=F_H, W=F_W, DX=F_DX, wind_ms=wind_ms, wind_from_deg=wind_from_deg,
                    ignition=F_IGNITION, use_mlx=True, max_sweeps=800, verbose=verbose,
                    terrain_fn=terrain_fn, fuel_fn=fuel_fn, wind_fn=wind_fn)
    r['T_min'] = r['T']/60.0
    r['cg'] = contourpy.contour_generator(z=np.nan_to_num(r['T_min'], nan=1e9))
    r['wind_meta'] = wind_meta            # provenance (e.g. real HRRR) for the title
    return r


def _fire_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_fire')


def build_fire_hrrr(date='2024-08-15 21:00', lat0=39.0, lon0=-121.0, verbose=False):
    """Coupled scenario driven by REAL HRRR wind via Herbie."""
    sys.path.insert(0, _fire_dir())
    from hrrr_wind import fetch_hrrr_wind
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=verbose)
    meta = (f"real HRRR {w['date']}Z @({lat0:.1f},{lon0:.1f}) "
            f"10 m {w['speed_10m']:.1f} m/s from {w['from_deg']:.0f}° → midflame "
            f"{w['midflame_ms']:.1f} m/s")
    return build_fire(verbose=verbose, wind_ms=w['midflame_ms'],
                      wind_from_deg=w['from_deg'], wind_meta=meta)


def build_fire_real(date='2024-08-15 21:00', lat0=38.8, lon0=-120.3, verbose=False):
    """Coupled scenario on REAL HRRR wind + REAL 3DEP terrain + REAL LANDFIRE fuel."""
    sys.path.insert(0, _fire_dir())
    from hrrr_wind import fetch_hrrr_wind
    from real_terrain import make_terrain_3dep
    from real_fuel import make_fuel_fbfm40
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=verbose)
    terr = make_terrain_3dep(lat0, lon0, verbose=verbose)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=verbose)
    meta = (f"real HRRR {w['date']}Z + 3DEP terrain + LANDFIRE fuel @({lat0:.1f},{lon0:.1f}) — "
            f"midflame {w['midflame_ms']:.1f} m/s from {w['from_deg']:.0f}°")
    return build_fire(verbose=verbose, wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                      wind_meta=meta, terrain_fn=terr, fuel_fn=fuel)


def _resample_closed(poly, n):
    """Resample an ordered (≈closed) loop to n points by arc length."""
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[:1]])
    seg = np.diff(poly, axis=0)
    d   = np.hypot(seg[:, 0], seg[:, 1])
    s   = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-6:
        return np.repeat(poly[:1], n, axis=0).astype('f4')
    u = np.linspace(0, total, n, endpoint=False)
    x = np.interp(u, s, poly[:, 0]);  y = np.interp(u, s, poly[:, 1])
    return np.column_stack([x, y]).astype('f4')


def front_at(cg, t_min, prev=None):
    """Largest closed fire front at fire time t_min, in world XZ, resampled."""
    lines = [L for L in cg.lines(float(t_min)) if len(L) > 3]
    if not lines:
        return prev
    L = max(lines, key=len)
    return _resample_closed(grid_to_world(L), N_FRONT)


def fire_time(step, n_steps):
    return FIRE_T0 + (FIRE_T1 - FIRE_T0)*step/max(n_steps-1, 1)


# ── Coupled pre-simulation ─────────────────────────────────────────────────────

def presim(fire=None, seed=7, n_steps=sc.N_STEPS, verbose=False):
    if fire is None:
        fire = build_fire(verbose=verbose)
    cg = fire['cg']
    rng = np.random.default_rng(seed)
    pos, vel = sc.init_state(rng)                      # deployment from BASE toward origin/fire
    N = sc.N_DRONES
    tp = np.empty((n_steps, N, 3), 'f4')
    tv = np.empty((n_steps, N, 3), 'f4')
    tc = np.empty(n_steps, 'f4')                       # coverage of the LIVE front
    terr = np.empty(n_steps, 'f4')                     # mean drone→front tracking error [m]
    fr = np.empty((n_steps, N_FRONT, 2), 'f4')
    ft = np.empty(n_steps, 'f4')
    front = None
    t0 = time.perf_counter()
    for t in range(n_steps):
        ftm = fire_time(t, n_steps)
        front = front_at(cg, ftm, prev=front)
        fr[t] = front;  ft[t] = ftm
        tp[t] = pos;  tv[t] = vel
        tc[t] = sc.coverage(pos[:, [0, 2]], perim=front)
        d = np.linalg.norm(front[None] - pos[:, [0, 2]][:, None], axis=2).min(axis=1)
        terr[t] = float(d.mean())
        if verbose and t % 200 == 0:
            print(f"  t={t*sc.DT:.0f}s  fire={ftm:.0f}min  cov={tc[t]*100:.0f}%  trackerr={terr[t]:.0f}m")
        pos, vel = sc.step_np(pos, vel, rng, perim=front)
    if verbose:
        print(f"coupled presim {n_steps} steps in {time.perf_counter()-t0:.2f}s")
    return dict(tp=tp, tv=tv, tc=tc, terr=terr, front=fr, ftime=ft, fire=fire)


# ── Rendering ──────────────────────────────────────────────────────────────────

def _front3(front, y=0.0):
    return np.column_stack([front[:, 0], np.full(len(front), y), front[:, 1]])


def build_figure(R):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    tp, tv, tc, terr, fr, ft = R['tp'], R['tv'], R['tc'], R['terr'], R['front'], R['ftime']
    fire = R['fire']
    N = sc.N_DRONES;  n_steps = len(tp);  n_frames = n_steps // b3.SPEED_X
    t_axis = np.arange(n_steps)*sc.DT
    Ex = [-CX, F_W*F_DX - CX];  Nz = [-CY, F_H*F_DX - CY]   # world bounds (m)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    try:
        fig.canvas.manager.set_window_title("Project Sycamore — Fire ↔ Swarm")
    except Exception:
        pass

    # View zoomed to the fire corridor (the action lives in ~−0.4..1.4 km East).
    VX = [-450, 1500];  VZ = [-750, 750]
    ax3 = fig.add_axes([0.02, 0.04, 0.66, 0.93], projection='3d')
    ax3.set_facecolor('#0d1117')
    ax3.set_xlim(VX);  ax3.set_ylim(0, 360);  ax3.set_zlim(VZ)
    ax3.set_box_aspect((3.0, 2.6, 2.1))           # strong vertical exaggeration so 150 m alt reads
    ax3.set_xlabel('East [m]', color='#8b949e', fontsize=8, labelpad=8)
    ax3.set_ylabel('Alt [m]',  color='#8b949e', fontsize=8, labelpad=4)
    ax3.set_zlabel('North [m]', color='#8b949e', fontsize=8, labelpad=8)
    ax3.tick_params(colors='#8b949e', labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False;  pane.set_edgecolor('#21262d')
    ax3.grid(True, color='#21262d', linewidth=0.4)
    ax3.view_init(elev=46, azim=-80)

    # unburned ground plane (drawn to the view bounds)
    ground = Poly3DCollection(
        [[(VX[0], 0, VZ[0]), (VX[1], 0, VZ[0]), (VX[1], 0, VZ[1]), (VX[0], 0, VZ[1])]],
        facecolor=(0.08, 0.13, 0.08, 0.92), edgecolor=(0.2, 0.3, 0.2, 0.4), zorder=0)
    ax3.add_collection3d(ground)

    # dynamic: burned area (filled front), active front line, drones, pins, links
    burned = Poly3DCollection([_front3(fr[0], 1.0)], facecolor=(0.28, 0.09, 0.04, 0.95),
                              edgecolor=(0.85, 0.32, 0.10, 0.9), linewidth=1.0, zorder=1)
    ax3.add_collection3d(burned)
    front_lc = Line3DCollection([_front3(fr[0], 4.0)], colors=(1.0, 0.50, 0.12, 1.0),
                                linewidths=3.0, zorder=4)
    ax3.add_collection3d(front_lc)
    drone_polys = Poly3DCollection(b3._disc_polys_all(tp[0], tv[0], scale=150.0),
                                   facecolor=[(0.31, 0.76, 0.97, 0.9)]*N,
                                   edgecolor=[(0.85, 0.9, 0.95, 0.5)]*N, linewidth=0.3, zorder=6)
    ax3.add_collection3d(drone_polys)
    pins = Line3DCollection(np.stack(
        [np.column_stack([tp[0, :, 0], np.zeros(N), tp[0, :, 2]]), tp[0]], axis=1),
        colors=(0.31, 0.76, 0.97, 0.16), linewidths=0.4, zorder=2)
    ax3.add_collection3d(pins)
    comm = Line3DCollection([[[0, 0, 0], [0, 0, 0]]], colors=(0.31, 0.76, 0.97, 0.14),
                            linewidths=0.4, zorder=3)
    ax3.add_collection3d(comm)

    # 2D fire-map inset
    ax2 = fig.add_axes([0.70, 0.55, 0.28, 0.4])
    ax2.set_facecolor('#0d1117')
    Tmask = np.ma.masked_greater(fire['T_min'], FIRE_T1)       # hide not-yet-burned
    cmap = __import__('matplotlib').colormaps['inferno'].copy()
    cmap.set_bad((0.09, 0.14, 0.09, 1.0))                      # unburned → ground green
    ax2.imshow(Tmask, cmap=cmap, origin='lower',
               extent=[Ex[0], Ex[1], Nz[0], Nz[1]], vmin=FIRE_T0, vmax=FIRE_T1)
    fr2d, = ax2.plot(fr[0][:, 0], fr[0][:, 1], color='#39d353', lw=1.4)
    dr2d, = ax2.plot(tp[0, :, 0], tp[0, :, 2], '.', color='#4fc3f7', ms=2)
    ax2.set_xlim(Ex);  ax2.set_ylim(Nz)
    ax2.set_title('fire arrival map + live front', color='#8b949e', fontsize=8)
    ax2.tick_params(colors='#8b949e', labelsize=6)

    # coverage + tracking-error chart
    axc = fig.add_axes([0.70, 0.08, 0.28, 0.36])
    axc.set_facecolor('#0d1117')
    axc.set_xlim(0, t_axis[-1]);  axc.set_ylim(0, 100)
    axc.plot(t_axis, tc*100, color='#21262d', lw=0.7)
    covl, = axc.plot([], [], color='#39d353', lw=1.5, label='front coverage %')
    axc.set_xlabel('t [s]', color='#8b949e', fontsize=8)
    axc.set_ylabel('coverage %', color='#39d353', fontsize=8)
    axc.tick_params(colors='#8b949e', labelsize=7)
    axc.grid(True, color='#21262d', linewidth=0.4)
    axt = axc.twinx()
    axt.set_ylim(0, max(R['terr'].max()*1.1, 50))
    axt.plot(t_axis, terr, color='#21262d', lw=0.7)
    terl, = axt.plot([], [], color='#f0883e', lw=1.2, label='track err [m]')
    axt.set_ylabel('track err [m]', color='#f0883e', fontsize=8)
    axt.tick_params(colors='#f0883e', labelsize=7)

    time_txt = fig.text(0.35, 0.97, '', ha='center', va='top', color='#c9d1d9',
                        fontsize=10, fontfamily='monospace')
    if fire.get('wind_meta'):
        fig.text(0.35, 0.945, fire['wind_meta'], ha='center', va='top',
                 color='#8b949e', fontsize=8)

    def update(frame):
        t = min(frame*b3.SPEED_X, n_steps-1)
        pt, vt = tp[t], tv[t]
        burned.set_verts([_front3(fr[t], 1.0)])
        front_lc.set_segments([_front3(fr[t], 4.0)])
        drone_polys.set_verts(b3._disc_polys_all(pt, vt, scale=150.0))
        spds = np.linalg.norm(vt[:, [0, 2]], axis=1)
        drone_polys.set_facecolors(b3._speed_color(spds))
        pins.set_segments(np.stack(
            [np.column_stack([pt[:, 0], np.zeros(N), pt[:, 2]]), pt], axis=1))
        pxz = pt[:, [0, 2]]
        D = np.linalg.norm(pxz[:, None] - pxz[None], axis=2)
        iu, ju = np.where(np.triu(D < b3.R_LINK_VIS, k=1))
        comm.set_segments(np.stack([pt[iu], pt[ju]], axis=1) if len(iu)
                          else [[[0, 0, 0], [0, 0, 0]]])
        fr2d.set_data(fr[t][:, 0], fr[t][:, 1])
        dr2d.set_data(pt[:, 0], pt[:, 2])
        covl.set_data(t_axis[:t+1], tc[:t+1]*100)
        terl.set_data(t_axis[:t+1], terr[:t+1])
        time_txt.set_text(f"t = {t*sc.DT:5.0f} s   fire t = {ft[t]:5.0f} min   "
                          f"front coverage = {tc[t]*100:4.0f}%   track err = {terr[t]:4.0f} m   N = {N}")
        return [burned, front_lc, drone_polys, pins, comm, fr2d, dr2d, covl, terl, time_txt]

    return fig, update, n_frames


def preview(frame_idx, out_path, R=None):
    import matplotlib.pyplot as plt
    if R is None:
        R = presim(verbose=True)
    fig, update, n_frames = build_figure(R)
    update(min(frame_idx, n_frames-1))
    fig.savefig(out_path, dpi=110, facecolor='#0d1117')
    plt.close(fig)
    print(f"Saved {out_path}")
    return R


def main():
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    R = presim(verbose=True)
    fig, update, n_frames = build_figure(R)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)
    globals()['_ani'] = ani
    plt.show()
    return ani


if __name__ == '__main__':
    main()
