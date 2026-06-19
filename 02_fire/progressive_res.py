"""
Progressive-resolution study — real terrain + real wind + real fuel.

Runs the SAME real-data fire scenario over a fixed geographic box at increasing
grid resolution (coarse → fine). The 3DEP DEM and LANDFIRE fuel are re-fetched at
each resolution (the REST services resample natively), so finer runs resolve finer
terrain ridges and fuel/firebreak structure. Reports cells, data volume, and the
MLX-vs-NumPy wall-clock — the GPU advantage grows as the grid scales.

Run: ~/ds/bin/python 02_fire/progressive_res.py
Output: output/fire_progressive.png
"""

import sys
import time
import numpy as np

import rothermel_ca as rc
from hrrr_wind import fetch_hrrr_wind
from real_terrain import make_terrain_3dep
from real_fuel import make_fuel_fbfm40

BOX_E, BOX_N = 8000.0, 5500.0          # fixed box [m] (East, North)
RESOLUTIONS  = [90, 50, 30, 15, 10]    # metres/cell, coarse → fine
LAT0, LON0   = 38.8, -120.3            # Eldorado NF, Sierra Nevada (contiguous wildland)
DATE         = '2024-08-15 21:00'
NP_MAX_CELLS = 600_000                 # run NumPy at all sizes to show the MLX crossover


def run(date=DATE, lat0=LAT0, lon0=LON0):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)
    terr = make_terrain_3dep(lat0, lon0, verbose=False)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=False)

    rows = []
    for DX in RESOLUTIONS:
        H, W = round(BOX_N/DX), round(BOX_E/DX)
        kw = dict(H=H, W=W, DX=float(DX), fuel=rc.FUEL_MODELS[2], M=0.08,
                  wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                  ignition=(0.32, 0.50), terrain_fn=terr, fuel_fn=fuel)
        r_mx = rc.simulate(**kw, use_mlx=True)
        cells = H*W
        np_s = None
        if cells <= NP_MAX_CELLS:
            np_s = rc.simulate(**kw, use_mlx=False)['wall']
        # data volume: DEM float32 + fuel uint16
        vol_mb = cells*(4+2)/1e6
        burned = np.isfinite(r_mx['T']).mean()*100
        rows.append(dict(DX=DX, H=H, W=W, cells=cells, mx=r_mx['wall'], np=np_s,
                         vol_mb=vol_mb, burned=burned, r=r_mx))
        sp = f"{np_s/r_mx['wall']:.1f}x" if np_s else "  —"
        print(f"DX={DX:>3} m  {H:>4}×{W:<4} = {cells:>7} cells  "
              f"MLX {r_mx['wall']:6.2f}s  NumPy {('%.2f s'%np_s) if np_s else '  (skip)':>8}  "
              f"speedup {sp:>5}  data {vol_mb:5.1f} MB  burned {burned:3.0f}%")
    return rows, w


def make_figure(rows, w, out='output/fire_progressive.png'):
    import os; os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(4*n, 7.5), facecolor='white')
    vmax = 180
    for i, row in enumerate(rows):
        r = row['r']; T = r['T']/60.0
        ax = fig.add_subplot(2, n, i+1)
        # terrain hillshade backdrop
        ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
        ax.imshow(ls.hillshade(r['z'], vert_exag=2.0, dx=r['DX'], dy=r['DX']),
                  cmap='gray', origin='lower', alpha=0.9,
                  extent=[0, BOX_E/1000, 0, BOX_N/1000])
        # nonburnable (firebreaks) shaded blue
        nb = ~r['burnable']
        ax.imshow(np.ma.masked_where(~nb, nb), cmap=matplotlib.colors.ListedColormap(['#3a7bd5']),
                  origin='lower', alpha=0.35, extent=[0, BOX_E/1000, 0, BOX_N/1000])
        ax.contourf(T, levels=np.linspace(0, vmax, 13), cmap='inferno', alpha=0.6,
                    origin='lower', extent=[0, BOX_E/1000, 0, BOX_N/1000], extend='max')
        ax.set_title(f"{row['DX']} m  ·  {row['cells']:,} cells", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # scaling chart
    ax2 = fig.add_subplot(2, 1, 2)
    cells = [r['cells'] for r in rows]
    mx = [r['mx'] for r in rows]
    npt = [r['np'] for r in rows]
    ax2.plot(cells, mx, 'o-', color='#1D9E75', label='MLX (M1 Pro GPU)')
    cn = [c for c, n in zip(cells, npt) if n]; nn = [n for n in npt if n]
    ax2.plot(cn, nn, 's--', color='#378ADD', label='NumPy (CPU)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('grid cells'); ax2.set_ylabel('wall-clock [s]')
    ax2.set_title('Fire-CA cost vs resolution (real terrain + wind + fuel)')
    ax2.grid(True, which='both', alpha=0.3); ax2.legend()
    for r in rows:
        ax2.annotate(f"{r['DX']}m", (r['cells'], r['mx']), fontsize=8,
                     textcoords='offset points', xytext=(4, 5))

    fig.suptitle(f"Progressive resolution — Sierra ({LAT0}°, {LON0}°), real 3DEP terrain + "
                 f"HRRR wind ({w['midflame_ms']:.1f} m/s) + LANDFIRE FBFM40 fuel",
                 fontsize=12, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    rows, w = run()
    make_figure(rows, w)
