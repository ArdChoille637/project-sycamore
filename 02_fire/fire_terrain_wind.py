"""
Terrain-downscaled wind vs uniform wind — what mass-conserving wind does to a fire.

Same real scenario (3DEP terrain + LANDFIRE fuel + HRRR ambient wind) run two ways:
the ambient wind applied uniformly, vs. the same ambient mass-conservingly downscaled
over the real DEM (terrain_wind.py). Shows the wind field and the two fire outcomes.

Run: ~/ds/bin/python 02_fire/fire_terrain_wind.py
Output: output/fire_terrain_wind.png
"""

import os
import numpy as np
import rothermel_ca as rc
from hrrr_wind import fetch_hrrr_wind
from real_terrain import make_terrain_3dep
from real_fuel import make_fuel_fbfm40
from terrain_wind import make_wind_terrain, downscale_wind


def main(date='2024-08-15 21:00', lat0=38.8, lon0=-120.3):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)
    terr = make_terrain_3dep(lat0, lon0, verbose=False)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=False)
    common = dict(H=220, W=320, DX=25.0, wind_from_deg=w['from_deg'],
                  ignition=(0.30, 0.50), terrain_fn=terr, fuel_fn=fuel, use_mlx=True)

    r_uni = rc.simulate(wind_ms=w['midflame_ms'], **common)
    wfn = make_wind_terrain(w['midflame_ms'], w['from_deg'], verbose=True)
    r_ter = rc.simulate(wind_fn=wfn, **common)

    sf = r_ter['wind_speed_field']
    m = np.isfinite(r_uni['T']) & np.isfinite(r_ter['T'])
    dmed = np.nanmedian(np.abs(r_uni['T'][m] - r_ter['T'][m]))/60
    print(f"\nambient midflame {w['midflame_ms']:.1f} m/s from {w['from_deg']:.0f}°")
    print(f"terrain wind {sf.min():.1f}–{sf.max():.1f} m/s "
          f"(+{(sf.max()/w['midflame_ms']-1)*100:.0f}% on ridges)")
    print(f"median |Δ arrival| uniform vs terrain-wind: {dmed:.0f} min")
    make_figure(r_uni, r_ter, w, lat0, lon0)


def make_figure(r_uni, r_ter, w, lat0, lon0, out='output/fire_terrain_wind.png'):
    os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DX, H, W = r_ter['DX'], r_ter['H'], r_ter['W']
    ext = [0, W*DX/1000, 0, H*DX/1000]
    z = r_ter['z']; gi, gj = r_ter['ignition']
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    hill = ls.hillshade(z, vert_exag=2.0, dx=DX, dy=DX)
    levels = np.array([15, 30, 45, 60, 90, 120, 180, 240], 'f4')
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.4), facecolor='white')

    # Panel 1 — downscaled wind field
    sf = r_ter['wind_speed_field']
    im = ax[0].imshow(sf, cmap='viridis', origin='lower', extent=ext)
    s = max(H, W)//22
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    _, frm = downscale_wind(z, DX, w['midflame_ms'], w['from_deg'])   # direction field for quiver
    to = np.deg2rad(frm[::s, ::s] + 180.0)
    ax[0].quiver(xx*DX/1000, yy*DX/1000, np.sin(to), np.cos(to), color='white',
                 scale=30, width=0.003, alpha=0.7)
    ax[0].contour(z, levels=10, colors='k', linewidths=0.3, extent=ext, origin='lower', alpha=0.3)
    fig.colorbar(im, ax=ax[0], shrink=0.8).set_label('wind speed [m/s]')
    ax[0].set_title(f"Mass-conserving terrain wind\n(ambient {w['midflame_ms']:.1f} m/s → "
                    f"{sf.min():.1f}–{sf.max():.1f} m/s)")
    ax[0].set(xlabel='East [km]', ylabel='North [km]')

    # Panels 2 & 3 — fire with uniform vs terrain wind
    for a, r, lab in [(ax[1], r_uni, 'uniform wind'), (ax[2], r_ter, 'terrain-downscaled wind')]:
        a.imshow(hill, cmap='gray', origin='lower', extent=ext, alpha=0.9)
        nb = ~r['burnable'] if r.get('burnable') is not None else np.zeros((H, W), bool)
        a.imshow(np.ma.masked_where(~nb, nb), cmap=matplotlib.colors.ListedColormap(['#3a7bd5']),
                 origin='lower', alpha=0.35, extent=ext)
        cs = a.contourf(r['T']/60, levels=levels, cmap='inferno', alpha=0.6,
                        origin='lower', extent=ext, extend='max')
        a.contour(r['T']/60, levels=levels, colors='k', linewidths=0.4, origin='lower',
                  extent=ext, alpha=0.4)
        a.plot(gj*DX/1000, gi*DX/1000, '*', color='cyan', ms=14, mec='k')
        a.set_title(f"Fire — {lab}");  a.set(xlabel='East [km]', ylabel='North [km]')
    fig.colorbar(cs, ax=ax[2], shrink=0.8, ticks=levels).set_label('arrival time [min]')

    fig.suptitle(f"Terrain-downscaled wind vs uniform — Eldorado NF ({lat0}°, {lon0}°), "
                 f"real 3DEP + HRRR + LANDFIRE. Sources: NOAA HRRR, USGS 3DEP, LANDFIRE.",
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
