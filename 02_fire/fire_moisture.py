"""
Spatial dead-fuel moisture — real HRRR weather × terrain microclimate.

Shows the moisture field (shaded high ground wetter, sunny SW slopes drier) and the
fire it produces, on the same real Sierra scenario (3DEP terrain + LANDFIRE fuel +
HRRR wind). Reports the *pure spatial* effect by also running uniform moisture set
to the field mean, so the comparison isn't just a global dry/wet offset.

Run: ~/ds/bin/python 02_fire/fire_moisture.py
Output: output/fire_moisture.png
"""

import os
import numpy as np
import rothermel_ca as rc
from hrrr_wind import fetch_hrrr_wind
from real_terrain import make_terrain_3dep
from real_fuel import make_fuel_fbfm40
from fuel_moisture import make_moisture_hrrr


def main(date='2024-08-15 21:00', lat0=38.8, lon0=-120.3):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)
    terr = make_terrain_3dep(lat0, lon0, verbose=False)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=False)
    mois = make_moisture_hrrr(date, lat0, lon0, verbose=True)
    common = dict(H=220, W=320, DX=25.0, wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                  ignition=(0.30, 0.50), terrain_fn=terr, fuel_fn=fuel, use_mlx=True)

    r_spatial = rc.simulate(moisture_fn=mois, **common)
    mf = r_spatial['M_field']
    r_uniform = rc.simulate(M=float(mf.mean()), **common)   # same MEAN → isolates spatial effect

    m = np.isfinite(r_uniform['T']) & np.isfinite(r_spatial['T'])
    dmed = np.nanmedian(np.abs(r_uniform['T'][m] - r_spatial['T'][m]))/60
    print(f"\nspatial fine dead-fuel moisture {mf.min()*100:.1f}–{mf.max()*100:.1f}% "
          f"(mean {mf.mean()*100:.1f}%)")
    print(f"pure spatial effect (vs uniform at same {mf.mean()*100:.1f}% mean): "
          f"median |Δ arrival| {dmed:.0f} min")
    make_figure(r_spatial, w, lat0, lon0, dmed)


def make_figure(r, w, lat0, lon0, dmed, out='output/fire_moisture.png'):
    os.makedirs('output', exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DX, H, W = r['DX'], r['H'], r['W']
    ext = [0, W*DX/1000, 0, H*DX/1000]
    z = r['z']; gi, gj = r['ignition']; mf = r['M_field']*100
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    hill = ls.hillshade(z, vert_exag=2.0, dx=DX, dy=DX)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), facecolor='white')

    # Panel 1 — moisture field (drier = warmer colour)
    ax[0].imshow(hill, cmap='gray', origin='lower', extent=ext, alpha=0.5)
    im = ax[0].imshow(mf, cmap='YlGnBu', origin='lower', extent=ext, alpha=0.75)
    ax[0].contour(z, levels=12, colors='k', linewidths=0.3, extent=ext, origin='lower', alpha=0.3)
    fig.colorbar(im, ax=ax[0], shrink=0.8).set_label('fine dead-fuel moisture [%]')
    ax[0].set_title("Spatial dead-fuel moisture\n(shaded/high ground wetter, sunny SW slopes drier)")
    ax[0].set(xlabel='East [km]', ylabel='North [km]')

    # Panel 2 — fire on spatial moisture
    levels = np.array([15, 30, 45, 60, 90, 120, 180, 240], 'f4')
    ax[1].imshow(hill, cmap='gray', origin='lower', extent=ext, alpha=0.9)
    nb = ~r['burnable'] if r.get('burnable') is not None else np.zeros((H, W), bool)
    ax[1].imshow(np.ma.masked_where(~nb, nb), cmap=matplotlib.colors.ListedColormap(['#3a7bd5']),
                 origin='lower', alpha=0.35, extent=ext)
    cs = ax[1].contourf(r['T']/60, levels=levels, cmap='inferno', alpha=0.6,
                        origin='lower', extent=ext, extend='max')
    ax[1].contour(r['T']/60, levels=levels, colors='k', linewidths=0.4, origin='lower',
                  extent=ext, alpha=0.4)
    ax[1].plot(gj*DX/1000, gi*DX/1000, '*', color='cyan', ms=14, mec='k')
    fig.colorbar(cs, ax=ax[1], shrink=0.8, ticks=levels).set_label('arrival time [min]')
    ax[1].set_title(f"Fire on spatial moisture\n(pure spatial effect: median |Δt| {dmed:.0f} min)")
    ax[1].set(xlabel='East [km]', ylabel='North [km]')

    fig.suptitle(f"HRRR-driven spatial fuel moisture — Eldorado NF ({lat0}°, {lon0}°): "
                 f"T {w['date']} 2 m weather × 3DEP terrain. Sources: NOAA HRRR, USGS 3DEP, LANDFIRE.",
                 fontsize=11, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
