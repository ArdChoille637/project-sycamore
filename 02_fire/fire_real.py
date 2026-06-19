"""
Rothermel CA over REAL wind + REAL terrain for a real place.

Wind   : NOAA HRRR 10 m → midflame (Herbie).
Terrain: USGS 3DEP bare-earth DEM (REST ImageServer).
Fuel/moisture are still a single uniform Anderson model — the fuel/flora layer
(LANDFIRE) is the next one to wire in.

Run: ~/ds/bin/python 02_fire/fire_real.py
     ~/ds/bin/python 02_fire/fire_real.py '2024-08-15 21:00' 39.0 -121.0
Output: output/fire_ca_real.png
"""

import sys
import numpy as np
import rothermel_ca as rc
from hrrr_wind import fetch_hrrr_wind
from real_terrain import make_terrain_3dep
from real_fuel import make_fuel_fbfm40, fuel_summary
from fuel_moisture import make_moisture_hrrr


def main(date='2024-08-15 21:00', lat0=38.8, lon0=-120.3):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)
    terr = make_terrain_3dep(lat0, lon0, verbose=True)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=True)
    mois = make_moisture_hrrr(date, lat0, lon0, verbose=True)

    r = rc.simulate(H=220, W=320, DX=25.0,
                    wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                    ignition=(0.32, 0.50), terrain_fn=terr, fuel_fn=fuel,
                    moisture_fn=mois, use_mlx=True, verbose=False)

    burned = np.isfinite(r['T']).mean()*100
    relief = f"{r['z'].min():.0f}–{r['z'].max():.0f} m"
    nb = (~r['burnable']).mean()*100
    mf = r['M_field']
    print(f"\nCA ({r['backend']}): {r['H']}×{r['W']} @ {r['DX']:.0f} m, relief {relief}, "
          f"{r['sweeps']} sweeps in {r['wall']:.2f}s, burned {burned:.0f}% of burnable area")
    print(f"head ROS range: {r['head_ros_ms'].min()*60:.1f}–{r['head_ros_ms'].max()*60:.1f} m/min; "
          f"moisture {mf.min()*100:.1f}–{mf.max()*100:.1f}%; nonburnable {nb:.0f}% (firebreaks)")
    print("fuel composition:")
    for code, name, pct in fuel_summary(r['fuel_code'])[:6]:
        print(f"   {name:>4} (code {code}): {pct:.0f}%")

    note = (f"ALL-REAL inputs.  Wind: NOAA HRRR {w['date']} UTC, midflame {w['midflame_ms']:.1f} m/s "
            f"from {w['from_deg']:.0f}°.  Terrain: USGS 3DEP, relief {relief}.  "
            f"Fuel: LANDFIRE FBFM40, {nb:.0f}% nonburnable.  "
            f"Moisture: HRRR-derived spatial {mf.min()*100:.1f}–{mf.max()*100:.1f}%.  "
            f"@ ({lat0:.2f}°, {lon0:.2f}°). Sources: NOAA HRRR (Herbie), USGS 3DEP, LANDFIRE.")
    rc.make_figures(r, out_prefix='output/fire_ca_real', note=note)
    return r, w


if __name__ == '__main__':
    a = sys.argv
    main(a[1], float(a[2]), float(a[3])) if len(a) >= 4 else main()
