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
from real_fuel import make_fuel_fbfm40, fetch_fbfm40, fuel_summary


def main(date='2024-08-15 21:00', lat0=38.8, lon0=-120.3):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)
    terr = make_terrain_3dep(lat0, lon0, verbose=True)
    fuel = make_fuel_fbfm40(lat0, lon0, verbose=True)

    r = rc.simulate(H=220, W=320, DX=25.0, M=0.08,
                    wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                    ignition=(0.32, 0.50), terrain_fn=terr, fuel_fn=fuel,
                    use_mlx=True, verbose=False)

    burned = np.isfinite(r['T']).mean()*100
    relief = f"{r['z'].min():.0f}–{r['z'].max():.0f} m"
    nb = (~r['burnable']).mean()*100
    print(f"\nCA ({r['backend']}): {r['H']}×{r['W']} @ {r['DX']:.0f} m, relief {relief}, "
          f"{r['sweeps']} sweeps in {r['wall']:.2f}s, burned {burned:.0f}% of burnable area")
    print(f"head ROS range: {r['head_ros_ms'].min()*60:.1f}–{r['head_ros_ms'].max()*60:.1f} m/min "
          f"(varies with real fuel + slope); nonburnable {nb:.0f}% (firebreaks)")
    print("fuel composition:")
    for code, name, pct in fuel_summary(r['fuel_code'])[:6]:
        print(f"   {name:>4} (code {code}): {pct:.0f}%")

    note = (f"Wind: real NOAA HRRR {w['date']} UTC, midflame {w['midflame_ms']:.1f} m/s from "
            f"{w['from_deg']:.0f}°.  Terrain: real USGS 3DEP, relief {relief}.  "
            f"Fuel: real LANDFIRE FBFM40 @ ({lat0:.2f}°, {lon0:.2f}°), {nb:.0f}% nonburnable. "
            f"Sources: NOAA HRRR (Herbie) + USGS 3DEP + LANDFIRE.")
    rc.make_figures(r, out_prefix='output/fire_ca_real', note=note)
    return r, w


if __name__ == '__main__':
    a = sys.argv
    main(a[1], float(a[2]), float(a[3])) if len(a) >= 4 else main()
