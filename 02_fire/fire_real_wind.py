"""
Drive the Rothermel CA with REAL HRRR wind (via Herbie) instead of a synthetic
value. This is the first "real-environment" scale-up step: a fire run whose wind
speed + direction come from an actual operational forecast for a real time and
place. (Terrain is still synthetic here — a real DEM is the next layer.)

Run: ~/ds/bin/python 02_fire/fire_real_wind.py
     ~/ds/bin/python 02_fire/fire_real_wind.py '2024-08-15 21:00' 39.0 -121.0
Outputs: output/fire_ca_realwind.png
"""

import sys
import numpy as np
import rothermel_ca as rc
from hrrr_wind import fetch_hrrr_wind


def main(date='2024-08-15 21:00', lat0=39.0, lon0=-121.0):
    w = fetch_hrrr_wind(date, lat0=lat0, lon0=lon0, verbose=True)

    r = rc.simulate(H=220, W=320, DX=25.0, fuel=rc.FUEL_MODELS[2], M=0.08,
                    wind_ms=w['midflame_ms'], wind_from_deg=w['from_deg'],
                    ignition=(0.30, 0.55), use_mlx=True, verbose=False)

    burned = np.isfinite(r['T']).mean()*100
    print(f"\nCA ({r['backend']}): {r['H']}×{r['W']} @ {r['DX']:.0f} m, "
          f"{r['sweeps']} sweeps in {r['wall']:.2f}s, burned {burned:.0f}%")
    print(f"head ROS range: {r['head_ros_ms'].min()*60:.1f}–{r['head_ros_ms'].max()*60:.1f} m/min")

    note = (f"Wind: real HRRR {w['source']} {w['date']} UTC @ "
            f"({lat0:.2f}°, {lon0:.2f}°) — 10 m {w['speed_10m']:.1f} m/s from "
            f"{w['from_deg']:.0f}°, midflame {w['midflame_ms']:.1f} m/s (WAF {w['waf']}). "
            f"Terrain synthetic. Source: NOAA HRRR via Herbie.")
    rc.make_figures(r, out_prefix='output/fire_ca_realwind', note=note)
    return r, w


if __name__ == '__main__':
    a = sys.argv
    if len(a) >= 4:
        main(a[1], float(a[2]), float(a[3]))
    else:
        main()
