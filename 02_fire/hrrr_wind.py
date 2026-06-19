"""
Real wind for the Rothermel CA — HRRR 10 m wind via Herbie.

Pulls the operational High-Resolution Rapid Refresh (HRRR, 3 km, NOAA) 10 m
U/V wind for a date + location straight from the AWS open-data archive, averages
it over a small box, and converts the 10 m freestream wind to a **midflame**
wind for Rothermel by a wind-adjustment factor (WAF). Feeding raw 10 m wind to
Rothermel would massively over-drive spread — Rothermel's wind input is the wind
at mid-flame height, typically a fraction of the 10 m / 20 ft wind.

WAF reference: Albini & Baughman (1979); Andrews (2012) RMRS-GTR-266. ~0.4 for
partly-open fuels; 0.1–0.3 under closed canopy; up to ~0.5 fully exposed.

Only the two 10 m wind GRIB messages are downloaded (byte-range subset, a few MB
of public NOAA data) to save_dir.

    from hrrr_wind import fetch_hrrr_wind
    w = fetch_hrrr_wind('2024-08-15 21:00', lat0=39.0, lon0=-121.0)
    # → dict with midflame_ms, from_deg, speed_10m, provenance …
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from herbie import Herbie

DEFAULT_WAF = 0.4


def fetch_hrrr_wind(date='2024-08-15 21:00', lat0=39.0, lon0=-121.0, halfdeg=0.4,
                    fxx=0, waf=DEFAULT_WAF, save_dir='/tmp/herbie', verbose=True):
    """Domain-mean HRRR 10 m wind → Rothermel midflame wind + provenance."""
    H = Herbie(date, model='hrrr', product='sfc', fxx=fxx, save_dir=save_dir, verbose=False)
    if H.grib is None:
        raise RuntimeError(f"No HRRR data found for {date} (try another recent date)")
    ds = H.xarray(":(U|V)GRD:10 m above ground:", verbose=False)

    lat = ds.latitude.values
    lon = ds.longitude.values                      # 0–360
    lonc = lon0 % 360
    box = (np.abs(lat - lat0) <= halfdeg) & \
          (np.abs(((lon - lonc + 180) % 360) - 180) <= halfdeg)
    if box.sum() == 0:
        raise RuntimeError(f"No HRRR cells within {halfdeg}° of ({lat0},{lon0})")

    u = float(ds.u10.values[box].mean())           # eastward
    v = float(ds.v10.values[box].mean())           # northward
    speed10 = float(np.hypot(u, v))
    from_deg = float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)   # met. "from" bearing
    spd_cells = np.hypot(ds.u10.values[box], ds.v10.values[box])

    out = dict(
        speed_10m=speed10,
        midflame_ms=speed10 * waf,
        from_deg=from_deg,
        waf=waf,
        max_10m=float(spd_cells.max()),
        min_10m=float(spd_cells.min()),
        n_cells=int(box.sum()),
        date=str(date), fxx=fxx, lat0=lat0, lon0=lon0,
        source=f"HRRR sfc f{fxx:02d}",
        grib=str(H.grib),
    )
    if verbose:
        print(f"HRRR {date} UTC  @ ({lat0:.2f}, {lon0:.2f})  [{out['n_cells']} cells]")
        print(f"  10 m wind : {speed10:.1f} m/s  from {from_deg:.0f}°  "
              f"(range {out['min_10m']:.1f}–{out['max_10m']:.1f} m/s over box)")
        print(f"  midflame  : {out['midflame_ms']:.1f} m/s   (WAF {waf})")
    return out


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else '2024-08-15 21:00'
    fetch_hrrr_wind(d)
