"""
Real fuel for the Rothermel CA — LANDFIRE FBFM40 (40 Scott & Burgan 2005 models).

Pulls the FBFM40 fuel-model raster from the LANDFIRE ImageServer (nearest-neighbour,
since the values are class codes) and maps each code to a single-class-equivalent
Rothermel fuel (w0 total dead load, σ characteristic dead SAV, δ fuel bed depth,
Mx dead moisture of extinction, h heat content). Nonburnable classes (urban,
water, agriculture, snow, barren) become firebreaks (no spread).

This makes every Rothermel fuel input spatial instead of one uniform model.

    from real_fuel import make_fuel_fbfm40
    rc.simulate(..., fuel_fn=make_fuel_fbfm40(39.0, -121.0))

Param table: single-class reduction of the multi-size-class Scott & Burgan models
(total dead load + characteristic dead SAV), consistent with this CA's single-class
formulation. Loads converted ton/acre → lb/ft² (×0.0459).
"""

import os
import io
import numpy as np
import requests
from pyproj import Transformer

FBFM40_URL = ("https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/"
              "LF2024_FBFM40_CONUS/ImageServer/exportImage")

TON_AC_TO_LB_FT2 = 2000.0 / 43560.0          # 0.04591
NONBURNABLE = {91, 92, 93, 98, 99}           # NB1 urban, NB2 snow, NB3 ag, NB8 water, NB9 barren

# Raw Scott & Burgan (2005, RMRS-GTR-153) FBFM40 parameters, cross-checked against
# the primary PDF (characteristic SAV, dead extinction moisture, fine-fuel loads,
# codes, constants verified directly; size-class loads from the canonical standard
# set, internally consistent with the verified fine-fuel-loads).
#   row = (name, load_1hr, load_10hr, load_100hr, load_live_herb, depth_ft,
#          sav_characteristic[1/ft], mx_dead[%], heat[BTU/lb])   loads in ton/acre
FBFM40_RAW = {
    101: ("GR1", 0.10, 0.00, 0.00, 0.30, 0.4, 2054, 15, 8000),
    102: ("GR2", 0.10, 0.00, 0.00, 1.00, 1.0, 1820, 15, 8000),
    103: ("GR3", 0.10, 0.40, 0.00, 1.50, 2.0, 1290, 30, 8000),
    104: ("GR4", 0.25, 0.00, 0.00, 1.90, 2.0, 1826, 15, 8000),
    105: ("GR5", 0.40, 0.00, 0.00, 2.50, 1.5, 1631, 40, 8000),
    106: ("GR6", 0.10, 0.00, 0.00, 3.40, 1.5, 2006, 40, 9000),
    107: ("GR7", 1.00, 0.00, 0.00, 5.40, 3.0, 1834, 15, 8000),
    108: ("GR8", 0.50, 1.00, 0.00, 7.30, 4.0, 1302, 30, 8000),
    109: ("GR9", 1.00, 1.00, 0.00, 9.00, 5.0, 1612, 40, 8000),
    121: ("GS1", 0.20, 0.00, 0.00, 0.50, 0.9, 1832, 15, 8000),
    122: ("GS2", 0.50, 0.50, 0.00, 0.60, 1.5, 1827, 15, 8000),
    123: ("GS3", 0.30, 0.25, 0.00, 1.45, 1.8, 1614, 40, 8000),
    124: ("GS4", 1.90, 0.30, 0.10, 3.40, 2.1, 1674, 40, 8000),
    141: ("SH1", 0.25, 0.25, 0.00, 0.15, 1.0, 1674, 15, 8000),
    142: ("SH2", 1.35, 2.40, 0.75, 0.00, 1.0, 1672, 15, 8000),
    143: ("SH3", 0.45, 3.00, 0.00, 0.00, 2.4, 1371, 40, 8000),
    144: ("SH4", 0.85, 1.15, 0.20, 0.00, 3.0, 1682, 30, 8000),
    145: ("SH5", 3.60, 2.10, 0.00, 0.00, 6.0, 1252, 15, 8000),
    146: ("SH6", 2.90, 1.45, 0.00, 0.00, 2.0, 1144, 30, 8000),
    147: ("SH7", 3.50, 5.30, 2.20, 0.00, 6.0, 1233, 15, 8000),
    148: ("SH8", 2.05, 3.40, 0.85, 0.00, 3.0, 1386, 40, 8000),
    149: ("SH9", 4.50, 2.45, 0.00, 1.55, 4.4, 1378, 40, 8000),
    161: ("TU1", 0.20, 0.90, 1.50, 0.20, 0.6, 1606, 20, 8000),
    162: ("TU2", 0.95, 1.80, 1.25, 0.00, 1.0, 1767, 30, 8000),
    163: ("TU3", 1.10, 0.15, 0.25, 0.65, 1.3, 1611, 30, 8000),
    164: ("TU4", 4.50, 0.00, 0.00, 0.00, 0.5, 2216, 12, 8000),
    165: ("TU5", 4.00, 4.00, 3.00, 0.00, 1.0, 1224, 25, 8000),
    181: ("TL1", 1.00, 2.20, 3.60, 0.00, 0.2, 1716, 30, 8000),
    182: ("TL2", 1.40, 2.30, 2.20, 0.00, 0.2, 1806, 25, 8000),
    183: ("TL3", 0.50, 2.20, 2.80, 0.00, 0.3, 1532, 20, 8000),
    184: ("TL4", 0.50, 1.50, 4.20, 0.00, 0.4, 1568, 25, 8000),
    185: ("TL5", 1.15, 2.50, 4.40, 0.00, 0.6, 1713, 25, 8000),
    186: ("TL6", 2.40, 1.20, 1.20, 0.00, 0.3, 1936, 25, 8000),
    187: ("TL7", 0.30, 1.40, 8.10, 0.00, 0.4, 1229, 25, 8000),
    188: ("TL8", 5.80, 1.40, 1.10, 0.00, 0.3, 1770, 35, 8000),
    189: ("TL9", 6.65, 3.30, 4.15, 0.00, 0.6, 1733, 35, 8000),
    201: ("SB1", 1.50, 3.00, 11.0, 0.00, 1.0, 1653, 25, 8000),
    202: ("SB2", 4.50, 4.25, 4.00, 0.00, 1.0, 1884, 25, 8000),
    203: ("SB3", 5.50, 2.75, 3.00, 0.00, 1.2, 1935, 25, 8000),
    204: ("SB4", 5.25, 3.50, 5.25, 0.00, 2.7, 1907, 25, 8000),
}

# Single-class reduction for this CA's single-fuel formulation:
#   w0    = (dead 1+10+100-hr + cured live-herb) — fire-season full-curing assumption;
#           live woody stays live and is omitted (the single-class dead model can't
#           carry a separate live class). Converted ton/ac → lb/ft².
#   sigma = published load-weighted CHARACTERISTIC SAV (the right single value).
#   delta = fuel bed depth; Mx = dead extinction moisture; h = heat content.
FBFM40_PARAMS = {}
FBFM40_NAMES = {}
for _code, (_nm, _l1, _l10, _l100, _lh, _dp, _sav, _mx, _ht) in FBFM40_RAW.items():
    FBFM40_PARAMS[_code] = dict(
        w0=(_l1 + _l10 + _l100 + _lh) * TON_AC_TO_LB_FT2,
        sigma=float(_sav), delta=float(_dp), Mx=_mx/100.0, h=float(_ht))
    FBFM40_NAMES[_code] = _nm

# Fallback for a burnable code missing from the table: a moderate grass-shrub.
_DEFAULT = dict(w0=0.10, sigma=1800.0, delta=1.0, Mx=0.20, h=8000.0)


def _utm_epsg(lat0, lon0):
    zone = int((lon0 + 180) / 6) + 1
    return (32600 if lat0 >= 0 else 32700) + zone


def fetch_fbfm40(H, W, DX, lat0, lon0, cache_dir='/tmp/sycamore_fuel', verbose=True):
    """Return (H,W) int FBFM40 code raster, origin lower (row 0 = south)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"fbfm40_{lat0:.4f}_{lon0:.4f}_{H}x{W}_{DX:.0f}.npy")
    if os.path.exists(path):
        c = np.load(path)
        if verbose:
            print(f"FBFM40 fuel (cached) {c.shape}")
        return c

    from PIL import Image
    epsg = _utm_epsg(lat0, lon0)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = tf.transform(lon0, lat0)
    xmin, xmax = x0 - W*DX/2, x0 + W*DX/2
    ymin, ymax = y0 - H*DX/2, y0 + H*DX/2
    params = dict(bbox=f"{xmin},{ymin},{xmax},{ymax}", bboxSR=epsg, imageSR=epsg,
                  size=f"{W},{H}", format="tiff", pixelType="U16",
                  interpolation="RSP_NearestNeighbor",      # classified → nearest
                  adjustAspectRatio="false", f="image")
    r = requests.get(FBFM40_URL, params=params, timeout=90)
    r.raise_for_status()
    if 'tiff' not in r.headers.get('Content-Type', ''):
        raise RuntimeError(f"LANDFIRE returned {r.headers.get('Content-Type')}: {r.content[:200]}")
    code = np.flipud(np.array(Image.open(io.BytesIO(r.content)))).astype(np.int32).copy()
    np.save(path, code)
    if verbose:
        print(f"FBFM40 fuel EPSG:{epsg}  {code.shape}  → cached")
    return code


def codes_to_fuel(code, verbose=True):
    """Map an FBFM40 code raster to per-cell Rothermel param arrays + burnable mask."""
    H, W = code.shape
    w0 = np.empty((H, W), 'f4'); delta = np.empty((H, W), 'f4')
    sigma = np.empty((H, W), 'f4'); h = np.empty((H, W), 'f4'); Mx = np.empty((H, W), 'f4')
    burnable = np.ones((H, W), bool)
    unknown = set()
    for c in np.unique(code):
        m = code == c
        ci = int(c)
        if ci in NONBURNABLE:
            p = _DEFAULT;  burnable[m] = False
        elif ci in FBFM40_PARAMS:
            p = FBFM40_PARAMS[ci]
        else:
            p = _DEFAULT;  unknown.add(ci)             # unknown burnable → moderate default
        w0[m] = p['w0']; delta[m] = p['delta']; sigma[m] = p['sigma']
        h[m] = p['h']; Mx[m] = p['Mx']
    if verbose and unknown:
        print(f"  note: codes {sorted(unknown)} not in table → moderate default")
    return dict(w0=w0, delta=delta, sigma=sigma, h=h, Mx=Mx, burnable=burnable, code=code)


def make_fuel_fbfm40(lat0, lon0, verbose=True):
    """Closure for rc.simulate(fuel_fn=...)."""
    def _fn(H, W, DX):
        return codes_to_fuel(fetch_fbfm40(H, W, DX, lat0, lon0, verbose=verbose), verbose=verbose)
    return _fn


def fuel_summary(code):
    """Human-readable composition of a fuel raster."""
    vals, counts = np.unique(code, return_counts=True)
    tot = code.size
    out = []
    for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: -t[1]):
        name = FBFM40_NAMES.get(int(v), 'nonburnable' if int(v) in NONBURNABLE else f'code {v}')
        out.append((int(v), name, 100*c/tot))
    return out
