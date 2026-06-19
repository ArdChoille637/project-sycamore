"""
Real terrain for the Rothermel CA — USGS 3DEP elevation.

Pulls a bare-earth DEM tile straight from the USGS 3DEP ImageServer REST endpoint,
sized to the CA grid (H×W at DX-metre pixels) and projected to the local UTM zone
so the grid is locally Euclidean (East = +col, North = +row, origin lower — the
same frame the CA and HRRR wind use). No GDAL/rasterio needed: pyproj for the
projection, PIL to read the float32 GeoTIFF. Result is cached per (lat,lon,grid).

Drop-in for rc.simulate's `terrain_fn`:
    from real_terrain import make_terrain_3dep
    rc.simulate(..., terrain_fn=make_terrain_3dep(39.0, -121.0))
"""

import os
import io
import numpy as np
import requests
from pyproj import Transformer

DEM_URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
           "3DEPElevation/ImageServer/exportImage")


def _utm_epsg(lat0, lon0):
    zone = int((lon0 + 180) / 6) + 1
    return (32600 if lat0 >= 0 else 32700) + zone


def fetch_3dep_dem(H, W, DX, lat0, lon0, cache_dir='/tmp/sycamore_dem', verbose=True):
    """Return a (H,W) float32 elevation array [m], origin lower (row 0 = south)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"dem_{lat0:.4f}_{lon0:.4f}_{H}x{W}_{DX:.0f}.npy")
    if os.path.exists(path):
        z = np.load(path)
        if verbose:
            print(f"3DEP DEM (cached) {z.shape}  {np.nanmin(z):.0f}–{np.nanmax(z):.0f} m")
        return z

    from PIL import Image
    epsg = _utm_epsg(lat0, lon0)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x0, y0 = tf.transform(lon0, lat0)
    xmin, xmax = x0 - W*DX/2, x0 + W*DX/2
    ymin, ymax = y0 - H*DX/2, y0 + H*DX/2
    params = dict(bbox=f"{xmin},{ymin},{xmax},{ymax}", bboxSR=epsg, imageSR=epsg,
                  size=f"{W},{H}", format="tiff", pixelType="F32",
                  interpolation="RSP_BilinearInterpolation",
                  adjustAspectRatio="false", f="image")
    r = requests.get(DEM_URL, params=params, timeout=90)
    r.raise_for_status()
    if 'tiff' not in r.headers.get('Content-Type', ''):
        raise RuntimeError(f"3DEP returned {r.headers.get('Content-Type')}: {r.content[:200]}")

    z = np.flipud(np.array(Image.open(io.BytesIO(r.content)), dtype='f4')).copy()
    z[z < -1e4] = np.nan                      # NoData sentinel
    if np.isnan(z).any():
        z[np.isnan(z)] = np.nanmedian(z)
    np.save(path, z)
    if verbose:
        print(f"3DEP DEM EPSG:{epsg}  {z.shape}  {z.min():.0f}–{z.max():.0f} m  → cached")
    return z


def make_terrain_3dep(lat0, lon0, verbose=True):
    """Closure matching make_terrain(H,W,DX) → (z, dzdx, dzdy)."""
    def _fn(H, W, DX):
        z = fetch_3dep_dem(H, W, DX, lat0, lon0, verbose=verbose)
        dzdy, dzdx = np.gradient(z, DX)       # ∂z/∂North(+row), ∂z/∂East(+col)
        return z.astype('f4'), dzdx.astype('f4'), dzdy.astype('f4')
    return _fn


if __name__ == '__main__':
    z = fetch_3dep_dem(220, 320, 25.0, 39.0, -121.0)
    dzdy, dzdx = np.gradient(z, 25.0)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    print(f"relief {z.min():.0f}–{z.max():.0f} m, slope median {np.median(slope):.1f}°, "
          f"max {slope.max():.1f}°")
