"""
Spatial dead-fuel moisture — HRRR weather × terrain microclimate.

Fine dead-fuel moisture (the Rothermel `M`) is set by the air it equilibrates
with. The ambient comes from real HRRR 2 m temperature + dewpoint (via Herbie);
terrain then modulates it two well-established ways:

  * elevation lapse — air cools ~6.5 K/km with height → higher RH → wetter fuel up high.
  * aspect heat load — sun-facing (SW) slopes run hotter/drier; shaded (NE) slopes
    stay cooler/wetter (a McCune & Keon 2002–style heat-load offset).

Equilibrium moisture content via Simard (1968); RH from temperature + dewpoint by
the Magnus formula. Where the result exceeds a fuel's moisture of extinction, that
cell simply won't carry fire — an emergent "wet firebreak" on shaded high ground.

    from fuel_moisture import make_moisture_hrrr
    rc.simulate(..., fuel_fn=..., moisture_fn=make_moisture_hrrr(date, lat, lon))
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from herbie import Herbie

LAPSE     = 6.5e-3      # environmental temperature lapse [K/m]
DEW_LAPSE = 2.0e-3      # dewpoint lapse [K/m]
HEAT_AMP  = 4.0        # max aspect heat-load temperature swing [°C] on steep SW vs NE


def _esat(Tc):
    """Saturation vapour pressure [hPa] (Magnus)."""
    return 6.112*np.exp(17.67*Tc/(Tc + 243.5))


def emc_simard(T_F, RH):
    """Equilibrium moisture content [%] (Simard 1968); T in °F, RH in %."""
    RH = np.clip(RH, 0.0, 100.0)
    emc = np.where(
        RH < 10.0, 0.03229 + 0.281073*RH - 0.000578*RH*T_F,
        np.where(RH <= 50.0, 2.22749 + 0.160107*RH - 0.014784*T_F,
                 21.0606 + 0.005565*RH**2 - 0.00035*RH*T_F - 0.483199*RH))
    return np.clip(emc, 1.0, None)


def moisture_field(z, dzdx, dzdy, DX, T_base_K, Td_base_K, M_floor=0.02, M_cap=0.35):
    """Spatial fine dead-fuel moisture [fraction] from ambient T/Td + terrain."""
    z = np.asarray(z, 'f8')
    z_ref = float(z.mean())
    Tc  = (T_base_K - 273.15) - LAPSE*(z - z_ref)         # air temp, elevation-lapsed [°C]
    Tdc = (Td_base_K - 273.15) - DEW_LAPSE*(z - z_ref)    # dewpoint, lapsed [°C]

    slope  = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0  # downslope azimuth (compass)
    heat   = HEAT_AMP*np.sin(slope)*np.cos(np.radians(aspect - 225.0))   # +SW … −NE [°C]
    Tc_fuel = Tc + heat                                    # fuel-surface microclimate temp

    Tdc = np.minimum(Tdc, Tc_fuel)                         # dewpoint ≤ temperature
    RH = np.clip(100.0*_esat(Tdc)/_esat(Tc_fuel), 1.0, 100.0)
    emc = emc_simard(Tc_fuel*9.0/5.0 + 32.0, RH)
    return np.clip(emc/100.0, M_floor, M_cap).astype('f4')


def fetch_hrrr_trh(date, lat0, lon0, halfdeg=0.4, fxx=0, save_dir='/tmp/herbie', verbose=True):
    """Domain-mean HRRR 2 m temperature + dewpoint [K]."""
    H = Herbie(date, model='hrrr', product='sfc', fxx=fxx, save_dir=save_dir, verbose=False)
    if H.grib is None:
        raise RuntimeError(f"No HRRR data for {date}")
    ds = H.xarray(":(TMP|DPT):2 m above ground:", verbose=False)
    lat = ds.latitude.values; lon = ds.longitude.values; lonc = lon0 % 360
    box = (np.abs(lat - lat0) <= halfdeg) & (np.abs(((lon - lonc + 180) % 360) - 180) <= halfdeg)
    T_K = float(ds.t2m.values[box].mean())
    Td_K = float(ds.d2m.values[box].mean())
    if verbose:
        print(f"HRRR 2 m @ ({lat0:.2f},{lon0:.2f}): T {T_K-273.15:.1f}°C, "
              f"Td {Td_K-273.15:.1f}°C, RH≈{100*_esat(Td_K-273.15)/_esat(T_K-273.15):.0f}%")
    return T_K, Td_K


def make_moisture_hrrr(date, lat0, lon0, verbose=True):
    """Closure for rc.simulate(moisture_fn=...)."""
    T_K, Td_K = fetch_hrrr_trh(date, lat0, lon0, verbose=verbose)

    def _fn(z, dzdx, dzdy, DX):
        return moisture_field(z, dzdx, dzdy, DX, T_K, Td_K)
    return _fn


# ── self-test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    H, W, DX = 120, 160, 30.0
    T_K, Td_K = 296.4, 275.1                               # the Eldorado fire-weather afternoon
    # flat → uniform
    zf = np.zeros((H, W)); g = np.zeros((H, W))
    Mf = moisture_field(zf, g, g, DX, T_K, Td_K)
    print(f"flat: M {Mf.min()*100:.2f}–{Mf.max()*100:.2f}% (uniform) → "
          f"{'PASS' if np.ptp(Mf) < 1e-4 else 'FAIL'}")
    # a ridge: compare a S-facing vs N-facing slope cell and high vs low elevation
    y, x = np.mgrid[0:H, 0:W]
    z = 500.0*np.exp(-(((y-H/2)/(0.18*H))**2))            # E–W ridge → N & S faces
    dzdy, dzdx = np.gradient(z, DX)
    M = moisture_field(z, dzdx, dzdy, DX, T_K, Td_K)
    # origin lower → high-row side faces North (shaded); low-row side faces South (sunny)
    n_face = M[H//2+18, W//2]*100      # N-facing slope, shaded
    s_face = M[H//2-18, W//2]*100      # S-facing slope, sunny
    print(f"aspect: N-face {n_face:.2f}% vs S-face {s_face:.2f}%  → "
          f"{'PASS' if n_face > s_face else 'FAIL'}: shaded fuel is moister")
    print(f"elevation: crest {M[H//2, W//2]*100:.2f}% vs base {M[5, W//2]*100:.2f}%")
