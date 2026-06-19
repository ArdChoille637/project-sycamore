"""
Terrain-downscaled wind — mass-conserving diagnostic (the core of WindNinja).

A spatially-uniform ambient wind (e.g. one HRRR value) is corrected so the flow
conserves mass over the real terrain: it accelerates where the flow layer is
squeezed thin over ridges and decelerates / steers around valleys. This is the
standard diagnostic ("conservation of mass") wind model:

    treat the wind as a depth-averaged flow in a layer of depth  d = H_top − z(x,y).
    Mass conservation:        ∇·(d V) = 0
    Helmholtz correction:     V = V0 − ∇φ
    →  solve the variable-coefficient Poisson eq.   ∇·(d∇φ) = ∇·(d V0)
       with φ = 0 on the boundary (ambient wind at the domain edge).

Solved here with conjugate gradient (the operator is symmetric positive-definite).
It is a stencil iteration — an MLX/Metal candidate for large grids — but kept in
NumPy here since it is a one-time precompute, not the CA hot loop.

    from terrain_wind import downscale_wind
    speed, from_deg = downscale_wind(z, DX, U0_ms=3.0, from_deg=256.0)
"""

import numpy as np

# neighbour shifts with Dirichlet (zero) fill outside the domain
def _E(a): r = np.zeros_like(a); r[:, :-1] = a[:, 1:]; return r   # a[i, j+1]
def _W(a): r = np.zeros_like(a); r[:, 1:] = a[:, :-1]; return r   # a[i, j-1]
def _N(a): r = np.zeros_like(a); r[:-1, :] = a[1:, :]; return r   # a[i+1, j]
def _S(a): r = np.zeros_like(a); r[1:, :] = a[:-1, :]; return r   # a[i-1, j]


def _faces(d):
    """Face-averaged flow depth (edge-replicated at the domain border)."""
    dE = 0.5*(d + np.pad(d, ((0, 0), (0, 1)), mode='edge')[:, 1:])
    dW = 0.5*(d + np.pad(d, ((0, 0), (1, 0)), mode='edge')[:, :-1])
    dN = 0.5*(d + np.pad(d, ((0, 1), (0, 0)), mode='edge')[1:, :])
    dS = 0.5*(d + np.pad(d, ((1, 0), (0, 0)), mode='edge')[:-1, :])
    return dE, dW, dN, dS


def _applyA(phi, dE, dW, dN, dS, DX2):
    """A φ = −∇·(d∇φ)  (SPD with Dirichlet φ=0); border rows/cols held at 0."""
    L = (dE*(_E(phi) - phi) - dW*(phi - _W(phi))
         + dN*(_N(phi) - phi) - dS*(phi - _S(phi))) / DX2
    A = -L
    A[0, :] = 0; A[-1, :] = 0; A[:, 0] = 0; A[:, -1] = 0
    return A


def downscale_wind(z, DX, U0_ms, from_deg, H_layer=None, tol=1e-6, max_iter=4000,
                   verbose=False):
    """Mass-conserving terrain wind. Returns (speed_field, from_deg_field)."""
    z = np.asarray(z, 'f8')
    relief = float(z.max() - z.min())
    if H_layer is None:
        # terrain-influenced flow-layer depth above the highest peak; a few hundred
        # metres over complex terrain (thinner layer → stronger ridge speed-up).
        H_layer = max(300.0, 0.4*relief)
    d = (z.max() + H_layer) - z                    # > 0 everywhere
    dzdy, dzdx = np.gradient(z, DX)                # ∂z/∂North(+i), ∂z/∂East(+j)

    to = np.deg2rad(from_deg + 180.0)              # direction the wind blows toward
    u0 = U0_ms*np.sin(to)                          # East
    v0 = U0_ms*np.cos(to)                          # North

    # RHS = ∇·(d V0) = u0 ∂d/∂x + v0 ∂d/∂y = −(u0 dzdx + v0 dzdy);  solve Aφ = b, A=−L, b=−RHS
    b = (u0*dzdx + v0*dzdy)
    b[0, :] = 0; b[-1, :] = 0; b[:, 0] = 0; b[:, -1] = 0
    DX2 = DX*DX
    dE, dW, dN, dS = _faces(d)

    # conjugate gradient
    phi = np.zeros_like(z)
    r = b - _applyA(phi, dE, dW, dN, dS, DX2)
    p = r.copy()
    rs = float(np.sum(r*r))
    b_norm = max(float(np.sqrt(np.sum(b*b))), 1e-30)
    it = 0
    for it in range(1, max_iter+1):
        Ap = _applyA(p, dE, dW, dN, dS, DX2)
        denom = float(np.sum(p*Ap))
        if abs(denom) < 1e-30:
            break
        alpha = rs/denom
        phi += alpha*p
        r -= alpha*Ap
        rs_new = float(np.sum(r*r))
        if np.sqrt(rs_new)/b_norm < tol:
            break
        p = r + (rs_new/rs)*p
        rs = rs_new

    # corrected wind  V = V0 − ∇φ
    dphidy, dphidx = np.gradient(phi, DX)
    u = u0 - dphidx
    v = v0 - dphidy
    speed = np.hypot(u, v)
    from_deg_field = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    if verbose:
        # mass-conservation residual of the corrected field (interior)
        flux = _div(d*u, d*v, DX)[1:-1, 1:-1]
        print(f"  CG {it} iters, residual {np.sqrt(rs_new)/b_norm:.1e}; "
              f"speed {speed.min():.1f}–{speed.max():.1f} m/s (ambient {U0_ms:.1f}); "
              f"|∇·(dV)| max {np.abs(flux).max():.2e}")
    return speed.astype('f4'), from_deg_field.astype('f4')


def _div(fx, fy, DX):
    dfydy, _ = np.gradient(fy, DX)
    _, dfxdx = np.gradient(fx, DX)
    return dfxdx + dfydy


def make_wind_terrain(U0_ms, from_deg, **kw):
    """Closure for rc.simulate(wind_fn=...): (z,dzdx,dzdy,DX) → (speed, from_deg) fields."""
    def _fn(z, dzdx, dzdy, DX):
        return downscale_wind(z, DX, U0_ms, from_deg, **kw)
    return _fn


# ── self-test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    H, W, DX = 120, 160, 30.0
    # 1) flat terrain → field must equal ambient
    zf = np.zeros((H, W))
    s, f = downscale_wind(zf, DX, 5.0, 270.0)
    print(f"flat: speed {s.min():.2f}–{s.max():.2f} (expect 5.00), "
          f"dir {f.min():.1f}–{f.max():.1f} (expect 270)")
    assert abs(s.mean()-5.0) < 1e-3 and np.ptp(s) < 1e-2, "flat terrain must be unchanged"

    # 2) a ridge across the flow → speed-up over the crest, mass conserved
    y, x = np.mgrid[0:H, 0:W]
    z = 400.0*np.exp(-(((x-W/2)/(0.10*W))**2))          # N–S ridge
    s, f = downscale_wind(z, DX, 5.0, 270.0, verbose=True)
    crest = s[:, W//2].mean(); upstream = s[:, 5].mean()
    print(f"ridge: crest speed {crest:.2f} vs upstream {upstream:.2f} m/s  "
          f"→ {'PASS' if crest > upstream else 'FAIL'}: wind accelerates over ridge")
