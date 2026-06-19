"""
Project Sycamore — D2Q9 Lattice-Boltzmann CFD on Apple-Silicon (MLX) / NumPy.

A native GPU CFD path for the samara / drone aerodynamics, in place of OpenFOAM
in a VM. LBM is an explicit, stencil-based, FP32 method — streaming is an array
roll, collision is element-wise — exactly the workload Apple's GPU accelerates
well (the same shape as this project's fire CA and terrain-wind Poisson solve).

Scheme: D2Q9, BGK (single-relaxation) collision, full-way bounce-back for solid
walls/obstacles, momentum-corrected bounce-back for moving walls (the cavity lid),
Zou/He-style velocity inlet + open outlet for external flow.

Validated against the Ghia et al. (1982) lid-driven cavity benchmark (see
lbm_validate.py). Backend selected at runtime: MLX (Metal GPU) when available.

Lattice units throughout (Δx = Δt = 1, c_s² = 1/3).
"""

import numpy as np

try:
    import mlx.core as mx
    HAVE_MLX = True
except Exception:                                        # pragma: no cover
    mx = None
    HAVE_MLX = False

# ── D2Q9 lattice ────────────────────────────────────────────────────────────────
#   index:  0      1     2     3      4      5      6      7      8
#   dir  :  rest   E     N     W      S      NE     NW     SW     SE
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], 'i4')
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], 'i4')
W9 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], 'f8')
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], 'i4')        # opposite direction (bounce-back)


class _BE:
    """Minimal NumPy/MLX shim."""
    def __init__(self, use_mlx):
        self.mlx = use_mlx and HAVE_MLX
        self.xp = mx if self.mlx else np
    def arr(self, a): return mx.array(np.asarray(a, 'f4')) if self.mlx else np.asarray(a, 'f4')
    def to_np(self, a): return np.array(a) if self.mlx else np.asarray(a)
    def eval(self, *a):
        if self.mlx: mx.eval(*a)
    def roll2(self, a, sx, sy):
        """Stream: shift field by (sx along x=axis1, sy along y=axis0)."""
        if self.mlx:
            return mx.roll(mx.roll(a, sy, axis=0), sx, axis=1)
        return np.roll(np.roll(a, sy, axis=0), sx, axis=1)


def equilibrium(be, rho, ux, uy):
    """D2Q9 equilibrium distribution feq (9, ny, nx)."""
    xp = be.xp
    usqr = 1.5*(ux*ux + uy*uy)
    feq = []
    for i in range(9):
        cu = 3.0*(CX[i]*ux + CY[i]*uy)
        feq.append(float(W9[i]) * rho * (1.0 + cu + 0.5*cu*cu - usqr))
    return xp.stack(feq, axis=0)


def macroscopic(be, f):
    """Density and velocity from the populations."""
    xp = be.xp
    rho = xp.sum(f, axis=0)
    ux = xp.sum(f * be.arr(CX.reshape(9, 1, 1).astype('f4')), axis=0) / rho
    uy = xp.sum(f * be.arr(CY.reshape(9, 1, 1).astype('f4')), axis=0) / rho
    return rho, ux, uy


class LBM:
    """D2Q9 BGK lattice-Boltzmann solver with bounce-back solids."""

    def __init__(self, ny, nx, nu, use_mlx=True):
        self.be = _BE(use_mlx)
        self.ny, self.nx = ny, nx
        self.nu = nu
        self.tau = 3.0*nu + 0.5
        self.omega = 1.0/self.tau
        self.solid = np.zeros((ny, nx), bool)            # all bounce-back nodes (walls + obstacle)
        self.obstacle = np.zeros((ny, nx), bool)         # obstacle only (force is measured here)
        self.uw = np.zeros((2, ny, nx), 'f4')            # wall velocity (moving solids)
        self.inlet_u = None                              # left Zou/He velocity inlet
        self.feq_in = None
        self.outlet = False                              # right zero-gradient outlet
        self._links = None                               # cached obstacle boundary links
        self._bc = None                                  # cached backend solid/wall arrays
        # initialise at rest, rho = 1
        be = self.be
        rho0 = be.arr(np.ones((ny, nx)))
        z = be.arr(np.zeros((ny, nx)))
        self.f = equilibrium(be, rho0, z, z)
        be.eval(self.f)

    # ── boundary configuration helpers ──
    def add_walls(self, top=False, bottom=False, left=False, right=False):
        if top:    self.solid[-1, :] = True
        if bottom: self.solid[0, :] = True
        if left:   self.solid[:, 0] = True
        if right:  self.solid[:, -1] = True
        self._bc = None

    def set_moving_wall(self, mask, ux, uy=0.0):
        self.solid |= mask
        self.uw[0][mask] = ux
        self.uw[1][mask] = uy
        self._bc = None

    def set_obstacle(self, mask):
        self.solid |= mask
        self.obstacle |= mask
        self._bc = None

    def set_inlet(self, u):
        """Left-column equilibrium-velocity inlet: u_x = u, u_y = 0, rho = 1."""
        self.inlet_u = float(u)
        usqr = 1.5*u*u
        self.feq_in = np.array([W9[i]*(1.0 + 3.0*CX[i]*u + 4.5*(CX[i]*u)**2 - usqr)
                                for i in range(9)], 'f4')

    def set_outlet(self):
        """Right-column zero-gradient (open) outlet."""
        self.outlet = True

    # ── one time step ──
    def step(self):
        be = self.be; xp = be.xp
        f = self.f
        rho, ux, uy = macroscopic(be, f)

        feq = equilibrium(be, rho, ux, uy)
        fout = f - self.omega*(f - feq)

        # full-way bounce-back on solids, with moving-wall momentum correction.
        # The masks are static during a run → build the backend arrays once and cache
        # them (rebuilt only when walls/obstacle/moving-wall config changes).
        if self._bc is None:
            self._bc = (be.arr(self.solid.astype('f4')), be.arr(self.uw[0]), be.arr(self.uw[1]))
        solid, uwx, uwy = self._bc
        parts = []
        for i in range(9):
            bb = _gather_opp(f, i) + 6.0*float(W9[i])*(CX[i]*uwx + CY[i]*uwy)
            parts.append(xp.where(solid > 0.5, bb, fout[i]))
        fout = xp.stack(parts, axis=0)

        # streaming
        f = xp.stack([be.roll2(fout[i], int(CX[i]), int(CY[i])) for i in range(9)], axis=0)

        # outlet: zero-gradient (copy second-to-last column into the last)
        if self.outlet:
            shifted = xp.roll(f, 1, axis=2)              # col j ← col j-1; last col ← col -2
            f = xp.where(self._cmask(self.nx-1, nd=True), shifted, f)
        # inlet: Zou/He velocity BC at the left column (enforces u_x=U, u_y=0
        # and the consistent density — a hard BC, unlike the soft equilibrium one)
        if self.inlet_u is not None:
            f = self._zou_he_inlet(f)

        self.f = f
        be.eval(self.f)

    def _zou_he_inlet(self, f):
        be = self.be; xp = be.xp; U = self.inlet_u
        f0 = f[0, :, 0:1]; f2 = f[2, :, 0:1]; f3 = f[3, :, 0:1]
        f4 = f[4, :, 0:1]; f6 = f[6, :, 0:1]; f7 = f[7, :, 0:1]      # known (ny,1)
        rho = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - U)
        f1 = f3 + (2.0/3.0)*rho*U                                   # unknown incoming pops
        f5 = f7 + (1.0/6.0)*rho*U - 0.5*(f2 - f4)
        f8 = f6 + (1.0/6.0)*rho*U + 0.5*(f2 - f4)
        cond = self._cmask(0)                                       # (1,nx)
        new1 = xp.where(cond, f1, f[1]); new5 = xp.where(cond, f5, f[5])
        new8 = xp.where(cond, f8, f[8])
        return xp.stack([f[0], new1, f[2], f[3], f[4], new5, f[6], f[7], new8], axis=0)

    def _cmask(self, j, nd=False):
        """Column-j selector. nd=True → (1,1,nx) for the (9,ny,nx) population array."""
        m = (np.arange(self.nx) == j).astype('f4')
        return self.be.arr(m.reshape(1, 1, -1) if nd else m.reshape(1, -1)) > 0.5

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def velocity(self):
        rho, ux, uy = macroscopic(self.be, self.f)
        ux = self.be.to_np(ux); uy = self.be.to_np(uy)
        m = self.solid                                   # solid nodes carry the wall velocity
        ux[m] = self.uw[0][m]; uy[m] = self.uw[1][m]
        return ux, uy

    def vorticity(self):
        ux, uy = self.velocity()
        duy_dx = np.gradient(uy, axis=1)
        dux_dy = np.gradient(ux, axis=0)
        return duy_dx - dux_dy

    def _build_links(self):
        """Cache the fluid→OBSTACLE boundary links for the momentum-exchange force
        (only the obstacle, not the domain walls)."""
        self._links = []
        for i in range(1, 9):
            ns = np.roll(np.roll(self.obstacle, -int(CY[i]), axis=0), -int(CX[i]), axis=1)
            link = (~self.solid) & ns                    # fluid node whose i-neighbour is the obstacle
            if link.any():
                self._links.append((i, self.be.arr(link.astype('f4'))))

    def force_on_solid(self):
        """Momentum-exchange force (Fx, Fy) on the obstacle (lattice units).
        Mei, Yu, Shyy & Luo (2002): F = Σ_links c_i (f_i + f_{ī}) at fluid boundary nodes."""
        if self._links is None:
            self._build_links()
        be = self.be; xp = be.xp; f = self.f
        sums = [xp.sum((f[i] + f[int(OPP[i])]) * lm) for i, lm in self._links]
        Fx = sum(int(CX[i]) * s for (i, _), s in zip(self._links, sums))
        Fy = sum(int(CY[i]) * s for (i, _), s in zip(self._links, sums))
        be.eval(Fx, Fy)
        return float(be.to_np(Fx)), float(be.to_np(Fy))


def _gather_opp(f, i):
    """f[OPP[i]] for both backends (f is (9,ny,nx))."""
    return f[int(OPP[i])]
