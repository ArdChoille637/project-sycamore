"""
Mass properties & rotational stability for the samara device — Project Sycamore
Phase-1 Wave 2.

This is the half of the §6 Phase-1 mandate the aero model alone does not cover:
"... nose-prop weight ratio, and mass distribution for STABLE autorotation ...".
The steady BEM (samara_bem.py) shows an autorotation equilibrium EXISTS; this
module asks whether it is STABLE and how the mass layout (wing vs hub, and the
nose-prop weight ratio) governs that.

Model (rigid body, tier: Indicative — exact rigid-body integrals on a
parameterised, not-yet-bench-measured mass breakdown):
  - Wing mass distributed along the span ∝ chord c(r) (reuses the aero grid).
  - Hub assembly on/near the spin axis (r≈0), split into:
        nose-prop + motor   (mass ABOVE the rotor plane, z = +z_prop)
        decoupled core       (sensors/battery BELOW the plane, z = −z_core)
        seed/structure       (in the plane)
Radial balance is a hard requirement for any spinning craft, so the spin axis is
taken through the CG (r_cg≈0); the wing's radial first moment is reported as the
counterweight the hub/seed must supply.

Stability screen (rigid body):
  - Max-inertia spin margin  I_zz / I_t  (≤1). By the perpendicular-axis theorem
    a PLANAR lamina spun about its normal has I_zz = I_x + I_y, i.e. I_zz = I_t —
    the spin axis is the maximum-inertia axis (the energetically/gyroscopically
    favourable spin). Axial hub mass (prop above, core below) adds a Σm·z² term
    to the transverse inertia, making I_t > I_zz and pushing the spin onto the
    INTERMEDIATE axis — rigid-body-destabilising. So the mass layout must stay
    near-planar (margin → 1) to keep gyroscopic help.
  - Pendulum term z_cg (CG BELOW the rotor plane, z_cg<0, is stabilising).
  - Self-correcting RPM slope dQ/dΩ<0 at the autorotation equilibrium (from the
    aero solver) — a Validated property of the torque curve.

Honest finding: a real samara is stabilised AERODYNAMICALLY (coning + LEV + aero
damping), not gyroscopically; this screen at best shows the spin axis is
*marginally* the max-inertia axis, with the margin eroded by axial mass. The
single-wing 1-per-rev (CG-offset orbit) forcing is NOT modelled here — flagged
as the dominant monowing dynamic for a later step. Absolute masses are
placeholders until a bill of materials exists; the stability BOUNDARIES
(signs/ratios) are more robust than the magnitudes.

Run:  ~/ds/bin/python samara_mass.py
"""

import numpy as np
from dataclasses import dataclass
import samara_bem as sb

G = 9.81


@dataclass
class MassModel:
    cfg:      sb.Config                      # aero geometry (gives the wing grid)
    m_total:  float = 0.075                  # total mass [kg]
    f_wing:   float = 0.45                   # wing mass fraction (distributed)
    f_prop:   float = 0.20                   # nose-prop + motor fraction (the mandate variable)
    f_core:   float = 0.30                   # decoupled-core (sensors/battery) fraction
    # remainder f_struct = 1 − f_wing − f_prop − f_core sits as seed/structure in-plane
    z_prop:   float = 0.04                   # prop mass height ABOVE rotor plane [m]
    z_core:   float = 0.05                   # core mass depth BELOW rotor plane [m]
    r_hub:    float = 0.0                    # radial offset of the hub masses [m] (~on axis)

    @property
    def m_wing(self):   return self.f_wing  * self.m_total
    @property
    def m_prop(self):   return self.f_prop  * self.m_total
    @property
    def m_core(self):   return self.f_core  * self.m_total
    @property
    def m_struct(self): return max(0.0, 1 - self.f_wing - self.f_prop - self.f_core) * self.m_total

    def _wing(self):
        """Spanwise wing mass elements dm(r) ∝ chord, on the aero grid."""
        r, c, _ = self.cfg.grid()
        dr = r[1] - r[0]
        w  = c * dr
        return r, self.m_wing * w / np.sum(w)

    # ── inertias, CG (all about the CG / spin axis) ───────────────────────
    def properties(self):
        r, dm = self._wing()
        m = self.m_total
        # wing radial first moment that the hub/seed counterweight must balance
        wing_first_moment = float(np.sum(dm * r))                 # kg·m
        # axial CG (prop above +z_prop, core below −z_core; wing & struct at z=0)
        z_cg = (self.m_prop*self.z_prop - self.m_core*self.z_core) / m
        # polar (spin-axis) inertia about the CG axis
        I_zz = float(np.sum(dm * r**2)) + (self.m_prop+self.m_core+self.m_struct)*self.r_hub**2
        # axial spread Σ m (z − z_cg)²  (the term that lifts the transverse inertia)
        axial = (self.m_wing*(0-z_cg)**2 + self.m_prop*(self.z_prop-z_cg)**2
                 + self.m_core*(-self.z_core-z_cg)**2 + self.m_struct*(0-z_cg)**2)
        # largest transverse inertia (in-plane axis ⊥ blade, through CG) = I_zz + axial.
        I_t = I_zz + axial
        return dict(m=m, wing_first_moment=wing_first_moment, z_cg=z_cg,
                    I_zz=I_zz, I_t=I_t, axial_spread=axial, k_gyr=np.sqrt(I_zz/m))

    # ── stability diagnostics (uses the aero equilibrium for Ω*, dQ/dΩ) ────
    def stability(self):
        p = self.properties()
        Vd, Om, info = sb.solve(self.cfg, self.m_total, return_info=True)
        I_zz, I_t = p['I_zz'], p['I_t']
        gyro_stiff = I_zz * Om**2 if not np.isnan(Om) else np.nan
        grav_stiff = self.m_total * G * abs(p['z_cg'])
        margin = I_zz / I_t
        return dict(
            **p,
            Vd=Vd, Omega=Om, rpm=(Om*60/(2*np.pi) if not np.isnan(Om) else np.nan),
            major_axis_margin=margin,                       # ≤1; →1 = max-inertia spin
            near_planar=bool(margin > 0.90),                # gyroscopically favourable
            pendulum_stable=bool(p['z_cg'] < 0),            # CG below plane
            dQ_dOmega=info.get('dQ_dOmega', np.nan),
            self_correcting_rpm=info.get('self_correcting', False),
            gyro_stiffness=gyro_stiff, gravity_stiffness=grav_stiff,
            gyro_ratio=(gyro_stiff/grav_stiff if grav_stiff > 1e-9 else np.inf),
        )


# ── analytic validation of the inertia integrals ──────────────────────────
def validate_inertia():
    """I_zz against closed forms: a uniform blade hub→tip → m R²/3 about the
    root axis; a tip point mass → m R²."""
    cfg = sb.Config(name='rod', R=1.0, c_root=0.05, c_tip=0.05, hub_frac=0.0, n_elem=400)
    mm  = MassModel(cfg=cfg, m_total=1.0, f_wing=1.0, f_prop=0.0, f_core=0.0)
    Izz = mm.properties()['I_zz']
    ok_rod = abs(Izz - 1/3) / (1/3) < 0.01
    cfg2 = sb.Config(name='pt', R=1.0, c_root=1e-6, c_tip=1.0, hub_frac=0.999, n_elem=50)
    Izz2 = MassModel(cfg=cfg2, m_total=1.0, f_wing=1.0, f_prop=0.0, f_core=0.0).properties()['I_zz']
    ok_pt = abs(Izz2 - 1.0) < 0.02
    # planar check: with no axial mass offsets, I_t must equal I_zz (margin=1)
    flat = MassModel(cfg=sb.SAMARA, m_total=0.075, f_prop=0.0, f_core=0.0, z_prop=0.0, z_core=0.0)
    pf = flat.properties(); ok_flat = abs(pf['I_t'] - pf['I_zz'])/pf['I_zz'] < 1e-9
    return dict(rod_Izz=Izz, tip_Izz=Izz2, planar_margin=pf['I_zz']/pf['I_t'],
                **{'pass': ok_rod and ok_pt and ok_flat})


# ── nose-prop weight-ratio trade study ────────────────────────────────────
def nose_prop_trade(cfg, m_total=0.075, f_props=None, z_prop=0.04, z_core=0.05):
    """Sweep the nose-prop mass fraction (prop traded against in-plane
    structure; wing and core budgets held); report the stability margins."""
    if f_props is None:
        f_props = np.linspace(0.05, 0.45, 9)
    rows = []
    for fp in f_props:
        mm = MassModel(cfg=cfg, m_total=m_total, f_wing=0.45, f_prop=fp,
                       f_core=0.30, z_prop=z_prop, z_core=z_core)
        s = mm.stability()
        rows.append((fp, s['z_cg'], s['major_axis_margin'], s['gyro_ratio'],
                     s['near_planar'], s['pendulum_stable']))
    return rows


def report(mm, title):
    s = mm.stability()
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")
    print(f"  masses: wing {mm.m_wing*1e3:.0f} g | prop {mm.m_prop*1e3:.0f} g | "
          f"core {mm.m_core*1e3:.0f} g | struct {mm.m_struct*1e3:.0f} g  (total {mm.m_total*1e3:.0f} g)")
    print(f"  I_zz = {s['I_zz']*1e4:.2f}e-4 kg·m²   I_t(transverse) = {s['I_t']*1e4:.2f}e-4 kg·m²   "
          f"k_gyr = {s['k_gyr']*100:.1f} cm")
    print(f"  CG axial z_cg = {s['z_cg']*100:+.1f} cm   |  wing radial first moment "
          f"{s['wing_first_moment']*1e3:.1f} g·m to counterbalance")
    print(f"  equilibrium: V_d={s['Vd']:.2f} m/s  Ω*={s['rpm']:.0f} RPM")
    print(f"  ── stability screen ──")
    print(f"    max-inertia spin margin I_zz/I_t = {s['major_axis_margin']:.2f}  "
          f"({'near-planar → gyroscopically favourable ✓' if s['near_planar'] else 'axial-mass-eroded → aero-reliant ✗'})")
    print(f"    pendulum z_cg={s['z_cg']*100:+.1f} cm  "
          f"({'below plane → stabilising ✓' if s['pendulum_stable'] else 'above plane → destabilising ✗'})")
    print(f"    self-correcting RPM dQ/dΩ={s['dQ_dOmega']:+.1e}  "
          f"({'✓' if s['self_correcting_rpm'] else '✗'})")
    print(f"    gyro stiffness I_zz·Ω² / gravity stiffness m·g·|z_cg| = {s['gyro_ratio']:.0f}")
    return s


if __name__ == '__main__':
    v = validate_inertia()
    print(f"[validation] inertia: rod I_zz={v['rod_Izz']:.4f} (mR²/3=0.3333), "
          f"tip I_zz={v['tip_Izz']:.4f} (mR²=1), planar margin={v['planar_margin']:.4f} (=1) "
          f"→ {'PASS ✓' if v['pass'] else 'FAIL ✗'}")

    report(MassModel(cfg=sb.SAMARA, m_total=0.075),
           'Sycamore device — baseline mass layout (samara-realistic aero)')

    mm = MassModel(cfg=sb.SAMARA)
    print(f"\n{'─'*60}\n  Nose-prop weight-ratio trade (z_prop={mm.z_prop*100:.0f}cm above, "
          f"z_core={mm.z_core*100:.0f}cm below)\n{'─'*60}")
    print(f"  {'f_prop':>7} {'z_cg[cm]':>9} {'I_zz/I_t':>9} {'gyro':>7}  verdict")
    for fp, zcg, maj, gyr, planar, pend in nose_prop_trade(sb.SAMARA):
        verdict = ('planar' if planar else 'AXIAL') + ('+pend✓' if pend else '+PEND✗')
        print(f"  {fp:7.2f} {zcg*100:9.2f} {maj:9.3f} {gyr:7.0f}  {verdict}")
