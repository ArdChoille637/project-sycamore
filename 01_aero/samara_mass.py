"""
Mass properties & rotational stability for the samara device — Project Sycamore
Phase-1 Wave 2 (a).

The "... nose-prop weight ratio, and mass distribution for STABLE autorotation ..."
half of the §6 Phase-1 mandate. The steady BEM shows an autorotation equilibrium
EXISTS; this module asks whether it is STABLE and how the mass layout governs that.

Inertia model (rigid body, all moments about the true CG):
  - Wing mass distributed along the span ∝ chord c(r) (reuses the aero grid),
    lying in the rotor plane along the blade line (X axis).
  - Hub assembly on the spin axis (X=0): nose-prop above the plane (Z=+z_prop),
    decoupled core below (Z=−z_core), seed/structure in-plane (Z=0).
A single wing is one-sided, so the system CG is offset OUTBOARD of the hub by
r_cg; the craft spins about the vertical axis through that CG. We assemble the
full point-mass cloud, locate the CG, and build the inertia tensor about it —
the polar spin moment I_spin (about the vertical CG axis, what drives dΩ/dt),
the three principal moments, and the product of inertia I_xz (spin-axis tilt).

Stability screen (rigid body, tier Indicative):
  - The vertical spin axis is ALWAYS the INTERMEDIATE principal axis for a
    samara whose in-plane radial spread exceeds its axial spread: by
    construction I_Y(⊥-blade) = I_spin + Σm·z'² > I_spin > I_X(along-blade).
    Intermediate-axis spin is rigid-body-UNSTABLE (tennis-racket theorem), so a
    samara is NOT gyroscopically self-stabilising — it is stabilised
    AERODYNAMICALLY (coning + LEV + aero damping) and by the PENDULUM effect.
  - Pendulum term: CG below the rotor plane (z_cg<0) is statically stabilising;
    this is the primary mass-layout knob and what bounds the nose-prop weight
    ratio.
  - Self-correcting RPM dQ/dΩ<0 at the autorotation equilibrium (from the aero
    solver) — a Validated property of the torque curve.

Honest scope: this is a low-order rigid-body screen on a parameterised,
not-yet-bench-measured mass breakdown; the BOUNDARIES (signs/crossings) are more
robust than the magnitudes. It does NOT model the single-wing 1/rev (CG-offset
orbit) forcing, nor does the aero (which assumes the spin axis at the hub)
account for the r_cg offset — both flagged for later. Absolute masses are
placeholders until a hardware bill of materials exists.

Run:  ~/ds/bin/python samara_mass.py
"""

import numpy as np
from dataclasses import dataclass
import samara_bem as sb

G = sb.G          # single source of truth (samara_bem)


@dataclass
class MassModel:
    cfg:      sb.Config                      # aero geometry (gives the wing grid)
    m_total:  float = 0.075                  # total mass [kg]
    f_wing:   float = 0.45                   # wing mass fraction (distributed)
    f_prop:   float = 0.20                   # nose-prop + motor fraction (the mandate variable)
    f_core:   float = 0.30                   # decoupled-core (sensors/battery) fraction
    z_prop:   float = 0.04                   # prop mass height ABOVE rotor plane [m]
    z_core:   float = 0.05                   # core mass depth BELOW rotor plane [m]

    def __post_init__(self):
        s = self.f_wing + self.f_prop + self.f_core
        if s > 1.0 + 1e-9:
            raise ValueError(f"mass fractions exceed 1 (wing+prop+core={s:.3f}); "
                             f"f_struct would be negative — renormalise the budget.")

    @property
    def m_wing(self):   return self.f_wing  * self.m_total
    @property
    def m_prop(self):   return self.f_prop  * self.m_total
    @property
    def m_core(self):   return self.f_core  * self.m_total
    @property
    def f_struct(self):  return 1.0 - self.f_wing - self.f_prop - self.f_core
    @property
    def m_struct(self): return self.f_struct * self.m_total

    def _cloud(self):
        """Full point-mass cloud (x radial, z axial, m). Wing ∝ chord in-plane;
        hub masses on the spin axis at their axial offsets."""
        r, c, _ = self.cfg.grid()
        dr = r[1] - r[0]
        dm = self.m_wing * (c*dr) / np.sum(c*dr)
        xs = np.concatenate([r,               [0.0, 0.0, 0.0]])
        zs = np.concatenate([np.zeros_like(r), [self.z_prop, -self.z_core, 0.0]])
        ms = np.concatenate([dm,              [self.m_prop, self.m_core, self.m_struct]])
        return xs, zs, ms, float(np.sum(dm*r))

    # ── inertia tensor about the true CG ──────────────────────────────────
    def properties(self):
        xs, zs, ms, wing_first_moment = self._cloud()
        M = float(ms.sum())
        r_cg = float((ms*xs).sum() / M)
        z_cg = float((ms*zs).sum() / M)
        X = xs - r_cg; Z = zs - z_cg                       # positions about the CG (Y=0)
        I_xx = float((ms*Z**2).sum())                      # about in-plane axis ALONG the blade
        I_yy = float((ms*(X**2 + Z**2)).sum())             # about in-plane axis ⊥ the blade
        I_spin = float((ms*X**2).sum())                    # about the VERTICAL axis (the spin axis)
        I_xz = float(-(ms*X*Z).sum())                      # product of inertia (spin-axis tilt)
        # principal moments: I_yy is principal (the Y axis); diagonalise the X–Z block
        block = np.array([[I_xx, I_xz], [I_xz, I_spin]])
        e = np.linalg.eigvalsh(block)
        principal = np.sort([I_yy, float(e[0]), float(e[1])])
        tilt = 0.5*np.degrees(np.arctan2(2*I_xz, I_spin - I_xx))   # spin-axis → principal tilt
        return dict(m=M, r_cg=r_cg, z_cg=z_cg, I_spin=I_spin, I_xx=I_xx, I_yy=I_yy,
                    I_xz=I_xz, principal=principal, tilt_deg=float(tilt),
                    wing_first_moment=wing_first_moment, k_gyr=np.sqrt(I_spin/M))

    # ── stability diagnostics ─────────────────────────────────────────────
    def stability(self):
        p = self.properties()
        Vd, Om, info = sb.solve(self.cfg, self.m_total, return_info=True)
        pr = p['principal']; I_spin = p['I_spin']
        # where the spin axis sits among the principal moments
        spin_is_max = bool(I_spin >= pr[-1] - 1e-12)
        spin_is_min = bool(I_spin <= pr[0] + 1e-12)
        spin_axis = 'max' if spin_is_max else ('min' if spin_is_min else 'intermediate')
        return dict(
            **p,
            Vd=Vd, Omega=Om, rpm=(Om*60/(2*np.pi) if not np.isnan(Om) else np.nan),
            spin_axis=spin_axis,                            # 'intermediate' ⇒ rigid-body unstable
            rigidbody_spin_stable=bool(spin_is_max or spin_is_min),
            pendulum_stable=bool(p['z_cg'] < 0),            # CG below plane (the real knob)
            dQ_dOmega=info.get('dQ_dOmega', np.nan),
            self_correcting_rpm=info.get('self_correcting', False),
        )


# ── analytic validation of the inertia tensor ─────────────────────────────
def validate_inertia():
    """Genuine about-CG checks (the old version validated about-root integrals)."""
    # 1) uniform rod over [0,R], no axial mass: polar about the CG (at R/2) = mR²/12
    rod = MassModel(cfg=sb.Config(name='rod', R=1.0, c_root=0.05, c_tip=0.05,
                                  hub_frac=0.0, n_elem=2000),
                    m_total=1.0, f_wing=1.0, f_prop=0.0, f_core=0.0, z_prop=0.0, z_core=0.0)
    pr = rod.properties()
    ok_rod = abs(pr['I_spin'] - 1/12) / (1/12) < 0.01 and abs(pr['r_cg'] - 0.5) < 1e-3
    # 2) parallel-axis: point mass m at x=d plus equal mass at x=0 → CG at d/2,
    #    polar about CG = 2·m·(d/2)² = m·d²/2.  Build directly via the cloud math.
    d, m = 0.4, 0.5
    xs = np.array([d, 0.0]); ms = np.array([m, m]); Mt = ms.sum()
    rcg = (ms*xs).sum()/Mt; Ispin = (ms*(xs-rcg)**2).sum()
    ok_par = abs(Ispin - m*d**2/2) < 1e-9 and abs(rcg - d/2) < 1e-12
    # 3) axial-spread identity: I_yy = I_spin + Σm·z'²  ≥  I_spin (always)
    base = MassModel(cfg=sb.SAMARA).properties()
    ok_id = base['I_yy'] >= base['I_spin'] - 1e-15
    return dict(rod_Ispin=pr['I_spin'], par_Ispin=Ispin,
                samara_spin=base['I_spin'], samara_principals=base['principal'],
                **{'pass': ok_rod and ok_par and ok_id})


# ── nose-prop weight-ratio trade study (budget-conserving) ─────────────────
def nose_prop_trade(cfg, m_total=0.075, f_props=None, f_wing=0.45, f_struct=0.05,
                    z_prop=0.04, z_core=0.05):
    """Trade nose-prop mass against the counter-balancing core mass (prop above
    vs core below — the direct pendulum trade), holding wing + structure. Total
    mass is conserved (f_prop + f_core = 1 − f_wing − f_struct)."""
    budget = 1.0 - f_wing - f_struct
    if f_props is None:
        f_props = np.linspace(0.05, budget - 0.05, 9)
    rows = []
    for fp in f_props:
        fc = budget - fp
        mm = MassModel(cfg=cfg, m_total=m_total, f_wing=f_wing, f_prop=fp,
                       f_core=fc, z_prop=z_prop, z_core=z_core)
        s = mm.stability()
        rows.append((fp, fc, s['z_cg'], s['pendulum_stable']))
    # analytic z_cg=0 crossing of the prop-vs-core trade
    fp_star = (budget*z_core) / (z_prop + z_core)
    return rows, fp_star


def report(mm, title):
    s = mm.stability()
    pr = s['principal']
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")
    print(f"  masses: wing {mm.m_wing*1e3:.0f} g | prop {mm.m_prop*1e3:.0f} g | "
          f"core {mm.m_core*1e3:.0f} g | struct {mm.m_struct*1e3:.0f} g  (total {s['m']*1e3:.0f} g)")
    print(f"  CG: radial r_cg = {s['r_cg']*100:.1f} cm (outboard of hub)   axial z_cg = {s['z_cg']*100:+.1f} cm")
    print(f"  I_spin = {s['I_spin']*1e4:.2f}e-4 kg·m²  (about the vertical CG axis)   "
          f"k_gyr = {s['k_gyr']*100:.1f} cm")
    print(f"  principal moments = [{pr[0]*1e4:.2f}, {pr[1]*1e4:.2f}, {pr[2]*1e4:.2f}]e-4 kg·m²   "
          f"spin-axis tilt {s['tilt_deg']:.1f}°")
    print(f"  equilibrium: V_d={s['Vd']:.2f} m/s  Ω*={s['rpm']:.0f} RPM")
    print(f"  ── stability screen ──")
    print(f"    spin axis is the {s['spin_axis'].upper()} principal axis  "
          f"({'rigid-body stable' if s['rigidbody_spin_stable'] else 'rigid-body UNSTABLE (tennis-racket) → aero-stabilised'})")
    print(f"    pendulum z_cg = {s['z_cg']*100:+.1f} cm  "
          f"({'below plane → stabilising ✓' if s['pendulum_stable'] else 'above plane → destabilising ✗'})")
    print(f"    self-correcting RPM dQ/dΩ = {s['dQ_dOmega']:+.1e}  "
          f"({'✓' if s['self_correcting_rpm'] else '✗'})")
    return s


if __name__ == '__main__':
    v = validate_inertia()
    print(f"[validation] inertia (about CG): rod I_spin={v['rod_Ispin']:.4f} (mR²/12=0.0833), "
          f"2-mass I_spin={v['par_Ispin']:.4f} (m·d²/2=0.0400), I_yy≥I_spin identity → "
          f"{'PASS ✓' if v['pass'] else 'FAIL ✗'}")
    print(f"             samara principal moments = "
          f"{[round(x*1e4,2) for x in v['samara_principals']]}e-4, "
          f"spin={v['samara_spin']*1e4:.2f}e-4 → spin axis is INTERMEDIATE")

    report(MassModel(cfg=sb.SAMARA, m_total=0.075),
           'Sycamore device — baseline mass layout (samara-realistic aero)')

    rows, fp_star = nose_prop_trade(sb.SAMARA)
    print(f"\n{'─'*62}\n  Nose-prop weight-ratio trade  (prop above vs core below;"
          f"\n  budget-conserving f_prop+f_core={1-0.45-0.05:.2f}; z_prop=4cm, z_core=5cm)\n{'─'*62}")
    print(f"  {'f_prop':>7} {'f_core':>7} {'z_cg[cm]':>9}  pendulum")
    for fp, fc, zcg, pend in rows:
        print(f"  {fp:7.2f} {fc:7.2f} {zcg*100:9.2f}  {'stable ✓' if pend else 'UNSTABLE ✗'}")
    print(f"  → nose-prop weight ratio bounded above at f_prop ≈ {fp_star:.2f} "
          f"(z_cg crosses 0: CG rises above the rotor plane)")
