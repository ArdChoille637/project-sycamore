"""
Single-wing 1-per-rev (1/rev) dynamics — Project Sycamore Phase-1.

A samara is a *single* wing spinning at Ω about its CG, so — unlike a balanced
multi-blade rotor — it carries a once-per-revolution (1/rev) asymmetry. This is
the dominant monowing dynamic flagged by the Wave-2 stability review, and it
governs the **onboard-sensor** design (motion blur, vibration) and the
decoupled-core isolation spec. It reuses the validated Wave-1 aero
(`samara_bem`) and Wave-2 mass model (`samara_mass`).

This module was developed as a **Claude Code ⇄ Gemini (Antigravity)
collaboration**: Claude proposed the four effects below; Gemini (3.1 Pro)
independently cross-checked and corrected two points in effect C — both adopted
here:
  - the body-frame-vs-inertial-frame nuance (an OFF-axis body-fixed sensor sees
    a 1/rev wobble, not a steady tilt), and
  - the gyroscopic restoring stiffness depends on the inertia DIFFERENCE,
    (I_t − I_spin)·Ω², not I_spin·Ω².

Four effects:
  A — Spin blur (geometric): the body spins at Ω, so a body-fixed sensor sees the
      world sweep past at Ω → exposure ≪ 1/Ω, i.e. de-spin or an Ω-synced strobe.
  B — 1/rev vibration: if not radially balanced, the hub (sensor/core) orbits the
      true CG at radius r_cg → centripetal accel a = Ω²·r_cg.
  C — Coning tilt: the single wing's offset thrust is a steady (body-frame)
      overturning moment M = W·r_cp that tilts the spin axis by a coning angle β,
      balanced by the gyroscopic restoring (I_t − I_spin)·Ω²·sinβcosβ. An
      off-axis body-fixed sensor sees β as a 1/rev wobble.
  D — Decoupled core: the ferrofluid-decoupled, non-spinning core at the CG is the
      mitigation for A+B+C; this module quantifies its isolation spec.

Tier: Indicative (rigid-body + aero-tier forces; absolute magnitudes are
order-of-magnitude, the dependencies/criteria are the deliverable).

Run:  ~/ds/bin/python samara_1rev.py
"""

import numpy as np
import samara_bem as sb
import samara_mass as sm

G = sb.G          # single source of truth (samara_bem)


def thrust_center(cfg, m_kg):
    """Spanwise centre of pressure r_cp = ∫r·dT / ∫dT at the autorotation
    equilibrium (the lever arm of the single wing's net thrust about the axis)."""
    Vd, Om = sb.solve(cfg, m_kg)
    r, c, theta = cfg.grid()
    dr = r[1] - r[0]
    _, _, phi, al, U2, cl, cd, _ = sb._state(Vd, Om, cfg, r, c, theta)
    q  = 0.5 * cfg.rho * U2[0] * c
    dT = sb.element_thrust(q, cl[0], cd[0], phi[0], cfg.n_blades)           # per unit span (shared kernel)
    T  = float(np.sum(dT * dr))
    r_cp = float(np.sum(r * dT * dr) / T)
    return Vd, Om, T, r_cp


def one_rev(cfg, m_kg=0.075, mass=None, blur_tol_mrad=5.0, g_budget=1.0):
    """Compute the four 1/rev effects for a config + mass layout."""
    mass = mass or sm.MassModel(cfg=cfg, m_total=m_kg)
    p = mass.properties()
    I_spin = p['I_spin']
    I_t    = p['principal'][-1]            # largest transverse principal moment
    r_cg   = p['r_cg']                      # CG offset from the hub (0 if balanced)
    Vd, Om, T, r_cp = thrust_center(cfg, m_kg)
    W = m_kg * G
    f = Om / (2*np.pi)

    # A — spin blur
    t_exp_max = (blur_tol_mrad*1e-3) / Om   # exposure for ≤ blur_tol angular smear

    # B — 1/rev orbit vibration (unbalanced hub) + the balance budget
    a_orbit   = Om**2 * r_cg
    r_cg_budget = g_budget*G / Om**2        # max CG offset to keep 1/rev < g_budget

    # C — coning tilt from the offset-thrust overturning moment
    M_over = W * r_cp                       # body-frame-steady overturning moment
    K = (I_t - I_spin) * Om**2              # gyroscopic restoring stiffness (Gemini's fix)
    ratio = 2*M_over / K                    # M_over = (K/2)·sin(2β)
    saturated = bool(abs(ratio) > 1.0)
    beta = 0.5*np.arcsin(np.clip(ratio, -1, 1))

    return dict(Vd=Vd, Om=Om, spin_hz=f, spin_rpm=Om*60/(2*np.pi),
                t_exp_max_us=t_exp_max*1e6,
                r_cg_cm=r_cg*100, a_orbit_ms2=a_orbit, a_orbit_g=a_orbit/G,
                r_cg_budget_mm=r_cg_budget*1e3, g_budget=g_budget,
                r_cp_cm=r_cp*100, M_over=M_over, K_restoring=K,
                beta_deg=np.degrees(beta), coning_saturated=saturated,
                I_spin=I_spin, I_t=I_t)


def report(cfg, m_kg=0.075, mass=None, title='Sycamore device (samara-realistic)'):
    d = one_rev(cfg, m_kg, mass)
    print(f"\n{'═'*64}\n  Single-wing 1/rev dynamics — {title}\n{'═'*64}")
    print(f"  equilibrium: V_d={d['Vd']:.2f} m/s,  Ω={d['spin_rpm']:.0f} RPM = "
          f"{d['spin_hz']:.1f} Hz  (I_spin={d['I_spin']*1e4:.2f}e-4, I_t={d['I_t']*1e4:.2f}e-4)")
    print(f"  [baseline note] SAMARA Ro≈5.45 — outside the optimiser's feasibility band [3,4]; "
          f"1/rev loads scale with Ω², so a feasible Ro≈3 (≈480 RPM) geometry would cut them ~0.55×.")
    print(f"\n  A — spin blur: a body-fixed sensor sweeps the world at {d['spin_hz']:.1f} Hz.")
    print(f"      For ≤5 mrad smear, exposure must be < {d['t_exp_max_us']:.0f} µs — "
          f"⇒ de-spin the sensor or strobe in sync with Ω.")
    print(f"\n  B — 1/rev vibration (if NOT radially balanced): the hub/core orbits the CG")
    print(f"      at r_cg={d['r_cg_cm']:.1f} cm ⇒ a = Ω²·r_cg = {d['a_orbit_ms2']:.0f} m/s² "
          f"= {d['a_orbit_g']:.0f} g of 1/rev shake.")
    print(f"      ⇒ balance the CG to within {d['r_cg_budget_mm']:.1f} mm of the hub to keep "
          f"1/rev < {d['g_budget']:.0f} g, or isolate via the decoupled core.")
    print(f"\n  C — coning tilt: offset thrust ⇒ overturning moment M=W·r_cp "
          f"(r_cp={d['r_cp_cm']:.1f} cm) = {d['M_over']*1e3:.0f} mN·m;")
    print(f"      gyroscopic restoring (I_t−I_spin)·Ω² = {d['K_restoring']*1e3:.0f} mN·m ⇒ "
          f"coning β ≈ {d['beta_deg']:.0f}°"
          + ("  ⚠ SATURATED (aero moment exceeds restoring → tumble risk)" if d['coning_saturated'] else ""))
    print(f"      An OFF-axis body-fixed sensor sees this β as a 1/rev wobble "
          f"(an on-spin-axis sensor sees a steady tilt).")
    print(f"\n  D — decoupled-core spec: the non-spinning core at the CG must reject "
          f"spin {d['spin_hz']:.1f} Hz / {d['spin_rpm']:.0f} RPM,")
    print(f"      the 1/rev vibration (up to ~{d['a_orbit_g']:.0f} g unbalanced) and the "
          f"β≈{d['beta_deg']:.0f}° coning wobble — all at {d['spin_hz']:.1f} Hz.")
    return d


if __name__ == '__main__':
    print("[note] developed as a Claude ⇄ Gemini collaboration; effect-C frame + "
          "gyroscopic-stiffness corrections are Gemini's (see /Users/home/Claude/"
          "shared-with-gemini/SHARED_MEMORY.md).")
    report(sb.SAMARA)
