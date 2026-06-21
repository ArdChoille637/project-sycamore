"""
Powered-mode propeller aero — Project Sycamore Phase-1.

The active mode is a single **nose propeller** that provides thrust for hover,
climb, and horizontal displacement before the craft cuts power and transitions
to passive samara autorotation. The 2-DOF transition solver (`samara_transient`)
currently treats the powered state as just an initial condition; this module
gives it an actual aero model, and answers the design questions it implies:

  - how much power to hover / climb the 75 g craft, and how it scales with the
    nose-prop diameter D_p (the disk-loading trade);
  - what active-mode operating point sets the transition initial condition;
  - whether the prop+motor mass fits under the Wave-2 nose-prop weight bound
    (f_prop ≲ 0.28 ⇒ ≲ 21 g of the 75 g budget).

Model: actuator-disk / momentum theory (Glauert), the same induced-velocity
backbone as the Wave-1 rotor inflow but applied to a powered disk.

Developed as a **Claude Code ⇄ Gemini (Antigravity) collaboration**: Claude
proposed the model + claims P1–P4; Gemini (3.1 Pro) independently re-derived and
confirmed all four (hover power 7.2 W at D_p=0.10 m, FM 0.65) with no
corrections (see /Users/home/Claude/shared-with-gemini/SHARED_MEMORY.md).

Tier: Indicative — the actuator-disk aero (P1–P3) is solid momentum theory; the
prop+motor *mass* model (P4) is a transparent parametric estimate (assumptions
printed), so the weight-bound verdict is a trade map, not a procurement spec.

Run:  ~/ds/bin/python samara_prop.py
"""

import numpy as np
import samara_bem as sb

G   = sb.G                   # single source of truth (samara_bem)
RHO = sb.SAMARA.rho          # 1.167 kg/m³ (match the aero model)

# --- prop+motor mass model (Indicative; assumptions are printed in the report) ---
PROP_G_AT_10CM = 5.0         # g of blade for a 0.10 m prop, scaled ∝ D_p² (disk area)
MOTOR_W_PER_G  = 2.0         # small BLDC+ESC continuous specific power (W per g)
MOTOR_FLOOR_G  = 4.0         # smallest practical motor mass (g)


def hover_disk(T, D_p, FM=0.65, rho=RHO):
    """Hover (axial V=0): T = 2ρA·v_i² ⇒ v_i, ideal power T·v_i, real power /FM."""
    A_p = np.pi * (D_p/2)**2
    v_i = np.sqrt(T / (2*rho*A_p))
    P_ideal = T * v_i
    return dict(D_p=D_p, A_p=A_p, v_i=v_i, P_ideal=P_ideal, P_real=P_ideal/FM,
                disk_loading=T/A_p, FM=FM)


def axial_v_i(T, V, A_p, rho=RHO):
    """Induced velocity in axial climb at speed V>0: solve v_i²+V·v_i−T/(2ρA)=0."""
    k = T / (2*rho*A_p)
    return (-V + np.sqrt(V*V + 4*k)) / 2


def climb_disk(T, V, D_p, FM=0.65, rho=RHO):
    """Steady axial climb at V (m/s): useful power T·(V+v_i), real /FM."""
    A_p = np.pi * (D_p/2)**2
    v_i = axial_v_i(T, V, A_p, rho)
    P_ideal = T * (V + v_i)
    return dict(D_p=D_p, V=V, v_i=v_i, P_ideal=P_ideal, P_real=P_ideal/FM)


def prop_motor_mass_g(D_p, P_real):
    """Indicative prop+motor mass: blade ∝ D_p² + BLDC sized by continuous power."""
    m_prop  = PROP_G_AT_10CM * (D_p/0.10)**2
    m_motor = max(MOTOR_FLOOR_G, P_real / MOTOR_W_PER_G)
    return m_prop + m_motor


def sweep(m_kg=0.075, FM=0.65, f_prop_max=0.28,
          diameters=(0.06, 0.08, 0.10, 0.12, 0.15)):
    """Hover power + disk loading + est. prop-group mass vs prop diameter."""
    W = m_kg * G
    m_budget_g = f_prop_max * m_kg * 1000
    rows = []
    for D_p in diameters:
        h = hover_disk(W, D_p, FM)
        m_g = prop_motor_mass_g(D_p, h['P_real'])
        rows.append(dict(D_p=D_p, v_i=h['v_i'], P_real=h['P_real'],
                         disk_loading=h['disk_loading'], mass_g=m_g,
                         fits=bool(m_g <= m_budget_g)))
    return rows, m_budget_g, W


def report(m_kg=0.075, D_p=0.10, FM=0.65, f_prop_max=0.28):
    W = m_kg * G
    h = hover_disk(W, D_p, FM)
    c = climb_disk(W, 1.0, D_p, FM)               # 1 m/s climb
    rows, m_budget_g, _ = sweep(m_kg, FM, f_prop_max)

    print(f"\n{'='*68}\n  Powered-mode propeller aero — Sycamore active mode\n{'='*68}")
    print(f"  craft: m={m_kg*1000:.0f} g, W={W:.3f} N, ρ={RHO} kg/m³, FM={FM}")

    print(f"\n  P1/P2 — hover at D_p={D_p*100:.0f} cm (A_p={h['A_p']*1e3:.1f}e-3 m²):")
    print(f"     induced velocity v_i = {h['v_i']:.2f} m/s,  disk loading = {h['disk_loading']:.0f} N/m²")
    print(f"     ideal power = {h['P_ideal']:.2f} W  →  real power = {h['P_real']:.1f} W  (/FM)")
    print(f"     +1 m/s climb: v_i={c['v_i']:.2f} m/s, real power = {c['P_real']:.1f} W "
          f"(+{100*(c['P_real']/h['P_real']-1):.0f}%)")

    print(f"\n  P3 — transition IC: the natural active→passive hand-off is engine cut from")
    print(f"     hover (V_z=0, prop thrust→0) while already at autorotation spin ⇒ the transient")
    print(f"     solver's nominal IC is V_z0≈0, Ω0=Ω* (≈650 RPM) — matching samara_transient.")

    print(f"\n  P4 — disk-loading vs weight trade (mass model: {PROP_G_AT_10CM:.0f} g blade @10 cm ∝D²,")
    print(f"     motor {MOTOR_W_PER_G:.0f} W/g, {MOTOR_FLOOR_G:.0f} g floor; budget f_prop≤{f_prop_max:.2f} ⇒ ≤{m_budget_g:.0f} g):")
    print(f"     {'D_p':>6} {'v_i':>7} {'P_real':>8} {'disk_load':>10} {'mass':>7}  fits?")
    for r in rows:
        print(f"     {r['D_p']*100:5.0f}cm {r['v_i']:6.2f}  {r['P_real']:6.1f}W {r['disk_loading']:8.0f}N/m² "
              f"{r['mass_g']:5.1f}g  {'✓' if r['fits'] else '✗ over'}")
    best = min((r for r in rows if r['fits']), key=lambda r: r['P_real'], default=None)
    if best:
        print(f"     ⇒ within the weight bound, the lowest-power option is "
              f"D_p={best['D_p']*100:.0f} cm at {best['P_real']:.1f} W ({best['mass_g']:.1f} g).")
    return dict(hover=h, climb=c, sweep=rows, m_budget_g=m_budget_g)


if __name__ == '__main__':
    print("[note] Claude⇄Gemini collaboration; Gemini independently confirmed claims "
          "P1–P4 (see shared-with-gemini/SHARED_MEMORY.md).")
    report()
