# Project Sycamore — status & resume point

**Last updated:** 2026-06-20 · **Phase:** 1 (single-vehicle autorotation aerodynamics) — *complete, reviewed, and verified*

This file is the "where we are / what's next" entry point. Read it first when resuming.

---

## TL;DR

Phase-1 of the samara airframe model is **done, adversarially reviewed, and merged to `main`**.
The aero/dynamics layer is internally consistent and honestly documented. There is **one open
question — a design decision, not a bug** — that is deliberately deferred until there is CFD /
wind-tunnel evidence (see [The one open decision](#the-one-open-decision)).

The PR queue is **empty**. Nothing is half-finished.

---

## What's on `main` (Phase-1)

The `01_aero/` build, in dependency order (all run with `~/ds/bin/python <module>.py` from `01_aero/`):

| Module | Role | Headline result |
|---|---|---|
| `samara_bem.py` | BEM autorotation solver (the equilibrium **hub**) | `SAMARA`: **V_d = 2.90 m/s, 650 RPM** |
| `samara_mass.py` | CG-referenced mass/inertia, pendulum stability | I_spin 6.08e-4, I_t 6.85e-4 kg·m² |
| `samara_transient.py` | 2-DOF active→passive transition | stable attractor; settles to autorotation |
| `samara_optimize.py` | wing-geometry optimisation over the Rossby band | optimum Ro 3.03 (see open decision) |
| `samara_1rev.py` | single-wing 1/rev dynamics (sensor/core spec) | 32 g vibration unbalanced; β≈24° coning |
| `samara_prop.py` | powered-mode actuator-disk prop aero | hover **7.2 W** @ 10 cm |

Every downstream module solves the **same** `SAMARA` config at the **same** reproduced equilibrium
(V_d=2.897, 650 RPM); the air-density split (1.167 operational / 1.225 validation) is clean; the
per-element thrust/torque is a single shared kernel (`sb.element_thrust` / `sb.element_torque`).

### Merged PRs (this work)
- **#8–#11** — Phase-1 waves 0–3: well-posed BEM → validated LEV aero → mass/stability/transition → geometry optimisation
- **#12** — CFD spot-check of the geometry optimum (honest qualitative LEV probe)
- **#13** — single-wing 1/rev dynamics  *(Claude⇄Gemini; Gemini caught 2 physics errors)*
- **#14** — powered-mode propeller aero  *(Claude⇄Gemini; Gemini confirmed clean)*
- **#15** — joint review fixes: disclose orphaned optimum + safe wiring cleanups  *(multi-agent review + Gemini-verified)*

---

## The one open decision

`samara_optimize.py` flags the **headline `SAMARA` geometry (Rossby ≈ 5.45)** as *infeasible* for the
stable-leading-edge-vortex mechanism the aero model assumes, and its feasible optimum is a wider-chord
geometry (**Ro ≈ 3.03, c_root 14.4 cm, V_d 2.56 m/s, 480 RPM**). The whole build currently uses the
Ro 5.45 baseline; the optimum is **not** adopted.

This is **disclosed** (README "Headline-geometry caveat" + a `NOTE` at the `SAMARA` config), not
silently swapped — because the optimum is itself locally non-slender (root chord ≈ ½ the hub radius)
and its descent edge is a soft-threshold artifact. **Neither geometry is yet the validated answer.**

**The tiebreaker is a resolved-Re CFD / wind-tunnel spot-check** of the leading-edge vortex at both
Ro 5.45 and Ro ≈ 3:
- if Ro 5.45's LEV holds → loosen the feasibility band, keep the baseline;
- if only Ro ≈ 3 sustains the LEV → adopt the optimum and re-run all downstream numbers.

---

## Next tasks (when resuming, roughly in priority order)

1. **Resolved-Re CFD spot-check** — the real unblock for the decision above. The existing `05_cfd/`
   LBM tooling (`lbm3d_rotor.py`, `lbm_samara.py`) is qualitative/Mach-limited (see PR #12's honest
   verdict); this needs a properly resolved-Re setup or an external CFD/wind-tunnel path.
2. **Wire the prop hover operating point into `samara_transient`'s IC** — connects #14 to the
   transition model so the active→passive hand-off is modelled end-to-end instead of assumed.
3. **Swarm-scale dispersal / coverage** — step up from the single airframe to the swarm (`03_swarm/`).
4. **Coning terminology rename** (low priority, noted in review) — `samara_bem.coning_deg` (blade
   flap-up) vs `samara_1rev.β` (gyroscopic spin-axis tilt) are different DOF but share the name
   "coning/β"; disambiguate to avoid the confusion it already caused one reviewer.

---

## Working setup

- **Python env:** `~/ds/bin/python` (uv venv with the full sci stack; numpy 2.x → `np.trapezoid`).
- **Run a module:** `cd 01_aero && ~/ds/bin/python samara_bem.py` (figures → `output/`).
- **Claude ⇄ Gemini collaboration:** the shared workspace is
  `/Users/home/Claude/shared-with-gemini/SHARED_MEMORY.md` (read it for the full per-task log).
  To resume a joint task, ask Claude to "collaborate with Gemini" — it posts to the shared file and
  relays via Gemini in the Antigravity IDE.

---

*Status convention: every result in the suite is tagged **Validated** (checked against an external
anchor) or **Indicative** (order-of-magnitude; dependencies/criteria are the deliverable). The Phase-1
aero is Validated against Sycamore A; the 1/rev, prop, and optimum magnitudes are Indicative pending CFD.*
