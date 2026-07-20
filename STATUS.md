# Project Sycamore — status & resume point

**Last updated:** 2026-07-20 · **Phase:** 1 (single-vehicle aero) *complete*; swarm layer *begun*

This file is the "where we are / what's next" entry point. Read it first when resuming.

---

## TL;DR

Phase-1 of the samara airframe model is **done, adversarially reviewed, and merged**. The **swarm
layer has begun**: passive-autorotation dispersal + coverage, with live and headless rendering, is
merged (PR #17). The old "one open decision" (which wing geometry) turned out to be **largely
resolved** — the Rossby "infeasibility" was a units artifact; keep the baseline (see below).

The PR queue is **empty**. Nothing is half-finished. Two small, ready engineering tasks and one
partly-blocked research thread remain (see [Next tasks](#next-tasks)).

---

## What's on `main`

### `01_aero/` — single-vehicle aerodynamics (Phase-1, all `~/ds/bin/python <module>.py`)

| Module | Role | Headline result |
|---|---|---|
| `samara_bem.py` | BEM autorotation solver (the equilibrium **hub**) | `SAMARA`: **V_d = 2.90 m/s, 650 RPM** |
| `samara_mass.py` | CG-referenced mass/inertia, pendulum stability | I_spin 6.08e-4, I_t 6.85e-4 kg·m² |
| `samara_transient.py` | 2-DOF active→passive transition | stable attractor; nominal cut ≈3 s / 8 m |
| `samara_optimize.py` | wing-geometry optimisation over the Rossby band | optimum Ro 3.03 (superseded — see below) |
| `samara_1rev.py` | single-wing 1/rev dynamics (sensor/core spec) | 32 g vibration unbalanced; axis-tilt ≈24° |
| `samara_prop.py` | powered-mode actuator-disk prop aero | hover **7.2 W** @ 10 cm |

Every downstream module solves the **same** `SAMARA` config at the **same** reproduced equilibrium
(V_d=2.897, 650 RPM); per-element thrust/torque is a single shared kernel.

### `03_swarm/` — swarm layer

| Module | Role | Notes |
|---|---|---|
| `dispersal.py` | **passive-autorotation dispersal + coverage** (PR #17) | imports the *Validated* V_d live; fire-AREA coverage; live + headless rendering |
| `boids_swarm_3d.py` | powered-patrol 3D boids renderer | generic boids (placeholder cruise speed); OBJ-silhouette fallback added |
| `fire_swarm.py` | coupled fire↔swarm patrol | reuses boids; Rothermel front as live perimeter |
| `swarm_core.py` | backend-free boids physics (NumPy + MLX) | patrol model — no samara physics yet |

`dispersal.py` is the one swarm module grounded in Phase-1: release a cloud upwind of a fire, each
autorotates down at the Validated V_d while drifting on wind; **a passive cloud seeds an *area*, it
can't ring a perimeter** (so the metric is fire-area coverage), and **drift = wind·H/V_d is bounded**,
so release altitude is the dominant design lever. Run modes: `--live` (native window), `--gif`/`--mp4`
(headless export), default static PNG, `--validate` (5/5 self-checks). The other two renderers model
the *powered patrol* and still use a placeholder cruise speed (no samara physics) — a future step-up.

### Merged PRs
- **#8–#11** — Phase-1 waves 0–3: well-posed BEM → validated LEV aero → mass/stability/transition → geometry optimisation
- **#12** — CFD spot-check of the geometry optimum (honest qualitative LEV probe)
- **#13** — single-wing 1/rev dynamics  *(Claude⇄Gemini)* · **#14** — powered-mode prop aero  *(Claude⇄Gemini)*
- **#15** — joint review fixes  · **#16** — this STATUS doc
- **#17** — swarm-scale dispersal + live rendering *(adversarial review: 9 findings fixed; Codex P2: nested sweep fixed)*

---

## The geometry decision — (now largely resolved)

**Conclusion: keep the `Ro ≈ 5.45` baseline.** `[Validated reasoning]`

STATUS previously framed this as an open choice between the baseline and the optimizer's `Ro ≈ 3.03`
wider-chord geometry, to be settled by CFD. A later cross-check (Claude Code, 2026-07-01, reproduced
two independent ways — recorded in `shared-with-claude-team/SHARED_MEMORY.md`) showed the "baseline
infeasible" flag was a **units artifact**:

- `samara_bem.py` computes `Ro = R/c̄` (tip-radius / mean-chord = single-wing **aspect ratio** ≈ 5.45)
  but gated it against a threshold defined on `R_g/c̄` (**radius of gyration**).
- On the correct `R_g/c̄` convention: baseline **3.07** (inside the stable-LEV window, *more central than
  the validated real samara at 2.39*), optimum 1.71. So the baseline is on the same footing as the
  validated Sycamore A — the infeasibility dissolves.

The optimizer's `Ro 3.03` optimum remains **documented but superseded** (not adopted, not erased).

**What is still genuinely open** (not a geometry choice):
- `[open]` **Threshold citation pin** — the code cites *Science* 324:1438 for the `Ro ≲ 3–4` gate, but
  that paper lacks the definition; the real source is *J. Exp. Biol.* 212:2705 (not on disk). Needs a
  holdable primary source before patching `samara_bem.py`'s `Ro_gyration`/citation/comments. *(This is
  the `@science` hand-off in the team file — user-relayed.)*
- `[Indicative]` **Reynolds residual** — outer-span LEV bursting at device Re ≈ 43–57k. The resolved-Re
  CFD / spin-rig (`HANDOFF-cfd-spotcheck.md`, uncommitted) is a **Reynolds confirmation**, *not* a
  geometry tiebreaker. It's the honest SBIR Phase-1→2 item.

---

## Next tasks

**Ready to execute now (small, low-risk, both scoped against the code):**

1. **Wire the prop hover point into `samara_transient`'s IC** — model the active→passive hand-off
   end-to-end instead of asserting it. Add `hover_cut_ic(cfg, m_kg)` to `samara_prop.py` returning
   `{Vz0=0.0, Om0=Ω*}` (hover holds altitude → V_z0=0; the samara is already spinning → Ω0=Ω*, **not**
   the prop throughflow v_i and **not** ~0); import it into `samara_transient.py` and replace the
   hard-coded nominal IC (`:156`). Result is numerically identical (Vz0=0, Om0=Om_eq already), so it's
   a provenance/wiring change — nominal transition must still reproduce ≈3 s / 8 m. No circular import.

2. **Coning terminology rename** — disambiguate the two DOF that share the name. `samara_bem.Config
   .coning_deg` (blade flap, an *active* physics input at `:133`) → `blade_coning_deg`; `samara_1rev`'s
   `beta`/`beta_deg`/`coning_saturated` (gyroscopic spin-axis tilt) → `axis_tilt`/`axis_tilt_deg`/
   `axis_tilt_saturated`. Rename-only, default 10.0 unchanged → **byte-identical results**. No external
   caller passes `coning_deg` (both `Config(...)` sites use the default). Validate by diffing stdout.

**Research thread (partly blocked / external):**

3. **Geometry citation pin** *(`@science`, user-relayed)* then patch `samara_bem.py`, and the
   **resolved-Re CFD / spin-rig** (Reynolds confirmation). See the geometry section above.

**Larger future step-up:** give the *powered-patrol* swarm (`boids_swarm_3d`/`fire_swarm`/`swarm_core`)
real samara physics (energy/endurance budget from the 7.2 W hover, autorotation as an actual mode)
instead of the placeholder cruise speed — the powered half of the bi-modal story, mirroring what
`dispersal.py` did for the passive half.

---

## Working setup

- **Python env:** `~/ds/bin/python` (uv venv). ⚠️ `~/ds` was deleted and **rebuilt 2026-07-20**
  (`uv venv ~/ds --python 3.12` + numpy/scipy/matplotlib/mlx/contourpy); heavy ML/anim packages are
  *not* reinstalled — add with `uv pip install` if needed. Rebuild it if `~/ds/bin/python` is missing.
- **Run a module:** `cd 01_aero && ~/ds/bin/python samara_bem.py` (figures → `output/`, now gitignored).
- **Live rendering:** `~/ds/bin/python 03_swarm/dispersal.py --live` (native window); headless anywhere
  via `MPLBACKEND=Agg` or the `--gif`/`--mp4` flags.
- **Claude ⇄ Gemini:** shared workspace `shared-with-gemini/SHARED_MEMORY.md`. **Claude Team** (Code ⇄
  Science ⇄ Design): `shared-with-claude-team/SHARED_MEMORY.md` (holds the geometry/`@science` thread).

---

*Status convention: every result is tagged **Validated** (checked against an external anchor or
independently reproduced) or **Indicative** (order-of-magnitude; dependencies/criteria are the
deliverable). Phase-1 aero is Validated against Sycamore A; the 1/rev, prop, dispersal-coverage, and
optimum magnitudes are Indicative.*
