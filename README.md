# Project Sycamore — biomimetic wildfire drone-swarm simulations

Simulation suite for an undergraduate research project (Eastern Kentucky University) on a
**sycamore-biomimetic, bi-modal wildfire drone swarm** — a UAV swarm whose airframe mimics the
autorotating maple/sycamore *samara*, operating in two modes (powered flight + passive autorotation
descent) for wildfire detection and suppression.

The code models the system end to end: single-vehicle **autorotation aerodynamics**, **wildfire
spread** over real landscapes, a **3-D swarm** that patrols and tracks a live fire, and the
**coupling** between them — accelerated on Apple-Silicon GPUs via [MLX](https://github.com/ml-explore/mlx),
with a NumPy fallback everywhere.

> **Real vs. synthetic, kept explicit.** Wind, terrain, and fuel can be driven by real public
> datasets (NOAA HRRR, USGS 3DEP, LANDFIRE). Every figure is labelled with its data provenance, and
> the remaining synthetic inputs are flagged rather than glossed over (see [Status](#status)).

---

## What's inside

| Module | What it does |
|---|---|
| `01_aero/samara_bem.py` | Blade-element-momentum autorotation solver for the samara airframe (descent rate, RPM, pitch schedule for autorotation equilibrium). |
| `05_cfd/lbm_mlx.py` | **D2Q9 Lattice-Boltzmann** CFD core (MLX/Metal GPU, NumPy fallback) — a native high-fidelity aero path that resolves the leading-edge vortex the BEM model can't. Streaming = array roll, collision = element-wise: the workload Apple Silicon accelerates well. |
| `05_cfd/lbm_validate.py` | Lid-driven cavity validation vs **Ghia et al. (1982)** — centreline RMSE ≈ 0.006 of lid speed. |
| `05_cfd/lbm_cylinder.py` | Flow past a cylinder — Kármán vortex street + momentum-exchange forces (qualitative; St/Cd elevated by channel confinement). |
| `05_cfd/lbm_samara.py` | Samara section at high angle of attack — the **leading-edge vortex** the attached-flow BEM model omits. |
| `05_cfd/lbm3d_mlx.py` | **3-D D3Q19 Lattice-Boltzmann** (MLX) — the "4-D" (3-D space + time) keystone. Fully GPU-resident, fused per-population collision, fp16-storage option. Validated vs the **Taylor-Green vortex** (analytic decay rate 4νk², **0.0% error**). 256³ fits 16 GB at ~8 GB peak (~22 MLUPS) — the practical ceiling on this machine. |
| `05_cfd/lbm3d_flow.py` | 3-D obstacle flow on the core (solid bounce-back + velocity inlet + outlet + momentum-exchange force) — the infrastructure toward the rotating samara. Flow validated on a sphere (clean recirculation wake); absolute force uncalibrated (same factor as the 2-D MEM). Next: a rotating-frame body force for the spanwise-stabilised LEV. |
| `02_fire/rothermel.py` | Rothermel (1972) surface-fire rate-of-spread + Anderson fuel models + Huygens elliptical perimeter growth. |
| `02_fire/rothermel_ca.py` | **Rothermel cellular automaton** — the point model turned into a spatial fire over terrain. Per-cell wind⊕slope ellipse spread; fire front as a GPU min-arrival-time relaxation. Spatial fuel + spatial wind aware. |
| `02_fire/real_terrain.py` | Real **USGS 3DEP** bare-earth DEM via the ImageServer REST API (pyproj + Pillow, no GDAL). |
| `02_fire/hrrr_wind.py` | Real **NOAA HRRR** 10 m wind via [Herbie](https://herbie.readthedocs.io), converted to a Rothermel mid-flame wind. |
| `02_fire/real_fuel.py` | Real **LANDFIRE FBFM40** fuel models (all 40 Scott & Burgan 2005 models + nonburnable firebreaks). |
| `02_fire/fuel_moisture.py` | Spatial **dead-fuel moisture** from real HRRR 2 m weather × terrain microclimate (elevation lapse + aspect heat load; EMC via Simard 1968). |
| `02_fire/terrain_wind.py` | **Mass-conserving terrain wind downscaling** (the WindNinja core) — solves ∇·(d∇φ)=∇·(dV₀) by conjugate gradient so the flow follows the real DEM. |
| `02_fire/fire_real.py` | Fire CA on **all-real** inputs — real wind, terrain, fuel, and moisture for a real place/time. |
| `02_fire/fire_terrain_wind.py` | Uniform vs. terrain-downscaled wind, side by side. |
| `02_fire/fire_moisture.py` | Spatial vs. uniform fuel moisture (isolates the terrain-microclimate effect). |
| `02_fire/progressive_res.py` | Resolution study: the same real-data fire at 90→10 m, with the MLX-vs-NumPy crossover. |
| `03_swarm/swarm_core.py` | 3-D boids physics (separation · alignment · fire-tracking · arc-index spread · tangential patrol · altitude PD). NumPy reference **and** MLX port, numerically validated. |
| `03_swarm/boids_swarm_3d.py` | 150-drone 3-D swarm: deploy from a base, self-organise to blanket a fire perimeter (matplotlib Axes3D). |
| `03_swarm/fire_swarm.py` | **Coupled fire ↔ swarm** — the swarm patrols the *live* Rothermel fire front as it grows and advances. |
| `03_swarm/validate_mlx.py` | MLX-vs-NumPy equivalence + benchmark + steady-state statistics. |
| `sycamore_teach.py` | Manim educational animation of the architecture, BEM, and boids. |

## Apple Silicon / MLX

The grid-local, FP32 workloads — the fire CA, the boids physics, the wind Poisson solve — map well to
the M-series GPU through MLX. Measured on an M1 Pro:

- **Boids step:** crossover at ~110 drones; **1.6× at N=150**, up to **7× at N=500**.
- **Fire CA:** crossover at ~30 m / ~50k cells; **~1.8× at 10 m (440k cells)** — the advantage grows
  with resolution.

FP64-heavy work (the BEM root-finding, any true CFD) stays on the CPU, where Apple-Silicon double
precision is emulated and slow.

## Performance cores

The M1 Pro has **8 performance + 2 efficiency cores**, and the scheduler places work on one or the
other by *Quality of Service* (QoS). Work pushed onto the E-cores runs **5–27× slower** (measured:
BLAS 1742→78 GFLOPS, the NumPy presim 1.9→9.7 s). A foreground terminal process already gets the
P-cores; the trap is lowering its QoS — `nohup`, `nice`, `taskpolicy -c background`, or a low-QoS
parent — which forces the E-cores and **cannot be undone from inside the process**.

So the single most important rule is: **don't background-clamp the heavy runs.** Beyond that,
[`perf.py`](perf.py) + the [`run`](run) launcher add two things for free:

```bash
./run 02_fire/fire_real.py        # P-core QoS bump + Accelerate thread caps (8), then runs it
python perf.py --audit            # report cores + a P-core/E-core sanity benchmark
```

- **Thread caps** (`VECLIB_MAXIMUM_THREADS=8`, set before NumPy imports) lift BLAS-heavy ops ~25–30%
  over the default (NumPy here uses Apple's **Accelerate**, the optimal Apple-Silicon BLAS).
- **A QoS bump** (`pthread_set_qos_class_self_np` → `USER_INITIATED`) pulls the process back onto the
  P-cores under a *soft* low-QoS launch (a UTILITY parent recovered 1392→1750 GFLOPS). It can't escape
  a *hard* `background` clamp — nothing can.

Honest scope: the genuinely heavy compute is already GPU-parallel via MLX, and the remaining CPU
paths (the BEM continuation, the wind CG, the relaxation sweeps) are *sequential* — not parallelisable
across cores. So this isn't about spreading work over more cores; it's about making sure the cores it
*does* run on are the fast ones.

## Quickstart

```bash
pip install -r requirements.txt          # numpy/scipy/matplotlib/mlx/herbie-data/pyproj/...

# Fire over a real Sierra landscape (real wind + terrain + fuel)
python 02_fire/fire_real.py

# Resolution scaling study (90 → 10 m) with the MLX/NumPy crossover
python 02_fire/progressive_res.py

# 150-drone 3-D swarm deploying to a fire perimeter (opens a window)
python 03_swarm/boids_swarm_3d.py

# Coupled fire ↔ swarm on real data (opens a window)
python 03_swarm/fire_swarm.py     # see build_fire_real() for the all-real scenario

# Numerical validation + benchmark
python 03_swarm/validate_mlx.py
python 02_fire/rothermel_ca.py --validate
```

Figures are written to each module's `output/` directory. The 3-D viewers open a native window;
run them from a terminal (not a headless/background process). Prefix any command with `./run`
(e.g. `./run 02_fire/fire_real.py`) for performance-core scheduling + tuned BLAS threads — see
[Performance cores](#performance-cores).

## Data sources & attribution

- **Wind** — NOAA [High-Resolution Rapid Refresh (HRRR)](https://rapidrefresh.noaa.gov/hrrr/), retrieved with [Herbie](https://herbie.readthedocs.io).
- **Terrain** — [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program).
- **Fuel** — [LANDFIRE](https://landfire.gov/) FBFM40, [Scott & Burgan (2005) RMRS-GTR-153](https://research.fs.usda.gov/treesearch/9521).
- Fire physics — Rothermel (1972) INT-115; Anderson (1982, 1983); Albini & Baughman (1979); Andrews (2012, 2018).

All datasets above are U.S. public-domain / open data. The DEM and fuel tiles are fetched on demand
and cached locally; nothing proprietary is redistributed here.

## Status

Real-data layers wired in: **wind** (HRRR + mass-conserving terrain downscaling), **terrain** (3DEP),
**fuel** (LANDFIRE FBFM40), **fuel moisture** (HRRR 2 m weather × terrain microclimate). Honestly
remaining simplified:

- **Wind is a single time snapshot** — time-varying wind needs a time-stepping front model, not the
  current static min-arrival-time relaxation.
- The fuel single-class reduction assumes fire-season herbaceous curing (documented in `real_fuel.py`);
  the moisture aspect heat-load is a parameterised offset (documented in `fuel_moisture.py`).

## License

[MIT](LICENSE) © 2026 Michael Ray Gregory. The bundled public datasets (HRRR, 3DEP, LANDFIRE) are
U.S. public-domain / open data under their own terms.

🤖 Simulation suite developed with [Claude Code](https://claude.com/claude-code).
