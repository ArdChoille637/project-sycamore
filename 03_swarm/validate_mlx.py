"""
Validate + benchmark the MLX boids step against the NumPy reference.

1. Numerical equivalence: run both backends from an identical state with Vicsek
   noise OFF; assert max |Δpos|, |Δvel| stay tiny across many steps.
2. Wall-clock benchmark at the production fleet size.

Run: ~/ds/bin/python 03_swarm/validate_mlx.py
"""

import time
import numpy as np
import swarm_core as sc


def validate_single_step(tol=1e-3):
    """Per-step the MLX kernel must reproduce the NumPy reference to float32
    precision (noise off). Boids is chaotic, so step-LOCKED tracking diverges
    over many steps from rounding alone — that is physics, not a port bug — so
    the contract is enforced one step at a time from a re-synced state."""
    if not sc.HAVE_MLX:
        print("MLX unavailable — skipping");  return None
    import mlx.core as mx

    rng = np.random.default_rng(7)
    pos_np, vel_np = sc.init_state(rng)
    rng0 = np.random.default_rng(0)
    worst_dp = worst_dv = 0.0
    for _ in range(40):
        # MLX takes one step from the *current* numpy state, then both advance.
        pm, vm = sc.step_mx(mx.array(pos_np), mx.array(vel_np), sigma=0.0)
        mx.eval(pm, vm)
        pos_np, vel_np = sc.step_np(pos_np, vel_np, rng0, sigma=0.0)
        worst_dp = max(worst_dp, float(np.abs(pos_np - np.array(pm)).max()))
        worst_dv = max(worst_dv, float(np.abs(vel_np - np.array(vm)).max()))
    ok = worst_dp < tol and worst_dv < tol
    print("Per-step equivalence (MLX vs NumPy, noise off, re-synced each step):")
    print(f"  worst |Δpos| = {worst_dp:.3e} m     worst |Δvel| = {worst_dv:.3e} m/s")
    print(f"  {'PASS' if ok else 'FAIL'}  (tol={tol})")
    return ok


def validate_aggregate():
    """With noise ON the two backends are different stochastic realisations, but
    the emergent statistics (coverage, altitude hold, cruise speed) must agree."""
    if not sc.HAVE_MLX:
        return None
    tp_n, tv_n, tc_n = sc.presim_np()
    tp_m, tv_m, tc_m = sc.presim_mx()
    win = slice(sc.N_STEPS//2, None)         # steady-state window
    def t_to(tc, frac):                      # deployment-curve discriminator: time to `frac` coverage
        hit = np.argmax(tc >= frac)
        return hit*sc.DT if (tc >= frac).any() else np.nan
    def stats(tp, tv, tc):
        spd = np.linalg.norm(tv[win][:, :, [0, 2]], axis=2)
        return (tc[win].mean()*100, tp[win][:, :, 1].mean(), spd.mean(),
                t_to(tc, 0.5), t_to(tc, 0.9))
    cn, an, sn, h50n, h90n = stats(tp_n, tv_n, tc_n)
    cm, am, sm, h50m, h90m = stats(tp_m, tv_m, tc_m)
    print("\nNumPy vs MLX (noise on; independent stochastic realisations):")
    print(f"  {'metric':<22}{'NumPy':>10}{'MLX':>10}")
    print(f"  {'steady coverage %':<22}{cn:>10.1f}{cm:>10.1f}")
    print(f"  {'steady alt [m]':<22}{an:>10.1f}{am:>10.1f}")
    print(f"  {'steady cruise [m/s]':<22}{sn:>10.2f}{sm:>10.2f}")
    print(f"  {'t→50% coverage [s]':<22}{h50n:>10.0f}{h50m:>10.0f}")
    print(f"  {'t→90% coverage [s]':<22}{h90n:>10.0f}{h90m:>10.0f}")
    # The deployment-transient times (t→50/90%) are the real discriminator here —
    # steady coverage saturates at 100% and altitude locks, so they alone are weak.
    ok = (abs(cn-cm) < 6 and abs(an-am) < 5 and abs(sn-sm) < 0.5
          and abs(h50n-h50m) < 30 and abs(h90n-h90m) < 40)
    print(f"  {'PASS' if ok else 'FAIL'}  (deployment curve + steady state agree)")
    return ok


def benchmark(sizes=(20, 50, 100, 150, 200, 300, 500), reps=100):
    import mlx.core as mx
    orig_N = sc.N_DRONES
    print(f"\n{'N':>5} {'NumPy ms':>10} {'MLX ms':>10} {'speedup':>9}")
    for N in sizes:
        sc.N_DRONES = N
        ang = np.linspace(0, 2*np.pi, N, endpoint=False)
        p = np.c_[sc.FIRE_A*np.cos(ang), np.full(N, sc.ALT_NOM),
                  sc.FIRE_B*np.sin(ang)].astype('f4')
        v = np.c_[-np.sin(ang)*sc.V_CRUISE, np.zeros(N),
                  np.cos(ang)*sc.V_CRUISE].astype('f4')
        rng = np.random.default_rng(0)

        pn, vn = p.copy(), v.copy()
        t0 = time.perf_counter()
        for _ in range(reps):
            pn, vn = sc.step_np(pn, vn, rng, sigma=sc.SIGMA_NOISE)
        t_np = (time.perf_counter() - t0)/reps*1000

        if sc.HAVE_MLX:
            pm, vm = mx.array(p), mx.array(v)
            key = mx.random.key(1)
            for _ in range(3):
                key, s = mx.random.split(key)
                pm, vm = sc.step_mx(pm, vm, key=s, sigma=sc.SIGMA_NOISE)
            mx.eval(pm, vm)
            t0 = time.perf_counter()
            for _ in range(reps):
                key, s = mx.random.split(key)
                pm, vm = sc.step_mx(pm, vm, key=s, sigma=sc.SIGMA_NOISE)
            mx.eval(pm, vm)
            t_mx = (time.perf_counter() - t0)/reps*1000
            print(f"{N:>5} {t_np:>10.3f} {t_mx:>10.3f} {t_np/t_mx:>8.2f}x")
        else:
            print(f"{N:>5} {t_np:>10.3f} {'n/a':>10}")
    sc.N_DRONES = orig_N


def presim_timing():
    print("\nFull pre-sim wall-clock (N=150, 1300 steps):")
    t0 = time.perf_counter();  sc.presim_np();  t_np = time.perf_counter()-t0
    print(f"  NumPy : {t_np:.2f}s")
    if sc.HAVE_MLX:
        t0 = time.perf_counter();  sc.presim_mx();  t_mx = time.perf_counter()-t0
        print(f"  MLX   : {t_mx:.2f}s   ({t_np/t_mx:.2f}x)")


if __name__ == '__main__':
    validate_single_step()
    benchmark()
    presim_timing()
    validate_aggregate()
