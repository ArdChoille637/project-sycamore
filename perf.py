"""
Project Sycamore — Apple-Silicon performance-core tuning.

On the M1 Pro (8 performance + 2 efficiency cores) the dominant scheduling fact is:
work placed on the *efficiency* cores runs 8–27× slower (measured: BLAS matmul
1742→78 GFLOPS, stencil 2.7→27 ms/iter). A foreground terminal process already
gets the P-cores, but anything launched at a lowered Quality-of-Service (QoS) —
`nohup`, `nice`, a background helper, a low-QoS parent — is pushed onto the E-cores.

This module does two things:

  1. Sets the Accelerate/BLAS thread caps (must happen *before* numpy imports) to
     the 8 P-cores. Accelerate's matmul saturates the AMX coprocessor at ~2 threads,
     but the default under-threads other ops, so setting this is a free ~25–30%.
  2. Raises the calling thread's QoS to USER_INITIATED via the pthread API, which
     pulls it back onto the P-cores under a *soft* low-QoS launch (e.g. a UTILITY
     parent: 1392→1750 GFLOPS recovered). A *hard* `background` clamp cannot be
     escaped — for that, simply don't background-clamp the process.

Two ways to use it:

    # A) the launcher (recommended — sets env BEFORE numpy, no code change):
    ./run 02_fire/fire_real.py

    # B) from a script, as the very first lines (before `import numpy`):
    import perf; perf.tune()
"""

import os
import sys
import ctypes

P_CORES = 8

# qos_class_t values from <sys/qos.h>
_QOS = {'USER_INTERACTIVE': 0x21, 'USER_INITIATED': 0x19,
        'DEFAULT': 0x15, 'UTILITY': 0x11, 'BACKGROUND': 0x09}

_THREAD_VARS = ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_MAX_THREADS')


def bump_qos(level='USER_INITIATED'):
    """Raise this thread's QoS so the scheduler prefers performance cores.
    Returns True on success (no effect under a hard `background` clamp)."""
    if sys.platform != 'darwin':
        return False
    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.dylib')
        rc = lib.pthread_set_qos_class_self_np(
            ctypes.c_uint(_QOS.get(level, _QOS['USER_INITIATED'])), ctypes.c_int(0))
        return rc == 0
    except Exception:
        return False


def set_thread_caps(threads=P_CORES):
    """Cap the BLAS/Accelerate thread pools at the P-core count. Only effective
    if called before numpy/Accelerate initialise; returns False (with a note) if
    numpy is already imported."""
    if 'numpy' in sys.modules:
        return False
    for v in _THREAD_VARS:
        os.environ.setdefault(v, str(threads))
    return True


def tune(threads=P_CORES, qos='USER_INITIATED', verbose=False):
    """Apply both: thread caps (pre-numpy) + QoS bump. Safe to call always."""
    capped = set_thread_caps(threads)
    qok = bump_qos(qos)
    if verbose:
        if not capped and 'numpy' in sys.modules:
            print("perf: numpy already imported — thread caps not applied; "
                  "use ./run or call perf.tune() before `import numpy`.")
        print(f"perf: QoS→{qos} ({'ok' if qok else 'no-op/clamped'}), "
              f"thread caps={'set' if capped else 'skipped'} ({threads})")
    return {'thread_caps': capped, 'qos': qok}


def _run_target(argv):
    """Launcher mode: tune, then run argv[0] as __main__ with its dir on sys.path
    (mimics `python <dir>/<script>.py` so same-dir imports keep working)."""
    tune()
    target = os.path.abspath(argv[0])
    sys.argv = list(argv)
    sys.path.insert(0, os.path.dirname(target))
    import runpy
    runpy.run_path(target, run_name='__main__')


def audit():
    """Print the core configuration and a quick P-core/E-core sanity benchmark."""
    import subprocess, time
    def sc(k):
        try: return subprocess.check_output(['sysctl', '-n', k], text=True).strip()
        except Exception: return '?'
    print(f"chip        : {sc('machdep.cpu.brand_string')}")
    print(f"cores       : {sc('hw.perflevel0.physicalcpu')} performance + "
          f"{sc('hw.perflevel1.physicalcpu')} efficiency")
    res = tune()
    print(f"thread caps : " + ', '.join(f"{v}={os.environ.get(v,'unset')}" for v in _THREAD_VARS))
    print(f"QoS bump    : {'applied (USER_INITIATED)' if res['qos'] else 'no-op / hard-clamped'}")
    import numpy as np
    N = 3072
    A = np.random.rand(N, N).astype('f4'); B = np.random.rand(N, N).astype('f4'); A @ B
    t = time.perf_counter()
    for _ in range(4): C = A @ B
    gf = 2*N**3/((time.perf_counter()-t)/4)/1e9
    M = 1200; u = np.random.rand(M, M)
    t = time.perf_counter()
    for _ in range(40):
        u = 0.25*(np.roll(u, 1, 0)+np.roll(u, -1, 0)+np.roll(u, 1, 1)+np.roll(u, -1, 1))
    ms = (time.perf_counter()-t)/40*1000
    where = 'PERFORMANCE cores' if gf > 800 else 'EFFICIENCY cores (!) — avoid nohup/nice/background'
    print(f"bench       : matmul {gf:.0f} GFLOPS, stencil {ms:.1f} ms/iter")
    print(f"running on  : {where}   (P-core matmul ≈1750, E-core ≈80 GFLOPS)")


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--audit':
        audit()
    elif len(sys.argv) >= 2:
        _run_target(sys.argv[1:])
    else:
        print("usage: ./run <script.py> [args...]      run with P-core tuning")
        print("       python perf.py --audit           report core config + benchmark")
        sys.exit(2)
