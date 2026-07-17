#!/usr/bin/env python3
"""Confirm + localize the parent-affinity clobber behind the '72 threads on core 0'
collapse (cell D: child libgomp saw OMP_PLACES='{0}' -- an inherited 1-core mask).
Prints os.sched_getaffinity(0) of the PARENT after every stage of the exact repro
sequence, then spawns one child per interesting point to show what it inherits.
Suspect: libgomp binds the master thread to one place (OMP_PROC_BIND=close) at the
parent's FIRST omp parallel region; every later subprocess inherits that mask."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def aff(label):
    m = os.sched_getaffinity(0)
    print(f'  parent affinity after {label:45s}: {len(m):3d} cpus '
          f'({"{" + str(min(m)) + ".." + str(max(m)) + "}" if len(m) > 1 else set(m)})', flush=True)

def child_aff(label):
    r = subprocess.run(['python3', '-c', 'import os; print(len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0))[:4])'],
                       capture_output=True, text=True, timeout=60)
    print(f'  CHILD spawned after {label:45s}: inherits {r.stdout.strip()}', flush=True)

aff('interpreter start')
import engine  # noqa: E402
aff('import engine')
engine.configure_dace_process()
aff('engine.configure_dace_process()')
import dace  # noqa: E402
import numpy as np  # noqa: E402
aff('import dace, numpy')

N = 2048

@dace.program
def gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = A @ B

sdfg = gemm.to_sdfg(simplify=True)
aff('to_sdfg')
sdfg = engine.pipeline_canon(sdfg, device='cpu')
aff('pipeline_canon')
sdfg.name = 'blas_affinity_trace_gemm'
csdfg = sdfg.compile()
aff('sdfg.compile() [loads .so]')
child_aff('compile')

rng = np.random.default_rng(0)
A = np.asfortranarray(rng.random((N, N)))
B = np.asfortranarray(rng.random((N, N)))
C = np.zeros((N, N), order='F')
csdfg(A=A, B=B, C=C)
aff('first csdfg call (omp region ran)')
child_aff('first csdfg call')
