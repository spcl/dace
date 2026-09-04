#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Optimize the npbench cavity_flow and channel_flow kernels with MPI suppressed.

A job script, run by hand -- NOT a test. Named ``test_two_stencils.py`` it matched
pytest.ini's ``python_files``, and being the only such name outside ``tests/`` it was the first
test module any unqualified run imported -- ahead of every module that imports dace. Its module
body then applied to the whole session: ``MPI4PY_RC_INITIALIZE=0`` (which is how MPI came up
single-threaded), an empty ``CUDA_VISIBLE_DEVICES``, and an ``OMPI_MCA_btl`` of its own. Keep the
name out of ``python_files``.
"""
import os
import sys
from pathlib import Path

# The checkout this job script lives in, so the ``tests.corpus.*`` imports below resolve.
DACE_REPO = Path(__file__).resolve().parent.parent

# 1. Suppress MPI / UCX / HWLOC initialization hangs BEFORE any imports
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')  # We still want 8 threads for DaCe
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader,tcp')
os.environ.setdefault('PMIX_MCA_gds', 'hash')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')
os.environ.setdefault('HWLOC_COMPONENTS', '-gl')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

# 2. Add the repo root to sys.path so we can import tests.corpus.*
if str(DACE_REPO) not in sys.path:
    sys.path.insert(0, str(DACE_REPO))

import time
import numpy as np
import dace
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target

from tests.corpus.npbench import npbench as NB

KERNELS = ('cavity_flow', 'channel_flow')
OUT_DIR = Path(__file__).resolve().parent / 'flow_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CANON_KW = dict(target='cpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True)

REPS = 7


def call_kwargs(c, arrays, params):
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call = NB._map_call(c["program"], work, params)
    symbols = {k: v for k, v in params.items() if not isinstance(v, float)}
    return {**call, **{k: v for k, v in symbols.items() if k not in call}}


def time_run(label, sdfg, kwargs):
    print(f'  [{label}] Compiling C++ (this may take a few seconds)...', flush=True)
    compiled = sdfg.compile()
    print(f'  [{label}] Running warmup...', flush=True)
    compiled(**kwargs)

    best = float('inf')
    print(f'  [{label}] Timing {REPS} reps...', flush=True)
    for _ in range(REPS):
        t0 = time.perf_counter()
        compiled(**kwargs)
        best = min(best, time.perf_counter() - t0)
    print(f'  {label:24s} best of {REPS} = {best * 1e3:9.3f} ms', flush=True)
    return best


def process(name):
    print(f'=== {name} ===', flush=True)

    print('  [setup] Loading corpus descriptor...', flush=True)
    c = NB.collect(name)[0]

    print('  [setup] Initializing inputs...', flush=True)
    arrays, params = NB.make_inputs(c, cap=None, preset='S')

    print('  [setup] Generating original SDFG...', flush=True)
    sdfg_orig = c["program"].to_sdfg(simplify=False)
    sdfg_orig.name = f"{c['name']}_{sdfg_orig.name}"
    sdfg_orig.save(str(OUT_DIR / f'{name}_original.sdfg'))

    print('  [setup] Generating auto_optimize SDFG...', flush=True)
    sdfg_auto = auto_optimize(c["program"].to_sdfg(simplify=False), dace.DeviceType.CPU)
    sdfg_auto.name = f'{sdfg_orig.name}_autoopt'
    sdfg_auto.save(str(OUT_DIR / f'{name}_autoopt.sdfg'))

    print('  [setup] Generating canonicalize SDFG...', flush=True)
    sdfg_canon = finalize_for_target(canonicalize(c["program"].to_sdfg(simplify=False), validate=True, **_CANON_KW),
                                     'cpu')
    sdfg_canon.name = f'{sdfg_orig.name}_canon'
    sdfg_canon.save(str(OUT_DIR / f'{name}_canon.sdfg'))

    kwargs = call_kwargs(c, arrays, params)

    time_run('original', sdfg_orig, kwargs)
    time_run('auto_optimize', sdfg_auto, kwargs)
    time_run('canonicalize', sdfg_canon, kwargs)


if __name__ == '__main__':
    for k in KERNELS:
        process(k)
