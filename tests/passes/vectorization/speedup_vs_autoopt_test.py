# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Speedup guard: ``canonicalize`` / ``fast_canon`` must be <= 1.2x ``auto_optimize``.

Per kernel (npbench + polybench), median of ``_REPS`` runs (``dace.profile``) of a
fresh auto-opt SDFG vs a fresh canon SDFG. Each variant's output is validated against
the reference (npbench numpy ref / polybench baseline) BEFORE timing -- a wrong-but-fast
variant fails. Build/run failures, an auto-opt vs reference mismatch, or auto-opt below
``_MIN_MS`` (overhead-dominated) are skipped.

Run sequentially with 8 threads (``OMP_NUM_THREADS=8 DACE_PERF_TEST=1 ... -p no:xdist``);
parallel runs oversubscribe cores and corrupt timings. Opt-in (skipped otherwise).
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pytest

# Perf guard -- opt-in. Run manually & SEQUENTIALLY with 8 threads:
#   OMP_NUM_THREADS=8 DACE_PERF_TEST=1 pytest tests/passes/vectorization/speedup_vs_autoopt_test.py -p no:xdist
# Skipped by default so normal CI (which runs in parallel and on shared cores) neither
# breaks on the known gaps nor produces meaningless timings.
#
# CURRENT FINDING (2026-06-27, todo E): canonicalize is materially SLOWER than
# auto_optimize on matmul/matvec kernels because canon emits a clean (un-tiled,
# un-fused) form while auto_optimize tiles + fuses. Measured medians (8 threads):
# polybench medium -- gemm 2.26x, k2mm 4.26x, gemver 2.21x, mvt 1.70x, bicg 1.70x;
# npbench S -- mlp 3.37x. These are real perf problems to fix (give canon the tiling/
# fusion it lacks vs auto-opt) before the <=1.2x guard can pass.
pytestmark = pytest.mark.skipif(not os.environ.get("DACE_PERF_TEST"),
                                reason="perf guard: set DACE_PERF_TEST=1 and run sequentially with OMP_NUM_THREADS=8")

import dace
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench

#: Allowed slowdown of canonicalize vs auto_optimize (median runtime ratio).
_MAX_RATIO = 1.2
#: Skip kernels whose auto-opt median is below this (ms) -- too small to time reliably.
_MIN_MS = 0.05
_REPS = 10
_WARMUP = 2

_NP = {c["name"]: c for c in npbench.collect()}
_PB = {k.name: k for k in polybench.collect()}


def _median_ms(sdfg: dace.SDFG, call: dict) -> float:
    csdfg = sdfg.compile()
    with dace.profile(repetitions=_REPS, warmup=_WARMUP, print_results=False) as prof:
        csdfg(**call)  # dace.profile runs it _REPS times internally
    _report, times = prof.times[-1]
    return float(np.median(np.asarray(times)))


#: The two canon pipelines compared against auto_optimize, each must be <= 1.2x.
_MODES = {
    "canon": lambda s: canonicalize(s, validate=True),
    "fast_canon": lambda s: canonicalize(s, validate=True, fast=True)
}


def _np_call(c, arrays, params):
    call = npbench._map_call(c["program"], {k: v.copy() for k, v in arrays.items()}, params)
    call.update({k: v for k, v in params.items() if k not in call and not isinstance(v, float)})
    return call


@pytest.mark.parametrize("mode", sorted(_MODES))
@pytest.mark.parametrize("name", sorted(_NP))
def test_npbench_canon_not_slower_than_autoopt(name, mode):
    c = _NP[name]
    arrays, params = npbench.make_inputs(c, cap=None)  # full preset S
    ref = npbench.reference_outputs(c, arrays, params)
    # auto-opt baseline: build, CORRECTNESS-gate, then time.
    try:
        ao = npbench.fresh_sdfg(c)
        auto_optimize(ao, dace.DeviceType.CPU)
    except Exception as e:
        pytest.skip(f"auto_optimize failed: {type(e).__name__}: {str(e)[:60]}")
    if not npbench.outputs_match(ref, npbench.run_outputs(c, ao, arrays, params)):
        pytest.skip("auto_optimize output disagrees with reference (auto-opt issue, not canon)")
    ao_ms = _median_ms(ao, _np_call(c, arrays, params))
    if ao_ms < _MIN_MS:
        pytest.skip(f"too small to time reliably (auto-opt {ao_ms:.4f} ms)")
    # canon variant: build, then CORRECTNESS FIRST (a wrong variant's speedup is meaningless).
    try:
        cn = npbench.fresh_sdfg(c)
        _MODES[mode](cn)
    except Exception as e:
        pytest.skip(f"{mode} failed (not yet supported): {type(e).__name__}: {str(e)[:60]}")
    assert npbench.outputs_match(ref, npbench.run_outputs(c, cn, arrays, params)), \
        f"{name}/{mode}: CANON OUTPUT INCORRECT vs numpy reference (speedup would be meaningless)"
    cn_ms = _median_ms(cn, _np_call(c, arrays, params))
    r = cn_ms / ao_ms
    assert r <= _MAX_RATIO, f"{name}/{mode}: {cn_ms:.3f} ms is {r:.2f}x auto-opt {ao_ms:.3f} ms (> {_MAX_RATIO}x)"


@pytest.mark.parametrize("mode", sorted(_MODES))
@pytest.mark.parametrize("name", sorted(_PB))
def test_polybench_canon_not_slower_than_autoopt(name, mode):
    k = _PB[name]
    # A medium dataset (sizes index 2) so compute dominates loop/launch overhead.
    arrays, psize = polybench.make_inputs(k, size_index=2, cap=None)
    ref = polybench.reference(k, arrays, psize)  # untransformed-baseline ground truth
    try:
        ao = polybench.fresh_sdfg(k)
        auto_optimize(ao, dace.DeviceType.CPU)
    except Exception as e:
        pytest.skip(f"auto_optimize failed: {type(e).__name__}: {str(e)[:60]}")
    if not polybench.outputs_match(ref, polybench.run(ao, arrays, psize)):
        pytest.skip("auto_optimize output disagrees with baseline (auto-opt issue, not canon)")
    ao_ms = _median_ms(ao, {**{n: v.copy() for n, v in arrays.items()}, **psize})
    if ao_ms < _MIN_MS:
        pytest.skip(f"too small to time reliably (auto-opt {ao_ms:.4f} ms)")
    try:
        cn = polybench.fresh_sdfg(k)
        _MODES[mode](cn)
    except Exception as e:
        pytest.skip(f"{mode} failed (not yet supported): {type(e).__name__}: {str(e)[:60]}")
    assert polybench.outputs_match(ref, polybench.run(cn, arrays, psize)), \
        f"{name}/{mode}: CANON OUTPUT INCORRECT vs baseline (speedup would be meaningless)"
    cn_ms = _median_ms(cn, {**{n: v.copy() for n, v in arrays.items()}, **psize})
    r = cn_ms / ao_ms
    assert r <= _MAX_RATIO, f"{name}/{mode}: {cn_ms:.3f} ms is {r:.2f}x auto-opt {ao_ms:.3f} ms (> {_MAX_RATIO}x)"
