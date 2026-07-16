# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU arm of the layout brute-force sweep: ``sweep(device="gpu")`` selects the device library
lowering and times each correct candidate with ``time_gpu`` (CUDA events on the default stream,
synchronizing on the stop event). The kernel is permuted on the pure-map path (no cuTENSOR
dependency); the sweep compiles with ``compiler.cuda.max_concurrent_streams = -1`` so dace emits the
legacy default stream, the single stream the timer records on. Correctness (vs the numpy oracle) is
the invariant; the measured time only has to be a finite positive number."""
import math

import numpy
import pytest

import dace
from dace.transformation.layout.brute_force import sweep, best, time_cpu, time_gpu
from dace.transformation.layout.permute_dimensions import PermuteDimensions

cupy = pytest.importorskip("cupy")

# Deliberately tiny: 48x32 float64 is 12 KiB per array (~36 KiB with the permuted transient). The
# sweep only has to VERIFY and produce a finite time; a big buffer would buy nothing and cost memory.
M, N = 48, 32


@dace.program
def add_one(A: dace.float64[M, N], C: dace.float64[M, N]):
    C[:] = A + 1.0


def _gpu_candidate(perm):
    """Build a fresh SDFG, permute A's layout, and move the whole thing to the GPU."""

    def make():
        sdfg = add_one.to_sdfg(simplify=True)
        PermuteDimensions(permute_map={"A": list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})
        sdfg.apply_gpu_transformations()
        return sdfg

    return make


@pytest.mark.gpu
def test_gpu_sweep_permute_times_and_verifies():
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")

    host_a = numpy.random.rand(M, N)
    reference = {"C": host_a + 1.0}

    def run(sdfg):
        a = host_a.copy()
        c = numpy.zeros((M, N))
        sdfg(A=a, C=c)  # gpu-transformed SDFG marshals host<->device internally
        return {"C": c}

    candidates = {f"permute_A_{''.join(map(str, p))}": _gpu_candidate(p) for p in ((0, 1), (1, 0))}

    # One stream for the whole program so the default-stream events bracket dace's kernels.
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        results = sweep(candidates, run, reference, device="gpu", reps=3, warmup=1)

    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    winner = best(results)
    assert winner is not None
    assert winner.time is not None and math.isfinite(winner.time) and winner.time > 0.0


@pytest.mark.gpu
def test_time_gpu_records_on_the_stream_dace_runs_on():
    """Regression guard for the stream the events are recorded on.

    dace does not launch on cupy's current stream, so events taken on a FRESH non-default stream
    would bracket no dace work and report ~0. Recorded on the default stream -- the one dace emits
    under ``max_concurrent_streams = -1`` -- the single-stream event window spans the same whole call
    the wall clock sees (both include dace's per-call marshalling), so the two agree to within a
    small factor. This pins the contract in :func:`time_gpu`'s docstring."""
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")

    host_a = numpy.random.rand(M, N)

    def run(sdfg):
        a = host_a.copy()
        c = numpy.zeros((M, N))
        sdfg(A=a, C=c)
        return c

    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        sdfg = _gpu_candidate((1, 0))()
        assert numpy.allclose(run(sdfg), host_a + 1.0)  # compile + correctness before timing
        t_gpu = time_gpu(lambda: run(sdfg), reps=3, warmup=1)
        t_wall = time_cpu(lambda: run(sdfg), reps=3, warmup=1)

    assert t_gpu > 0.0 and math.isfinite(t_gpu)
    assert 0.25 * t_wall < t_gpu < 4.0 * t_wall, f"gpu={t_gpu} wall={t_wall}: events missed the call"


if __name__ == "__main__":
    test_gpu_sweep_permute_times_and_verifies()
    test_time_gpu_records_on_the_stream_dace_runs_on()
    print("gpu sweep test PASS")
