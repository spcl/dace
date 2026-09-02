# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fork isolation for the layout sweep: a segfault or runaway in
generated code must be a non-viable candidate, not a dead campaign -- and the OpenMP pool the
parent's parallel kernels spun up must be torn down before the fork, or the child deadlocks on its
first parallel region (libgomp, gcc's default, installs no pthread_atfork handler)."""
import ctypes
import time

import numpy
import pytest

import dace
from dace.transformation.layout.brute_force import best, sweep
from dace.transformation.layout.isolation import (OMP_PAUSE_MODES, OMP_RUNTIME_SONAMES, pause_openmp_pools,
                                                  run_isolated, set_openmp_thread_count)
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.timing import compute_region_stats_timer

N = dace.symbol("N")


@dace.program
def scale(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] * 2.0


def segfault():
    """Call a function pointer at address 0 -> SIGSEGV (a real crash, not a Python exception)."""
    ctypes.CFUNCTYPE(None)(0)()


# --------------------------------------------------------------------------- #
#  run_isolated in isolation
# --------------------------------------------------------------------------- #
def test_run_isolated_returns_child_dict():
    assert run_isolated(lambda: {"ok": True, "v": 3}) == {"ok": True, "v": 3}


def test_run_isolated_survives_segfault():
    out = run_isolated(segfault)
    assert "error" in out and "signal" in out["error"], out  # the parent survives; verdict is an error


def test_run_isolated_reports_python_error_not_crash():
    out = run_isolated(lambda: {"x": 1 / 0})
    assert "error" in out and "ZeroDivisionError" in out["error"]


def test_run_isolated_times_out():
    out = run_isolated(lambda: (time.sleep(5), {"never": True})[1], timeout=0.5)
    assert "error" in out and "timeout" in out["error"]


def test_pause_openmp_pools_is_a_safe_noop_and_accepts_modes():
    pause_openmp_pools()  # must never raise, whatever is or is not loaded
    for mode in OMP_PAUSE_MODES.values():
        pause_openmp_pools(mode)


def omp_probe():
    """In a forked child: bring a runtime up at 4 threads, show the environment write is ignored,
    then pin it. Forked so the parent's own thread count is left where the root conftest put it."""
    import ctypes
    import os
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname)  # a plain load, not RTLD_NOLOAD: bring one up if none is
        except OSError:
            continue
        break
    else:
        return {"runtime": None}
    lib.omp_get_max_threads.restype = ctypes.c_int
    lib.omp_set_num_threads.argtypes = [ctypes.c_int]
    lib.omp_set_num_threads.restype = None
    lib.omp_set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "1"
    ignored = lib.omp_get_max_threads()  # the runtime never re-reads the variable
    return {
        "runtime": soname,
        "ignored": ignored,
        "pinned": set_openmp_thread_count(1),
        "after": lib.omp_get_max_threads(),
        "env": os.environ["OMP_NUM_THREADS"],
    }


def test_set_openmp_thread_count_pins_a_runtime_that_already_read_the_environment():
    """Writing OMP_NUM_THREADS does not move a runtime that is already up: libgomp parses the
    variable in its initialiser and caches the count, so a later write reaches nobody.

    This is not academic. It is what made ``polybench/lu`` diverge between the legacy and the
    experimental CPU generator: a full-tree collection maps libgomp before
    ``tests/codegen/readable/conftest.py`` sets the variable, the 4-thread team survived, and the
    two generators then accumulated the same dot product in two different orders -- a 6.6e-04
    spread on a float32 factorisation whose emitted arithmetic is character-for-character equal.
    """
    out = run_isolated(omp_probe)
    assert "error" not in out, out
    if out["runtime"] is None:
        pytest.skip("no OpenMP runtime on this machine, so there is no thread count to pin")
    assert out["ignored"] == 4, (f"{out['runtime']}: expected the environment write to be ignored and the "
                                 f"count to stay 4, got {out['ignored']}")
    assert out["pinned"] is True, f"{out['runtime']}: set_openmp_thread_count reported the pin did not take"
    assert out["after"] == 1, f"{out['runtime']}: still at {out['after']} threads after the pin"
    assert out["env"] == "1", "the environment must also be set, for a runtime dlopened afterwards"


# --------------------------------------------------------------------------- #
#  sweep(isolate=True): the campaign survives a crash and the OMP pool is torn down
# --------------------------------------------------------------------------- #
def make_scale_sdfg(tag, perm):

    def make():
        sdfg = scale.to_sdfg(simplify=True)
        sdfg.name = f"scale_{tag}"
        if perm is not None:
            PermuteDimensions(permute_map={"A": list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})
        return sdfg

    return make


def test_isolate_sweep_survives_a_crashing_candidate(n=32):
    """Three candidates through the forked path; the middle one segfaults during its run. The
    campaign returns all three -- the crasher marked non-viable, the others correct and timed --
    instead of dying with the child."""
    a = numpy.random.default_rng(0).random((n, n))
    reference = {"C": a * 2.0}

    def run(sdfg):
        if "CRASH" in sdfg.name:
            segfault()
        c = numpy.zeros((n, n))
        sdfg(A=a.copy(), C=c, N=n)
        return {"C": c}

    candidates = {
        "identity": make_scale_sdfg("identity", None),
        "CRASH": make_scale_sdfg("CRASH", None),
        "permute": make_scale_sdfg("permute", (1, 0)),
    }
    results = sweep(candidates, run, reference, reps=2, warmup=1, timer=compute_region_stats_timer, isolate=True)
    by_name = {r.name: r for r in results}
    assert by_name["CRASH"].correct is False and "signal" in by_name["CRASH"].error
    assert by_name["identity"].correct and by_name["permute"].correct  # the campaign survived the crash
    assert by_name["identity"].time is not None  # timed inside the child, verdict marshalled back
    assert best(results).name in ("identity", "permute")


def test_isolate_sweep_runs_parallel_kernels_without_deadlock(n=48):
    """The fixture maps are OpenMP-parallel, so running one in the parent spins up the libgomp pool.
    A subsequent forked child that enters its own parallel region would deadlock unless the pool was
    torn down first -- so an isolate sweep that COMPLETES (does not time out) is the OMP-safety
    proof. Correctness is checked against the numpy oracle for good measure."""
    a = numpy.random.default_rng(1).random((n, n))
    reference = {"C": a * 2.0}

    # Poison the parent: run a parallel kernel here so the pool is live across the coming forks.
    warm = scale.to_sdfg(simplify=True)
    warm(A=a.copy(), C=numpy.zeros((n, n)), N=n)

    def run(sdfg):
        c = numpy.zeros((n, n))
        sdfg(A=a.copy(), C=c, N=n)
        return {"C": c}

    candidates = {"identity": make_scale_sdfg("identity", None), "permute": make_scale_sdfg("permute", (1, 0))}
    results = sweep(candidates, run, reference, reps=2, warmup=1, timer=compute_region_stats_timer, isolate=True)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert all(r.time is not None for r in results)


def test_isolate_refuses_gpu():
    with pytest.raises(ValueError, match="CPU-only"):
        sweep({"x": lambda: dace.SDFG("x")}, lambda s: {}, {}, device="gpu", isolate=True)


if __name__ == "__main__":
    test_run_isolated_returns_child_dict()
    test_run_isolated_survives_segfault()
    test_run_isolated_reports_python_error_not_crash()
    test_run_isolated_times_out()
    test_pause_openmp_pools_is_a_safe_noop_and_accepts_modes()
    test_isolate_sweep_survives_a_crashing_candidate()
    test_isolate_sweep_runs_parallel_kernels_without_deadlock()
    test_isolate_refuses_gpu()
    print("isolation tests PASS")
