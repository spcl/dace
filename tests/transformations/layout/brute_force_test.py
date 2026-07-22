# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the brute-force GLOBAL layout sweep engine.

The sweep compiles/runs/verifies each global layout candidate against a numpy oracle and ranks the
correct ones. Correctness (every transparent candidate verifies, incorrect ones are flagged and not
timed, best() returns a correct one) is the invariant asserted here; timing is best-effort and NOT
asserted (noisy on a shared host)."""
import numpy
import dace

from dace.transformation.layout.brute_force import (sweep, best, time_cpu, permutation_candidates, SweepResult)

N = dace.symbol("N")


@dace.program
def ew2d(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 2.0 + 1.0


def test_time_cpu_returns_positive():
    t = time_cpu(lambda: sum(range(1000)), reps=3, warmup=1)
    assert isinstance(t, float) and t >= 0.0


def test_sweep_permutation_candidates_all_verify():
    """A global dimension permutation of A is transparent (add_permute_maps wraps the input), so
    every permutation candidate verifies against the numpy oracle."""
    _N = 8
    A = numpy.random.rand(_N, _N)
    reference = {"C": A * 2.0 + 1.0}

    def make_for(apply):

        def make():
            sdfg = ew2d.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in permutation_candidates("A", 2)}
    assert set(candidates) == {"permute_A_01", "permute_A_10"}

    def run(sdfg):
        C = numpy.zeros((_N, _N))
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None and best(results).correct


def test_sweep_flags_and_ranks():
    """An incorrect candidate is flagged (not timed) and ranked after correct ones; a correct one is
    timed and comes first."""
    _N = 8
    A = numpy.random.rand(_N, _N)
    good_ref = {"C": A * 2.0 + 1.0}

    def make_good():
        return ew2d.to_sdfg(simplify=True)

    def run(sdfg):
        C = numpy.zeros((_N, _N))
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    # "bad" candidate: compare against a wrong reference to force a mismatch.
    wrong_ref = {"C": A * 5.0}
    results_bad = sweep({"id": make_good}, run, wrong_ref, do_time=True)
    assert results_bad[0].correct is False and results_bad[0].time is None  # incorrect -> not timed

    results_good = sweep({"id": make_good}, run, good_ref, do_time=True, reps=2, warmup=1)
    assert results_good[0].correct and results_good[0].time is not None and results_good[0].time >= 0.0


def test_sweep_build_failure_is_not_viable():
    """A candidate whose builder raises is recorded as not-correct with the error, not propagated."""

    def boom():
        raise RuntimeError("cannot build")

    results = sweep({"boom": boom}, lambda s: {}, {"C": numpy.zeros(1)}, do_time=False)
    assert results[0].correct is False and "cannot build" in results[0].error
    assert best(results) is None


def test_sweep_timing_failure_does_not_demote_correct():
    """A timer that raises (e.g. cupy absent, no device) must NOT flip a verified-correct candidate
    to incorrect: timing is advisory. best() still returns it; the timing error is kept in metadata."""
    _N = 8
    A = numpy.random.rand(_N, _N)
    reference = {"C": A * 2.0 + 1.0}

    def make():
        sdfg = ew2d.to_sdfg(simplify=True)
        return sdfg

    def run(sdfg):
        C = numpy.zeros((_N, _N))
        sdfg(A=A.copy(), C=C, N=_N)
        return {"C": C}

    def raising_timer(sdfg, run_, reps, warmup):
        raise ModuleNotFoundError("No module named 'cupy'")

    results = sweep({"id": make}, run, reference, do_time=True, timer=raising_timer)
    assert results[0].correct is True  # verified despite the timer blowing up
    assert results[0].time is None
    assert "ModuleNotFoundError" in results[0].metadata.get("timing_error", "")
    assert best(results) is not None and best(results).correct


def test_sweep_bad_device_rejected():
    """sweep validates device up front rather than swallowing it per-candidate."""
    import pytest
    with pytest.raises(ValueError):
        sweep({}, lambda s: {}, {}, device="tpu")


def test_single_default_stream_forces_minus_one():
    """The 1-stream mode is ALWAYS -1 (legacy default stream) inside the context, restored after."""
    from dace.transformation.layout.brute_force import single_default_stream
    before = dace.Config.get("compiler", "cuda", "max_concurrent_streams")
    with single_default_stream():
        assert dace.Config.get("compiler", "cuda", "max_concurrent_streams") == -1
    assert dace.Config.get("compiler", "cuda", "max_concurrent_streams") == before


if __name__ == "__main__":
    test_time_cpu_returns_positive()
    test_sweep_permutation_candidates_all_verify()
    test_sweep_flags_and_ranks()
    test_sweep_build_failure_is_not_viable()
    test_sweep_timing_failure_does_not_demote_correct()
    test_sweep_bad_device_rejected()
    test_single_default_stream_forces_minus_one()
    print("brute_force tests PASS")
