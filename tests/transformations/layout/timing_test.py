# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the layout timing pass: barrier the copy-in/copy-out relayout states with an empty
side-effect tasklet (so StateFusion cannot merge them into the compute) and Timer-instrument the
compute region, so a run times the compute alone -- excluding the one-time relayout copies."""
import numpy
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.timing import (InsertLayoutTiming, is_copy_state, add_fusion_barrier, time_compute)

N = dace.symbol("N")


@dace.program
def ew(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 2.0


def _permuted():
    sdfg = ew.to_sdfg(simplify=True)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    return sdfg


def test_is_copy_state_classifies_relayout_vs_compute():
    sdfg = _permuted()
    labels = {s.label: is_copy_state(s) for s in sdfg.states()}
    assert labels["permute_in"] and labels["permute_out"]  # pure relayout copies
    assert not labels["MapState"]  # has an arithmetic tasklet


def test_insert_timing_instruments_compute_only():
    sdfg = _permuted()
    assert InsertLayoutTiming().apply_pass(sdfg, {}) == 1  # one compute state
    instrumented = {s.label for s in sdfg.states() if s.instrument != dace.InstrumentationType.No_Instrumentation}
    assert instrumented == {"MapState"}
    # the copy-in/out states carry a side-effect fusion barrier
    for label in ("permute_in", "permute_out"):
        state = next(s for s in sdfg.states() if s.label == label)
        assert any(isinstance(n, dace.nodes.Tasklet) and n.side_effects for n in state.nodes())
    sdfg.validate()


def test_barrier_prevents_state_fusion():
    from dace.transformation.interstate import StateFusion
    sdfg = _permuted()
    InsertLayoutTiming().apply_pass(sdfg, {})
    before = sdfg.number_of_nodes()
    sdfg.apply_transformations_repeated(StateFusion)
    assert sdfg.number_of_nodes() == before  # copy-in/out never merged into compute


def test_time_compute_and_correctness():
    sdfg = _permuted()
    InsertLayoutTiming().apply_pass(sdfg, {})
    sdfg.validate()

    _N = 48
    A = numpy.random.rand(_N, _N)

    def run(g):
        C = numpy.zeros((_N, _N))
        g(A=A.copy(), C=C, N=_N)
        return C

    # correctness is preserved through barrier + instrumentation
    C = run(sdfg)
    assert numpy.allclose(C, A * 2.0)
    # a compute-region time is reported
    ms = time_compute(sdfg, run, reps=3, warmup=1)
    assert ms is not None and ms >= 0.0


def test_add_fusion_barrier_is_side_effecting():
    sdfg = ew.to_sdfg(simplify=True)
    state = next(iter(sdfg.states()))
    t = add_fusion_barrier(state)
    assert t.side_effects and t.has_side_effects(sdfg)


def test_sweep_with_compute_region_timer():
    """The sweep can time the compute region (excluding relayout) via compute_region_timer: correct
    candidates get a compute time, and ranking still holds."""
    from dace.transformation.layout.brute_force import sweep, best, permutation_candidates
    from dace.transformation.layout.timing import compute_region_timer

    _N = 32
    A = numpy.random.rand(_N, _N)
    reference = {"C": A * 2.0}

    def make_for(apply):

        def make():
            g = ew.to_sdfg(simplify=True)
            apply(g)
            return g

        return make

    candidates = {name: make_for(apply) for name, apply in permutation_candidates("A", 2)}

    def run(g):
        C = numpy.zeros((_N, _N))
        g(A=A.copy(), C=C, N=_N)
        return {"C": C}

    results = sweep(candidates, run, reference, reps=3, warmup=1, timer=compute_region_timer)
    assert all(r.correct for r in results)
    assert all(r.time is not None and r.time >= 0.0 for r in results)  # compute-region time recorded
    assert best(results) is not None


if __name__ == "__main__":
    test_is_copy_state_classifies_relayout_vs_compute()
    test_insert_timing_instruments_compute_only()
    test_barrier_prevents_state_fusion()
    test_time_compute_and_correctness()
    test_add_fusion_barrier_is_side_effecting()
    test_sweep_with_compute_region_timer()
    print("timing tests PASS")
