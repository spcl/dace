# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The facts z3 may assume in LoopToMap's dependence queries (``parallelization.facts.SolverFacts``): symbol facts at the
loop, and array facts of the index arrays a subset reads through a per-iteration binding. Each positive lifts only
with the fact; each negative removes one premise."""

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.symbolic_facts import DataFact, DataFactKind
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.analysis import smt_dependence
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.parallelization.facts import SolverFacts
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

pytestmark = pytest.mark.skipif(not smt_dependence.has_z3(), reason="needs z3")

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")


@dace.program
def strided_store(a: dace.float64[N * N], b: dace.float64[N]):
    for i in range(N):
        a[K * i] = b[i]


@dace.program
def quadratic_store(a: dace.float64[N * N + N], b: dace.float64[N]):
    for i in range(N):
        a[K * i * i + i] = b[i]


@dace.program
def store_through_bound_index(perm: dace.int64[M], out: dace.float64[M]):
    for r in range(M):
        row = perm[r]
        out[row] = 1.0


@dace.program
def store_through_rewritten_index(perm: dace.int64[M], out: dace.float64[M]):
    for r in range(M):
        row = perm[r]
        out[row] = 1.0
        perm[r] = 0


@dace.program
def read_ahead_through_index(a: dace.float64[2 * N], idx: dace.int64[N]):
    for i in range(N):
        k = idx[i]
        a[i] = a[i + 1 + k] * 0.5


def only_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))


def loop_to_map_probe(sdfg: dace.SDFG) -> LoopToMap:
    """LoopToMap's own probe of the one loop, without ParallelizeLoops' array-facts retry."""
    loop = only_loop(sdfg)
    probe = LoopToMap()
    probe.setup_match(loop.sdfg, -1, -1, {LoopToMap.loop: loop}, 0, override=True)
    probe.accepted = probe.can_be_applied(loop.parent_graph, 0, loop.sdfg)
    return probe


@pytest.mark.parametrize("program", [strided_store, quadratic_store])
def test_a_store_injective_only_for_a_positive_coefficient_lifts_under_that_fact(program):
    free = program.to_sdfg(simplify=True)
    assert not ParallelizeLoops().accepts(only_loop(free)), "without K > 0 the store may repeat"

    sdfg = program.to_sdfg(simplify=True)
    sdfg.set_symbol_assumptions("K", frozenset({symbolic.Predicate.POSITIVE}))
    ParallelizeLoops().apply_pass(sdfg, {})
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions())

    # The quadratic store stays inside ``N * N + N`` only for ``K = 1``.
    n, k = 10, 1 if program is quadratic_store else 3
    b = np.arange(n, dtype=np.float64) + 1.0
    a = np.zeros(n * n + n if program is quadratic_store else n * n)
    sdfg(a=a, b=b, N=n, K=k)
    ref = np.zeros_like(a)
    for i in range(n):
        ref[k * i * i + i if program is quadratic_store else k * i] = b[i]
    assert np.array_equal(a, ref)


def test_a_relation_on_a_symbol_the_body_reassigns_is_not_assumed():
    sdfg = strided_store.to_sdfg(simplify=True)
    sdfg.set_symbol_assumptions("K", frozenset({symbolic.Predicate.POSITIVE}))
    loop = only_loop(sdfg)
    facts = loop.sdfg.facts()
    assert SolverFacts(loop, facts, set()).symbol_axioms({"K"}), "K > 0 is assumed while K is fixed"
    assert not SolverFacts(loop, facts, {"K"}).symbol_axioms({"K"}), "a reassigned K is not"


def test_a_declared_permutation_makes_loop_to_map_prove_a_store_through_a_bound_index():
    sdfg = store_through_bound_index.to_sdfg(simplify=True)
    sdfg.add_data_assumption("perm", symbolic.permutation(0, M - 1))
    probe = loop_to_map_probe(sdfg)
    assert probe.accepted, probe.last_refusal_reason
    assert [f.describe() for f in probe.solver_facts.used] == ["perm injective with values in [0, M - 1] (declared)"]

    ParallelizeLoops().apply_pass(sdfg, {})
    perm = np.random.default_rng(3).permutation(20).astype(np.int64)
    out = np.zeros(20)
    sdfg(perm=perm, out=out, M=20)
    assert np.array_equal(out, np.ones(20))


def test_a_store_through_an_undeclared_index_stays_sequential():
    probe = loop_to_map_probe(store_through_bound_index.to_sdfg(simplify=True))
    assert not probe.accepted


def test_a_fact_about_an_index_array_the_loop_rewrites_is_not_assumed():
    sdfg = store_through_rewritten_index.to_sdfg(simplify=True)
    sdfg.add_data_assumption("perm", symbolic.permutation(0, M - 1))
    probe = loop_to_map_probe(sdfg)
    assert not probe.accepted
    assert not probe.solver_facts.used


@pytest.mark.parametrize("declared", [False, True])
def test_break_anti_dependence_snapshots_a_read_ahead_only_under_a_nonnegative_index(declared):
    sdfg = read_ahead_through_index.to_sdfg(simplify=True)
    if declared:
        sdfg.add_data_assumption("idx", DataFact(DataFactKind.NONNEGATIVE))
    renamed = BreakAntiDependence().apply_pass(sdfg, {})
    assert (renamed == 1) is declared, "a[i + 1 + idx[i]] reads ahead only when idx >= 0"
    if not declared:
        return
    n = 16
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n - 1, n).astype(np.int64)
    a = rng.random(2 * n)
    ref = a.copy()
    for i in range(n):
        ref[i] = ref[i + 1 + idx[i]] * 0.5
    sdfg(a=a, idx=idx, N=n)
    assert np.allclose(a, ref)


if __name__ == "__main__":
    for program in (strided_store, quadratic_store):
        test_a_store_injective_only_for_a_positive_coefficient_lifts_under_that_fact(program)
    test_a_relation_on_a_symbol_the_body_reassigns_is_not_assumed()
    test_a_declared_permutation_makes_loop_to_map_prove_a_store_through_a_bound_index()
    test_a_store_through_an_undeclared_index_stays_sequential()
    test_a_fact_about_an_index_array_the_loop_rewrites_is_not_assumed()
    for declared in (False, True):
        test_break_anti_dependence_snapshots_a_read_ahead_only_under_a_nonnegative_index(declared)
