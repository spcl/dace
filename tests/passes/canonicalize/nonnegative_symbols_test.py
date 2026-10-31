# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization assumes nonnegative exactly the symbols its caller declares so (``nonnegative_symbols``): they are
registered in the SDFG's facts before the first pass and read by every proof. Any other symbol has an unknown sign,
and a lift that needs its sign is versioned on it."""

import numpy as np

import dace
from dace import symbolic
from dace.libraries.tileops.alignment import linear_base_offset
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import collect_assumptions
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol("N", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


def relation_le(lhs: str, rhs: str) -> symbolic.Relation:
    return symbolic.Relation(symbolic.RelationKind.LE, symbolic.pystr_to_symbolic(lhs), symbolic.pystr_to_symbolic(rhs))


@dace.program
def shifted_recurrence(a: dace.float64[N + 16], b: dace.float64[N + 16]):
    for i in range(8, N):
        b[i] = b[i + K] + a[i]


def shifted_recurrence_reference(a: np.ndarray, b: np.ndarray, n: int, k: int) -> None:
    for i in range(8, n):
        b[i] = b[i + k] + a[i]


def run_shifted_recurrence(sdfg: dace.SDFG, values: tuple[int, ...]) -> None:
    compiled = sdfg.compile()
    n = 40
    rng = np.random.default_rng(0)
    for k in values:
        a, b0 = rng.random(n + 16), rng.random(n + 16)
        got, expected = b0.copy(), b0.copy()
        compiled(a=a, b=got, N=n, K=k)
        shifted_recurrence_reference(a, expected, n, k)
        assert np.allclose(got, expected), k


def versions(sdfg: dace.SDFG) -> list[str]:
    return [
        c.as_string
        for r in sdfg.all_control_flow_regions()
        if isinstance(r, ConditionalBlock)
        for c, branch in r.branches
        if c is not None
    ]


def test_an_undeclared_symbol_is_neither_assumed_nor_trapped_and_its_sign_is_versioned():
    sdfg = shifted_recurrence.to_sdfg(simplify=True)
    sdfg.name = "contract_undeclared_shift"
    canonicalize(sdfg, validate=True, nonnegative_symbols={"N"})
    predicates = sdfg.symbol_repo.params.predicates
    assert symbolic.Predicate.NONNEGATIVE not in predicates.get("K", frozenset())
    assert symbolic.Predicate.NONNEGATIVE in predicates.get("N", frozenset())
    assert not [a for a in collect_assumptions(sdfg) if "K" in {str(s) for s in a.free_symbols}]
    assert any("K" in condition for condition in versions(sdfg))
    run_shifted_recurrence(sdfg, (-3, -1, 0, 2, 5))


def test_a_declared_nonnegative_offset_renames_without_a_version():
    """``b[i] = b[i + K] + a[i]`` reads before it writes: an offset ``0`` reads the element its own iteration writes
    afterwards, so ``K >= 0`` (the declaration) is all the rename needs."""
    sdfg = shifted_recurrence.to_sdfg(simplify=True)
    sdfg.name = "contract_declared_shift"
    canonicalize(sdfg, validate=True, nonnegative_symbols={"N", "K"})
    assert versions(sdfg) == []
    run_shifted_recurrence(sdfg, (0, 2, 5))


@dace.program
def written_then_read_ahead(a: dace.float64[N + 8], d: dace.float64[N + 8], x: dace.float64[N + 8]):
    for i in range(2, N):
        a[i] = x[i] * 2.0
        d[i] = a[i] + a[i + K]


def test_a_read_ahead_after_a_write_of_the_same_iteration_is_versioned_on_positivity():
    """``a[i] = ..; d[i] = a[i] + a[i + K]``: at ``K == 0`` the read must see this iteration's write, which the
    snapshot does not hold, so even a declared nonnegative ``K`` is versioned on ``K >= 1``."""
    sdfg = written_then_read_ahead.to_sdfg(simplify=True)
    sdfg.name = "contract_written_then_read_ahead"
    canonicalize(sdfg, validate=True, nonnegative_symbols={"N", "K"})
    assert versions(sdfg) == ["(1 <= K)"]
    compiled = sdfg.compile()
    n = 30
    rng = np.random.default_rng(1)
    for k in (0, 1, 3):
        x, a0, d0 = rng.random(n + 8), rng.random(n + 8), rng.random(n + 8)
        got_a, got_d, ref_a, ref_d = a0.copy(), d0.copy(), a0.copy(), d0.copy()
        compiled(a=got_a, d=got_d, x=x, N=n, K=k)
        for i in range(2, n):
            ref_a[i] = x[i] * 2.0
            ref_d[i] = ref_a[i] + ref_a[i + k]
        assert np.allclose(got_a, ref_a) and np.allclose(got_d, ref_d), k


def test_the_declared_symbols_are_in_the_facts_before_the_first_pass():
    sdfg = shifted_recurrence.to_sdfg(simplify=True)
    sdfg.name = "contract_registered_first"
    canonicalize(sdfg, stages=[], nonnegative_symbols={"N"})
    facts = sdfg.facts()
    assert symbolic.provably_nonnegative(symbolic.pystr_to_symbolic("N"), facts)
    assert not symbolic.provably_nonnegative(symbolic.pystr_to_symbolic("K"), facts)


def test_nothing_declared_assumes_and_traps_nothing():
    sdfg = shifted_recurrence.to_sdfg(simplify=True)
    sdfg.name = "contract_none_declared"
    canonicalize(sdfg, stages=[])
    assert not sdfg.symbol_repo.params.predicates.get("N", frozenset())
    assert collect_assumptions(sdfg) == []


@dace.program
def read_ahead_by_relation(a: dace.float64[N], b: dace.float64[2 * N]):
    for i in range(N):
        b[i] = b[i + M - K] + a[i]


def test_a_read_ahead_by_a_relation_is_not_a_scan():
    """``b[i] = b[i + M - K]`` reads ahead for ``K <= M``, a relation of the facts: the stride ``K - M`` is no scan
    stride, so no ``stride >= 1`` specialization is built for it."""
    sdfg = read_ahead_by_relation.to_sdfg(simplify=True)
    sdfg.add_symbol_relation(relation_le("K", "M"))
    LiftPreprocess().apply_pass(sdfg, {})
    LoopToScan().apply_pass(sdfg, {})
    assert not [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]


def test_an_alignment_stride_nonnegative_by_a_relation_is_bounded():
    """The row stride ``N - K`` of ``A[M, N - K]`` is nonnegative for ``K <= N``: the facts prove it, the per-symbol
    contract alone did not, and the tile alignment proof refused the access."""
    sdfg = dace.SDFG("alignment_stride_by_relation")
    for name in ("M", "N", "K"):
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_symbol_relation(relation_le("K", "N"))
    sdfg.add_array("A", ["M", "N - K"], dace.float64)
    sdfg.add_array("B", ["M", "N - K"], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet, entry, exit_node = state.add_mapped_tasklet(
        "copy",
        {"i": "0:M", "j": "0:N - K"},
        {"x": dace.Memlet("A[i, j]")},
        "y = x",
        {"y": dace.Memlet("B[i, j]")},
        external_edges=True,
    )
    assert isinstance(tasklet, nodes.Tasklet)
    edge = state.in_edges(tasklet)[0]
    assert linear_base_offset(tasklet, state, sdfg, edge) is not None


if __name__ == "__main__":
    test_an_undeclared_symbol_is_neither_assumed_nor_trapped_and_its_sign_is_versioned()
    test_a_declared_nonnegative_offset_renames_without_a_version()
    test_a_read_ahead_after_a_write_of_the_same_iteration_is_versioned_on_positivity()
    test_the_declared_symbols_are_in_the_facts_before_the_first_pass()
    test_nothing_declared_assumes_and_traps_nothing()
    test_a_read_ahead_by_a_relation_is_not_a_scan()
    test_an_alignment_stride_nonnegative_by_a_relation_is_bounded()
