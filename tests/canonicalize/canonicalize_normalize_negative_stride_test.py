# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`NormalizeNegativeStride`."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

N = dace.symbol("N")


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def test_reverse_loop_normalizes_to_positive_stride():
    """``for i in range(N-2, -1, -1): a[i+1] = a[i] + b[i]`` rewrites to a
    positive-stride loop driving ``_loop_pos_<N>``, with ``i`` rebound on
    each iteration so the body's memlets stay valid and iteration order
    (and therefore semantics) is preserved."""

    @dace.program
    def reverse_recurrence_reverse_loop_normalizes_to_positive_stride(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    sdfg = reverse_recurrence_reverse_loop_normalizes_to_positive_stride.to_sdfg(simplify=True)
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops = _loops(sdfg)
    assert len(loops) == 1
    loop = loops[0]
    assert loop.loop_variable.startswith("_loop_pos_")
    stride = loop_analysis.get_loop_stride(loop)
    assert stride == 1, f"expected positive stride, got {stride}"


def test_reverse_recurrence_value_preserving():
    """End-to-end: the normalized loop produces the same numeric result as the
    original Python reference (iteration order preserved by the rebinding)."""

    @dace.program
    def reverse_recurrence_reverse_recurrence_value_preserving(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    n = 8
    rng = np.random.default_rng(112)
    a0 = rng.standard_normal(n)
    b0 = rng.standard_normal(n)

    ref = a0.copy()
    for i in range(n - 2, -1, -1):
        ref[i + 1] = ref[i] + b0[i]

    sdfg = reverse_recurrence_reverse_recurrence_value_preserving.to_sdfg(simplify=True)
    NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b0.copy(), N=n)
    assert np.allclose(got, ref)


def test_positive_stride_loop_is_noop():
    """A loop with positive stride is left untouched."""

    @dace.program
    def forward(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i + 1] = a[i] + b[i]

    sdfg = forward.to_sdfg(simplify=True)
    before = sdfg.to_json()
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    assert res is None
    assert sdfg.to_json() == before, "a refusal must leave the SDFG byte-identical"


@dace.program
def reverse_stride2(a: dace.float64[N], b: dace.float64[N]):
    """``for i in range(N-1, -1, -2)`` -- stride -2, so the last iteration lands on
    ``0`` for odd ``N`` and on ``1`` for even ``N``."""
    for i in range(N - 1, -1, -2):
        a[i] = b[i] + 1.0


def test_strided_negative_loop_normalizes():
    """A stride of -2 also rewrites to a positive-stride loop."""
    sdfg = reverse_stride2.to_sdfg(simplify=True)
    res = NormalizeNegativeStride().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops = _loops(sdfg)
    assert len(loops) == 1
    assert loop_analysis.get_loop_stride(loops[0]) == 1


@pytest.mark.parametrize("n", [8, 9])
def test_strided_negative_loop_value_preserving(n):
    """Stride -2 keeps the trip count and the touched indices for an even and an odd ``N``.

    The rewrite recomputes the start from the trip count, which is where the classic
    ``range(N-1, -1, -2)`` off-by-one lives: one extra or one missing iteration shifts every
    index by two and the loop count alone would not notice."""
    rng = np.random.default_rng(113)
    a0 = rng.standard_normal(n)
    b0 = rng.standard_normal(n)
    ref = a0.copy()
    for i in range(n - 1, -1, -2):
        ref[i] = b0[i] + 1.0

    sdfg = reverse_stride2.to_sdfg(simplify=True)
    assert NormalizeNegativeStride().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    got = a0.copy()
    sdfg(a=got, b=b0.copy(), N=n)
    assert np.allclose(got, ref), f"stride -2 rewrite changed the touched indices for N={n}"


def test_loop_whose_body_reassigns_its_start_symbol_is_left_unchanged():
    """``for i = K; i >= 0; i -= 1`` with ``K = K - 1`` in the body: the rebinding would read the moved ``K``."""
    sdfg = dace.SDFG("negative_stride_start_reassigned")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("a", [16], dace.int64)
    loop = LoopRegion("L", "i >= 0", "i", "i = K", "i = i - 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_edge(body.add_tasklet("t", {}, {"o"}, "o = i"), "o", body.add_write("a"), None, dace.Memlet("a[i]"))
    loop.add_edge(body, loop.add_state("bump"), dace.InterstateEdge(assignments={"K": "K - 1"}))
    before = sdfg.to_json()

    assert NormalizeNegativeStride().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before
    a = np.full(16, -1, dtype=np.int64)
    sdfg(a=a, K=12)
    assert np.array_equal(a[:13], np.arange(13)) and np.all(a[13:] == -1)


def _down_by_symbol(predicates):
    """``for i = N - 1; i >= 0; i -= S: a[i] = i``, with ``S`` declared ``predicates``."""
    sdfg = dace.SDFG("negative_stride_symbolic")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("S", dace.int64, predicates=predicates)
    sdfg.add_array("a", ["N"], dace.int64)
    loop = LoopRegion("L", "i >= 0", "i", "i = N - 1", "i = i - S")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_edge(body.add_tasklet("t", {}, {"o"}, "o = i"), "o", body.add_write("a"), None, dace.Memlet("a[i]"))
    return sdfg


def test_symbolic_stride_the_facts_prove_negative_normalizes():
    """``i -= S`` with ``S`` positive steps by ``-S``, provably negative: rewritten, same indices touched."""
    sdfg = _down_by_symbol(frozenset({dace.symbolic.Predicate.POSITIVE}))
    assert NormalizeNegativeStride().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert loop_analysis.get_loop_stride(_loops(sdfg)[0]) == 1
    a = np.full(9, -1, dtype=np.int64)
    sdfg(a=a, N=9, S=2)
    ref = np.full(9, -1, dtype=np.int64)
    ref[8::-2] = np.arange(8, -1, -2)
    assert np.array_equal(a, ref)


def test_symbolic_stride_of_unknown_sign_is_left_unchanged():
    """``i -= S`` with nothing known of ``S``: ``-S`` may be zero or positive, so the loop stays."""
    sdfg = _down_by_symbol(frozenset())
    assert NormalizeNegativeStride().apply_pass(sdfg, {}) is None


def test_the_rebound_iterator_keeps_its_range():
    """After the rewrite ``i`` is an interstate-assigned symbol, ``i = N - 1 - _loop_pos_``; the body still knows
    ``0 <= i <= N - 1`` from that assignment and the new iterator's range."""
    from dace import symbolic
    from dace.sdfg.state import SymbolResolver

    sdfg = dace.SDFG("negative_stride_rebound_range")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("a", ["N"], dace.int64)
    loop = LoopRegion("L", "i >= 0", "i", "i = N - 1", "i = i - 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_edge(body.add_tasklet("t", {}, {"o"}, "o = i"), "o", body.add_write("a"), None, dace.Memlet("a[i]"))
    assert NormalizeNegativeStride().apply_pass(sdfg, {}) == 1
    facts = SymbolResolver().facts_at(body)
    for lhs, rhs in (("0", "i"), ("i", "N - 1")):
        relation = symbolic.Relation(
            symbolic.RelationKind.LE, symbolic.pystr_to_symbolic(lhs), symbolic.pystr_to_symbolic(rhs)
        )
        assert symbolic.ask(relation, facts) is symbolic.Truth.TRUE, (lhs, rhs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
