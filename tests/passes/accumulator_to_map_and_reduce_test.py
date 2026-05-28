# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.accumulator_to_map_and_reduce.AccumulatorToMapAndReduce`.

Covers the canonical scalar-accumulator shapes the pass rewrites, the
disqualifying patterns it must refuse, and idempotence on a fixed SDFG.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.accumulator_to_map_and_reduce import AccumulatorToMapAndReduce


def _count_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _count_map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _has_buf_transient(sdfg: dace.SDFG) -> bool:
    return any(name.startswith('_accum_buf_') for name in sdfg.arrays)


def _count_reduce_nodes(sdfg: dace.SDFG) -> int:
    import dace.libraries.standard as stdlib
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, stdlib.Reduce))


def test_scalar_sum_reduce_value_preserving():
    """``acc = acc + src[i]`` on a 1-D source array; the canonical sum reduction.

    After the pass: a fresh ``_accum_buf`` transient exists, the SDFG validates
    and runs, the result matches the numpy oracle, and the resulting structure is
    (sequential loop + Reduce libnode) -- LoopToMap then parallelizes the loop.
    """

    @dace.program
    def sum_reduce(acc: dace.float64[1], src: dace.float64[10]):
        for i in range(10):
            acc[0] = acc[0] + src[i]

    sdfg = sum_reduce.to_sdfg(simplify=True)
    assert _count_loops(sdfg) == 1

    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten and len(rewritten) == 1
    assert _has_buf_transient(sdfg)
    assert _count_reduce_nodes(sdfg) == 1

    src = np.random.default_rng(0).random(10)
    acc = np.array([1.5])
    ref = np.array([1.5 + src.sum()])
    sdfg(acc=acc, src=src)
    assert np.allclose(acc, ref)

    # The delta-buffer loop parallelizes; the Reduce stays a libnode.
    maps_before = _count_map_entries(sdfg)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _count_map_entries(sdfg) > maps_before


def test_scalar_max_reduce():
    """``acc = max(acc, src[i])`` lifts to a max ``Reduce`` libnode."""

    @dace.program
    def max_reduce(acc: dace.float64[1], src: dace.float64[8]):
        for i in range(8):
            acc[0] = max(acc[0], src[i])

    sdfg = max_reduce.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten

    src = np.random.default_rng(1).random(8)
    acc = np.array([0.4])
    ref = np.array([max(0.4, src.max())])
    sdfg(acc=acc, src=src)
    assert np.allclose(acc, ref)


def test_computed_delta_not_a_direct_subscript():
    """``acc = acc + (a[i] * b[i] + c[i])`` -- the per-iteration delta is a *computed*
    expression, not a clean ``arr[i]`` slice. ``LoopToReduce`` would refuse this shape;
    this pass takes it and emits a Map that computes the delta, then a sum ``Reduce``.
    """

    @dace.program
    def computed(acc: dace.float64[1], a: dace.float64[12], b: dace.float64[12], c: dace.float64[12]):
        for i in range(12):
            acc[0] = acc[0] + (a[i] * b[i] + c[i])

    sdfg = computed.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten

    rng = np.random.default_rng(2)
    a, b, c = rng.random(12), rng.random(12), rng.random(12)
    acc = np.array([0.25])
    ref = np.array([0.25 + (a * b + c).sum()])
    sdfg(acc=acc, a=a, b=b, c=c)
    assert np.allclose(acc, ref)


def test_accumulator_with_extra_per_iteration_side_effect():
    """``LoopToReduce`` refuses a body whose accumulator is paired with a
    per-iteration write to another non-transient array (the running accumulator
    value would be observed every iteration). This pass takes it: the per-iteration
    side effect stays inside the per-iteration buffer-writing Map, and the
    accumulator becomes a Reduce libnode over the buffer.
    """

    @dace.program
    def accum_with_tap(acc: dace.float64[1], src: dace.float64[8], tap: dace.float64[8]):
        for i in range(8):
            acc[0] = acc[0] + src[i]
            tap[i] = src[i] * 2.0

    sdfg = accum_with_tap.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten

    src = np.random.default_rng(7).random(8)
    acc = np.array([0.3])
    tap = np.zeros(8)
    sdfg(acc=acc, src=src, tap=tap)
    assert np.allclose(acc, np.array([0.3 + src.sum()]))
    assert np.allclose(tap, src * 2.0)


def test_refuses_non_constant_write_index():
    """``arr[i] = arr[i] + delta`` is a per-iteration write, not an accumulator;
    nothing should be rewritten.
    """

    @dace.program
    def per_iter(arr: dace.float64[8], delta: dace.float64[8]):
        for i in range(8):
            arr[i] = arr[i] + delta[i]

    sdfg = per_iter.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert rewritten is None
    assert not _has_buf_transient(sdfg)


def test_refuses_non_associative_op():
    """``acc = acc - src[i]`` is not associative (only left-fold); the pass refuses."""

    @dace.program
    def left_sub(acc: dace.float64[1], src: dace.float64[6]):
        for i in range(6):
            acc[0] = acc[0] - src[i]

    sdfg = left_sub.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert rewritten is None


def test_idempotent_on_already_rewritten():
    """Re-running the pass on its own output is a no-op (the matcher skips loops
    whose accumulator is a ``_accum_buf_`` transient, but more fundamentally the
    rewritten loops are gone after the first run)."""

    @dace.program
    def sum_reduce(acc: dace.float64[1], src: dace.float64[7]):
        for i in range(7):
            acc[0] = acc[0] + src[i]

    sdfg = sum_reduce.to_sdfg(simplify=True)
    AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    second = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert second is None


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
