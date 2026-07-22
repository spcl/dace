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
    # Falsy, not ``is None``: the pass's internal trivial-tasklet cleanup may still have
    # edited the graph, and it reports that with an empty dict so callers that treat
    # ``None`` as "SDFG untouched" (the canonicalization pipeline skips validation on it)
    # are not misled. What this test pins is that no ACCUMULATOR was rewritten.
    assert not rewritten
    assert not _has_buf_transient(sdfg)


def test_refuses_non_associative_op():
    """``acc = acc - src[i]`` is not associative (only left-fold); the pass refuses."""

    @dace.program
    def left_sub(acc: dace.float64[1], src: dace.float64[6]):
        for i in range(6):
            acc[0] = acc[0] - src[i]

    sdfg = left_sub.to_sdfg(simplify=True)
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert not rewritten  # falsy, not ``is None`` -- see test_refuses_non_constant_write_index
    assert not _has_buf_transient(sdfg)


def test_map_wcr_scalar_sum_rewritten_to_buffer_and_reduce():
    """A ``dace.map`` whose body writes the accumulator via a ``wcr=+`` edge --
    the shape produced by :class:`AugAssignToWCR` plus a permissive lift -- is
    rewritten into a buffer-writing Map (WCR removed) plus a ``Reduce`` libnode.

    Verifies: the WCR edge is gone, an IntegerSort-free Reduce libnode appears
    in the SDFG, and the numeric result matches the un-rewritten reference.
    """
    import dace.libraries.standard as stdlib

    @dace.program
    def map_wcr(acc: dace.float64[1], src: dace.float64[16]):
        for i in dace.map[0:16]:
            acc[0] += src[i]

    sdfg = map_wcr.to_sdfg(simplify=True)
    reduce_before = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, stdlib.Reduce))

    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten
    assert _has_buf_transient(sdfg)
    reduce_after = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, stdlib.Reduce))
    assert reduce_after == reduce_before + 1, 'Expected exactly one new Reduce libnode.'

    # No WCR-carrying edge remains in the SDFG.
    wcr_edges = [
        e for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for e in st.edges()
        if e.data is not None and e.data.wcr is not None
    ]
    assert not wcr_edges, f'Expected no WCR edges after rewrite; got {len(wcr_edges)}.'

    src = np.random.default_rng(42).random(16)
    acc = np.array([0.7])
    ref = np.array([0.7 + src.sum()])
    sdfg(acc=acc, src=src)
    assert np.allclose(acc, ref)


def test_map_wcr_via_aug_assign_pipeline():
    """End-to-end: parallel ``+=`` loop runs ``AugAssignToWCR`` then this pass.

    The frontend lifts ``for i in dace.map[...]: acc[0] += src[i]`` directly to a
    Map with a WCR edge, but in the more general pipeline ``AugAssignToWCR`` is
    what creates the WCR shape from an explicit RMW. Run the two-step pipeline
    here to confirm composition with the existing transform.
    """
    from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

    @dace.program
    def map_rmw(acc: dace.float64[1], src: dace.float64[20]):
        for i in dace.map[0:20]:
            acc[0] += src[i]  # frontend lifts ``+=`` directly to a WCR edge

    sdfg = map_rmw.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([AugAssignToWCR()]).apply_pass(sdfg, {})
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten

    src = np.random.default_rng(11).random(20)
    acc = np.array([2.5])
    ref = np.array([2.5 + src.sum()])
    sdfg(acc=acc, src=src)
    assert np.allclose(acc, ref)


def test_seeded_inplace_accumulator_distinct_subsets():
    """A NON-fresh accumulator (a plain seed write then a WCR accumulate) routes the
    accumulator subset onto both the seed-read and the result-write edge of the
    ``combine`` tasklet. These must be DISTINCT ``Range`` objects -- sharing one
    object makes ``sdfg.validate()`` raise ``InvalidSDFGEdgeError: Duplicate subset
    detected`` (the regression that broke polybench ``lu`` / ``cholesky``).
    """
    from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

    @dace.program
    def seeded_map_rmw(acc: dace.float64[1], src: dace.float64[12]):
        acc[0] = 0.5  # plain seed write -> accumulator is NOT fresh -> seeded combine path
        for i in dace.map[0:12]:
            acc[0] += src[i]

    sdfg = seeded_map_rmw.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([AugAssignToWCR()]).apply_pass(sdfg, {})
    rewritten = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert rewritten
    # The seeded combine emits two ``acc`` memlets (seed-read + result-write); the
    # fix deep-copies the subset so validation does not flag a duplicate.
    sdfg.validate()

    # Every edge that names ``acc`` must own a distinct subset object.
    acc_subsets = [id(e.data.subset) for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for e in st.edges()
                   if e.data is not None and e.data.data == 'acc' and e.data.subset is not None]
    assert len(acc_subsets) == len(set(acc_subsets)), "shared subset object across acc memlets"

    src = np.random.default_rng(7).random(12)
    acc = np.array([0.0])
    ref = np.array([0.5 + src.sum()])
    sdfg(acc=acc, src=src)
    assert np.allclose(acc, ref)


def test_converges_on_already_rewritten():
    """Re-running the pass rewrites no further accumulator, and settles to a true no-op.

    Not a no-op on the *second* run: the first run's rewrite emits ``out = other_in``
    tasklets, which the second run's internal ``TrivialTaskletElimination`` then collapses.
    So run 2 legitimately reports a modification (empty dict = "changed, no accumulator")
    and run 3 is the first ``None``. Asserting ``second is None`` -- as this test used to --
    only held while the pass under-reported its own cleanup as no modification."""

    @dace.program
    def sum_reduce(acc: dace.float64[1], src: dace.float64[7]):
        for i in range(7):
            acc[0] = acc[0] + src[i]

    sdfg = sum_reduce.to_sdfg(simplify=True)
    first = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert first, 'the accumulator loop should have been rewritten on the first run'

    second = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert not second, 'no further accumulator may be rewritten'

    third = AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    assert third is None, 'the pass must settle to a genuine no-op'


def test_map_wcr_in_state_consumer_dependency_ordering():
    """When the reduced accumulator is consumed IN THE SAME STATE, the rewrite must
    keep the ``Reduce`` in-state and sequence it as
    ``buffer-fill map -> buf -> Reduce -> accum -> consumer``: (a) the whole buffer is
    written before it is reduced, and (b) the consumer never reads the accumulator
    before the Reduce has produced it (the atax-style 'reduce then immediately
    consume' ordering). The edges themselves are the dependency, so we assert the
    chain structurally and check the numeric result."""
    import dace.libraries.standard as stdlib

    sdfg = dace.SDFG('dep_order')
    sdfg.add_array('src', [16], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('acc', dace.float64, transient=True)
    st = sdfg.add_state('main')

    src_r = st.add_read('src')
    acc_an = st.add_access('acc')
    me, mx = st.add_map('reduce_map', {'i': '0:16'})
    t = st.add_tasklet('copy', {'__s'}, {'__a'}, '__a = __s')
    st.add_memlet_path(src_r, me, t, dst_conn='__s', memlet=dace.Memlet('src[i]'))
    st.add_memlet_path(t,
                       mx,
                       acc_an,
                       src_conn='__a',
                       memlet=dace.Memlet(data='acc', subset='0', wcr='lambda x, y: x + y'))

    # In-state consumer of the accumulator (reads acc right after the reduction).
    consume = st.add_tasklet('consume', {'__a'}, {'__o'}, '__o = __a * 2.0')
    out_w = st.add_write('out')
    st.add_edge(acc_an, None, consume, '__a', dace.Memlet('acc[0]'))
    st.add_edge(consume, '__o', out_w, None, dace.Memlet('out[0]'))
    sdfg.validate()

    assert AccumulatorToMapAndReduce().apply_pass(sdfg, {})
    sdfg.validate()

    state = sdfg.nodes()[0]
    reduces = [n for n in state.nodes() if isinstance(n, stdlib.Reduce)]
    assert len(reduces) == 1, 'expected exactly one in-state Reduce'
    red = reduces[0]

    # buffer-fill map -> buf -> Reduce: the buffer is filled by the map, then reduced.
    red_ins = state.in_edges(red)
    assert len(red_ins) == 1
    buf_an = red_ins[0].src
    assert isinstance(buf_an, nodes.AccessNode) and buf_an.data.startswith('_accum_buf_')
    assert any(isinstance(e.src, nodes.MapExit) for e in state.in_edges(buf_an)), \
        'buffer must be filled by the map exit before the Reduce reads it'

    # Reduce -> accum -> consumer: the consumer is strictly downstream of the Reduce.
    red_outs = state.out_edges(red)
    assert len(red_outs) == 1
    acc_out = red_outs[0].dst
    assert isinstance(acc_out, nodes.AccessNode) and acc_out.data == 'acc'
    assert consume in [e.dst for e in state.out_edges(acc_out)], \
        'consumer must read the accumulator produced by the Reduce (not before it)'

    src = np.random.default_rng(3).random(16)
    out = np.zeros(1)
    sdfg(src=src, out=out)
    assert np.allclose(out, src.sum() * 2.0)


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
