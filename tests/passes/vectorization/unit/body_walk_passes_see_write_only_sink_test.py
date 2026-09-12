# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vectorization passes must inspect a map body that ends in a write-only scratch scalar.

CloudSC's ``zanew_0`` is an ordinary shape: a one-element ``Register`` transient with ``Scope``
lifetime, one in-edge and no out-edge -- computed and never read again.
``SDFGState.all_nodes_between`` discards its ENTIRE walk on reaching such a node
(``dace/sdfg/graph.py``), so a pass reading that walk as "the body" answered over nothing: an
invariant reported itself satisfied, an innermost-only guard admitted a map nesting another map,
a body-nesting pass called a populated map empty, and a re-nest clustered the wrong nodes into a
cyclic state.

Each pass below is checked BOTH ways over a body carrying that sink: it must still refuse or act
on what it exists to catch, and it must still leave alone what is genuinely fine. Every fixture
also pins ``all_nodes_between`` at empty, so the tests keep naming the degeneracy they cover
rather than silently passing once some unrelated change makes the walk healthy again.
"""
import numpy as np
import pytest

import dace
from dace import SDFGState, nodes
from dace.libraries.standard.nodes import Reduce
from dace.transformation.passes.vectorization.lift_map_reduction import (LiftMapReductionToReduce, _pure_wcr_map_ok)
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import StageGlobalArrayThroughScalars
from dace.transformation.passes.vectorization.utils.map_predicates import (get_single_nsdfg_inside_map, map_body_nodes,
                                                                           map_consists_of_single_nsdfg_or_no_nsdfg)
from dace.transformation.passes.vectorization.utils.pass_invariants import assert_invariant, no_wcr_in_map_body
from dace.transformation.passes.vectorization.utils.reductions import recognize_map_reduction
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig

N = 16
SINK = 'zanew_0'


def attach_scratch_sink(sdfg: dace.SDFG, state: SDFGState, map_entry: nodes.MapEntry) -> nodes.AccessNode:
    """A ``zanew_0``-shaped write-only scratch scalar inside ``map_entry``'s scope."""
    sdfg.add_array(SINK, [1],
                   dace.float64,
                   storage=dace.dtypes.StorageType.Register,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.Scope)
    stage = state.add_tasklet('stage', {}, {'o'}, 'o = 0.0')
    state.add_nedge(map_entry, stage, dace.Memlet())
    sink = state.add_access(SINK)
    state.add_edge(stage, 'o', sink, None, dace.Memlet(f'{SINK}[0]'))
    return sink


def walk_is_degenerate(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """``all_nodes_between`` threw the whole body away on this map."""
    return len(state.all_nodes_between(map_entry, state.exit_node(map_entry))) == 0


# ---------------------------------------------------------------------------
# Fixtures, one shape per gate.
# ---------------------------------------------------------------------------


def elementwise_map(with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """``B[i] = A[i, 0] * 2`` -- an innermost, flat, WCR-free body."""
    sdfg = dace.SDFG('elementwise')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    t = state.add_tasklet('scale', {'a'}, {'o'}, 'o = a * 2.0')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    if with_sink:
        attach_scratch_sink(sdfg, state, me)
    sdfg.validate()
    return sdfg, state, me


def scatter_wcr_map(with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """``hist[idx[i]] += A[i]`` -- a per-element scatter, so the WCR lives INSIDE the body."""
    sdfg = dace.SDFG('scatter')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('IDX', [N], dace.int64)
    sdfg.add_array('H', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    t = state.add_tasklet('scatter', {'a': None, 'k': None}, {'o': None}, 'o = a')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(state.add_access('IDX'), me, t, dst_conn='k', memlet=dace.Memlet('IDX[i]'))
    partial = state.add_access('H')
    state.add_edge(t, 'o', partial, None, dace.Memlet(data='H', subset=f'0:{N}', wcr='lambda a, b: a + b',
                                                      dynamic=True))
    state.add_nedge(partial, mx, dace.Memlet())
    state.add_nedge(mx, state.add_access('H'), dace.Memlet())
    if with_sink:
        attach_scratch_sink(sdfg, state, me)
    return sdfg, state, me


def body_nest() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_array('p', [N, N], dace.float64)
    inner.add_array('q', [N], dace.float64)
    st = inner.add_state('compute', is_start_block=True)
    t = st.add_tasklet('acc', {'cur'}, {'out'}, 'out = cur * 2.0')
    st.add_edge(st.add_access('p'), None, t, 'cur', dace.Memlet('p[i, 0]'))
    st.add_edge(t, 'out', st.add_access('q'), None, dace.Memlet('q[i]'))
    return inner


def nested_body_map(with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """A map whose body is one NestedSDFG -- the shape the tile emitters require."""
    sdfg = dace.SDFG('nested_body')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    ns = state.add_nested_sdfg(body_nest(), {'p': None}, {'q': None}, symbol_mapping={'i': 'i'})
    state.add_memlet_path(state.add_access('A'), me, ns, dst_conn='p', memlet=dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_memlet_path(ns, mx, state.add_access('B'), src_conn='q', memlet=dace.Memlet(f'B[0:{N}]'))
    if with_sink:
        attach_scratch_sink(sdfg, state, me)
    sdfg.validate()
    return sdfg, state, me


def elementwise_scaled_sdfg_with_surviving_sink() -> dace.SDFG:
    """``B[i] = A[i] * 2 + 1`` plus a scratch scalar a later state reads, so no DCE removes it."""
    sym_n = dace.symbol('N', dtype=dace.int64)
    sdfg = dace.SDFG('surviving_sink')
    sdfg.add_array('A', [sym_n], dace.float64)
    sdfg.add_array('B', [sym_n], dace.float64)
    sdfg.add_array('last', [1], dace.float64)
    sdfg.add_array(SINK, [1],
                   dace.float64,
                   storage=dace.dtypes.StorageType.Register,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.SDFG)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i='0:N'))
    t = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0 + 1.0')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    stage = state.add_tasklet('stage', {'inp'}, {'o'}, 'o = inp * 0.5')
    state.add_memlet_path(state.add_access('A'), me, stage, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_edge(stage, 'o', state.add_access(SINK), None, dace.Memlet(f'{SINK}[0]'))
    post = sdfg.add_state_after(state, 'post')
    post.add_nedge(post.add_access(SINK), post.add_access('last'), dace.Memlet(f'{SINK}[0] -> [0]'))
    sdfg.validate()
    return sdfg


def map_nesting_a_map(with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """``B[i, j] = A[i, j] * 2`` as two maps -- the OUTER one is not innermost."""
    sdfg = dace.SDFG('nested_map')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    ime, imx = state.add_map('col', dict(j=f'0:{N}'))
    t = state.add_tasklet('scale', {'a'}, {'o'}, 'o = a * 2.0')
    state.add_memlet_path(state.add_access('A'), me, ime, t, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(t, imx, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i, j]'))
    if with_sink:
        attach_scratch_sink(sdfg, state, me)
    sdfg.validate()
    return sdfg, state, me


def lifted_reduction_with_dynamic_trip(
        with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry, nodes.MapExit, nodes.AccessNode, Reduce]:
    """The post-lift shape: a product-fill map over a data-dependent trip, its buffer, and Reduce."""
    sdfg = dace.SDFG('lifted')
    sdfg.add_symbol('row_end', dace.int64)
    sdfg.add_scalar('re_s', dace.int64)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_transient('buf', ['row_end'], dace.float64)
    pre = sdfg.add_state('pre', is_start_block=True)
    state = sdfg.add_state('main')
    sdfg.add_edge(pre, state, dace.InterstateEdge(assignments={'row_end': 're_s'}))
    me, mx = state.add_map('fill', dict(k='0:row_end'))
    t = state.add_tasklet('product', {'a'}, {'o'}, 'o = a * a')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[k]'))
    buf_node = state.add_access('buf')
    state.add_memlet_path(t, mx, buf_node, src_conn='o', memlet=dace.Memlet('buf[k]'))
    red = Reduce('sum', wcr='lambda a, b: a + b', identity=0.0)
    state.add_node(red)
    state.add_edge(buf_node, None, red, '_in', dace.Memlet('buf[0:row_end]'))
    state.add_edge(red, '_out', state.add_access('out'), None, dace.Memlet('out[0]'))
    if with_sink:
        attach_scratch_sink(sdfg, state, me)
    return sdfg, state, me, mx, buf_node, red


# ---------------------------------------------------------------------------
# pass_invariants.no_wcr_in_map_body -- the invariant must be able to FAIL.
# ---------------------------------------------------------------------------


def test_scatter_reduction_is_reported_although_the_body_ends_in_a_scratch_scalar() -> None:
    sdfg, state, map_entry = scatter_wcr_map(with_sink=True)
    violation = no_wcr_in_map_body(sdfg)

    assert walk_is_degenerate(state, map_entry)
    assert violation is not None
    assert 'carries WCR' in violation
    assert 'scatter -> H' in violation


def test_the_wcr_free_precondition_raises_on_a_scatter_body_ending_in_a_scratch_scalar() -> None:
    sdfg, _, _ = scatter_wcr_map(with_sink=True)

    with pytest.raises(AssertionError, match='no WCR survives inside a map body'):
        assert_invariant(no_wcr_in_map_body(sdfg), 'Vectorize', 'no WCR survives inside a map body')


def test_elementwise_body_ending_in_a_scratch_scalar_carries_no_write_conflict() -> None:
    sdfg, state, map_entry = elementwise_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert no_wcr_in_map_body(sdfg) is None
    assert all(e.data.wcr is None for e in state.edges() if e.data is not None)


# ---------------------------------------------------------------------------
# map_predicates -- single-NSDFG body recognition.
# ---------------------------------------------------------------------------


def test_a_body_nest_beside_a_scratch_scalar_is_not_a_single_nsdfg_body() -> None:
    _, state, map_entry = nested_body_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert map_consists_of_single_nsdfg_or_no_nsdfg(state, map_entry) is False
    assert get_single_nsdfg_inside_map(state, map_entry) is None
    assert sum(isinstance(n, nodes.NestedSDFG) for n in map_body_nodes(state, map_entry)) == 1


def test_a_body_that_is_only_a_nest_is_recognized_as_a_single_nsdfg_body() -> None:
    _, state, map_entry = nested_body_map(with_sink=False)
    nest = next(n for n in map_body_nodes(state, map_entry) if isinstance(n, nodes.NestedSDFG))

    assert map_consists_of_single_nsdfg_or_no_nsdfg(state, map_entry) is True
    assert get_single_nsdfg_inside_map(state, map_entry) is nest


def test_a_bare_tasklet_body_ending_in_a_scratch_scalar_holds_no_nest() -> None:
    _, state, map_entry = elementwise_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert map_consists_of_single_nsdfg_or_no_nsdfg(state, map_entry) is True
    assert get_single_nsdfg_inside_map(state, map_entry) is None


# ---------------------------------------------------------------------------
# NestInnermostMapBodyIntoNSDFG -- the body the emitters require gets built.
# ---------------------------------------------------------------------------


def test_a_body_ending_in_a_scratch_scalar_is_left_unnested_so_the_emitters_stay_away() -> None:
    """The deliberate refusal, not an oversight -- see the comment at the selection loop.

    Nesting such a body hands the tile widener a dead-end scalar it cannot lower, and the
    ``TileBinop`` refusal surfaces from the orchestrator's final ``validate`` rather than through
    its ``VectorizeUnsupported`` handler, killing the whole leg.
    ``test_a_map_body_ending_in_a_scratch_scalar_computes_every_element`` pins what keeps the
    skipped map correct instead.
    """
    sdfg, state, map_entry = elementwise_map(with_sink=True)

    NestInnermostMapBodyIntoNSDFG(vector_width=8, nest_provably_divisible=True).apply_pass(sdfg, {})

    assert walk_is_degenerate(state, map_entry)
    assert not any(isinstance(n, nodes.NestedSDFG) for n in map_body_nodes(state, map_entry))
    sdfg.validate()


def test_a_map_body_ending_in_a_scratch_scalar_computes_every_element() -> None:
    """End to end: the map the nesting pass skips must not come out strided over a scalar body."""
    sdfg = elementwise_scaled_sdfg_with_surviving_sink()
    with pytest.warns(UserWarning, match='tiled nothing'):
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa='SCALAR', validate=True)).apply_pass(sdfg, {})

    a = np.arange(1.0, 21.0)
    b = np.full(20, -7.0)
    last = np.zeros(1)
    sdfg(A=a, B=b, last=last, N=20)

    steps = [str(n.map.range.ranges[-1][2]) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert np.array_equal(b, a * 2.0 + 1.0)
    assert steps == ['1', '1']


def test_a_map_with_nothing_in_its_scope_is_left_unnested() -> None:
    sdfg = dace.SDFG('empty_scope')
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    state.add_nedge(me, mx, dace.Memlet())
    state.add_nedge(mx, state.add_access('B'), dace.Memlet())

    NestInnermostMapBodyIntoNSDFG(vector_width=8, nest_provably_divisible=True).apply_pass(sdfg, {})

    assert map_body_nodes(state, me) == []
    assert not any(isinstance(n, nodes.NestedSDFG) for n in state.nodes())


def test_a_body_already_nested_beside_its_reduction_partial_is_not_renested() -> None:
    sdfg = dace.SDFG('nested_reduction')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('part', [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    ns = state.add_nested_sdfg(body_nest(), {'p': None}, {'q': None}, symbol_mapping={'i': 'i'})
    state.add_memlet_path(state.add_access('A'), me, ns, dst_conn='p', memlet=dace.Memlet(f'A[0:{N}, 0:{N}]'))
    partial = state.add_access('part')
    state.add_edge(ns, 'q', partial, None, dace.Memlet(f'part[0] -> [0:{N}]'))
    state.add_edge(partial, None, mx, 'IN_B', dace.Memlet(data='B', subset='0', wcr='lambda a, b: a + b'))
    mx.add_in_connector('IN_B')
    mx.add_out_connector('OUT_B')
    state.add_edge(mx, 'OUT_B', state.add_access('B'), None, dace.Memlet(data='B', subset='0',
                                                                         wcr='lambda a, b: a + b'))
    pass_instance = NestInnermostMapBodyIntoNSDFG(vector_width=8, nest_provably_divisible=True)

    assert pass_instance._body_is_nested_reduction(state, me) is True


def test_a_bare_tasklet_body_beside_a_scratch_scalar_is_not_an_already_nested_reduction() -> None:
    _, state, map_entry = elementwise_map(with_sink=True)
    pass_instance = NestInnermostMapBodyIntoNSDFG(vector_width=8, nest_provably_divisible=True)

    assert walk_is_degenerate(state, map_entry)
    assert pass_instance._body_is_nested_reduction(state, map_entry) is False


# ---------------------------------------------------------------------------
# lift_map_reduction -- the innermost guard and the re-nest cluster.
# ---------------------------------------------------------------------------


def test_a_map_nesting_another_map_is_no_pure_wcr_candidate_although_it_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_nesting_a_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert _pure_wcr_map_ok(state, map_entry) is None


def test_an_innermost_map_ending_in_a_scratch_scalar_still_passes_the_pure_wcr_map_guards() -> None:
    _, state, map_entry = elementwise_map(with_sink=True)
    ok = _pure_wcr_map_ok(state, map_entry)

    assert ok is not None
    map_exit, inner, param = ok
    assert map_exit is state.exit_node(map_entry)
    assert param == 'i'
    assert any(n.data == SINK for n in inner if isinstance(n, nodes.AccessNode))


def test_the_dynamic_range_wrap_keeps_the_fill_body_inside_its_own_map() -> None:
    sdfg, state, me, mx, buf_node, red = lifted_reduction_with_dynamic_trip(with_sink=True)
    degenerate_before = walk_is_degenerate(state, me)

    LiftMapReductionToReduce._scope_dynamic_range_symbols(state, me, mx, buf_node, red)

    scope = state.scope_dict()
    product = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'product')
    wrap = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.label == 'reduce_scope')
    assert degenerate_before
    assert scope[product] is me
    assert scope[me] is wrap
    assert 'row_end' in wrap.in_connectors
    sdfg.validate()


# ---------------------------------------------------------------------------
# stage_global_array_through_scalars -- body composition gate.
# ---------------------------------------------------------------------------


def test_a_map_holding_a_body_nest_is_ineligible_for_staging_although_it_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = nested_body_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert StageGlobalArrayThroughScalars()._eligible_map_bodies(state) == {}


def test_a_flat_map_body_ending_in_a_scratch_scalar_stays_eligible_and_is_enumerated() -> None:
    _, state, map_entry = elementwise_map(with_sink=True)
    eligible = StageGlobalArrayThroughScalars()._eligible_map_bodies(state)

    assert list(eligible) == [map_entry]
    assert sorted(n.label if isinstance(n, nodes.Tasklet) else n.data
                  for n in eligible[map_entry]) == ['scale', 'stage', SINK]


# ---------------------------------------------------------------------------
# reduction_scalar_local_prep -- innermost-only widening candidate.
# ---------------------------------------------------------------------------


def test_a_map_nesting_another_map_is_no_widening_candidate_although_it_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_nesting_a_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert PrepareReductionForWidening._map_is_widening_candidate(state, state.exit_node(map_entry)) is False


def test_an_innermost_map_ending_in_a_scratch_scalar_stays_a_widening_candidate() -> None:
    _, state, map_entry = elementwise_map(with_sink=True)

    assert PrepareReductionForWidening._map_is_widening_candidate(state, state.exit_node(map_entry)) is True


# ---------------------------------------------------------------------------
# utils.reductions -- innermost-only map reduction recognition.
# ---------------------------------------------------------------------------


def test_a_map_nesting_another_map_is_not_recognized_as_a_carried_reduction() -> None:
    _, state, map_entry = map_nesting_a_map(with_sink=True)

    assert walk_is_degenerate(state, map_entry)
    assert recognize_map_reduction(state, map_entry) is None


def gather_body_nest() -> dace.SDFG:
    """An opaque body nest combining ``acc_out = acc_in + a`` -- the spmv row-reduction shape."""
    inner = dace.SDFG('combine')
    inner.add_array('a', [1], dace.float64)
    inner.add_array('acc_in', [1], dace.float64)
    inner.add_array('acc_out', [1], dace.float64)
    st = inner.add_state('c', is_start_block=True)
    t = st.add_tasklet('combine', {'cur': None, 'val': None}, {'o': None}, 'o = cur + val')
    st.add_edge(st.add_access('acc_in'), None, t, 'cur', dace.Memlet('acc_in[0]'))
    st.add_edge(st.add_access('a'), None, t, 'val', dace.Memlet('a[0]'))
    st.add_edge(t, 'o', st.add_access('acc_out'), None, dace.Memlet('acc_out[0]'))
    return inner


def test_an_opaque_body_nest_carrying_a_scalar_accumulator_is_recognized_as_a_reduction() -> None:
    sdfg = dace.SDFG('carried')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    ns = state.add_nested_sdfg(gather_body_nest(), {'a': None, 'acc_in': None}, {'acc_out': None})
    state.add_memlet_path(state.add_access('A'), me, ns, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(state.add_access('acc'), me, ns, dst_conn='acc_in', memlet=dace.Memlet('acc[0]'))
    state.add_memlet_path(ns, mx, state.add_access('acc'), src_conn='acc_out', memlet=dace.Memlet('acc[0]'))
    sdfg.validate()
    info = recognize_map_reduction(state, me)

    assert info is not None
    assert info.op == '+'
