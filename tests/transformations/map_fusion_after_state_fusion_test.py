# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MapFusion applied to SDFGs produced by ``StateFusionExtended``.

State fusion collapses the interstate barrier and re-imposes the lost ordering with
happens-before edges (empty memlets) for the WAR / WAW hazards, while RAW keeps flowing
through a merged data node. Those artefacts land exactly where MapFusion pattern-matches,
so this file pins the interaction from all three sides:

  * correctness -- fusing two Maps turns a whole-Map ordering ("all of the first Map, then
    all of the second") into a PER-ITERATION one. That is only valid when no iteration can
    collide with a *different* iteration; for a cross-iteration WAR/WAW it is a miscompile,
    and MapFusion must refuse.
  * opportunity, RAW -- RAW is handled by merging the common data node rather than by an
    ordering edge, so the ``MapExit -> AccessNode -> MapEntry`` pattern MapFusionVertical
    needs survives state fusion intact.
  * opportunity, WAR/WAW -- when the hazard *is* iteration-private, MapFusionHorizontal
    fuses the Maps anyway: it drops the outer ordering edges (they would become a self loop
    on the fused Map) and re-establishes the ordering *inside* the fused scope, where per
    iteration is all that is needed.
"""

import copy

import networkx as nx
import numpy as np

import dace
from dace import SDFG, InterstateEdge, Memlet, dtypes
from dace.sdfg import nodes as dnodes
from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
from dace.transformation.interstate import StateFusionExtended

N = 8


def _count_maps(sdfg: SDFG) -> int:
    return sum(1 for st in sdfg.states() for n in st.nodes() if isinstance(n, dnodes.MapEntry))


def _count_empty_edges(sdfg: SDFG) -> int:
    return sum(1 for st in sdfg.states() for e in st.edges() if e.data is not None and e.data.is_empty())


def _run(sdfg: SDFG, tag: str, **arrays):
    s = copy.deepcopy(sdfg)
    s.name = f'{sdfg.name}_{tag}'
    args = {k: v.copy() for k, v in arrays.items()}
    s(**args)
    return args


def _state_fuse(sdfg: SDFG) -> SDFG:
    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(StateFusionExtended)
    fused.validate()
    return fused


def _map_fuse(sdfg: SDFG) -> SDFG:
    fused = copy.deepcopy(sdfg)
    fused.apply_transformations_repeated(MapFusionVertical)
    fused.apply_transformations_repeated(MapFusionHorizontal)
    fused.validate()
    return fused


def _assert_matches(ref, got, where: str):
    for name in ref:
        assert np.allclose(got[name], ref[name], rtol=1e-13, atol=1e-13), \
            f'{where}: {name} diverges: {got[name]} vs ref {ref[name]}'


def _tasklet(sdfg: SDFG, code: str) -> dnodes.Tasklet:
    """The single tasklet whose body is `code`."""
    found = [
        n for st in sdfg.states() for n in st.nodes() if isinstance(n, dnodes.Tasklet) and n.code.as_string == code
    ]
    assert len(found) == 1, f'expected exactly one tasklet {code!r}, found {len(found)}'
    return found[0]


def _assert_ordered_inside_one_map(sdfg: SDFG, first_code: str, second_code: str):
    """`first_code` must run before `second_code`, and both inside the same Map scope.

    Numerics alone do not prove this: an unordered pair may still be *emitted* in the
    right order by chance, and would then be reordered by the next pass that runs.
    """
    state = sdfg.states()[0]
    first, second = _tasklet(sdfg, first_code), _tasklet(sdfg, second_code)
    assert nx.has_path(state._nx, first, second), f'{first_code!r} is not ordered before {second_code!r}'
    scope = state.scope_dict()
    assert isinstance(scope[first], dnodes.MapEntry), f'{first_code!r} is not inside a Map'
    assert scope[first] is scope[second], 'the two tasklets did not end up in the same Map scope'


def _war_two_maps(read_offset: int) -> SDFG:
    """s1: ``B[i] = A[i + read_offset]``; s2: ``A[i] = 99``.

    A non-zero offset makes it a CROSS-ITERATION anti-dependency (iteration i's write
    would clobber a later iteration's read); offset zero is element-wise, safe per
    iteration."""
    size = N + read_offset
    sdfg = SDFG(f'war_two_maps_off{read_offset}')
    for arr in ('A', 'B'):
        sdfg.add_array(arr, [size], dtypes.float64)
    s1 = sdfg.add_state('read_map', is_start_block=True)
    s2 = sdfg.add_state('write_map')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_access('A')
    bw = s1.add_access('B')
    me, mx = s1.add_map('m_read', {'i': f'0:{N}'})
    t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=Memlet(f'A[i + {read_offset}]'))
    s1.add_memlet_path(t, mx, bw, src_conn='_out', memlet=Memlet('B[i]'))

    aw = s2.add_access('A')
    me2, mx2 = s2.add_map('m_write', {'i': f'0:{N}'})
    t2 = s2.add_tasklet('set', {}, {'_out'}, '_out = 99.0')
    s2.add_memlet_path(me2, t2, memlet=Memlet())
    s2.add_memlet_path(t2, mx2, aw, src_conn='_out', memlet=Memlet('A[i]'))
    sdfg.validate()
    return sdfg


def _waw_two_maps(write_offset: int) -> SDFG:
    """s1: ``A[i] = 1``; s2: ``A[i + write_offset] = 2`` -- an output dependency."""
    size = N + write_offset
    sdfg = SDFG(f'waw_two_maps_off{write_offset}')
    sdfg.add_array('A', [size], dtypes.float64)
    s1 = sdfg.add_state('first_write', is_start_block=True)
    s2 = sdfg.add_state('second_write')
    sdfg.add_edge(s1, s2, InterstateEdge())
    for state, label, value, index in ((s1, 'm_first', '1.0', 'i'), (s2, 'm_second', '2.0', f'i + {write_offset}')):
        aw = state.add_access('A')
        me, mx = state.add_map(label, {'i': f'0:{N}'})
        t = state.add_tasklet('set', {}, {'_out'}, f'_out = {value}')
        state.add_memlet_path(me, t, memlet=Memlet())
        state.add_memlet_path(t, mx, aw, src_conn='_out', memlet=Memlet(f'A[{index}]'))
    sdfg.validate()
    return sdfg


def test_cross_iteration_war_between_maps_is_not_map_fused():
    """``B[i] = A[i+1]`` then ``A[i] = 99``. After state fusion the two Maps are ordered
    only by happens-before edges. Fusing them would make the ordering per-iteration, so
    iteration i's write to ``A[i]`` would clobber the value a later iteration still has to
    read -- a miscompile. MapFusion must leave them apart."""
    sdfg = _war_two_maps(read_offset=1)
    arrays = {'A': np.arange(N + 1, dtype=np.float64) + 1.0, 'B': np.zeros(N + 1, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    state_fused = _state_fuse(sdfg)
    assert state_fused.number_of_nodes() == 1, 'the two states should fuse'
    assert _count_empty_edges(state_fused) > 0, 'the WAR must be pinned by a happens-before edge'
    _assert_matches(ref, _run(state_fused, 'sf', **arrays), 'after state fusion')

    map_fused = _map_fuse(state_fused)
    assert _count_maps(map_fused) == 2, \
        'MapFusion must not fuse across a happens-before edge carrying a cross-iteration WAR'
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_cross_iteration_waw_between_maps_is_not_map_fused():
    """``A[i] = 1`` then ``A[i+1] = 2``. The two writes of iterations i and i+1 overlap, so
    per-iteration ordering does not cover the output dependency and MapFusion must refuse:
    fused, iteration i+1 could write ``A[i+1] = 2`` before iteration i writes ``A[i+1] = 1``,
    leaving a 1 where the sequential program leaves a 2."""
    sdfg = _waw_two_maps(write_offset=1)
    arrays = {'A': np.zeros(N + 1, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    state_fused = _state_fuse(sdfg)
    map_fused = _map_fuse(state_fused)
    assert _count_maps(map_fused) == 2, 'a cross-iteration WAW must not be reduced to a per-iteration one'
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_raw_between_maps_still_map_fuses_after_state_fusion():
    """``T[i] = A[i]*2`` then ``B[i] = T[i]+1``. RAW is a TRUE dependency and is handled by
    merging the common ``T`` node -- no ordering edge -- so the
    ``MapExit -> AccessNode -> MapEntry`` pattern survives and MapFusion still collapses
    the two Maps. Pins that state fusion does not cost us a legitimate Map fusion."""
    sdfg = SDFG('raw_two_maps')
    for arr in ('A', 'B'):
        sdfg.add_array(arr, [N], dtypes.float64)
    sdfg.add_transient('T', [N], dtypes.float64)
    s1 = sdfg.add_state('produce', is_start_block=True)
    s2 = sdfg.add_state('consume')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_access('A')
    tw = s1.add_access('T')
    me, mx = s1.add_map('m_prod', {'i': f'0:{N}'})
    t = s1.add_tasklet('x2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=Memlet('A[i]'))
    s1.add_memlet_path(t, mx, tw, src_conn='_out', memlet=Memlet('T[i]'))

    tr = s2.add_access('T')
    bw = s2.add_access('B')
    me2, mx2 = s2.add_map('m_cons', {'i': f'0:{N}'})
    t2 = s2.add_tasklet('p1', {'_in'}, {'_out'}, '_out = _in + 1.0')
    s2.add_memlet_path(tr, me2, t2, dst_conn='_in', memlet=Memlet('T[i]'))
    s2.add_memlet_path(t2, mx2, bw, src_conn='_out', memlet=Memlet('B[i]'))
    sdfg.validate()

    arrays = {'A': np.arange(N, dtype=np.float64) + 1.0, 'B': np.zeros(N, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    state_fused = _state_fuse(sdfg)
    assert state_fused.number_of_nodes() == 1
    assert _count_empty_edges(state_fused) == 0, 'RAW must flow through a merged node, not an ordering edge'
    _assert_matches(ref, _run(state_fused, 'sf', **arrays), 'after state fusion')

    map_fused = _map_fuse(state_fused)
    assert _count_maps(map_fused) == 1, 'the RAW Map pair must still fuse after state fusion'
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_elementwise_war_between_maps_fuses_with_inner_ordering():
    """``B[i] = A[i]`` then ``A[i] = 99``: the anti-dependency is ELEMENT-WISE, so per
    iteration the read of ``A[i]`` still precedes the overwrite of ``A[i]`` and the Maps may
    be fused. MapFusion drops the outer happens-before edges and re-establishes the
    dependency inside the fused scope."""
    sdfg = _war_two_maps(read_offset=0)
    arrays = {'A': np.arange(N, dtype=np.float64) + 1.0, 'B': np.zeros(N, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    state_fused = _state_fuse(sdfg)
    assert _count_empty_edges(state_fused) > 0, 'the WAR is pinned by an ordering edge before Map fusion'

    map_fused = _map_fuse(state_fused)
    assert _count_maps(map_fused) == 1, 'an element-wise WAR must not block Map fusion'
    _assert_ordered_inside_one_map(map_fused, '_out = _in', '_out = 99.0')
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_elementwise_waw_between_maps_fuses_with_inner_ordering():
    """``A[i] = 1`` then ``A[i] = 2``: the output dependency is element-wise, so the Maps
    fuse and the surviving write must still be the second one. The ordering edge inside the
    fused Map is what guarantees that -- without it the two writes are free siblings."""
    sdfg = _waw_two_maps(write_offset=0)
    arrays = {'A': np.zeros(N, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)
    assert np.allclose(ref['A'], 2.0), 'the reference must keep the second write'

    map_fused = _map_fuse(_state_fuse(sdfg))
    assert _count_maps(map_fused) == 1, 'an element-wise WAW must not block Map fusion'
    _assert_ordered_inside_one_map(map_fused, '_out = 1.0', '_out = 2.0')
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def _war_two_dimensional(read_whole_first_dim: bool) -> SDFG:
    """s1 reads ``A[i, j]`` (or all of ``A[:, j]``), s2 writes ``A[i, j]``.

    Reading the whole first dimension leaves the Map parameter ``i`` unpinned: iterations
    (0, j) and (1, j) then touch the same elements, so the hazard is not iteration-private."""
    sdfg = SDFG(f'war_2d_{"whole" if read_whole_first_dim else "point"}')
    for arr in ('A', 'B'):
        sdfg.add_array(arr, [N, N], dtypes.float64)
    s1 = sdfg.add_state('read_map', is_start_block=True)
    s2 = sdfg.add_state('write_map')
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar = s1.add_access('A')
    bw = s1.add_access('B')
    me, mx = s1.add_map('m_read', {'i': f'0:{N}', 'j': f'0:{N}'})
    if read_whole_first_dim:
        t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in[0]')
        read_memlet = Memlet(f'A[0:{N}, j]')
    else:
        t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
        read_memlet = Memlet('A[i, j]')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=read_memlet)
    s1.add_memlet_path(t, mx, bw, src_conn='_out', memlet=Memlet('B[i, j]'))

    aw = s2.add_access('A')
    me2, mx2 = s2.add_map('m_write', {'i': f'0:{N}', 'j': f'0:{N}'})
    t2 = s2.add_tasklet('set', {}, {'_out'}, '_out = 99.0')
    s2.add_memlet_path(me2, t2, memlet=Memlet())
    s2.add_memlet_path(t2, mx2, aw, src_conn='_out', memlet=Memlet('A[i, j]'))
    sdfg.validate()
    return sdfg


def test_two_dimensional_elementwise_war_fuses():
    """Both Map parameters are pinned by the ``A[i, j]`` accesses, so the hazard cannot
    cross iterations and the Maps fuse."""
    sdfg = _war_two_dimensional(read_whole_first_dim=False)
    arrays = {'A': np.arange(N * N, dtype=np.float64).reshape(N, N), 'B': np.zeros((N, N), dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    map_fused = _map_fuse(_state_fuse(sdfg))
    assert _count_maps(map_fused) == 1
    _assert_ordered_inside_one_map(map_fused, '_out = _in', '_out = 99.0')
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_war_with_unpinned_map_parameter_is_not_fused():
    """The read ``A[0:N, j]`` does not depend on ``i``, so iteration (0, j) reads what
    iteration (1, j) overwrites. Per-iteration ordering does not cover that and the Maps
    must stay apart -- the case an "are the subsets equal?" test would wrongly accept."""
    sdfg = _war_two_dimensional(read_whole_first_dim=True)
    arrays = {'A': np.arange(N * N, dtype=np.float64).reshape(N, N), 'B': np.zeros((N, N), dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    map_fused = _map_fuse(_state_fuse(sdfg))
    assert _count_maps(map_fused) == 2, 'an unpinned Map parameter leaves a cross-iteration hazard'
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_ordering_edge_of_a_third_map_is_not_dropped():
    """``B[i] = A[i]`` (0:N) then ``A[k] = 1`` (0:N/2) then ``A[i] = 2`` (0:N).

    The first and third Maps have the same range and an iteration-private hazard, so they
    look fusable -- but the second Map is ordered between them. Fusing the outer pair would
    have to drop an ordering edge that belongs to the middle Map, so MapFusion must refuse
    the pair rather than silently lose that dependency."""
    sdfg = SDFG('third_map_ordering')
    for arr in ('A', 'B'):
        sdfg.add_array(arr, [N], dtypes.float64)
    s1 = sdfg.add_state('read', is_start_block=True)
    s2 = sdfg.add_state('write_half')
    s3 = sdfg.add_state('write_all')
    sdfg.add_edge(s1, s2, InterstateEdge())
    sdfg.add_edge(s2, s3, InterstateEdge())

    ar = s1.add_access('A')
    bw = s1.add_access('B')
    me, mx = s1.add_map('m_read', {'i': f'0:{N}'})
    t = s1.add_tasklet('cp', {'_in'}, {'_out'}, '_out = _in')
    s1.add_memlet_path(ar, me, t, dst_conn='_in', memlet=Memlet('A[i]'))
    s1.add_memlet_path(t, mx, bw, src_conn='_out', memlet=Memlet('B[i]'))

    for state, label, params, index, value in ((s2, 'm_half', {
            'k': f'0:{N // 2}'
    }, 'k', '1.0'), (s3, 'm_all', {
            'i': f'0:{N}'
    }, 'i', '2.0')):
        aw = state.add_access('A')
        entry, exit_ = state.add_map(label, params)
        tw = state.add_tasklet('set', {}, {'_out'}, f'_out = {value}')
        state.add_memlet_path(entry, tw, memlet=Memlet())
        state.add_memlet_path(tw, exit_, aw, src_conn='_out', memlet=Memlet(f'A[{index}]'))
    sdfg.validate()

    arrays = {'A': np.arange(N, dtype=np.float64) + 1.0, 'B': np.zeros(N, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    map_fused = _map_fuse(_state_fuse(sdfg))
    assert _count_maps(map_fused) == 3, 'the three Maps must stay apart, the middle one orders the outer pair'
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


def test_war_from_dace_program_fuses_and_matches_reference():
    """The same element-wise WAR, but built by the frontend rather than by hand, so the
    Memlet/scope shapes are the ones the pipeline really sees."""

    @dace.program
    def war_program(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = A[i] * 2.0
        for i in dace.map[0:N]:
            A[i] = 99.0

    sdfg = war_program.to_sdfg(simplify=False)
    arrays = {'A': np.arange(N, dtype=np.float64) + 1.0, 'B': np.zeros(N, dtype=np.float64)}
    ref = _run(sdfg, 'ref', **arrays)

    state_fused = _state_fuse(sdfg)
    _assert_matches(ref, _run(state_fused, 'sf', **arrays), 'after state fusion')
    map_fused = _map_fuse(state_fused)
    _assert_matches(ref, _run(map_fused, 'mf', **arrays), 'after MapFusion')


if __name__ == '__main__':
    test_cross_iteration_war_between_maps_is_not_map_fused()
    test_cross_iteration_waw_between_maps_is_not_map_fused()
    test_raw_between_maps_still_map_fuses_after_state_fusion()
    test_elementwise_war_between_maps_fuses_with_inner_ordering()
    test_elementwise_waw_between_maps_fuses_with_inner_ordering()
    test_two_dimensional_elementwise_war_fuses()
    test_war_with_unpinned_map_parameter_is_not_fused()
    test_ordering_edge_of_a_third_map_is_not_dropped()
    test_war_from_dace_program_fuses_and_matches_reference()
