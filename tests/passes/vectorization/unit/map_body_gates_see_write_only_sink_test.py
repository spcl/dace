# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map body's safety gates must inspect a body that ends in a write-only scratch scalar.

CloudSC's ``zanew_0`` is an ordinary shape: a one-element ``Register`` transient with ``Scope``
lifetime, one in-edge and no out-edge -- a value computed and never read again. ``all_nodes_between``
discards its ENTIRE walk on reaching such a node, so every gate built on that walk reported a clean
body having looked at nothing, and the tile path strode a map by the vector width over a body that
was never widened. Each gate below is therefore checked BOTH ways over a body carrying that sink:
it must still refuse what it exists to refuse, and it must still admit what is genuinely fine. A
gate that can only answer "clean" proves nothing.
"""
from collections.abc import Callable

import dace
from dace import SDFGState
from dace.libraries.standard.nodes import Reduce
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.utils.map_predicates import (
    is_vectorizable_map, map_body_has_foreign_language_tasklet, map_body_has_inner_loop, map_body_has_library_node,
    map_body_has_mixed_conditional_tasklet, map_body_nodes, map_body_per_lane_subsets)

N = 16
SINK = 'zanew_0'

BodyBuilder = Callable[[dace.SDFG, SDFGState, dace.nodes.MapEntry, dace.nodes.MapExit], None]


def elementwise_body(sdfg: dace.SDFG, state: SDFGState, me: dace.nodes.MapEntry, mx: dace.nodes.MapExit) -> None:
    t = state.add_tasklet('scale', {'a'}, {'o'}, 'o = a * 2.0')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))


def reduce_library_node_body(sdfg: dace.SDFG, state: SDFGState, me: dace.nodes.MapEntry,
                             mx: dace.nodes.MapExit) -> None:
    red = Reduce('sum', wcr='lambda a, b: a + b', identity=0.0)
    state.add_node(red)
    state.add_memlet_path(state.add_access('A'), me, red, dst_conn='_in', memlet=dace.Memlet(f'A[i, 0:{N}]'))
    state.add_memlet_path(red, mx, state.add_access('B'), src_conn='_out', memlet=dace.Memlet('B[i]'))


def cpp_tasklet_body(sdfg: dace.SDFG, state: SDFGState, me: dace.nodes.MapEntry, mx: dace.nodes.MapExit) -> None:
    t = state.add_tasklet('native', {'a'}, {'o'}, 'o = a * 2.0;', language=dace.dtypes.Language.CPP)
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))


def mixed_conditional_tasklet_body(sdfg: dace.SDFG, state: SDFGState, me: dace.nodes.MapEntry,
                                   mx: dace.nodes.MapExit) -> None:
    t = state.add_tasklet('clamp', {'a'}, {'o'}, 'o = a * 2.0\nif o <= 0.1:\n    o = 1.0')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))


def body_sdfg_shell() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_array('p', [N, N], dace.float64)
    inner.add_array('q', [N], dace.float64)
    return inner


def fill_body_write(body: SDFGState, index: str) -> None:
    t = body.add_tasklet('acc', {'cur'}, {'out'}, 'out = cur * 2.0')
    body.add_edge(body.add_access('p'), None, t, 'cur', dace.Memlet(f'p[i, {index}]'))
    body.add_edge(t, 'out', body.add_access('q'), None, dace.Memlet('q[i]'))


def flat_body_sdfg() -> dace.SDFG:
    """A body nest that is pure dataflow -- the shape the tile emitters can widen."""
    inner = body_sdfg_shell()
    fill_body_write(inner.add_state('compute', is_start_block=True), '0')
    return inner


def sweeping_body_sdfg() -> dace.SDFG:
    """A body nest whose write sits under a sequential ``LoopRegion`` -- a carried recurrence."""
    inner = body_sdfg_shell()
    loop = LoopRegion('sweep', f'j < {N}', 'j', 'j = 1', 'j = j + 1', sdfg=inner)
    inner.add_node(loop, is_start_block=True)
    fill_body_write(loop.add_state('compute', is_start_block=True), 'j')
    return inner


def nested_sdfg_body(make_inner: Callable[[], dace.SDFG]) -> BodyBuilder:

    def build(sdfg: dace.SDFG, state: SDFGState, me: dace.nodes.MapEntry, mx: dace.nodes.MapExit) -> None:
        ns = state.add_nested_sdfg(make_inner(), {'p': None}, {'q': None}, symbol_mapping={'i': 'i'})
        state.add_memlet_path(state.add_access('A'), me, ns, dst_conn='p', memlet=dace.Memlet(f'A[0:{N}, 0:{N}]'))
        state.add_memlet_path(ns, mx, state.add_access('B'), src_conn='q', memlet=dace.Memlet(f'B[0:{N}]'))

    return build


def map_with_scratch_sink(body: BodyBuilder) -> tuple[dace.SDFG, SDFGState, dace.nodes.MapEntry]:
    """``body`` inside a row map, alongside a ``zanew_0``-shaped write-only scratch scalar."""
    sdfg = dace.SDFG('gate_probe')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array(SINK, [1],
                   dace.float64,
                   storage=dace.dtypes.StorageType.Register,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.Scope)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    body(sdfg, state, me, mx)
    scratch = state.add_tasklet('stage', {}, {'o'}, 'o = 0.0')
    state.add_nedge(me, scratch, dace.Memlet())
    state.add_edge(scratch, 'o', state.add_access(SINK), None, dace.Memlet(f'{SINK}[0]'))
    sdfg.validate()
    return sdfg, state, me


def sink_node(state: SDFGState) -> dace.nodes.AccessNode:
    return next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == SINK)


def test_scratch_scalar_fixture_has_the_write_only_shape_the_gates_tripped_on() -> None:
    sdfg, state, map_entry = map_with_scratch_sink(elementwise_body)
    sink = sink_node(state)
    desc = sdfg.arrays[SINK]

    assert (len(state.in_edges(sink)), len(state.out_edges(sink))) == (1, 0)
    assert tuple(str(s) for s in desc.shape) == ('1', )
    assert desc.transient is True
    assert desc.storage is dace.dtypes.StorageType.Register
    assert desc.lifetime is dace.dtypes.AllocationLifetime.Scope
    assert sink in map_body_nodes(state, map_entry)


def test_map_wrapping_a_reduce_is_refused_although_the_body_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(reduce_library_node_body)

    assert map_body_has_library_node(state, map_entry) is True
    assert any(isinstance(n, Reduce) for n in map_body_nodes(state, map_entry))


def test_elementwise_map_ending_in_a_scratch_scalar_holds_no_library_node() -> None:
    _, state, map_entry = map_with_scratch_sink(elementwise_body)

    assert map_body_has_library_node(state, map_entry) is False
    assert not any(isinstance(n, dace.nodes.LibraryNode) for n in map_body_nodes(state, map_entry))


def test_map_wrapping_a_cpp_tasklet_is_refused_although_the_body_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(cpp_tasklet_body)

    assert map_body_has_foreign_language_tasklet(state, map_entry) is True
    assert any(
        isinstance(n, dace.nodes.Tasklet) and n.language is dace.dtypes.Language.CPP
        for n in map_body_nodes(state, map_entry))


def test_elementwise_map_ending_in_a_scratch_scalar_holds_no_foreign_language_tasklet() -> None:
    _, state, map_entry = map_with_scratch_sink(elementwise_body)

    assert map_body_has_foreign_language_tasklet(state, map_entry) is False
    assert all(n.language is dace.dtypes.Language.Python for n in map_body_nodes(state, map_entry)
               if isinstance(n, dace.nodes.Tasklet))


def test_map_sweeping_a_sequential_loop_is_refused_although_the_body_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(nested_sdfg_body(sweeping_body_sdfg))
    nest = next(n for n in map_body_nodes(state, map_entry) if isinstance(n, dace.nodes.NestedSDFG))

    assert map_body_has_inner_loop(state, map_entry) is True
    assert any(isinstance(r, LoopRegion) for r in nest.sdfg.all_control_flow_regions(recursive=True))


def test_pure_dataflow_nest_ending_in_a_scratch_scalar_holds_no_inner_loop() -> None:
    _, state, map_entry = map_with_scratch_sink(nested_sdfg_body(flat_body_sdfg))
    nest = next(n for n in map_body_nodes(state, map_entry) if isinstance(n, dace.nodes.NestedSDFG))

    assert map_body_has_inner_loop(state, map_entry) is False
    assert not any(isinstance(r, LoopRegion) for r in nest.sdfg.all_control_flow_regions(recursive=True))


def test_tasklet_mixing_a_statement_with_a_branch_is_refused_although_the_body_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(mixed_conditional_tasklet_body)

    assert map_body_has_mixed_conditional_tasklet(state, map_entry) is True


def test_single_statement_tasklet_ending_in_a_scratch_scalar_is_not_a_mixed_conditional() -> None:
    _, state, map_entry = map_with_scratch_sink(elementwise_body)

    assert map_body_has_mixed_conditional_tasklet(state, map_entry) is False


def test_per_lane_write_inside_a_body_nest_is_enumerated_although_the_body_ends_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(nested_sdfg_body(flat_body_sdfg))
    writes = list(map_body_per_lane_subsets(state, map_entry))

    assert [str(w.subset) for w in writes] == [f'0:{N}', 'i']
    assert [w.desc.transient for w in writes] == [False, False]
    assert writes[1].sdfg is not state.sdfg


def test_a_body_without_a_nest_yields_only_the_map_exit_boundary_write() -> None:
    _, state, map_entry = map_with_scratch_sink(elementwise_body)
    writes = list(map_body_per_lane_subsets(state, map_entry))

    assert [str(w.subset) for w in writes] == ['i']
    assert [w.sdfg for w in writes] == [state.sdfg]


def test_the_shared_tile_gate_refuses_a_library_node_body_ending_in_a_scratch_scalar() -> None:
    _, state, map_entry = map_with_scratch_sink(reduce_library_node_body)

    assert is_vectorizable_map(state, map_entry, K=1) is False
