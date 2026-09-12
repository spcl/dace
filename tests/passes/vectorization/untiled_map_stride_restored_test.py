# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map the tile emitters skipped must not keep the tile stride.

``StrideMapByTileWidths`` sets ``step = W`` from the shared candidate gate; the emitters that turn
the body into W lanes select through a predicate of their own (the scope must be a single body
``NestedSDFG``). Where the two disagree the map runs one element per W iterations and W-1 of every
W are never computed -- the CloudSC vectorize leg lost three whole map regions that way.
:class:`RestoreUntiledMapStride` reads the finished graph instead of re-deriving a predicate: no
tile library node in the scope means the stride has to go back.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops.nodes import TileLoad
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.restore_untiled_map_stride import (TILE_NODES, RestoreUntiledMapStride)
from dace.transformation.passes.vectorization.utils.errors import VectorizeUnsupported

WIDTH = 8
N = dace.symbol('N', dtype=dace.int64)


def strided_map_sdfg(name: str, step: int) -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """``A -> map(0:N:step) -> tasklet -> B``, the shape a stride pass leaves behind."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, exit_node = state.add_map('body', {'i': f'0:{N}:{step}'})
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(state.add_access('A'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_access('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg, state, entry


def innermost_steps(sdfg: dace.SDFG) -> list[str]:
    """The last-dim step of every map in ``sdfg``, as printed."""
    return [
        str(node.map.range.ranges[-1][2]) for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry)
    ]


def tile_node_count(sdfg: dace.SDFG) -> int:
    """How many tile library nodes the emit stage left in ``sdfg``."""
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, TILE_NODES))


def test_a_strided_map_the_emitters_left_untiled_gets_its_unit_step_back():
    sdfg, state, entry = strided_map_sdfg('untiled_strided', WIDTH)

    repaired = RestoreUntiledMapStride(widths=(WIDTH, )).apply_pass(sdfg, {})

    assert repaired == 1
    assert str(entry.map.range.ranges[-1][2]) == '1'
    sdfg.validate()


def test_a_map_carrying_a_tile_op_keeps_its_tile_stride():
    """The control the previous test needs: with a tile op present the pass must do nothing."""
    sdfg, state, entry = strided_map_sdfg('tiled_strided', WIDTH)
    sdfg.add_array('A_tile', [WIDTH], dace.float64, transient=True, storage=dace.StorageType.Register)
    tile_load = TileLoad(name='load', widths=(WIDTH, ), src_kind='Tile')
    state.add_node(tile_load)
    state.add_edge(entry, 'OUT_1', tile_load, '_src', dace.Memlet('A[i:i+8]'))
    entry.add_out_connector('OUT_1')
    entry.add_in_connector('IN_1')
    state.add_edge(state.add_access('A'), None, entry, 'IN_1', dace.Memlet('A[0:N]'))
    state.add_edge(tile_load, '_dst', state.add_access('A_tile'), None, dace.Memlet('A_tile[0:8]'))

    repaired = RestoreUntiledMapStride(widths=(WIDTH, )).apply_pass(sdfg, {})

    assert repaired is None
    assert str(entry.map.range.ranges[-1][2]) == str(WIDTH)


def test_an_untiled_map_whose_body_was_already_widened_is_refused():
    """Widened per-lane buffers under a step-1 map would be a second miscompile, so refuse."""
    sdfg, state, entry = strided_map_sdfg('widened_untiled', WIDTH)
    inner = dace.SDFG('widened_body')
    inner.add_array('lane', [WIDTH], dace.float64, transient=True, storage=dace.StorageType.Register)
    inner.add_state('empty', is_start_block=True)
    nested = state.add_nested_sdfg(inner, {}, {})
    state.add_edge(entry, None, nested, None, dace.Memlet())

    with pytest.raises(VectorizeUnsupported, match='never lowered to tile ops'):
        RestoreUntiledMapStride(widths=(WIDTH, )).apply_pass(sdfg, {})


def test_the_vectorizer_leaves_no_strided_map_without_a_tile_op():
    """The whole-pipeline invariant the repair exists to hold, on a kernel that does tile."""

    @dace.program
    def scale(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0 + 1.0

    sdfg = scale.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(WIDTH, ), target_isa='SCALAR', validate=True)).apply_pass(sdfg, {})

    assert tile_node_count(sdfg) > 0, 'nothing was tiled -- the invariant below would hold vacuously'
    for node, graph in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.MapEntry) or not isinstance(graph, dace.SDFGState):
            continue
        if str(node.map.range.ranges[-1][2]) != str(WIDTH):
            continue
        scope = graph.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
        tiled = any(isinstance(n, TILE_NODES) for n in scope) or any(
            isinstance(descendant, TILE_NODES) for n in scope if isinstance(n, dace.nodes.NestedSDFG)
            for descendant, _ in n.sdfg.all_nodes_recursive())
        assert tiled, (f'map {node.map.label!r} steps by {WIDTH} with no tile op in its body: '
                       f'{WIDTH - 1} of every {WIDTH} iterations are never computed')


def test_a_strided_untiled_map_computes_every_element_after_the_repair():
    """The numbers, not only the shape: before the repair the map skips 7 of every 8 elements."""
    sdfg, state, entry = strided_map_sdfg('numeric_untiled', WIDTH)
    RestoreUntiledMapStride(widths=(WIDTH, )).apply_pass(sdfg, {})

    a = np.arange(1.0, 21.0)
    b = np.full(20, -7.0)
    sdfg(A=a, B=b, N=20)

    assert np.array_equal(b, a * 2.0)
    assert str(entry.map.range.ranges[-1][2]) == '1'
