# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List
import dace
from dace.transformation.dataflow import MapTiling

N = dace.symbol('N')


def test_map_tiling_with_strides():
    s = 33

    @dace.program
    def vector_copy_strides(A: dace.uint32[N], B: dace.uint32[N]):
        for i in dace.map[0:N:s] @ dace.dtypes.ScheduleType.CPU_Multicore:
            A[i] = B[i]

    sdfg = vector_copy_strides.to_sdfg()
    sdfg.simplify()
    assert len(list(sdfg.all_states())) == 1

    state = next(iter(sdfg.all_states()))
    state_nodes = state.nodes()
    map_entries: List[dace.nodes.MapEntry] = [n for n in state_nodes if isinstance(n, dace.nodes.MapEntry)]
    assert len(map_entries) == 1
    map_entry = map_entries[0]

    tile_sizes = [32]
    MapTiling.apply_to(sdfg=sdfg,
                       options={
                           "prefix": "b",
                           "tile_sizes": tile_sizes,
                           "divides_evenly": False,
                           "tile_trivial": True,
                           "skew": False
                       },
                       map_entry=map_entry)
    inner_map_entry = map_entry
    outer_map_entry = state.entry_node(inner_map_entry)

    b_i = dace.symbol("b_i")
    inner_rangelist = [(b_i, dace.symbolic.SymExpr("Min(N - 1, b_i + 32*33 - 1)"), 33)]
    outer_rangelist = [(0, N - 1, 32 * 33)]
    inner_range = dace.subsets.Range(inner_rangelist)
    outer_range = dace.subsets.Range(outer_rangelist)

    sdfg.validate()

    assert inner_map_entry.map.range == inner_range
    assert outer_map_entry.map.range == outer_range


def _get_sdfg_with_memlet_tree():
    sdfg = dace.SDFG("test")
    state = sdfg.add_state(is_start_block=True)

    for aname in "ab":
        sdfg.add_array(
            aname,
            shape=(10, 2),
            dtype=dace.float64,
            storage=dace.dtypes.StorageType.GPU_Global,
            transient=False,
        )
    sdfg.add_scalar(
        "s",
        dtype=dace.float64,
        transient=True,
    )

    a, b, s = (state.add_access(name) for name in "abs")
    me, mx = state.add_map("comp", ndrange={"__i": "0:10"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    tlet = state.add_tasklet(
        "tlet",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + 1.0",
    )

    state.add_edge(
        a,
        None,
        me,
        "IN_a1",
        dace.Memlet("a[0:10, 0]"),
    )
    state.add_edge(
        me,
        "OUT_a1",
        tlet,
        "__in",
        dace.Memlet("a[__i, 0]"),
    )
    me.add_scope_connectors("a1")

    state.add_edge(
        tlet,
        "__out",
        mx,
        "IN_b1",
        dace.Memlet("b[__i, 0]"),
    )
    state.add_edge(
        mx,
        "OUT_b1",
        b,
        None,
        dace.Memlet("b[0:10, 0]"),
    )
    mx.add_scope_connectors("b1")

    state.add_edge(
        me,
        # It is also important that we read from the same as the tasklet.
        "OUT_a1",
        s,
        None,
        # According to my understanding the error is here, that the data of this
        #  Memlet refers to `s` instead of `a` as the outer data does.
        dace.Memlet("s[0] -> [__i, 0]"),
    )

    state.add_edge(
        s,
        None,
        mx,
        "IN_b2",
        dace.Memlet("b[__i, 1] -> [0]"),
    )
    state.add_edge(
        mx,
        "OUT_b2",
        b,
        None,
        dace.Memlet("b[0:10, 1]"),
    )
    mx.add_scope_connectors("b2")

    sdfg.validate()
    return sdfg


def test_memlet_tree():
    sdfg = _get_sdfg_with_memlet_tree()
    all_maps_and_states = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    for map_entry, state in all_maps_and_states:
        MapTiling.apply_to(state.sdfg, {}, 0, True, map_entry=map_entry)


def test_symbol_rename_reaches_strip_mined_bound():
    """Renaming the tile iterator must rewrite the whole inner range, not just its start.

    ``MapTiling`` gives the inner map the bound ``Min(N - 1, tile_i + 31)``, which DaCe stores as
    a ``SymExpr`` (exact expression plus over-approximation). Symbol replacement used to hand any
    non-``sympy.Basic`` value straight back, so the range came out half-renamed --
    ``q:Min(N - 1, tile_i + 31)`` -- naming a symbol nothing binds any more.
    """

    @dace.program
    def scale(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(MapTiling, options=dict(tile_sizes=(32, ))) == 1
    state = sdfg.states()[0]
    outer = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is None)
    inner = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is outer)
    tile_param = outer.map.params[0]
    assert tile_param in str(inner.map.range), 'the inner bound must depend on the tile iterator'

    sdfg.replace(tile_param, 'q')

    assert tile_param not in str(inner.map.range), f'stale tile iterator left in {inner.map.range}'
    stale = [str(e.data) for e in state.edges() if e.data is not None and tile_param in str(e.data.subset)]
    assert not stale, f'stale tile iterator left in memlets: {stale}'


if __name__ == '__main__':
    test_map_tiling_with_strides()
    test_memlet_tree()
    test_symbol_rename_reaches_strip_mined_bound()
