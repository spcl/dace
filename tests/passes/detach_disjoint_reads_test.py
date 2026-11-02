# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map reading what a sibling map's write misses is detached from it; one that needs the order is not."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.detach_disjoint_reads import DetachDisjointReads

N = 6


def slices(read: int) -> dace.SDFG:
    """``t[:, 2] = t[:, 0] + 1`` then ``t[:, 3] = t[:, read] + 1``, chained through the access nodes of ``t``."""
    sdfg = dace.SDFG(f"detach_slices_{read}")
    sdfg.add_array("t", [N, 4], dace.float64)
    state = sdfg.add_state()
    first, between, last = (state.add_access("t") for _ in range(3))
    for target, column, source, sink in ((2, 0, first, between), (3, read, between, last)):
        state.add_mapped_tasklet(
            f"slice_{target}",
            {"i": f"0:{N}"},
            {"a": dace.Memlet(f"t[i, {column}]")},
            "b = a + 1",
            {"b": dace.Memlet(f"t[i, {target}]")},
            input_nodes={"t": source},
            output_nodes={"t": sink},
            external_edges=True,
        )
    return sdfg


def maps_are_ordered(sdfg: dace.SDFG) -> bool:
    state = sdfg.start_block
    first, second = sorted((n for n in state.nodes() if isinstance(n, nodes.MapEntry)), key=lambda m: m.map.label)
    return second in state.bfs_nodes(first) or first in state.bfs_nodes(second)


def test_a_read_a_write_misses_detaches_and_computes_the_same():
    sdfg = slices(read=1)

    assert DetachDisjointReads().apply_pass(sdfg, {}) == 1

    sdfg.validate()
    assert not maps_are_ordered(sdfg)
    start = np.arange(4.0 * N).reshape(N, 4)
    reference = start.copy()
    reference[:, 2] = reference[:, 0] + 1
    reference[:, 3] = reference[:, 1] + 1
    sdfg(t=start)
    assert np.array_equal(start, reference)


def test_a_read_of_the_write_keeps_the_order():
    sdfg = slices(read=2)

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None

    assert maps_are_ordered(sdfg)


def test_a_map_ordered_only_by_the_chain_on_another_container_keeps_the_order():
    """The first map reads ``u`` before the second overwrites it: only the chain on ``t`` says so."""
    sdfg = slices(read=1)
    sdfg.add_array("u", [N], dace.float64)
    state = sdfg.start_block
    first, second = sorted((n for n in state.nodes() if isinstance(n, nodes.MapEntry)), key=lambda m: m.map.label)
    load = next(n for n in state.scope_children()[first] if isinstance(n, nodes.Tasklet))
    store = next(n for n in state.scope_children()[second] if isinstance(n, nodes.Tasklet))
    load.add_in_connector("c")
    load.code.as_string = "b = a + c"
    state.add_memlet_path(state.add_access("u"), first, load, dst_conn="c", memlet=dace.Memlet("u[i]"))
    store.add_out_connector("c")
    store.code.as_string = "b = a + 1\nc = a"
    state.add_memlet_path(
        store, state.exit_node(second), state.add_access("u"), src_conn="c", memlet=dace.Memlet("u[i]")
    )
    sdfg.validate()

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_a_read_a_write_misses_detaches_and_computes_the_same()
    test_a_read_of_the_write_keeps_the_order()
    test_a_map_ordered_only_by_the_chain_on_another_container_keeps_the_order()
