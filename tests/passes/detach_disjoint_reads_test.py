# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map reading what a sibling map's write misses is detached from it; one that needs the order is not."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.detach_disjoint_reads import DetachDisjointReads, detach

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


def chain(name: str, *extra: str) -> tuple[dace.SDFG, dace.SDFGState, list[nodes.AccessNode]]:
    """Three access nodes of ``t``, and one of each ``extra`` vector."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("t", [N, 8], dace.float64)
    for vector in extra:
        sdfg.add_array(vector, [N], dace.float64)
    state = sdfg.add_state()
    return sdfg, state, [state.add_access("t") for _ in range(3)]


def add_map(state: dace.SDFGState, label: str, reads: dict[str, nodes.AccessNode], write: str, sink: nodes.AccessNode):
    state.add_mapped_tasklet(
        label,
        {"i": f"0:{N}"},
        {f"a{k}": dace.Memlet(memlet) for k, memlet in enumerate(reads)},
        "b = " + " + ".join(f"a{k}" for k in range(len(reads))),
        {"b": dace.Memlet(write)},
        input_nodes={memlet.split("[")[0]: source for memlet, source in reads.items()},
        output_nodes={sink.data: sink},
        external_edges=True,
    )


def test_a_map_also_ordered_through_a_sibling_cannot_join_its_write_to_the_chain():
    sdfg, state, (first, between, last) = chain("detach_cycle", "u")
    vector = state.add_access("u")
    add_map(state, "writer", {"t[i, 0]": first}, "t[i, 6]", between)
    add_map(state, "sibling", {"t[i, 6]": between}, "u[i]", vector)
    add_map(state, "reader", {"t[i, 1]": between, "u[i]": vector}, "t[i, 7]", last)

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None

    sdfg.validate()


def test_a_read_ahead_of_the_writer_stays_ahead_of_the_reader_write():
    """``load`` reads ``t[i, 7]`` before ``reader`` overwrites it, only because it feeds ``other``."""
    sdfg, state, (first, between, last) = chain("detach_load", "u")
    vector = state.add_access("u")
    add_map(state, "load", {"t[i, 7]": first}, "u[i]", vector)
    add_map(state, "other", {"u[i]": vector}, "t[i, 6]", between)
    add_map(state, "writer", {"t[i, 0]": first}, "t[i, 5]", between)
    add_map(state, "reader", {"t[i, 1]": between}, "t[i, 7]", last)

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None


def test_a_copy_is_a_write_of_its_destination():
    sdfg, state, (first, between, last) = chain("detach_copy_write", "s", "y")
    copied, again = state.add_access("s"), state.add_access("s")
    state.add_nedge(state.add_access("y"), copied, dace.Memlet(f"y[0:{N}]"))
    add_map(state, "writer", {"t[i, 0]": first, "s[i]": copied}, "t[i, 6]", between)
    add_map(state, "reader", {"t[i, 1]": between, "s[i]": again}, "t[i, 7]", last)

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None


def test_a_copy_memlet_naming_its_destination_still_reads_the_source():
    sdfg, state, (first, between, last) = chain("detach_copy_read")
    sdfg.add_array("y", [N, 8], dace.float64)
    add_map(state, "writer", {"t[i, 0]": first}, "t[i, 6]", between)
    state.add_edge(between, None, state.add_access("y"), None, dace.Memlet(f"y[0:{N}, 7]"))
    add_map(state, "reader", {"t[i, 1]": between}, "t[i, 7]", last)

    assert DetachDisjointReads().apply_pass(sdfg, {}) is None


def test_a_write_a_detach_moved_in_need_not_come_from_a_map():
    sdfg, state, (first, between, last) = chain("detach_moved_write")
    add_map(state, "writer", {"t[i, 0]": first}, "t[i, 6]", between)
    add_map(state, "reader", {"t[i, 1]": between}, "t[i, 7]", last)
    add_map(state, "second_reader", {"t[i, 2]": between}, "t[i, 3]", state.add_access("t"))
    state.add_edge(state.add_tasklet("one", {}, {"x"}, "x = 1.0"), "x", last, None, dace.Memlet("t[0, 5]"))

    assert DetachDisjointReads().apply_pass(sdfg, {}) == 1

    sdfg.validate()


def test_a_writer_outside_any_map_keeps_the_reader():
    sdfg = dace.SDFG("detach_top_level_writer")
    sdfg.add_array("a", [4], dace.float64)
    sdfg.add_array("b", [4], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    written = state.add_access("a")
    state.add_edge(state.add_tasklet("w", {}, {"o"}, "o = 1.0"), "o", written, None, dace.Memlet("a[0]"))
    entry, exit_ = state.add_map("reader", {"i": "1:3"})
    tasklet = state.add_tasklet("t", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(written, entry, tasklet, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(tasklet, exit_, state.add_write("b"), src_conn="y", memlet=dace.Memlet("b[i]"))
    assert not detach(state, written, entry, sdfg.facts())


if __name__ == "__main__":
    test_a_read_a_write_misses_detaches_and_computes_the_same()
    test_a_read_of_the_write_keeps_the_order()
    test_a_map_ordered_only_by_the_chain_on_another_container_keeps_the_order()
    test_a_map_also_ordered_through_a_sibling_cannot_join_its_write_to_the_chain()
    test_a_read_ahead_of_the_writer_stays_ahead_of_the_reader_write()
    test_a_copy_is_a_write_of_its_destination()
    test_a_copy_memlet_naming_its_destination_still_reads_the_source()
    test_a_write_a_detach_moved_in_need_not_come_from_a_map()
    test_a_writer_outside_any_map_keeps_the_reader()
