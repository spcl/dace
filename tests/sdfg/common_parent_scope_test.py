# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``dace.sdfg.scope.common_parent_scope`` on disjoint scopes."""

import dace
from dace.sdfg import scope
from dace.sdfg.scope import common_parent_scope


def _nested_maps() -> dace.SDFGState:
    """An outer map that contains two sibling maps, next to a second top-level map."""
    sdfg = dace.SDFG("common_parent_scope")
    sdfg.add_array("A", [10, 10], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", {"i": "0:10"})
    for name in ("first", "second"):
        inner_entry, inner_exit = state.add_map(name, {"j": "0:5"})
        tasklet = state.add_tasklet(name, {}, {"o"}, "o = 1")
        state.add_nedge(outer_entry, inner_entry, dace.Memlet())
        state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
        state.add_memlet_path(
            tasklet,
            inner_exit,
            outer_exit,
            state.add_write("A"),
            src_conn="o",
            memlet=dace.Memlet(f"A[i, j + {5 if name == 'second' else 0}]"),
        )
    state.add_mapped_tasklet("other", {"k": "0:10"}, {}, "o = 2", {"o": dace.Memlet("A[k, 0]")}, external_edges=True)
    return state


def test_sibling_scopes_share_their_parent():
    state = _nested_maps()
    sdict = state.scope_dict()
    outer = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label == "outer")
    first, second = (
        next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == name)
        for name in ("first", "second")
    )
    assert scope.common_parent_scope(sdict, sdict[first], sdict[second]) is outer


def test_containing_scope_is_the_common_parent():
    state = _nested_maps()
    sdict = state.scope_dict()
    outer = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label == "outer")
    first = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label == "first")
    assert scope.common_parent_scope(sdict, first, outer) is outer
    assert scope.common_parent_scope(sdict, outer, first) is outer


def test_disjoint_top_level_scopes_have_no_common_parent():
    state = _nested_maps()
    sdict = state.scope_dict()
    first = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == "first")
    other = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == "other")
    assert scope.common_parent_scope(sdict, sdict[first], sdict[other]) is None


def nested_pair(state, label, a, b, rng, outer=None):
    """A map over ``j`` on ``rng`` holding a map over ``k``, reading ``A`` into ``B``, optionally inside ``outer``."""
    entry, exit_node = state.add_map(label, dict(j=rng))
    inner_entry, inner_exit = state.add_map(label + "_inner", dict(k="0:2"))
    tasklet = state.add_tasklet(label, {"x"}, {"o"}, "o = x + 1")
    index = "i, j + k" if outer else "j, k"
    entries = [outer[0], entry, inner_entry] if outer else [entry, inner_entry]
    exits = [inner_exit, exit_node, outer[1]] if outer else [inner_exit, exit_node]
    state.add_memlet_path(a, *entries, tasklet, dst_conn="x", memlet=dace.Memlet(f"A[{index}]"))
    state.add_memlet_path(tasklet, *exits, b, src_conn="o", memlet=dace.Memlet(f"B[{index}]"))
    return entry, inner_entry


def test_two_scopes_in_sibling_branches_share_the_scope_around_them():
    """The path to the top level never advanced past the scope's own parent, and the paths were
    walked with ``reversed(zip(...))``, which Python refuses: two maps nested in sibling maps of one
    kernel (the fp64 x 2 vectorized CloudSC's tile main and remainder slabs) ran the allocation
    lifetime analysis out of 500 GB of memory."""
    sdfg = dace.SDFG("sibling_branches")
    sdfg.add_array("A", [4, 8], dace.float64)
    sdfg.add_array("B", [4, 8], dace.float64)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    outer = state.add_map("outer", dict(i="0:4"))
    _, first = nested_pair(state, "first", a, b, "0:2", outer)
    _, second = nested_pair(state, "second", a, b, "4:6", outer)
    sdfg.validate()
    assert common_parent_scope(state.scope_dict(), first, second) is outer[0]


def test_two_scopes_in_separate_top_level_maps_share_no_scope():
    sdfg = dace.SDFG("separate_branches")
    sdfg.add_array("A", [8, 2], dace.float64)
    sdfg.add_array("B", [8, 2], dace.float64)
    state = sdfg.add_state()
    _, first = nested_pair(state, "first", state.add_read("A"), state.add_write("B"), "0:4")
    _, second = nested_pair(state, "second", state.add_read("A"), state.add_write("B"), "4:8")
    sdfg.validate()
    assert common_parent_scope(state.scope_dict(), first, second) is None


if __name__ == "__main__":
    test_sibling_scopes_share_their_parent()
    test_containing_scope_is_the_common_parent()
    test_disjoint_top_level_scopes_have_no_common_parent()
