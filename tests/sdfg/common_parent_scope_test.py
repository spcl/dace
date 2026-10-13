# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``dace.sdfg.scope.common_parent_scope`` on disjoint scopes."""

import dace
from dace.sdfg import scope


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


if __name__ == "__main__":
    test_sibling_scopes_share_their_parent()
    test_containing_scope_is_the_common_parent()
    test_disjoint_top_level_scopes_have_no_common_parent()
