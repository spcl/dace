# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``match_patterns`` skips a state that holds no node of a pattern's types before collapsing it: the
matches are the ones a full scan finds, and a state with nothing to match is never collapsed."""
from unittest import mock

import dace
from dace.transformation.dataflow import MapToForLoop, TrivialMapElimination
from dace.transformation.passes import pattern_matching


def make_sdfg() -> dace.SDFG:
    """Four states: two tasklet-only states, then two states with one map each."""
    sdfg = dace.SDFG("pattern_type_prefilter")
    sdfg.add_array("a", shape=(10, ), dtype=dace.float64)
    sdfg.add_scalar("s", dtype=dace.float64, transient=True)
    previous = None
    for index in range(4):
        state = sdfg.add_state(f"s{index}", is_start_block=(previous is None))
        if previous is not None:
            sdfg.add_edge(previous, state, dace.InterstateEdge())
        previous = state
        if index < 2:
            tasklet = state.add_tasklet(f"t{index}", {}, {"__out"}, "__out = 1.0")
            state.add_edge(tasklet, "__out", state.add_write("s"), None, dace.Memlet("s[0]"))
        else:
            state.add_mapped_tasklet(f"m{index}",
                                     map_ranges={"__i": "0:10"},
                                     inputs={},
                                     outputs={"__out": dace.Memlet("a[__i]")},
                                     code="__out = 2.0",
                                     external_edges=True)
    sdfg.validate()
    return sdfg


def matched(sdfg: dace.SDFG, node_match) -> list:
    """``(state label, matched map label)`` of every MapToForLoop match."""
    return [(sdfg.cfg_list[m.cfg_id].node(m.state_id).label, m.map_entry.map.label)
            for m in pattern_matching.match_patterns(sdfg, [MapToForLoop], node_match=node_match)]


def test_prefilter_finds_the_matches_of_a_full_scan():
    sdfg = make_sdfg()
    # Any node_match other than type_match turns the prefilter off: that is the full scan.
    full_scan = matched(sdfg, lambda a, b: pattern_matching.type_match(a, b))
    assert full_scan == [("s2", "m2_map"), ("s3", "m3_map")]
    assert matched(sdfg, pattern_matching.type_match) == full_scan


def test_state_without_pattern_types_is_not_collapsed():
    sdfg = make_sdfg()
    original = pattern_matching.collapse_multigraph_to_nx
    with mock.patch.object(pattern_matching, "collapse_multigraph_to_nx", side_effect=original) as spy:
        assert len(matched(sdfg, pattern_matching.type_match)) == 2
    # The pattern graphs are collapsed too (building the metadata); only the states count here.
    collapsed = [call.args[0] for call in spy.call_args_list if isinstance(call.args[0], dace.SDFGState)]
    assert [state.label for state in collapsed] == ["s2", "s3"]


def test_pattern_types_present_needs_every_pattern_node_type():
    (_, _, map_pattern, _, _), = pattern_matching.get_transformation_metadata([MapToForLoop])[1]
    (_, _, trivial_pattern, _, _), = pattern_matching.get_transformation_metadata([TrivialMapElimination])[1]
    assert pattern_matching.pattern_types_present(map_pattern, {dace.nodes.MapEntry, dace.nodes.Tasklet})
    assert not pattern_matching.pattern_types_present(map_pattern, {dace.nodes.Tasklet, dace.nodes.AccessNode})
    assert not pattern_matching.pattern_types_present(map_pattern, set())
    assert pattern_matching.pattern_types_present(trivial_pattern, {dace.nodes.MapEntry})


if __name__ == "__main__":
    test_prefilter_finds_the_matches_of_a_full_scan()
    test_state_without_pattern_types_is_not_collapsed()
    test_pattern_types_present_needs_every_pattern_node_type()
