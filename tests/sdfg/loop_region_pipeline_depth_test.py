# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``pipeline_depth`` property of :class:`LoopRegion`."""

import copy
import os
import tempfile

import pytest

import dace
from dace.sdfg.state import LoopRegion


def _loop_sdfg(pipeline_depth: int = 0) -> dace.SDFG:
    sdfg = dace.SDFG("pipelined_loop")
    sdfg.add_array("A", [16], dace.float64)
    loop = LoopRegion("loop", "i < 16", "i", "i = 0", "i = i + 1", pipeline_depth=pipeline_depth)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("write", {}, {"o"}, "o = i")
    body.add_edge(tasklet, "o", body.add_write("A"), None, dace.Memlet("A[i]"))
    return sdfg


def _loop(sdfg: dace.SDFG) -> LoopRegion:
    (loop,) = [region for region in sdfg.all_control_flow_regions() if isinstance(region, LoopRegion)]
    return loop


def test_default_is_zero():
    assert LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1").pipeline_depth == 0


def test_json_round_trip():
    sdfg = _loop_sdfg(3)
    loaded = dace.SDFG.from_json(sdfg.to_json())
    assert _loop(loaded).pipeline_depth == 3


def test_deepcopy():
    sdfg = _loop_sdfg(4)
    assert _loop(copy.deepcopy(sdfg)).pipeline_depth == 4


def test_schedule_tree_round_trip():
    sdfg = _loop_sdfg(2)
    rebuilt = sdfg.as_schedule_tree().as_sdfg()
    assert _loop(rebuilt).pipeline_depth == 2


def test_set_after_construction():
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    loop.pipeline_depth = 5
    assert loop.pipeline_depth == 5
    loop.pipeline_depth = 0
    assert loop.pipeline_depth == 0


def test_non_integer_is_rejected():
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    with pytest.raises((TypeError, ValueError)):
        loop.pipeline_depth = "deep"


def test_json_without_property_loads_as_default():
    """SDFG files written before the property existed have no ``pipeline_depth`` entry."""
    json_sdfg = _loop_sdfg(3).to_json()
    removed = []

    def strip(node):
        if isinstance(node, dict):
            if "pipeline_depth" in node:
                del node["pipeline_depth"]
                removed.append(node)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(json_sdfg)
    assert removed
    assert _loop(dace.SDFG.from_json(json_sdfg)).pipeline_depth == 0


def test_default_survives_round_trip_and_depth_one():
    for depth in (0, 1):
        loaded = dace.SDFG.from_json(_loop_sdfg(depth).to_json())
        assert _loop(loaded).pipeline_depth == depth


def test_file_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pipelined.sdfg")
        _loop_sdfg(6).save(path)
        assert _loop(dace.SDFG.from_file(path)).pipeline_depth == 6


def test_deepcopy_is_independent():
    sdfg = _loop_sdfg(4)
    clone = copy.deepcopy(sdfg)
    _loop(clone).pipeline_depth = 2
    assert _loop(sdfg).pipeline_depth == 4


def test_nested_loops_keep_their_own_depth():
    sdfg = dace.SDFG("nested_pipelined_loops")
    outer = LoopRegion("outer", "i < 4", "i", "i = 0", "i = i + 1", pipeline_depth=3)
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "j < 4", "j", "j = 0", "j = j + 1", pipeline_depth=2)
    outer.add_node(inner, is_start_block=True)
    inner.add_state("body", is_start_block=True)
    loaded = dace.SDFG.from_json(sdfg.to_json())
    depths = {
        r.label: r.pipeline_depth for r in loaded.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)
    }
    assert depths == {"outer": 3, "inner": 2}


if __name__ == "__main__":
    test_default_is_zero()
    test_json_round_trip()
    test_deepcopy()
    test_schedule_tree_round_trip()
    test_set_after_construction()
    test_non_integer_is_rejected()
    test_json_without_property_loads_as_default()
    test_default_survives_round_trip_and_depth_one()
    test_file_round_trip()
    test_deepcopy_is_independent()
    test_nested_loops_keep_their_own_depth()
