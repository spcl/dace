# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``pipeline_depth`` property of :class:`LoopRegion`."""

import copy

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


if __name__ == "__main__":
    test_default_is_zero()
    test_json_round_trip()
    test_deepcopy()
    test_schedule_tree_round_trip()
