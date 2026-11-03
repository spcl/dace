# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineMultistateSDFG leaves a callee that can reach the end of the nest other than through a sink: spliced in, that
path would end the caller instead of returning to it."""

import dace
from dace.sdfg.state import ReturnBlock
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG


def caller_of(callee: dace.SDFG) -> dace.SDFG:
    sdfg = dace.SDFG(f"caller_of_{callee.name}")
    sdfg.add_symbol("x", dace.int64)
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_edge(state, sdfg.add_state(), dace.InterstateEdge())
    state.add_nested_sdfg(callee, {}, {}, symbol_mapping={"x": "x"})
    return sdfg


def test_an_early_return_stays_nested():
    callee = dace.SDFG("early_return")
    callee.add_symbol("x", dace.int64)
    first = callee.add_state(is_start_block=True)
    exit_block = ReturnBlock("exit")
    callee.add_node(exit_block)
    callee.add_edge(first, exit_block, dace.InterstateEdge("x <= 0"))
    callee.add_edge(first, callee.add_state(), dace.InterstateEdge("x > 0"))
    assert caller_of(callee).apply_transformations_repeated(InlineMultistateSDFG) == 0


def test_a_condition_that_can_fail_stays_nested():
    callee = dace.SDFG("falls_off")
    callee.add_symbol("x", dace.int64)
    callee.add_edge(callee.add_state(is_start_block=True), callee.add_state(), dace.InterstateEdge("x > 0"))
    assert caller_of(callee).apply_transformations_repeated(InlineMultistateSDFG) == 0


if __name__ == "__main__":
    test_an_early_return_stays_nested()
    test_a_condition_that_can_fail_stays_nested()
