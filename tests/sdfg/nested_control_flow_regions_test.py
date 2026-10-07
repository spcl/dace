# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def test_is_start_state_deprecation():
    sdfg = dace.SDFG("deprecation_test")
    with pytest.deprecated_call():
        sdfg.add_state("state1", is_start_state=True)
    sdfg2 = dace.SDFG("deprecation_test2")
    state = dace.SDFGState("state2")
    with pytest.deprecated_call():
        sdfg2.add_node(state, is_start_state=True)


def test_states_recurse_into_nested_regions_in_graph_order():
    sdfg = dace.SDFG("states_recurse_into_nested_regions")
    first = sdfg.add_state("first", is_start_block=True)
    loop = LoopRegion("loop", "i < 4", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    body = loop.add_state("body", is_start_block=True)
    branch = ControlFlowRegion("branch")
    then_state = branch.add_state("then_state", is_start_block=True)
    conditional = ConditionalBlock("conditional")
    conditional.add_branch(CodeBlock("i > 1"), branch)
    loop.add_node(conditional)
    loop.add_edge(body, conditional, dace.InterstateEdge())
    sdfg.add_edge(first, loop, dace.InterstateEdge())

    assert sdfg.states() == [first, body, then_state], sdfg.states()
    assert loop.states() == [body, then_state], loop.states()


if __name__ == "__main__":
    test_is_start_state_deprecation()
    test_states_recurse_into_nested_regions_in_graph_order()
