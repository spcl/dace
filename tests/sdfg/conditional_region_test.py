# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalRegion, ControlFlowRegion
import dace.serialize


def test_cond_region_if():
    sdfg = dace.SDFG('regular_if')
    sdfg.add_symbol("i", dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    
    if1 = ConditionalRegion("if1")
    if_body = ControlFlowRegion("if_body")
    state1 = if_body.add_state("state1", is_start_block=True)
    state2 = if_body.add_state("state2")
    if_body.add_edge(state1, state2, InterstateEdge(assignments={"i": "100"}))
    if1.branches.append((CodeBlock("i == 1"), if_body))

    sdfg.add_edge(state0, if1, InterstateEdge())
    
    assert sdfg.is_valid()

    json = sdfg.to_json()
    new_sdfg = SDFG.from_json(json)

    assert new_sdfg.is_valid()

if __name__ == '__main__':
    test_cond_region_if()
