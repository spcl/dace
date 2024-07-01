# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import dace
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalRegion, ControlFlowRegion
import dace.serialize


def test_cond_region_if():
    sdfg = dace.SDFG('regular_if')
    sdfg.add_array("A", (1,), dace.float32)
    sdfg.add_symbol("i", dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    
    if1 = ConditionalRegion("if1")
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion("if_body", sdfg=sdfg)
    if1.branches.append((CodeBlock("i == 1"), if_body))

    state1 = if_body.add_state("state1", is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet("t1", None, {"a"}, "a = 100")
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))
    
    def assertions(sdfg):
        assert sdfg.is_valid()
        A = np.ones((1,), dtype=np.float32)
        sdfg(i=1, A=A)
        assert A[0] == 100

        A = np.ones((1,), dtype=np.float32)
        sdfg(i=0, A=A)
        assert A[0] == 1
    
    assertions(sdfg)

    json = sdfg.to_json()
    new_sdfg = SDFG.from_json(json)
    assertions(new_sdfg)

def test_serialization():
    sdfg = SDFG("test_serialization")
    cond_region = ConditionalRegion("cond_region")
    sdfg.add_node(cond_region, is_start_block=True)
    sdfg.add_symbol("i", dace.int32)

    for j in range(10):
        cfg = ControlFlowRegion(f"cfg_{j}", sdfg)
        cond_region.branches.append((CodeBlock(f"i == {j}"), cfg))
    
    assert sdfg.is_valid()

    new_sdfg = SDFG.from_json(sdfg.to_json())
    assert new_sdfg.is_valid()
    new_cond_region: ConditionalRegion = new_sdfg.nodes()[0]
    for j in range(10):
        condition, cfg = new_cond_region.branches[j]
        assert condition == CodeBlock(f"i == {j}")
        assert cfg.label == f"cfg_{j}"

if __name__ == '__main__':
    test_cond_region_if()
    test_serialization()
