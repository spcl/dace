# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

import dace
import dace.serialize
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def test_cond_region_if():
    sdfg = dace.SDFG("regular_if")
    sdfg.add_array("A", (1,), dace.float32)
    sdfg.add_symbol("i", dace.int32)
    state0 = sdfg.add_state("state0", is_start_block=True)

    if1 = ConditionalBlock("if1")
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion("if_body", sdfg=sdfg)
    if1.add_branch(CodeBlock("i == 1"), if_body)

    state1 = if_body.add_state("state1", is_start_block=True)
    acc_a = state1.add_access("A")
    t1 = state1.add_tasklet("t1", None, {"a"}, "a = 100")
    state1.add_edge(t1, "a", acc_a, None, dace.Memlet("A[0]"))

    assert sdfg.is_valid()
    A = np.ones((1,), dtype=np.float32)
    sdfg(i=1, A=A)
    assert A[0] == 100

    A = np.ones((1,), dtype=np.float32)
    sdfg(i=0, A=A)
    assert A[0] == 1


def test_serialization():
    sdfg = SDFG("test_serialization")
    cond_region = ConditionalBlock("cond_region")
    sdfg.add_node(cond_region, is_start_block=True)
    sdfg.add_symbol("i", dace.int32)

    for j in range(10):
        cfg = ControlFlowRegion(f"cfg_{j}", sdfg)
        cfg.add_state("noop")
        cond_region.add_branch(CodeBlock(f"i == {j}"), cfg)

    assert sdfg.is_valid()

    new_sdfg = SDFG.from_json(sdfg.to_json())
    assert new_sdfg.is_valid()
    new_cond_region: ConditionalBlock = new_sdfg.nodes()[0]
    for j in range(10):
        condition, cfg = new_cond_region.branches[j]
        assert condition == CodeBlock(f"i == {j}")
        assert cfg.label == f"cfg_{j}"


def test_nested_conditionals_serialize_each_state_once():
    """A conditional's branches are its nodes: serializing them as nodes too doubled the work per nesting level."""
    depth = 12
    sdfg = SDFG("nested_conditionals_serialize_once")
    sdfg.add_symbol("i", dace.int32)
    region = sdfg
    for level in range(depth):
        cond = ConditionalBlock(f"cond_{level}")
        region.add_node(cond, is_start_block=True)
        branch = ControlFlowRegion(f"branch_{level}", sdfg)
        branch.add_state(f"state_{level}", is_start_block=True)
        cond.add_branch(CodeBlock(f"i > {level}"), branch)
        region = branch
    assert sdfg.is_valid()

    serialized = []
    state_to_json = dace.SDFGState.to_json
    dace.SDFGState.to_json = lambda self, parent=None: serialized.append(self) or state_to_json(self, parent)
    try:
        new_sdfg = SDFG.from_json(sdfg.to_json())
    finally:
        dace.SDFGState.to_json = state_to_json

    assert len(serialized) == depth
    assert new_sdfg.is_valid()
    assert sorted(s.label for s in new_sdfg.states()) == sorted(f"state_{level}" for level in range(depth))


def test_if_else():
    sdfg = dace.SDFG("regular_if_else")
    sdfg.add_array("A", (1,), dace.float32)
    sdfg.add_symbol("i", dace.int32)
    state0 = sdfg.add_state("state0", is_start_block=True)

    if1 = ConditionalBlock("if1")
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion("if_body", sdfg=sdfg)
    state1 = if_body.add_state("state1", is_start_block=True)
    acc_a = state1.add_access("A")
    t1 = state1.add_tasklet("t1", None, {"a"}, "a = 100")
    state1.add_edge(t1, "a", acc_a, None, dace.Memlet("A[0]"))
    if1.add_branch(CodeBlock("i == 1"), if_body)

    else_body = ControlFlowRegion("else_body", sdfg=sdfg)
    state2 = else_body.add_state("state1", is_start_block=True)
    acc_a2 = state2.add_access("A")
    t2 = state2.add_tasklet("t2", None, {"a"}, "a = 200")
    state2.add_edge(t2, "a", acc_a2, None, dace.Memlet("A[0]"))
    if1.add_branch(CodeBlock("i == 0"), else_body)

    assert sdfg.is_valid()
    A = np.ones((1,), dtype=np.float32)
    sdfg(i=1, A=A)
    assert A[0] == 100

    A = np.ones((1,), dtype=np.float32)
    sdfg(i=0, A=A)
    assert A[0] == 200


if __name__ == "__main__":
    test_cond_region_if()
    test_serialization()
    test_nested_conditionals_serialize_each_state_once()
    test_if_else()
