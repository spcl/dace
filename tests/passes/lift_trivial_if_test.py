# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.sdfg import CodeBlock, ConditionalBlock
from dace import ControlFlowRegion
from dace.transformation.passes.lift_trivially_true_if import LiftTriviallyTrueIf
import pytest

# Always True conditions
always_true = [
    "True",
    "1 == 1",
    "2 > 1",
    "5 >= 5",
    "not False",
    "2 + 2 == 4",
    "abs(-5) == 5",
    "max(1, 2, 3) == 3",
]

# Always False conditions
always_false = [
    "False",
    "1 == 2",
    "5 < 3",
    "10 <= 9",
    "not True",
    "2 + 2 == 5",
    "abs(-5) == -5",
]

cant_eval = ["a < 5", "c == 0", "d >= 1"]


def _get_sdfg(condition: str):
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg = ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition), branch=cfg)
    s1 = cfg.add_state(label="s1")
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_if_else_sdfg(condition: str, body_in_else_branch: bool):
    sdfg = dace.SDFG("if_else_1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg_if_true = ControlFlowRegion(label="cfg_if_true", sdfg=cb.sdfg, parent=cb)
    cfg_if_false = ControlFlowRegion(label="cfg_if_false", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition), branch=cfg_if_true)
    cb.add_branch(condition=None, branch=cfg_if_false)
    if body_in_else_branch:
        s1 = cfg_if_true.add_state(label="s1", is_start_block=True)
        s2 = cfg_if_false.add_state(label="fs1", is_start_block=True)
    else:
        s1 = cfg_if_false.add_state(label="s1", is_start_block=True)
        s2 = cfg_if_true.add_state(label="ts1", is_start_block=True)
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


def _get_nested_sdfg(condition1: str, condition2: str):
    sdfg = dace.SDFG("nested1")
    sdfg = dace.SDFG("basic1")
    _, A = sdfg.add_array(name="A", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    _, B = sdfg.add_array(name="B", shape=[
        5,
    ], dtype=dace.float64, transient=False)
    cb = ConditionalBlock(label="cfb1", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    cfg1 = ControlFlowRegion(label="cfg1", sdfg=cb.sdfg, parent=cb)
    cb.add_branch(condition=CodeBlock(condition1), branch=cfg1)
    cb2 = ConditionalBlock(label="cfb2", sdfg=cfg1.sdfg, parent=cfg1)
    cfg1.add_node(cb2, is_start_block=True)
    cfg2 = ControlFlowRegion(label="cfg2", sdfg=cb2.sdfg, parent=cb2)
    cb2.add_branch(condition=CodeBlock(condition2), branch=cfg2)
    s1 = cfg2.add_state(label="s1", is_start_block=True)
    aA = s1.add_access("A")
    aB = s1.add_access("B")
    s1.add_edge(aA, None, aB, None, dace.memlet.Memlet.from_array("A", A))

    return sdfg


@pytest.mark.parametrize("condition", always_true)
def test_single_condition(condition: str):
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition", cant_eval)
def test_single_condition_cant_eval(condition: str):
    sdfg = _get_sdfg(condition)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 1


@pytest.mark.parametrize("condition1,condition2",
                         [(always_true[i], always_true[i + 1]) for i in range(len(always_true) - 1)])
def test_nested_condition(condition1: str, condition2: str):
    sdfg = _get_nested_sdfg(condition1, condition2)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition1,condition2", [(cant_eval[i], cant_eval[i + 1]) for i in range(len(cant_eval) - 1)])
def test_nested_condition_cant_eval(condition1: str, condition2: str):
    sdfg = _get_nested_sdfg(condition1, condition2)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 2


@pytest.mark.parametrize("condition", always_true)
def test_if_else_cond_is_trivially_true(condition: str):
    sdfg = _get_if_else_sdfg(condition, False)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


@pytest.mark.parametrize("condition", always_false)
def test_if_else_cond_is_trivially_false(condition: str):
    sdfg = _get_if_else_sdfg(condition, True)
    sdfg.validate()
    LiftTriviallyTrueIf().apply_pass(sdfg, {})
    sdfg.validate()
    assert len({n for n in sdfg.all_control_flow_blocks() if isinstance(n, ConditionalBlock)}) == 0


if __name__ == "__main__":
    for c in always_true:
        test_single_condition(c)

    for i in range(len(always_true) - 1):
        c1 = always_true[i]
        c2 = always_true[i + 1]
        test_nested_condition(c1, c2)

    for c in always_true:
        test_if_else_cond_is_trivially_true(c)

    for c in always_false:
        test_if_else_cond_is_trivially_false(c)
