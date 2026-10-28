# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An access every execution makes is in bounds, so it bounds the extent symbols below."""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.sdfg.symbol_repo import program_assumptions
from dace.transformation.passes.canonicalize.in_bounds_facts import declare_in_bounds_extents


def write_at(parent, index: str) -> None:
    state = parent.add_state("write", is_start_block=True)
    tasklet = state.add_tasklet("one", {}, {"o"}, "o = 1")
    state.add_edge(tasklet, "o", state.add_write("A"), None, dace.Memlet(f"A[{index}]"))


def sdfg_over(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("NZ", dace.int64)
    sdfg.add_array("A", ["NZ"], dace.float64)
    return sdfg


def test_an_always_executed_access_bounds_its_extent():
    sdfg = sdfg_over("always")
    write_at(sdfg, "NZ - 2")
    assert declare_in_bounds_extents(sdfg) == {"NZ": 2}
    assert any("NZ" in str(relation) for relation in program_assumptions(sdfg))


def test_an_access_in_a_branch_bounds_nothing():
    sdfg = sdfg_over("branch")
    guard = ConditionalBlock("guard")
    sdfg.add_node(guard, is_start_block=True)
    arm = ControlFlowRegion("arm", sdfg=sdfg)
    guard.add_branch(CodeBlock("NZ > 3"), arm)
    write_at(arm, "NZ - 2")
    assert declare_in_bounds_extents(sdfg) == {}
