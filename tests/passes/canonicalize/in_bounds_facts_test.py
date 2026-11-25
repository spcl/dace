# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An access every execution makes is in bounds, so it bounds the extent symbols below."""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, ReturnBlock
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


def test_an_access_after_a_possible_return_bounds_nothing():
    sdfg = sdfg_over("after_return")
    guard = ConditionalBlock("guard")
    sdfg.add_node(guard, is_start_block=True)
    arm = ControlFlowRegion("arm", sdfg=sdfg)
    arm.add_node(ReturnBlock("leave"), is_start_block=True)
    guard.add_branch(CodeBlock("NZ > 3"), arm)
    after = sdfg.add_state_after(guard, "after")
    tasklet = after.add_tasklet("one", {}, {"o"}, "o = 1")
    after.add_edge(tasklet, "o", after.add_write("A"), None, dace.Memlet("A[NZ - 2]"))
    assert declare_in_bounds_extents(sdfg) == {}


def read_under_map(name: str, extent: str) -> dace.SDFG:
    sdfg = sdfg_over(name)
    sdfg.add_symbol("M", dace.int64)
    sdfg.add_array("B", [extent], dace.float64)
    state = sdfg.add_state("broadcast", is_start_block=True)
    state.add_mapped_tasklet(
        "broadcast",
        {"i": f"0:{extent}"},
        {"a": dace.Memlet("A[NZ - 3]")},
        "b = a",
        {"b": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def write_in_nested_under_map(name: str) -> dace.SDFG:
    sdfg = sdfg_over(name)
    sdfg.add_symbol("M", dace.int64)
    body = sdfg_over(f"{name}_body")
    write_at(body, "NZ - 2")
    state = sdfg.add_state("lifted", is_start_block=True)
    entry, exit_node = state.add_map("lifted", {"i": "0:M"})
    nested = state.add_nested_sdfg(body, {}, {"A"}, symbol_mapping={"NZ": "NZ"})
    state.add_nedge(entry, nested, dace.Memlet())
    state.add_memlet_path(nested, exit_node, state.add_write("A"), src_conn="A", memlet=dace.Memlet("A[NZ - 2]"))
    return sdfg


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_an_access_under_a_map_that_may_not_run_bounds_nothing():
    """With ``M == 0`` neither map runs, so ``NZ == 1`` is a valid input and no bound on ``NZ`` holds."""
    assert declare_in_bounds_extents(read_under_map("possibly_empty_map", "M")) == {}
    assert declare_in_bounds_extents(write_in_nested_under_map("possibly_empty_nested")) == {}
    assert declare_in_bounds_extents(read_under_map("nonempty_map", "4")) == {"NZ": 3}


if __name__ == "__main__":
    test_an_always_executed_access_bounds_its_extent()
    test_an_access_in_a_branch_bounds_nothing()
    test_an_access_after_a_possible_return_bounds_nothing()
    test_an_access_under_a_map_that_may_not_run_bounds_nothing()
