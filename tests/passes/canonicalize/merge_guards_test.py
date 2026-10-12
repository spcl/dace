# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MergeGuards``: equal or opposite consecutive guards merge, a guard alone inside another folds into it, and
anything else stays apart, so no body is ever copied."""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.canonicalize.merge_guards import MergeGuards


def guard(
    sdfg: dace.SDFG, label: str, condition: str, target: str, value: str, read: str | None = None
) -> ConditionalBlock:
    """``if condition: target[0] = value``, ``value`` naming ``i`` for ``read[0]`` if ``read`` is given."""
    block = ConditionalBlock(label, sdfg=sdfg)
    body = ControlFlowRegion(f"{label}_body")
    state = body.add_state(is_start_block=True)
    write = state.add_tasklet(f"{label}_write", {"i"} if read else set(), {"o"}, f"o = {value}")
    if read:
        state.add_edge(state.add_read(read), None, write, "i", dace.Memlet(f"{read}[0]"))
    state.add_edge(write, "o", state.add_write(target), None, dace.Memlet(f"{target}[0]"))
    block.add_branch(CodeBlock(condition), body)
    return block


def two_guards(name: str, first: str, second: str, second_value: str = "2.0") -> dace.SDFG:
    """``if first: R[0] = 1``, then ``if second: S[0] = second_value``."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("c", dace.int64)
    sdfg.add_array("R", [1], dace.float64)
    sdfg.add_array("S", [1], dace.float64)
    one = guard(sdfg, "one", first, "R", "1.0")
    two = guard(sdfg, "two", second, "S", second_value)
    sdfg.add_node(one, is_start_block=True)
    sdfg.add_node(two)
    sdfg.add_edge(one, two, dace.InterstateEdge())
    return sdfg


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]


def run(sdfg: dace.SDFG, c: int) -> tuple[float, float]:
    R, S = np.zeros(1), np.zeros(1)
    sdfg(R=R, S=S, c=c)
    return R[0], S[0]


def test_equal_guards_merge_into_one_branch():
    sdfg = two_guards("merge_guards_equal", "c > 0", "c > 0")
    assert MergeGuards().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (block,) = conditionals(sdfg)
    assert len(block.branches) == 1
    assert run(sdfg, 1) == (1.0, 2.0) and run(sdfg, 0) == (0.0, 0.0)


def test_opposite_guards_become_an_if_else():
    sdfg = two_guards("merge_guards_opposite", "c > 0", "c <= 0")
    assert MergeGuards().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (block,) = conditionals(sdfg)
    assert [condition is None for condition, _ in block.branches] == [False, True]
    assert run(sdfg, 1) == (1.0, 0.0) and run(sdfg, 0) == (0.0, 2.0)


def test_different_guards_stay_apart():
    """Merging ``c > 0`` with ``c > 5`` takes a copy of a body per combination of branches: no merge."""
    sdfg = two_guards("merge_guards_different", "c > 0", "c > 5")
    assert MergeGuards().apply_pass(sdfg, {}) is None
    assert len(conditionals(sdfg)) == 2


def test_a_second_guard_reading_what_the_first_body_writes_stays_apart():
    """``if R[0] > 0`` after the body that writes ``R`` must test the written value."""
    sdfg = dace.SDFG("merge_guards_reads_write")
    sdfg.add_array("R", [1], dace.float64)
    sdfg.add_array("S", [1], dace.float64)
    one = guard(sdfg, "one", "R[0] > 0", "R", "i - 1.0", read="R")
    two = guard(sdfg, "two", "R[0] > 0", "S", "2.0")
    sdfg.add_node(one, is_start_block=True)
    sdfg.add_node(two)
    sdfg.add_edge(one, two, dace.InterstateEdge())
    sdfg.validate()
    assert MergeGuards().apply_pass(sdfg, {}) is None


def test_a_link_with_assignments_keeps_its_guards_apart():
    """Merged, ``k = 3`` between the guards would run only on the guarded path."""
    sdfg = two_guards("merge_guards_link_assignment", "c > 0", "c > 0")
    sdfg.add_symbol("k", dace.int64)
    (edge,) = sdfg.edges()
    edge.data.assignments["k"] = "3"
    assert MergeGuards().apply_pass(sdfg, {}) is None


def test_a_guard_alone_inside_another_folds_into_it():
    sdfg = dace.SDFG("merge_guards_nested")
    sdfg.add_symbol("c", dace.int64)
    sdfg.add_array("R", [1], dace.float64)
    sdfg.add_array("S", [1], dace.float64)
    outer = ConditionalBlock("outer", sdfg=sdfg)
    branch = ControlFlowRegion("outer_body")
    inner = guard(sdfg, "inner", "c < 5", "R", "1.0")
    branch.add_node(inner, is_start_block=True)
    outer.add_branch(CodeBlock("c > 0"), branch)
    sdfg.add_node(outer, is_start_block=True)
    sdfg.validate()
    assert MergeGuards().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (block,) = conditionals(sdfg)
    assert len(block.branches) == 1 and "and" in block.branches[0][0].as_string
    assert [run(sdfg, c)[0] for c in (0, 3, 7)] == [0.0, 1.0, 0.0]


if __name__ == "__main__":
    test_equal_guards_merge_into_one_branch()
    test_opposite_guards_become_an_if_else()
    test_different_guards_stay_apart()
    test_a_second_guard_reading_what_the_first_body_writes_stays_apart()
    test_a_link_with_assignments_keeps_its_guards_apart()
    test_a_guard_alone_inside_another_folds_into_it()
