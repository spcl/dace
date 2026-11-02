# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrites of the loop structure that must refuse, or keep a value, on a shape they used to miscompile."""

import copy

import numpy as np

import dace
from dace import SDFG, InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg.state import BreakBlock, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import MaterializeLoopExitSymbols
from dace.transformation.passes.canonicalize.perfect_loop_nesting import distribute_loops
from dace.transformation.passes.canonicalize.while_to_for import WhileToFor
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

N = dace.symbol("N", dace.int64)


def copy_statement(state, source, target):
    """``target = source`` in ``state``; no ``source`` writes a constant."""
    tasklet = state.add_tasklet("copy", {"x"} if source else {}, {"y"}, "y = x + 1" if source else "y = 1")
    if source:
        state.add_edge(state.add_read(source.split("[")[0]), None, tasklet, "x", dace.Memlet(source))
    state.add_edge(tasklet, "y", state.add_write(target.split("[")[0]), None, dace.Memlet(target))


def counted_loop(label, var, start=0):
    return LoopRegion(label, f"{var} < N", var, f"{var} = {start}", f"{var} = {var} + 1")


def int_program(name, **arrays):
    sdfg = SDFG(name)
    sdfg.add_symbol("N", dace.int64)
    for array, shape in arrays.items():
        sdfg.add_array(array, shape, dace.int64)
    return sdfg


def break_guard(loop, condition):
    guard = ConditionalBlock("guard")
    loop.add_node(guard, is_start_block=True)
    branch = ControlFlowRegion("branch")
    guard.add_branch(CodeBlock(condition), branch)
    branch.add_node(BreakBlock("stop"), is_start_block=True)
    return guard


def test_distribution_keeps_row_writer_with_column_reader():
    sdfg = int_program("row_column", A=[N, N], B=[N, N])
    outer = counted_loop("outer", "i")
    sdfg.add_node(outer, is_start_block=True)
    first, second = counted_loop("first", "j"), counted_loop("second", "k")
    outer.add_node(first, is_start_block=True)
    outer.add_node(second)
    outer.add_edge(first, second, InterstateEdge())
    copy_statement(first.add_state("write", is_start_block=True), "A[i, j]", "A[i, j]")
    copy_statement(second.add_state("read", is_start_block=True), "A[k, i]", "B[i, k]")
    assert distribute_loops(sdfg) == 0


def test_distribution_keeps_a_while_loop_whole():
    sdfg = int_program("while_body", c=[1], d=[1])
    loop = LoopRegion("loop", "c[0] > 0")
    sdfg.add_node(loop, is_start_block=True)
    first, second = loop.add_state("first", is_start_block=True), loop.add_state("second")
    loop.add_edge(first, second, InterstateEdge())
    copy_statement(first, "c[0]", "c[0]")
    copy_statement(second, "d[0]", "d[0]")
    assert distribute_loops(sdfg) == 0


def test_distribution_keeps_a_break_with_the_statements_it_skips():
    sdfg = int_program("break_body", a=[N], b=[N])
    loop = counted_loop("loop", "i")
    sdfg.add_node(loop, is_start_block=True)
    guard = break_guard(loop, "a[i] < 0")
    body = loop.add_state("body")
    loop.add_edge(guard, body, InterstateEdge())
    copy_statement(body, None, "b[i]")
    assert distribute_loops(sdfg) == 0


def guarded_prep(name, condition):
    sdfg = int_program(name, x=[1], a=[N])
    guard = ConditionalBlock("guard")
    sdfg.add_node(guard, is_start_block=True)
    branch = ControlFlowRegion("branch")
    guard.add_branch(CodeBlock(condition), branch)
    prep = branch.add_state("prep", is_start_block=True)
    copy_statement(prep, None, "x[0]")
    loop = counted_loop("loop", "i")
    branch.add_node(loop)
    branch.add_edge(prep, loop, InterstateEdge())
    copy_statement(loop.add_state("body", is_start_block=True), None, "a[i]")
    return sdfg


def test_guard_stays_above_a_prep_that_writes_its_condition():
    assert MoveIfIntoLoop().apply_pass(guarded_prep("clobbered_guard", "x[0] > 0"), {}) is None


def test_guard_stays_above_a_prep_an_empty_loop_would_skip():
    assert MoveIfIntoLoop().apply_pass(guarded_prep("empty_loop_prep", "N > 1"), {}) is None


def recurrence_pair(name, first_guard=None, second_code="y = x + 1"):
    sdfg = int_program(name, a=[N], b=[N], m=[N])
    first, second = counted_loop("first", "i", 1), counted_loop("second", "j", 1)
    sdfg.add_node(first, is_start_block=True)
    sdfg.add_node(second)
    sdfg.add_edge(first, second, InterstateEdge())
    guard = break_guard(first, first_guard) if first_guard else None
    body = first.add_state("first_body", is_start_block=guard is None)
    if guard:
        first.add_edge(guard, body, InterstateEdge())
    copy_statement(body, "a[i - 1]", "a[i]")
    body = second.add_state("second_body", is_start_block=True)
    tasklet = body.add_tasklet("recur", {"x"}, {"y"}, second_code)
    body.add_edge(body.add_read("b"), None, tasklet, "x", dace.Memlet("b[j - 1]"))
    body.add_edge(tasklet, "y", body.add_write("b"), None, dace.Memlet("b[j]"))
    return sdfg, first, second


def test_fusion_refuses_a_loop_that_breaks():
    sdfg, first, second = recurrence_pair("fused_break", first_guard="m[i] < 0")
    assert not LoopFusion.can_be_applied_to(sdfg, first=first, second=second)


def test_fusion_refuses_a_read_of_the_first_iterators_exit_value():
    sdfg, first, second = recurrence_pair("fused_exit_read", second_code="y = x + i")
    assert not LoopFusion.can_be_applied_to(sdfg, first=first, second=second)


def exit_reader(name, start, after=None):
    """A loop bumping ``k``, then ``o[0] = k`` (``after``: the name read, or an assignment on the exit edge)."""
    sdfg = int_program(name, o=[1], a=[8])
    sdfg.add_symbol("k", dace.int64)
    head = sdfg.add_state("head", is_start_block=True)
    loop = counted_loop("loop", "i", start)
    sdfg.add_node(loop)
    sdfg.add_edge(head, loop, InterstateEdge(assignments={"k": "3"}))
    body, tail = loop.add_state("body", is_start_block=True), loop.add_state("tail")
    loop.add_edge(body, tail, InterstateEdge(assignments={"k": "k + 2"}))
    copy_statement(body, None, "a[i]")
    last = sdfg.add_state("last")
    sdfg.add_edge(loop, last, InterstateEdge(assignments=after or {}))
    tasklet = last.add_tasklet("read", {}, {"y"}, "y = k")
    last.add_edge(tasklet, "y", last.add_write("o"), None, dace.Memlet("o[0]"))
    return sdfg


def run_both(sdfg, transform, **symbols):
    reference = copy.deepcopy(sdfg)
    reference.name += "_reference"
    transform(sdfg)
    results = []
    for program in (reference, sdfg):
        out = np.zeros(1, dtype=np.int64)
        program(o=out, a=np.zeros(8, dtype=np.int64), **symbols)
        results.append(int(out[0]))
    return results


def test_while_to_for_keeps_a_start_value_read_from_an_array_written_before_the_loop():
    sdfg = SDFG("start_reads_array")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_array("o", [1], dace.int64)
    first, second = sdfg.add_state("first", is_start_block=True), sdfg.add_state("second")
    sdfg.add_edge(first, second, InterstateEdge(assignments={"i": "o[0]"}))
    copy_statement(second, None, "o[0]")
    loop = LoopRegion("loop", "i < 8")
    sdfg.add_node(loop)
    sdfg.add_edge(second, loop, InterstateEdge())
    body, tail = loop.add_state("body", is_start_block=True), loop.add_state("tail")
    loop.add_edge(body, tail, InterstateEdge(assignments={"i": "i + 1"}))
    assert WhileToFor().apply_pass(sdfg, {}) is None


def flag_loop(name, entry_flag, flag_first):
    """``i = 0; flag = entry_flag; while flag: ...``, the body assigning ``flag = i < N`` around ``i = i + 1``."""
    sdfg = int_program(name, o=[1])
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("flag", dace.bool_)
    head = sdfg.add_state("head", is_start_block=True)
    loop = LoopRegion("loop", "flag")
    sdfg.add_node(loop)
    sdfg.add_edge(head, loop, InterstateEdge(assignments={"i": "0", "flag": entry_flag}))
    sdfg.add_edge(loop, sdfg.add_state("after"), InterstateEdge())
    body, mid, tail = loop.add_state("body", is_start_block=True), loop.add_state("mid"), loop.add_state("tail")
    bump, test = {"i": "i + 1"}, {"flag": "i < N"}
    loop.add_edge(body, mid, InterstateEdge(assignments=test if flag_first else bump))
    loop.add_edge(mid, tail, InterstateEdge(assignments=bump if flag_first else test))
    return sdfg


def test_while_to_for_needs_the_condition_symbol_to_enter_and_stay_as_its_expression():
    assert WhileToFor().apply_pass(flag_loop("flag_entry", "i < N", False), {}) == 1
    assert WhileToFor().apply_pass(flag_loop("flag_true_on_entry", "1", False), {}) is None
    assert WhileToFor().apply_pass(flag_loop("flag_before_bump", "i < N", True), {}) is None


def test_exit_symbol_of_a_loop_that_runs_no_iteration():
    expected, got = run_both(
        exit_reader("exit_empty", 5), lambda s: MaterializeLoopExitSymbols().apply_pass(s, {}), N=3
    )
    assert expected == got == 3


def test_exit_symbol_leaves_a_later_assignment_alone():
    sdfg = exit_reader("exit_reassigned", 0, {"k": "0"})
    expected, got = run_both(sdfg, lambda s: MaterializeLoopExitSymbols().apply_pass(s, {}), N=3)
    assert expected == got == 0


def test_exit_symbol_of_a_counter_a_branch_also_bumps_stays_a_read():
    sdfg = exit_reader("exit_nested_bump", 0)
    loop = next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))
    guard = ConditionalBlock("guard")
    arm = ControlFlowRegion("arm")
    guard.add_branch(CodeBlock("i > 0"), arm)
    bump = arm.add_state("bump", is_start_block=True)
    arm.add_edge(bump, arm.add_state("bumped"), InterstateEdge(assignments={"k": "k + 1"}))
    loop.add_node(guard)
    loop.add_edge(next(b for b in loop.nodes() if b.label == "tail"), guard, InterstateEdge())
    sdfg.reset_cfg_list()
    expected, got = run_both(sdfg, lambda s: MaterializeLoopExitSymbols().apply_pass(s, {}), N=3)
    assert expected == got == 11


def test_iterator_exit_value_of_a_loop_that_runs_no_iteration():
    sdfg = int_program("iterator_exit_empty", o=[1], a=[8])
    loop = counted_loop("loop", "i", 5)
    sdfg.add_node(loop, is_start_block=True)
    copy_statement(loop.add_state("body", is_start_block=True), None, "a[i]")
    last = sdfg.add_state("last")
    sdfg.add_edge(loop, last, InterstateEdge())
    tasklet = last.add_tasklet("read", {}, {"y"}, "y = i")
    last.add_edge(tasklet, "y", last.add_write("o"), None, dace.Memlet("o[0]"))
    expected, got = run_both(sdfg, lambda s: UniqueLoopIterators().apply_pass(s, {}), N=3)
    assert expected == got == 5


def test_while_to_for_keeps_a_start_value_changed_before_the_loop():
    sdfg = SDFG("start_changed")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_array("o", [1], dace.int64)
    first, second, third = (
        sdfg.add_state("first", is_start_block=True),
        sdfg.add_state("second"),
        sdfg.add_state("third"),
    )
    sdfg.add_edge(first, second, InterstateEdge(assignments={"m": "2"}))
    sdfg.add_edge(second, third, InterstateEdge(assignments={"i": "m"}))
    fourth = sdfg.add_state("fourth")
    sdfg.add_edge(third, fourth, InterstateEdge(assignments={"m": "6"}))
    loop = LoopRegion("loop", "i < 8")
    sdfg.add_node(loop)
    sdfg.add_edge(fourth, loop, InterstateEdge())
    body, tail = loop.add_state("body", is_start_block=True), loop.add_state("tail")
    loop.add_edge(body, tail, InterstateEdge(assignments={"i": "i + 1"}))
    copy_statement(body, "o[0]", "o[0]")
    assert WhileToFor().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_distribution_keeps_row_writer_with_column_reader()
    test_distribution_keeps_a_while_loop_whole()
    test_distribution_keeps_a_break_with_the_statements_it_skips()
    test_guard_stays_above_a_prep_that_writes_its_condition()
    test_guard_stays_above_a_prep_an_empty_loop_would_skip()
    test_fusion_refuses_a_loop_that_breaks()
    test_fusion_refuses_a_read_of_the_first_iterators_exit_value()
    test_exit_symbol_of_a_loop_that_runs_no_iteration()
    test_exit_symbol_leaves_a_later_assignment_alone()
    test_exit_symbol_of_a_counter_a_branch_also_bumps_stays_a_read()
    test_iterator_exit_value_of_a_loop_that_runs_no_iteration()
    test_while_to_for_keeps_a_start_value_changed_before_the_loop()
    test_while_to_for_keeps_a_start_value_read_from_an_array_written_before_the_loop()
    test_while_to_for_needs_the_condition_symbol_to_enter_and_stay_as_its_expression()
