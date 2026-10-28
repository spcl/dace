# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RemoveRedundantLoopGuard`` sinks ``if count != 0`` back into a loop whose iterations are already guarded by the
predicate ``count`` counted, and leaves every guard it cannot prove implied."""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.simplification.remove_redundant_loop_guard import RemoveRedundantLoopGuard

K, N = 5, 37


def clipped_rows(
    inner_check: str = "clip[k, j] != 0", unguarded_write: bool = False, count_read_after: bool = False
) -> dace.SDFG:
    """Per row ``k``: count the clipped ``i``, then, if any, double ``A`` where clipped (ICON's CFL ``clip_count``)."""
    sdfg = dace.SDFG("clipped_rows")
    sdfg.add_array("A", [K, N], dace.float64)
    sdfg.add_array("B", [K, N], dace.float64)
    sdfg.add_array("clip", [K, N], dace.int32)
    sdfg.add_array("counts", [K], dace.int64)
    for name in ("count", "p", "q", "seen"):
        sdfg.add_symbol(name, dace.int64)

    rows = LoopRegion("rows", f"k < {K}", "k", "k = 0", "k = k + 1")
    sdfg.add_node(rows, is_start_block=True)
    reset = rows.add_state("reset", is_start_block=True)

    counting = LoopRegion("counting", f"i < {N}", "i", "i = 0", "i = i + 1")
    rows.add_node(counting)
    rows.add_edge(reset, counting, dace.InterstateEdge(assignments={"count": "0"}))
    mark = counting.add_state("mark", is_start_block=True)
    tasklet = mark.add_tasklet("mark", {"a"}, {"c"}, "c = abs(a) > 1.0")
    mark.add_edge(mark.add_read("A"), None, tasklet, "a", dace.Memlet("A[k, i]"))
    mark.add_edge(tasklet, "c", mark.add_write("clip"), None, dace.Memlet("clip[k, i]"))
    increment = ConditionalBlock("increment")
    counting.add_node(increment)
    counting.add_edge(mark, increment, dace.InterstateEdge(assignments={"p": "clip[k, i] != 0"}))
    branch = ControlFlowRegion("increment_body")
    increment.add_branch("p", branch)
    first = branch.add_state("first", is_start_block=True)
    branch.add_edge(first, branch.add_state("second"), dace.InterstateEdge(assignments={"count": "count + 1"}))

    guard = ConditionalBlock("guard")
    rows.add_node(guard)
    rows.add_edge(counting, guard, dace.InterstateEdge())
    body = ControlFlowRegion("guard_body")
    guard.add_branch("count != 0", body)
    guarded = LoopRegion("guarded", f"j < {N}", "j", "j = 0", "j = j + 1")
    body.add_node(guarded, is_start_block=True)
    check = guarded.add_state("check", is_start_block=True)
    if unguarded_write:
        tasklet = check.add_tasklet("touch", {}, {"b"}, "b = -1.0")
        check.add_edge(tasklet, "b", check.add_write("B"), None, dace.Memlet("B[k, j]"))
    work = ConditionalBlock("work")
    guarded.add_node(work)
    guarded.add_edge(check, work, dace.InterstateEdge(assignments={"q": inner_check}))
    work_body = ControlFlowRegion("work_body")
    work.add_branch("q", work_body)
    double = work_body.add_state("double", is_start_block=True)
    tasklet = double.add_tasklet("double", {"a"}, {"b"}, "b = 2 * a")
    double.add_edge(double.add_read("A"), None, tasklet, "a", dace.Memlet("A[k, j]"))
    double.add_edge(tasklet, "b", double.add_write("B"), None, dace.Memlet("B[k, j]"))

    if count_read_after:
        record = rows.add_state("record")
        rows.add_edge(guard, record, dace.InterstateEdge(assignments={"seen": "count"}))
        tasklet = record.add_tasklet("record", {}, {"out"}, "out = seen")
        record.add_edge(tasklet, "out", record.add_write("counts"), None, dace.Memlet("counts[k]"))
    return sdfg


def guards_on(sdfg: dace.SDFG, name: str) -> list[ConditionalBlock]:
    return [
        b
        for b in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(b, ConditionalBlock) and any(c and name in c.get_free_symbols() for c, _ in b.branches)
    ]


def inputs() -> dict[str, np.ndarray]:
    a = np.random.default_rng(7).uniform(-2.0, 2.0, (K, N))
    a[1] = 0.5  # a row with nothing clipped
    return dict(A=a, B=np.zeros((K, N)), clip=np.zeros((K, N), np.int32), counts=np.zeros(K, np.int64))


def run(sdfg: dace.SDFG) -> dict[str, np.ndarray]:
    args = inputs()
    sdfg(**args)
    return args


def test_a_guard_on_a_count_of_the_inner_predicate_is_dropped():
    expected = run(clipped_rows())
    sdfg = clipped_rows()

    assert RemoveRedundantLoopGuard().apply_pass(sdfg, {}) == 1

    sdfg.validate()
    assert guards_on(sdfg, "count") == []
    assert all(
        "count" not in e.data.assignments for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges()
    )
    assert len(guards_on(sdfg, "q")) == 1  # The per-iteration guard stays
    got = run(sdfg)
    np.testing.assert_array_equal(got["B"], expected["B"])
    np.testing.assert_array_equal(got["clip"], expected["clip"])


def test_a_write_outside_the_inner_guard_keeps_the_guard():
    sdfg = clipped_rows(unguarded_write=True)
    assert RemoveRedundantLoopGuard().apply_pass(sdfg, {}) is None
    assert len(guards_on(sdfg, "count")) == 1


def test_a_different_inner_predicate_keeps_the_guard():
    sdfg = clipped_rows(inner_check="clip[k, j] != 1")
    assert RemoveRedundantLoopGuard().apply_pass(sdfg, {}) is None
    assert len(guards_on(sdfg, "count")) == 1


def test_a_predicate_on_another_element_keeps_the_guard():
    sdfg = clipped_rows(inner_check="clip[k, 0] != 0")
    assert RemoveRedundantLoopGuard().apply_pass(sdfg, {}) is None
    assert len(guards_on(sdfg, "count")) == 1


def test_a_count_read_after_the_guard_keeps_the_guard():
    sdfg = clipped_rows(count_read_after=True)
    assert RemoveRedundantLoopGuard().apply_pass(sdfg, {}) is None
    assert len(guards_on(sdfg, "count")) == 1


if __name__ == "__main__":
    test_a_guard_on_a_count_of_the_inner_predicate_is_dropped()
    test_a_write_outside_the_inner_guard_keeps_the_guard()
    test_a_different_inner_predicate_keeps_the_guard()
    test_a_predicate_on_another_element_keeps_the_guard()
    test_a_count_read_after_the_guard_keeps_the_guard()
