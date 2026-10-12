# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for must-writes and reachability of values between the statements of schedule trees."""

import sympy

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.dataflow import ReachingAnalysis, covered
from dace.sdfg.state import LoopRegion


def _root(*children: tn.ScheduleTreeNode) -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG("dataflow")
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    stree.add_children(list(children))
    return stree


def _loop(var: str, start, end, body: list, step: int = 1) -> tn.ForScope:
    condition = f"{var} < {end}" if step > 0 else f"{var} > {end}"
    loop = LoopRegion(f"loop_{var}", condition, var, f"{var} = {start}", f"{var} = {var} + {step}")
    return tn.ForScope(loop=loop, children=body)


def _tasklet(code: str, inputs: dict[str, str], outputs: dict[str, str]) -> tn.TaskletNode:
    tasklet = dace.nodes.Tasklet("compute", set(inputs), set(outputs), code)
    return tn.TaskletNode(
        node=tasklet,
        in_memlets={c: dace.Memlet(m) for c, m in inputs.items()},
        out_memlets={c: dace.Memlet(m) for c, m in outputs.items()},
    )


def _write(memlet: str) -> tn.TaskletNode:
    return _tasklet("o = 1", {}, {"o": memlet})


def _read(memlet: str, out: str = "out[0]") -> tn.TaskletNode:
    return _tasklet("o = x", {"x": memlet}, {"o": out})


def _reaches(stree: tn.ScheduleTreeRoot, write: tn.TaskletNode, read: tn.TaskletNode) -> bool:
    return ReachingAnalysis(stree).may_reach(write, write.out_memlets["o"], read, read.in_memlets["x"])


def _boxes(writes: dict, name: str) -> list:
    return [[(sympy.simplify(lo), sympy.simplify(hi)) for lo, hi in box] for box in writes.get(name, [])]


def test_must_writes_over_iterations():
    analysis = ReachingAnalysis(_root())
    nest = _loop("k", 0, 8, [_loop("i", 0, 16, [_write("A[k, i]")])])
    assert _boxes(analysis.must_writes(nest), "A") == [[(0, 7), (0, 15)]]
    backwards = _loop("k", 7, -1, [_write("A[k]")], step=-1)
    assert _boxes(analysis.must_writes(backwards), "A") == [[(0, 7)]]
    shifted = _loop("k", 1, 8, [_write("A[k - 1]")])
    assert _boxes(analysis.must_writes(shifted), "A") == [[(0, 6)]]


def test_must_writes_exclude_gaps_and_possibly_empty_loops():
    analysis = ReachingAnalysis(_root())
    assert analysis.must_writes(_loop("k", 0, 8, [_write("A[2 * k]")])) == {}  # Every other element
    assert analysis.must_writes(_loop("k", 0, 8, [_write("A[k, k]")])) == {}  # A diagonal
    assert analysis.must_writes(_loop("k", 0, 8, [_write("A[k]")], step=2)) == {}
    assert analysis.must_writes(_loop("k", 0, "N", [_write("A[k]")])) == {}  # May run no iteration
    assert (
        analysis.must_writes(_loop("k", 0, 8, [tn.IfScope(condition=CodeBlock("c > 0"), children=[_write("A[k]")])]))
        == {}
    )


def test_must_writes_of_both_branches():
    analysis = ReachingAnalysis(_root())
    branches = [
        tn.IfScope(condition=CodeBlock("c > 0"), children=[_write("A[k]"), _write("B[k]")]),
        tn.ElseScope(children=[_write("A[k]")]),
    ]
    writes = analysis.must_writes(_loop("k", 0, 8, branches))
    assert _boxes(writes, "A") == [[(0, 7)]]
    assert "B" not in writes


def test_covered_symbolic_bounds():
    k, n = sympy.symbols("k N")
    assert covered([(k, k), (0, 3)], [[(k - 1, k + 1), (0, 1)], [(k, k), (2, 3)]])
    assert not covered([(k, k), (0, 3)], [[(k - 1, k + 1), (0, 1)]])
    assert covered([(0, n - 1)], [[(0, n - 1)]])
    assert not covered([(0, n - 1)], [[(0, n)]])  # Symbolic extents must match exactly


def test_overwritten_value_not_reached():
    producer, consumer = _write("A[i]"), _read("A[i]")
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, 8, [_write("A[i]")]), _loop("i", 0, 8, [consumer]))
    assert not _reaches(stree, producer, consumer)


def test_conditionally_overwritten_value_reached():
    producer, consumer = _write("A[i]"), _read("A[i]")
    overwrite = tn.IfScope(condition=CodeBlock("c > 0"), children=[_write("A[i]")])
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, 8, [overwrite]), _loop("i", 0, 8, [consumer]))
    assert _reaches(stree, producer, consumer)


def test_value_overwritten_in_both_branches_not_reached():
    producer, consumer = _write("A[i]"), _read("A[i]")
    branches = [
        tn.IfScope(condition=CodeBlock("c > 0"), children=[_write("A[i]")]),
        tn.ElseScope(children=[_tasklet("o = 2", {}, {"o": "A[i]"})]),
    ]
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, 8, branches), _loop("i", 0, 8, [consumer]))
    assert not _reaches(stree, producer, consumer)


def test_partly_overwritten_value_reached():
    producer, consumer = _write("A[i]"), _read("A[i]")
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 1, 8, [_write("A[i]")]), _loop("i", 0, 8, [consumer]))
    assert _reaches(stree, producer, consumer)


def test_overwrite_in_possibly_empty_loop_does_not_hide():
    producer, consumer = _write("A[i]"), _read("A[i]")
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, "N", [_write("A[i]")]), _loop("i", 0, 8, [consumer]))
    assert _reaches(stree, producer, consumer)


def test_disjoint_elements_not_reached():
    producer, consumer = _write("A[i]"), _read("A[i + 8]")
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, 8, [consumer]))
    assert not _reaches(stree, producer, consumer)


def test_independent_chain_in_shared_temporary():
    """A temporary shared by unrelated values: the producer writes levels 0..n-1; a later loop over levels 1..n
    rewrites each level of the temporary and reads it back in the same iteration, and a final read of level 0 sees the
    producer's value."""
    producer = _write("T[k, i]")
    inner_read = _read("T[k, i]", "out[k, i]")
    final_read = _read("T[0, i]", "out[0, i]")
    stree = _root(
        _loop("k", 0, 8, [_loop("i", 0, 4, [producer])]),
        _loop("k", 1, 9, [_loop("i", 0, 4, [_write("T[k, i]")]), _loop("i", 0, 4, [inner_read])]),
        _loop("i", 0, 4, [final_read]),
    )
    assert not _reaches(stree, producer, inner_read)
    assert _reaches(stree, producer, final_read)


def test_independent_chain_missing_an_element_reached():
    producer = _write("T[k, i]")
    inner_read = _read("T[k, i]", "out[k, i]")
    stree = _root(
        _loop("k", 0, 8, [_loop("i", 0, 4, [producer])]),
        _loop("k", 1, 9, [_loop("i", 1, 4, [_write("T[k, i]")]), _loop("i", 0, 4, [inner_read])]),
    )
    assert _reaches(stree, producer, inner_read)


def test_read_before_write():
    write, read = _write("A[0]"), _read("A[0]")
    assert not _reaches(_root(read, write), write, read)
    # In a loop, the value of the previous iteration is read, unless overwritten in between
    write, read = _write("A[0]"), _read("A[0]")
    assert _reaches(_root(_loop("t", 0, 4, [read, write])), write, read)
    write, read = _write("A[0]"), _read("A[0]")
    assert not _reaches(_root(_loop("t", 0, 4, [_write("A[0]"), read, write])), write, read)


def test_kills_after_the_write_belong_to_an_earlier_iteration():
    """``for k: R: x = A[k]; W: A[k + 1] = 1; X: A[k] = 2``: ``R`` reads in iteration ``k + 1`` what ``W`` wrote in
    iteration ``k``. ``X`` writes ``A[k]`` after ``W``, but in iteration ``k``, so it does not hide that value."""
    write, read = _write("A[k + 1]"), _read("A[k]")
    stree = _root(_loop("k", 0, 4, [read, write, _tasklet("o = 2", {}, {"o": "A[k]"})]))
    assert _reaches(stree, write, read)


def test_accumulation_reaches_itself_in_loops_only():
    accumulate = _tasklet("o = s + x", {"s": "S[0]", "x": "X[t]"}, {"o": "S[0]"})
    analysis = ReachingAnalysis(_root(_loop("t", 0, 4, [accumulate])))
    assert analysis.may_reach(accumulate, accumulate.out_memlets["o"], accumulate, accumulate.in_memlets["s"])
    accumulate = _tasklet("o = s + x", {"s": "S[0]", "x": "X[0]"}, {"o": "S[0]"})
    analysis = ReachingAnalysis(_root(accumulate))
    assert not analysis.may_reach(accumulate, accumulate.out_memlets["o"], accumulate, accumulate.in_memlets["s"])


def test_reach_in_later_executions():
    write, read = _write("A[i]"), _read("A[i]")
    inner = [_loop("i", 0, 8, [write]), _loop("i", 0, 8, [read])]
    outer = _loop("t", 0, 4, inner)
    analysis = ReachingAnalysis(_root(outer))
    # Every iteration of ``t`` writes what it reads before reading it
    assert not analysis.may_reach_later(write, write.out_memlets["o"], read, read.in_memlets["x"], outer)
    # Read first: the previous iteration's value
    write, read = _write("A[i]"), _read("A[i]")
    outer = _loop("t", 0, 4, [_loop("i", 0, 8, [read]), _loop("i", 0, 8, [write])])
    analysis = ReachingAnalysis(_root(outer))
    assert analysis.may_reach_later(write, write.out_memlets["o"], read, read.in_memlets["x"], outer)
    # A scope that runs once has no later execution
    write, read = _write("A[i]"), _read("A[i]")
    block = _loop("t", 0, 4, [_loop("i", 0, 8, [read]), _loop("i", 0, 8, [write])])
    once = tn.IfScope(condition=CodeBlock("c > 0"), children=[block])
    analysis = ReachingAnalysis(_root(once))
    assert not analysis.may_reach_later(write, write.out_memlets["o"], read, read.in_memlets["x"], once)


def test_condition_read():
    producer = _write("A[i]")
    guard = tn.IfScope(condition=CodeBlock("A[i] > 0"), children=[_write("B[i]")])
    stree = _root(_loop("i", 0, 8, [producer]), _loop("i", 0, 8, [guard]))
    analysis = ReachingAnalysis(stree)
    assert analysis.may_reach(producer, producer.out_memlets["o"], guard, dace.Memlet("A[i]"))
    # A write in the body of the ``if`` is read by the condition of a later iteration only
    inner = _write("A[i]")
    guard = tn.IfScope(condition=CodeBlock("A[0] > 0"), children=[inner])
    stree = _root(_loop("i", 0, 8, [guard]))
    assert ReachingAnalysis(stree).may_reach(inner, inner.out_memlets["o"], guard, dace.Memlet("A[0]"))
    inner = _write("A[i]")
    guard = tn.IfScope(condition=CodeBlock("A[0] > 0"), children=[inner])
    stree = _root(guard)
    assert not ReachingAnalysis(stree).may_reach(inner, inner.out_memlets["o"], guard, dace.Memlet("A[0]"))


def test_no_kills_with_goto():
    producer, consumer = _write("A[i]"), _read("A[i]")
    stree = _root(
        _loop("i", 0, 8, [producer]),
        _loop("i", 0, 8, [_write("A[i]")]),
        tn.GotoNode(target="end"),
        tn.StateLabel(state=None),
        _loop("i", 0, 8, [consumer]),
    )
    assert _reaches(stree, producer, consumer)


if __name__ == "__main__":
    test_must_writes_over_iterations()
    test_must_writes_exclude_gaps_and_possibly_empty_loops()
    test_must_writes_of_both_branches()
    test_covered_symbolic_bounds()
    test_overwritten_value_not_reached()
    test_conditionally_overwritten_value_reached()
    test_value_overwritten_in_both_branches_not_reached()
    test_partly_overwritten_value_reached()
    test_overwrite_in_possibly_empty_loop_does_not_hide()
    test_disjoint_elements_not_reached()
    test_independent_chain_in_shared_temporary()
    test_independent_chain_missing_an_element_reached()
    test_read_before_write()
    test_kills_after_the_write_belong_to_an_earlier_iteration()
    test_accumulation_reaches_itself_in_loops_only()
    test_reach_in_later_executions()
    test_condition_read()
    test_no_kills_with_goto()
