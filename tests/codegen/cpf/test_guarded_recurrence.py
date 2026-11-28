# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A recurrence whose read sits under a branch is proven when the branch can hold at the iterations that conflict.

CloudSC's level loop reads the level above only under ``IF (JK > NCLDTOP)`` or a flag of the input, so a witness that
used only accesses outside every branch left it ``unknown``. The solver now takes each branch condition at its own
iteration, an input array the condition reads as values it may choose, and names those values in the example.
"""

import re

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import (
    PROVEN_PREFIX,
    AnnotateLoopKinds,
    unseen_writes,
    witnessed_carrying_access,
)

N = dace.symbol("N")


@dace.program
def level_guard(x: dace.float64[N], y: dace.float64[N]):
    for k in range(N):
        if k > 0:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def data_guard(x: dace.float64[N], y: dace.float64[N], flag: dace.int32[N]):
    for k in range(N):
        if flag[k]:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def dead_guard(x: dace.float64[N], y: dace.float64[N]):
    for k in range(N):
        if k < 0:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


@dace.program
def below_top(x: dace.float64[N], y: dace.float64[N], top: dace.int64[1]):
    first = top[0]
    for k in range(first, N):
        if k > first:
            y[k] = x[k - 1]
        x[k] = y[k] * 2.0


def class_line(sdfg: dace.SDFG) -> str:
    AnnotateLoopKinds().apply_pass(sdfg, {})
    (loop,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return loop.specialization_hint.splitlines()[0]


def test_a_read_under_a_counter_guard_proves_the_recurrence():
    line = class_line(level_guard.to_sdfg(simplify=True))
    assert line.startswith(PROVEN_PREFIX), line
    assert line.endswith("(example the solver found: N=2, iterations k=0 and k=1)"), line


def test_a_read_under_an_input_flag_proves_the_recurrence_naming_the_flag():
    line = class_line(data_guard.to_sdfg(simplify=True))
    assert line.startswith(PROVEN_PREFIX), line
    assert "flag[1]=1, iterations k=0 and k=1)" in line, line


def test_a_read_under_a_guard_false_where_it_meets_the_write_proves_nothing():
    # Unsimplified, so the dead branch stays for the witness search to refuse
    sdfg = dead_guard.to_sdfg(simplify=False)
    (loop,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert witnessed_carrying_access(loop) is None
    assert not class_line(sdfg).startswith(PROVEN_PREFIX)


def test_a_level_loop_from_an_input_top_read_below_the_top_proves_the_recurrence():
    """CloudSC's level loop starts at ``ncldtop`` and reads the level above only below it. A top that leaves one level
    orders nothing, so the witness needs only every top under which the loop runs at least twice."""
    line = class_line(below_top.to_sdfg(simplify=True))
    assert line.startswith(PROVEN_PREFIX), line
    # The top bounds the loop, so the example names the value it rests on
    example = re.search(
        r"\(example the solver found: N=(\d+), first=(-?\d+), iterations k=(-?\d+) and k=(-?\d+)\)", line
    )
    assert example, line
    size, first, earlier, later = (int(g) for g in example.groups())
    assert first <= earlier and later == earlier + 1 < size, line


def test_a_write_in_a_loop_without_a_constant_stride_is_unseen():
    """minife and qe_vexx_k crashed here: the loop's states were read through a method regions do not have."""
    sdfg = dace.SDFG("doubling")
    sdfg.add_array("y", [N], dace.float64)
    loop = LoopRegion("doubling", "j < N", "j", "j = 1", "j = j * 2")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("one", {}, {"out"}, "out = 1.0")
    body.add_edge(tasklet, "out", body.add_write("y"), None, dace.Memlet("y[j]"))
    assert "y" in unseen_writes(sdfg)


if __name__ == "__main__":
    test_a_read_under_a_counter_guard_proves_the_recurrence()
    test_a_read_under_an_input_flag_proves_the_recurrence_naming_the_flag()
    test_a_read_under_a_guard_false_where_it_meets_the_write_proves_nothing()
    test_a_level_loop_from_an_input_top_read_below_the_top_proves_the_recurrence()
    test_a_write_in_a_loop_without_a_constant_stride_is_unseen()
