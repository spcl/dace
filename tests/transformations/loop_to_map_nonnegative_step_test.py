# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose symbolic step is only known nonnegative still runs upwards: a zero step never terminates (and Python's
``range`` refuses it), so ``n3 >= 0`` orients the map as surely as ``n3 > 0`` (TSVC ``s122``)."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.symbolic import Predicate
from dace.transformation.interstate.loop_to_map import LoopToMap


def strided_increment() -> dace.SDFG:
    sdfg = dace.SDFG("strided_increment")
    sdfg.add_symbol("N", dace.int64, predicates=frozenset({Predicate.NONNEGATIVE}))
    sdfg.add_symbol("n3", dace.int64, predicates=frozenset({Predicate.NONNEGATIVE}))
    sdfg.add_array("a", ["N"], dace.float64)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + n3")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("inc", {"x"}, {"y"}, "y = x + 1.0")
    body.add_edge(body.add_read("a"), None, tasklet, "x", dace.Memlet("a[i]"))
    body.add_edge(tasklet, "y", body.add_write("a"), None, dace.Memlet("a[i]"))
    return sdfg


def test_a_nonnegative_symbolic_step_lifts_to_an_ascending_map():
    sdfg = strided_increment()
    assert sdfg.apply_transformations(LoopToMap) == 1
    a = np.zeros(10)
    sdfg(a=a, N=10, n3=3)
    assert np.array_equal(a, np.array([1.0 if i % 3 == 0 else 0.0 for i in range(10)]))


def test_a_write_in_a_sibling_loop_is_not_judged_by_the_facts_of_the_read():
    """``i = 2`` writes ``A[15, 2]`` at ``j = 13``, which ``i = 0`` reads: the facts at the read bound ``j`` by 9."""
    sdfg = dace.SDFG("two_inner_loops")
    sdfg.add_array("A", [100, 10], dace.float64)
    sdfg.add_array("B", [10, 10], dace.float64)
    loop = LoopRegion("outer", "i < 5", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = LoopRegion("first", "j < 10", "j", "j = 0", "j = j + 1")
    second = LoopRegion("second", "j < 26", "j", "j = 10", "j = j + 1")
    loop.add_node(first, is_start_block=True)
    loop.add_node(second)
    loop.add_edge(first, second, dace.InterstateEdge())
    read = first.add_state("read", is_start_block=True)
    tasklet = read.add_tasklet("copy", {"a"}, {"b"}, "b = a")
    read.add_edge(read.add_read("A"), None, tasklet, "a", dace.Memlet("A[i + 15, 2]"))
    read.add_edge(tasklet, "b", read.add_write("B"), None, dace.Memlet("B[i, j]"))
    write = second.add_state("write", is_start_block=True)
    fill = write.add_tasklet("fill", {}, {"o"}, "o = 1.0")
    write.add_edge(fill, "o", write.add_write("A"), None, dace.Memlet("A[i + j, i]"))
    transformation = LoopToMap()
    transformation.loop = loop
    assert not transformation.can_be_applied(sdfg, 0, sdfg, False)


if __name__ == "__main__":
    test_a_nonnegative_symbolic_step_lifts_to_an_ascending_map()
    test_a_write_in_a_sibling_loop_is_not_judged_by_the_facts_of_the_read()
