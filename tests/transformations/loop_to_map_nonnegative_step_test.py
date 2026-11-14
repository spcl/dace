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


if __name__ == "__main__":
    test_a_nonnegative_symbolic_step_lifts_to_an_ascending_map()
