# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``symbols_defined_at`` types an assignment that reads a symbol an earlier interstate edge defines."""

import dace
from dace.sdfg.sdfg import InterstateEdge


def chained_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.Tasklet]:
    """``start = A[i]`` on one edge, ``cursor = start`` on the next, read by a tasklet after both:
    the shape of a while loop whose cursor starts at an index read from an array (gromacs_nbnxm)."""
    sdfg = dace.SDFG("chained")
    sdfg.add_array("A", [8], dace.int32)
    sdfg.add_array("B", [1], dace.int32)
    first = sdfg.add_state("first", is_start_block=True)
    middle = sdfg.add_state("middle")
    last = sdfg.add_state("last")
    sdfg.add_edge(first, middle, InterstateEdge(assignments={"start": "A[1]"}))
    sdfg.add_edge(middle, last, InterstateEdge(assignments={"cursor": "start"}))
    tasklet = last.add_tasklet("use", {}, {"out"}, "out = cursor")
    last.add_edge(tasklet, "out", last.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, last, tasklet


def test_an_assignment_reading_an_earlier_edges_symbol_is_typed():
    sdfg, state, tasklet = chained_sdfg()
    defined = state.symbols_defined_at(tasklet)
    assert defined["start"] == dace.int32
    assert defined["cursor"] == dace.int32


def test_the_chained_sdfg_compiles_and_runs():
    sdfg, _, _ = chained_sdfg()
    A = dace.ndarray([8], dace.int32)
    A[:] = list(range(10, 18))
    B = dace.ndarray([1], dace.int32)
    B[0] = 0
    sdfg(A=A, B=B)
    assert B[0] == 11


if __name__ == "__main__":
    test_an_assignment_reading_an_earlier_edges_symbol_is_typed()
    test_the_chained_sdfg_compiles_and_runs()
