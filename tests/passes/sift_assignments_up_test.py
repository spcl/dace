# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SiftAssignmentsUp``: an assignment moves up through the states that neither read its symbol nor write what its
value reads, and stops at the first that does."""

import numpy as np

import dace
from dace.transformation.passes.sift_assignments_up import SiftAssignmentsUp


def chain_with_a_late_assignment(reader_writes_bounds: bool) -> dace.SDFG:
    """``init -> a -> b -> c``; the edge ``b -> c`` assigns ``k = bounds[0]`` and ``c`` reads ``k``. ``a`` writes
    ``bounds`` when ``reader_writes_bounds``."""
    sdfg = dace.SDFG(f"sift_{int(reader_writes_bounds)}")
    sdfg.add_array("bounds", [1], dace.int64)
    sdfg.add_array("out", [1], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    a, b, c = sdfg.add_state("a"), sdfg.add_state("b"), sdfg.add_state("c")
    sdfg.add_edge(init, a, dace.InterstateEdge())
    sdfg.add_edge(a, b, dace.InterstateEdge())
    sdfg.add_edge(b, c, dace.InterstateEdge(assignments={"k": "bounds[0]"}))
    if reader_writes_bounds:
        tasklet = a.add_tasklet("bump", {}, {"o"}, "o = 7")
        a.add_edge(tasklet, "o", a.add_write("bounds"), None, dace.Memlet("bounds[0]"))
    tasklet = c.add_tasklet("use", {}, {"o"}, "o = k")
    c.add_edge(tasklet, "o", c.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def assigned_on(sdfg: dace.SDFG, src: str) -> dict[str, str]:
    edge = next(e for e in sdfg.edges() if e.src.label == src)
    return {k: str(v) for k, v in edge.data.assignments.items()}


def test_an_assignment_moves_to_the_head_of_the_chain():
    sdfg = chain_with_a_late_assignment(reader_writes_bounds=False)
    assert SiftAssignmentsUp().apply_pass(sdfg, {}) == 2
    sdfg.validate()
    assert "k" in assigned_on(sdfg, "init") and not assigned_on(sdfg, "b")


def test_an_assignment_stops_below_a_state_writing_what_it_reads():
    sdfg = chain_with_a_late_assignment(reader_writes_bounds=True)
    assert SiftAssignmentsUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "k" in assigned_on(sdfg, "a") and not assigned_on(sdfg, "init")

    bounds, out = np.array([3], dtype=np.int64), np.zeros(1, dtype=np.int64)
    sdfg(bounds=bounds, out=out)
    assert out[0] == 7


if __name__ == "__main__":
    test_an_assignment_moves_to_the_head_of_the_chain()
    test_an_assignment_stops_below_a_state_writing_what_it_reads()
