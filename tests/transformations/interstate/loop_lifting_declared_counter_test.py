# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A state machine may count with a declared symbol; once lifted, the loop binds it, and an iterator is never a
declared symbol. Left declared, the lifted SDFG no longer validated ("bound by a loop or map scope")."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_lifting import LoopLifting


def counting_state_machine() -> dace.SDFG:
    sdfg = dace.SDFG("counting_state_machine")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_array("A", [10], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    guard, body, done = sdfg.add_state("guard"), sdfg.add_state("body"), sdfg.add_state("done")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i < 10"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"i": "i + 1"}))
    sdfg.add_edge(guard, done, dace.InterstateEdge(condition="i >= 10"))
    tasklet = body.add_tasklet("set", {}, {"out"}, "out = i")
    body.add_edge(tasklet, "out", body.add_write("A"), None, dace.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


def test_a_lifted_counter_is_no_longer_declared():
    sdfg = counting_state_machine()
    assert sdfg.apply_transformations_repeated(LoopLifting) == 1
    assert any(isinstance(block, LoopRegion) and block.loop_variable == "i" for block in sdfg.nodes())
    assert "i" not in sdfg.symbols
    sdfg.validate()
    a = np.zeros(10)
    sdfg(A=a)
    assert np.array_equal(a, np.arange(10, dtype=np.float64))


if __name__ == "__main__":
    test_a_lifted_counter_is_no_longer_declared()
