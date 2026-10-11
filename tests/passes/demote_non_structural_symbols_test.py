# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Only iteration domains and data accesses keep their symbols; every other interstate symbol becomes a scalar."""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.dataflow import AugAssignToWCR
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.demote_non_structural_symbols import DemoteNonStructuralSymbols


def counting_sdfg() -> dace.SDFG:
    """ICON's ``clip_count``: ``n = N; k = 0; for i < n: if A[i] > 0.5: k = k + 1``, then ``C[0] = k``."""
    sdfg = dace.SDFG("count_above_half")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_array("A", ["N"], dace.float64)
    sdfg.add_array("C", [1], dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    loop = LoopRegion("loop", "i < n", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(start, loop, dace.InterstateEdge(assignments={"n": "N", "k": "0"}))
    cond = ConditionalBlock("above")
    branch = ControlFlowRegion("count")
    first = branch.add_state(is_start_block=True)
    branch.add_state_after(first, assignments={"k": "k + 1"})
    cond.add_branch(dace.properties.CodeBlock("A[i] > 0.5"), branch)
    loop.add_node(cond, is_start_block=True)
    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    write = end.add_tasklet("store", {}, {"o"}, "o = k")
    end.add_edge(write, "o", end.add_write("C"), None, dace.Memlet("C[0]"))
    return sdfg


def test_a_counter_becomes_a_scalar_and_its_loop_maps():
    """``k`` shapes nothing, so it becomes a scalar; ``n`` bounds the loop and stays a symbol."""
    sdfg = counting_sdfg()
    sdfg.validate()
    assert DemoteNonStructuralSymbols().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "k" in sdfg.arrays and "k" not in sdfg.symbols
    assert "n" in sdfg.symbols and "n" not in sdfg.arrays

    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 1
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1

    A = np.arange(32, dtype=np.float64) / 31.0
    C = np.zeros(1, dtype=np.int64)
    sdfg(A=A, C=C, N=32)
    assert C[0] == np.count_nonzero(A > 0.5)


def test_a_demoted_value_keeps_its_arithmetic_as_written():
    """CloudSC's ``zrhc = ramid + (1 - ramid) * ((s - 0.8) / 0.2) ** 2`` must round exactly as the interstate edge
    did: a sympy round-trip distributes the divisor (``5.0 * s - 4.0``) and changes the last bits."""
    sdfg = dace.SDFG("demoted_value_as_written")
    sdfg.add_symbol("z", dace.float64)
    sdfg.add_array("S", [1], dace.float64)
    sdfg.add_array("R", [1], dace.float64)
    start = sdfg.add_state("start", is_start_block=True)
    use = sdfg.add_state_after(start, "use", assignments={"z": "0.3 + (1.0 - 0.3) * ((S[0] - 0.8) / 0.2) ** 2"})
    write = use.add_tasklet("store", {}, {"o"}, "o = z")
    use.add_edge(write, "o", use.add_write("R"), None, dace.Memlet("R[0]"))
    sdfg.validate()
    assert DemoteNonStructuralSymbols().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    assign = next(
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and n.label == "assign_z"
    )
    assert "/ 0.2" in assign.code.as_string, assign.code.as_string

    S = np.array([0.9137], dtype=np.float64)
    R = np.zeros(1)
    sdfg(S=S, R=R)
    assert R[0] == 0.3 + (1.0 - 0.3) * ((S[0] - 0.8) / 0.2) ** 2


def test_a_tasklet_reading_a_demoted_symbol_keeps_its_expression():
    """The read of a demoted symbol becomes a connector without reordering the tasklet: ``x - z + y`` stays in that
    order (sympy writes ``x + y - z``, which rounds differently)."""
    sdfg = dace.SDFG("tasklet_keeps_expression")
    sdfg.add_symbol("z", dace.float64)
    for name in ("S", "X", "Y", "R"):
        sdfg.add_array(name, [1], dace.float64)
    start = sdfg.add_state("start", is_start_block=True)
    use = sdfg.add_state_after(start, "use", assignments={"z": "S[0]"})
    tasklet = use.add_tasklet("combine", {"x", "y"}, {"o"}, "o = x - z + y")
    use.add_edge(use.add_read("X"), None, tasklet, "x", dace.Memlet("X[0]"))
    use.add_edge(use.add_read("Y"), None, tasklet, "y", dace.Memlet("Y[0]"))
    use.add_edge(tasklet, "o", use.add_write("R"), None, dace.Memlet("R[0]"))
    sdfg.validate()
    assert DemoteNonStructuralSymbols().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "(x - _in_z) + y" in tasklet.code.as_string, tasklet.code.as_string

    S, X, Y, R = np.array([1e16]), np.array([1.0]), np.array([-1e16]), np.zeros(1)
    sdfg(S=S, X=X, Y=Y, R=R)
    assert R[0] == X[0] - S[0] + Y[0]


def test_an_index_read_inside_a_branch_condition_stays_a_symbol():
    """``j = I[0]`` then ``if A[j] > 0.5``: ``j`` selects the element the condition reads, so it is a data-access
    symbol, while ``w = A[j]`` (a value only compared) becomes a scalar."""
    sdfg = dace.SDFG("index_in_condition")
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_symbol("w", dace.float64)
    sdfg.add_array("I", [1], dace.int64)
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("R", [1], dace.float64)
    start = sdfg.add_state("start", is_start_block=True)
    read = sdfg.add_state_after(start, "read", assignments={"j": "I[0]"})
    cond = ConditionalBlock("above")
    branch = ControlFlowRegion("store")
    body = branch.add_state(is_start_block=True)
    write = body.add_tasklet("store", {}, {"o"}, "o = w")
    body.add_edge(write, "o", body.add_write("R"), None, dace.Memlet("R[0]"))
    cond.add_branch(dace.properties.CodeBlock("A[j] > 0.5"), branch)
    sdfg.add_node(cond)
    sdfg.add_edge(read, cond, dace.InterstateEdge(assignments={"w": "A[j]"}))
    sdfg.validate()
    assert DemoteNonStructuralSymbols().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "j" in sdfg.symbols and "w" in sdfg.arrays

    I, A, R = np.array([2], dtype=np.int64), np.array([0.0, 0.1, 0.9, 0.2]), np.zeros(1)
    sdfg(I=I, A=A, R=R)
    assert R[0] == 0.9


if __name__ == "__main__":
    test_a_counter_becomes_a_scalar_and_its_loop_maps()
    test_a_demoted_value_keeps_its_arithmetic_as_written()
    test_a_tasklet_reading_a_demoted_symbol_keeps_its_expression()
    test_an_index_read_inside_a_branch_condition_stays_a_symbol()
