# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate import LoopNormalize
from dace.transformation.passes import SymbolPropagation


def _count_loops(sdfg: dace.SDFG):
    loops = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion):
            loops += 1
    return loops


def test_loop_carried_symbol():
    """
    Tests SymbolPropagation respects loop carried dependencies.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_symbol("LB", dace.int32)
    sdfg.add_symbol("UB", dace.int32)
    sdfg.add_symbol("idx", dace.int32)
    sdfg.add_symbol("cnt", dace.int32)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < UB", "i", "i = LB", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"cnt": "0"}))

    s = loop.add_state(is_start_block=True)
    s1 = loop.add_state()
    s2 = loop.add_state()
    e = loop.add_state()
    loop.add_edge(s, s1, dace.InterstateEdge(assignments={"a": "i", "c": "cnt + 1"}))
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={"b": "a+1"}))
    loop.add_edge(
        s2, e, dace.InterstateEdge(assignments={"idx": "b - 1 - LB", "cnt": "c"})
    )
    task = e.add_tasklet("init", {}, {"out"}, "out = 0")
    access = e.add_access("A")
    e.add_edge(task, "out", access, None, dace.Memlet("A[idx]"))

    e = sdfg.add_state()
    sdfg.add_edge(loop, e, dace.InterstateEdge(assignments={"cnt": "cnt+1"}))
    sdfg.validate()

    # Count loops before transformation
    assert _count_loops(sdfg) == 1

    # Apply LoopNormalize, SymbolPropagation, and LoopToMap
    sdfg.apply_transformations_repeated(LoopNormalize)
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()

    # Should not have transformed the loop if the loop-carried dependency is respected as the
    assert _count_loops(sdfg) == 1

    # Validate correctness
    A = dace.ndarray([64], dtype=dace.float32)
    A[:] = np.random.rand(64).astype(dace.float32.type)
    sdfg(A=A, LB=0, UB=64)

    assert np.allclose(A[:], 0)


def test_nested_symbol():
    """
    Tests that SymbolPropagation does not overwrite nested symbols.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("v", dace.int32)

    s = sdfg.add_state(is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    edge1 = sdfg.add_edge(s, cond, dace.InterstateEdge(assignments={"v": "0"}))
    b1 = ControlFlowRegion("b1", sdfg)
    b1s = b1.add_state()
    b1e = b1.add_state()
    edge2 = b1.add_edge(b1s, b1e, dace.InterstateEdge(assignments={"v": "5"}))
    cond.add_branch(CodeBlock("v == 0"), b1)

    b2 = ControlFlowRegion("b2", sdfg)
    b2s = b2.add_state()
    b2e = b2.add_state()
    edge3 = b2.add_edge(b2s, b2e, dace.InterstateEdge(assignments={"v": "8"}))
    cond.add_branch(CodeBlock("v == 3"), b2)

    e = sdfg.add_state()
    edge4 = sdfg.add_edge(cond, e, dace.InterstateEdge(assignments={"v": "v+1"}))
    sdfg.validate()

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # No assignment should have been changed
    assert edge1.data.assignments["v"] == "0"
    assert edge2.data.assignments["v"] == "5"
    assert edge3.data.assignments["v"] == "8"
    assert edge4.data.assignments["v"] == "v+1"


def test_multiple_sources():
    """
    Tests that SymbolPropagation handles multiple sources correctly.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("v", dace.int32)
    sdfg.add_symbol("a", dace.int32)

    s1 = sdfg.add_state(is_start_block=True)
    s2 = sdfg.add_state()
    s3 = sdfg.add_state()
    c = sdfg.add_state()
    e = sdfg.add_state()

    edge1 = sdfg.add_edge(s1, c, dace.InterstateEdge(assignments={"v": "0"}))
    edge2 = sdfg.add_edge(s2, c, dace.InterstateEdge(assignments={"a": "5"}))
    edge3 = sdfg.add_edge(s3, c, dace.InterstateEdge(assignments={"v": "8"}))
    edge4 = sdfg.add_edge(c, e, dace.InterstateEdge(assignments={"v": "v+a"}))

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # No assignment should have been changed
    assert edge1.data.assignments["v"] == "0"
    assert edge2.data.assignments["a"] == "5"
    assert edge3.data.assignments["v"] == "8"
    assert edge4.data.assignments["v"] == "v+a"


if __name__ == "__main__":
    test_loop_carried_symbol()
    test_nested_symbol()
    test_multiple_sources()
