# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate import LoopNormalize
from dace.transformation.passes import SymbolPropagation
from dace.transformation.passes import ArrayElimination
from dace.transformation.passes import analysis as ap


def _count_loops(sdfg):
    loops = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion):
            loops += 1
    return loops


def test_l2m_pipeline():
    """
    Tests that the Loop2Map pipeline (LoopNormalize -> SymbolPropagation) enables additional Loop2Map transformations.
    """
    sdfg = dace.SDFG("l2m_pipeline")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_symbol("LB", dace.int32)
    sdfg.add_symbol("UB", dace.int32)
    sdfg.add_symbol("idx", dace.int32)

    loop = LoopRegion("loop", "i < UB", "i", "i = LB", "i = i + 1")
    sdfg.add_node(loop)
    s = loop.add_state(is_start_block=True)
    e = loop.add_state()
    loop.add_edge(s, e, dace.InterstateEdge(assignments={"idx": "i - LB"}))
    task = e.add_tasklet("init", {}, {"out"}, "out = 0")
    access = e.add_access("A")
    e.add_edge(task, "out", access, None, dace.Memlet("A[idx]"))

    sdfg.validate()

    # Count loops before transformation
    assert _count_loops(sdfg) == 1

    # Apply Loop2Map directly
    sdfg.apply_transformations_repeated(LoopToMap)

    # Should not have changed
    assert _count_loops(sdfg) == 1

    # Apply LoopNormalize and SymbolPropagation
    sdfg.apply_transformations_repeated(LoopNormalize)
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)

    # Should have transformed the loop
    assert _count_loops(sdfg) == 0

    # Validate correctness
    A = dace.ndarray([64], dtype=dace.float32)
    A[:] = np.random.rand(64).astype(dace.float32.type)
    sdfg(A=A, LB=0, UB=64)

    assert np.allclose(A[:], 0)


def test_l2m_views():
    """
    Tests that the Loop2Map works correctly with views.
    """
    sdfg = dace.SDFG("l2m_pipeline")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)
    sdfg.add_array("B", [64], dace.float32)

    loop = LoopRegion("loop", "i < 64", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    s = loop.add_state(is_start_block=True)
    access_A = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    access_B = s.add_access("B")
    s.add_edge(access_A, None, access_view, "views", dace.Memlet("A[0]"))
    s.add_edge(access_view, None, access_B, None, dace.Memlet("B[i]"))

    sdfg.validate()

    # Try to apply Loop2Map directly
    try:
        sdfg.apply_transformations_repeated(LoopToMap)
        sdfg.validate()
    except Exception as e:
        assert False, f"LoopToMap failed: {e}"

    # Validate correctness
    A = dace.ndarray([64], dtype=dace.float32)
    A[:] = np.random.rand(64).astype(dace.float32.type)
    B = dace.ndarray([64], dtype=dace.float32)
    B[:] = np.random.rand(64).astype(dace.float32.type)
    sdfg(A=A, B=B)

    assert np.allclose(B[:], A[0])

def test_array_elimination_nested_edge_view():
    """
    Tests if the ArrayElimination pass works correctly with nested eges using views.
    """
    sdfg = dace.SDFG("l2m_pipeline")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)

    loop = LoopRegion("loop1", "i < 64", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    s = loop.add_state(is_start_block=True)
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))
    s2 = loop.add_state()
    loop.add_edge(s, s2, dace.InterstateEdge(assignments={"v": "A_view[0]"}))

    s = sdfg.add_state()
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))
    sdfg.add_edge(loop, s, dace.InterstateEdge())

    sdfg.validate()
    prev_nodes = list(sdfg.all_nodes_recursive())

    # Apply ArrayElimination
    try:
        res1 = ap.StateReachability().apply_pass(sdfg, {})
        res2 = ap.FindAccessStates().apply_pass(sdfg, {})
        # combine both dicts
        pipe_res = {
            ap.StateReachability.__name__: res1,
            ap.FindAccessStates.__name__: res2,
        }
        ArrayElimination().apply_pass(sdfg, pipe_res)
    except Exception as e:
        assert False, f"ArrayElimination failed: {e}"

    # Should not remove anything
    assert len(list(sdfg.all_nodes_recursive())) == len(prev_nodes)


def test_array_elimination_view():
    """
    Tests if the ArrayElimination pass works correctly with views.
    """
    sdfg = dace.SDFG("l2m_pipeline")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_view("A_view", [1], dace.float32)

    s = sdfg.add_state()
    access = s.add_access("A")
    access_view = s.add_access("A_view")
    access_view.add_in_connector("views")
    s.add_edge(access, None, access_view, "views", dace.Memlet("A[0]"))

    sdfg.validate()

    # Apply ArrayElimination
    try:
        res1 = ap.StateReachability().apply_pass(sdfg, {})
        res2 = ap.FindAccessStates().apply_pass(sdfg, {})
        # combine both dicts
        pipe_res = {
            ap.StateReachability.__name__: res1,
            ap.FindAccessStates.__name__: res2,
        }
        ArrayElimination().apply_pass(sdfg, pipe_res)
        sdfg.validate()
    except Exception as e:
        assert False, f"ArrayElimination failed: {e}"

    # Should remove everything
    assert len(list(sdfg.all_nodes_recursive())) == 0


if __name__ == "__main__":
    test_l2m_pipeline()
    test_l2m_views()

    test_array_elimination_nested_edge_view()
    test_array_elimination_view()
