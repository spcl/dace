"""
Unit tests for SSALoopIterators pass.
"""
import dace
import numpy as np
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.ssa_loop_iterators import SSALoopIterators
from dace.transformation.passes.analysis import loop_analysis


@dace.program
def foo(A: dace.float64[10, 10], idx: dace.int32[10, 10], B: dace.float64[5, 10]):
    for i in range(5):
        for j, k in dace.map[0:10, 0:10]:
            A[j, k] = 1.1 * A[j, k] + 1.2 * B[i, idx[j, k]]


def test_nested_sdfg_symbol_mapping():
    """
    The map inside the loop body becomes a nested SDFG.
    The loop variable `i` must appear in the nested SDFG's symbol_mapping.
    After SSALoopIterators, the symbol_mapping should reference the new
    SSA name (_it_N), not the original `i`.
    """
    SSALoopIterators.loop_var_counter = 0

    sdfg = foo.to_sdfg(simplify=False)

    # Before: confirm `i` is the loop variable and appears in a nested SDFG mapping
    loops_before = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops_before) == 1
    assert loops_before[0].loop_variable == 'i'

    found_i_in_mapping = False
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if 'i' in node.symbol_mapping:
                    found_i_in_mapping = True
    assert found_i_in_mapping, "Expected 'i' in nested SDFG symbol_mapping before pass"

    # Apply pass
    SSALoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # After: the nested SDFG symbol_mapping should have _it_0, not i
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                assert 'i' not in node.symbol_mapping, \
                    f"Original loop var 'i' should not be in symbol_mapping, got {node.symbol_mapping}"
                assert '_it_0' in node.symbol_mapping, \
                    f"SSA loop var '_it_0' should be in symbol_mapping, got {node.symbol_mapping}"

    # Verify correctness
    A = np.random.rand(10, 10)
    idx = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    B = np.random.rand(5, 10)

    A_ref = A.copy()
    for i in range(5):
        for j in range(10):
            for k in range(10):
                A_ref[j, k] = 1.1 * A_ref[j, k] + 1.2 * B[i, idx[j, k]]

    csdfg = sdfg.compile()
    csdfg(A=A, idx=idx, B=B)
    assert np.allclose(A, A_ref), f"Max error: {np.max(np.abs(A - A_ref))}"


# ============================================================================
# Test 2: Loop variable used after the loop (reconstruction check)
# ============================================================================
@dace.program
def loop_var_used_after(A: dace.float64[10], B: dace.float64[10]):
    for i in range(10):
        A[i] = 2.0 * B[i]
    # After the loop, i should be 9. The pass should insert
    # an assignment i = loop_end - 1 so downstream usage is correct.


def test_loop_var_reconstruction():
    """
    After SSALoopIterators, a reconstruction state should assign
    the original loop variable to (loop_end - 1) so that any
    downstream use of the variable sees the correct final value.
    """
    SSALoopIterators.loop_var_counter = 0

    sdfg = loop_var_used_after.to_sdfg(simplify=False)

    SSALoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # Check that a reconstruction state was added
    reconstruction_states = [
        s for s in sdfg.all_states() if hasattr(s, 'label') and 'SSA_loop_var_reconstruction' in s.label
    ]
    assert len(reconstruction_states) == 1, f"Expected 1 reconstruction state, found {len(reconstruction_states)}"

    # Check that assignment is correct
    loops = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops) == 1
    loop = loops[0]

    out_edges = loop.parent_graph.out_edges(loop)
    assert len(out_edges) == 1

    assignments = out_edges[0].data.assignments
    assert 'i' in assignments, f"Expected assignment to 'i', got {assignments}"
    assert str(
        assignments['i']
    ) == f"({(str(loop_analysis.get_loop_end(loop)))})", f"Expected loop end assignment, got {assignments['i']}"

    # Verify correctness
    A = np.zeros(10)
    B = np.random.rand(10)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B)
    assert np.allclose(A, 2.0 * B)


# ============================================================================
# Test 3: Nested loops — both variables should be renamed independently
# ============================================================================
@dace.program
def nested_loops(A: dace.float64[8, 8]):
    for i in range(8):
        for j in range(8):
            A[i, j] = A[i, j] + 1.0


def test_nested_loops():
    """
    Two nested LoopRegions with variables i and j.
    Both should be renamed to distinct SSA names (_it_0, _it_1),
    and both should get reconstruction states.
    """
    SSALoopIterators.loop_var_counter = 0

    sdfg = nested_loops.to_sdfg(simplify=False)

    # Before: should have 2 loop regions
    loops_before = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops_before) == 2
    loop_vars_before = {l.loop_variable for l in loops_before}
    assert loop_vars_before == {'i', 'j'}

    SSALoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # Verify correctness
    A = np.random.rand(8, 8)
    A_ref = A.copy() + 1.0
    csdfg = sdfg.compile()
    csdfg(A=A)
    assert np.allclose(A, A_ref), f"Max error: {np.max(np.abs(A - A_ref))}"


if __name__ == '__main__':
    test_nested_sdfg_symbol_mapping()
    test_loop_var_reconstruction()
    test_nested_loops()
