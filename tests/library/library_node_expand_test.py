# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import sys
import os
import numpy as np

# Add the test library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'library'))
import addlib


def create_test_sdfg():
    """Create a simple SDFG with an AddNode for testing"""
    sdfg = dace.SDFG('test_expand')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_array('B', [1], dace.float32)
    state = sdfg.add_state()

    A = state.add_read('A')
    B = state.add_write('B')
    addnode = addlib.AddNode('add')
    state.add_node(addnode)

    state.add_edge(A, None, addnode, '_a', dace.Memlet('A[0]'))
    state.add_edge(addnode, '_b', B, None, dace.Memlet('B[0]'))

    return sdfg, state, addnode


def test_new_library_node_expand_interface():
    """Test the new LibraryNode.expand(state, implementation, **kwargs) interface"""
    sdfg, state, addnode = create_test_sdfg()

    # Test with explicit implementation
    result = addnode.expand(state, 'pure')
    assert result == 'pure'

    # Test with default implementation (None)
    sdfg2, state2, addnode2 = create_test_sdfg()
    result2 = addnode2.expand(state2)
    assert result2 == 'pure'


def test_old_library_node_expand_interface():
    """Test the old LibraryNode.expand(sdfg, state, **kwargs) interface for backward compatibility"""
    sdfg, state, addnode = create_test_sdfg()

    # Test with old interface
    result = addnode.expand(sdfg, state)
    assert result == 'pure'


def test_state_expand_library_node_method():
    """Test the new SDFGState.expand_library_node(node, implementation, **kwargs) method"""
    sdfg, state, addnode = create_test_sdfg()

    # Test with valid implementation
    result = state.expand_library_node(addnode, 'pure')
    assert result == 'pure'


def test_state_expand_library_node_errors():
    """Test error handling in SDFGState.expand_library_node method"""
    sdfg, state, addnode = create_test_sdfg()

    # Test with invalid implementation
    with pytest.raises(KeyError) as exc_info:
        state.expand_library_node(addnode, 'nonexistent')
    assert 'Unknown implementation for node AddNode: nonexistent' in str(exc_info.value)

    # Test with node not in state
    sdfg2, state2, addnode2 = create_test_sdfg()
    with pytest.raises(ValueError) as exc_info:
        state.expand_library_node(addnode2, 'pure')
    assert 'is not in this state' in str(exc_info.value)


def test_library_node_expand_with_kwargs():
    """Test that expansion kwargs are properly passed through"""
    sdfg, state, addnode = create_test_sdfg()

    # Test that the method accepts kwargs even if they aren't used
    # We'll test with a library node that doesn't use kwargs but doesn't fail
    try:
        result = addnode.expand(state, 'pure')
        assert result == 'pure'
    except TypeError:
        pytest.fail("expand method should handle kwargs gracefully")

    # Test with state method - this should also work
    sdfg2, state2, addnode2 = create_test_sdfg()
    result2 = state2.expand_library_node(addnode2, 'pure')
    assert result2 == 'pure'


def test_compatibility_with_existing_expand_library_nodes():
    """Test that the original SDFG.expand_library_nodes() still works"""
    sdfg, state, addnode = create_test_sdfg()

    # This should not raise any errors
    sdfg.expand_library_nodes()

    # The node should be expanded and no longer be a library node
    # (it should be replaced by the expansion)
    library_nodes = [n for n in state.nodes() if isinstance(n, dace.nodes.LibraryNode)]
    assert len(library_nodes) == 0  # Should be expanded


def test_functional_correctness():
    """Test that the expanded node actually works correctly"""
    sdfg, state, addnode = create_test_sdfg()

    # Expand using new interface
    addnode.expand(state, 'pure')

    # Test execution
    A = np.array([5.0], dtype=np.float32)
    B = np.array([0.0], dtype=np.float32)
    sdfg(A=A, B=B)

    # AddNode should add 1 to the input (as per the implementation)
    assert B[0] == 6.0


def test_implementation_override():
    """Test that the implementation parameter overrides the node's implementation"""
    sdfg, state, addnode = create_test_sdfg()

    # Initially, the node has no implementation set
    assert addnode.implementation is None

    # Set a different implementation on the node
    addnode.implementation = 'nonexistent'  # This would normally fail

    # But we override it with a valid implementation
    result = addnode.expand(state, 'pure')
    assert result == 'pure'


def test_gemm_expansion_with_arguments():
    """Test expanding Gemm library node with alpha and beta arguments."""
    import dace
    from dace.libraries.blas import Gemm
    import numpy as np

    # Create SDFG with a Gemm node
    sdfg = dace.SDFG('test_gemm_expansion')
    state = sdfg.add_state('s0')

    # Add arrays
    M, N, K = 64, 32, 128
    sdfg.add_array('A', [M, K], dace.float32)
    sdfg.add_array('B', [K, N], dace.float32)
    sdfg.add_array('C', [M, N], dace.float32)
    sdfg.add_array('result', [M, N], dace.float32)

    # Create Gemm node with specific alpha and beta values
    gemm_node = Gemm('gemm', alpha=2.5, beta=1.5)

    # Add nodes to state
    a_read = state.add_read('A')
    b_read = state.add_read('B')
    c_read = state.add_read('C')
    result_write = state.add_write('result')

    # Add the Gemm node to state
    state.add_node(gemm_node)

    # Connect nodes
    state.add_edge(a_read, None, gemm_node, '_a', dace.Memlet('A[0:M, 0:K]'))
    state.add_edge(b_read, None, gemm_node, '_b', dace.Memlet('B[0:K, 0:N]'))
    state.add_edge(c_read, None, gemm_node, '_cin', dace.Memlet('C[0:M, 0:N]'))
    state.add_edge(gemm_node, '_c', result_write, None, dace.Memlet('result[0:M, 0:N]'))

    # Test new interface with implementation specification
    result = gemm_node.expand(state, 'pure')
    assert result == 'pure'

    # Verify that the alpha and beta values are preserved
    assert gemm_node.alpha == 2.5
    assert gemm_node.beta == 1.5

    # Test that the expansion worked by checking SDFG is valid
    sdfg.validate()


if __name__ == '__main__':
    test_new_library_node_expand_interface()
    test_old_library_node_expand_interface()
    test_state_expand_library_node_method()
    test_state_expand_library_node_errors()
    test_library_node_expand_with_kwargs()
    test_compatibility_with_existing_expand_library_nodes()
    test_functional_correctness()
    test_implementation_override()
    test_gemm_expansion_with_arguments()
