# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import sys
import os
import numpy as np
import copy
from dace import SDFGState, SDFG, library, dtypes
from dace.transformation.transformation import ExpandTransformation

# Add the test library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'library'))
import addlib


# Create a tiled expansion for testing keyword arguments
@library.register_expansion(addlib.AddNode, 'tiled')
class ExpandAddTiled(ExpandTransformation):
    """Test expansion that accepts tile_size keyword argument"""
    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG, tile_size=8, **kwargs):
        in_edge = parent_state.in_edges(node)[0]
        out_edge = parent_state.out_edges(node)[0]

        sdfg = dace.SDFG("nested_tiled")
        sdfg.add_datadesc("_a", copy.deepcopy(parent_sdfg.arrays[in_edge.data.data]))
        sdfg.add_datadesc("_b", copy.deepcopy(parent_sdfg.arrays[out_edge.data.data]))
        sdfg.arrays["_a"].transient = False
        sdfg.arrays["_b"].transient = False
        state = sdfg.add_state()

        inp = state.add_access("_a")
        outp = state.add_access("_b")

        # Create a map with the specified tile_size - use the tile_size in the range
        me, mx = state.add_map("tiled_map", {"i": f"0:{tile_size}"})

        tasklet = state.add_tasklet("add_tiled", {"inp"}, {"outp"}, "outp = inp + 1")

        state.add_edge(inp, None, me, None, sdfg.make_array_memlet("_a"))
        state.add_edge(me, None, tasklet, "inp", sdfg.make_array_memlet("_a"))

        state.add_edge(tasklet, "outp", mx, None, dace.Memlet("_b[0]"))
        state.add_edge(mx, None, outp, None, dace.Memlet("_b[0]"))
        sdfg.fill_scope_connectors()

        return sdfg


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
    with pytest.warns(DeprecationWarning, match=r"The expand\(sdfg, state\) interface is deprecated"):
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


def test_expansion_with_keyword_arguments():
    """Test expanding library node with keyword arguments passed to expansion method."""
    sdfg, state, addnode = create_test_sdfg()

    # Test expansion with tile_size keyword argument
    result = addnode.expand(state, 'tiled', tile_size=16)
    assert result == 'tiled'

    # Find the nested SDFG that was created by the expansion
    nested_sdfg_nodes = [n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nested_sdfg_nodes) == 1

    nested_sdfg = nested_sdfg_nodes[0].sdfg

    # Verify the map range uses the correct tile_size
    found_tiled_map = False
    for sdfg_state in nested_sdfg.nodes():
        for node in sdfg_state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and node.label == 'tiled_map':
                # Check that the map range is "0:16" (range end is tile_size-1 since ranges are inclusive)
                assert node.map.range.ranges[0] == (0, 15, 1)  # 0:16 becomes (0, 15, 1)
                found_tiled_map = True
                break

    assert found_tiled_map, "Could not find the tiled_map in the expanded SDFG"

    # Test with different tile_size
    sdfg2, state2, addnode2 = create_test_sdfg()
    result2 = addnode2.expand(state2, 'tiled', tile_size=32)
    assert result2 == 'tiled'

    nested_sdfg_nodes2 = [n for n in state2.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    nested_sdfg2 = nested_sdfg_nodes2[0].sdfg

    found_tiled_map2 = False
    for sdfg_state in nested_sdfg2.nodes():
        for node in sdfg_state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and node.label == 'tiled_map':
                assert node.map.range.ranges[0] == (0, 31, 1)  # 0:32 becomes (0, 31, 1)
                found_tiled_map2 = True
                break

    assert found_tiled_map2, "Could not find the tiled_map in the second expanded SDFG"


if __name__ == '__main__':
    test_new_library_node_expand_interface()
    test_old_library_node_expand_interface()
    test_state_expand_library_node_method()
    test_state_expand_library_node_errors()
    test_library_node_expand_with_kwargs()
    test_compatibility_with_existing_expand_library_nodes()
    test_functional_correctness()
    test_implementation_override()
    test_expansion_with_keyword_arguments()
