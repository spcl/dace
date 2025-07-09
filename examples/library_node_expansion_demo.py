#!/usr/bin/env python3
"""
Example demonstrating the new simpler library node expansion interface.
"""

import dace
import numpy as np
import sys
import os

# Add the test library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'library'))
import addlib


def example_old_interface():
    """Example using the old interface (still supported)"""
    print("=== Old Interface Example ===")

    # Create SDFG
    sdfg = dace.SDFG('old_interface_example')
    sdfg.add_array('input', [5], dace.float32)
    sdfg.add_array('output', [5], dace.float32)
    state = sdfg.add_state()

    # Add library node
    input_node = state.add_read('input')
    output_node = state.add_write('output')
    add_node = addlib.AddNode('add_old')
    state.add_node(add_node)

    # Connect nodes
    state.add_edge(input_node, None, add_node, '_a', dace.Memlet('input[0:5]'))
    state.add_edge(add_node, '_b', output_node, None, dace.Memlet('output[0:5]'))

    # Expand using old interface
    result = add_node.expand(sdfg, state)
    print(f"Expanded with implementation: {result}")

    # Test execution
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    output_data = np.zeros(5, dtype=np.float32)

    sdfg(input=input_data, output=output_data)
    print(f"Input: {input_data}")
    print(f"Output: {output_data}")
    print()


def example_new_library_node_interface():
    """Example using the new LibraryNode.expand() interface"""
    print("=== New LibraryNode Interface Example ===")

    # Create SDFG
    sdfg = dace.SDFG('new_interface_example')
    sdfg.add_array('input', [5], dace.float32)
    sdfg.add_array('output', [5], dace.float32)
    state = sdfg.add_state()

    # Add library node
    input_node = state.add_read('input')
    output_node = state.add_write('output')
    add_node = addlib.AddNode('add_new')
    state.add_node(add_node)

    # Connect nodes
    state.add_edge(input_node, None, add_node, '_a', dace.Memlet('input[0:5]'))
    state.add_edge(add_node, '_b', output_node, None, dace.Memlet('output[0:5]'))

    # Expand using new interface - much simpler!
    result = add_node.expand(state, 'pure')
    print(f"Expanded with implementation: {result}")

    # Test execution
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    output_data = np.zeros(5, dtype=np.float32)

    sdfg(input=input_data, output=output_data)
    print(f"Input: {input_data}")
    print(f"Output: {output_data}")
    print()


def example_state_interface():
    """Example using the new SDFGState.expand_library_node() interface"""
    print("=== New SDFGState Interface Example ===")

    # Create SDFG
    sdfg = dace.SDFG('state_interface_example')
    sdfg.add_array('input', [5], dace.float32)
    sdfg.add_array('output', [5], dace.float32)
    state = sdfg.add_state()

    # Add library node
    input_node = state.add_read('input')
    output_node = state.add_write('output')
    add_node = addlib.AddNode('add_state')
    state.add_node(add_node)

    # Connect nodes
    state.add_edge(input_node, None, add_node, '_a', dace.Memlet('input[0:5]'))
    state.add_edge(add_node, '_b', output_node, None, dace.Memlet('output[0:5]'))

    # Expand using state interface - even cleaner!
    result = state.expand_library_node(add_node, 'pure')
    print(f"Expanded with implementation: {result}")

    # Test execution
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    output_data = np.zeros(5, dtype=np.float32)

    sdfg(input=input_data, output=output_data)
    print(f"Input: {input_data}")
    print(f"Output: {output_data}")
    print()


def example_auto_expand():
    """Example showing automatic expansion still works"""
    print("=== Automatic Expansion Example ===")

    # Create SDFG
    sdfg = dace.SDFG('auto_expand_example')
    sdfg.add_array('input', [5], dace.float32)
    sdfg.add_array('output', [5], dace.float32)
    state = sdfg.add_state()

    # Add library node
    input_node = state.add_read('input')
    output_node = state.add_write('output')
    add_node = addlib.AddNode('add_auto')
    state.add_node(add_node)

    # Connect nodes
    state.add_edge(input_node, None, add_node, '_a', dace.Memlet('input[0:5]'))
    state.add_edge(add_node, '_b', output_node, None, dace.Memlet('output[0:5]'))

    # Expand all library nodes automatically
    sdfg.expand_library_nodes()
    print("Automatically expanded all library nodes")

    # Test execution
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    output_data = np.zeros(5, dtype=np.float32)

    sdfg(input=input_data, output=output_data)
    print(f"Input: {input_data}")
    print(f"Output: {output_data}")
    print()


if __name__ == '__main__':
    print("DaCe Library Node Expansion Interface Examples")
    print("=" * 50)

    example_old_interface()
    example_new_library_node_interface()
    example_state_interface()
    example_auto_expand()

    print("All examples completed successfully!")
