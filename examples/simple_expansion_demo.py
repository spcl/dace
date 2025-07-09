#!/usr/bin/env python3
"""
Simple example demonstrating the new simpler library node expansion interface.
"""

import dace
import sys
import os

# Add the test library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'library'))
import addlib


def example_comparison():
    """Compare old vs new interface"""
    print("DaCe Library Node Expansion Interface Comparison")
    print("=" * 50)

    # Create SDFG
    sdfg = dace.SDFG('comparison_example')
    sdfg.add_array('input', [5], dace.float32)
    sdfg.add_array('output', [5], dace.float32)
    state = sdfg.add_state()

    # Add library node
    input_node = state.add_read('input')
    output_node = state.add_write('output')
    add_node = addlib.AddNode('add_comparison')
    state.add_node(add_node)

    # Connect nodes
    state.add_edge(input_node, None, add_node, '_a', dace.Memlet('input[0:5]'))
    state.add_edge(add_node, '_b', output_node, None, dace.Memlet('output[0:5]'))

    print("Available implementations:", list(add_node.implementations.keys()))
    print("Default implementation:", add_node.default_implementation)
    print()

    # Show the different ways to expand
    print("1. OLD INTERFACE (still supported):")
    print("   result = add_node.expand(sdfg, state)")
    print()

    print("2. NEW LIBRARYNODE INTERFACE:")
    print("   result = add_node.expand(state, 'pure')")
    print()

    print("3. NEW SDFGSTATE INTERFACE:")
    print("   result = state.expand_library_node(add_node, 'pure')")
    print()

    print("4. AUTOMATIC EXPANSION (unchanged):")
    print("   sdfg.expand_library_nodes()")
    print()

    # Test the new interface
    print("Testing new interface...")
    result = add_node.expand(state, 'pure')
    print(f"✓ Expansion successful with implementation: {result}")
    print()

    # Test state interface
    sdfg2 = dace.SDFG('state_example')
    sdfg2.add_array('input', [5], dace.float32)
    sdfg2.add_array('output', [5], dace.float32)
    state2 = sdfg2.add_state()

    input_node2 = state2.add_read('input')
    output_node2 = state2.add_write('output')
    add_node2 = addlib.AddNode('add_state')
    state2.add_node(add_node2)

    state2.add_edge(input_node2, None, add_node2, '_a', dace.Memlet('input[0:5]'))
    state2.add_edge(add_node2, '_b', output_node2, None, dace.Memlet('output[0:5]'))

    print("Testing state interface...")
    result2 = state2.expand_library_node(add_node2, 'pure')
    print(f"✓ State expansion successful with implementation: {result2}")
    print()

    print("Benefits of the new interface:")
    print("- Cleaner API: no need to pass both sdfg and state")
    print("- Direct implementation selection")
    print("- Better error handling")
    print("- More intuitive for users")
    print("- Backward compatible")
    print()

    print("All interface tests completed successfully!")


if __name__ == '__main__':
    example_comparison()
