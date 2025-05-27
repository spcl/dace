#!/usr/bin/env python3
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import numpy as np

import dace
from dace import symbolic


def test_undefined_symbol_in_sdfg():
    """Tests that an UndefinedSymbol can be used in an SDFG but raises an error when compiled."""
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [20, undefined_dim], dace.float64)
    
    # Create state and add nodes
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    tmp = state.add_access('tmp')
    B = state.add_write('B')
    
    # Add a map for i
    outer_map_entry, outer_map_exit = state.add_map('i', dict(i='0:20'))
    
    # Add a map for j within i
    inner_map_entry, inner_map_exit = state.add_map('j', dict(j='0:20'))
    
    # Add compute tasklet
    tasklet = state.add_tasklet('compute', {'a'}, {'t'}, 't = a')
    
    # Add tasklet for using tmp
    read_tasklet = state.add_tasklet('read_tmp', {'t'}, {'b'}, 'b = t')
    
    # Connect everything
    state.add_edge(A, None, outer_map_entry, None, dace.Memlet())
    state.add_edge(outer_map_entry, None, inner_map_entry, None, dace.Memlet())
    state.add_edge(inner_map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[i, j]'))
    state.add_edge(tasklet, 't', tmp, None, dace.Memlet('tmp[i, 0]'))
    state.add_edge(inner_map_entry, None, read_tasklet, None, dace.Memlet())
    state.add_edge(tmp, None, read_tasklet, 't', dace.Memlet('tmp[i, 0]'))
    state.add_edge(read_tasklet, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(read_tasklet, None, inner_map_exit, None, dace.Memlet())
    state.add_edge(inner_map_exit, None, outer_map_exit, None, dace.Memlet())
    
    # Try to generate code - this should fail because of the undefined symbol
    with pytest.raises((TypeError, dace.sdfg.InvalidSDFGError), match=r"undefined symbol|contains undefined symbol"):
        sdfg.compile()


def test_undefined_symbol_in_unused_dimension():
    """Tests that an UndefinedSymbol in an unused dimension allows code generation to proceed."""
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_unused_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG - in this case the undefined dimension is in the array A
    # which is an input (non-transient) so it doesn't need to be allocated
    sdfg.add_array('A', [undefined_dim, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [20, 20], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    tmp = state.add_access('tmp')
    B = state.add_write('B')
    
    # Add map
    map_entry, map_exit = state.add_map('compute', dict(i='0:20'))
    
    # Add tasklets
    tasklet_write = state.add_tasklet('write', {'a'}, {'t'}, 't = a')
    tasklet_read = state.add_tasklet('read', {'t'}, {'b'}, 'b = t')
    
    # Connect everything - only using the defined dimension (index 1),
    # avoiding the undefined one (index 0)
    state.add_edge(A, None, map_entry, None, dace.Memlet())
    state.add_edge(map_entry, None, tasklet_write, None, dace.Memlet())
    state.add_edge(A, None, tasklet_write, 'a', dace.Memlet('A[0, i]'))
    state.add_edge(tasklet_write, 't', tmp, None, dace.Memlet('tmp[i, 0]'))
    state.add_edge(tmp, None, tasklet_read, 't', dace.Memlet('tmp[i, 0]'))
    state.add_edge(tasklet_read, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(tasklet_read, None, map_exit, None, dace.Memlet())
    
    # This should compile successfully since the undefined dimension is never used
    sdfg.compile()


def test_undefined_symbol_value_assignment():
    """Tests that an UndefinedSymbol can be assigned a value before code generation."""
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_assignment_test')
    
    # Define symbols and register the undefined symbol in the SDFG
    undefined_dim = symbolic.UndefinedSymbol()
    sdfg.add_symbol('ud', undefined_dim.dtype)
    
    # Add arrays to SDFG - use the undefined symbol in the shape
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [20, undefined_dim], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    tmp = state.add_access('tmp')
    B = state.add_write('B')
    
    # Add map
    map_entry, map_exit = state.add_map('compute', dict(i='0:20', j='0:10'))
    
    # Add tasklet
    tasklet = state.add_tasklet('compute', {'a'}, {'b', 't'}, 'b = a; t = a')
    
    # Connect with edges that use the undefined symbol
    state.add_edge(A, None, map_entry, None, dace.Memlet())
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[i, j]'))
    state.add_edge(tasklet, 't', tmp, None, dace.Memlet('tmp[i, j]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(tasklet, None, map_exit, None, dace.Memlet())
    
    # Replace undefined symbols in the SDFG with concrete values
    for desc in sdfg.arrays.values():
        if hasattr(desc, 'shape'):
            new_shape = []
            for dim in desc.shape:
                if isinstance(dim, symbolic.UndefinedSymbol):
                    # Replace undefined symbol with a concrete value
                    new_shape.append(10)  # Assign a value of 10
                else:
                    new_shape.append(dim)
            desc.shape = tuple(new_shape)
    
    # Replace any map ranges that use the undefined symbol
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                new_ranges = {}
                for k, v in node.map.range.items():
                    start, end, step = v
                    if isinstance(start, symbolic.UndefinedSymbol):
                        start = 0
                    if isinstance(end, symbolic.UndefinedSymbol):
                        end = 10
                    if isinstance(step, symbolic.UndefinedSymbol):
                        step = 1
                    new_ranges[k] = (start, end, step)
                node.map.range = new_ranges
    
    # This should compile successfully since we've replaced the undefined symbols
    sdfg.compile()


def test_legitimate_undefined_symbol_in_argument():
    """
    Test a legitimate use of undefined symbols: a 1D array argument with runtime-defined
    size that is only read at a constant index.
    """
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_legitimate_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG - use undefined symbol in the input array (not transient)
    sdfg.add_array('A', [undefined_dim], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    B = state.add_write('B')
    
    # Add map
    map_entry, map_exit = state.add_map('compute', dict(i='0:5'))
    
    # Add tasklet
    tasklet = state.add_tasklet('compute', {'a'}, {'b'}, 'b = a')
    
    # Connect edges - only read A at constant index 0
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(tasklet, None, map_exit, None, dace.Memlet())
    
    # This should compile successfully since we only use constant index
    sdfg.compile()


def test_undefined_symbols_not_in_arglist():
    """
    Test that undefined symbols do not make it into SDFG.arglist if unused.
    """
    
    # Create an SDFG with an undefined symbol that isn't used in execution
    sdfg = dace.SDFG('unused_undefined_symbol_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    concrete_dim = symbolic.symbol('N')
    
    # Add a symbol that isn't used in the SDFG
    sdfg.add_symbol('unused_symbol', undefined_dim.dtype)
    sdfg.add_symbol('N', concrete_dim.dtype)
    
    # Add arrays not using the undefined symbol
    sdfg.add_array('A', [concrete_dim], dace.float64)
    sdfg.add_array('B', [concrete_dim], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    B = state.add_write('B')
    
    # Add map
    map_entry, map_exit = state.add_map('compute', dict(i='0:N'))
    
    # Add tasklet
    tasklet = state.add_tasklet('compute', {'a'}, {'b'}, 'b = a + 1')
    
    # Connect everything
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[i]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(tasklet, None, map_exit, None, dace.Memlet())
    
    # Get the argument list - undefined symbol should not be in it
    arglist = sdfg.arglist()
    
    # The unused undefined symbol should not be in the arglist
    assert 'unused_symbol' not in arglist
    
    # The concrete symbol N should be in the arglist
    assert 'N' in arglist


if __name__ == '__main__':
    test_undefined_symbol_in_unused_dimension()
    test_undefined_symbol_value_assignment()
    test_legitimate_undefined_symbol_in_argument()
    test_undefined_symbols_not_in_arglist()
    # test_undefined_symbol_in_sdfg() - This would fail as expected
