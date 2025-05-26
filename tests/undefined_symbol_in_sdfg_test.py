#!/usr/bin/env python3
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import numpy as np

import dace
from dace import symbolic


def test_undefined_symbol_in_sdfg():
    """Tests that an UndefinedSymbol can be used in an SDFG but raises an error when compiled."""
    
    # Create an SDFG manually instead of using @dace.program
    sdfg = dace.SDFG('undefined_symbol_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [20, undefined_dim], dace.float64)
    
    # Create state and add nodes
    state = sdfg.add_state()
    
    # Validate SDFG - it should pass validation at this point
    sdfg.validate()
    
    # Try to generate code - this should fail because of the undefined symbol
    with pytest.raises(TypeError, match="undefined symbol"):
        sdfg.compile()


def test_undefined_symbol_in_unused_dimension():
    """Tests that an UndefinedSymbol in an unused dimension allows code generation to proceed."""
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_unused_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    
    # Add transient with undefined dimension in the first position
    sdfg.add_transient('tmp', [undefined_dim, 20], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Create access nodes
    A = state.add_read('A')
    tmp = state.add_access('tmp')
    B = state.add_write('B')
    
    # Add MapEntry and MapExit nodes
    map_entry, map_exit = state.add_map('compute', dict(i='0:20'))
    
    # Add tasklets
    tasklet_write = state.add_tasklet('write', {'a'}, {'b'}, 
                                     'b = a')
    tasklet_read = state.add_tasklet('read', {'a'}, {'b'}, 
                                    'b = a')
    
    # Connect everything with only using the defined dimension (avoid the undefined one)
    state.add_edge(A, None, tasklet_write, 'a', dace.Memlet.simple('A[0, 0]'))
    state.add_edge(tasklet_write, 'b', tmp, None, dace.Memlet.simple('tmp[0, 0]'))
    state.add_edge(tmp, None, tasklet_read, 'a', dace.Memlet.simple('tmp[0, 0]'))
    state.add_edge(tasklet_read, 'b', B, None, dace.Memlet.simple('B[0]'))
    
    # Validate SDFG - it should pass validation
    sdfg.validate()
    
    # This should compile successfully since the undefined dimension is never used
    sdfg.compile()


def test_undefined_symbol_value_assignment():
    """Tests that an UndefinedSymbol can be assigned a value before code generation."""
    
    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_assignment_test')
    
    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()
    
    # Add arrays to SDFG
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_transient('tmp', [20, undefined_dim], dace.float64)
    
    # Create state
    state = sdfg.add_state()
    
    # Find all data descriptors with undefined dimensions and replace them
    for desc in sdfg.arrays.values():
        if hasattr(desc, 'shape'):
            new_shape = []
            for dim in desc.shape:
                if isinstance(dim, symbolic.UndefinedSymbol):
                    # Replace undefined symbol with a concrete value
                    new_shape.append(10)
                else:
                    new_shape.append(dim)
            desc.shape = new_shape
    
    # Validate SDFG - it should pass validation
    sdfg.validate()
    
    # This should compile successfully since we've replaced the undefined symbols
    sdfg.compile()


if __name__ == '__main__':
    test_undefined_symbol_in_unused_dimension()
    test_undefined_symbol_value_assignment()
    # test_undefined_symbol_in_sdfg() - This would fail as expected
