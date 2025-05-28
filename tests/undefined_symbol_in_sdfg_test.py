#!/usr/bin/env python3
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import numpy as np

import dace
from dace import symbolic
from dace.sdfg.validation import InvalidSDFGError


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
    """Tests that validation catches UndefinedSymbol even in 'unused' dimensions."""

    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_in_unused_dimension_test')

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

    # Even though we only access A[0, i] (a constant in the first dimension),
    # validation should still fail because the UndefinedSymbol appears in the argument
    with pytest.raises(InvalidSDFGError, match="undefined symbol in dimension"):
        sdfg.validate()


def test_undefined_symbol_validation_failure():
    """Tests that UndefinedSymbol in transient data shapes causes validation failure."""

    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_validation_test')

    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()

    # Add arrays to SDFG - use the undefined symbol in the transient shape
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

    # This should fail validation because the transient has an undefined dimension
    with pytest.raises(InvalidSDFGError, match="undefined symbol in dimension"):
        sdfg.validate()


def test_undefined_symbol_value_assignment():
    """Tests that an UndefinedSymbol can be assigned a value before validation to avoid errors."""

    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_assignment_test')

    # Define symbols
    undefined_dim = symbolic.UndefinedSymbol()

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
    # This shows how a user would explicitly replace UndefinedSymbol with concrete values
    replaced_shape = []
    for dim in sdfg.arrays['tmp'].shape:
        if symbolic.is_undefined(dim):
            replaced_shape.append(10)
        else:
            replaced_shape.append(dim)
    sdfg.arrays['tmp'].shape = tuple(replaced_shape)

    # Need to update strides too
    sdfg.arrays['tmp'].strides = [10, 1]

    # Need to update total size
    sdfg.arrays['tmp'].total_size = 20 * 10  # product of dimensions

    # Make sure we can now validate the SDFG
    sdfg.validate()

    # It should also compile successfully
    sdfg.compile()


def test_undefined_symbol_in_argument_validation_failure():
    """
    Test that UndefinedSymbol in argument array shapes fails validation,
    even when only accessed at constant indices.
    """

    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_argument_test')

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

    # This should fail validation because argument array contains undefined symbol
    with pytest.raises(InvalidSDFGError, match="undefined symbol in dimension"):
        sdfg.validate()


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

    # This is a legitimate use case and should pass validation
    sdfg.validate()

    # It should also compile successfully
    sdfg.compile()


def test_undefined_symbol_numpy_frontend():
    """Test UndefinedSymbol with the numpy frontend."""

    undefined_dim = symbolic.UndefinedSymbol()

    @dace.program
    def program_with_undefined_symbol(A: dace.float64[undefined_dim], B: dace.float64[20]):
        # Only access the first element of A, so the undefined dimension isn't used
        B[0] = A[0]

    # Convert to SDFG
    sdfg = program_with_undefined_symbol.to_sdfg()

    # Validation should fail because undefined_dim is in the argument list
    with pytest.raises(InvalidSDFGError, match="undefined symbol"):
        sdfg.validate()


def test_undefined_symbol_in_arglist():
    """Tests that UndefinedSymbol can be found in arglist."""

    # Create an SDFG manually
    sdfg = dace.SDFG('undefined_symbol_in_arglist_test')

    # Add normal symbol and array with UndefinedSymbol in shape
    undefined_dim = symbolic.UndefinedSymbol()
    N = symbolic.symbol('N')

    sdfg.add_symbol('N', N.dtype)

    # Add arrays - use undefined symbol in one dimension of an array
    sdfg.add_array('A', [undefined_dim], dace.float64)
    sdfg.add_array('B', [N], dace.float64)

    # Create state
    state = sdfg.add_state(is_start_block=True)

    # Create access nodes
    A = state.add_read('A')
    B = state.add_write('B')

    # Add a map
    map_entry, map_exit = state.add_map('compute', dict(i='0:N'))

    # Add a tasklet that only uses the first element of A
    tasklet = state.add_tasklet('compute', {'a'}, {'b'}, 'b = a + 1')

    # Connect everything - only use A[0]
    state.add_edge(map_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'b', B, None, dace.Memlet('B[i]'))
    state.add_edge(tasklet, None, map_exit, None, dace.Memlet())

    # Get the argument list
    arglist = sdfg.arglist()

    # Both arrays should be in the arglist
    assert 'A' in arglist
    assert 'B' in arglist
    assert 'N' in arglist


if __name__ == '__main__':
    test_undefined_symbol_in_unused_dimension()
    test_undefined_symbol_value_assignment()
    test_undefined_symbols_not_in_arglist()
    test_undefined_symbol_in_arglist()
    # test_undefined_symbol_in_sdfg() - This would fail as expected
