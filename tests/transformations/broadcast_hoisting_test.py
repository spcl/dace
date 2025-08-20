# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests dimension creation transformations."""

import numpy as np
import pytest
import dace
from dace.transformation.dataflow import BroadcastHoisting
from copy import deepcopy


def _get_small_input_data(sdfg: dace.SDFG) -> dict:
    """
    Generate small input data for the given SDFG with non-trivial entries.
    """
    input_data = {}
    sym_data = {}
    for sym in sdfg.symbols:
        if sym not in sdfg.constants:
            sym_data[sym] = 3  # Default value for symbolic constants
            input_data[sym] = 3  # Default value for symbolic arguments

    for argName, argType in sdfg.arglist().items():
        if isinstance(argType, dace.data.Scalar):
            input_data[argName] = 3
            continue

        shape = []
        for entry in argType.shape:
            shape.append(dace.symbolic.evaluate(entry, sdfg.constants or sym_data))
        shape = tuple(shape)
        arr = dace.ndarray(shape=shape, dtype=argType.dtype)
        # prime number to avoid nice alignment
        arr[:] = (np.arange(arr.size).reshape(arr.shape) + 3) % 7 + 3
        input_data[argName] = arr
    return input_data

def _test_for_unchanged_behavior(prog, array_name, dim_increase = 0):
    sdfg: dace.SDFG = prog.to_sdfg(simplify=True)
    sdfg.validate()

    # Get the dimension of the array
    old_array_dim = len(sdfg.arrays[array_name].shape)

    # Get ground truth values if we expect dimension increase
    if dim_increase > 0:
        input_data = _get_small_input_data(sdfg)
        ground_truth = deepcopy(input_data)
        sdfg(**ground_truth)

    # Apply the transformation
    sdfg.apply_transformations_repeated(BroadcastHoisting)
    sdfg.validate()
    sdfg.save("test_broadcast_hoisting.sdfg")

    # Check that the array shape has been modified correctly
    new_array_dim = len(sdfg.arrays[array_name].shape)
    assert new_array_dim == old_array_dim + dim_increase

    # Test if the behavior is unchanged if we expect dimension increase
    if dim_increase > 0:
        output_data = deepcopy(input_data)
        sdfg(**output_data)
        np.testing.assert_equal(output_data, ground_truth)


def test_broadcast_hoisting_basic():

    @dace.program
    def tester(B: dace.float32[10], C: dace.float32[10, 10]):
        A = dace.define_local([10], dace.float32)
        for i in dace.map[0:10]:
                A[i] = B[i]
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
              C[i, j] = A[i]

    _test_for_unchanged_behavior(tester, 'A', 1)



def test_broadcast_hoisting_non_transient():
    """Tests that the transformation doesn't apply to non-transient arrays."""
    
    @dace.program
    def non_transient_test(A: dace.float32[10], C: dace.float32[10, 10]):
        # A is an input array, not transient
        for i in dace.map[0:10]:
            A[i] = A[i] + 1  # Modify but don't create
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[i]  # Broadcast
    
    _test_for_unchanged_behavior(non_transient_test, 'A')


def test_broadcast_hoisting_equal_volume():
    """Tests that the transformation doesn't apply when read volume equals write volume."""
    
    @dace.program
    def equal_volume_test(B: dace.float32[10], C: dace.float32[10]):
        A = dace.define_local([10], dace.float32)
        for i in dace.map[0:10]:
            A[i] = B[i]  # Write to A
        for i in dace.map[0:10]:
            C[i] = A[i]  # Read from A with same volume
    
    _test_for_unchanged_behavior(equal_volume_test, 'A')


def test_broadcast_hoisting_less_read_volume():
    """Tests that the transformation doesn't apply when read volume is less than write volume."""
    
    @dace.program
    def less_read_test(B: dace.float32[10, 10], C: dace.float32[5]):
        A = dace.define_local([10, 10], dace.float32)
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                A[i, j] = B[i, j]  # Write to full array
        for i in dace.map[0:5]:
            C[i] = A[i, 0]  # Read only part of the array
    
    _test_for_unchanged_behavior(less_read_test, 'A')


def test_broadcast_hoisting_multiple_dims():
    """Test broadcasting with multiple additional dimensions."""
    
    @dace.program
    def multi_dim_test(B: dace.float32[10], C: dace.float32[10, 10, 10]):
        A = dace.define_local([10], dace.float32)
        for i in dace.map[0:10]:
            A[i] = B[i]
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                for k in dace.map[0:10]:
                    C[i, j, k] = A[i]  # Broadcasting to two additional dimensions
    
    _test_for_unchanged_behavior(multi_dim_test, 'A', 2)


def test_broadcast_hoisting_symbolic():
    """Test broadcasting with symbolic dimensions."""
    
    N = dace.symbol('N')
    
    @dace.program
    def symbolic_test(B: dace.float32[N], C: dace.float32[N, N]):
        A = dace.define_local([N], dace.float32)
        for i in dace.map[0:N]:
            A[i] = B[i]
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                C[i, j] = A[i]
    
    _test_for_unchanged_behavior(symbolic_test, 'A', 1)


def test_broadcast_hoisting_strided_access():
    """Test broadcasting with strided access patterns."""
    
    @dace.program
    def strided_test(B: dace.float32[20], C: dace.float32[10, 10]):
        A = dace.define_local([10], dace.float32)
        for i in dace.map[0:10]:
            A[i] = B[i * 2]  # Strided read from B
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[i]  # Broadcast
    
    _test_for_unchanged_behavior(strided_test, 'A', 1)


def test_broadcast_hoisting_multiple_consumers():
    """Test broadcasting with multiple consumers of the broadcast data."""
    
    @dace.program
    def multiple_consumers_test(B: dace.float32[10], C: dace.float32[10, 10], D: dace.float32[10, 5]):
        A = dace.define_local([10], dace.float32)
        for i in dace.map[0:10]:
            A[i] = B[i]
        # Multiple consumers of A
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[i]
        for i in dace.map[0:10]:
            for j in dace.map[0:5]:
                D[i, j] = A[i] * 2
    
    _test_for_unchanged_behavior(multiple_consumers_test, 'A', 1)


def test_broadcast_hoisting_nested():
    """Test with nested broadcasting operations."""
    
    @dace.program
    def nested_broadcast_test(B: dace.float32[10], D: dace.float32[10, 10, 10]):
        A = dace.define_local([10], dace.float32)
        C = dace.define_local([10, 10], dace.float32)
        
        for i in dace.map[0:10]:
            A[i] = B[i]
        
        # First broadcast
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[i]
        
        # Second broadcast
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                for k in dace.map[0:10]:
                    D[i, j, k] = C[i, j]
    
    _test_for_unchanged_behavior(nested_broadcast_test, 'A', 1)
    _test_for_unchanged_behavior(nested_broadcast_test, 'C', 1)


def test_broadcast_hoisting_small_ranges():
    """Test broadcasting with small ranges, including size 1."""
    
    @dace.program
    def small_range_test(B: dace.float32[5], C: dace.float32[5, 1]):
        A = dace.define_local([5], dace.float32)
        for i in dace.map[0:5]:
            A[i] = B[i]
        for i in dace.map[0:5]:
            for j in dace.map[0:1]:  # Single iteration map
                C[i, j] = A[i]
    
    _test_for_unchanged_behavior(small_range_test, 'A', 1)


def test_broadcast_hoisting_non_consecutive_maps():
    """Test with operations between the two maps."""
    
    @dace.program
    def non_consecutive_test(B: dace.float32[10], C: dace.float32[10, 10]):
        A = dace.define_local([10], dace.float32)
        E = dace.define_local([1], dace.float32)
        
        for i in dace.map[0:10]:
            A[i] = B[i]
        
        # Operation between maps
        E[0] = 1.0
        
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[i] * E[0]
    
    _test_for_unchanged_behavior(non_consecutive_test, 'A', 1)


def test_broadcast_hoisting_indirect_access():
    """Test with indirect array access."""
    
    @dace.program
    def indirect_access_test(B: dace.float32[10], idx: dace.int32[10], C: dace.float32[10, 10]):
        A = dace.define_local([10], dace.float32)
        
        for i in dace.map[0:10]:
            A[i] = B[i]
        
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                C[i, j] = A[idx[i]]  # Indirect access
    
    # Should not transform due to indirect access
    _test_for_unchanged_behavior(indirect_access_test, 'A')


def test_broadcast_hoisting_dynamic_range():
    """Test with dynamically computed range bounds."""
    
    @dace.program
    def dynamic_range_test(B: dace.float32[10], N: dace.int32, C: dace.float32[10, 10]):
        A = dace.define_local([10], dace.float32)
        
        for i in dace.map[0:10]:
            A[i] = B[i]
        
        # Dynamic upper bound
        for i in dace.map[0:10]:
            for j in dace.map[0:N]:  # Dynamic bound
                if j < 10:  # Safety check
                    C[i, j] = A[i]
    
    _test_for_unchanged_behavior(dynamic_range_test, 'A', 1)


def test_broadcast_hoisting_mixed_dimensions():
    """Test with mixed dimensions in arrays."""
    
    @dace.program
    def mixed_dim_test(B: dace.float32[10, 5], C: dace.float32[10, 5, 8]):
        # A is already multidimensional
        A = dace.define_local([10, 5], dace.float32)
        
        for i in dace.map[0:10]:
            for j in dace.map[0:5]:
                A[i, j] = B[i, j]
        
        for i in dace.map[0:10]:
            for j in dace.map[0:5]:
                for k in dace.map[0:8]:
                    C[i, j, k] = A[i, j]  # Broadcast along third dimension
    
    _test_for_unchanged_behavior(mixed_dim_test, 'A', 1)


if __name__ == "__main__":
    test_broadcast_hoisting_basic()
    test_broadcast_hoisting_non_transient()
    test_broadcast_hoisting_equal_volume()
    test_broadcast_hoisting_less_read_volume()
    test_broadcast_hoisting_multiple_dims()
    test_broadcast_hoisting_symbolic()
    test_broadcast_hoisting_strided_access()
    test_broadcast_hoisting_multiple_consumers()
    test_broadcast_hoisting_nested()
    test_broadcast_hoisting_small_ranges()
    test_broadcast_hoisting_non_consecutive_maps()
    test_broadcast_hoisting_indirect_access()
    test_broadcast_hoisting_dynamic_range()
    test_broadcast_hoisting_mixed_dimensions()

