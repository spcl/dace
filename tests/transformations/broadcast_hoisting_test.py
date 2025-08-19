# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests dimension creation transformations."""

import numpy as np
import pytest
import dace
from dace.transformation.dataflow import BroadcastHoisting
from copy import deepcopy


def _test_for_unchanged_behavior(prog, array_name, dim_increase = 0):
    sdfg: dace.SDFG = prog.to_sdfg(simplify=True)
    sdfg.validate()

    # Get the dimension of the array
    old_array_dim = len(sdfg.arrays[array_name].shape)

    # Get ground truth values if we expect dimension increase
    if dim_increase > 0:
        input_data = {}
        for argName, argType in sdfg.arglist().items():
            arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
            arr[:] = np.random.rand(*argType.shape).astype(argType.dtype.type)
            input_data[argName] = arr
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


if __name__ == "__main__":
    test_broadcast_hoisting_basic()
    test_broadcast_hoisting_non_transient()
    test_broadcast_hoisting_equal_volume()
    test_broadcast_hoisting_less_read_volume()
    test_broadcast_hoisting_multiple_dims()

