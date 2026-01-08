# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from functools import reduce
from operator import mul
from typing import Dict, Collection
import warnings

from dace import SDFG, Memlet, dtypes
from dace.codegen import codegen
from dace.codegen.targets import cpp
from dace.subsets import Range


def test_reshape_strides_multidim_array_all_dims_unit():
    r = Range([(0, 0, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1]
    assert strides == [1]

    # To equal-sized shape
    target_dims = [1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1]
    assert strides == [1, 1]

    # To larger-sized shape
    target_dims = [1, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1, 1]
    assert strides == [1, 1, 1]


def test_reshape_strides_multidim_array_some_dims_unit():
    r = Range([(0, 1, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [2, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1]
    # To equal-sized shape, but units first.
    target_dims = [1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape.
    target_dims = [2, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1, 1]
    # To larger-sized shape, but units first.
    target_dims = [1, 1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 2, 1]


def test_reshape_strides_multidim_array_different_shape():
    r = Range([(0, 4, 1), (0, 5, 1)])

    # To smaller-sized shape
    target_dims = [30]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [15, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape
    target_dims = [3, 5, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [10, 2, 1]


def test_reshape_strides_from_strided_range():
    r = Range([(0, 4, 2), (0, 6, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def test_reshape_strides_from_strided_and_offset_range():
    r = Range([(10, 14, 2), (10, 16, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def test_arrays_bigger_than_max_stack_size_get_deallocated():
    # Setup SDFG with array A that is too big to be allocated on the stack.
    sdfg = SDFG("test")
    sdfg.add_array(name="A", shape=(10000,), dtype=dtypes.float64, storage=dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("state", is_start_block=True)
    read = state.add_access("A")
    tasklet = state.add_tasklet("dummy", {"a"}, {}, "a = 1")
    state.add_memlet_path(read, tasklet, dst_conn="a", memlet=Memlet("A[0]"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Generate code for the program by traversing the SDFG state by state
        program_objects = codegen.generate_code(sdfg)

        # Assert that we get the expected warning message
        assert w
        assert any("was allocated on the heap instead of" in str(warn.message) for warn in w)

        # In code, assert that we allocate _and_ deallocate on the heap
        code = program_objects[0].clean_code
        assert code.find("A = new double") > 0, "A is allocated on the heap."
        assert code.find("delete[] A") > 0, "A is deallocated from the heap."


if __name__ == '__main__':
    test_reshape_strides_multidim_array_all_dims_unit()
    test_reshape_strides_multidim_array_some_dims_unit()
    test_reshape_strides_multidim_array_different_shape()
    test_reshape_strides_from_strided_range()
    test_reshape_strides_from_strided_and_offset_range()

    test_arrays_bigger_than_max_stack_size_get_deallocated()
