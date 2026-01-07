# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import dace.subsets


# Helper function
def create_array(shape, strides):
    """Helper to create a dace.data.Array with specific strides"""
    sdfg = dace.SDFG('test_sdfg')
    name, array = sdfg.add_array(name='test_array', shape=shape, dtype=dace.float64, strides=strides)
    return array


def create_subset(ranges):
    """Helper to create a dace.subsets.Range from list of (begin, end, step) tuples"""
    return dace.subsets.Range(ranges)


def test_fortran_full_range_all_dimensions():
    """Test Fortran array with full range in all dimensions - should be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 9, 1), (0, 19, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_partial_first_dim_rest_size_one():
    """Test Fortran array with partial first dim, rest size 1 - should be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 4, 1), (0, 0, 1), (0, 0, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_partial_first_dim_rest_full():
    """Test Fortran array with partial first dim, rest full - should NOT be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 4, 1), (0, 19, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_fortran_full_first_partial_second_rest_size_one():
    """Test Fortran array with full first, partial second, rest size 1 - should be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 9, 1), (0, 9, 1), (8, 8, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_full_first_partial_second_rest_full():
    """Test Fortran array with full first, partial second, rest full - should NOT be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 9, 1), (0, 9, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_fortran_partial_last_dim_only():
    """Test Fortran array with only last dim partial - should be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(0, 9, 1), (0, 19, 1), (0, 5, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_single_element_all_dims():
    """Test Fortran array with single element (all dims size 1) - should be contiguous"""
    array = create_array((10, 20, 30), (1, 10, 200))
    subset = create_subset([(5, 5, 1), (10, 10, 1), (15, 15, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_full_range_all_dimensions():
    """Test C array with full range in all dimensions - should be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 9, 1), (0, 19, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_partial_last_dim_rest_size_one():
    """Test C array with partial last dim, rest size 1 - should be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 0, 1), (0, 0, 1), (6, 14, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_partial_last_dim_rest_full():
    """Test C array with partial last dim, rest full - should NOT be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 9, 1), (0, 19, 1), (0, 14, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_c_full_last_partial_second_rest_size_one():
    """Test C array with full last, partial second, rest size 1 - should be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 0, 1), (0, 9, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_full_last_partial_second_rest_full():
    """Test C array with full last, partial second, rest full - should NOT be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 9, 1), (0, 9, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_c_partial_first_dim_only():
    """Test C array with only first dim partial - should be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(0, 4, 1), (0, 19, 1), (0, 29, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_single_element_all_dims():
    """Test C array with single element (all dims size 1) - should be contiguous"""
    array = create_array((10, 20, 30), (600, 30, 1))
    subset = create_subset([(5, 5, 1), (10, 10, 1), (15, 15, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_2d_full_first_partial_second():
    """Test Fortran 2D array with full first, partial second - should be contiguous"""
    array = create_array((50, 100), (1, 50))
    subset = create_subset([(0, 49, 1), (10, 89, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_2d_partial_first_full_second():
    """Test Fortran 2D array with partial first, full second - should NOT be contiguous"""
    array = create_array((50, 100), (1, 50))
    subset = create_subset([(10, 39, 1), (0, 99, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_c_2d_full_first_partial_second():
    """Test C 2D array with full first, partial second - should NOT be contiguous"""
    array = create_array((50, 100), (100, 1))
    subset = create_subset([(0, 49, 1), (10, 89, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_c_2d_partial_first_full_second():
    """Test C 2D array with partial first, full second - should be contiguous"""
    array = create_array((50, 100), (100, 1))
    subset = create_subset([(10, 39, 1), (0, 99, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_1d_full_range():
    """Test 1D array with full range - should be contiguous"""
    array = create_array((100, ), (1, ))
    subset = create_subset([(0, 99, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_1d_partial_range():
    """Test 1D array with partial range - should be contiguous"""
    array = create_array((100, ), (1, ))
    subset = create_subset([(25, 74, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_4d_complex_contiguous():
    """Test Fortran 4D array - full first two dims, partial third, size 1 last"""
    array = create_array((5, 10, 15, 20), (1, 5, 50, 750))
    subset = create_subset([(0, 4, 1), (0, 9, 1), (0, 7, 1), (0, 0, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_4d_complex_contiguous():
    """Test C 4D array - full last two dims, partial second, size 1 first"""
    array = create_array((5, 10, 15, 20), (3000, 300, 20, 1))
    subset = create_subset([(0, 0, 1), (0, 4, 1), (0, 14, 1), (0, 19, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_fortran_4d_non_contiguous():
    """Test Fortran 4D array - partial middle dims without trailing size 1"""
    array = create_array((5, 10, 15, 20), (1, 5, 50, 750))
    subset = create_subset([(0, 4, 1), (0, 4, 1), (0, 7, 1), (0, 10, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_fortran_4d_contiguous_first_partial_second_rest_one():
    """Test Fortran 4D array - partial middle dims without trailing size 1"""
    array = create_array((5, 10, 15, 20), (1, 5, 50, 750))
    subset = create_subset([(0, 4, 1), (0, 4, 1), (1, 1, 1), (1, 1, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_c_4d_non_contiguous():
    """Test C 4D array - partial middle dims without leading size 1"""
    array = create_array((5, 10, 15, 20), (3000, 300, 20, 1))
    subset = create_subset([(0, 2, 1), (0, 4, 1), (0, 7, 1), (0, 19, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is False


def test_c_4d_contiguous_first_partial_second_rest_one():
    """Test Fortran 4D array - partial middle dims without trailing size 1"""
    array = create_array((5, 10, 15, 20), (3000, 300, 20, 1))
    subset = create_subset([(1, 1, 1), (1, 1, 1), (0, 5, 1), (0, 19, 1)])
    result = subset.is_contiguous_subset(array)
    assert result is True


def test_non_standard_strides_returns_false():
    """Test array with non-standard strides - should return False"""
    array = create_array((10, 20, 30), (2, 20, 400))  # Custom strides
    subset = create_subset([(0, 9, 1), (0, 19, 1), (0, 29, 1)])

    result = subset.is_contiguous_subset(array)
    assert result is False


if __name__ == "__main__":
    tests = [obj for name, obj in globals().items() if callable(obj) and name.startswith("test_")]
    for test_function in tests:
        test_function()
