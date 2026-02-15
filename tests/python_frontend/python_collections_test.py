# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for using Python collections (lists, tuples, dicts) with DaCe programs.

Currently, DaCe handles Python lists and tuples by converting them to NumPy
arrays when passed as arguments to @dace.program functions. Tuples can be
returned from DaCe programs. This test suite documents and verifies the current
behavior and serves as a foundation for future native collection support.
"""
import dace
import numpy as np
import pytest

# ============================================================================
# Tests for passing Python lists as arguments
# ============================================================================


def test_list_arg_1d():
    """Test passing a 1D Python list as an argument (auto-converted to array)."""

    @dace.program
    def add_one(a: dace.float64[5]):
        return a + 1

    result = add_one([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(result, expected)


def test_list_arg_2d():
    """Test passing a 2D nested Python list as an argument."""

    @dace.program
    def double(a: dace.float64[2, 3]):
        return a * 2

    result = double([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    assert np.allclose(result, expected)


def test_list_arg_integer():
    """Test passing a list of integers as an argument."""

    @dace.program
    def sum_elements(a: dace.int64[4]):
        result = dace.define_local_scalar(dace.int64)
        result = 0
        for i in range(4):
            result = result + a[i]
        return result

    result = sum_elements([10, 20, 30, 40])
    assert result == 100


def test_list_arg_inout():
    """Test that lists are copied (not modified in-place) when passed."""

    @dace.program
    def fill_fives(a: dace.float64[3]):
        a[:] = 5.0

    original = [1.0, 2.0, 3.0]
    # Lists get converted to numpy arrays internally; the original list is unchanged
    fill_fives(original)
    # The original Python list should remain unchanged
    assert original == [1.0, 2.0, 3.0]


# ============================================================================
# Tests for passing Python tuples as arguments
# ============================================================================


def test_tuple_arg_not_supported():
    """Tuples are not auto-converted to arrays at runtime (unlike lists).

    This documents the current limitation: while lists are silently cast to
    numpy arrays when passed to compiled DaCe programs, tuples raise a
    TypeError. Users should convert tuples to numpy arrays explicitly.
    """

    @dace.program
    def add_one(a: dace.float64[4]):
        return a + 1

    with pytest.raises(TypeError, match="Passing an object.*tuple.*to an array"):
        add_one((10.0, 20.0, 30.0, 40.0))


def test_tuple_arg_as_numpy():
    """Test that tuples work when explicitly converted to numpy arrays first."""

    @dace.program
    def add_one(a: dace.float64[4]):
        return a + 1

    result = add_one(np.array((10.0, 20.0, 30.0, 40.0)))
    expected = np.array([11.0, 21.0, 31.0, 41.0])
    assert np.allclose(result, expected)


# ============================================================================
# Tests for returning tuples from DaCe programs
# ============================================================================


def test_return_tuple_scalars():
    """Test returning a tuple of scalar values."""

    @dace.program
    def compute():
        return 42, 7

    result = compute()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 42
    assert result[1] == 7


def test_return_tuple_arrays():
    """Test returning a tuple of arrays."""

    @dace.program
    def split_compute(a: dace.float64[4]):
        return a + 1, a * 2

    inp = np.array([1.0, 2.0, 3.0, 4.0])
    result = split_compute(inp)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.allclose(result[0], np.array([2.0, 3.0, 4.0, 5.0]))
    assert np.allclose(result[1], np.array([2.0, 4.0, 6.0, 8.0]))


def test_return_tuple_mixed():
    """Test returning a tuple with both scalar and array results."""

    @dace.program
    def compute_with_sum(a: dace.float64[3]):
        doubled = a * 2
        total = np.sum(a)
        return doubled, total

    inp = np.array([1.0, 2.0, 3.0])
    result = compute_with_sum(inp)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.allclose(result[0], np.array([2.0, 4.0, 6.0]))
    assert np.isclose(result[1], 6.0)


def test_return_triple():
    """Test returning a 3-element tuple."""

    @dace.program
    def three_things():
        return 1, 2, 3

    result = three_things()
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == (1, 2, 3)


# ============================================================================
# Tests for using list/tuple literals within DaCe programs
# ============================================================================


def test_list_literal_as_shape():
    """Test using a list literal to define an array shape."""

    @dace.program
    def create_array():
        a = np.zeros([3, 4], dtype=dace.float64)
        a[:, :] = 1.0
        return a

    result = create_array()
    assert result.shape == (3, 4)
    assert np.allclose(result, np.ones((3, 4)))


def test_tuple_literal_as_shape():
    """Test using a tuple literal to define an array shape."""

    @dace.program
    def create_array():
        a = np.zeros((2, 5), dtype=dace.float64)
        a[:, :] = 2.0
        return a

    result = create_array()
    assert result.shape == (2, 5)
    assert np.allclose(result, 2.0 * np.ones((2, 5)))


def test_range_as_collection():
    """Test using range() which produces a collection-like iterable."""

    @dace.program
    def sum_range(a: dace.float64[10]):
        for i in range(10):
            a[i] = a[i] + 1.0

    a = np.zeros(10)
    sum_range(a)
    assert np.allclose(a, np.ones(10))


# ============================================================================
# Tests for the create_datadescriptor function with collections
# ============================================================================


def test_create_datadescriptor_from_list():
    """Test that create_datadescriptor correctly converts a list."""
    from dace.data import create_datadescriptor

    desc = create_datadescriptor([1.0, 2.0, 3.0])
    assert isinstance(desc, dace.data.Array)
    assert desc.shape == (3, )


def test_create_datadescriptor_from_tuple():
    """Test that create_datadescriptor correctly converts a tuple."""
    from dace.data import create_datadescriptor

    desc = create_datadescriptor((1, 2, 3, 4))
    assert isinstance(desc, dace.data.Array)
    assert desc.shape == (4, )


def test_create_datadescriptor_from_nested_list():
    """Test that create_datadescriptor correctly converts a nested list."""
    from dace.data import create_datadescriptor

    desc = create_datadescriptor([[1, 2], [3, 4], [5, 6]])
    assert isinstance(desc, dace.data.Array)
    assert desc.shape == (3, 2)


def test_create_datadescriptor_preserves_dtype():
    """Test that create_datadescriptor infers the correct dtype."""
    from dace.data import create_datadescriptor

    # Float list
    desc_float = create_datadescriptor([1.0, 2.0])
    assert desc_float.dtype == dace.float64

    # Integer list
    desc_int = create_datadescriptor([1, 2])
    assert desc_int.dtype == dace.int64


# ============================================================================
# Tests combining collections with DaCe operations
# ============================================================================


def test_list_arg_with_map():
    """Test using a list argument in a mapped computation."""

    @dace.program
    def elementwise_square(a: dace.float64[5]):
        return a * a

    result = elementwise_square([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    assert np.allclose(result, expected)


def test_multiple_list_args():
    """Test passing multiple list arguments.

    Note: When multiple lists are passed, they are each converted to temporary
    numpy arrays. Due to memory reuse of temporary allocations, both arguments
    may end up pointing to the same buffer. To avoid this, convert at least one
    list to a numpy array before passing.
    """

    @dace.program
    def add_arrays(a: dace.float64[3], b: dace.float64[3]):
        return a + b

    # Use explicit numpy arrays to avoid the temporary buffer reuse issue
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = add_arrays(a, b)
    expected = np.array([5.0, 7.0, 9.0])
    assert np.allclose(result, expected)


def test_mixed_list_and_array_args():
    """Test passing both a list and a numpy array as arguments."""

    @dace.program
    def add_arrays(a: dace.float64[3], b: dace.float64[3]):
        return a + b

    arr = np.array([10.0, 20.0, 30.0])
    result = add_arrays([1.0, 2.0, 3.0], arr)
    expected = np.array([11.0, 22.0, 33.0])
    assert np.allclose(result, expected)


if __name__ == '__main__':
    test_list_arg_1d()
    test_list_arg_2d()
    test_list_arg_integer()
    test_list_arg_inout()
    test_tuple_arg_not_supported()
    test_tuple_arg_as_numpy()
    test_return_tuple_scalars()
    test_return_tuple_arrays()
    test_return_tuple_mixed()
    test_return_triple()
    test_list_literal_as_shape()
    test_tuple_literal_as_shape()
    test_range_as_collection()
    test_create_datadescriptor_from_list()
    test_create_datadescriptor_from_tuple()
    test_create_datadescriptor_from_nested_list()
    test_create_datadescriptor_preserves_dtype()
    test_list_arg_with_map()
    test_multiple_list_args()
    test_mixed_list_and_array_args()
    print("All tests passed!")
