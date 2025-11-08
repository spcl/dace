# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import tempfile
import os

N = dace.symbol("N")


# Helper function
def create_array(shape, strides):
    sdfg = dace.SDFG('test_sdfg')
    _, array = sdfg.add_array(name='test_array', shape=shape, dtype=dace.float64, strides=strides)
    return sdfg, array


def test_get_packed_fortran_strides_1d():
    """Test Fortran strides for 1D array"""
    _, array = create_array((10, ), None)
    expected = (1, )
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_2d():
    """Test Fortran strides for 2D array"""
    _, array = create_array((10, 20), None)
    expected = (1, 10)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_3d():
    """Test Fortran strides for 3D array"""
    _, array = create_array((10, 20, 30), None)
    expected = (1, 10, 200)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_4d():
    """Test Fortran strides for 4D array"""
    _, array = create_array((5, 10, 15, 20), None)
    expected = (1, 5, 50, 750)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_c_strides_1d():
    """Test C strides for 1D array"""
    _, array = create_array((10, ), None)
    expected = (1, )
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_2d():
    """Test C strides for 2D array"""
    _, array = create_array((10, 20), None)
    expected = (20, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_3d():
    """Test C strides for 3D array"""
    _, array = create_array((10, 20, 30), None)
    expected = (600, 30, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_4d():
    """Test C strides for 4D array"""
    _, array = create_array((5, 10, 15, 20), None)
    expected = (3000, 300, 20, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_is_packed_fortran_strides_true_3d():
    """Test 3D array with Fortran strides"""
    _, array = create_array((10, 20, 30), (1, 10, 200))
    result = array.is_packed_fortran_strides()
    assert result is True


def test_is_packed_fortran_strides_false_c_layout():
    """Test array with C strides returns False"""
    _, array = create_array((10, 20, 30), (600, 30, 1))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_fortran_strides_false_custom_strides():
    """Test array with custom strides returns False"""
    _, array = create_array((10, 20, 30), (2, 20, 400))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_fortran_strides_false_wrong_order():
    """Test array with incorrect stride ordering"""
    _, array = create_array((10, 20, 30), (10, 1, 200))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_c_strides_true_1d():
    """Test 1D array with C strides"""
    _, array = create_array((10, ), (1, ))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_true_2d():
    """Test 2D array with C strides"""
    _, array = create_array((10, 20), (20, 1))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_true_3d():
    """Test 3D array with C strides"""
    _, array = create_array((10, 20, 30), (600, 30, 1))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_false_fortran_layout():
    """Test array with Fortran strides returns False"""
    _, array = create_array((10, 20, 30), (1, 10, 200))
    result = array.is_packed_c_strides()
    assert result is False


def test_is_packed_c_strides_false_custom_strides():
    """Test array with custom strides returns False"""
    _, array = create_array((10, 20, 30), (1200, 60, 2))
    result = array.is_packed_c_strides()
    assert result is False


def test_is_packed_c_strides_false_wrong_order():
    """Test array with incorrect stride ordering"""
    _, array = create_array((10, 20, 30), (1, 30, 600))
    result = array.is_packed_c_strides()
    assert result is False


def test_fortran_and_c_equivalent_for_1d():
    """Test that 1D arrays have same strides for both layouts"""
    _, array = create_array((100, ), (1, ))
    assert array.is_packed_fortran_strides() is True
    assert array.is_packed_c_strides() is True


def test_c_strides_calculation_accumulation():
    """Test stride accumulation for C layout"""
    _, array = create_array((2, 3, 4), None)
    result = array._get_packed_c_strides()
    assert result == (12, 4, 1)


def test_explicit_fortran_strides():
    """Test explicitly set Fortran strides"""
    _, array = create_array((5, 7, 9), (1, 5, 35))
    assert array.is_packed_fortran_strides() is True
    assert array.is_packed_c_strides() is False


def test_explicit_fortran_strides_not_packed():
    """Test explicitly set Fortran strides"""
    _, array = create_array((5, 7, 9), (10, 5 * 10, 35 * 10))
    assert array.is_packed_fortran_strides() is False
    assert array.is_packed_c_strides() is False


def test_explicit_c_strides():
    """Test explicitly set C strides"""
    _, array = create_array((5, 7, 9), (63, 9, 1))
    assert array.is_packed_c_strides() is True
    assert array.is_packed_fortran_strides() is False


def test_explicit_c_strides_not_packed():
    """Test explicitly set C strides"""
    _, array = create_array((5, 7, 9), (63 * 5, 9 * 5, 1))
    assert array.is_packed_c_strides() is False
    assert array.is_packed_fortran_strides() is False


def test_set_shape_reshape_1d_to_2d():
    """Reshape 1D array to 2D and verify packed Fortran/C strides."""
    _, array = create_array((12, ), None)

    before_fortran = array._get_packed_fortran_strides()
    before_c = array._get_packed_c_strides()
    assert before_fortran == (1, )
    assert before_c == (1, )

    array.set_shape((3, 4))
    after_fortran = array._get_packed_fortran_strides()
    after_c = array._get_packed_c_strides()
    assert after_fortran == (1, 3)
    assert after_c == (4, 1)


def test_set_shape_reshape_2d_to_3d():
    """Reshape 2D array to 3D and verify packed Fortran/C strides."""
    _, array = create_array((5, 10), None)

    before_fortran = array._get_packed_fortran_strides()
    before_c = array._get_packed_c_strides()
    assert before_fortran == (1, 5)
    assert before_c == (10, 1)

    array.set_shape((2, 5, 5))
    after_fortran = array._get_packed_fortran_strides()
    after_c = array._get_packed_c_strides()
    assert after_fortran == (1, 2, 10)
    assert after_c == (25, 5, 1)


def test_set_shape_reshape_square_to_rectangular():
    """Reshape square 2D array to rectangular 2D and check strides."""
    _, array = create_array((4, 4), None)

    before_fortran = array._get_packed_fortran_strides()
    before_c = array._get_packed_c_strides()
    assert before_fortran == (1, 4)
    assert before_c == (4, 1)

    array.set_shape((8, 2))
    after_fortran = array._get_packed_fortran_strides()
    after_c = array._get_packed_c_strides()
    assert after_fortran == (1, 8)
    assert after_c == (2, 1)


def test_set_shape_reshape_3d_to_1d():
    """Reshape 3D array to 1D and check strides."""
    _, array = create_array((2, 3, 4), None)

    before_fortran = array._get_packed_fortran_strides()
    before_c = array._get_packed_c_strides()
    assert before_fortran == (1, 2, 6)
    assert before_c == (12, 4, 1)

    array.set_shape((24, ))
    after_fortran = array._get_packed_fortran_strides()
    after_c = array._get_packed_c_strides()
    assert after_fortran == (1, )
    assert after_c == (1, )


def test_sdfg_save_and_reload_strides_tempfile():
    """Ensure packed strides remain consistent after SDFG save/load using tempfile."""
    sdfg, array = create_array((3, 4, 5), None)

    before_fortran = array._get_packed_fortran_strides()
    before_c = array._get_packed_c_strides()

    # Save SDFG to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".sdfg", delete=False) as tmp_file:
        sdfg_path = tmp_file.name
    try:
        sdfg.save(sdfg_path)

        # Load SDFG from file
        loaded_sdfg = dace.SDFG.from_file(sdfg_path)
        loaded_array = loaded_sdfg.arrays['test_array']

        after_fortran = loaded_array._get_packed_fortran_strides()
        after_c = loaded_array._get_packed_c_strides()

        # Strides should remain identical
        assert after_fortran == before_fortran
        assert after_c == before_c
    finally:
        os.remove(sdfg_path)


def test_get_packed_fortran_strides_2d_symbolic_N():
    """Test Fortran strides for 2D array with symbolic N"""
    _, array = create_array((10, N), None)  # N added to the second dimension
    expected = (1, 10)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_c_strides_3d_symbolic_N():
    """Test C strides for 3D array with symbolic N"""
    _, array = create_array((10, N, 30), None)  # N added to the second dimension
    expected = (30 * N, 30, 1)  # Stride calculation with symbolic N
    result = array._get_packed_c_strides()
    assert result == expected


def test_is_packed_fortran_strides_true_3d_symbolic_N():
    """Test 3D array with Fortran strides and symbolic N"""
    _, array = create_array((10, N, 30), (1, 10, 10 * N))  # Adjust stride for N
    result = array.is_packed_fortran_strides()
    assert result is True


def test_fortran_and_c_equivalent_for_1d_symbolic_N():
    """Test that 1D arrays have same strides for both layouts with symbolic N"""
    _, array = create_array((N, ), (1, ))
    assert array.is_packed_fortran_strides() is True
    assert array.is_packed_c_strides() is True


if __name__ == "__main__":
    tests = [obj for name, obj in globals().items() if callable(obj) and name.startswith("test_")]
    for test_function in tests:
        test_function()
