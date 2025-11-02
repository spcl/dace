import dace
import pytest


# Helper function
def create_array(shape, strides):
    sdfg = dace.SDFG('test_sdfg')
    _, array = sdfg.add_array(name='test_array', shape=shape, dtype=dace.float64, strides=strides)
    return array


def test_get_packed_fortran_strides_1d():
    """Test Fortran strides for 1D array"""
    array = create_array((10, ), None)
    expected = (1, )
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_2d():
    """Test Fortran strides for 2D array"""
    array = create_array((10, 20), None)
    expected = (1, 10)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_3d():
    """Test Fortran strides for 3D array"""
    array = create_array((10, 20, 30), None)
    expected = (1, 10, 200)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_fortran_strides_4d():
    """Test Fortran strides for 4D array"""
    array = create_array((5, 10, 15, 20), None)
    expected = (1, 5, 50, 750)
    result = array._get_packed_fortran_strides()
    assert result == expected


def test_get_packed_c_strides_1d():
    """Test C strides for 1D array"""
    array = create_array((10, ), None)
    expected = (1, )
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_2d():
    """Test C strides for 2D array"""
    array = create_array((10, 20), None)
    expected = (20, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_3d():
    """Test C strides for 3D array"""
    array = create_array((10, 20, 30), None)
    expected = (600, 30, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_get_packed_c_strides_4d():
    """Test C strides for 4D array"""
    array = create_array((5, 10, 15, 20), None)
    expected = (3000, 300, 20, 1)
    result = array._get_packed_c_strides()
    assert result == expected


def test_is_packed_fortran_strides_true_3d():
    """Test 3D array with Fortran strides"""
    array = create_array((10, 20, 30), (1, 10, 200))
    result = array.is_packed_fortran_strides()
    assert result is True


def test_is_packed_fortran_strides_false_c_layout():
    """Test array with C strides returns False"""
    array = create_array((10, 20, 30), (600, 30, 1))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_fortran_strides_false_custom_strides():
    """Test array with custom strides returns False"""
    array = create_array((10, 20, 30), (2, 20, 400))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_fortran_strides_false_wrong_order():
    """Test array with incorrect stride ordering"""
    array = create_array((10, 20, 30), (10, 1, 200))
    result = array.is_packed_fortran_strides()
    assert result is False


def test_is_packed_c_strides_true_1d():
    """Test 1D array with C strides"""
    array = create_array((10, ), (1, ))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_true_2d():
    """Test 2D array with C strides"""
    array = create_array((10, 20), (20, 1))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_true_3d():
    """Test 3D array with C strides"""
    array = create_array((10, 20, 30), (600, 30, 1))
    result = array.is_packed_c_strides()
    assert result is True


def test_is_packed_c_strides_false_fortran_layout():
    """Test array with Fortran strides returns False"""
    array = create_array((10, 20, 30), (1, 10, 200))
    result = array.is_packed_c_strides()
    assert result is False


def test_is_packed_c_strides_false_custom_strides():
    """Test array with custom strides returns False"""
    array = create_array((10, 20, 30), (1200, 60, 2))
    result = array.is_packed_c_strides()
    assert result is False


def test_is_packed_c_strides_false_wrong_order():
    """Test array with incorrect stride ordering"""
    array = create_array((10, 20, 30), (1, 30, 600))
    result = array.is_packed_c_strides()
    assert result is False


def test_empty_shape():
    """Test with empty shape (scalar)"""
    array = create_array((), ())
    fortran_result = array._get_packed_fortran_strides()
    c_result = array._get_packed_c_strides()
    assert fortran_result == ()
    assert c_result == ()


def test_fortran_and_c_equivalent_for_1d():
    """Test that 1D arrays have same strides for both layouts"""
    array = create_array((100, ), (1, ))
    assert array.is_packed_fortran_strides() is True
    assert array.is_packed_c_strides() is True


def test_c_strides_calculation_accumulation():
    """Test stride accumulation for C layout"""
    array = create_array((2, 3, 4), None)
    result = array._get_packed_c_strides()
    assert result == (12, 4, 1)


def test_explicit_fortran_strides():
    """Test explicitly set Fortran strides"""
    array = create_array((5, 7, 9), (1, 5, 35))
    assert array.is_packed_fortran_strides() is True
    assert array.is_packed_c_strides() is False


def test_explicit_fortran_strides_not_packed():
    """Test explicitly set Fortran strides"""
    array = create_array((5, 7, 9), (10, 5 * 10, 35 * 10))
    assert array.is_packed_fortran_strides() is False
    assert array.is_packed_c_strides() is False


def test_explicit_c_strides():
    """Test explicitly set C strides"""
    array = create_array((5, 7, 9), (63, 9, 1))
    assert array.is_packed_c_strides() is True
    assert array.is_packed_fortran_strides() is False


def test_explicit_c_strides_not_packed():
    """Test explicitly set C strides"""
    array = create_array((5, 7, 9), (63 * 5, 9 * 5, 1))
    assert array.is_packed_c_strides() is False
    assert array.is_packed_fortran_strides() is False


if __name__ == "__main__":

    all_tests = [
        test_get_packed_fortran_strides_1d,
        test_get_packed_fortran_strides_2d,
        test_get_packed_fortran_strides_3d,
        test_get_packed_fortran_strides_4d,
        test_get_packed_c_strides_1d,
        test_get_packed_c_strides_2d,
        test_get_packed_c_strides_3d,
        test_get_packed_c_strides_4d,
        test_is_packed_fortran_strides_true_3d,
        test_is_packed_fortran_strides_false_c_layout,
        test_is_packed_fortran_strides_false_custom_strides,
        test_is_packed_fortran_strides_false_wrong_order,
        test_is_packed_c_strides_true_1d,
        test_is_packed_c_strides_true_2d,
        test_is_packed_c_strides_true_3d,
        test_is_packed_c_strides_false_fortran_layout,
        test_is_packed_c_strides_false_custom_strides,
        test_is_packed_c_strides_false_wrong_order,
        test_empty_shape,
        test_fortran_and_c_equivalent_for_1d,
        test_c_strides_calculation_accumulation,
        test_explicit_fortran_strides,
        test_explicit_fortran_strides_not_packed,
        test_explicit_c_strides,
        test_explicit_c_strides_not_packed,
    ]

    results = []
    for test in all_tests:
        test()
