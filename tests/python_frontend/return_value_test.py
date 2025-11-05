# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_return_scalar():

    @dace.program
    def return_scalar():
        return 5

    res = return_scalar()
    assert res == 5

    # Don't be fooled by the test above the return value is an array. If you would
    #  add the return value annotation to the program, i.e. `-> dace.int32` you would
    #  get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int64


def test_return_scalar_in_nested_function():

    @dace.program
    def nested_function() -> dace.int32:
        return 5

    @dace.program
    def return_scalar():
        return nested_function()

    res = return_scalar()
    assert res == 5

    # Don't be fooled by the test above the return value is an array. If you would
    #  add the return value annotation to the program, i.e. `-> dace.int32` you would
    #  get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int32


def test_return_array():

    @dace.program
    def return_array():
        return 5 * np.ones(5)

    res = return_array()
    assert np.allclose(res, 5 * np.ones(5))


def test_return_tuple():

    @dace.program
    def return_tuple():
        return 5, 6

    res = return_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res == (5, 6)


def test_return_array_tuple():

    @dace.program
    def return_array_tuple():
        return 5 * np.ones(5), 6 * np.ones(6)

    res = return_array_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert np.allclose(res[0], 5 * np.ones(5))
    assert np.allclose(res[1], 6 * np.ones(6))


def test_return_void():

    @dace.program
    def return_void(a: dace.float64[20]):
        a[:] += 1
        return
        a[:] = 5

    a = np.random.rand(20)
    ref = a + 1
    res = return_void(a)
    assert res is None
    assert np.allclose(a, ref)


def test_return_tuple_1_element():

    @dace.program
    def return_one_element_tuple(a: dace.float64[20]):
        return (a + 3.5, )

    a = np.random.rand(20)
    ref = a + 3.5
    res = return_one_element_tuple(a)
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert np.allclose(res[0], ref)


def test_return_void_in_if():

    @dace.program
    def return_void(a: dace.float64[20]):
        if a[0] < 0:
            return
        a[:] = 5

    a = np.random.rand(20)
    return_void(a)
    assert np.allclose(a, 5)
    a[:] = np.random.rand(20)
    a[0] = -1
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


def test_return_void_in_for():

    @dace.program
    def return_void(a: dace.float64[20]):
        for _ in range(20):
            return
        a[:] = 5

    a = np.random.rand(20)
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


if __name__ == '__main__':
    test_return_scalar()
    test_return_scalar_in_nested_function()
    test_return_array()
    test_return_tuple()
    test_return_array_tuple()
    test_return_void()
    test_return_void_in_if()
    test_return_void_in_for()
