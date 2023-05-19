# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


def test_expand_if_python():

    def function_0(array_0, array_1, scalar):

        a = array_0 + 5
        b = array_1 - 5
        c = (a + b) / 2

        if scalar > 0:
            tmp_array_0 = array_0
            tmp_array_1 = array_1
        else:
            tmp_array_0 = array_1
            tmp_array_1 = array_0
        
        tmp_array_0[:] = c
        d = tmp_array_0 * 0.25
        e = tmp_array_1 * 0.75
        f = (d + e) / 2
        tmp_array_1[:] = f
    
    @dace.expand_if
    def function_1(array_0, array_1, scalar):

        a = array_0 + 5
        b = array_1 - 5
        c = (a + b) / 2

        if scalar > 0:
            tmp_array_0 = array_0
            tmp_array_1 = array_1
        else:
            tmp_array_0 = array_1
            tmp_array_1 = array_0
        
        tmp_array_0[:] = c
        d = tmp_array_0 * 0.25
        e = tmp_array_1 * 0.75
        f = (d + e) / 2
        tmp_array_1[:] = f
    
    rng = np.random.default_rng(42)
    A_ref = rng.random((10, ), dtype=np.float32)
    B_ref = rng.random((10, ), dtype=np.float32)
    A_val = A_ref.copy()
    B_val = B_ref.copy()
    scalar = rng.random(dtype=np.float32)

    function_0(A_ref, B_ref, scalar)
    function_1(A_val, B_val, scalar)

    assert np.allclose(A_ref, A_val)
    assert np.allclose(B_ref, B_val)


@pytest.mark.skip
def test_expand_if_dace():

    @dace.program(expand_if=True)
    def function(array_0, array_1, scalar):

        a = array_0 + 5
        b = array_1 - 5
        c = (a + b) / 2

        if scalar > 0:
            tmp_array_0 = array_0
            tmp_array_1 = array_1
        else:
            tmp_array_0 = array_1
            tmp_array_1 = array_0
        
        tmp_array_0[:] = c
        d = tmp_array_0 * 0.25
        e = tmp_array_1 * 0.75
        f = (d + e) / 2
        tmp_array_1[:] = f
    
    rng = np.random.default_rng(42)
    A_ref = rng.random((10, ), dtype=np.float32)
    B_ref = rng.random((10, ), dtype=np.float32)
    A_val = A_ref.copy()
    B_val = B_ref.copy()
    scalar = rng.random(dtype=np.float32)

    function.f(A_ref, B_ref, scalar)
    function(A_val, B_val, scalar)

    assert np.allclose(A_ref, A_val)
    assert np.allclose(B_ref, B_val)


def test_if_array_none():

    @dace.program(expand_if=True)
    def if_array_none(array, scalar, optional_array):
        if optional_array is not None:
            tmp_array = optional_array
        else:
            tmp_array = array
        tmp_array[:] = scalar
    
    rng = np.random.default_rng(42)
    A_ref = rng.random((10, ), dtype=np.float32)
    B_ref = rng.random((10, ), dtype=np.float32)
    A_val = A_ref.copy()
    B_val = B_ref.copy()
    scalar = rng.random(dtype=np.float32)

    if_array_none.f(A_ref, scalar, None)
    if_array_none(A_val, scalar, None)

    assert np.allclose(A_ref, A_val)

    if_array_none.f(A_ref, scalar, B_ref)
    if_array_none(A_val, scalar, B_val)

    assert np.allclose(A_ref, A_val)
    assert np.allclose(B_ref, B_val)


if __name__ == '__main__':
    test_expand_if_python()
    test_expand_if_dace()
    test_if_array_none()
