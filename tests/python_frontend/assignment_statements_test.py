# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.frontend.python.common import DaceSyntaxError


@dace.program
def single_target(a: dace.float32[1]):
    b = dace.ndarray((1, ), dtype=dace.float32)
    if (a[0] < 0):
        b = 0
    elif (a[0] < 1):
        b = 1
    else:
        b = a
    return b


def test_single_target():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b = single_target(a=a)
    assert (b[0] == np.float32(np.pi))


@dace.program
def single_target_parentheses(a: dace.float32[1]):
    (b) = a
    return b


def test_single_target_parentheses():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b = single_target_parentheses(a=a)
    assert (b[0] == np.float32(np.pi))


@dace.program
def multiple_targets(a: dace.float32[1]):
    b, c = a, 2 * a
    return b, c


def test_multiple_targets():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c = multiple_targets(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))


@dace.program
def multiple_targets_parentheses(a: dace.float32[1]):
    (b, c) = (a, 2 * a)
    return b, c


def test_multiple_targets_parentheses():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c = multiple_targets_parentheses(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))


@dace.program
def starred_target(a: dace.float32[1]):
    b, *c, d, e = a, 2 * a, 3 * a, 4 * a, 5 * a, 6 * a
    return b, c, d, e


@pytest.mark.skip
def test_starred_target():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c, d, e = starred_target(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))
    assert (c[1] == np.float32(3) * np.float32(np.pi))
    assert (c[2] == np.float32(4) * np.float32(np.pi))
    assert (d[0] == np.float32(5) * np.float32(np.pi))
    assert (e[0] == np.float32(6) * np.float32(np.pi))


mystruct = dace.struct('mystruct', a=dace.int32, b=dace.float32)


@dace.program
def attribute_reference(a: mystruct[1]):
    a.a[0] = 5
    a.b[0] = 6


@pytest.mark.skip
def test_attribute_reference():
    a = np.ndarray((1, ), dtype=np.dtype(mystruct.as_ctypes()))
    attribute_reference(a=a)
    assert (a[0]['a'] == np.int32(5))
    assert (a[0]['b'] == np.float32(6))


@dace.program
def ann_assign_supported_type():
    a: dace.uint16 = 5
    return a


def test_ann_assign_supported_type():
    a = ann_assign_supported_type()
    assert (a.dtype == np.uint16)


def test_assignment_to_nonexistent_variable():
    @dace.program
    def badprog(B: dace.float64):
        A[...] = B

    with pytest.raises(DaceSyntaxError):
        badprog.to_sdfg()


def test_assign_return_symbols():

    @dace.program
    def assign_symbols():
        a = 6
        for i in range(10):
            a = 5
        a -= 1
        return i, a
    
    result = assign_symbols()
    a = result[1][0]
    assert a == 4


if __name__ == "__main__":
    test_single_target()
    test_single_target_parentheses()
    test_multiple_targets()
    test_multiple_targets_parentheses()

    # test_starred_target()
    # test_attribute_reference()

    test_ann_assign_supported_type()
    test_assignment_to_nonexistent_variable()

    test_assign_return_symbols()
