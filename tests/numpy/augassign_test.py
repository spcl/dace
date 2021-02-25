# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


@compare_numpy_output()
def test_augadd(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B += A
    return B


@compare_numpy_output()
def test_augsub(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B -= A
    return B


@compare_numpy_output()
def test_augmult(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B *= A
    return B


@compare_numpy_output(non_zero=True, positive=True)
def test_augdiv(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    B /= A
    return B


@compare_numpy_output(non_zero=True, positive=True)
def test_augfloordiv(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B //= A
    return B


@compare_numpy_output(non_zero=True, positive=True)
def test_augmod(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B %= A
    return B


@compare_numpy_output(positive=True)
def test_augpow(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B **= A
    return B


@compare_numpy_output(positive=True)
def test_auglshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B <<= A
    return B


@compare_numpy_output(positive=True)
def test_augrshift(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B >>= A
    return B


@compare_numpy_output()
def test_augbitor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B |= A
    return B


@compare_numpy_output()
def test_augbitxor(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B ^= A
    return B


@compare_numpy_output()
def test_augbitand(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B &= A
    return B


if __name__ == '__main__':
    # Generate with cat augassign_test.py | grep -oP '(?<=f ).*(?=\()' | awk '{print $0 "()"}'
    test_augadd()
    test_augsub()
    test_augmult()
    test_augdiv()
    test_augfloordiv()
    test_augmod()
    test_augpow()
    test_auglshift()
    test_augrshift()
    test_augbitor()
    test_augbitxor()
    test_augbitand()
