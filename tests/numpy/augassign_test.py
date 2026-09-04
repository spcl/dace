# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
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


@compare_numpy_output()
def test_augbitand_bool(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    B &= A
    return B


def augbitand(A, B):
    B &= A


def augbitor(A, B):
    B |= A


def augbitxor(A, B):
    B ^= A


def auglshift(A, B):
    B <<= A


def augrshift(A, B):
    B >>= A


BITWISE_AUGASSIGNS = (augbitand, augbitor, augbitxor, auglshift, augrshift)


# The augassign path builds its tasklet directly instead of going through the operator
# replacements, so it used to accept these and only fail in the C++ compiler.
@pytest.mark.parametrize('kernel', BITWISE_AUGASSIGNS, ids=lambda k: k.__name__)
def test_bitwise_augassign_float_rejected(kernel):
    with pytest.raises(TypeError):
        dace.program(kernel).to_sdfg(simplify=False, A=np.ones((5, 5), np.float64), B=np.ones((5, 5), np.float64))


@pytest.mark.parametrize('kernel', BITWISE_AUGASSIGNS, ids=lambda k: k.__name__)
def test_bitwise_augassign_signed_uint64_rejected(kernel):
    with pytest.raises(TypeError):
        dace.program(kernel).to_sdfg(simplify=False, A=np.ones((5, 5), np.uint64), B=np.ones((5, 5), np.int32))


def auglshift_by_literal(A):
    A <<= 2


def test_bitwise_augassign_float_literal_rejected():
    with pytest.raises(TypeError):
        dace.program(auglshift_by_literal).to_sdfg(simplify=False, A=np.ones((5, 5), np.float64))


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
    test_augbitand_bool()
    for k in BITWISE_AUGASSIGNS:
        test_bitwise_augassign_float_rejected(k)
        test_bitwise_augassign_signed_uint64_rejected(k)
    test_bitwise_augassign_float_literal_rejected()
