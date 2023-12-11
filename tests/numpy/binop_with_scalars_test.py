# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from common import compare_numpy_output

### Left #####################################################################


@compare_numpy_output()
def test_addl(A: dace.int64[5, 5], B: dace.int64):
    return A + B


@compare_numpy_output()
def test_subl(A: dace.int64[5, 5], B: dace.int64):
    return A - B


@compare_numpy_output()
def test_multl(A: dace.int64[5, 5], B: dace.int64):
    return A * B


@compare_numpy_output()
def test_divl(A: dace.float64[5, 5], B: dace.float64):
    return A / B


# A // B is not implemented correctly in dace for negative numbers
@compare_numpy_output(non_zero=True, positive=True)
def test_floordivl(A: dace.int64[5, 5], B: dace.int64):
    return A // B


# A % B is not implemented correctly in dace for negative numbers
@compare_numpy_output(non_zero=True, positive=True)
def test_modl(A: dace.int64[5, 5], B: dace.int64):
    return A % B


# numpy throws an error for negative B, dace doesn't
@compare_numpy_output(positive=True, casting=np.float64)
def test_powl(A: dace.int64[5, 5], B: dace.int64):
    return A**B


# dace has weird behavior here too
@compare_numpy_output(positive=True, max_value=10)
def test_lshiftl(A: dace.int64[5, 5], B: dace.int64):
    return A << B


@compare_numpy_output(positive=True, max_value=10)
def test_rshiftl(A: dace.int64[5, 5], B: dace.int64):
    return A >> B


@compare_numpy_output()
def test_bitorl(A: dace.int64[5, 5], B: dace.int64):
    return A | B


@compare_numpy_output()
def test_bitxorl(A: dace.int64[5, 5], B: dace.int64):
    return A ^ B


@compare_numpy_output()
def test_bitandl(A: dace.int64[5, 5], B: dace.int64):
    return A & B


@compare_numpy_output()
def test_eql(A: dace.int64[5, 5], B: dace.int64):
    return A == B


@compare_numpy_output()
def test_noteql(A: dace.int64[5, 5], B: dace.int64):
    return A != B


@compare_numpy_output()
def test_ltl(A: dace.int64[5, 5], B: dace.int64):
    return A < B


@compare_numpy_output()
def test_ltel(A: dace.int64[5, 5], B: dace.int64):
    return A <= B


@compare_numpy_output()
def test_gtl(A: dace.int64[5, 5], B: dace.int64):
    return A > B


@compare_numpy_output()
def test_gtel(A: dace.int64[5, 5], B: dace.int64):
    return A >= B


### Right #####################################################################


@compare_numpy_output()
def test_addr(A: dace.int64, B: dace.int64[5, 5]):
    return A + B


@compare_numpy_output()
def test_subr(A: dace.int64, B: dace.int64[5, 5]):
    return A - B


@compare_numpy_output()
def test_multr(A: dace.int64, B: dace.int64[5, 5]):
    return A * B


@compare_numpy_output()
def test_divr(A: dace.float64, B: dace.float64[5, 5]):
    return A / B


@compare_numpy_output(non_zero=True, positive=True)
def test_floordivr(A: dace.int64, B: dace.int64[5, 5]):
    return A // B


@compare_numpy_output(non_zero=True, positive=True)
def test_modr(A: dace.int64, B: dace.int64[5, 5]):
    return A % B


@compare_numpy_output(positive=True, casting=np.float64)
def test_powr(A: dace.int64, B: dace.int64[5, 5]):
    return A**B


@compare_numpy_output(positive=True, max_value=10)
def test_lshiftr(A: dace.int64, B: dace.int64[5, 5]):
    return A << B


@compare_numpy_output(positive=True, max_value=10)
def test_rshiftr(A: dace.int64, B: dace.int64[5, 5]):
    return A >> B


@compare_numpy_output()
def test_bitorr(A: dace.int64, B: dace.int64[5, 5]):
    return A | B


@compare_numpy_output()
def test_bitxorr(A: dace.int64, B: dace.int64[5, 5]):
    return A ^ B


@compare_numpy_output()
def test_bitandr(A: dace.int64, B: dace.int64[5, 5]):
    return A & B


@compare_numpy_output()
def test_eqr(A: dace.int64, B: dace.int64[5, 5]):
    return A == B


@compare_numpy_output()
def test_noteqr(A: dace.int64, B: dace.int64[5, 5]):
    return A != B


@compare_numpy_output()
def test_ltr(A: dace.int64, B: dace.int64[5, 5]):
    return A < B


@compare_numpy_output()
def test_lter(A: dace.int64, B: dace.int64[5, 5]):
    return A <= B


@compare_numpy_output()
def test_gtr(A: dace.int64, B: dace.int64[5, 5]):
    return A > B


@compare_numpy_output()
def test_gter(A: dace.int64, B: dace.int64[5, 5]):
    return A >= B


if __name__ == '__main__':
    # generate this with
    # cat binop_with_scalars_test.py | grep -oP '(?<=f ).*(?=\()' | awk '{print $0 "()"}'
    test_addl()
    test_subl()
    test_multl()
    test_divl()
    test_floordivl()
    test_modl()
    test_powl()
    test_lshiftl()
    test_rshiftl()
    test_bitorl()
    test_bitxorl()
    test_bitandl()
    test_eql()
    test_noteql()
    test_ltl()
    test_ltel()
    test_gtl()
    test_gtel()
    test_addr()
    test_subr()
    test_multr()
    test_divr()
    test_floordivr()
    test_modr()
    test_powr()
    test_lshiftr()
    test_rshiftr()
    test_bitorr()
    test_bitxorr()
    test_bitandr()
    test_eqr()
    test_noteqr()
    test_ltr()
    test_lter()
    test_gtr()
    test_gter()
