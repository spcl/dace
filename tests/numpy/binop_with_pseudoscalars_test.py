import dace
import numpy as np
import pytest
from common import compare_numpy_output

### Left #####################################################################


@compare_numpy_output
def test_addl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A + B


@compare_numpy_output
def test_subl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A - B


@compare_numpy_output
def test_multl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A * B


@compare_numpy_output
def test_divl(A: dace.float64[5, 5], B: dace.float64[1]):
    return A / B


@compare_numpy_output
def test_floordivl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A // B


@compare_numpy_output
def test_modl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A % B


@compare_numpy_output
def test_powl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A**B


@compare_numpy_output
def test_lshiftl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A << B


@compare_numpy_output
def test_rshiftl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A >> B


@compare_numpy_output
def test_bitorl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A | B


@compare_numpy_output
def test_bitxorl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A ^ B


@compare_numpy_output
def test_bitandl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A & B


@compare_numpy_output
def test_eql(A: dace.int64[5, 5], B: dace.int64[1]):
    return A == B


@compare_numpy_output
def test_noteql(A: dace.int64[5, 5], B: dace.int64[1]):
    return A != B


@compare_numpy_output
def test_ltl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A < B


@compare_numpy_output
def test_ltel(A: dace.int64[5, 5], B: dace.int64[1]):
    return A <= B


@compare_numpy_output
def test_gtl(A: dace.int64[5, 5], B: dace.int64[1]):
    return A > B


@compare_numpy_output
def test_gtel(A: dace.int64[5, 5], B: dace.int64[1]):
    return A >= B


### Right #####################################################################


@compare_numpy_output
def test_addr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A + B


@compare_numpy_output
def test_subr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A - B


@compare_numpy_output
def test_multr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A * B


@compare_numpy_output
def test_divr(A: dace.float64[1], B: dace.float64[5, 5]):
    return A / B


@compare_numpy_output
def test_floordivr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A // B


@compare_numpy_output
def test_modr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A % B


@compare_numpy_output
def test_powr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A**B


@compare_numpy_output
def test_lshiftr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A << B


@compare_numpy_output
def test_rshiftr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A >> B


@compare_numpy_output
def test_bitorr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A | B


@compare_numpy_output
def test_bitxorr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A ^ B


@compare_numpy_output
def test_bitandr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A & B


@compare_numpy_output
def test_eqr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A == B


@compare_numpy_output
def test_noteqr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A != B


@compare_numpy_output
def test_ltr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A < B


@compare_numpy_output
def test_lter(A: dace.int64[1], B: dace.int64[5, 5]):
    return A <= B


@compare_numpy_output
def test_gtr(A: dace.int64[1], B: dace.int64[5, 5]):
    return A > B


@compare_numpy_output
def test_gter(A: dace.int64[1], B: dace.int64[5, 5]):
    return A >= B


if __name__ == '__main__':
    import __main__ as main
    exit(pytest.main([main.__file__]))
