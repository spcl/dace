import dace
import numpy as np
import pytest
from common import compare_numpy_output

### Left, match first pos ######################################################


@compare_numpy_output
def test_subl1(A: dace.float64[5, 3], B: dace.float64[3]):
    return A - B


@compare_numpy_output
def test_multl1(A: dace.int64[5, 3], B: dace.int64[3]):
    return A * B


@compare_numpy_output
def test_bitorl1(A: dace.int64[5, 3], B: dace.int64[3]):
    return A | B


@compare_numpy_output
def test_bitxorl1(A: dace.int64[5, 3], B: dace.int64[3]):
    return A ^ B


@compare_numpy_output
def test_noteql1(A: dace.int64[5, 3], B: dace.int64[3]):
    return A != B


@compare_numpy_output
def test_ltl1(A: dace.int64[5, 3], B: dace.int64[3]):
    return A < B


### Right, match first pos #####################################################


@compare_numpy_output
def test_subr1(A: dace.float64[5], B: dace.float64[3, 5]):
    return A - B


@compare_numpy_output
def test_multr1(A: dace.int64[5], B: dace.int64[3, 5]):
    return A * B


@compare_numpy_output
def test_bitorr1(A: dace.int64[5], B: dace.int64[3, 5]):
    return A | B


@compare_numpy_output
def test_bitxorr1(A: dace.int64[5], B: dace.int64[3, 5]):
    return A ^ B


@compare_numpy_output
def test_noteqr1(A: dace.int64[5], B: dace.int64[3, 5]):
    return A != B


@compare_numpy_output
def test_ltr1(A: dace.int64[5], B: dace.int64[3, 5]):
    return A < B


### Left, first pos 1, match second pos ########################################


@compare_numpy_output
def test_subl2(A: dace.float64[5, 3], B: dace.float64[5, 1]):
    return A - B


@compare_numpy_output
def test_multl2(A: dace.int64[5, 3], B: dace.int64[5, 1]):
    return A * B


@compare_numpy_output
def test_bitorl2(A: dace.int64[5, 3], B: dace.int64[5, 1]):
    return A | B


@compare_numpy_output
def test_bitxorl2(A: dace.int64[5, 3], B: dace.int64[5, 1]):
    return A ^ B


@compare_numpy_output
def test_noteql2(A: dace.int64[5, 3], B: dace.int64[5, 1]):
    return A != B


@compare_numpy_output
def test_ltl2(A: dace.int64[5, 3], B: dace.int64[5, 1]):
    return A < B


### Right, first pos 1, match second ###########################################


@compare_numpy_output
def test_subr2(A: dace.float64[3, 1], B: dace.float64[3, 5]):
    return A - B


@compare_numpy_output
def test_multr2(A: dace.int64[3, 1], B: dace.int64[3, 5]):
    return A * B


@compare_numpy_output
def test_bitorr2(A: dace.int64[3, 1], B: dace.int64[3, 5]):
    return A | B


@compare_numpy_output
def test_bitxorr2(A: dace.int64[3, 1], B: dace.int64[3, 5]):
    return A ^ B


@compare_numpy_output
def test_noteqr2(A: dace.int64[3, 1], B: dace.int64[3, 5]):
    return A != B


@compare_numpy_output
def test_ltr2(A: dace.int64[3, 1], B: dace.int64[3, 5]):
    return A < B


### Left, first pos 1, match second pos, None last pos ########################


@compare_numpy_output
def test_subl3(A: dace.float64[5, 3], B: dace.float64[2, 5, 1]):
    return A - B


@compare_numpy_output
def test_bitxorl3(A: dace.int64[5, 3], B: dace.int64[2, 5, 1]):
    return A ^ B


@compare_numpy_output
def test_ltl3(A: dace.int64[5, 3], B: dace.int64[2, 5, 1]):
    return A < B


### Right, first pos 1, match second pos, None last pos #######################


@compare_numpy_output
def test_multr3(A: dace.int64[4, 3, 1], B: dace.int64[3, 5]):
    return A * B


@compare_numpy_output
def test_bitorr3(A: dace.int64[4, 3, 1], B: dace.int64[3, 5]):
    return A | B


@compare_numpy_output
def test_noteqr3(A: dace.int64[4, 3, 1], B: dace.int64[3, 5]):
    return A != B


### Left Errors ###############################################################


@compare_numpy_output
def test_subl4(A: dace.float64[5, 3], B: dace.float64[2]):
    return A - B


@compare_numpy_output
def test_bitxorl4(A: dace.int64[5, 3], B: dace.int64[2, 3]):
    return A ^ B


@compare_numpy_output
def test_ltl4(A: dace.int64[5, 3], B: dace.int64[3, 2, 3]):
    return A < B


### Right Errors ##############################################################


@compare_numpy_output
def test_multr4(A: dace.int64[4], B: dace.int64[3, 5]):
    return A * B


@compare_numpy_output
def test_bitorr4(A: dace.int64[4, 1], B: dace.int64[3, 5]):
    return A | B


# this is broken as of numpy 1.18: numpy doesn't raise an error
#
# >>> import numpy as np
# >>> a = np.random.rand(3, 2)
# >>> b = np.random.rand(2)
# >>> a == b # this works as expected
# array([[False, False],
#        [False, False],
#        [False, False]])
# >>> b = np.random.rand(3)
# >>> a == b # ?
# <stdin>:1: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
# False
#
# this test can be reenabled when this is fixed

#@compare_numpy_output
#def test_noteqr4(A: dace.int64[3, 3, 2], B: dace.int64[3, 5]):
#    return A != B


if __name__ == '__main__':
    import __main__ as main
    exit(pytest.main([main.__file__]))
