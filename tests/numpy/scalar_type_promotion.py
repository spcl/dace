"""
See https://github.com/numpy/numpy/issues/13591 for some wacky examples
"""
import dace

from common import compare_numpy_output


@compare_numpy_output()
def int32_plus_one_test(A: dace.int32[5, 5]):
    return A + 1


@compare_numpy_output()
def one_plus_int32_test(A: dace.int32[5, 5]):
    return 1 + A


@compare_numpy_output()
def float_plus_int32_test(A: dace.int32[5, 5]):
    return 1.0 + A


@compare_numpy_output()
def float_plus_int8_test(A: dace.int8[5, 5]):
    return 0.0 + A


if __name__ == "__main__":
    int32_plus_one_test()
    one_plus_int32_test()
    float_plus_int32_test()
    float_plus_int8_test()
