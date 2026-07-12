# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace


@dace.program
def cpp_float(A: dace.float32[1], B: dace.float32[2]):

    @dace.tasklet('CPP')
    def asin():
        a << A[0]
        b0 >> B[0]
        b1 >> B[1]
        """
        b0 = asin(a);       // from <math.h> (only double precision)
        b1 = std::asin(a);  // from <cmath>
        """


@dace.program
def cpp_double(A: dace.float64[1], B: dace.float64[2]):

    @dace.tasklet('CPP')
    def asin():
        a << A[0]
        b0 >> B[0]
        b1 >> B[1]
        """
        b0 = asin(a);       // from <math.h> (only double precision)
        b1 = std::asin(a);  // from <cmath>
        """


@dace.program
def dace_32(A: dace.float32[1], B: dace.float32[2]):

    @dace.tasklet
    def asin():
        a << A[0]
        b0 = asin(a)
        b1 = dace.math.asin(a)
        b0 >> B[0]
        b1 >> B[1]


@dace.program
def dace_64(A: dace.float64[1], B: dace.float64[2]):

    @dace.tasklet
    def asin():
        a << A[0]
        b0 = asin(a)
        b1 = dace.math.asin(a)
        b0 >> B[0]
        b1 >> B[1]


def test_math_precision():
    in_32 = dace.ndarray((1, ), dace.float32)
    in_64 = dace.ndarray((1, ), dace.float64)
    in_32[:] = [0.5]
    in_64[:] = [0.5]

    cpp_out_32 = dace.ndarray((2, ), dace.float32)
    cpp_out_64 = dace.ndarray((2, ), dace.float64)
    cpp_float(in_32, cpp_out_32)
    cpp_double(in_64, cpp_out_64)

    dace_out_32 = dace.ndarray((2, ), dace.float32)
    dace_out_64 = dace.ndarray((2, ), dace.float64)
    dace_32(in_32, dace_out_32)
    dace_64(in_64, dace_out_64)

    # Assert single & double precision version don't return the same response.
    assert (dace_out_32 != dace_out_64).all()

    # Assert single & double precision versions match the cpp baseline.
    assert (cpp_out_32 == dace_out_32).all()
    assert (cpp_out_64 == dace_out_64).all()


if __name__ == "__main__":
    test_math_precision()
