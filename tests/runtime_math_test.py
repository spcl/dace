# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace


@dace.program
def cpp_float(A: dace.float32[1], B: dace.float32[2]):

    @dace.tasklet("CPP")
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

    @dace.tasklet("CPP")
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


@dace.program
def cpp_fractional_pow(
    H: dace.float16[1],
    F: dace.float32[1],
    D: dace.float64[1],
    I: dace.int32[1],
    OH: dace.float16[1],
    OF: dace.float32[1],
    OD: dace.float64[1],
    OI: dace.float64[1],
    OQ: dace.float16[1],
):

    @dace.tasklet("CPP")
    def fractional_pow():
        h << H[0]
        f << F[0]
        d << D[0]
        i << I[0]
        oh >> OH[0]
        of >> OF[0]
        od >> OD[0]
        oi >> OI[0]
        oq >> OQ[0]
        """
        static_assert(std::is_same<decltype(dace::math::pow(h, 1, 2)), dace::float16>::value, "half stays half");
        static_assert(std::is_same<decltype(dace::math::pow(f, 1, 3)), float>::value, "float stays float");
        static_assert(std::is_same<decltype(dace::math::pow(d, -2, 3)), double>::value, "double stays double");
        static_assert(std::is_same<decltype(dace::math::pow(i, 1, 3)), double>::value, "int becomes double");
        oh = dace::math::pow(h, 1, 2);
        of = dace::math::pow(f, 1, 3);
        od = dace::math::pow(d, -2, 3);
        oi = dace::math::pow(i, 1, 3);
        oq = dace::math::pow(h, 100000, 100000);
        """


def test_math_precision():
    in_32 = dace.ndarray((1,), dace.float32)
    in_64 = dace.ndarray((1,), dace.float64)
    in_32[:] = [0.5]
    in_64[:] = [0.5]

    cpp_out_32 = dace.ndarray((2,), dace.float32)
    cpp_out_64 = dace.ndarray((2,), dace.float64)
    cpp_float(in_32, cpp_out_32)
    cpp_double(in_64, cpp_out_64)

    dace_out_32 = dace.ndarray((2,), dace.float32)
    dace_out_64 = dace.ndarray((2,), dace.float64)
    dace_32(in_32, dace_out_32)
    dace_64(in_64, dace_out_64)

    # Assert single & double precision version don't return the same response.
    assert (dace_out_32 != dace_out_64).all()

    # Assert single & double precision versions match the cpp baseline.
    assert (cpp_out_32 == dace_out_32).all()
    assert (cpp_out_64 == dace_out_64).all()


def test_fractional_power_keeps_the_floating_type_of_the_base():
    import numpy as np

    h, f, d, i = (
        np.array([4.0], np.float16),
        np.array([8.0], np.float32),
        np.array([8.0], np.float64),
        np.array([8], np.int32),
    )
    oh, of, od, oi = np.zeros(1, np.float16), np.zeros(1, np.float32), np.zeros(1, np.float64), np.zeros(1, np.float64)
    oq = np.zeros(1, np.float16)
    cpp_fractional_pow(h, f, d, i, oh, of, od, oi, oq)
    assert (oh[0], of[0], od[0], oi[0]) == (2.0, 2.0, 0.25, 2.0)
    assert oq[0] == h[0]


if __name__ == "__main__":
    test_math_precision()
    test_fractional_power_keeps_the_floating_type_of_the_base()
