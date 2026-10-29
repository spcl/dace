# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace

N = dace.symbol("N")


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
    O_BIG: dace.float16[1],
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
        o_big >> O_BIG[0]
        """
        static_assert(std::is_same<decltype(dace::math::pow(h, 1, 2)), dace::float16>::value, "half stays half");
        static_assert(std::is_same<decltype(dace::math::pow(f, 1, 3)), float>::value, "float stays float");
        static_assert(std::is_same<decltype(dace::math::pow(d, -2, 3)), double>::value, "double stays double");
        static_assert(std::is_same<decltype(dace::math::pow(i, 1, 3)), double>::value, "int becomes double");
        oh = dace::math::pow(h, 1, 2);
        of = dace::math::pow(f, 1, 3);
        od = dace::math::pow(d, -2, 3);
        oi = dace::math::pow(i, 1, 3);
        o_big = dace::math::pow(h, 100000, 100000);
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
    h, f, d, i = (
        np.array([4.0], np.float16),
        np.array([8.0], np.float32),
        np.array([8.0], np.float64),
        np.array([8], np.int32),
    )
    oh, of, od, oi = np.zeros(1, np.float16), np.zeros(1, np.float32), np.zeros(1, np.float64), np.zeros(1, np.float64)
    o_big = np.zeros(1, np.float16)
    cpp_fractional_pow(h, f, d, i, oh, of, od, oi, o_big)
    assert (oh[0], of[0], od[0], oi[0]) == (2.0, 2.0, 0.25, 2.0)
    # p and q overflow a half: the exponent must be divided in float
    assert o_big[0] == h[0]


@dace.program
def complex_math(a: dace.complex128[N], out: dace.complex128[16, N]):
    out[0] = np.exp(a)
    out[1] = np.log(a)
    out[2] = np.sqrt(a)
    for i in dace.map[0:N]:
        with dace.tasklet:
            z << a[i]
            o3 = dace.math.log10(z)
            o4 = dace.math.sin(z)
            o5 = dace.math.cos(z)
            o6 = dace.math.tan(z)
            o7 = dace.math.sinh(z)
            o8 = dace.math.cosh(z)
            o9 = dace.math.tanh(z)
            o10 = dace.math.asin(z)
            o11 = dace.math.acos(z)
            o12 = dace.math.atan(z)
            o13 = dace.math.asinh(z)
            o14 = dace.math.acosh(z)
            o15 = dace.math.atanh(z)
            o3 >> out[3, i]
            o4 >> out[4, i]
            o5 >> out[5, i]
            o6 >> out[6, i]
            o7 >> out[7, i]
            o8 >> out[8, i]
            o9 >> out[9, i]
            o10 >> out[10, i]
            o11 >> out[11, i]
            o12 >> out[12, i]
            o13 >> out[13, i]
            o14 >> out[14, i]
            o15 >> out[15, i]


COMPLEX_MATH_ORACLES = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
}


@pytest.mark.gpu
def test_complex_math_functions_on_the_gpu_match_numpy():
    """``dace::math`` forwards to ``std::``, which has no ``thrust::complex`` overloads."""
    sdfg = complex_math.to_sdfg()
    sdfg.apply_gpu_transformations()
    cuda = "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")
    missing = [name for name in COMPLEX_MATH_ORACLES if f"dace::math::{name}(" not in cuda]
    assert not missing, f"the device code does not call dace::math for {missing}"

    n = 32
    rng = np.random.default_rng(20260914)
    # Imaginary parts in (0.2, 0.9) keep every input off the real- and imaginary-axis branch cuts.
    a = rng.uniform(-0.9, 0.9, size=n) + 1j * rng.uniform(0.2, 0.9, size=n)
    out = np.zeros((16, n), dtype=np.complex128)
    sdfg(a=a, out=out, N=n)
    expected = np.stack([oracle(a) for oracle in COMPLEX_MATH_ORACLES.values()])
    np.testing.assert_allclose(out, expected, rtol=1e-12)


if __name__ == "__main__":
    test_math_precision()
    test_fractional_power_keeps_the_floating_type_of_the_base()
