# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

N = dace.symbol('N')

SPECIAL_VALUES = np.array([-2.5, -0.75, -0.0, 0.0, 0.75, 3.0, -3.0, np.inf, -np.inf, np.nan, 1e30, 1e-45, 2.0**23 + 1])
INTEGERS = np.array([-7, -1, 0, 3, 2**40], dtype=np.int64)


@dace.program
def modf_float64(x: dace.float64[N]):
    fractional, integral = np.modf(x)
    return fractional, integral


@dace.program
def modf_float32(x: dace.float32[N]):
    fractional, integral = np.modf(x)
    return fractional, integral


@dace.program
def modf_int64(x: dace.int64[N]):
    fractional, integral = np.modf(x)
    return fractional, integral


@dace.program
def frexp_float64(x: dace.float64[N]):
    mantissa, exponent = np.frexp(x)
    return mantissa, exponent


@dace.program
def frexp_float32(x: dace.float32[N]):
    mantissa, exponent = np.frexp(x)
    return mantissa, exponent


def identical(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Elementwise equality that also tells -0.0 from 0.0 and counts NaN as equal to NaN."""
    return (np.isnan(got) & np.isnan(want)) | ((got == want) & (np.signbit(got) == np.signbit(want)))


@pytest.mark.parametrize('program, ufunc, x', [
    pytest.param(modf_float64, np.modf, SPECIAL_VALUES, id='modf-float64'),
    pytest.param(modf_float32, np.modf, SPECIAL_VALUES.astype(np.float32), id='modf-float32'),
    pytest.param(modf_int64, np.modf, INTEGERS, id='modf-int64'),
    pytest.param(frexp_float64, np.frexp, SPECIAL_VALUES, id='frexp-float64'),
    pytest.param(frexp_float32, np.frexp, SPECIAL_VALUES.astype(np.float32), id='frexp-float32'),
])
def test_each_output_of_a_multi_output_ufunc_is_numpys_output_at_that_position(program, ufunc, x):
    """A swapped pair of outputs still has the right dtypes and shapes, so only the values can tell."""
    got = program(x.copy(), N=x.size)
    want = ufunc(x)
    assert len(got) == len(want)
    for position, (got_part, want_part) in enumerate(zip(got, want)):
        assert got_part.dtype == want_part.dtype, (position, got_part.dtype, want_part.dtype)
        same = identical(got_part, want_part)
        assert same.all(), (f'output {position}: for {x[~same]} dace gives {got_part[~same]}, '
                            f'numpy {want_part[~same]}')
