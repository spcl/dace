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


def division_family(dtype):
    """``np.divmod``, ``//`` and ``%`` over whole arrays, and ``//`` and ``%`` element by element in a map."""

    @dace.program
    def family(a: dtype[N], b: dtype[N], scalar_floor: dtype[N], scalar_mod: dtype[N]):
        quotient, remainder = np.divmod(a, b)
        for i in dace.map[0:N]:
            scalar_floor[i] = a[i] // b[i]
            scalar_mod[i] = a[i] % b[i]
        return quotient, remainder, a // b, a % b

    return family


DIVISION_FAMILIES = {
    name: division_family(getattr(dace, name))
    for name in ('float64', 'float32', 'int32', 'int64', 'uint32', 'uint64')
}

INT32_MIN = int(np.iinfo(np.int32).min)
INT64_MIN = int(np.iinfo(np.int64).min)

DIVISION_ROWS = [
    pytest.param('float64', [1e300, -1e300, 1e300], [7.0, 7.0, -7.0], id='large-magnitude'),
    pytest.param('float32', [1e30, -1e30], [7.0, 7.0], id='large-magnitude-float32'),
    pytest.param('float64', [5.0, -5.0, 5.0, -5.0], [np.inf, np.inf, -np.inf, -np.inf], id='modulo-infinity'),
    pytest.param('float64', [1.0], [0.1], id='inexact-divisor'),
    pytest.param('float64', [1.0, -1.0, 1.0, -1.0, 7.0, -7.0, 7.0, -7.0], [0.1, 0.1, -0.1, -0.1, 3.0, 3.0, -3.0, -3.0],
                 id='sign-combinations'),
    pytest.param('float64', [0.0, -0.0, 0.0, -0.0, 6.0, -6.0], [3.0, 3.0, -3.0, -3.0, -3.0, 3.0], id='signed-zero'),
    pytest.param('float64', [np.nan, 5.0, np.inf, 5.0, -5.0, 0.0], [2.0, np.nan, 5.0, 0.0, -0.0, 0.0],
                 id='nan-and-zero-divisor'),
    pytest.param('int32', [7, -7, 7, -7], [3, 3, -3, -3], id='int32'),
    pytest.param('int64', [7, -7, 7, -7, 2**40], [3, 3, -3, -3, -3], id='int64'),
    pytest.param('int32', [2**31 - 1, 2**31 - 2], [3, 5], id='int32-dividend-near-max'),
    pytest.param('int64', [2**63 - 1, 2**63 - 2], [3, 5], id='int64-dividend-near-max'),
    pytest.param('uint32', [7, 0, 2**32 - 1], [3, 5, 2], id='uint32'),
    pytest.param('uint64', [7, 2**64 - 1], [3, 2], id='uint64'),
    pytest.param('int32', [7, -7, 0, INT32_MIN], [0, 0, 0, -1], id='int32-zero-divisor-and-overflow'),
    pytest.param('int64', [7, -7, 0, INT64_MIN], [0, 0, 0, -1], id='int64-zero-divisor-and-overflow'),
    pytest.param('uint32', [7, 0], [0, 0], id='uint32-zero-divisor'),
]


@pytest.mark.parametrize('dtype, numerators, denominators', DIVISION_ROWS)
def test_divmod_floor_divide_and_remainder_are_numpys_for_every_spelling(dtype, numerators, denominators):
    """``//`` and ``%`` are the two halves of ``np.divmod``; a value, a signed zero or a dtype that differs
    from numpy in any of the six results is a silent wrong answer. An integer zero divisor gives 0, as
    numpy does, instead of trapping."""
    a = np.array(numerators, dtype=dtype)
    b = np.array(denominators, dtype=dtype)
    scalar_floor = np.zeros_like(a)
    scalar_mod = np.zeros_like(a)
    quotient, remainder, floor_divide, modulo = DIVISION_FAMILIES[dtype](a.copy(),
                                                                         b.copy(),
                                                                         scalar_floor,
                                                                         scalar_mod,
                                                                         N=a.size)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        want_quotient, want_remainder = np.divmod(a, b)
        want_floor_divide = np.floor_divide(a, b)
        want_remainder_ufunc = np.remainder(a, b)
    results = {
        'np.divmod quotient': (quotient, want_quotient),
        'np.divmod remainder': (remainder, want_remainder),
        'a // b': (floor_divide, want_floor_divide),
        'a % b': (modulo, want_remainder_ufunc),
        'a[i] // b[i]': (scalar_floor, want_floor_divide),
        'a[i] % b[i]': (scalar_mod, want_remainder_ufunc),
    }
    for spelling, (got, want) in results.items():
        assert got.dtype == want.dtype, (spelling, got.dtype, want.dtype)
        same = identical(got, want)
        assert same.all(), (f'{spelling}: for {a[~same]} and {b[~same]} dace gives {got[~same]}, '
                            f'numpy {want[~same]}')
