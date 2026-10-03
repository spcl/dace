# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generated C++ combining a complex value with a scalar of another type, checked elementwise against numpy.

``std::complex<T>`` only takes a ``T`` scalar, so a ``double`` against a ``complex<float>`` fell through to the runtime's
``int`` overload and scaled by zero.
"""
from typing import Callable

import numpy as np
import pytest

import dace

COMPLEX_TYPES = [
    pytest.param(np.complex64, dace.complex64, id='complex64'),
    pytest.param(np.complex128, dace.complex128, id='complex128'),
]
SCALARS = [
    pytest.param(np.float32(1.0 / 30.0), dace.float32, id='float32'),
    pytest.param(np.float64(1.0 / 30.0), dace.float64, id='float64'),
    pytest.param(np.int32(-7), dace.int32, id='int32'),
    pytest.param(np.int64(3000000000), dace.int64, id='int64_beyond_int'),
]
BINARY_OPERATIONS = [
    pytest.param('c * s', lambda c, s: c * s, id='complex_times_scalar'),
    pytest.param('s * c', lambda c, s: s * c, id='scalar_times_complex'),
    pytest.param('c / s', lambda c, s: c / s, id='complex_over_scalar'),
    pytest.param('s / c', lambda c, s: s / c, id='scalar_over_complex'),
    pytest.param('c + s', lambda c, s: c + s, id='complex_plus_scalar'),
    pytest.param('s + c', lambda c, s: s + c, id='scalar_plus_complex'),
    pytest.param('c - s', lambda c, s: c - s, id='complex_minus_scalar'),
    pytest.param('s - c', lambda c, s: s - c, id='scalar_minus_complex'),
]
COMPOUND_OPERATIONS = [
    pytest.param('*=', lambda c, s: c * s, id='times_assign'),
    pytest.param('/=', lambda c, s: c / s, id='over_assign'),
    pytest.param('+=', lambda c, s: c + s, id='plus_assign'),
    pytest.param('-=', lambda c, s: c - s, id='minus_assign'),
]
BINARY_OPERATION_CODES = [p.values[0] for p in BINARY_OPERATIONS]
COMPOUND_OPERATORS = [p.values[0] for p in COMPOUND_OPERATIONS]
SIZE = 6


def tolerance(ctype: type) -> float:
    return 1e-6 if ctype == np.complex64 else 1e-12


def random_complex(shape, ctype: type, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(ctype)


def run_scalar_tasklet(name: str, ctype: type, cdtype: dace.typeclass, scalar, sdtype: dace.typeclass, code: str,
                       language: dace.Language, values: np.ndarray) -> np.ndarray:
    """Maps ``code`` over a complex array ``c`` with the scalar ``s``, writing each element's ``o``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('C', [SIZE], cdtype)
    sdfg.add_array('S', [1], sdtype)
    sdfg.add_array('O', [SIZE], cdtype)
    state = sdfg.add_state()
    state.add_mapped_tasklet('op', {'k': f'0:{SIZE}'}, {
        'c': dace.Memlet('C[k]'),
        's': dace.Memlet('S[0]')
    },
                             code, {'o': dace.Memlet('O[k]')},
                             external_edges=True,
                             language=language)
    out = np.zeros(SIZE, ctype)
    sdfg(C=values.copy(), S=np.array([scalar]), O=out)
    return out


@pytest.mark.parametrize('code,numpy_operation', BINARY_OPERATIONS)
@pytest.mark.parametrize('scalar,sdtype', SCALARS)
@pytest.mark.parametrize('ctype,cdtype', COMPLEX_TYPES)
def test_a_complex_and_a_scalar_of_another_type_combine_like_numpy(ctype: type, cdtype: dace.typeclass, scalar,
                                                                   sdtype: dace.typeclass, code: str,
                                                                   numpy_operation: Callable):
    """The tasklet stores the result in the complex input's type, so numpy's result is cast to it."""
    name = f'binop_{cdtype.to_string()}_{sdtype.to_string()}_{BINARY_OPERATION_CODES.index(code)}'
    values = random_complex(SIZE, ctype, 1)
    got = run_scalar_tasklet(name, ctype, cdtype, scalar, sdtype, f'o = {code}', dace.Language.Python, values)
    want = numpy_operation(values, scalar).astype(ctype)
    np.testing.assert_allclose(got, want, rtol=tolerance(ctype), atol=tolerance(ctype))


@pytest.mark.parametrize('operator,numpy_operation', COMPOUND_OPERATIONS)
@pytest.mark.parametrize('scalar,sdtype', SCALARS)
@pytest.mark.parametrize('ctype,cdtype', COMPLEX_TYPES)
def test_compound_assignment_by_a_scalar_of_another_type_matches_numpy(ctype: type, cdtype: dace.typeclass, scalar,
                                                                       sdtype: dace.typeclass, operator: str,
                                                                       numpy_operation: Callable):
    name = f'compound_{cdtype.to_string()}_{sdtype.to_string()}_{COMPOUND_OPERATORS.index(operator)}'
    values = random_complex(SIZE, ctype, 2)
    got = run_scalar_tasklet(name, ctype, cdtype, scalar, sdtype, f'o = c; o {operator} s;', dace.Language.CPP, values)
    want = numpy_operation(values, scalar).astype(ctype)
    np.testing.assert_allclose(got, want, rtol=tolerance(ctype), atol=tolerance(ctype))


@pytest.mark.parametrize('ctype,cdtype', COMPLEX_TYPES)
def test_normalized_inverse_fft_matches_numpy(ctype: type, cdtype: dace.typeclass):
    """The pure DFT scales every element by a ``double`` factor, which zeroed a complex64 result."""

    @dace.program
    def tester(x: cdtype[30]):
        return np.fft.ifft(x)

    x = random_complex(30, ctype, 3)
    np.testing.assert_allclose(tester(x.copy()), np.fft.ifft(x), rtol=tolerance(ctype) * 10, atol=tolerance(ctype))


@pytest.mark.parametrize('ctype,cdtype', COMPLEX_TYPES)
def test_orthonormal_fft_matches_numpy(ctype: type, cdtype: dace.typeclass):

    @dace.program
    def tester(x: cdtype[30]):
        return np.fft.fft(x, norm='ortho')

    x = random_complex(30, ctype, 4)
    np.testing.assert_allclose(tester(x.copy()),
                               np.fft.fft(x, norm='ortho'),
                               rtol=tolerance(ctype) * 10,
                               atol=tolerance(ctype))


@pytest.mark.parametrize('axes', [None, (1, ), (0, 2)], ids=['all_axes', 'middle_axis', 'outer_axes'])
@pytest.mark.parametrize('ctype,cdtype', COMPLEX_TYPES)
def test_normalized_inverse_fftn_matches_numpy(ctype: type, cdtype: dace.typeclass, axes):

    @dace.program
    def tester(x: cdtype[4, 5, 6]):
        return np.fft.ifftn(x, axes=axes)

    x = random_complex((4, 5, 6), ctype, 5)
    np.testing.assert_allclose(tester(x.copy()),
                               np.fft.ifftn(x, axes=axes),
                               rtol=tolerance(ctype) * 10,
                               atol=tolerance(ctype))
