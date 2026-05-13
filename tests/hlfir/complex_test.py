"""End-to-end tests for Fortran ``COMPLEX(4)`` / ``COMPLEX(8)`` through
the HLFIR bridge.

All tests use 1-D arrays even for "scalar" cases  --  Python 3.12 ctypes has
no ``c_double_complex``, so DaCe currently cannot pass a complex value
by-value (would silently drop the imaginary part).  Length-1 arrays
match the existing scalar-output convention and keep the test surface
ABI-clean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# Arithmetic  --  parametrised over (kind, op, numpy-equivalent)
# ---------------------------------------------------------------------------

_COMPLEX_BIN_OPS = [
    # (fortran_op, numpy_op, label, marks)
    ('+', np.add, 'add', ()),
    ('-', np.subtract, 'sub', ()),
    ('*', np.multiply, 'mul', ()),
    # Complex / lowers to ``__divdc3`` / ``__divsc3`` (overflow-safe
    # Smith's algorithm).  The bridge recognises the 4-real call shape
    # and reconstructs the original complex operands.
    ('/', np.divide, 'div', ()),
]

_COMPLEX_KINDS = [
    # (kind, np_dtype, fortran_decl, label)
    (4, np.complex64, 'complex(4)', 'c4'),
    (8, np.complex128, 'complex(8)', 'c8'),
]


@pytest.mark.parametrize("kind,np_dtype,decl,klabel", _COMPLEX_KINDS, ids=[k[3] for k in _COMPLEX_KINDS])
@pytest.mark.parametrize("fop,np_op,oplabel",
                         [pytest.param(o[0], o[1], o[2], marks=o[3], id=o[2]) for o in _COMPLEX_BIN_OPS])
def test_complex_arithmetic(tmp_path: Path, kind, np_dtype, decl, klabel, fop, np_op, oplabel):
    """Complex arithmetic on length-N arrays, compared against numpy."""
    src = f"""
subroutine main(n, a, b, out)
  integer,    intent(in)  :: n
  {decl}, intent(in)  :: a(n), b(n)
  {decl}, intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = a(i) {fop} b(i)
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 8
    rng = np.random.default_rng(0)
    a = (rng.random(n) + 1j * rng.random(n) + 0.1).astype(np_dtype)
    b = (rng.random(n) + 1j * rng.random(n) + 0.1).astype(np_dtype)
    out = np.zeros(n, dtype=np_dtype)
    sdfg(n=n, a=a, b=b, out=out)
    rtol = 1e-6 if kind == 4 else 1e-12
    np.testing.assert_allclose(out, np_op(a, b), rtol=rtol)


# ---------------------------------------------------------------------------
# Transcendentals  --  Fortran intrinsic vs numpy
# ---------------------------------------------------------------------------

_COMPLEX_UNARY_FUNCS = [
    # (fortran_intrinsic, numpy_func, label)
    ('SIN', np.sin, 'sin'),
    ('COS', np.cos, 'cos'),
    ('TAN', np.tan, 'tan'),
    ('SINH', np.sinh, 'sinh'),
    ('COSH', np.cosh, 'cosh'),
    ('TANH', np.tanh, 'tanh'),
    ('EXP', np.exp, 'exp'),
    ('LOG', np.log, 'log'),
    ('SQRT', np.sqrt, 'sqrt'),
]


@pytest.mark.parametrize("kind,np_dtype,decl,klabel", _COMPLEX_KINDS, ids=[k[3] for k in _COMPLEX_KINDS])
@pytest.mark.parametrize("fname,np_func,label", _COMPLEX_UNARY_FUNCS, ids=[f[2] for f in _COMPLEX_UNARY_FUNCS])
def test_complex_transcendentals(tmp_path: Path, kind, np_dtype, decl, klabel, fname, np_func, label):
    """Each complex transcendental lowers to ``c<func>`` (kind=8) or
    ``c<func>f`` (kind=4); the bridge maps both to the bare Python name."""
    src = f"""
subroutine main(n, a, out)
  integer,    intent(in)  :: n
  {decl}, intent(in)  :: a(n)
  {decl}, intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = {fname}(a(i))
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 6
    rng = np.random.default_rng(1)
    # Restrict to a tame domain so principal-branch funcs match numpy
    # bit-for-bit (avoid imag(a) ~= +/-pi/2 for tan, real(a) <= 0 for log, etc.)
    a = (0.1 + 0.5 * rng.random(n) + 1j * (0.1 + 0.4 * rng.random(n))).astype(np_dtype)
    out = np.zeros(n, dtype=np_dtype)
    sdfg(n=n, a=a, out=out)
    rtol = 1e-5 if kind == 4 else 1e-12
    np.testing.assert_allclose(out, np_func(a), rtol=rtol)


def test_complex8_abs_returns_real(tmp_path: Path):
    """``ABS(complex(8))`` returns ``real(8)``  --  lowered as ``cabs``."""
    src = """
subroutine main(n, a, out)
  integer,    intent(in)  :: n
  complex(8), intent(in)  :: a(n)
  real(8),    intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = abs(a(i))
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 4
    a = np.array([3 + 4j, 5 + 12j, -1 + 0j, 0 + 1j], dtype=np.complex128)
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=n, a=a, out=out)
    np.testing.assert_allclose(out, np.abs(a), rtol=1e-12)
