"""Reduction intrinsics as inline expression operands.

When a Fortran reduction (``MAXVAL``, ``MINVAL``, ``SUM``, ``PRODUCT``,
``ANY``, ``ALL``) appears as the immediate RHS of an assignment
(``out = MAXVAL(arr)``), the bridge's existing
``buildReduceNode`` / ``buildSectionReduceAssign`` paths handle it.

When the reduction appears as an OPERAND of a larger expression
(``out = max(scalar, MAXVAL(arr(s:e)))``), ``buildExpr`` previously
returned ``"?"`` for the reduction subexpression, the resulting
tasklet code failed Python ``ast.parse``, and the SDFG could not
build.

The new ``hlfir-lift-reduction-operands`` pass rewrites every nested
reduction into a preceding scalar-temp assign:

    %tmp = fir.alloca f64
    %tmp_decl = hlfir.declare %tmp ...
    hlfir.assign %maxval_result to %tmp_decl#0
    %loaded = fir.load %tmp_decl#0
    ...uses of %maxval_result rewritten to %loaded...

After the pass, the lifted ``temp = MAXVAL(...)`` is a top-level
reduction the dispatcher already handles, and the consuming
expression sees a clean scalar load.

Each test below pairs an SDFG run against an f2py / numpy reference
on identical random inputs.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_run(src: str, tmp_path: Path, **kwargs) -> dict:
    """Build an SDFG via the bridge, call it with kwargs, return the
    final state of every numpy buffer the caller passed in."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    sdfg(**kwargs)
    return kwargs


def test_inline_maxval_in_max_expression(tmp_path: Path):
    """``out = max(scalar, MAXVAL(arr(1:n)))`` — the failure repro from
    the velocity_tendencies probe."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = max(scalar, maxval(arr(1:n)))
end subroutine kernel
"""
    rng = np.random.default_rng(0)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(0.5)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    assert out[0] == max(scalar, arr.max())


def test_inline_minval_in_min_expression(tmp_path: Path):
    """``out = min(scalar, MINVAL(arr))`` — symmetric of the maxval
    case."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = min(scalar, minval(arr(1:n)))
end subroutine kernel
"""
    rng = np.random.default_rng(1)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(-0.5)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    assert out[0] == min(scalar, arr.min())


def test_inline_sum_in_arithmetic(tmp_path: Path):
    """``out = scalar + SUM(arr(1:n))`` — sum used additively in a
    larger expression."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = scalar + sum(arr(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(2)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(1.25)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    np.testing.assert_allclose(out[0], scalar + arr.sum(), rtol=1e-12)


def test_inline_product_in_arithmetic(tmp_path: Path):
    """``out = scalar * PRODUCT(arr(1:n))`` — product used
    multiplicatively in a larger expression."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = scalar * product(arr(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(3)
    n = 4  # small n to keep product manageable
    arr = np.asfortranarray(rng.uniform(0.5, 1.5, n))
    scalar = np.float64(2.0)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    np.testing.assert_allclose(out[0], scalar * arr.prod(), rtol=1e-12)


def test_two_inline_reductions_in_same_expression(tmp_path: Path):
    """``out = MAXVAL(a(1:n)) + MINVAL(b(1:n))`` — two distinct
    reductions in one expression, each must lift independently."""
    src = """
subroutine kernel(a, b, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n)
  real(8), intent(out) :: out
  out = maxval(a(1:n)) + minval(b(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(4)
    n = 16
    a = np.asfortranarray(rng.standard_normal(n))
    b = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, a=a, b=b, out=out, n=n)
    np.testing.assert_allclose(out[0], a.max() + b.min(), rtol=1e-12)


def test_inline_maxval_no_section(tmp_path: Path):
    """``out = max(scalar, MAXVAL(arr))`` — whole-array reduction
    (no slice) used inline.  Exercises the same lift but routes the
    lifted assign through the whole-array Reduce path instead of the
    section-reduce loop."""
    src = """
subroutine kernel(arr, scalar, out)
  implicit none
  real(8), intent(in) :: arr(8)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = max(scalar, maxval(arr))
end subroutine kernel
"""
    rng = np.random.default_rng(5)
    arr = np.asfortranarray(rng.standard_normal(8))
    scalar = np.float64(0.3)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out)
    assert out[0] == max(scalar, arr.max())
