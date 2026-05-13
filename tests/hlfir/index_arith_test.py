"""Closed-form integer arithmetic inside array subscripts.

Flang lowers ``arr(..., nlev-1, ...)`` as ``arith.subi %nlev, %c1``
+ ``fir.convert`` before handing the value to ``hlfir.designate``.
``buildIndexExpr`` has to recognise those integer arith ops  --  without
it the subscript surfaces as ``?`` in the memlet and DaCe picks up
``?`` as a free symbol at call time (``KeyError: '?'`` from the
argslist resolver).

These tests are intentionally minimal  --  one pattern per test  --  so a
regression immediately points at the arith op that broke.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py_ref(src: str, out_dir: Path, name: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.f90").write_text(src)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c", f"{name}.f90", "-m", name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(name)
    return sys.modules[name]


def test_literal_integer_subscript(tmp_path: Path):
    """``arr(i, 1, jb)``  --  a literal integer in the mid dim.  Flang
    emits this as a plain ``arith.constant 1 : index``, which
    ``buildIndexExpr`` must render as ``1``."""
    src = """
subroutine lit_mid(a, b, n, m, p)
  implicit none
  integer, intent(in)    :: n, m, p
  real(8), intent(in)    :: a(n, m, p)
  real(8), intent(inout) :: b(n, p)
  integer :: i, k
  do k = 1, p
    do i = 1, n
      b(i, k) = a(i, 1, k)
    end do
  end do
end subroutine lit_mid
"""
    mod = _f2py_ref(src, tmp_path / "ref", "lit_mid")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="lit_mid").build()

    rng = np.random.default_rng(0)
    n, m, p = 4, 3, 5
    a = np.asfortranarray(rng.random((n, m, p)))
    b_ref = np.zeros((n, p), order="F")
    mod.lit_mid(a, b_ref)
    b_sdfg = np.zeros((n, p), order="F")
    sdfg(a=a, b=b_sdfg, n=n, m=m, p=p)
    np.testing.assert_array_equal(b_sdfg, b_ref)


def test_subi_in_subscript(tmp_path: Path):
    """``arr(i, k-1, jb)``  --  subtract-from-loop-iter.  Exercises the
    ``arith.subi`` branch of ``buildIndexExpr``."""
    src = """
subroutine sub_idx(a, b, n, m)
  implicit none
  integer, intent(in)    :: n, m
  real(8), intent(in)    :: a(n, m)
  real(8), intent(inout) :: b(n, m)
  integer :: i, k
  do k = 2, m
    do i = 1, n
      b(i, k) = a(i, k-1)
    end do
  end do
end subroutine sub_idx
"""
    mod = _f2py_ref(src, tmp_path / "ref", "sub_idx")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sub_idx").build()

    rng = np.random.default_rng(1)
    n, m = 4, 6
    a = np.asfortranarray(rng.random((n, m)))
    b_ref = np.zeros((n, m), order="F")
    mod.sub_idx(a, b_ref)
    b_sdfg = np.zeros((n, m), order="F")
    sdfg(a=a, b=b_sdfg, n=n, m=m)
    np.testing.assert_array_equal(b_sdfg, b_ref)


def test_subi_from_symbol_in_subscript(tmp_path: Path):
    """``arr(i, nlev-1)``  --  subtract-constant-from-symbol.  The
    ``nlev`` dummy's presence in the subscript (not just as a loop
    bound) is the pattern velocity loopnest 5 hits."""
    src = """
subroutine sym_sub(a, b, nlev, n)
  implicit none
  integer, intent(in)    :: nlev, n
  real(8), intent(in)    :: a(n, nlev)
  real(8), intent(inout) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = a(i, nlev-1)
  end do
end subroutine sym_sub
"""
    mod = _f2py_ref(src, tmp_path / "ref", "sym_sub")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sym_sub").build()

    rng = np.random.default_rng(2)
    n, nlev = 5, 7
    a = np.asfortranarray(rng.random((n, nlev)))
    b_ref = np.zeros(n, order="F")
    mod.sym_sub(a, b_ref)
    b_sdfg = np.zeros(n, order="F")
    sdfg(a=a, b=b_sdfg, n=n, nlev=nlev)
    np.testing.assert_array_equal(b_sdfg, b_ref)


def test_addi_in_subscript(tmp_path: Path):
    """``arr(i+1, k)``  --  add-constant-to-loop-iter.  Sibling of the
    subi case; guards the ``arith.addi`` branch."""
    src = """
subroutine add_idx(a, b, n, m)
  implicit none
  integer, intent(in)    :: n, m
  real(8), intent(in)    :: a(n, m)
  real(8), intent(inout) :: b(n, m)
  integer :: i, k
  do k = 1, m
    do i = 1, n-1
      b(i, k) = a(i+1, k)
    end do
  end do
end subroutine add_idx
"""
    mod = _f2py_ref(src, tmp_path / "ref", "add_idx")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="add_idx").build()

    rng = np.random.default_rng(3)
    n, m = 6, 4
    a = np.asfortranarray(rng.random((n, m)))
    b_ref = np.zeros((n, m), order="F")
    mod.add_idx(a, b_ref)
    b_sdfg = np.zeros((n, m), order="F")
    sdfg(a=a, b=b_sdfg, n=n, m=m)
    # Only the i=1..n-1 region was written; last row stays zero in both.
    np.testing.assert_array_equal(b_sdfg, b_ref)
