"""Bridge hazard-guard coverage: RAW / WAR / WAW sibling assigns.

Several Fortran statements in one loop body that touch the same array
land as separate tasklets in one SDFG state with no dataflow edge
between them (they back the same non-transient storage).  The codegen
scheduler only honours RAW edges on shared AccessNodes, so without the
hazard guard it is free to reorder write-before-read siblings and
clobber the value -- the bug diagnosed in cloudsc Section 4.5.

The guard (``emit_assign`` for IF/structured bodies, ``_raw_hazard``
for the loop-body batch) must force a new state whenever a new assign
read/write collides with a prior read/write in the current state:

* RAW -- a later sibling reads what an earlier one wrote;
* WAR -- a later sibling writes what an earlier one read;
* WAW -- two siblings write the same array (final value must be the
  textually-last write).

Each kernel is built through the bridge and compared against an
f2py-compiled reference of the same source (non-transformed
reference, per ``feedback_e2e_numerical``); a reorder produces a
grossly wrong result, so exact equality is the right assertion.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run(src: str, entry: str, tmp_path: Path, **arrays):
    """Build ``src`` through the bridge and via f2py; run both.

    :param src: inline Fortran with an ``intent(inout)`` output array.
    :param entry: mangled flang symbol (``_QP<name>``).
    :param tmp_path: pytest scratch dir.
    :param arrays: input/output arrays by Fortran dummy name; the
        ``out`` entry is duplicated for the two runs and compared.
    :returns: ``(out_sdfg, out_ref)`` after both executions.
    """
    name = entry.split("P")[-1]
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sdfg = build_sdfg(src, sdfg_dir, name=name, entry=entry).build()
    sdfg.validate()
    ref = f2py_compile(src, ref_dir, f"{name}_ref")

    n = next(iter(arrays.values())).shape[0]
    ref_kw = {k: np.array(v, order="F", copy=True) for k, v in arrays.items()}
    sdfg_kw = {k: np.array(v, order="F", copy=True) for k, v in arrays.items()}

    getattr(ref, name)(**ref_kw)
    sdfg(n=np.int32(n), **sdfg_kw)
    return sdfg_kw, ref_kw


def test_raw_sibling_read_after_write(tmp_path: Path):
    """``t = a*2 ; out = t+1`` -- the second statement reads what the
    first wrote.  Reordering makes ``out`` read stale ``t``."""
    src = """
subroutine raw_kern(n, a, t, out)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: t(n)
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    t(i)   = a(i) * 2.0d0
    out(i) = t(i) + 1.0d0
  end do
end subroutine raw_kern
"""
    a = np.arange(1, 9, dtype=np.float64)
    s, r = _run(src, "_QPraw_kern", tmp_path, a=a, t=np.zeros(8), out=np.zeros(8))
    np.testing.assert_array_equal(s["out"], r["out"])
    np.testing.assert_array_equal(s["out"], a * 2.0 + 1.0)


def test_war_sibling_write_after_read(tmp_path: Path):
    """``out = f+1 ; f = a*2`` -- the second statement writes what the
    first read.  Reordering makes ``out`` read the new ``f`` (this is
    the cloudsc 4.5 ``cv=...e/f ; f=f-e`` shape)."""
    src = """
subroutine war_kern(n, a, f, out)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: f(n)
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = f(i) + 1.0d0
    f(i)   = a(i) * 2.0d0
  end do
end subroutine war_kern
"""
    a = np.arange(1, 9, dtype=np.float64)
    f0 = np.full(8, 7.0)
    s, r = _run(src, "_QPwar_kern", tmp_path, a=a, f=f0.copy(), out=np.zeros(8))
    np.testing.assert_array_equal(s["out"], r["out"])
    np.testing.assert_array_equal(s["out"], f0 + 1.0)  # old f, not a*2
    np.testing.assert_array_equal(s["f"], a * 2.0)


def test_waw_then_read_final_write_wins(tmp_path: Path):
    """``x=a ; y=x*3 ; x=b`` -- ``x`` is written twice with a read in
    between.  Correct: ``y`` uses the first write (``a``), final ``x``
    is the last write (``b``).  A WAW/WAR reorder makes ``y`` read
    ``b``."""
    src = """
subroutine waw_kern(n, a, b, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(in)    :: b(n)
  real(8), intent(inout) :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    x(i) = a(i)
    y(i) = x(i) * 3.0d0
    x(i) = b(i)
  end do
end subroutine waw_kern
"""
    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(11, 19, dtype=np.float64)
    s, r = _run(src, "_QPwaw_kern", tmp_path, a=a, b=b, x=np.zeros(8), y=np.zeros(8))
    np.testing.assert_array_equal(s["y"], r["y"])
    np.testing.assert_array_equal(s["x"], r["x"])
    np.testing.assert_array_equal(s["y"], a * 3.0)  # first write of x
    np.testing.assert_array_equal(s["x"], b)  # last write wins


def test_hazard_chain_in_nested_if(tmp_path: Path):
    """The cloudsc-4.5 shape: WAR chain inside a nested IF in a
    multi-pass loop, exercising the ``emit_assign`` guard (not the
    loop-batch ``_raw_hazard``).  ``cv`` reads ``f`` then ``f`` is
    overwritten -- a reorder makes ``cv`` use the new ``f``."""
    src = """
subroutine haz_if(n, a, b, f, cv)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(in)    :: b(n)
  real(8), intent(inout) :: f(n)
  real(8), intent(inout) :: cv(n)
  integer :: i, p
  do p = 1, 2
    do i = 1, n
      if (a(i) > 0.0d0) then
        cv(i) = cv(i) + f(i) * b(i)
        f(i)  = f(i) - a(i)
      end if
    end do
  end do
end subroutine haz_if
"""
    a = np.arange(1, 9, dtype=np.float64)
    b = np.full(8, 0.5)
    s, r = _run(src, "_QPhaz_if", tmp_path, a=a, b=b, f=np.full(8, 10.0), cv=np.zeros(8))
    np.testing.assert_array_equal(s["cv"], r["cv"])
    np.testing.assert_array_equal(s["f"], r["f"])
