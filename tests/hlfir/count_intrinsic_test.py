"""End-to-end tests for the Fortran ``COUNT`` intrinsic flowing through
the HLFIR frontend → ``CountLibraryNode``.

Three lowering modes:

- **Mode A** — ``COUNT(mask)`` over a plain logical / integer mask
  (whole-array reduce; output is a scalar).
- **Mode B** — ``COUNT(mask, dim=k)`` per-dim reduction (output is a
  rank-(N-1) array).
- **Mode C** — ``COUNT(<expression>)`` where the source is an inline
  ``hlfir.elemental`` (typically a comparison ``arr1 .eq. arr2`` or a
  compound boolean).  The bridge synthesises a transient int32 mask via
  a per-element loop, then routes it through ``CountLibraryNode``.

Each test compares against a numpy reference; numerical correctness is
the bar.  The library node's own pure expansion is covered separately
in ``count_library_node_test.py``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# Mode A — plain mask
# ---------------------------------------------------------------------------


def test_count_mode_a_logical_mask_1d(tmp_path: Path):
    """``COUNT(mask)`` on a 1-D logical mask.  Source is a plain
    ``hlfir.declare`` array; bridge picks the kLibTable ``count`` entry,
    emits a CountLibraryNode (no ``dim``), output is a scalar."""
    src = """
subroutine main(mask, n, res)
  integer, intent(in)  :: n
  logical, intent(in)  :: mask(n)
  integer, intent(out) :: res
  res = COUNT(mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(0)
    n = 16
    mask = (rng.random(n) > 0.5)
    res = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, n=n, res=res)
    assert int(res[0]) == int(mask.sum())


def test_count_mode_a_integer_mask_2d_whole(tmp_path: Path):
    """``COUNT(mask)`` on a 2-D integer mask — same path as 1-D, just
    a different rank.  Whole-array reduce; output is a scalar."""
    src = """
subroutine main(mask, n, m, res)
  integer, intent(in)  :: n, m
  logical, intent(in)  :: mask(n, m)
  integer, intent(out) :: res
  res = COUNT(mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(1)
    n, m = 6, 8
    mask = np.asfortranarray(rng.random((n, m)) > 0.5)
    res = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, n=n, m=m, res=res)
    assert int(res[0]) == int(mask.sum())


# ---------------------------------------------------------------------------
# Mode B — with explicit ``dim`` argument
# ---------------------------------------------------------------------------


def test_count_mode_b_with_dim_2(tmp_path: Path):
    """``COUNT(mask, dim=2)`` on a 2-D mask — output is rank-1 (collapses
    the second dim).  Bridge traces the dim operand via traceConstInt
    and stamps it on the ASTNode's reduce_axes; emit_libcall converts
    back to Fortran 1-based for CountLibraryNode's constructor."""
    src = """
subroutine main(mask, n, m, res)
  integer, intent(in)  :: n, m
  logical, intent(in)  :: mask(n, m)
  integer, intent(out) :: res(n)
  res = COUNT(mask, dim=2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(2)
    n, m = 5, 7
    mask = np.asfortranarray(rng.random((n, m)) > 0.5)
    res = np.zeros(n, dtype=np.int32)
    sdfg(mask=mask, n=n, m=m, res=res)
    np.testing.assert_array_equal(res, mask.sum(axis=1))


def test_count_mode_b_with_dim_1(tmp_path: Path):
    """``COUNT(mask, dim=1)`` on a 2-D mask — output is rank-1 along
    the second axis."""
    src = """
subroutine main(mask, n, m, res)
  integer, intent(in)  :: n, m
  logical, intent(in)  :: mask(n, m)
  integer, intent(out) :: res(m)
  res = COUNT(mask, dim=1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(3)
    n, m = 4, 6
    mask = np.asfortranarray(rng.random((n, m)) > 0.5)
    res = np.zeros(m, dtype=np.int32)
    sdfg(mask=mask, n=n, m=m, res=res)
    np.testing.assert_array_equal(res, mask.sum(axis=0))


# ---------------------------------------------------------------------------
# Mode C — inline elemental source (comparison-as-mask)
# ---------------------------------------------------------------------------


def test_count_mode_c_array_comparison(tmp_path: Path):
    """``COUNT(a .eq. b)`` — comparison inline.  Flang lowers the mask
    as ``hlfir.elemental { arith.cmpi eq; yield_element }``; the bridge
    walks the body, materialises a transient int32 mask, and routes
    through CountLibraryNode."""
    src = """
subroutine main(a, b, n, res)
  integer, intent(in)  :: n
  integer, intent(in)  :: a(n), b(n)
  integer, intent(out) :: res
  res = COUNT(a .eq. b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(4)
    n = 12
    a = np.ascontiguousarray(rng.integers(0, 5, size=n, dtype=np.int32))
    b = np.ascontiguousarray(rng.integers(0, 5, size=n, dtype=np.int32))
    res = np.zeros(1, dtype=np.int32)
    sdfg(a=a, b=b, n=n, res=res)
    assert int(res[0]) == int((a == b).sum())


def test_count_mode_c_compound_boolean(tmp_path: Path):
    """``COUNT((a .gt. lo) .and. (a .lt. hi))`` — nested elemental
    wrapping two comparisons combined by ``and``.  Same Mode C path."""
    src = """
subroutine main(a, n, lo, hi, res)
  integer, intent(in)  :: n, lo, hi
  integer, intent(in)  :: a(n)
  integer, intent(out) :: res
  res = COUNT((a .gt. lo) .and. (a .lt. hi))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(5)
    n, lo_val, hi_val = 20, 3, 8
    a = np.ascontiguousarray(rng.integers(0, 12, size=n, dtype=np.int32))
    res = np.zeros(1, dtype=np.int32)
    sdfg(a=a, n=n, lo=lo_val, hi=hi_val, res=res)
    assert int(res[0]) == int(((a > lo_val) & (a < hi_val)).sum())
