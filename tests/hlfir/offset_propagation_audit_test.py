"""Audit probes for offset-propagation corners not yet covered.

Each test targets a Fortran shape the bridge's lower-bound
inference *could* mis-handle.  A test that PASSES pins coverage;
an XFAIL documents a real gap with a minimal reproducer.

Covered surface (``extract_vars.cpp``):
  * ``traceConstInt`` -- peels arith.constant / fir.convert /
    arith.select(false=0) ONLY (no arith.subi/addi/muli folding).
  * ``traceConstIntThroughLoad`` -- peels hlfir.associate /
    hlfir.declare / fir.convert / fir.load<-fir.store /
    fir.load<-hlfir.assign, recursively.
  * ``lowerBoundsFromAllocSite`` -- local ALLOCATE shape_shift.
  * dummy-arg deferred-shape free-symbol fallback.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp_path: Path, entry: str):
    """Compile ``src`` to a built SDFG.

    :param src: inline Fortran.
    :param tmp_path: pytest scratch dir.
    :param entry: mangled subroutine symbol.
    :returns: built SDFG.
    """
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, d, name=entry.split('P')[-1], entry=entry).build()
    sdfg.validate()
    return sdfg


def test_two_level_inlined_callee_literal(tmp_path: Path):
    """Literal flows through TWO nested inlined subroutines before
    reaching the indexed access.  Exercises ``traceConstIntThroughLoad``
    recursion across more than one associate/declare/assign hop."""
    src = """
module mo_two
  implicit none
  contains
  subroutine innermost(arr, k, out)
    integer, allocatable, intent(in) :: arr(:)
    integer, intent(in) :: k
    integer, intent(out) :: out
    integer :: kk
    kk = k
    out = arr(kk)
  end subroutine innermost
  subroutine middle(arr, k, out)
    integer, allocatable, intent(in) :: arr(:)
    integer, intent(in) :: k
    integer, intent(out) :: out
    call innermost(arr, k, out)
  end subroutine middle
end module mo_two

subroutine two_outer(arr, out)
  use mo_two, only: middle
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out
  call middle(arr, -4, out)
end subroutine two_outer
"""
    sdfg = _build(src, tmp_path, "_QPtwo_outer")
    assert dict(sdfg.constants).get('offset_arr_d0') == -4, (f"2-level inline literal should propagate -4; got "
                                                             f"{dict(sdfg.constants).get('offset_arr_d0')}")
    arr = np.asfortranarray(np.array([10, 20, 30, 40, 50], dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out, arr_d0=np.int64(5))
    assert out[0] == 10  # arr(-4) = first element with lb=-4


def test_assumed_shape_explicit_negative_lower_bound(tmp_path: Path):
    """``REAL :: arr(-5:)`` -- explicit negative lower bound, assumed
    extent.  ``resolveLowerBounds`` should read -5 from the declare's
    ``fir.shape_shift`` operand."""
    src = """
subroutine assumed_lb(arr, n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: arr(-5:n)
  integer, intent(out) :: out
  out = arr(-5) + arr(-3) + arr(0)
end subroutine assumed_lb
"""
    sdfg = _build(src, tmp_path, "_QPassumed_lb")
    assert dict(
        sdfg.constants).get('offset_arr_d0') == -5, (f"explicit arr(-5:) lower bound should specialise to -5; got "
                                                     f"{dict(sdfg.constants).get('offset_arr_d0')}")
    # Buffer arr(-5..5): 11 elements; arr(-5)=buf[0], arr(-3)=buf[2],
    # arr(0)=buf[5].
    arr = np.asfortranarray(np.array([(i - 5) * 10 for i in range(11)], dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, n=np.int32(5), out=out)
    # arr(-5)= -50, arr(-3)= -30, arr(0)= 0 -> sum -80
    assert out[0] == -80


def test_section_assignment_negative_bounds(tmp_path: Path):
    """Whole-section assignment ``dst(-3:3) = src(-3:3)`` on local
    allocatables.  The designate carries triplet operands; the bound
    should come from the local ALLOCATE shape_shift."""
    src = """
subroutine sec_assign(out)
  implicit none
  integer, intent(out) :: out(7)
  integer, allocatable :: dst(:), src(:)
  integer :: i
  allocate(dst(-3:3), src(-3:3))
  do i = -3, 3
    src(i) = i * 11
  end do
  dst(-3:3) = src(-3:3)
  do i = -3, 3
    out(i + 4) = dst(i)
  end do
  deallocate(dst, src)
end subroutine sec_assign
"""
    sdfg = _build(src, tmp_path, "_QPsec_assign")
    consts = dict(sdfg.constants)
    assert consts.get('offset_dst_d0') == -3, (f"dst lower bound should be -3; got {consts.get('offset_dst_d0')}")
    assert consts.get('offset_src_d0') == -3, (f"src lower bound should be -3; got {consts.get('offset_src_d0')}")
    out = np.zeros(7, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [i * 11 for i in range(-3, 4)])


def test_struct_allocatable_member_literal_negative_index(tmp_path: Path):
    """Dummy struct with an ALLOCATABLE member read at a literal
    negative index -- the velocity_tendencies shape, isolated.
    The flattened companion ``p_tbl`` should get offset -7 from the
    literal-index inference."""
    src = """
module mo_s
  implicit none
  type :: holder_t
    integer, allocatable :: tbl(:)
  end type holder_t
end module mo_s

subroutine read_member(p, out)
  use mo_s, only: holder_t
  implicit none
  type(holder_t), intent(in) :: p
  integer, intent(out) :: out
  out = p%tbl(-7) + p%tbl(0) + p%tbl(5)
end subroutine read_member
"""
    sdfg = _build(src, tmp_path, "_QPread_member")
    assert dict(
        sdfg.constants).get('offset_p_tbl_d0') == -7, (f"struct allocatable member literal -7 should specialise; got "
                                                       f"{dict(sdfg.constants).get('offset_p_tbl_d0')}")
    # p_tbl(-7..5): 13 elements.
    tbl = np.asfortranarray(np.array([(i - 7) for i in range(13)], dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(p_tbl=tbl, out=out, p_tbl_d0=np.int64(13))
    # p%tbl(-7)=-7, p%tbl(0)=0, p%tbl(5)=5 -> sum -2
    assert out[0] == -2


def test_computed_index_dummy_arith_parameter_fold(tmp_path: Path):
    """``arr(lb - 1)`` on a DUMMY allocatable where ``lb`` is a
    negative PARAMETER.  Flang constant-folds ``lb - 1`` (-4-1) to a
    plain ``arith.constant -5`` *before* the bridge sees it, so the
    literal-index inference picks -5 up directly -- no ``arith.subi``
    folding needed in ``traceConstInt``.  Pins this as a covered
    (not gap) case."""
    src = """
subroutine computed_dummy(arr, out)
  implicit none
  integer, parameter :: lb = -4
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out
  out = arr(lb - 1)
end subroutine computed_dummy
"""
    sdfg = _build(src, tmp_path, "_QPcomputed_dummy")
    assert dict(sdfg.constants).get('offset_arr_d0') == -5, (f"PARAMETER arithmetic should fold to literal -5; got "
                                                             f"{dict(sdfg.constants).get('offset_arr_d0')}")
    # arr(lb-1) = arr(-5).  Buffer arr(-5..5): 11 elements,
    # offset -5 -> arr(-5) reads buf[0].
    arr = np.asfortranarray(np.array([777] + [0] * 10, dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out, arr_d0=np.int64(11))
    assert out[0] == 777
