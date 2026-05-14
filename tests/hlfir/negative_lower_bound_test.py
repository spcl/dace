"""Negative-lower-bound Fortran arrays.

ICON uses refined-cell-tag indexing extensively: arrays like
``start_block(min_rlcell_int:max_rlcell_int)`` carry indices that span
roughly ``[-10, 7]``.  The bridge handles every array access as
``arr[(fortran_index) - offset_<arr>_d<i>]``, so the lower bound
needs to land on the SDFG side as the offset symbol's value.

Two distinct paths:

1. **Explicit-shape declare** -- ``INTEGER :: arr(-5:5)`` makes the
   declare carry the bounds; the bridge's extract_vars reads them
   into ``v.lower_bounds`` and ``descriptors.py`` specialises
   ``offset_arr_d0 = -5``.  This case should already work.

2. **Deferred-shape allocatable** -- ``INTEGER, ALLOCATABLE :: arr(:)``
   with the actual bounds set at runtime via
   ``ALLOCATE(arr(-5:5))``.  The bridge never sees the ALLOCATE, so
   ``lower_bounds`` defaults to ``"1"`` and accesses at ``arr(-3)``
   lower to ``arr[-4]`` -- invalid pointer.  Currently a real gap;
   marked xfail here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_explicit_negative_lower_bound(tmp_path: Path):
    """Explicit-shape ``arr(-5:5)`` -- the bridge SHOULD see the
    bound on the declare and specialise the offset symbol."""
    src = """
subroutine read_arr(arr, idx, out)
  implicit none
  integer, intent(in) :: arr(-5:5)
  integer, intent(in) :: idx
  integer, intent(out) :: out
  out = arr(idx)
end subroutine read_arr
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_arr", entry="_QPread_arr").build()
    sdfg.validate()

    # Populate arr so position-i holds value (i * 10) -- distinguishes
    # offsets-by-1 from offsets-by-(-5).
    arr = np.asfortranarray(np.array([(i - 5) * 10 for i in range(11)], dtype=np.int32))  # values -50 ... 50
    out = np.zeros(1, dtype=np.int32, order='F')

    # arr(-3) should return -30 (because arr(-5) = -50, arr(-4) = -40, etc).
    # If the bridge mishandles the offset, it'd return a different cell
    # value (or segfault).
    sdfg(arr=arr, idx=np.int32(-3), out=out)
    assert out[0] == -30, f"arr(-3) should be -30; got {out[0]}"

    sdfg(arr=arr, idx=np.int32(5), out=out)
    assert out[0] == 50, f"arr(5) should be 50; got {out[0]}"

    sdfg(arr=arr, idx=np.int32(0), out=out)
    assert out[0] == 0, f"arr(0) should be 0; got {out[0]}"


def test_deferred_shape_allocatable_offset_is_one_by_default(tmp_path: Path):
    """Deferred-shape ``ALLOCATABLE :: arr(:)`` -- the bridge can't see
    the runtime ``ALLOCATE(arr(-5:5))``, so it defaults the offset
    symbol to 1.  This test asserts the current behavior (no segfault,
    correct values when accessed with 1-based indices) and documents
    the gap: any kernel that allocates with a negative lower bound and
    then reads via a negative Fortran index will produce wrong indexing
    on the SDFG side.

    The right fix is either (a) accept an offset override as a free
    symbol the caller binds at call time, or (b) scan the body for the
    most-negative index used and infer that as the lower bound.
    """
    src = """
subroutine read_alloc(idx, out)
  implicit none
  integer, intent(in) :: idx
  integer, intent(out) :: out
  integer, allocatable :: arr(:)
  integer :: i
  allocate(arr(1:11))
  do i = 1, 11
    arr(i) = i * 10
  end do
  out = arr(idx)
  deallocate(arr)
end subroutine read_alloc
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_alloc", entry="_QPread_alloc").build()
    sdfg.validate()

    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(idx=np.int32(3), out=out)
    # 1-based: arr(3) = 30.  Confirms the default offset=1 path works
    # for positive-only indices.
    assert out[0] == 30, f"arr(3) should be 30; got {out[0]}"


def test_deferred_shape_allocatable_negative_lower_bound(tmp_path: Path):
    """ICON refined-cell-tag pattern: ``ALLOCATABLE :: arr(:)`` declare
    plus a runtime ``ALLOCATE(arr(-5:5))``, with body accesses at
    literal negative indices.  Bridge's static-inference pass scans
    ``hlfir.designate`` ops rooted at the declare's result and takes
    the per-dim min of literal indices; here the most-negative literal
    is ``arr(-5)``, so the inferred lower bound is ``-5``.  Subsequent
    ``arr(N)`` accesses lower to ``arr[N - (-5)] = arr[N+5]`` -- in
    bounds for any N in [-5, 5].

    Pre-fix behavior: bridge defaulted ``offset_d0 = 1``, lowering
    ``arr(-3)`` to ``arr[-4]`` -> segfault.  This test holds the line
    on the inference."""
    src = """
subroutine read_alloc(out)
  implicit none
  integer, intent(out) :: out(4)
  integer, allocatable :: arr(:)
  allocate(arr(-5:5))
  arr(-5) = -50
  arr(-3) = -30
  arr( 0) =   0
  arr( 5) =  50
  out(1) = arr(-5)
  out(2) = arr(-3)
  out(3) = arr( 0)
  out(4) = arr( 5)
  deallocate(arr)
end subroutine read_alloc
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_alloc", entry="_QPread_alloc").build()
    sdfg.validate()

    # Inference should pick -5 as the lower bound (most-negative literal
    # designate index in the body).
    assert dict(sdfg.constants).get('offset_arr_d0') == -5, (f"expected offset_arr_d0 inferred to -5; got "
                                                             f"{dict(sdfg.constants).get('offset_arr_d0')}")

    out = np.zeros(4, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-50, -30, 0, 50])


def test_zero_based_allocatable(tmp_path: Path):
    """0-based deferred-shape array -- common in ICON
    (``sfc%tsoil(nproma, 0:nlev_soil, nblks_c)``).  The bridge defaults
    offset_d0 to 1; the inference pass should drop it to 0 after
    seeing the literal ``arr(0)`` write."""
    src = """
subroutine zero_based(out)
  implicit none
  integer, intent(out) :: out(3)
  integer, allocatable :: arr(:)
  allocate(arr(0:2))
  arr(0) = 100
  arr(1) = 101
  arr(2) = 102
  out(1) = arr(0)
  out(2) = arr(1)
  out(3) = arr(2)
  deallocate(arr)
end subroutine zero_based
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="zero_based", entry="_QPzero_based").build()
    sdfg.validate()
    assert dict(sdfg.constants).get('offset_arr_d0') == 0

    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [100, 101, 102])


def test_icon_min_rledge_pattern(tmp_path: Path):
    """ICON's deepest negative bound: ``min_rledge = -13`` (from
    ``min_rledge_int - (2*max_hw+1)`` in mo_impl_constants).  Real
    ``mo_alloc_patches.f90`` does
    ``ALLOCATE(p_patch%edges%start_block(min_rledge:max_rledge))``
    and ``mo_velocity_advection`` later reads at literal negative
    indices like ``end_block(-13)``."""
    src = """
subroutine icon_edge_blocks(out)
  implicit none
  integer, parameter :: min_rl = -13
  integer, parameter :: max_rl =   8
  integer, intent(out) :: out(3)
  integer, allocatable :: end_block(:)
  allocate(end_block(min_rl:max_rl))
  end_block(-13) = -1300
  end_block( -8) =  -800
  end_block(  5) =   500
  out(1) = end_block(-13)
  out(2) = end_block( -8)
  out(3) = end_block(  5)
  deallocate(end_block)
end subroutine icon_edge_blocks
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="icon_edge_blocks", entry="_QPicon_edge_blocks").build()
    sdfg.validate()
    assert dict(sdfg.constants).get('offset_end_block_d0') == -13

    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-1300, -800, 500])


def test_multidim_mixed_negative_bounds(tmp_path: Path):
    """ICON's ``start_idx(min_rlcell:max_rlcell, max_childdom)``: rank-2
    array, dim-0 has a negative lower bound and dim-1 is default 1-based.
    Inference must adjust only dim-0, leave dim-1 at 1."""
    src = """
subroutine mixed_bounds(out)
  implicit none
  integer, intent(out) :: out(4)
  integer, allocatable :: arr(:, :)
  allocate(arr(-4:4, 1:3))
  arr(-4, 1) = -41
  arr( 0, 2) =   2
  arr( 4, 3) =  43
  arr(-2, 2) = -22
  out(1) = arr(-4, 1)
  out(2) = arr( 0, 2)
  out(3) = arr( 4, 3)
  out(4) = arr(-2, 2)
  deallocate(arr)
end subroutine mixed_bounds
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="mixed_bounds", entry="_QPmixed_bounds").build()
    sdfg.validate()
    assert dict(sdfg.constants).get('offset_arr_d0') == -4
    # dim 1 has no literal index below 1, so stays at default 1
    assert dict(sdfg.constants).get('offset_arr_d1') == 1

    out = np.zeros(4, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-41, 2, 43, -22])


def test_symbolic_index_local_allocatable(tmp_path: Path):
    """Loop-iterator access into a negative-bound *local* allocatable.
    No literal indices in the designates, so the literal-index pass
    yields nothing.  But the ``ALLOCATE(arr(-5:5))`` lowers to a
    ``fir.shape_shift -5, 11`` operand on the embox; option (B) reads
    that lower-bound operand directly and writes -5 into
    ``v.lower_bounds[0]``.  The kernel then lowers correctly without
    any literal-index hint."""
    src = """
subroutine sym_idx(out)
  implicit none
  integer, intent(out) :: out(11)
  integer, allocatable :: arr(:)
  integer :: i
  allocate(arr(-5:5))
  do i = -5, 5
    arr(i) = i * 100
  end do
  do i = -5, 5
    out(i + 6) = arr(i)
  end do
  deallocate(arr)
end subroutine sym_idx
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sym_idx", entry="_QPsym_idx").build()
    sdfg.validate()
    assert dict(
        sdfg.constants).get('offset_arr_d0') == -5, (f"shape_shift inference should pick -5 from ALLOCATE(arr(-5:5)); "
                                                     f"got {dict(sdfg.constants).get('offset_arr_d0')}")
    out = np.zeros(11, dtype=np.int32, order='F')
    sdfg(out=out)
    expected = np.array([(i - 5) * 100 for i in range(11)], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_dummy_arg_allocatable_literal_negative_index(tmp_path: Path):
    """Dummy-arg deferred-shape allocatable with literal negative
    indices in the body.  Mirrors the way ICON's velocity_tendencies
    reads ``p_patch%edges%start_block(-10)`` etc. through the flattened
    companion arrays.  The literal-index inference catches the
    negative literals and sets ``offset_arr_d0`` correctly without
    any bindings-layer involvement -- direct call works."""
    src = """
subroutine read_dummy(arr, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out(3)
  out(1) = arr(-3)
  out(2) = arr( 0)
  out(3) = arr( 5)
end subroutine read_dummy
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_dummy", entry="_QPread_dummy").build()
    sdfg.validate()
    # literal-index inference sees -3 as the most-negative literal
    assert dict(sdfg.constants).get('offset_arr_d0') == -3, (f"expected offset_arr_d0 == -3 (most-negative literal); "
                                                             f"got {dict(sdfg.constants).get('offset_arr_d0')}")

    # Buffer is allocated 1-based by numpy; caller positions data
    # so that what the SDFG reads at offset (arr[N - (-3)] = arr[N+3])
    # matches the Fortran semantic value.  For arr(N) with N in [-3, 5]
    # the buffer index is N+3, so the buffer has 9 elements with
    # buf[0]=arr(-3)=-30, buf[3]=arr(0)=0, buf[8]=arr(5)=50.
    arr = np.asfortranarray(np.array([(i - 3) * 10 for i in range(9)], dtype=np.int32))
    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, [-30, 0, 50])


def test_dummy_arg_allocatable_symbolic_loop(tmp_path: Path):
    """Dummy-arg deferred-shape allocatable + loop-iterator access.
    No literal indices in the body, so literal-index inference can't
    determine the lower bound.  Bridge falls back to leaving
    ``offset_arr_d0`` as a free symbol on the SDFG signature; the
    direct-call test passes it explicitly (``offset_arr_d0=-5``)."""
    src = """
subroutine sum_arr(arr, n, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -5, 5
    total = total + arr(i)
  end do
  out = total
end subroutine sum_arr
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sum_arr", entry="_QPsum_arr").build()
    sdfg.validate()
    # Loop bounds are literals but the designate index is symbolic;
    # the bridge leaves the offset free on the SDFG signature.
    assert 'offset_arr_d0' in sdfg.arglist(), (f"expected offset_arr_d0 to be free on SDFG signature; "
                                               f"arglist: {list(sdfg.arglist().keys())}")

    # Buffer holds arr(-5)..arr(5) at buf[0..10]; sum is 0.
    arr = np.asfortranarray(np.array(range(-5, 6), dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, n=np.int32(11), out=out, offset_arr_d0=np.int64(-5), arr_d0=np.int64(11))
    assert out[0] == 0, f"sum(arr(-5..5)) should be 0; got {out[0]}"


def test_dummy_arg_allocatable_multidim_mixed(tmp_path: Path):
    """Dummy-arg deferred-shape rank-2 allocatable, mixed bounds:
    dim 0 negative, dim 1 default 1-based.  Body uses symbolic
    indexing on dim 0 (via a loop iterator) and a literal on dim 1.
    Bridge: offset_arr_d0 is free (caller binds), offset_arr_d1 = 1
    (default, never bound below 1)."""
    src = """
subroutine sum_col(arr, n, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:, :)
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -3, 3
    total = total + arr(i, 1)
  end do
  out = total
end subroutine sum_col
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sum_col", entry="_QPsum_col").build()
    sdfg.validate()
    arglist = sdfg.arglist()
    assert 'offset_arr_d0' in arglist, "offset_arr_d0 should be free"
    # dim 1 has a literal index 1 in the body; that's a positive
    # literal, so inference doesn't fire (min >= 1) and the bridge
    # *does* set the dummy-arg-allocatable fallback for dim 1 too
    # since lit observation doesn't change the lower-bound default.
    # Document the actual behavior: free if seenLit was False, baked
    # at 1 if seenLit[d] was True (positive literal).

    # Buffer: column 0 holds arr(-3..3, 1) = values -3..3 -> sum=0.
    arr = np.asfortranarray(np.array([[i] for i in range(-3, 4)], dtype=np.int32))  # 7 rows, 1 col
    out = np.zeros(1, dtype=np.int32, order='F')
    kw = dict(arr=arr, n=np.int32(7), out=out, offset_arr_d0=np.int64(-3), arr_d0=np.int64(7), arr_d1=np.int64(1))
    if 'offset_arr_d1' in arglist:
        kw['offset_arr_d1'] = np.int64(1)
    sdfg(**kw)
    assert out[0] == 0, f"sum(arr(-3..3, 1)) should be 0; got {out[0]}"


def test_dummy_arg_allocatable_inout_symbolic_write(tmp_path: Path):
    """Dummy-arg ALLOCATABLE, INTENT(INOUT) -- kernel writes through
    a symbolic-index loop.  Verifies the runtime offset works for
    writebacks too, not just reads."""
    src = """
subroutine write_arr(arr)
  implicit none
  integer, allocatable, intent(inout) :: arr(:)
  integer :: i
  do i = -4, 4
    arr(i) = i * 10
  end do
end subroutine write_arr
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="write_arr", entry="_QPwrite_arr").build()
    sdfg.validate()
    assert 'offset_arr_d0' in sdfg.arglist()

    arr = np.zeros(9, dtype=np.int32, order='F')
    sdfg(arr=arr, offset_arr_d0=np.int64(-4), arr_d0=np.int64(9))
    expected = np.array([i * 10 for i in range(-4, 5)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


def test_dummy_arg_allocatable_two_arrays_independent_offsets(tmp_path: Path):
    """Two dummy-arg deferred-shape allocatables with DIFFERENT
    negative bounds.  Each gets its own free offset symbol; the
    caller passes the right one for each."""
    src = """
subroutine pair_sum(a, b, out)
  implicit none
  integer, allocatable, intent(in) :: a(:), b(:)
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -3, 3
    total = total + a(i)
  end do
  do i = -7, 7
    total = total + b(i)
  end do
  out = total
end subroutine pair_sum
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="pair_sum", entry="_QPpair_sum").build()
    sdfg.validate()
    arglist = sdfg.arglist()
    assert 'offset_a_d0' in arglist, "a should have a free offset symbol"
    assert 'offset_b_d0' in arglist, "b should have a free offset symbol"

    a = np.asfortranarray(np.array(range(-3, 4), dtype=np.int32))  # sum = 0
    b = np.asfortranarray(np.array(range(-7, 8), dtype=np.int32))  # sum = 0
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out, offset_a_d0=np.int64(-3), offset_b_d0=np.int64(-7), a_d0=np.int64(7), b_d0=np.int64(15))
    assert out[0] == 0, f"sum(a)+sum(b) should be 0; got {out[0]}"


def test_explicit_shape_parameter_negative_bound(tmp_path: Path):
    """Explicit-shape declare using a PARAMETER for the lower bound
    (``INTEGER :: arr(lb:5)`` where ``lb = -8``).  The bridge's
    ``resolveLowerBounds`` path reads the ShapeShiftOp directly and
    traces the parameter through ``traceConstInt``; this stays
    working alongside the literal-access inference."""
    src = """
subroutine param_bound(arr, out)
  implicit none
  integer, parameter :: lb = -8
  integer, parameter :: ub =  5
  integer, intent(in) :: arr(lb:ub)
  integer, intent(out) :: out(3)
  out(1) = arr(-8)
  out(2) = arr( 0)
  out(3) = arr( 5)
end subroutine param_bound
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="param_bound", entry="_QPparam_bound").build()
    sdfg.validate()
    assert dict(sdfg.constants).get('offset_arr_d0') == -8

    arr = np.asfortranarray(np.array([(i - 8) * 100 for i in range(14)], dtype=np.int32))  # arr(-8) = -800, etc
    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, [-800, 0, 500])
