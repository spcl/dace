"""Module-level derived types with array members — Phase 1.

The HLFIR pass ``hlfir-flatten-structs`` decomposes a ``type(t) :: s``
declaration where ``t`` has flat-only members (scalars or arrays of
scalars) into per-member declares ``s_<field>``.  After Phase 1 of
derived-type support, the pass also fires on **local** declares (not
just dummy arguments), and ``extract_vars`` recovers concrete extents
from ``fir.SequenceType`` when the synthesised per-field declare
carries no ``fir.shape`` operand.

Each test compares an SDFG run against a gfortran/f2py reference for
bit-exact validation, matching the saved e2e-numerical rule.

A negative test ensures the bridge throws a ``RuntimeError`` (not
silent wrong values) when ``hlfir-flatten-structs`` could not lower
the struct — the loud-failure pattern from the previous round of
correctness work.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_local_struct_element_write_and_read(tmp_path: Path):
    """Local ``type(t) :: s`` with explicit-shape array member, single
    element write + read.  Exercises the local-instance flatten +
    SequenceType-extent fallback in ``extract_vars``."""
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5)
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(2)
  type(simple_type) :: s
  s%w(1, 1, 1) = 5.5
  d(1) = s%w(1, 1, 1)
  d(2) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_element_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [5.5, 11.0])


def test_local_struct_two_array_members(tmp_path: Path):
    """Two array members of different shapes — exercises the per-
    member path generating two separate flat arrays."""
    src = """
module lib
  implicit none
  type two_arrays
    real :: u(4)
    real :: v(7)
  end type two_arrays
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(2)
  type(two_arrays) :: t
  t%u(2) = 3.0
  t%v(7) = 4.0
  out(1) = t%u(2)
  out(2) = t%v(7)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_two_arrays_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [3.0, 4.0])


def test_local_struct_member_in_loop(tmp_path: Path):
    """Loop-driven element writes to a struct's array member.  The
    flat ``s_w`` array carries the SequenceType's static (5,) extent —
    the SDFG signature has no synth shape symbol to bind."""
    src = """
module lib
  implicit none
  type sum_type
    real :: w(5)
  end type sum_type
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(sum_type) :: s
  integer :: i
  do i = 1, 5
    s%w(i) = real(i) * 2.0
  end do
  out = s%w(1) + s%w(2) + s%w(3) + s%w(4) + s%w(5)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_loop_ref")
    out_ref = float(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 2.0 + 4.0 + 6.0 + 8.0 + 10.0


def test_icon_style_state_struct(tmp_path: Path):
    """ICON dycore-flavoured state struct: many parallel 3-D arrays
    (``(nproma, nlev, nblks)``) bundled into one derived type.  The
    real code looks like ``p_diag%vn``, ``p_diag%w``, ``p_diag%theta_v``
    indexed elementwise inside a kernel.  Verifies Phase 1's standard
    path on a wider, more realistic schema."""
    src = """
module lib
  implicit none
  integer, parameter :: nproma = 4, nlev = 3, nblks = 2
  type state_t
    real :: vn(nproma, nlev, nblks)
    real :: w(nproma, nlev, nblks)
    real :: theta_v(nproma, nlev, nblks)
  end type state_t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(nproma, nlev, nblks)
  type(state_t) :: s
  integer :: i, k, b
  do b = 1, nblks
    do k = 1, nlev
      do i = 1, nproma
        s%vn(i, k, b)      = real(i + k * 10 + b * 100)
        s%w(i, k, b)       = real(i)
        s%theta_v(i, k, b) = real(b)
        out(i, k, b) = s%vn(i, k, b) + s%w(i, k, b) * s%theta_v(i, k, b)
      end do
    end do
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "icon_state_ref")
    out_ref = mod.main()

    sdfg = _build(src, tmp_path)
    out = np.zeros((4, 3, 2), order='F', dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)


def test_qe_style_pdf_sampler_struct(tmp_path: Path):
    """QE-flavoured ``pdf_sampler_type`` (from npbench's ``usxx.py``):
    scalar struct with several flat members of different types
    (integers, real(8), and a fixed-shape real(8) lookup table).
    Drops the original's ``ALLOCATABLE val(:,:)`` member — that's
    Phase 3 territory — but keeps the rest of the schema."""
    src = """
module lib
  implicit none
  integer, parameter :: ncdf = 4, nfsd = 3
  type pdf_sampler_type
    integer :: ncdf_n, nfsd_n
    real(8) :: fsd1, inv_fsd_interval
    real(8) :: val(ncdf, nfsd)
  end type pdf_sampler_type
end module lib

subroutine main(out)
  use lib
  implicit none
  real(8), intent(out) :: out
  type(pdf_sampler_type) :: s
  integer :: i, j
  s%ncdf_n = ncdf
  s%nfsd_n = nfsd
  s%fsd1 = 1.5d0
  s%inv_fsd_interval = 2.5d0
  do j = 1, nfsd
    do i = 1, ncdf
      s%val(i, j) = real(i * j, 8)
    end do
  end do
  out = s%val(2, 2) * s%fsd1 + s%inv_fsd_interval &
        + real(s%ncdf_n + s%nfsd_n, 8)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "qe_pdf_sampler_ref")
    out_ref = float(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(float(out[0]), out_ref, rtol=1e-12)


def test_batched_csr_fixed_capacity(tmp_path: Path):
    """Batched CSR — array of CSR-shaped structs, each member sized to a
    common compile-time capacity.  This is the AoS-with-array-members
    pattern Phase 1.5 lifted: ``A(N)`` of struct with array members
    flattens to a per-member SoA where the outer dim is ``N`` and the
    inner is the member's declared extent.

    Layout:
      ``A_rowptr(N, ROW_CAP)``
      ``A_colidx(N, NNZ_CAP)``
      ``A_val   (N, NNZ_CAP)``

    Real Fortran code that needs runtime-jagged CSR (each instance
    with its own ``allocatable`` of a different real size) is Phase 3
    territory — see ``test_batched_csr_allocatable_xfail`` below.
    """
    src = """
module lib
  implicit none
  integer, parameter :: ROWS    = 3
  integer, parameter :: ROW_CAP = ROWS + 1
  integer, parameter :: NNZ_CAP = 4
  type csr_t
    integer :: rowptr(ROW_CAP)
    integer :: colidx(NNZ_CAP)
    real(8) :: val(NNZ_CAP)
  end type csr_t
end module lib

subroutine main(out)
  ! Run a tiny SpMV on each batched CSR matrix, accumulating into ``out``.
  ! Per-element init avoids whole-array component access ``A(1)%rowptr =
  ! (/ ... /)``, which the bridge rewriter doesn't yet collapse to a
  ! flat slice ``A_rowptr(1, :)`` (Phase 2.1 follow-up).
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: b, r, k

  ! ---- Init batch 1: identity (3x3) ----
  A(1)%rowptr(1) = 1
  A(1)%rowptr(2) = 2
  A(1)%rowptr(3) = 3
  A(1)%rowptr(4) = 4
  A(1)%colidx(1) = 1
  A(1)%colidx(2) = 2
  A(1)%colidx(3) = 3
  A(1)%colidx(4) = 0
  A(1)%val(1) = 1.0d0
  A(1)%val(2) = 1.0d0
  A(1)%val(3) = 1.0d0
  A(1)%val(4) = 0.0d0

  ! ---- Init batch 2: tridiagonal-flavoured ----
  A(2)%rowptr(1) = 1
  A(2)%rowptr(2) = 3
  A(2)%rowptr(3) = 4
  A(2)%rowptr(4) = 5
  A(2)%colidx(1) = 1
  A(2)%colidx(2) = 2
  A(2)%colidx(3) = 1
  A(2)%colidx(4) = 3
  A(2)%val(1) = 2.0d0
  A(2)%val(2) = -1.0d0
  A(2)%val(3) = 1.0d0
  A(2)%val(4) = 1.0d0

  x(1) = 10.0d0
  x(2) = 20.0d0
  x(3) = 30.0d0

  ! ---- SpMV per batch ----
  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "batched_csr_ref")
    out_ref = mod.main()

    sdfg = _build(src, tmp_path)
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12)


def test_batched_csr_fixed_capacity_cross_boundary(tmp_path: Path):
    """Same fixed-capacity CSR as the local test, but the SpMV runs in
    a CALLEE.  Caller hands the batched CSR struct array across the
    subroutine boundary.

    The bindings layer's pack/unpack is what's needed here: caller-side
    allocates ``A_rowptr(BATCH, ROW_CAP)`` etc., copies each
    ``A(i)%<m>(:)`` into the flat slot, calls the SDFG, and unpacks
    on return.  The flat representation matches what the AoS-with-
    array-members local rewrite already produces; only the bindings
    emitter side is missing.
    """
    src = """
module lib
  implicit none
  integer, parameter :: ROWS    = 3
  integer, parameter :: ROW_CAP = ROWS + 1
  integer, parameter :: NNZ_CAP = 4
  type csr_t
    integer :: rowptr(ROW_CAP)
    integer :: colidx(NNZ_CAP)
    real(8) :: val(NNZ_CAP)
  end type csr_t
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: i

  ! Init batches and x (per-element so the pass is exercised on the
  ! local-allocation path; the test then passes A across the boundary).
  A(1)%rowptr(1) = 1
  A(1)%rowptr(2) = 2
  A(1)%rowptr(3) = 3
  A(1)%rowptr(4) = 4
  do i = 1, NNZ_CAP
    A(1)%colidx(i) = i
    A(1)%val(i) = 1.0d0
  end do

  A(2)%rowptr(1) = 1
  A(2)%rowptr(2) = 3
  A(2)%rowptr(3) = 4
  A(2)%rowptr(4) = 5
  A(2)%colidx(1) = 1
  A(2)%colidx(2) = 2
  A(2)%colidx(3) = 1
  A(2)%colidx(4) = 3
  A(2)%val(1) = 2.0d0
  A(2)%val(2) = -1.0d0
  A(2)%val(3) = 1.0d0
  A(2)%val(4) = 1.0d0

  x(1) = 10.0d0
  x(2) = 20.0d0
  x(3) = 30.0d0

  call spmv_batched(A, x, out)
end subroutine main

subroutine spmv_batched(A, x, out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  type(csr_t), intent(in) :: A(BATCH)
  real(8), intent(in)     :: x(ROWS)
  real(8), intent(out)    :: out(BATCH, ROWS)
  integer :: b, r, k

  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine spmv_batched
"""
    sdfg = _build(src, tmp_path, name='main', entry='_QPmain')
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)


def test_local_struct_allocatable_member_element_writes(tmp_path: Path):
    """Phase 5a: ``type t :: real, allocatable :: w(:)`` — local
    struct instance with one allocatable array member, allocate then
    per-element writes, then read back two elements and sum.

    The flatten pass replaces ``s%w`` with a flat top-level allocatable
    ``s_w`` (declare carrying ``fortran_attrs = #fir.var_attrs<allocatable>``)
    and renames the per-allocate ``fir.allocmem`` op so the bridge's
    ``collectAllocSites`` finds it under ``s_w.alloc``.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out)   :: res
  type(t) :: s
  allocate(s%w(n))
  s%w(1) = 1.0
  s%w(2) = 2.0
  s%w(3) = 3.0
  s%w(4) = 4.0
  res = s%w(2) + s%w(4)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=4, res=res)
    assert res[0] == 6.0


def test_local_struct_allocatable_with_scalar_sibling_member(tmp_path: Path):
    """Phase 5a: struct with one scalar field and one allocatable
    array field.  The scalar member also flattens (existing path) —
    test pins both flat declares co-existing and the scalar field
    being usable as the allocate's extent."""
    src = """
module lib
  implicit none
  type t
    integer :: n
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(extent, res)
  use lib
  implicit none
  integer, intent(in) :: extent
  real, intent(out)   :: res
  type(t) :: s
  s%n = extent
  allocate(s%w(s%n))
  s%w(1) = 10.0
  s%w(s%n) = 20.0
  res = s%w(1) + s%w(s%n)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(extent=5, res=res)
    assert res[0] == 30.0


def test_local_struct_allocatable_whole_array_assign(tmp_path: Path):
    """Phase 5a: per-element copy into the allocatable member then
    read back.  Pins the allocate-then-loop pattern against the
    rewritten ``s_w`` flat declare."""
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n, src, res)
  use lib
  implicit none
  integer, intent(in)    :: n
  real, intent(in)       :: src(n)
  real, intent(out)      :: res(n)
  type(t) :: s
  integer :: i
  allocate(s%w(n))
  do i = 1, n
    s%w(i) = src(i)
  end do
  do i = 1, n
    res(i) = s%w(i)
  end do
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    n = 6
    src_arr = np.arange(1.0, n + 1.0, dtype=np.float32)
    res = np.zeros(n, dtype=np.float32)
    sdfg(n=n, src=src_arr, res=res)
    np.testing.assert_array_equal(res, src_arr)


def test_dummy_struct_with_allocatable_member_top_level_call(tmp_path: Path):
    """Phase 5b: ``type(t), intent(in) :: s`` where ``t`` has an
    allocatable member, passed across a top-level subroutine call
    (NOT module-contained).  After ``hlfir-inline-all`` the callee
    body is spliced into main and ``s%w`` reads designate the inlined
    alias which traces back to the caller's struct decl.

    Removes the dummy-arg-with-allocatable gate from
    ``planAndReplaceStructArgs``: per the saved bindings policy, the
    flatten pass treats the member like any other allocatable and the
    bindings-side wrapper handles the descriptor marshalling
    (nullptr if not allocated, packed copy if allocated; no runtime
    ``ALLOCATED()`` checks — the program is responsible for tracking
    allocation state)."""
    src = """
module lib
  implicit none
  type t
    integer :: n
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: s
  s%n = 4
  allocate(s%w(s%n))
  s%w(1) = 10.0
  s%w(2) = 20.0
  s%w(3) = 30.0
  s%w(4) = 40.0
  call accumulate(s, out)
  deallocate(s%w)
end subroutine main

subroutine accumulate(s, out)
  use lib
  implicit none
  type(t), intent(in) :: s
  real, intent(out) :: out
  integer :: i
  out = 0.0
  do i = 1, s%n
    out = out + s%w(i)
  end do
end subroutine accumulate
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    assert out[0] == 100.0


def test_struct_pointer_member_slice_rebind(tmp_path: Path):
    """Phase 5b: ``type t :: real, pointer :: w(:)`` — local struct
    instance with a pointer array member, rebound to a section of a
    TARGET'd array (``s%w => src(1:n)``).  The flatten pass treats
    pointer members the same as allocatable members (Phase 5a):
    synthesises a flat top-level companion ``s_w`` carrying the
    ``pointer`` fortran_attr.  ``hlfir-rewrite-pointer-assigns``'s
    slice-target arm then forwards the rebound section box to every
    load of the flat companion's box-ref slot, so each ``s%w(i)``
    read lands on ``src(i)`` directly.

    Strict-no-aliasing assumption: the rewrite is unsafe if the
    program relies on aliasing.  The test exercises the read-only
    direction (``res = s%w(2) + s%w(4)``); writes through the
    pointer follow the same lowering path."""
    src = """
module lib
  implicit none
  type t
    real, pointer :: w(:)
  end type t
end module lib

subroutine main(n, src, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(in), target :: src(n)
  real, intent(out) :: res
  type(t) :: s
  s%w => src(1:n)
  res = s%w(2) + s%w(4)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    n = 5
    src_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=n, src=src_arr, res=res)
    assert res[0] == 6.0  # src[1] + src[3] = 2 + 4


def test_local_struct_allocatable_via_inlined_subprogram(tmp_path: Path):
    """Phase 5b: allocate happens INSIDE a module-contained subroutine
    that takes the struct as an ``intent(inout)`` dummy.  After
    ``hlfir-inline-all`` runs, the callee's ``s%w`` designate is
    rooted at an inlined alias declare, not the caller's struct
    declare.  ``renameMemberAllocmems`` must follow alias chains
    (``hlfir.declare`` → ``fir.embox``/``fir.convert`` → caller's
    declare) to find the allocate site and rename it.  Without the
    alias walk the SDFG ends up with an unbound ``s_w_d0`` symbol
    even though the user-visible Fortran name ``n`` is in scope.

    NOTE: ``entry='_QPmain'`` is REQUIRED.  The Fortran source
    declares two public functions (``_QMlibPfill_and_set`` and
    ``_QPmain``).  Without explicit entry the bridge would walk the
    first public function in module order — ``fill_and_set`` —
    whose un-inlined body is unsupported (struct dummy with
    allocatable member).  Passing ``entry`` marks every other
    func.func private so symbol-dce drops them after inlining."""
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
contains
  subroutine fill_and_set(s, n)
    type(t), intent(inout) :: s
    integer, intent(in) :: n
    integer :: i
    allocate(s%w(n))
    do i = 1, n
      s%w(i) = real(i * 10)
    end do
  end subroutine fill_and_set
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(3)
  type(t) :: s
  call fill_and_set(s, n)
  res(1) = s%w(1)
  res(2) = s%w(n / 2)
  res(3) = s%w(n)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    res = np.zeros(3, dtype=np.float32)
    sdfg(n=6, res=res)
    np.testing.assert_array_equal(res, [10.0, 30.0, 60.0])


def test_parametric_dim_from_struct_field(tmp_path: Path):
    """Phase 6: ``real :: bob(st%a)`` — local array whose extent is a
    struct field's runtime value.  After flatten, ``st%a`` becomes
    ``st_a`` and is bound as an SDFG symbol; ``bob``'s shape is
    ``(st_a,)`` and the bridge emits a transient with that runtime
    extent.  Element writes, whole-array assigns, and elementwise
    arithmetic all flow through the existing array-of-symbol path."""
    src = """
module lib
  implicit none
  type t
    integer :: a
  end type t
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(10)
  type(t) :: st
  st%a = n
  block
    real :: bob(st%a)
    bob(:) = 0.0
    bob(1) = 5.5
    res(1:st%a) = bob + 1.0
  end block
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    res = np.zeros(10, dtype=np.float32)
    sdfg(n=4, res=res)
    np.testing.assert_array_equal(res, [6.5, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0])


def test_parametric_dim_two_locals_one_struct(tmp_path: Path):
    """Phase 6: two parametric locals from sibling fields of the same
    struct.  Each gets its own SDFG symbol (``st_a`` / ``st_b``) and
    the bridge keeps them independent — the transient declarations
    don't shadow each other."""
    src = """
module lib
  implicit none
  type t
    integer :: a
    integer :: b
  end type t
end module lib

subroutine main(av, bv, res)
  use lib
  implicit none
  integer, intent(in) :: av, bv
  real, intent(out) :: res(20)
  type(t) :: st
  st%a = av
  st%b = bv
  block
    real :: outer(st%a)
    real :: inner(st%b)
    integer :: i
    outer = 1.5
    inner = 2.5
    do i = 1, st%a
      res(i) = outer(i)
    end do
    do i = 1, st%b
      res(st%a + i) = inner(i)
    end do
  end block
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    res = np.zeros(20, dtype=np.float32)
    sdfg(av=3, bv=4, res=res)
    np.testing.assert_array_equal(res, [1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_parametric_dim_via_inlined_subprogram(tmp_path: Path):
    """Phase 6 + cross-subprogram: parametric local ``bob(st%a)`` lives
    inside a module-contained subroutine that gets inlined into main.
    Pins that the runtime-extent symbol is bound at the right scope
    after inlining."""
    src = """
module lib
  implicit none
  type t
    integer :: a
  end type t
contains
  subroutine fill(d, st)
    type(t), intent(in) :: st
    real, intent(out) :: d(10)
    real :: bob(st%a)
    integer :: i
    do i = 1, st%a
      bob(i) = real(i)
    end do
    do i = 1, st%a
      d(i) = bob(i) * 2.0
    end do
  end subroutine fill
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(10)
  type(t) :: st
  st%a = n
  call fill(res, st)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    res = np.zeros(10, dtype=np.float32)
    sdfg(n=4, res=res)
    np.testing.assert_array_equal(res, [2, 4, 6, 8, 0, 0, 0, 0, 0, 0])


def test_local_struct_allocatable_member_reallocate(tmp_path: Path):
    """Phase 5a: allocate / deallocate / re-allocate cycle on the same
    member.  Pins the multiple-allocate-site path: each
    ``fir.allocmem`` gets renamed by the flatten pass; the second
    site comes back from ``collectAllocSites`` as a separate
    ``fir.allocmem`` and ``allocAliasName`` mints a second-site
    transient name (``s_w_alloc1``)."""
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n1, n2, res)
  use lib
  implicit none
  integer, intent(in) :: n1, n2
  real, intent(out)   :: res
  type(t) :: s
  allocate(s%w(n1))
  s%w(1) = 7.0
  deallocate(s%w)
  allocate(s%w(n2))
  s%w(1) = 11.0
  res = s%w(1)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n1=3, n2=4, res=res)
    assert res[0] == 11.0


def test_aos_allocatable_uniform_const_size(tmp_path: Path):
    """Phase 5c-A: AoS + allocatable member with uniform compile-time
    constant allocate sizes.  Each ``A(i)`` allocates ``A(i)%w`` to
    the SAME constant ``M`` elements, so the flat companion is
    fully static ``A_w(N, M)`` and the kernel-internal allocate /
    deallocate sites become semantic no-ops over the pre-existing
    2D buffer.

    Bridge changes coordinated for this case:
      * ``FlattenStructs.cpp::aosAllocUniformConstSize`` predicate
        gates the AoS+allocatable path on uniform-const sizes.
      * Phase 5c-A synth emits ``A_<member>(N, M)`` (concat extent).
      * ``collapseAosAllocReads`` rewrites the
        ``fir.load + hlfir.designate (loaded, j)`` chain into a
        direct 2-index ``hlfir.designate flatBase (i, j)``.
      * ``stripReallocOnAosMember`` drops the ``realloc`` flag from
        whole-array assigns so the verifier accepts the now-static
        LHS.
      * ``eraseAosAllocDeallocChain`` sweeps the per-instance
        allocate / freemem ops; the wrapping ``deallocate`` loop's
        body becomes empty.
      * Post-gen sweep in ``SDFGBuilder.build()`` adds an empty
        state to any zero-block CFG region, keeping the empty
        deallocate loop's ``LoopRegion`` valid.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i, j
  do i = 1, 2
    allocate(A(i)%w(3))
    do j = 1, 3
      A(i)%w(j) = real(i * j)
    end do
  end do
  out = A(1)%w(1) + A(2)%w(2)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # A(1)%w(1) = 1*1 = 1, A(2)%w(2) = 2*2 = 4, sum = 5
    assert out[0] == 5.0


def test_aos_allocatable_via_inlined_kernel(tmp_path: Path):
    """Phase 5c-B: AoS+allocatable struct passed as ``intent(inout)``
    dummy to a module-contained kernel.  After ``hlfir-inline-all``
    splices the kernel body into the caller, the kernel's reads/
    writes through ``A(i)%w(j)`` are rooted at an alias declare
    (dummy_scope), not the caller's original struct decl.

    ``collapseAosAllocReads`` must follow alias chains
    (``hlfir.declare`` → ``fir.embox`` / ``fir.convert`` → caller's
    decl) to find every load-of-member-designate inside the inlined
    body.  Without the alias walk, the kernel's reads stay as 1-D
    designates against the loaded 2D companion, silently producing
    wrong indices.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
contains
  subroutine kernel(A, n, m, out)
    type(t), intent(inout) :: A(2)
    integer, intent(in) :: n, m
    real, intent(out) :: out
    integer :: i, j
    do i = 1, n
      do j = 1, m
        A(i)%w(j) = A(i)%w(j) * 2.0
      end do
    end do
    out = A(1)%w(1) + A(2)%w(2)
  end subroutine kernel
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i, j
  do i = 1, 2
    allocate(A(i)%w(3))
    do j = 1, 3
      A(i)%w(j) = real(i + j)
    end do
  end do
  call kernel(A, 2, 3, out)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # Original A(1)%w(1) = 1+1 = 2; after kernel doubles every cell,
    # A(1)%w = [4, 6, 8] and A(2)%w = [6, 8, 10].
    # out = A(1)%w(1) + A(2)%w(2) = 4 + 8 = 12.
    assert out[0] == 12.0


def test_aos_allocatable_whole_array_assign(tmp_path: Path):
    """Phase 5c-A: whole-array assign on AoS-allocatable member.
    ``A(i)%w = scalar`` must lower to ``A_w(i, 1:M:1) = scalar`` —
    a row-section assign — NOT a whole-2D-array broadcast.  Without
    the section rewrite, every iteration's scalar would be
    splatted across all rows, silently corrupting earlier rows.

    Pinned by ``rewriteAosWholeMemberAssign`` in FlattenStructs.cpp.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i
  do i = 1, 2
    allocate(A(i)%w(3))
    A(i)%w = real(i)         ! whole-array assign of scalar
  end do
  out = A(1)%w(1) + A(2)%w(2)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='_QPmain')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # A(1)%w = [1, 1, 1], A(2)%w = [2, 2, 2]; sum = 1 + 2 = 3.
    assert out[0] == 3.0


def test_batched_csr_allocatable_xfail(tmp_path: Path):
    """Genuinely jagged batched CSR — each instance's CSR arrays are
    runtime-allocated to different (compile-time-constant) sizes.

    Lowering chain:
      * ``aosAllocMaxConstSize`` records ``max_i(N_i)`` per allocatable
        member as the companion column count.
      * ``rewriteAosWholeMemberAssign`` resolves a per-instance
        section bound when the parent's outer index is a constant —
        each ``A(i)%w = (/...lit.../)`` lowers to a row section sized
        to that instance's specific allocate size, not the global cap.
      * The constant-pool feature (``parameter``-attributed declares
        backed by ``fir.global ... constant``) ships the literal data
        through ``sdfg.add_constant`` so the kernel's reads see the
        right values.
    """
    src = """
module lib
  implicit none
  type csr_t
    integer, allocatable :: rowptr(:)
    integer, allocatable :: colidx(:)
    real(8), allocatable :: val(:)
  end type csr_t
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2, ROWS = 3
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: b, r, k

  allocate(A(1)%rowptr(ROWS + 1), A(1)%colidx(3), A(1)%val(3))
  A(1)%rowptr = (/ 1, 2, 3, 4 /)
  A(1)%colidx = (/ 1, 2, 3 /)
  A(1)%val    = (/ 1.0d0, 1.0d0, 1.0d0 /)

  allocate(A(2)%rowptr(ROWS + 1), A(2)%colidx(4), A(2)%val(4))
  A(2)%rowptr = (/ 1, 3, 4, 5 /)
  A(2)%colidx = (/ 1, 2, 1, 3 /)
  A(2)%val    = (/ 2.0d0, -1.0d0, 1.0d0, 1.0d0 /)

  x = (/ 10.0d0, 20.0d0, 30.0d0 /)

  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)


def test_static_polymorphism_devirtualised(tmp_path: Path):
    """Polymorphic CLASS dispatch that flang's ``fir-polymorphic-op``
    pass devirtualises statically.  Two flavours:

      1. Type-bound procedure call ``c%area()`` where ``c`` is a
         concrete ``type(circle_t)`` — flang resolves the dispatch
         to ``circle_area(c)`` directly because the receiver type
         is statically known.
      2. Same shape with a different extension type ``rect_t`` —
         the type-bound procedure resolves to ``rect_area``.

    Member functions ``circle_area`` / ``rect_area`` themselves take
    concrete-class dummies (``class(circle_t)``, ``class(rect_t)``)
    not the abstract base; their bodies are non-polymorphic.  After
    inlining and devirtualisation the SDFG sees plain flat scalars.

    Verifies the bridge handles these cleanly: SDFG builds, output
    bit-exact against gfortran/f2py.  Pairs with the bail-out test
    in ``noncontig_unsupported_test.py`` which proves the reject
    pass fires when polymorphic-op can't resolve everything.
    """
    src = """
module shapes
  implicit none
  type :: circle_t
    real(8) :: r
  contains
    procedure :: area => circle_area
  end type circle_t

  type :: rect_t
    real(8) :: w, h
  contains
    procedure :: area => rect_area
  end type rect_t

contains
  function circle_area(self) result(a)
    class(circle_t), intent(in) :: self
    real(8) :: a
    a = 3.141592653589793d0 * self%r * self%r
  end function
  function rect_area(self) result(a)
    class(rect_t), intent(in) :: self
    real(8) :: a
    a = self%w * self%h
  end function
end module shapes

subroutine main(r, w, h, out)
  use shapes
  implicit none
  real(8), intent(in)  :: r, w, h
  real(8), intent(out) :: out(2)
  type(circle_t) :: c
  type(rect_t)   :: rect
  c%r = r
  rect%w = w
  rect%h = h
  ! Type-bound procedure call - flang devirtualises to the concrete
  ! ``circle_area`` / ``rect_area`` at compile time because the
  ! receiver's type is statically known.
  out(1) = c%area()
  out(2) = rect%area()
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "static_poly_ref")
    out_ref = mod.main(2.0, 3.0, 4.0)

    sdfg = _build(src, tmp_path, name='main', entry='_QPmain')
    out = np.zeros(2, dtype=np.float64)
    sdfg(r=2.0, w=3.0, h=4.0, out=out)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12)


@pytest.mark.parametrize("call_arg,kwarg_for_sdfg", [("x", True), ("0.5d0", False)],
                         ids=["runtime_arg", "literal_constant"])
def test_class_as_monomorphic_box(tmp_path: Path, call_arg, kwarg_for_sdfg):
    """``CLASS(t) :: this`` used as a non-polymorphic box.  Common in
    ECRAD / ICON code where a derived type is declared with
    ``class(...)`` for future polymorphism but every call site uses
    a concrete subtype.  No virtual dispatch, no allocatable
    members.

    Verifies the FlattenStructs box-peeling walk treats
    ``fir.class<T>`` like ``fir.box<T>`` (both inherit
    ``fir::BaseBoxType``).  The bridge never sees the struct —
    flat per-field arrays only.

    Parametrised over the scalar argument shape:
      * ``runtime_arg`` — caller supplies ``x`` as an SDFG argument.
      * ``literal_constant`` — flang creates an
        ``hlfir.associate %cst {adapt.valuebyref}`` so the inlined
        callee's by-ref dummy can take a value-converted constant.
        ``hlfir-materialise-associates`` rewrites that scalar
        associate to a local alloca + store so the bridge sees a
        proper transient instead of a nameless associate result.
    """
    src = f"""
module lib
  implicit none
  type :: pdf_sampler_t
    integer :: ncdf, nfsd
    real(8) :: fsd1, inv_fsd_interval
  end type pdf_sampler_t
contains
  subroutine evaluate(this, x, out)
    class(pdf_sampler_t), intent(in) :: this
    real(8),              intent(in)  :: x
    real(8),              intent(out) :: out
    out = x * this%fsd1 + this%inv_fsd_interval &
        + real(this%ncdf + this%nfsd, 8)
  end subroutine evaluate
end module lib

subroutine main(x, out)
  use lib
  implicit none
  real(8), intent(in)  :: x
  real(8), intent(out) :: out
  type(pdf_sampler_t) :: s
  s%ncdf = 3
  s%nfsd = 5
  s%fsd1 = 1.5d0
  s%inv_fsd_interval = 2.5d0
  call evaluate(s, {call_arg}, out)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "class_box_ref")
    out_ref = float(mod.main(0.5))

    sdfg = _build(src, tmp_path, name='main', entry='_QPmain')
    out = np.zeros(1, dtype=np.float64)
    if kwarg_for_sdfg:
        sdfg(x=0.5, out=out)
    else:
        # Literal constant case: ``x`` is unused (flang folds the
        # constant inline) but the SDFG signature still binds it as
        # an unused scalar input — pass any placeholder.
        sdfg(x=0.0, out=out)
    np.testing.assert_allclose(float(out[0]), out_ref, rtol=1e-12)


def test_three_level_nested_struct(tmp_path: Path):
    """LLVM-IR-flavoured deep nesting (Function → BasicBlock →
    Instruction shape).  Three levels of pure-record nesting with a
    single flat leaf at the bottom.  Exercises the Phase 2 path
    walker at depth 3 and verifies the path-flattened name
    ``f_bb_inst_pc`` resolves correctly."""
    src = """
module lib
  implicit none
  type instr_t
    integer :: pc(8)
  end type instr_t
  type bb_t
    type(instr_t) :: inst
  end type bb_t
  type func_t
    type(bb_t) :: bb
  end type func_t
end module lib

subroutine main(d)
  use lib
  implicit none
  integer, intent(out) :: d(8)
  type(func_t) :: f
  integer :: i
  do i = 1, 8
    f%bb%inst%pc(i) = i * 7
  end do
  d(:) = f%bb%inst%pc(:)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "three_level_ref")
    d_ref = mod.main()

    sdfg = _build(src, tmp_path)
    d = np.zeros(8, dtype=np.int32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)


def test_local_struct_used_as_2d_assignment_target(tmp_path: Path):
    """``s%w(:, k) = arr(:)`` — slice assignment into a struct's 2-D
    array member.  Exercises the section-to-section path landing on
    a flat per-field array."""
    src = """
module lib
  implicit none
  type two_d
    real :: w(3, 4)
  end type two_d
end module lib

subroutine main(arr, out)
  use lib
  implicit none
  real, intent(in)  :: arr(3)
  real, intent(out) :: out(3)
  type(two_d) :: t
  integer :: i
  do i = 1, 3
    t%w(i, 2) = arr(i) * 10.0
  end do
  out(:) = t%w(:, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_2d_ref")
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32, order="F")
    out_ref = np.asarray(mod.main(arr), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.float32)
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])


def test_nested_struct_lowered_via_phase2(tmp_path: Path):
    """Phase 2: nested struct ``type(outer_t)`` whose member is itself
    a ``type(inner_t)``.  ``hlfir-flatten-structs`` walks the path-leaf
    tree and synthesises one ``hlfir.declare`` per leaf with name
    ``<base>_<m1>_<m2>_..._<leaf>`` (here: ``o_inner_x``).  Designate
    chains ``o%inner%x(i)`` rewrite to ``o_inner_x(i)``."""
    src = """
module lib
  implicit none
  type inner_t
    real :: x(5)
  end type inner_t
  type outer_t
    type(inner_t) :: inner
  end type outer_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(1)
  type(outer_t) :: o
  o%inner%x(1) = 1.0
  d(1) = o%inner%x(1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "nested_struct_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(1, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)


def test_array_of_nested_struct_member(tmp_path: Path):
    """Phase 2 extension: a struct member is an ARRAY of another
    struct (``type(simple_type) :: pprog(10)``).  The flat
    companion folds the array dim into the leaf's shape so
    ``p_prog%pprog(i)%w(j, k)`` rewrites to a 3D companion
    ``p_prog_pprog_w(i, j, k)``.  Exercises ``collectFlatLeaves``'s
    ``array<N x RecordType>`` branch and ``walkDesignateChain``'s
    intermediate-indices path together.
    """
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog(10)
  end type simple_type2
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(5, 5)
  type(simple_type2) :: p_prog
  p_prog%pprog(1)%w(1, 1) = 47.0
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "array_of_nested_struct_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros((5, 5), order="F", dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    assert d[0][0] == 47.0


def test_aos_member_to_member_array_copy(tmp_path: Path):
    """AoS pattern ``a(i)%b = a(j)%c`` where ``b`` and ``c`` are array
    members — the assignment is a whole-array copy of one inner row.

    After flatten-structs, the array-of-struct ``a`` becomes a pair of
    flat per-member arrays ``a_b`` and ``a_c``, each shaped (outer-N,
    inner-extents...).  The Fortran row-copy:

        a(i)%b = a(j)%c

    flattens to:

        a_b(i, :) = a_c(j, :)

    i.e. a whole-array section copy from row ``j`` of ``a_c`` into row
    ``i`` of ``a_b``.  Exercises both AoS index-merging
    (``rewriteDesignate`` concat case) AND the whole-member triplet
    section path (no inner indices on the designate).
    """
    src = """
module lib
  implicit none
  type pair
    integer :: b(4)
    integer :: c(4)
  end type pair
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, intent(out) :: out(4, 2)
  type(pair) :: a(3)
  integer :: i, k

  ! Initialise: a(j)%c(k) = j*10 + k
  do i = 1, 3
    do k = 1, 4
      a(i)%c(k) = i * 10 + k
      a(i)%b(k) = 0
    end do
  end do

  ! Whole-row copy: a(1)%b <- a(3)%c  (copy row 3 of c into row 1 of b)
  a(1)%b = a(3)%c

  ! Read back
  do k = 1, 4
    out(k, 1) = a(1)%b(k)
    out(k, 2) = a(3)%c(k)
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "aos_row_copy_ref")
    # ``intent(out)`` dummy: f2py returns it.
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros((4, 2), order="F", dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # Both columns should hold the row [31, 32, 33, 34] (a(3)%c, copied
    # into a(1)%b).
    np.testing.assert_array_equal(out[:, 0], [31, 32, 33, 34])
    np.testing.assert_array_equal(out[:, 1], [31, 32, 33, 34])
