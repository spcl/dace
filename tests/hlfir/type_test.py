"""Verbatim port of f2dace/dev:tests/fortran/type_test.py."""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang
from _helpers import xfail

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_basic_type(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: s
  s%w(1, 1, 1) = 5.5
  d(2, 1) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_basic_type2(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: w(5, 5, 5), z(5)
    integer:: a
  end type simple_type
  type comlex_type
    type(simple_type):: s
    real:: b
  end type comlex_type
  type meta_type
    type(comlex_type):: cc
    real:: omega
  end type meta_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: s(3)
  type(comlex_type) :: c
  type(meta_type) :: m
  c%b = 1.0
  c%s%w(1, 1, 1) = 5.5
  m%cc%s%a = 17
  s(1)%w(1, 1, 1) = 5.5 + c%b
  d(2, 1) = c%s%w(1, 1, 1) + s(1)%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 12)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_symbol(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5)
    integer:: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d, st)
end subroutine main

subroutine internal_function(d, st)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: st
  real bob(st%a)
  bob(1) = 5.5
  d(2, 1) = 2*bob(1)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_pardecl(tmp_path):
    """Parametric struct-dim local (``real bob2(st%a)``) plus a
    PARAMETER-sized companion (``real bob(n)``).

    Originally xfailed claiming "parametric array dimension not lowered",
    but the actual blocker was test-side bugs: the test passed
    ``np.full([4, 5])`` for a ``d(5, 5)`` dummy (shape mismatch) AND
    wrote ``d(:, 1) = bob(1) + bob2`` — illegal Fortran since LHS is
    rank-1 length 5 and RHS broadcasts to length 10 (the size of
    ``bob2``).  Flang lowered the illegal assign by writing 10
    elements column-major, spilling into column 2 of ``d`` — undefined
    behaviour that made the assertion ``a[1, 1] == 42`` fail (the
    value got overwritten to 5.5).

    Fixes (preserve original intent):
      * Truncate ``bob2`` to length 5 via the slice ``bob2(1:5)`` so the
        assignment is valid Fortran.  ``bob2(st%a=10)`` retains its
        full declared extent so the parametric-dim feature is still
        exercised.
      * ``np.full([5, 5])`` matches the dummy shape.

    Parametric struct dim works today (Phase 5a + 6); this test is the
    cross-subroutine variant of ``derived_type_test.py::
    test_parametric_dim_via_inlined_subprogram``.
    """
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5, 5, 5)
    integer:: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d, st)
end subroutine main

subroutine internal_function(d, st)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type) :: st

  integer, parameter :: n = 5
  real bob(n)
  real bob2(st%a)
  bob(1) = 5.5
  bob2(:) = 0
  bob2(1) = 5.5
  d(:, 1) = bob(1) + bob2(1:5)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 11)
    assert (a[1, 0] == 5.5)
    assert (a[2, 0] == 5.5)
    assert (a[1, 1] == 42)


def test_fortran_frontend_type_struct(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real:: z(5, 5, 5)
    integer:: a
    !real, allocatable :: unknown(:)
    !INTEGER :: unkown_size
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%a = 10
  call internal_function(d,st)
end subroutine main

subroutine internal_function(d,st)
  use lib
  implicit none
  !! WHAT DOES THIS MEAN?
  ! st.a.shape = [st.a_size]
  real d(5, 5)
  type(simple_type) :: st
  real bob(st%a)
  integer, parameter :: n = 5
  real BOB2(n)
  bob(1) = 5.5
  bob2(1) = 5.5
  st%z(1, :, 2:3) = bob(1)
  d(2, 1) = bob(1) + bob2(1)
end subroutine internal_function
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_circular_type(tmp_path):
    src = """
module lib
  implicit none
  type a_t
    real :: w(5, 5, 5)
    type(b_t), pointer :: b
  end type a_t
  type b_t
    type(a_t) :: a
    integer :: x
  end type b_t
  type c_t
    type(d_t), pointer :: ab
    integer :: xz
  end type c_t
  type d_t
    type(c_t) :: ac
    integer :: xy
  end type d_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(a_t) :: s
  type(b_t) :: b(3)
  s%w(1, 1, 1) = 5.5
  ! s%b=>b(1)
  ! s%b%a=>s
  b(1)%x = 1
  d(2, 1) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("parent-pointer round-trip (s%b%a%w === s%w) not collapsed — needs a "
       "CollapseParentPointer pre-pass that rewrites the designate chain "
       "before FlattenStructs.  Structural-candidacy detection: `a_t.b: "
       "pointer<b_t>` is a candidate because b_t's embedded-field closure "
       "contains a_t (via b_t.a).  The rewrite matches chain prefix "
       "`<root>%b%a` (pointer-chase + named back-ref) and re-anchors at "
       "<root>.  See project_circular_type_plan.md.")
def test_fortran_frontend_circular_type_parent_pointer_chase(tmp_path):
    """End-to-end correctness for the parent-pointer round-trip.

    Contract the (deferred) rewrite would enforce: ``s%b%a === s`` —
    the pointer-chase through ``b`` followed by the embedded
    back-reference ``a`` returns to the same ``a_t`` instance.  Under
    that contract, ``s%b%a%w(...) === s%w(...)``.

    Numerical check: compare the SDFG's output to a gfortran/f2py-
    compiled reference of the same Fortran source on multiple writes
    + reads through the parent-pointer chain.  Xfails today at SDFG
    build (no rewrite pass); when the rewrite lands, both paths
    succeed and the assertion catches semantic regressions in the
    rewrite itself.
    """
    src = """
subroutine kernel(d, x, y, z)
  implicit none
  type a_t
    real :: w(5, 5, 5)
    type(b_t), pointer :: b
  end type a_t
  type b_t
    type(a_t), pointer :: a
    integer :: x
  end type b_t

  real, intent(in)    :: x, y, z
  real, intent(inout) :: d(3)
  type(a_t), target :: s
  type(b_t), target :: bb

  s%b => bb
  bb%a => s

  ! Write three distinct values into s%w at distinct positions.
  s%w(1, 1, 1) = x
  s%w(2, 1, 1) = y
  s%w(1, 2, 1) = z

  ! Read them back through the parent-pointer round-trip.  Under
  ! the contract s%b%a === s, these must equal the writes above.
  d(1) = s%b%a%w(1, 1, 1)
  d(2) = s%b%a%w(2, 1, 1)
  d(3) = s%b%a%w(1, 2, 1)
end subroutine kernel
"""
    # SDFG via HLFIR bridge — xfails today at build (no rewrite pass).
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='kernel').build()

    # f2py reference — always builds; serves as the oracle once the
    # bridge clears the rewrite.
    ref = f2py_compile(src, tmp_path / "ref", "parent_pointer_ref")

    rng = np.random.default_rng(0)
    x, y, z = (np.float32(rng.standard_normal()) for _ in range(3))

    d_sdfg = np.zeros(3, dtype=np.float32)
    d_ref = np.zeros(3, dtype=np.float32)

    sdfg(d=d_sdfg, x=x, y=y, z=z)
    ref.kernel(d_ref, x, y, z)

    np.testing.assert_allclose(d_sdfg, d_ref, rtol=0, atol=0)
    # Sanity: contract requires d to carry the inputs back verbatim.
    np.testing.assert_allclose(d_ref, [x, y, z], rtol=0, atol=0)


def test_fortran_frontend_type_in_call(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type), target :: s
  real, pointer :: tmp(:, :, :)
  tmp => s%w
  tmp(1, 1, 1) = 11.0
  d(2, 1) = max(1.0, tmp(1, 1, 1))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_array(tmp_path):
    src = """
module lib
  implicit none

  type simple_type3
    integer :: a
  end type simple_type3

  type simple_type2
    type(simple_type3) :: w(7:12, 8:13)
  end type simple_type2

  type simple_type
    type(simple_type2) :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type) :: s
  call f2(s)
  d(1, 1) = s%name%w(8, 10)%a
end subroutine main

subroutine f2(s)
  use lib
  implicit none
  type(simple_type) :: s
  s%name%w(8, 10)%a = 42
end subroutine f2
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_array2(tmp_path):
    src = """
module lib
  implicit none

  type simple_type3
    integer :: a
  end type simple_type3

  type simple_type2
    type(simple_type3) :: w(7:12, 8:13)
    integer :: wx(7:12, 8:13)
  end type simple_type2

  type simple_type
    type(simple_type2) :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  integer :: x(3, 3, 3)
  type(simple_type) :: s
  call f2(s, x)
  !d(1,1) = s%name%w(8, x(3,3,3))%a
  d(1, 2) = s%name%wx(8, x(3, 3, 3))
end subroutine main

subroutine f2(s, x)
  use lib
  implicit none
  type(simple_type) :: s
  integer :: x(3, 3, 3)
  x(3, 3, 3) = 10
  !s%name%w(8,x(3,3,3))%a = 42
  s%name%wx(8, x(3, 3, 3)) = 43
end subroutine f2
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_pointer(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real d(5, 5)
  type(simple_type), target :: s
  real, dimension(:, :, :), pointer :: tmp
  tmp => s%w
  tmp(1, 1, 1) = 11.0
  d(2, 1) = max(1.0, tmp(1, 1, 1))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


@xfail("`type(t), allocatable :: pprog(:)` member — needs `LiftAllocArrayOfRecords` "
       "pre-pass to rewrite the chain into accesses on a synthesised top-level "
       "companion of shape `(NPPROG, *leaf_shape)`.  See the active plan in "
       "`~/.claude/plans/vectorized-fluttering-pumpkin.md`.")
def test_fortran_frontend_type_arg(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real, pointer, contiguous :: w(:, :)
  end type simple_type
  type simple_type2
    type(simple_type), allocatable :: pprog(:)
  end type simple_type2
contains
  subroutine f2(stuff)
    type(simple_type) :: stuff
    call deepest(stuff%w)
  end subroutine f2

  subroutine deepest(my_arr)
    real :: my_arr(:, :)
    my_arr(1, 1) = 42
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  call f2(p_prog%pprog(1))
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_arg2(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog(10)
  end type simple_type2
contains
  subroutine deepest(my_arr, d)
    real :: my_arr(:, :)
    real :: d(5, 5)
    my_arr(1, 1) = 5.5
    d(1, 1) = my_arr(1, 1)
  end subroutine deepest
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: d(5, 5)
  type(simple_type2) :: p_prog
  integer :: i
  i = 1

  !p_prog%pprog(1)%w(1,1) = 5.5
  call deepest(p_prog%pprog(i)%w, d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)


def test_fortran_frontend_type_view(tmp_path):
    src = """
module lib
  implicit none
  type simple_type
    real :: z(3, 3)
    integer :: a
  end type simple_type
contains
  subroutine internal_function(d, sta)
    real d(5, 5)
    real sta(:, :)
    d(2, 1) = 2*sta(1, 1)
  end subroutine internal_function
end module lib

subroutine main(d)
  use lib
  implicit none
  type(simple_type) :: st
  real :: d(5, 5)
  st%z(1, 1) = 5.5
  call internal_function(d, st%z)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    # Should NOT need to bind ``sta_d0`` / ``sta_d1`` — ``st_z`` is
    # concretely (3, 3) and ``sta`` is just an inlined alias.  The
    # SDFG signature surfaces these synth symbols today only because
    # ``asAssumedShapeAlias`` doesn't trace through a flattened-field
    # designate; once that's fixed they should disappear.
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_func_type_prefix(tmp_path):
    src = """
module lib
  implicit none
contains
  real function custom_sum(d)
    real :: d(5, 5)
    integer :: i, j
    do i = 1, 5
      do j = 1, 5
        custom_sum = custom_sum + d(i, j)
      end do
    end do
  end function custom_sum
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: norm
  real :: d(5, 5)
  d(1, 1) = custom_sum(d) ** 2.0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)


def test_fortran_frontend_func_type_body(tmp_path):
    src = """
module lib
  implicit none
contains
  function custom_sum(d)
    real :: custom_sum
    real :: d(5, 5)
    integer :: i, j
    do i = 1, 5
      do j = 1, 5
        custom_sum = custom_sum + d(i, j)
      end do
    end do
  end function custom_sum
end module lib

subroutine main(d)
  use lib
  implicit none
  real :: norm
  real :: d(5, 5)
  d(1, 1) = custom_sum(d) ** 2.0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([5, 5], 1, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 625)


# ---------------------------------------------------------------------------
# Alloc-array-of-records member — the `LiftAllocArrayOfRecords` pre-pass
# target.  Both tests below xfail today; flip to passing when the pass lands.
# ---------------------------------------------------------------------------


@xfail("LiftAllocArrayOfRecords pre-pass not yet implemented.  Simple "
       "stand-alone version of the alloc-array-of-records pattern: const "
       "index, single pointer-array leaf, no inlining/callee aliasing.  "
       "Smallest reproducer for the lift transformation.")
def test_lift_alloc_array_of_records_simple(tmp_path):
    """Minimal unit test for the LiftAllocArrayOfRecords pre-pass.

    Structure: outer struct holds an alloc-array of records, each
    record holds one pointer-array member.  Write a value through a
    const-index access, read it back via the same path, assert
    against a gfortran/f2py reference compiled from the same source.
    """
    src = """
module lift_simple_mod
  implicit none
  type inner_t
    real(kind=8), pointer, contiguous :: w(:, :)
  end type inner_t
  type outer_t
    type(inner_t), allocatable :: items(:)
  end type outer_t
end module lift_simple_mod

subroutine kernel(d, val)
  use lift_simple_mod
  implicit none
  real(kind=8), intent(in)    :: val
  real(kind=8), intent(inout) :: d(2)
  type(outer_t) :: s
  ! Storage laid out so the LAST dim selects the record — keeps each
  ! ``storage(:, :, jg)`` slice contiguous for the pointer rebind.
  real(kind=8), target :: storage(4, 5, 3)

  ! Allocate the AoS spine + bind each record's pointer to a slice.
  allocate(s%items(3))
  s%items(1)%w => storage(:, :, 1)
  s%items(2)%w => storage(:, :, 2)
  s%items(3)%w => storage(:, :, 3)

  ! Write a value, read it back through the AoS chain.
  s%items(2)%w(1, 1) = val
  d(1) = s%items(2)%w(1, 1)
  d(2) = s%items(2)%w(1, 1) * 2.0d0

  deallocate(s%items)
end subroutine kernel
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='kernel').build()
    ref = f2py_compile(src, tmp_path / "ref", "lift_simple_ref")

    val = np.float64(3.5)
    d_sdfg = np.zeros(2, dtype=np.float64)
    d_ref = np.zeros(2, dtype=np.float64)

    sdfg(d=d_sdfg, val=val)
    ref.kernel(d_ref, val)

    np.testing.assert_allclose(d_sdfg, d_ref, rtol=0, atol=0)
    np.testing.assert_allclose(d_ref, [val, val * 2.0], rtol=0, atol=0)


@xfail("LiftAllocArrayOfRecords pre-pass not yet implemented.  "
       "ICON-derived snippet: `t_nh_state` holds `TYPE(t_nh_prog), "
       "ALLOCATABLE :: prog(:)` where each `t_nh_prog` carries multiple "
       "pointer-array members.  Access pattern mirrors "
       "`mo_solve_nonhydro.f90:1576`: `p_nh%prog(nvar)%rho(jc, jk, jb)` "
       "with runtime `nvar`.")
def test_lift_alloc_array_of_records_icon_pattern(tmp_path):
    """ICON-derived test for the LiftAllocArrayOfRecords pre-pass.

    Mirrors the canonical solve_nh pattern: outer state struct holds
    an alloc-array of prognostic records, each carrying several
    rank-3 pointer-array members (rho, theta_v, w).  Access uses a
    runtime element index (`nvar`), exactly as ICON's two-time-level
    scheme does.  Compares the SDFG output against a gfortran/f2py
    reference compiled from the same source.
    """
    src = """
module lift_icon_mod
  implicit none
  type t_nh_prog
    real(kind=8), pointer, contiguous :: rho(:, :, :)
    real(kind=8), pointer, contiguous :: theta_v(:, :, :)
    real(kind=8), pointer, contiguous :: w(:, :, :)
  end type t_nh_prog
  type t_nh_state
    type(t_nh_prog), allocatable :: prog(:)
  end type t_nh_state
end module lift_icon_mod

subroutine kernel(out_rho, out_theta, out_w, nvar, wgt, rho_val, theta_val, w_val)
  use lift_icon_mod
  implicit none
  integer, intent(in)         :: nvar
  real(kind=8), intent(in)    :: wgt, rho_val, theta_val, w_val
  real(kind=8), intent(inout) :: out_rho(2, 3, 2)
  real(kind=8), intent(inout) :: out_theta(2, 3, 2)
  real(kind=8), intent(inout) :: out_w(2, 3, 2)

  type(t_nh_state) :: p_nh
  ! Trailing-dim record index keeps each ``store(:, :, :, n)`` slice
  ! contiguous for the pointer rebind.
  real(kind=8), target :: rho_store(2, 3, 2, 3)
  real(kind=8), target :: theta_store(2, 3, 2, 3)
  real(kind=8), target :: w_store(2, 3, 2, 3)
  integer :: jc, jk, jb, n

  allocate(p_nh%prog(3))
  do n = 1, 3
    p_nh%prog(n)%rho     => rho_store(:, :, :, n)
    p_nh%prog(n)%theta_v => theta_store(:, :, :, n)
    p_nh%prog(n)%w       => w_store(:, :, :, n)
  end do

  do jb = 1, 2
    do jk = 1, 3
      do jc = 1, 2
        p_nh%prog(nvar)%rho(jc, jk, jb)     = rho_val   + 0.01d0 * (jc + jk + jb)
        p_nh%prog(nvar)%theta_v(jc, jk, jb) = theta_val + 0.02d0 * (jc + jk + jb)
        p_nh%prog(nvar)%w(jc, jk, jb)       = w_val     + 0.03d0 * (jc + jk + jb)
      end do
    end do
  end do

  do jb = 1, 2
    do jk = 1, 3
      do jc = 1, 2
        out_rho(jc, jk, jb)   = wgt * p_nh%prog(nvar)%rho(jc, jk, jb)
        out_theta(jc, jk, jb) = wgt * p_nh%prog(nvar)%theta_v(jc, jk, jb)
        out_w(jc, jk, jb)     = wgt * p_nh%prog(nvar)%w(jc, jk, jb)
      end do
    end do
  end do

  deallocate(p_nh%prog)
end subroutine kernel
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='kernel').build()
    ref = f2py_compile(src, tmp_path / "ref", "lift_icon_ref")

    rng = np.random.default_rng(0)
    nvar = np.int32(2)
    wgt = np.float64(rng.standard_normal())
    rho_val = np.float64(rng.standard_normal())
    theta_val = np.float64(rng.standard_normal())
    w_val = np.float64(rng.standard_normal())

    shape = (2, 3, 2)
    out_rho_sdfg = np.zeros(shape, dtype=np.float64, order='F')
    out_theta_sdfg = np.zeros(shape, dtype=np.float64, order='F')
    out_w_sdfg = np.zeros(shape, dtype=np.float64, order='F')
    out_rho_ref = np.zeros(shape, dtype=np.float64, order='F')
    out_theta_ref = np.zeros(shape, dtype=np.float64, order='F')
    out_w_ref = np.zeros(shape, dtype=np.float64, order='F')

    sdfg(out_rho=out_rho_sdfg,
         out_theta=out_theta_sdfg,
         out_w=out_w_sdfg,
         nvar=nvar,
         wgt=wgt,
         rho_val=rho_val,
         theta_val=theta_val,
         w_val=w_val)
    ref.kernel(out_rho_ref, out_theta_ref, out_w_ref, nvar, wgt, rho_val, theta_val, w_val)

    np.testing.assert_allclose(out_rho_sdfg, out_rho_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_theta_sdfg, out_theta_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_w_sdfg, out_w_ref, rtol=1e-12, atol=1e-12)
