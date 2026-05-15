"""Pass->bridge->FlattenPlan round-trip.  Verifies that
``hlfir-flatten-structs`` stamps a structurally correct
``hlfir.flatten_plan`` attribute that the bridge decodes back into a
usable ``FlattenPlan`` dataclass.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _util import have_flang  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dace" / "frontend" / "hlfir"))
from build_bridge import hb  # noqa: E402

from dace.frontend.hlfir.bindings import FlattenPlan  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_FLANG = "flang-new-21"


def _plan_from_fortran(src: str, tmp_path: Path) -> FlattenPlan:
    f90 = tmp_path / "src.f90"
    f90.write_text(src)
    hlfir = tmp_path / "src.hlfir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(f90), "-o", str(hlfir)])
    m = hb.HLFIRModule()
    assert m.parse_file(str(hlfir))
    m.run_passes("hlfir-flatten-structs")
    return FlattenPlan.from_dict(m.get_flatten_plan())


def test_no_struct_gives_empty_plan(tmp_path: Path):
    plan = _plan_from_fortran(
        """
subroutine kernel(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = dble(i)
  end do
end subroutine
""", tmp_path)
    assert plan.entries == ()


def test_two_real_array_struct_emits_aliased_recipe(tmp_path: Path):
    """Static-shape struct with two real members -> one FlattenEntry
    per member (aliasable, rank-2 shape exprs).

    The bridge emits one entry per member rather than bundling every
    member into a single multi-flat recipe so mixed-dtype structs
    (e.g. ``complex(c_double)`` next to ``real(c_double)``) carry the
    correct per-flat ``scratch_dtype``; the emitter walks
    ``plan.entries`` and renders each with its own dtype.
    """
    plan = _plan_from_fortran(
        """
module state_mod
  use iso_c_binding
  type, public :: state_t
    real(c_double) :: u(10, 20)
    real(c_double) :: v(10, 20)
  end type
end module

subroutine kernel(st)
  use iso_c_binding
  use state_mod
  implicit none
  type(state_t), intent(inout) :: st
  integer :: i, j
  do j = 1, 20
    do i = 1, 10
      st%u(i, j) = st%u(i, j) + 1.0d0
      st%v(i, j) = st%v(i, j) * 2.0d0
    end do
  end do
end subroutine
""", tmp_path)
    assert len(plan.entries) == 2
    e_u, e_v = plan.entries
    assert e_u.outer_expr == "st%u"
    assert e_v.outer_expr == "st%v"
    assert e_u.writeback_intent == "inout"
    assert e_v.writeback_intent == "inout"
    assert e_u.recipe.flat_names == ("st_u", )
    assert e_v.recipe.flat_names == ("st_v", )
    assert e_u.recipe.read_exprs == ("st%u($i1, $i2)", )
    assert e_v.recipe.read_exprs == ("st%v($i1, $i2)", )
    for r in (e_u.recipe, e_v.recipe):
        assert r.rank == 2
        assert r.aliasable is True
        assert r.write_expr == ""
        assert r.scratch_dtype == "float64"
        assert len(r.shape_exprs) == 2
        assert r.shape_exprs[0].startswith("size(st%")
        assert "dim=1" in r.shape_exprs[0]
        assert "dim=2" in r.shape_exprs[1]


def test_intent_in_carries_through(tmp_path: Path):
    """A read-only struct dummy should produce writeback_intent='in'
    on every per-member entry."""
    plan = _plan_from_fortran(
        """
module state_mod
  use iso_c_binding
  type, public :: ro_t
    real(c_double) :: a(8, 8)
    real(c_double) :: b(8, 8)
  end type
end module

subroutine sink(ro, acc)
  use iso_c_binding
  use state_mod
  implicit none
  type(ro_t),  intent(in)    :: ro
  real(8),     intent(inout) :: acc
  integer :: i, j
  do j = 1, 8
    do i = 1, 8
      acc = acc + ro%a(i, j) + ro%b(i, j)
    end do
  end do
end subroutine
""", tmp_path)
    assert len(plan.entries) == 2
    assert all(e.writeback_intent == "in" for e in plan.entries)


def test_aos_allocatable_dummy_emits_aos_alloc_recipe(tmp_path: Path):
    """Phase 5c-B (true SDFG boundary): an AoS+allocatable struct
    dummy yields one ``aos_alloc=True`` recipe per allocatable
    member, with a ``cap_<base>_<member>`` symbol on the inner dim
    and rank == outer_rank + 1.
    """
    plan = _plan_from_fortran(
        """
module aos_mod
  implicit none
  type, public :: row_t
    real(8), allocatable :: w(:)
  end type
end module

subroutine kernel(A, n, m, out)
  use aos_mod
  implicit none
  type(row_t), intent(inout) :: A(2)
  integer, intent(in) :: n, m
  real(8), intent(out) :: out
  integer :: i, j
  do i = 1, n
    do j = 1, m
      A(i)%w(j) = A(i)%w(j) * 2.0d0
    end do
  end do
  out = A(1)%w(1)
end subroutine
""", tmp_path)
    # Exactly one entry  --  the allocatable member.  No regular entry
    # (every member excluded from the regular path).  Flang lowercases
    # all identifiers, so the outer name comes out as ``a`` and the
    # flat companion as ``a_w``.
    assert len(plan.entries) == 1
    e = plan.entries[0]
    assert e.outer_expr == "a"
    assert e.writeback_intent == "inout"
    r = e.recipe
    assert r.aos_alloc is True
    assert r.aliasable is False
    assert r.flat_names == ("a_w", )
    assert r.cap_symbol == "cap_a_w"
    assert r.rank == 2
    assert r.shape_exprs == ("size(a, dim=1)", "cap_a_w")
    assert r.read_exprs == ("a($i1)%w($i2)", )
    assert r.scratch_dtype == "float64"


def test_mixed_aos_alloc_and_plain_member_split_into_two_entries(tmp_path: Path):
    """A struct with both an allocatable and a plain (static-shape)
    member  --  Phase 5c-B emits a separate ``aos_alloc=True`` recipe
    per allocatable member, AND a regular aliasable recipe covering
    the non-allocatable members.
    """
    plan = _plan_from_fortran(
        """
module aos_mod
  implicit none
  type, public :: row2_t
    real(8), allocatable :: w(:)
    integer :: tag
  end type
end module

subroutine kernel(A, n, m, out)
  use aos_mod
  implicit none
  type(row2_t), intent(inout) :: A(3)
  integer, intent(in) :: n, m
  real(8), intent(out) :: out
  integer :: i, j
  do i = 1, n
    A(i)%tag = i
    do j = 1, m
      A(i)%w(j) = A(i)%w(j) + real(A(i)%tag, kind=8)
    end do
  end do
  out = A(1)%w(1)
end subroutine
""", tmp_path)
    # Two entries: one aos_alloc for w, one regular for tag.  Flang
    # lowercases identifiers  --  the flat companions are ``a_w`` / ``a_tag``.
    aos_entries = [e for e in plan.entries if e.recipe.aos_alloc]
    plain_entries = [e for e in plan.entries if not e.recipe.aos_alloc]
    assert len(aos_entries) == 1
    assert len(plain_entries) == 1
    aos = aos_entries[0].recipe
    assert aos.flat_names == ("a_w", )
    assert aos.cap_symbol == "cap_a_w"
    plain = plain_entries[0].recipe
    assert plain.flat_names == ("a_tag", )
    assert "tag" in plain.read_exprs[0]
