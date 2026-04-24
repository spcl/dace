"""Pass→bridge→FlattenPlan round-trip.  Verifies that
``hlfir-flatten-structs`` stamps a structurally correct
``hlfir.flatten_plan`` attribute that the bridge decodes back into a
usable ``FlattenPlan`` dataclass.
"""
from __future__ import annotations

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
    """Static-shape struct with two real members → one FlattenEntry,
    aliasable, rank-2 shape exprs."""
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
    assert len(plan.entries) == 1
    e = plan.entries[0]
    assert e.outer_expr == "st"
    assert e.writeback_intent == "inout"
    r = e.recipe
    assert r.flat_names == ("st_u", "st_v")
    assert r.read_exprs == ("st%u($i1, $i2)", "st%v($i1, $i2)")
    assert r.rank == 2
    assert r.aliasable is True
    assert r.write_expr == ""
    assert r.scratch_dtype == "float64"
    assert len(r.shape_exprs) == 2
    assert r.shape_exprs[0].startswith("size(st%")
    assert "dim=1" in r.shape_exprs[0]
    assert "dim=2" in r.shape_exprs[1]


def test_intent_in_carries_through(tmp_path: Path):
    """A read-only struct dummy should produce writeback_intent='in'."""
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
    assert len(plan.entries) == 1
    assert plan.entries[0].writeback_intent == "in"
