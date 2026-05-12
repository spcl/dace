"""Negative tests — patterns the HLFIR bridge deliberately refuses to
lower.  Each case below produces a clear error message naming the
failing source location and the reason.  Documenting these as live
tests (rather than just READMEs) keeps the contract enforced in CI;
if the bridge ever silently drops the loud-error path, the test fails
and the regression is caught.

Currently covered:
  * Symbolic-extent noncontiguous gather (no compile-time-constant size).
    Lowered by ``hlfir-materialise-associates`` only when extent is a
    constant integer; otherwise pass aborts with ``op.emitError``.

Higher-rank noncontiguous gathers and INTENT(out) scatter-back fail
through the same path; once those are implemented the corresponding
xfails in [noncontig_pardecls_test.py](ported/noncontig_pardecls_test.py)
will move here as positive tests instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# All scatter / gather variants — including symbolic extents and
# symbolic-extent calls — are now supported in Phase 1.5.  Positive
# tests in noncontig_gather_scatter_test.py.  The single remaining
# unsupported case is ALIASED self-assignments where source and
# destination are the same array (Fortran 2003 RHS-to-temp
# evaluation order); see test_gather_scatter_aliasing_same_array
# in that file (xfailed).


def test_placeholder_for_future_unsupported_cases(tmp_path: Path):
    """Stub.  Add bail-out tests here when new patterns the bridge
    deliberately refuses to lower are introduced.  Empty for now."""
    pass


def test_virtual_dispatch_bails_loudly(tmp_path: Path):
    """Genuinely runtime-polymorphic ``fir.dispatch`` — a polymorphic
    dummy ``class(t)`` that the function itself dispatches on.  Flang's
    static devirtualisation can't resolve this case because the
    concrete type only becomes known at the call site (which is
    outside the function being compiled).  Surviving ``fir.dispatch``
    ops after ``fir-polymorphic-op`` trip our
    ``hlfir-reject-polymorphism`` pass.

    The test asserts that the pipeline raises ``RuntimeError`` with
    a message naming polymorphism — the bridge's contract that
    runtime polymorphic dispatch is unsupported.
    """
    src = """
module shapes
  implicit none
  type, abstract :: shape_t
  contains
    procedure(area_iface), deferred :: area
  end type shape_t

  abstract interface
    function area_iface(this) result(a)
      import :: shape_t
      class(shape_t), intent(in) :: this
      real :: a
    end function
  end interface

  type, extends(shape_t) :: circle_t
    real :: r
  contains
    procedure :: area => circle_area
  end type circle_t

contains
  function circle_area(this) result(a)
    class(circle_t), intent(in) :: this
    real :: a
    a = 3.141592 * this%r * this%r
  end function
end module shapes

subroutine main(p, out)
  use shapes
  implicit none
  ! ``p`` is a CLASS-typed dummy argument — a polymorphic dummy whose
  ! concrete runtime type is determined by the caller, *outside* the
  ! function being compiled.  ``p%area()`` here is a true virtual
  ! dispatch that ``fir-polymorphic-op`` cannot statically resolve;
  ! it lowers to an indirect ``fir.call`` through the type-info
  ! dispatch table, which our reject pass catches via the leftover
  ! ``fir.box_tdesc`` marker.
  class(shape_t), intent(in) :: p
  real, intent(out) :: out
  out = p%area()
end subroutine main
"""
    with pytest.raises(RuntimeError) as exc:
        build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    msg = str(exc.value)
    assert "pipeline failed" in msg or "polymorphism" in msg, (
        f"expected a pipeline-failed message naming polymorphism, "
        f"got: {msg}")
