"""E10 regression: ``fir.box_dims`` lower bound leaking ``?`` into a
loop-bound expression.

An assumed-shape dummy ``a(:)`` lowers to ``hlfir.declare ... :
!fir.box<...>`` (no shape operand); its bounds are read at runtime via
``fir.box_dims`` (a 3-tuple lower-bound/extent/stride).  Using it with
``do i = lbound(a,1), ubound(a,1)`` makes the ``ubound`` lowering
``box_dims#0 + box_dims#1 - 1`` flow into the loop-bound expression.

``fir.box_dims`` is handled in ``expressions.cpp`` / ``assigns.cpp``
but NOT in ``control_flow.cpp``'s loop-bound ``buildExpr``: result
``#0`` (the lower bound) fell through to the generic ``"?"`` sentinel,
producing ``((? - 1) + a_d0)`` which DaCe's ``unique_loop_iterators``
then ``ast.parse``s -> ``SyntaxError``.

This test pins the fix: the assumed-shape sum builds and equals the
closed-form ``sum(a)`` (f2py's crackfortran can't wrap an
assumed-shape dummy, so the non-transformed reference is the exact
closed-form result; the build itself failing is the repro).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ``a(10:)`` -> explicit local lower bound -> ``fir.shift %c10`` shape
# operand (assumed-shape).  ``ubound(a,1)`` lowers to
# ``box_dims#0(lb) + box_dims#1(extent) - 1``; the box_dims lower
# bound must become the ``offset_a_d0`` symbol, not the ``?``
# sentinel, in the loop-bound expression.
_SRC = """
subroutine sum_as(a, out)
  implicit none
  real(8), intent(in)  :: a(10:)
  real(8), intent(out) :: out
  integer :: i
  out = 0.0d0
  do i = lbound(a, 1), ubound(a, 1)
    out = out + a(i)
  end do
end subroutine sum_as
"""


def _bind_free_syms(sdfg, n: int) -> dict:
    """Bind assumed-shape free symbols from the arglist.

    :param sdfg: built SDFG.
    :param n: the actual extent of ``a``.
    :returns: kwargs for every ``a_d<i>`` extent (= n) and
        ``offset_a_d<i>`` lower bound (= 1, the assumed-shape default).
    """
    out = {}
    for k in sdfg.arglist():
        if k.startswith("offset_") and k.endswith(tuple(f"_d{i}" for i in range(4))):
            out[k] = np.int64(1)
        elif k == "a" or k == "out":
            continue
        elif k.endswith(tuple(f"_d{i}" for i in range(4))):
            out[k] = np.int64(n)
    return out


def test_assumed_shape_lbound_ubound_loop(tmp_path: Path):
    """``do i = lbound(a,1), ubound(a,1); out += a(i)`` over an
    assumed-shape ``a(10:)``.  Builds (no ``?`` leak) and equals
    ``sum(a)``."""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, d, name="sum_as", entry="_QPsum_as").build()
    sdfg.validate()

    n = 7
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.random(n))
    out = np.zeros(1, dtype=np.float64, order="F")
    sdfg(a=a.copy(order="F"), out=out, **_bind_free_syms(sdfg, n))

    np.testing.assert_allclose(out[0], a.sum(), rtol=1e-12, atol=1e-12)
