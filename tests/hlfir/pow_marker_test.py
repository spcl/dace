"""Exponentiation lowering — Flang's ``**`` becomes a Python ``**``
in the tasklet body.

All four Flang variants (``math.fpowi`` / ``math.powf`` / ``math.powi``
/ ``math.ipowi``) collapse to the same ``a ** b`` form.  A downstream
SDFG-level simplify pass rewrites ``**`` based on the tasklet's
input / output types — no extra variant markers are propagated at
this layer.  These tests assert that the bridge surfaces ``**`` and
that ``hlfir.no_reassoc`` passes through transparently (otherwise
the inner ``math.fpowi`` would be stranded as ``?``).  Each test is
also paired with an e2e numerical check against a numpy reference —
the structural ``**``-in-tasklet check guards the lowering shape;
the numerical check guards that ``**`` evaluates to the right value
at run time.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _tasklet_codes(sdfg) -> list[str]:
    from dace.sdfg.state import SDFGState, ControlFlowRegion
    from dace.sdfg import nodes as nd
    out = []

    def walk(region):
        if isinstance(region, SDFGState):
            for n in region.nodes():
                if isinstance(n, nd.Tasklet):
                    out.append(n.code.as_string)
        if isinstance(region, ControlFlowRegion):
            for n in region.nodes():
                walk(n)

    walk(sdfg)
    return out


def test_fpowi_surfaces_as_python_pow(tmp_path: Path):
    """``r = x**2`` (float ** integer literal, Flang emits
    ``math.fpowi``) lands in the tasklet as ``r = (x ** 2)`` and
    evaluates to ``x*x`` at run time."""
    src = """
subroutine pow_fpowi(x, r)
  implicit none
  real(8), intent(in)  :: x
  real(8), intent(out) :: r
  r = x**2
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="pow_fpowi", pipeline="hlfir-propagate-shapes").build()
    body = "\n".join(_tasklet_codes(sdfg))
    assert "**" in body, f"expected ** in tasklet, got: {body!r}"
    assert "?" not in body, f"bridge left a ? fallback, got: {body!r}"

    # Scalar real(8) intent(in) lands as a length-1 Array on the SDFG signature.
    x = np.array([2.5], dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)
    sdfg(x=x, r=r)
    np.testing.assert_allclose(r[0], 2.5**2, rtol=1e-12, atol=1e-12)


def test_powf_surfaces_as_python_pow(tmp_path: Path):
    """``r = x**y`` (both float, ``math.powf``) also lowers to ``**``
    and evaluates to ``x**y`` numerically."""
    src = """
subroutine pow_powf(x, y, r)
  implicit none
  real(8), intent(in)  :: x, y
  real(8), intent(out) :: r
  r = x**y
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="pow_powf", pipeline="hlfir-propagate-shapes").build()
    body = "\n".join(_tasklet_codes(sdfg))
    assert "**" in body
    assert "?" not in body

    x = np.array([3.0], dtype=np.float64)
    y = np.array([1.7], dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)
    sdfg(x=x, y=y, r=r)
    np.testing.assert_allclose(r[0], 3.0**1.7, rtol=1e-12, atol=1e-12)


def test_no_reassoc_wrapper_does_not_strand_inner_pow(tmp_path: Path):
    """Flang wraps parenthesised sums (``(x**2 + y**2)``) in
    ``hlfir.no_reassoc`` to block FP reassociation.  The bridge must
    recurse through the wrapper or the inner ``**`` surfaces as ``?``;
    the numerical check makes sure the surviving ``**`` actually
    delivers the right sum-of-squares."""
    src = """
subroutine pow_sum(x, y, r)
  implicit none
  real(8), intent(in)  :: x, y
  real(8), intent(out) :: r
  r = 0.5d0 * (x**2 + y**2)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="pow_sum", pipeline="hlfir-propagate-shapes").build()
    body = "\n".join(_tasklet_codes(sdfg))
    assert body.count("**") == 2, f"expected two **, got: {body!r}"
    assert "?" not in body

    x = np.array([3.0], dtype=np.float64)
    y = np.array([4.0], dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)
    sdfg(x=x, y=y, r=r)
    np.testing.assert_allclose(r[0], 0.5 * (3.0**2 + 4.0**2), rtol=1e-12, atol=1e-12)
