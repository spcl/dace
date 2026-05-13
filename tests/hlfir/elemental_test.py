"""ELEMENTAL procedures -> loop-over-array + scalar-body tasklet.

Flang lowers a call to an elemental subroutine on an array argument as a
``fir.do_loop`` that per-element invokes the scalar procedure.  After
``hlfir-inline-all`` splices the callee's body in, the remaining
``hlfir.declare`` ops whose memref is an ``hlfir.designate`` of the outer
array are element-scoped aliases  --  they carry the callee's Fortran name
into the inlined body but add no semantics.  ``hlfir-fold-element-aliases``
erases them so the SDFG builder sees the loop body as plain indexed
access into the outer array, i.e. a loop + scalar tasklet shape
identical to any hand-written per-element loop.

Tests cover the subroutine form (explicit inout update) and the function
form (via an ``hlfir.elemental`` block) on the same source so both
Flang-emitted shapes are exercised.  References are NumPy  --  f2py's
module-contained-elemental parsing is shaky, so we check against a
hand-rolled per-element implementation instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_elemental_subroutine_with_inout(tmp_path: Path):
    """Elemental subroutine with inout scalars applied to three arrays  --
    Flang emits ``fir.do_loop`` + ``fir.call`` to the scalar body;
    inline-all + fold-element-aliases collapse the body into indexed
    updates on the outer arrays."""
    src = """
subroutine apply_delta(od, scat_od, g)
  implicit none
  real(8), intent(inout) :: od(14), scat_od(14), g(14)
  call delta(od, scat_od, g)
contains
  elemental subroutine delta(a, b, c)
    real(8), intent(inout) :: a, b, c
    real(8) :: f
    f = c * c
    a = a - b * f
    b = b * (1.0d0 - f)
    c = c / (1.0d0 + c)
  end subroutine delta
end subroutine apply_delta
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_delta").build()

    rng = np.random.default_rng(0)
    od = np.asfortranarray(rng.random(14, dtype=np.float64))
    scat_od = np.asfortranarray(rng.random(14, dtype=np.float64))
    g = np.asfortranarray(rng.random(14, dtype=np.float64))

    # NumPy reference: scalar body applied per element.
    od_ref, sod_ref, g_ref = od.copy(), scat_od.copy(), g.copy()
    f_ref = g_ref * g_ref
    od_ref = od_ref - sod_ref * f_ref
    sod_ref = sod_ref * (1.0 - f_ref)
    g_ref = g_ref / (1.0 + g_ref)

    sdfg(od=od, scat_od=scat_od, g=g)

    np.testing.assert_allclose(od, od_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(scat_od, sod_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(g, g_ref, atol=1e-12, rtol=0)


def test_elemental_function_via_hlfir_elemental(tmp_path: Path):
    """Pointwise expression on arrays  --  Flang wraps the RHS in
    ``hlfir.elemental`` + ``hlfir.yield_element``, the exact shape
    ``buildElementalAssign`` consumes.  This test guards that
    ``FoldElementAliases`` (which targets inlined-scalar-body aliases)
    leaves the standard intrinsic-elemental path untouched."""
    src = """
subroutine apply_square_shift(x, y, n)
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: x(n)
  real(8), intent(out) :: y(n)
  y = x * x - 1.0d0
end subroutine apply_square_shift
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_square_shift").build()

    rng = np.random.default_rng(1)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    y = np.zeros(n, dtype=np.float64, order="F")

    sdfg(x=x, y=y, n=n)

    np.testing.assert_allclose(y, x * x - 1.0, atol=1e-12, rtol=0)


def test_fold_element_aliases_drops_inlined_declares(tmp_path: Path):
    """Structural: after the pipeline has run, the inlined elemental
    body should NOT carry its own Fortran-named scalar as a separate
    SDFG array.  (Before the FoldElementAliases pass, the callee's
    per-element dummies showed up as stray ``a`` / ``b`` / ``c`` /
    ``f`` scalars on the SDFG argslist.)"""
    src = """
subroutine driver(x)
  implicit none
  real(8), intent(inout) :: x(8)
  call doubler(x)
contains
  elemental subroutine doubler(v)
    real(8), intent(inout) :: v
    v = v * 2.0d0
  end subroutine doubler
end subroutine driver
"""
    b = build_sdfg(src, tmp_path, name="driver")
    sdfg = b.build()

    # The outer array ``x`` is the only dummy of the driver.  The
    # inlined callee's scalar ``v`` must NOT show up as its own array.
    assert "v" not in b.arrays, \
        f"elemental inner dummy 'v' leaked into arrays: {list(b.arrays.keys())}"
    assert "x" in sdfg.arrays, list(sdfg.arrays.keys())


def test_elemental_body_with_intrinsic(tmp_path: Path):
    """Elemental subroutine whose scalar body invokes an intrinsic
    (``exp``)  --  after ``hlfir-inline-all`` + ``FoldElementAliases``
    the intrinsic must survive and land in the tasklet code as a
    Python call (``exp``)."""
    src = """
subroutine apply_soft(x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  call soft(x)
contains
  elemental subroutine soft(v)
    real(8), intent(inout) :: v
    v = exp(v) - 1.0d0
  end subroutine soft
end subroutine apply_soft
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_soft").build()

    rng = np.random.default_rng(2)
    n = 16
    x = np.asfortranarray(rng.random(n, dtype=np.float64))
    x_ref = np.exp(x) - 1.0

    sdfg(x=x, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)


def test_elemental_subroutine_relu(tmp_path: Path):
    """Elemental subroutine implementing ReLU via a Fortran if/else on
    each element  --  exercises conditional control flow *inside* the
    inlined scalar body, on top of the loop-over-array shape."""
    src = """
subroutine apply_relu(x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  call relu(x)
contains
  elemental subroutine relu(v)
    real(8), intent(inout) :: v
    if (v <= 0.0d0) v = 0.0d0
  end subroutine relu
end subroutine apply_relu
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_relu").build()

    rng = np.random.default_rng(3)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    x_ref = np.maximum(x, 0.0)

    sdfg(x=x, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)


def test_elemental_subroutine_softmax_step(tmp_path: Path):
    """Elemental subroutine implementing one step of a softmax-style
    normalisation  --  exercises a two-statement body where the first
    statement's write feeds the second (``t = exp(x); x = t / s``).
    Drives the RAW-hazard serialisation in ``emit_loop``'s child-
    assigns path: without a fresh state per statement, the second
    tasklet's read of ``t`` would race with the first tasklet's
    write."""
    src = """
subroutine apply_softmax_step(x, s, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: s
  real(8), intent(inout) :: x(n)
  call smstep(x, s)
contains
  elemental subroutine smstep(v, norm)
    real(8), intent(inout) :: v
    real(8), intent(in)    :: norm
    real(8) :: t
    t = exp(v)
    v = t / norm
  end subroutine smstep
end subroutine apply_softmax_step
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_softmax_step").build()

    rng = np.random.default_rng(4)
    n = 16
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    s_val = float(np.sum(np.exp(x)))
    x_ref = np.exp(x) / s_val

    sdfg(x=x, s=s_val, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)
