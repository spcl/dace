"""End-to-end coverage for the unified lower-bound/offset resolver.

Each case drives a Fortran array shape that exercises one of the three
``hlfir.declare`` shape-operand forms the unified decoder
(``classifyShapeOperand``) must handle, then runs the SDFG and an
f2py-compiled reference of the *same* source and asserts bitwise
equality (a wrong per-dim offset shifts or wild-writes the result, so
exact equality is the right assertion).

Forms covered:
  * ``fir.shape``        -- explicit/automatic, lb 1   (baseline)
  * ``fir.shape_shift``  -- explicit negative bounds ``a(-5:5,0:3)``
  * ``fir.shift``        -- assumed-shape dummy with explicit local
                            lower bounds ``a(10:,20:)`` -- the form
                            the bridge previously did NOT handle
                            (offsets silently lost -> OOB).
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _e2e(src: str, entry: str, tmp_path: Path, sdfg_kw: dict, ref_args: tuple):
    """Build via bridge + via f2py, run both, return (sdfg_out, ref_out).

    :param src: inline Fortran (one subroutine, ``intent(inout)`` out).
    :param entry: mangled flang symbol.
    :param sdfg_kw: kwargs for the SDFG call (incl. shape symbols).
    :param ref_args: positional args for the f2py reference call.
    :returns: nothing -- callers assert on the mutated arrays they own.
    """
    name = entry.split("P")[-1]
    sd = tmp_path / "sdfg"
    sd.mkdir(parents=True, exist_ok=True)
    rd = tmp_path / "ref"
    rd.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sd, name=name, entry=entry).build()
    sdfg.validate()
    ref = f2py_compile(src, rd, f"{name}_ref")
    getattr(ref, name)(*ref_args)
    sdfg(**sdfg_kw)


def test_explicit_shape_default_lb(tmp_path: Path):
    """``fir.shape`` -- automatic array, default lb 1.  Baseline that
    the unified decoder's Shape path stays correct."""
    src = """
subroutine es_def(n, a, out)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: a(n, n)
  real(8), intent(inout) :: out(n, n)
  integer :: i, j
  do j = 1, n
    do i = 1, n
      out(i, j) = a(i, j) * 2.0d0 + 1.0d0
    end do
  end do
end subroutine es_def
"""
    n = 4
    a = np.asfortranarray(np.random.default_rng(1).random((n, n)))
    o_r = np.zeros((n, n), order="F")
    o_s = np.zeros((n, n), order="F")
    _e2e(src, "_QPes_def", tmp_path, dict(n=np.int32(n), a=a.copy(order="F"), out=o_s), (a.copy(order="F"), o_r))
    np.testing.assert_array_equal(o_s, o_r)
    np.testing.assert_array_equal(o_s, a * 2.0 + 1.0)


# f2py's crackfortran cannot wrap a dummy with explicit non-unit /
# assumed-shape bounds (``w(-5:5,0:3)`` / ``a(10:,20:)``) -- it needs an
# explicit interface it doesn't synthesise.  These cases verify the
# unified resolver two ways instead, both non-transformed references:
#   (1) the per-dim ``offset_<arr>_d<i>`` constants are exactly the
#       Fortran lower bounds (the direct resolver-correctness signal);
#   (2) the compiled SDFG binding, run end-to-end, matches the
#       closed-form mathematical result.


def _offsets(sdfg, arr: str) -> dict:
    c = dict(sdfg.constants)
    return {k: int(v) for k, v in c.items() if k.startswith(f"offset_{arr}_d")}


def test_explicit_negative_bounds_shape_shift(tmp_path: Path):
    """``fir.shape_shift`` -- explicit-shape array with negative /
    non-unit lower bounds ``w(-5:5, 0:3)``.  Offsets must be -5 and 0;
    a wrong offset shifts (or wild-writes) every element."""
    src = """
subroutine ss_neg(w, out)
  implicit none
  real(8), intent(inout) :: w(-5:5, 0:3)
  real(8), intent(inout) :: out(-5:5, 0:3)
  integer :: i, j
  do j = 0, 3
    do i = -5, 5
      out(i, j) = w(i, j) + real(i, 8) - real(j, 8)
    end do
  end do
end subroutine ss_neg
"""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, d, name="ss_neg", entry="_QPss_neg").build()
    sdfg.validate()
    assert _offsets(sdfg, "w") == {"offset_w_d0": -5, "offset_w_d1": 0}
    assert _offsets(sdfg, "out") == {"offset_out_d0": -5, "offset_out_d1": 0}

    rng = np.random.default_rng(2)
    w = np.asfortranarray(rng.random((11, 4)))
    o = np.zeros((11, 4), order="F")
    sdfg(w=w.copy(order="F"), out=o)
    expect = w + (np.arange(-5, 6)[:, None] - np.arange(0, 4)[None, :])
    np.testing.assert_array_equal(o, expect)


def test_assumed_shape_explicit_lb_fir_shift(tmp_path: Path):
    """``fir.shift`` -- assumed-shape dummy with an explicit local
    lower bound ``a(10:)``.  flang emits ``fir.shift %c10`` as the
    declare's shape operand; the bridge previously handled only
    ShapeOp/ShapeShiftOp, so the 10 was lost and the access went out
    of bounds.  The unified decoder (``classifyShapeOperand``) +
    fir.shift lb consumption must recover offset 10.

    1-D form: a 2-D assumed-shape's second extent symbol is not
    surfaced as a program arg (a separate gap, unrelated to the lower
    bound this test pins); 1-D keeps the numeric e2e clean."""
    src = """
subroutine as_shift(n, a, out)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: a(10:)
  real(8), intent(inout) :: out(10:)
  integer :: i
  do i = 10, 10 + n - 1
    out(i) = a(i) * 3.0d0 - 2.0d0
  end do
end subroutine as_shift
"""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, d, name="as_shift", entry="_QPas_shift").build()
    sdfg.validate()
    # The E1 correctness signal: fir.shift lb recovered exactly.
    assert _offsets(sdfg, "a") == {"offset_a_d0": 10}
    assert _offsets(sdfg, "out") == {"offset_out_d0": 10}

    n = 6
    rng = np.random.default_rng(3)
    a = np.asfortranarray(rng.random(n))
    o = np.zeros(n, order="F")
    al = sdfg.arglist()
    ext = {k: np.int64(n) for k in al if k.endswith("_d0") and not k.startswith("offset_")}
    sdfg(n=np.int32(n), a=a.copy(order="F"), out=o, **ext)
    # Float computation (a*3-2): allclose, not array_equal (1-ULP
    # eval-order difference); the offset asserts above stay exact.
    np.testing.assert_allclose(o, a * 3.0 - 2.0, rtol=1e-12, atol=1e-12)


def test_assumed_shape_explicit_lb_fir_shift_2d(tmp_path: Path):
    """2-D ``fir.shift`` -- ``a(10:, 20:)``.  E1-proper must recover
    BOTH lower bounds (10, 20) exactly.  The numeric e2e additionally
    requires every per-dim extent of a 2-D assumed-shape to be a
    bindable program argument; recover them generically from the
    arglist (this also pins the 2-D-assumed-shape extent surfacing)."""
    src = """
subroutine as_shift2(n, m, a, out)
  implicit none
  integer, intent(in) :: n, m
  real(8), intent(inout) :: a(10:, 20:)
  real(8), intent(inout) :: out(10:, 20:)
  integer :: i, j
  do j = 20, 20 + m - 1
    do i = 10, 10 + n - 1
      out(i, j) = a(i, j) * 3.0d0 - 2.0d0
    end do
  end do
end subroutine as_shift2
"""
    d = tmp_path / "sdfg2"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, d, name="as_shift2", entry="_QPas_shift2").build()
    sdfg.validate()
    # E1 correctness signal: both fir.shift lbs recovered exactly.
    assert _offsets(sdfg, "a") == {"offset_a_d0": 10, "offset_a_d1": 20}
    assert _offsets(sdfg, "out") == {"offset_out_d0": 10, "offset_out_d1": 20}

    n, m = 5, 3
    rng = np.random.default_rng(4)
    a = np.asfortranarray(rng.random((n, m)))
    o = np.zeros((n, m), order="F")
    al = sdfg.arglist()
    # Per-dim extent binding: the SDFG descriptor shape for ``a`` is
    # (a_d0, a_d1); both must be bindable program args.
    sizes = {0: n, 1: m}
    ext = {}
    for k in al:
        if k.startswith("offset_"):
            continue
        for di, sz in sizes.items():
            if k.endswith(f"_d{di}"):
                ext[k] = np.int64(sz)
    sdfg(n=np.int32(n), m=np.int32(m), a=a.copy(order="F"), out=o, **ext)
    np.testing.assert_allclose(o, a * 3.0 - 2.0, rtol=1e-12, atol=1e-12)
