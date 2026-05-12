"""Simple FaCe-native test for the Fortran ``MERGE`` intrinsic.

``merge(t, f, mask)`` returns ``t`` where ``mask`` is true and ``f``
otherwise.  Flang lowers it to a bare ``arith.select`` for scalars and
to an ``hlfir.elemental`` wrapping ``arith.select`` for arrays.  The
bridge's generic ``arith.select`` ternary fallback in ``buildExpr``
emits a Python ``(t if cond else f)`` form that the C++ codegen lowers
to a conditional expression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_merge_scalar(tmp_path: Path):
    """``merge`` on scalar real(8) inputs with a logical mask."""
    src = """
subroutine probe(a, b, mask, out)
  real(8), intent(in)   :: a, b
  logical, intent(in)   :: mask
  real(8), intent(out)  :: out
  out = merge(a, b, mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for mask_val in (True, False):
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=1.5, b=2.5, mask=mask_val, out=out)
        expected = 1.5 if mask_val else 2.5
        assert float(out[0]) == expected, f"mask={mask_val}: got {out[0]}, want {expected}"


def test_merge_comparison_as_mask(tmp_path: Path):
    """``merge(a, b, a > b)`` — an inline comparison as the mask."""
    src = """
subroutine probe(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out
  out = merge(a, b, a > b)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for a_in, b_in, expected in [(3.0, 2.0, 3.0), (1.0, 5.0, 5.0), (4.0, 4.0, 4.0)]:
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=a_in, b=b_in, out=out)
        assert float(out[0]) == expected


def test_merge_array_via_library_node(tmp_path: Path):
    """Array ``merge(t, f, mask)`` — Flang lowers to ``hlfir.elemental {
    hlfir.designate; arith.select; yield_element }``; the bridge
    detects the simple shape (three loaded designates of declared
    arrays) and routes through ``MergeLibraryNode`` directly.  Verify
    end-to-end against numpy ``np.where``."""
    src = """
subroutine main(t, f, mask, out, n)
  integer, intent(in)  :: n
  real(8), intent(in)  :: t(n), f(n)
  logical, intent(in)  :: mask(n)
  real(8), intent(out) :: out(n)
  out = MERGE(t, f, mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(0)
    n = 12
    t = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    f = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    mask = np.ascontiguousarray(rng.random(n) > 0.5)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out, n=n)
    np.testing.assert_array_equal(out, np.where(mask, t, f))


def test_merge_array_verifies_libnode_present(tmp_path: Path):
    """Same as above, but also asserts a ``MergeLibraryNode`` is present
    in the built SDFG — pins the bridge → library-node routing
    structurally so a future regression (e.g. a refactor that quietly
    falls back to the per-element-tasklet path) trips this test."""
    from dace.libraries.standard.nodes import MergeLibraryNode
    src = """
subroutine main(t, f, mask, out, n)
  integer, intent(in)  :: n
  real(8), intent(in)  :: t(n), f(n)
  logical, intent(in)  :: mask(n)
  real(8), intent(out) :: out(n)
  out = MERGE(t, f, mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    libnodes = [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, MergeLibraryNode)]
    assert len(libnodes) == 1, f"expected exactly one MergeLibraryNode, got {len(libnodes)}"
