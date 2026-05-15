"""Regression: literal-lower-bound heuristic must not poison an
explicit-shape (plain ``fir.ShapeOp``) array's offset.

cloudsc shape (lines 1574-1591 of cloudsc_bottom_lower.F90): a local
automatic array ``ZQX(KLON,KLEV,NCLV)`` (NCLV a PARAMETER) is written
with a *mix* of 3rd-dimension subscripts -- a PARAMETER constant index
``ZQX(JL,JK,NCLDQV)`` and a loop index ``ZQX(JL,JK,JM)`` in a
``DO JM=1,NCLV-1`` loop.

``inferLowerBoundsFromLiteralAccesses`` (meant only to recover ICON's
*explicit negative* lower bounds on deferred-shape ALLOCATABLEs) used
to run on this plain-``fir.ShapeOp`` array too.  ``traceConstIntThrough
Load`` resolved one of the non-literal subscripts to a sentinel, so
the per-dim ``min`` produced ``offset_<arr>_d2 = -999`` -- turning the
write subset into a wild out-of-bounds store (segfault / 1e228
garbage).  The fix gates the heuristic off plain-``fir.ShapeOp``
declares (lower bound is provably 1 there).

This test pins it: every ``offset_a_d*`` must be ``1`` (never a
sentinel / negative) and the kernel must match a gfortran/f2py
reference of the same source.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Mirrors the cloudsc trigger: explicit-shape local 3-D array, PARAMETER
# last extent, written with a constant PARAMETER 3rd index in one loop
# nest and a loop 3rd index in another.
_SRC = """
subroutine mixed_idx(n, m, q, cl, a)
  implicit none
  integer, parameter :: nc = 5
  integer, parameter :: klast = 5     ! NCLDQV-style constant index
  integer, intent(in) :: n, m
  real(8), intent(in)    :: q(n, m)
  real(8), intent(in)    :: cl(n, m, nc)
  real(8), intent(inout) :: a(n, m, nc)
  integer :: i, k, jm
  ! constant PARAMETER 3rd-dim index (no JM loop)
  do k = 1, m
    do i = 1, n
      a(i, k, klast) = q(i, k) * 2.0d0
    end do
  end do
  ! loop 3rd-dim index
  do jm = 1, nc - 1
    do k = 1, m
      do i = 1, n
        a(i, k, jm) = cl(i, k, jm) + q(i, k)
      end do
    end do
  end do
end subroutine mixed_idx
"""


def test_explicit_shape_offsets_not_poisoned(tmp_path: Path):
    """Build the SDFG: every ``offset_a_d*`` (and q/cl) must be exactly
    1 -- never a ``-999``-style sentinel from the literal heuristic --
    and the result must match the f2py reference."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sdfg = build_sdfg(_SRC, sdfg_dir, name="mixed_idx", entry="_QPmixed_idx").build()
    sdfg.validate()

    consts = dict(sdfg.constants)
    offsets = {k: int(v) for k, v in consts.items() if k.startswith("offset_")}
    assert offsets, "expected per-dim offset constants"
    bad = {k: v for k, v in offsets.items() if v != 1}
    assert not bad, (f"explicit-shape arrays must have offset 1 in every dim; "
                     f"poisoned offsets: {bad}")

    ref = f2py_compile(_SRC, ref_dir, "mixed_idx_ref")

    n, m, nc = 3, 4, 5
    rng = np.random.default_rng(0)
    q = np.asfortranarray(rng.random((n, m)))
    cl = np.asfortranarray(rng.random((n, m, nc)))
    a0 = np.asfortranarray(rng.random((n, m, nc)))

    q_r, cl_r, a_r = q.copy(order="F"), cl.copy(order="F"), a0.copy(order="F")
    ref.mixed_idx(q_r, cl_r, a_r)  # n/m auto-derived

    q_s, cl_s, a_s = q.copy(order="F"), cl.copy(order="F"), a0.copy(order="F")
    sdfg(n=np.int32(n), m=np.int32(m), q=q_s, cl=cl_s, a=a_s)

    np.testing.assert_array_equal(a_s, a_r)
