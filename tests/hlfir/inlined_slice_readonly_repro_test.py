"""Minimal reproducer for the cloudsc_full / bottom_lower structural
failure.

cloudsc shape:  ``CLOUDSCOUTER`` owns ``PLUDE(KLON,KLEV,NBLOCKS)`` and,
in a block loop ``DO IBL=1,NBLOCKS``, calls ``CLOUDSC(..., PLUDE(:,:,IBL),
...)``.  ``CLOUDSC`` declares ``PLUDE(KLON,KLEV)`` ``INTENT(INOUT)`` but
in the bottom-lower carve-out only *reads* it (``... - ZALFAW*PLUDE(JL,JK)
...``).  After ``hlfir-inline-all`` the read of the 2-D slice dummy
must alias the caller's 3-D ``PLUDE`` storage at block ``IBL``;
otherwise the SDFG reads uninitialised memory (~1e228) and every
downstream flux output is garbage.

This kernel isolates exactly that: a 3-D ``intent(inout)`` array
passed as a 2-D section ``a(:,:,ib)`` to an inlined subroutine that
only reads it, inside an outer block loop.  SDFG vs gfortran/f2py on
the same source -- a broken slice-read alias shows up as a gross
mismatch (uninitialised values), not a tolerance issue.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_inner
  implicit none
contains
  subroutine inner(n, lev, plude, out)
    integer, intent(in) :: n, lev
    real(8), intent(inout) :: plude(n, lev)   ! read-only in this body
    real(8), intent(inout) :: out(n, lev)
    integer :: i, k
    do k = 1, lev
      do i = 1, n
        out(i, k) = out(i, k) + plude(i, k) * 2.0d0
      end do
    end do
  end subroutine inner
end module mo_inner

subroutine outer(n, lev, nb, plude, out)
  use mo_inner, only: inner
  implicit none
  integer, intent(in) :: n, lev, nb
  real(8), intent(inout) :: plude(n, lev, nb)
  real(8), intent(inout) :: out(n, lev, nb)
  integer :: ib
  do ib = 1, nb
    call inner(n, lev, plude(:, :, ib), out(:, :, ib))
  end do
end subroutine outer
"""


def test_inlined_2d_slice_readonly_of_3d_array(tmp_path: Path):
    """``out(:,:,ib) += plude(:,:,ib)*2`` via an inlined callee.  The
    inlined read of the 2-D slice dummy must alias the caller's 3-D
    ``plude`` at block ``ib``; a broken alias reads uninitialised
    memory and the result diverges grossly from the f2py reference."""
    n, lev, nb = 3, 4, 2
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sdfg = build_sdfg(_SRC, sdfg_dir, name="outer", entry="_QPouter").build()
    sdfg.validate()
    ref = f2py_compile(_SRC, ref_dir, "slice_ref", only=("outer", ))

    rng = np.random.default_rng(0)
    plude = np.asfortranarray(rng.random((n, lev, nb)))
    out0 = np.asfortranarray(rng.random((n, lev, nb)))

    p_ref, o_ref = plude.copy(order="F"), out0.copy(order="F")
    ref.outer(p_ref, o_ref)  # n/lev/nb auto-derived from shapes

    p_sdfg, o_sdfg = plude.copy(order="F"), out0.copy(order="F")
    sdfg(n=np.int32(n), lev=np.int32(lev), nb=np.int32(nb), plude=p_sdfg, out=o_sdfg)

    np.testing.assert_allclose(o_sdfg, o_ref, rtol=1e-12, atol=1e-12)
    # Closed-form: out is incremented once per block by plude*2.
    np.testing.assert_allclose(o_sdfg, out0 + plude * 2.0, rtol=1e-12, atol=1e-12)


_SRC_FLUX = """
module mo_flux
  implicit none
contains
  subroutine flux(n, lev, plude, pfsqlf)
    integer, intent(in) :: n, lev
    real(8), intent(inout) :: plude(n, lev)        ! read-only here
    real(8), intent(inout) :: pfsqlf(n, lev + 1)   ! KLEV+1, shifted idx
    integer :: i, k
    do i = 1, n
      pfsqlf(i, 1) = 0.0d0
    end do
    do k = 1, lev
      do i = 1, n
        pfsqlf(i, k + 1) = pfsqlf(i, k) + plude(i, k) * 0.5d0
      end do
    end do
  end subroutine flux
end module mo_flux

subroutine outer_flux(n, lev, nb, plude, pfsqlf)
  use mo_flux, only: flux
  implicit none
  integer, intent(in) :: n, lev, nb
  real(8), intent(inout) :: plude(n, lev, nb)
  real(8), intent(inout) :: pfsqlf(n, lev + 1, nb)
  integer :: ib
  do ib = 1, nb
    call flux(n, lev, plude(:, :, ib), pfsqlf(:, :, ib))
  end do
end subroutine outer_flux
"""


def test_inlined_flux_accumulation_shifted_index(tmp_path: Path):
    """``pfsqlf(:,1)=0 ; pfsqlf(:,k+1)=pfsqlf(:,k)+plude(:,k)*0.5``
    over a ``(KLEV+1)`` slice ``pfsqlf(:,:,ib)``, inlined, block loop --
    the cloudsc Section-6 flux-accumulation shape that produces the
    ~1e228 garbage in bottom_lower."""
    n, lev, nb = 3, 4, 2
    sdfg_dir = tmp_path / "sdfg2"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = tmp_path / "ref2"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sdfg = build_sdfg(_SRC_FLUX, sdfg_dir, name="outer_flux", entry="_QPouter_flux").build()
    sdfg.validate()
    ref = f2py_compile(_SRC_FLUX, ref_dir, "flux_ref", only=("outer_flux", ))

    rng = np.random.default_rng(1)
    plude = np.asfortranarray(rng.random((n, lev, nb)))
    pf0 = np.asfortranarray(rng.random((n, lev + 1, nb)))

    p_ref, f_ref = plude.copy(order="F"), pf0.copy(order="F")
    ref.outer_flux(p_ref, f_ref)

    p_sdfg, f_sdfg = plude.copy(order="F"), pf0.copy(order="F")
    sdfg(n=np.int32(n), lev=np.int32(lev), nb=np.int32(nb), plude=p_sdfg, pfsqlf=f_sdfg)

    np.testing.assert_allclose(f_sdfg, f_ref, rtol=1e-12, atol=1e-12)
