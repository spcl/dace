"""Reproducer for the CLOUDSC Section-8 flux-accumulation divergence.

The full CLOUDSC flux loop (cloudsc.F90 lines ~3748-3793) is a
loop-carried prefix scan over a ``(KLON, KLEV+1)`` ``INTENT(OUT)``
array, written through an inlined-callee section slice
``PFSQLF(:,:,IBL)``.  Each JK iteration:

  PFSQLF(JL,JK+1) = PFSQLF(JL,JK)            ! carry the running sum
  PFSQRF(JL,JK+1) = PFSQLF(JL,JK)            ! cross-array carry (RF<-LF)
  PFSQLF(JL,JK+1) = PFSQLF(JL,JK+1) + termL  ! accumulate this level
  PFSQRF(JL,JK+1) = PFSQRF(JL,JK+1) + termR

cloudsc_full diverges O(0.4) on exactly PFSQLF/PFSQIF/PFSQRF/PFSQSF
(~130 cells) while PCOVPTOT and every non-flux output match at 1e-12.
This carve-out isolates the four ingredients: (1) loop-carried scan,
(2) two writers to the same ``(JK+1)`` element per body, (3) the
cross-array ``RF<-LF(JK)`` carry, (4) the inlined-callee
``INTENT(OUT)`` section slice through a block wrapper.

f2py-compiled gfortran is the non-transformed reference (flat-arg
subroutine, crackfortran-wrappable); SDFG is built through the full
default pipeline (inline-all must fire for the slice path).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
subroutine cloudsc(klon, klev, paph, za, zb, pfsqlf, pfsqrf)
  implicit none
  integer, intent(in) :: klon, klev
  real(8), intent(in)  :: paph(klon, klev+1)
  real(8), intent(in)  :: za(klon, klev), zb(klon, klev)
  real(8), intent(out) :: pfsqlf(klon, klev+1)
  real(8), intent(out) :: pfsqrf(klon, klev+1)
  integer :: jk, jl
  real(8) :: zgdph_r
  do jl = 1, klon
    pfsqlf(jl, 1) = 0.0d0
    pfsqrf(jl, 1) = 0.0d0
  end do
  do jk = 1, klev
    do jl = 1, klon
      zgdph_r = -(paph(jl, jk+1) - paph(jl, jk))
      pfsqlf(jl, jk+1) = pfsqlf(jl, jk)
      pfsqrf(jl, jk+1) = pfsqlf(jl, jk)
      pfsqlf(jl, jk+1) = pfsqlf(jl, jk+1) + za(jl, jk) * zgdph_r
      pfsqrf(jl, jk+1) = pfsqrf(jl, jk+1) + zb(jl, jk) * zgdph_r
    end do
  end do
end subroutine cloudsc

subroutine cloudscouter(klon, klev, nblocks, paph, za, zb, pfsqlf, pfsqrf)
  implicit none
  integer, intent(in) :: klon, klev, nblocks
  real(8), intent(in)  :: paph(klon, klev+1, nblocks)
  real(8), intent(in)  :: za(klon, klev, nblocks), zb(klon, klev, nblocks)
  real(8), intent(out) :: pfsqlf(klon, klev+1, nblocks)
  real(8), intent(out) :: pfsqrf(klon, klev+1, nblocks)
  integer :: ibl
  do ibl = 1, nblocks
    call cloudsc(klon, klev, paph(:, :, ibl), za(:, :, ibl), &
                 zb(:, :, ibl), pfsqlf(:, :, ibl), pfsqrf(:, :, ibl))
  end do
end subroutine cloudscouter
"""


def _f2py_build(src_text: str, out_dir: Path, mod_name: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mod_name}.f90").write_text(src_text)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c", f"{mod_name}.f90", "-m", mod_name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def test_cloudsc_flux_recurrence(tmp_path: Path):
    """The loop-carried flux scan written through an inlined-callee
    ``INTENT(OUT)`` slice must equal the gfortran reference."""
    ref = _f2py_build(_SRC, tmp_path / "ref", "flux_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="cloudsc", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(7)
    klon, klev, nblocks = 4, 6, 3

    def _f(shape):
        return np.asfortranarray(rng.standard_normal(shape, dtype=np.float64))

    paph = _f((klon, klev + 1, nblocks))
    za = _f((klon, klev, nblocks))
    zb = _f((klon, klev, nblocks))

    lf_ref, rf_ref = ref.cloudscouter(paph, za, zb)

    lf_sdfg = np.zeros((klon, klev + 1, nblocks), dtype=np.float64, order="F")
    rf_sdfg = np.zeros_like(lf_sdfg)
    sdfg(klon=klon,
         klev=klev,
         nblocks=nblocks,
         paph=paph.copy(order="F"),
         za=za.copy(order="F"),
         zb=zb.copy(order="F"),
         pfsqlf=lf_sdfg,
         pfsqrf=rf_sdfg)

    np.testing.assert_allclose(lf_sdfg, lf_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rf_sdfg, rf_ref, rtol=1e-12, atol=1e-12)
