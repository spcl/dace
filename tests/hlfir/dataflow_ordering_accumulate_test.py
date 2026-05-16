"""Data-dependency-ordering tests for self-accumulate / prefix-scan
patterns, modelled on the CLOUDSC Section-8 flux loop.

The flux family in ``cloudsc_full`` diverges because the per-level
increment ``(ZQXN2D - ZQX0 + ...)`` is wrong at ~20 cells even though
every operand read in isolation is bit-identical to gfortran; the
loop-carried prefix sum ``PFSQLF(JK+1)=PFSQLF(JK)+incr`` then cascades
it.  Bisection ruled out aliasing, connector collision, FP flags,
indexing, buffer reuse, and the carry+accumulate two-write split
(fusing them changed nothing).  What is left is a producer->consumer
state-ordering hazard: the consumer reads an array that is saved in an
EARLY loop (``ZQX0``) together with one produced in the LATE main loop
(``ZQXN2D``), across a large intervening ``LoopRegion``.

These tests pin that family of shapes (each compared element-wise to
an f2py/gfortran reference of the same source).  They escalate from
the bare pattern to long multi-operand tasklets and large
producer/consumer state distance, so a future regression in the
bridge's data-dependency ordering fails loudly here rather than only
in the full kernel.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py(src_text: str, out_dir: Path, mod: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mod}.f90").write_text(src_text)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c", f"{mod}.f90", "-m", mod, "--quiet",
            "--f90flags=-O0 -fno-fast-math -ffp-contract=off"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod)
    return sys.modules[mod]


def _sdfg_kw(sdfg, ints: dict) -> dict:
    from dace.data import Scalar
    al = sdfg.arglist()
    out = {}
    for k, v in ints.items():
        d = al.get(k)
        out[k] = v if (d is None or isinstance(d, Scalar)) else np.array([v], np.int32)
    return out


# ---------------------------------------------------------------------------
# 1. Loop-carried prefix scan: f(jk+1)=f(jk)+(qn(jk)-q0(jk))*g, where
#    q0 is saved in an EARLY loop and qn produced in a LATE loop, with a
#    big intervening loop -- the exact CLOUDSC flux producer/consumer
#    state-distance shape, run per block (inlined-callee slice).
# ---------------------------------------------------------------------------

_PREFIX_SCAN = """
subroutine kern(klon, klev, paph, pclv, ptend, pwork_in, q0, qn, flux)
  implicit none
  integer, intent(in) :: klon, klev
  real(8), intent(in)  :: paph(klon, klev+1)
  real(8), intent(in)  :: pclv(klon, klev), ptend(klon, klev), pwork_in(klon, klev)
  real(8), intent(out) :: q0(klon, klev), qn(klon, klev), flux(klon, klev+1)
  real(8) :: zwork(klon, klev)
  integer :: jk, jl
  real(8) :: zg
  ! EARLY save loop: q0 = pclv + ptend  (the ZQX0 analogue)
  do jk = 1, klev
    do jl = 1, klon
      q0(jl, jk) = pclv(jl, jk) + 0.5d0 * ptend(jl, jk)
    end do
  end do
  ! BIG intervening main loop (no flux scalars; produces zwork & qn)
  do jk = 1, klev
    do jl = 1, klon
      zwork(jl, jk) = pwork_in(jl, jk) * pclv(jl, jk) + ptend(jl, jk)
      qn(jl, jk) = q0(jl, jk) + zwork(jl, jk) * 0.25d0
    end do
  end do
  ! POST-loop prefix-scan flux (ZQXN2D-ZQX0 analogue + carry)
  do jl = 1, klon
    flux(jl, 1) = 0.0d0
  end do
  do jk = 1, klev
    do jl = 1, klon
      zg = -(paph(jl, jk+1) - paph(jl, jk))
      flux(jl, jk+1) = flux(jl, jk)
      flux(jl, jk+1) = flux(jl, jk+1) + (qn(jl, jk) - q0(jl, jk)) * zg
    end do
  end do
end subroutine kern

subroutine outer(klon, klev, nb, paph, pclv, ptend, pwork_in, q0, qn, flux)
  implicit none
  integer, intent(in) :: klon, klev, nb
  real(8), intent(in)  :: paph(klon, klev+1, nb)
  real(8), intent(in)  :: pclv(klon, klev, nb), ptend(klon, klev, nb), pwork_in(klon, klev, nb)
  real(8), intent(out) :: q0(klon, klev, nb), qn(klon, klev, nb), flux(klon, klev+1, nb)
  integer :: ib
  do ib = 1, nb
    call kern(klon, klev, paph(:,:,ib), pclv(:,:,ib), ptend(:,:,ib), &
              pwork_in(:,:,ib), q0(:,:,ib), qn(:,:,ib), flux(:,:,ib))
  end do
end subroutine outer
"""


def test_prefix_scan_early_save_vs_late_produced(tmp_path: Path):
    """``flux(jk+1)=flux(jk)+(qn-q0)*g`` where ``q0`` is saved early and
    ``qn`` produced in a later loop, per block.  Must equal gfortran."""
    ref = _f2py(_PREFIX_SCAN, tmp_path / "ref", "scan_ref")
    sdfg = build_sdfg(_PREFIX_SCAN, _mkd(tmp_path / "sdfg"), name="outer", entry="_QPouter").build()

    klon, klev, nb = 1, 40, 4
    rng = np.random.default_rng(0)

    def F(*s):
        return np.asfortranarray(rng.standard_normal(s))

    paph, pclv = F(klon, klev + 1, nb), F(klon, klev, nb)
    ptend, pwin = F(klon, klev, nb), F(klon, klev, nb)
    q0r, qnr, fr = ref.outer(paph, pclv, ptend, pwin)

    q0s = np.zeros((klon, klev, nb), order="F")
    qns = np.zeros((klon, klev, nb), order="F")
    fs = np.zeros((klon, klev + 1, nb), order="F")
    sdfg(paph=paph.copy(order="F"),
         pclv=pclv.copy(order="F"),
         ptend=ptend.copy(order="F"),
         pwork_in=pwin.copy(order="F"),
         q0=q0s,
         qn=qns,
         flux=fs,
         **_sdfg_kw(sdfg, dict(klon=klon, klev=klev, nb=nb)))

    np.testing.assert_allclose(fs, fr, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Very long multi-operand self-accumulate tasklet (the 4+-term
#    CLOUDSC increment shape, extended): many early/late producers in a
#    single accumulate expression, prefix-scanned.
# ---------------------------------------------------------------------------

_LONG_TASKLET = """
subroutine kern(n, klev, a, b, c, d, e, f, g, h, s0, sn, acc)
  implicit none
  integer, intent(in) :: n, klev
  real(8), intent(in)  :: a(n,klev), b(n,klev), c(n,klev), d(n,klev)
  real(8), intent(in)  :: e(n,klev), f(n,klev), g(n,klev), h(n,klev)
  real(8), intent(out) :: s0(n,klev), sn(n,klev), acc(n,klev+1)
  real(8) :: w(n,klev)
  integer :: jk, jl
  do jk = 1, klev
    do jl = 1, n
      s0(jl,jk) = a(jl,jk) + 0.5d0*b(jl,jk)
    end do
  end do
  do jk = 1, klev
    do jl = 1, n
      w(jl,jk) = c(jl,jk)*d(jl,jk) - e(jl,jk)
      sn(jl,jk) = s0(jl,jk) + w(jl,jk) + f(jl,jk)*g(jl,jk)
    end do
  end do
  do jl = 1, n
    acc(jl,1) = 0.0d0
  end do
  do jk = 1, klev
    do jl = 1, n
      acc(jl,jk+1) = acc(jl,jk) + &
        ((sn(jl,jk) - s0(jl,jk) + a(jl,jk)*b(jl,jk) - c(jl,jk)*d(jl,jk) &
          + e(jl,jk)*f(jl,jk) - g(jl,jk)*h(jl,jk)) * (a(jl,jk) - b(jl,jk)))
    end do
  end do
end subroutine kern
"""


def test_very_long_accumulate_tasklet(tmp_path: Path):
    """A long (8-input + self) accumulate expression mixing an
    early-saved (``s0``) and late-produced (``sn``) operand, prefix-
    scanned.  Pins the CLOUDSC ``ZQXN2D-ZQX0+...`` increment shape."""
    ref = _f2py(_LONG_TASKLET, tmp_path / "ref", "long_ref")
    sdfg = build_sdfg(_LONG_TASKLET, _mkd(tmp_path / "sdfg"), name="kern", entry="_QPkern").build()

    n, klev = 1, 50
    rng = np.random.default_rng(1)

    def F():
        return np.asfortranarray(rng.standard_normal((n, klev)))

    arrs = [F() for _ in range(8)]
    s0r, snr, accr = ref.kern(*arrs)

    s0s = np.zeros((n, klev), order="F")
    sns = np.zeros((n, klev), order="F")
    accs = np.zeros((n, klev + 1), order="F")
    sdfg(a=arrs[0].copy(order="F"),
         b=arrs[1].copy(order="F"),
         c=arrs[2].copy(order="F"),
         d=arrs[3].copy(order="F"),
         e=arrs[4].copy(order="F"),
         f=arrs[5].copy(order="F"),
         g=arrs[6].copy(order="F"),
         h=arrs[7].copy(order="F"),
         s0=s0s,
         sn=sns,
         acc=accs,
         **_sdfg_kw(sdfg, dict(n=n, klev=klev)))

    np.testing.assert_allclose(accs, accr, rtol=1e-12, atol=1e-12)


def _mkd(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
