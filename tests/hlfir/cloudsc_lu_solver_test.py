"""LU forward+back substitution loopnest from CLOUDSC.

Lifts the LU-solve block (cloudscexp2_simplified.F90 lines 3308-3336)
that solves the per-cell ``NCLV x NCLV`` cloud-microphysics linear
system.  The result ``ZQXN`` feeds ``ZQXN2D``, which feeds every
cumulative flux output (``PFSQLF``, ``PFSQIF``, ``PFSQRF``, ``PFSQSF``).
A wrong ZQXN here propagates as a per-step delta to all of those.

Structure:

* Non-pivoting recursive factorisation (forward sweep over JN, inner
  JM/IK loops; modifies ZQLHS in place).
* Forward-substitute (step 1): ``DO JN=2,NCLV / DO JM=1,JN-1`` —
  ``ZQXN(:, JN) -= ZQLHS(:, JN, JM) * ZQXN(:, JM)``.
* Diagonal divide on the last row: ``ZQXN(:, NCLV) /= ZQLHS(:, NCLV, NCLV)``.
* Back-substitute: ``DO JN=NCLV-1,1,-1 / DO JM=JN+1,NCLV`` with both
  the row-update and the row-divide.

All four loop nests use Fortran array slice notation on the column
dimension (``KIDIA:KFDIA``), which the bridge has to lower as elemental
JL loops without breaking the JN/JM sequential-dependency contract.
A bug in iteration order, slice-write tasklet wiring, or section
aliasing would produce small per-cell errors that look exactly like
the cloudsc cumulative-flux mismatches.

E2e against an f2py-compiled reference of the same Fortran source.
"""
import dace
import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_cloudsc_lu_solver(tmp_path):
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE kernel(zqlhs, zqxn, kidia, kfdia, klon, nclv)
integer, intent(in) :: kidia, kfdia, klon, nclv
double precision, intent(inout) :: zqlhs(klon, nclv, nclv)
double precision, intent(inout) :: zqxn(klon, nclv)
integer :: jn, jm, ik, jl

! Non pivoting recursive factorization
DO jn = 1, nclv - 1
    DO jm = jn + 1, nclv
        zqlhs(kidia:kfdia, jm, jn) = zqlhs(kidia:kfdia, jm, jn) &
         &                          / zqlhs(kidia:kfdia, jn, jn)
        DO ik = jn + 1, nclv
            DO jl = kidia, kfdia
                zqlhs(jl, jm, ik) = zqlhs(jl, jm, ik) - zqlhs(jl, jm, jn) * zqlhs(jl, jn, ik)
            ENDDO
        ENDDO
    ENDDO
ENDDO

! Backsubstitution step 1 (forward sweep on rhs)
DO jn = 2, nclv
    DO jm = 1, jn - 1
        zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) &
         &                    * zqxn(kidia:kfdia, jm)
    ENDDO
ENDDO

! Backsubstitution step 2 (diagonal divide on last row then sweep up)
zqxn(kidia:kfdia, nclv) = zqxn(kidia:kfdia, nclv) / zqlhs(kidia:kfdia, nclv, nclv)
DO jn = nclv - 1, 1, -1
    DO jm = jn + 1, nclv
        zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) - zqlhs(kidia:kfdia, jn, jm) &
         &                    * zqxn(kidia:kfdia, jm)
    ENDDO
    zqxn(kidia:kfdia, jn) = zqxn(kidia:kfdia, jn) / zqlhs(kidia:kfdia, jn, jn)
ENDDO
END SUBROUTINE kernel

SUBROUTINE driver(zqlhs, zqxn, klon, nclv, nblocks)
integer, intent(in) :: klon, nclv, nblocks
double precision, intent(inout) :: zqlhs(klon, nclv, nclv, nblocks)
double precision, intent(inout) :: zqxn(klon, nclv, nblocks)
integer :: ibl
DO ibl = 1, nblocks
    CALL kernel(zqlhs(:, :, :, ibl), zqxn(:, :, ibl), 1, klon, klon, nclv)
ENDDO
END SUBROUTINE driver
END MODULE kernel_mod
"""
    ref = f2py(src, tmp_path / 'ref', 'cloudsc_lu_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='cloudsc_lu', entry='_QMkernel_modPdriver').build()

    klon, nclv, nblocks = 1, 5, 4
    rng = np.random.default_rng(11)

    # Diagonally-dominant ZQLHS so the LU stays numerically stable.
    base = rng.random((klon, nclv, nclv, nblocks))
    for ibl in range(nblocks):
        for jl in range(klon):
            for jn in range(nclv):
                base[jl, jn, jn, ibl] += nclv  # diag dominance
    zqlhs_in = np.asfortranarray(base)
    zqxn_in = np.asfortranarray(rng.random((klon, nclv, nblocks)))

    zqlhs_ref = np.asfortranarray(zqlhs_in.copy())
    zqxn_ref = np.asfortranarray(zqxn_in.copy())
    ref.kernel_mod.driver(zqlhs_ref, zqxn_ref, klon, nclv, nblocks)

    zqlhs = np.asfortranarray(zqlhs_in.copy())
    zqxn = np.asfortranarray(zqxn_in.copy())
    sdfg(zqlhs=zqlhs, zqxn=zqxn, klon=klon, nclv=nclv, nblocks=nblocks)

    np.testing.assert_allclose(zqxn, zqxn_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zqlhs, zqlhs_ref, rtol=1e-12, atol=1e-12)


_PY_KLON = dace.symbol('_PY_KLON')
_PY_NCLV = dace.symbol('_PY_NCLV')


@dace.program
def _py_lu_solver(zqlhs: dace.float64[_PY_KLON + 1, _PY_NCLV + 1, _PY_NCLV + 1], zqxn: dace.float64[_PY_KLON + 1,
                                                                                                    _PY_NCLV + 1]):
    """Same LU solve as the Fortran kernel, written in the DaCe Python
    frontend with 1-indexed sequential loops + 1-indexed ``dace.map``
    on the parallel JL dimension.

    Defined at module scope so that ``@dace.program``'s annotation
    resolution can see ``_PY_KLON`` and ``_PY_NCLV`` as module-level
    symbols."""
    for jn in range(1, _PY_NCLV):
        for jm in range(jn + 1, _PY_NCLV + 1):
            for jl in dace.map[1:_PY_KLON + 1]:
                a = zqlhs[jl, jm, jn]
                b = zqlhs[jl, jn, jn]
                zqlhs[jl, jm, jn] = a / b
            for ik in range(jn + 1, _PY_NCLV + 1):
                for jl in dace.map[1:_PY_KLON + 1]:
                    a = zqlhs[jl, jm, ik]
                    b = zqlhs[jl, jm, jn]
                    c = zqlhs[jl, jn, ik]
                    zqlhs[jl, jm, ik] = a - b * c
    for jn in range(2, _PY_NCLV + 1):
        for jm in range(1, jn):
            for jl in dace.map[1:_PY_KLON + 1]:
                a = zqxn[jl, jn]
                b = zqlhs[jl, jn, jm]
                c = zqxn[jl, jm]
                zqxn[jl, jn] = a - b * c
    for jl in dace.map[1:_PY_KLON + 1]:
        a = zqxn[jl, _PY_NCLV]
        b = zqlhs[jl, _PY_NCLV, _PY_NCLV]
        zqxn[jl, _PY_NCLV] = a / b
    for jn in range(_PY_NCLV - 1, 0, -1):
        for jm in range(jn + 1, _PY_NCLV + 1):
            for jl in dace.map[1:_PY_KLON + 1]:
                a = zqxn[jl, jn]
                b = zqlhs[jl, jn, jm]
                c = zqxn[jl, jm]
                zqxn[jl, jn] = a - b * c
        for jl in dace.map[1:_PY_KLON + 1]:
            a = zqxn[jl, jn]
            b = zqlhs[jl, jn, jn]
            zqxn[jl, jn] = a / b


@pytest.mark.xfail(
    strict=True,
    reason="DaCe Python frontend produces NaNs when sequential outer "
    "``range()`` loops are nested with two sibling ``dace.map`` "
    "blocks accessing 1-indexed offsets.  The Fortran-bridge port of "
    "the same algorithm passes, so the bug sits in DaCe core's "
    "offset/map sequencing, not the HLFIR bridge.",
)
def test_python_frontend_cloudsc_lu_solver_one_indexed():
    """Same LU solve via the DaCe Python frontend with 1-indexed
    ``dace.map[1:N+1]`` ranges.

    Verifies that DaCe core handles offset (non-zero-based) map ranges
    + sequential outer (range) loops + parallel inner JL maps without
    the HLFIR bridge in the loop.  If the HLFIR bridge produces correct
    SDFG for the Fortran LU but the Python-frontend variant fails, the
    bug is in DaCe's offset/map lowering rather than the bridge.
    """
    klon, nclv = 3, 5
    rng = np.random.default_rng(11)
    base = rng.random((klon, nclv, nclv))
    for jl in range(klon):
        for jn in range(nclv):
            base[jl, jn, jn] += nclv  # diag dominance
    zqlhs_in = np.zeros((klon + 1, nclv + 1, nclv + 1), dtype=np.float64, order="F")
    zqlhs_in[1:, 1:, 1:] = base
    zqxn_in = np.zeros((klon + 1, nclv + 1), dtype=np.float64, order="F")
    zqxn_in[1:, 1:] = rng.random((klon, nclv))

    def _np_lu(zqlhs, zqxn):
        for jn in range(1, nclv):
            for jm in range(jn + 1, nclv + 1):
                zqlhs[1:, jm, jn] /= zqlhs[1:, jn, jn]
                for ik in range(jn + 1, nclv + 1):
                    zqlhs[1:, jm, ik] -= zqlhs[1:, jm, jn] * zqlhs[1:, jn, ik]
        for jn in range(2, nclv + 1):
            for jm in range(1, jn):
                zqxn[1:, jn] -= zqlhs[1:, jn, jm] * zqxn[1:, jm]
        zqxn[1:, nclv] /= zqlhs[1:, nclv, nclv]
        for jn in range(nclv - 1, 0, -1):
            for jm in range(jn + 1, nclv + 1):
                zqxn[1:, jn] -= zqlhs[1:, jn, jm] * zqxn[1:, jm]
            zqxn[1:, jn] /= zqlhs[1:, jn, jn]

    zqlhs_ref = zqlhs_in.copy(order="F")
    zqxn_ref = zqxn_in.copy(order="F")
    _np_lu(zqlhs_ref, zqxn_ref)

    zqlhs = zqlhs_in.copy(order="F")
    zqxn = zqxn_in.copy(order="F")
    _py_lu_solver(zqlhs=zqlhs, zqxn=zqxn, _PY_NCLV=nclv, _PY_KLON=klon)

    np.testing.assert_allclose(zqxn[1:, 1:], zqxn_ref[1:, 1:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(zqlhs[1:, 1:, 1:], zqlhs_ref[1:, 1:, 1:], rtol=1e-12, atol=1e-12)
