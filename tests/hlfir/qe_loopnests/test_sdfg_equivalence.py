"""End-to-end tests for the QE / SC26-Layout-AD experiment kernels
(E1-E5) translated to Fortran.

Each kernel is in ``tests/hlfir/qe_loopnests/qe_eN_*.f90``;
this harness compiles it through the bridge AND through ``f2py``,
then asserts numerical equivalence on a small random input.

Source experiments (without the NUMA / multi-allocator harness):
  * E1 MatrixAdd   --  ``C(i,j) = C(i,j) + A(i,j) + B(i,j)``
  * E2 Conjugate   --  ``b(i) = conjg(b(i))`` complex(8)
  * E3 Transpose   --  ``B(j,i) = A(i,j)``
  * E4 GAS/zaxpy   --  ``Y(i) = a*X(i) + Y(i)`` complex(8)
  * E5 USXX scatter  --  ``rhoc(nl(i)) = rhoc(nl(i)) + aux2(i)``
                       (the addusxx_g hot inner loop)
  * E5 USXX phase    --  ``eigqts(na) = cos(arg) - i*sin(arg)``
                       (the per-atom phase factor)

Each test compares SDFG output against gfortran/f2py reference
bit-exact (``rtol=1e-12`` real(8); ``rtol=1e-12`` complex(8) too).
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_LOOPNESTS_DIR = Path(__file__).parent


def _src(name: str) -> str:
    p = _LOOPNESTS_DIR / f"{name}.f90"
    if not p.is_file():
        pytest.skip(f"missing kernel source: {p}")
    return p.read_text()


# ---------------------------------------------------------------------------
# E1  --  MatrixAdd
# ---------------------------------------------------------------------------


def test_e1_matrix_add(tmp_path: Path):
    src = _src("qe_e1_matrix_add")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e1_matrix_add_ref")

    rng = np.random.default_rng(0)
    m, n = 32, 48
    a = np.asfortranarray(rng.random((m, n))).astype(np.float64)
    b = np.asfortranarray(rng.random((m, n))).astype(np.float64)
    c = np.asfortranarray(rng.random((m, n))).astype(np.float64)
    # f2py auto-derives m, n from ``a``  --  they're intent(hide).  ``c``
    # is intent(inout) so we have to pass a Fortran-contiguous copy.
    c_ref = np.asfortranarray(c.copy())
    mod.kernel(a, b, c_ref)
    sdfg(m=m, n=n, a=a, b=b, c=c)
    np.testing.assert_allclose(c, c_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# E2  --  Conjugate
# ---------------------------------------------------------------------------


def test_e2_conjugate_inplace(tmp_path: Path):
    src = _src("qe_e2_conjugate_inplace")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e2_conjugate_ref")

    rng = np.random.default_rng(1)
    n = 64
    b = (rng.random(n) + 1j * rng.random(n)).astype(np.complex128)
    b_ref = b.copy()
    mod.kernel(b_ref)
    sdfg(n=n, b=b)
    np.testing.assert_allclose(b, b_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# E3  --  Transpose
# ---------------------------------------------------------------------------


def test_e3_transpose(tmp_path: Path):
    src = _src("qe_e3_transpose")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e3_transpose_ref")

    rng = np.random.default_rng(2)
    n = 24
    a = np.asfortranarray(rng.random((n, n))).astype(np.float64)
    b_sdfg = np.zeros((n, n), order='F', dtype=np.float64)
    # f2py converts ``b`` (intent(out)) to a return value.
    b_ref = mod.kernel(a)
    sdfg(n=n, a=a, b=b_sdfg)
    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# E4  --  GAS / zaxpy
# ---------------------------------------------------------------------------


def test_e4_zaxpy(tmp_path: Path):
    src = _src("qe_e4_zaxpy")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e4_zaxpy_ref")

    rng = np.random.default_rng(3)
    n = 32
    # ``a`` as length-1 array  --  see comment in the kernel source.
    a = np.array([0.7 - 0.3j], dtype=np.complex128)
    x = (rng.random(n) + 1j * rng.random(n)).astype(np.complex128)
    y = (rng.random(n) + 1j * rng.random(n)).astype(np.complex128)
    y_ref = y.copy()
    mod.kernel(a, x, y_ref)
    sdfg(n=n, a=a, x=x, y=y)
    np.testing.assert_allclose(y, y_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# E5  --  USXX scatter (the addusxx_g hot inner loop)
# ---------------------------------------------------------------------------


def test_e5_usxx_scatter(tmp_path: Path):
    src = _src("qe_e5_usxx_scatter")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e5_usxx_scatter_ref")

    rng = np.random.default_rng(4)
    blocksize = 8
    nrxxs = 32
    # ``nl`` is 1-based and may have repeats  --  the kernel ACCUMULATES
    # into rhoc, so duplicate indices must add multiple aux2 values
    # into the same slot.  We don't repeat here for simplicity.
    nl = rng.permutation(nrxxs)[:blocksize].astype(np.int32) + 1
    aux2 = (rng.random(blocksize) + 1j * rng.random(blocksize)).astype(np.complex128)
    rhoc = (rng.random(nrxxs) + 1j * rng.random(nrxxs)).astype(np.complex128)
    rhoc_ref = rhoc.copy()
    mod.kernel(nl, aux2, rhoc_ref)
    sdfg(blocksize=blocksize, nrxxs=nrxxs, nl=nl, aux2=aux2, rhoc_out=rhoc)
    np.testing.assert_allclose(rhoc, rhoc_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# E5  --  USXX phase factor (per-atom)
# ---------------------------------------------------------------------------


def test_e5_usxx_phase(tmp_path: Path):
    src = _src("qe_e5_usxx_phase")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    mod = f2py_compile(src, tmp_path / "ref", "e5_usxx_phase_ref")

    rng = np.random.default_rng(5)
    nat = 16
    xk = rng.random(3).astype(np.float64)
    xkq = rng.random(3).astype(np.float64)
    tau = np.asfortranarray(rng.random((3, nat))).astype(np.float64)
    eigqts_sdfg = np.zeros(nat, dtype=np.complex128)
    # f2py converts ``eigqts`` (intent(out)) to a return value;
    # ``nat`` is auto-derived from ``tau.shape[1]``.
    eigqts_ref = mod.kernel(xk, xkq, tau)
    sdfg(nat=nat, xk=xk, xkq=xkq, tau=tau, eigqts=eigqts_sdfg)
    np.testing.assert_allclose(eigqts_sdfg, eigqts_ref, rtol=1e-12)
