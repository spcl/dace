"""Linear-algebra Fortran intrinsics -> DaCe library nodes.

Exercises the four ``hlfir.*`` linalg ops that bypass the elemental
path and go straight to dedicated library nodes:

    hlfir.matmul       -> blas.MatMul   (GEMM / GEMV shapes)
    hlfir.transpose    -> standard.Transpose
    hlfir.dot_product  -> blas.Dot

Each result is compared numerically against the gfortran/f2py-compiled
Fortran reference on seeded random input.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "linalg_intrinsics.f90"


def _f2py(src: Path, out_dir: Path, mod_name: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c", str(src), "-m", mod_name, "--quiet"], cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def test_linalg_ops_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "linalg_ops_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="linalg_ops", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(11)
    n, m, k = 5, 4, 3
    a = rng.standard_normal((n, m))
    b = rng.standard_normal((m, k))
    v = rng.standard_normal(m)
    u = rng.standard_normal(n)

    # Reference  --  gfortran-compiled.
    c_ref = np.zeros((n, k), order="F")
    at_ref = np.zeros((m, n), order="F")
    w_ref = np.zeros(n, order="F")
    s_ref = np.zeros(1, order="F")
    mod.linalg_ops(np.asfortranarray(a), np.asfortranarray(b), c_ref, at_ref, np.asfortranarray(v), w_ref,
                   np.asfortranarray(u), s_ref)

    # SDFG  --  frontend now emits Fortran-order strides for rank>1
    # descriptors, so pass F-order arrays to the matmul/transpose ops
    # to match the caller-side convention.
    c_sdfg = np.zeros((n, k), dtype=np.float64, order="F")
    at_sdfg = np.zeros((m, n), dtype=np.float64, order="F")
    w_sdfg = np.zeros(n, dtype=np.float64)
    s_sdfg = np.zeros(1, dtype=np.float64)
    sdfg(a=np.asfortranarray(a),
         b=np.asfortranarray(b),
         c=c_sdfg,
         at=at_sdfg,
         v=np.asfortranarray(v),
         w=w_sdfg,
         u=np.asfortranarray(u),
         s=s_sdfg,
         n=n,
         m=m,
         k=k)

    np.testing.assert_allclose(c_sdfg, c_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(at_sdfg, at_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_sdfg, w_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_sdfg[0], s_ref[0], rtol=1e-12, atol=1e-12)


_TRANSPOSE_OF_ELEMENTAL_SRC = """
subroutine probe(d, res)
  implicit none
  double precision, intent(in)    :: d(16, 5)
  double precision, intent(inout) :: res(5, 16)
  res = transpose(1.0d0 - d)
end subroutine probe
"""


def test_transpose_of_elemental(tmp_path):
    """Regression: ``transpose(<inline elementwise expr>)``  --  the
    transpose's operand is an ``hlfir.expr`` produced by an inline
    ``hlfir.elemental`` (here ``1.0d0 - d``), not a named array.  Without
    the libcall-over-elemental materialise path, ``buildLibCallNode``'s
    ``traceToDecl`` on the operand returns ``""``, ``emit_libcall``
    looks up ``ctx.sdfg.arrays['']``, and the build raises
    ``KeyError: ''``.

    Triggers Phase 2's ``materialiseElementalForLibcall`` at the direct
    libcall dispatch site (``dispatch.cpp``).  Pure isolation  --  no
    gather, no triplet, so the test fails iff the elemental-source
    materialise specifically regresses."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    src_path = tmp_path / "transpose_of_elem.f90"
    src_path.write_text(_TRANSPOSE_OF_ELEMENTAL_SRC)
    mod = _f2py(src_path, tmp_path / "ref", "transpose_of_elem_ref")

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_TRANSPOSE_OF_ELEMENTAL_SRC, sdfg_dir, name="probe", entry="_QPprobe").build()

    rng = np.random.default_rng(7)
    d = np.asfortranarray(rng.standard_normal((16, 5)))
    res_ref = np.zeros((5, 16), order="F", dtype=np.float64)
    res_sdfg = np.zeros((5, 16), order="F", dtype=np.float64)
    mod.probe(d, res_ref)
    sdfg(d=d, res=res_sdfg)
    np.testing.assert_allclose(res_sdfg, res_ref, rtol=1e-12, atol=1e-12)


def test_linalg_ops_structure(tmp_path):
    """The SDFG should carry two MatMul, one Transpose, and one Dot
    library node  --  one per intrinsic call."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="linalg_ops", pipeline="hlfir-propagate-shapes").build()

    from dace.sdfg.state import LoopRegion, SDFGState
    from dace.libraries.blas.nodes import MatMul, Dot
    from dace.libraries.standard.nodes import Transpose

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif isinstance(n, SDFGState):
                yield n

    nodes = [n for s in iter_states(sdfg) for n in s.nodes()]
    matmuls = [n for n in nodes if isinstance(n, MatMul)]
    transposes = [n for n in nodes if isinstance(n, Transpose)]
    dots = [n for n in nodes if isinstance(n, Dot)]
    assert len(matmuls) == 2, f"expected 2 MatMul nodes, got {len(matmuls)}"
    assert len(transposes) == 1, f"expected 1 Transpose node, got {len(transposes)}"
    assert len(dots) == 1, f"expected 1 Dot node, got {len(dots)}"
