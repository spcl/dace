"""Whole-array scalar reductions → DaCe ``standard.Reduce`` library node.

Checks that ``sum``, ``product``, ``minval``, ``maxval`` each lower
through Flang's dedicated HLFIR op into an SDFG Reduce node, and that
the numerical result matches the gfortran/f2py-compiled reference.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "reduce_intrinsics.f90"


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


def test_scalar_reductions_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "reduce_scalar_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="reduce_scalar", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(7)
    n = 24
    # Positive values so product / minval / maxval are well-conditioned.
    a = rng.uniform(0.1, 2.0, size=n)

    # Reference.
    t_ref, p_ref, lo_ref, hi_ref = (np.zeros(1, order="F"), np.zeros(1, order="F"), np.zeros(1, order="F"),
                                    np.zeros(1, order="F"))
    mod.reduce_scalar(np.asfortranarray(a), t_ref, p_ref, lo_ref, hi_ref)

    # SDFG — ``intent(inout)`` scalar parameters land as size-1 Array
    # descriptors (DaCe can't put Scalars on the external signature), so
    # the caller binds size-1 numpy arrays.
    t_sdfg = np.zeros(1, dtype=np.float64)
    p_sdfg = np.zeros(1, dtype=np.float64)
    lo_sdfg = np.zeros(1, dtype=np.float64)
    hi_sdfg = np.zeros(1, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), total=t_sdfg, prod=p_sdfg, lo=lo_sdfg, hi=hi_sdfg, n=n)

    np.testing.assert_allclose(t_sdfg[0], t_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p_sdfg[0], p_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lo_sdfg[0], lo_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hi_sdfg[0], hi_ref[0], rtol=1e-12, atol=1e-12)


def test_scalar_reductions_structure(tmp_path):
    """The SDFG should contain four ``Reduce`` library nodes, one per
    ``sum / product / minval / maxval`` call, plus the scalar outputs as
    non-transient SDFG data."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="reduce_scalar", pipeline="hlfir-propagate-shapes").build()

    from dace.sdfg.state import LoopRegion, SDFGState
    from dace.libraries.standard.nodes.reduce import Reduce

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif isinstance(n, SDFGState):
                yield n

    reduces = [n for s in iter_states(sdfg) for n in s.nodes() if isinstance(n, Reduce)]
    assert len(reduces) == 4, (f"expected 4 Reduce library nodes; got {len(reduces)}")

    wcrs = sorted(r.wcr for r in reduces)
    assert 'lambda a, b: a + b' in wcrs
    assert 'lambda a, b: a * b' in wcrs
    assert 'lambda a, b: min(a, b)' in wcrs
    assert 'lambda a, b: max(a, b)' in wcrs
