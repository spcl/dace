"""A local scalar variable used only as an array index should become an
SDFG symbol, so writes to it emit an interstate-edge assignment that
bumps the state machine forward.

This is a stricter classification than the current bridge uses  --  shape
and loop-bound scalars already land as symbols, but an "index-only
scalar" like ``ix`` in a ``b(i) = a(ix); ix = ix + 1`` pattern does not.
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
_SRC_PATH = _HERE / "symbol_from_index.f90"


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


def test_symbol_from_index(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "idx_sym_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="idx_sym").build()
    sdfg.validate()

    n = 8
    a = np.arange(n, dtype=np.float64) + 1.0

    b_ref = np.zeros(n, order="F", dtype=np.float64)
    mod.idx_sym(np.asfortranarray(a), b_ref)

    b_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, n=n)
    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
