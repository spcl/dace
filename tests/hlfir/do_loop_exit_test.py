"""Fortran EXIT inside a counted DO.

Flang doesn't emit a structured ``fir.do_loop`` when a loop body contains
``EXIT``: it drops to ``cf.br`` / ``cf.cond_br``.  The ``lift-cf-to-scf``
pass folds the exit edge into an ``scf.while`` whose ``scf.condition``
encodes the combined keep-going predicate.  This test pins that shape
end-to-end against the gfortran reference.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "do_loop_exit.f90"


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


def test_do_loop_exit_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "do_exit_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="do_exit").build()
    sdfg.validate()

    rng = np.random.default_rng(23)
    n = 12
    # Mix values below and above the 100.0 exit threshold so at least one
    # early exit actually fires.
    a = rng.uniform(-50.0, 150.0, size=n)

    b_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    mod.do_exit(np.asfortranarray(a), b_ref)

    b_sdfg = np.full(n, 9.0, dtype=np.float64)
    # ``i`` is a local counter-symbol.  The SDFG initialises it via an
    # interstate-edge ``i = 1`` on the first state, but DaCe's free-symbol
    # analysis still lists it so the caller has to pass a placeholder.
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, n=n, i=0)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
