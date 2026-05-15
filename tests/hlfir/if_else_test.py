"""IF / ELSE branching inside a DO loop.

Exercises ``_emit_cond`` end-to-end: a THEN/ELSE pair (writing ``b``) and
a THEN-only IF (writing ``c``), both guarded on an element-wise numeric
condition.  The SDFG result must match the gfortran-compiled reference
on seeded random input.
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
_SRC_PATH = _HERE / "if_else.f90"


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


def test_if_else_branch_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "if_else_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="if_else_branch", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(17)
    n = 16
    a = rng.standard_normal(n)

    # Sentinel values so dropped branches are visible.
    b_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    c_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    mod.if_else_branch(np.asfortranarray(a), b_ref, c_ref)

    b_sdfg = np.full(n, 9.0, dtype=np.float64)
    c_sdfg = np.full(n, 9.0, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, c=c_sdfg, n=n)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_sdfg, c_ref, rtol=1e-12, atol=1e-12)
