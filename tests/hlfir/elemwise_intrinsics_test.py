"""End-to-end test for elementwise Fortran intrinsics.

Builds ``elemwise_sin(a, b, n)`` as an SDFG through the HLFIR frontend —
Flang lowers ``b = sin(a) + 2.0d0 * a`` into composed ``hlfir.elemental``
ops, our bridge walks them into a nested ``kind="loop"`` + ``kind="assign"``
AST, and the SDFG emitter produces a DaCe ``LoopRegion`` whose tasklet
body is ``_out_b = sin(_in_a_0) + (2.0 * _in_a_1)``.

The reference is compiled with gfortran through ``numpy.f2py``; both run
on seeded random input and outputs must match to ``1e-12``.
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
_SRC_PATH = _HERE / "elemwise_intrinsics.f90"


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


def test_elemwise_sin_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "elemwise_sin_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="elemwise_sin", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(42)
    n = 16
    a = rng.standard_normal(n)

    # f2py reads Fortran-order; DaCe default is C — same logical data, one
    # layout-matching copy each.
    b_ref = np.zeros(n, order="F")
    mod.elemwise_sin(np.asfortranarray(a), b_ref)

    b_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, n=n)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)


def test_elemwise_sin_structure(tmp_path):
    """Sanity check the SDFG shape: one LoopRegion over the shape dim, one
    Python-language tasklet whose body calls ``sin`` with the bare name."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="elemwise_sin", pipeline="hlfir-propagate-shapes").build()

    from dace.sdfg.state import LoopRegion, SDFGState
    loops = [n for n in sdfg.nodes() if isinstance(n, LoopRegion)]
    assert len(loops) >= 1, "expected at least one LoopRegion"

    # Walk into every state and look for the sin tasklet.
    def walk(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from walk(n)
            elif isinstance(n, SDFGState):
                yield n

    tasklets = [t for s in walk(sdfg) for t in s.nodes() if isinstance(t, nd := __import__('dace').sdfg.nodes.Tasklet)]
    assert any("sin(" in t.code.as_string
               for t in tasklets), ("no tasklet body calls sin; got: " + repr([t.code.as_string for t in tasklets]))
