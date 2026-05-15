"""Whole-array copy -> ``CopyLibraryNode`` and zero-fill -> ``MemsetLibraryNode``.

Exercises the two ``hlfir.assign`` shapes that skip the tasklet/loop path
and go straight to library nodes on FaCe.  Compared numerically against
the gfortran/f2py-compiled reference on seeded random input.
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
_SRC_PATH = _HERE / "copy_memset.f90"


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


def test_copy_and_memset_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "copy_and_memset_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="copy_and_memset",
                      pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(9)
    n = 17
    a = rng.standard_normal(n)

    # Start b and c with nonzero values to catch cases where the copy or
    # memset didn't run.
    b_sdfg = np.full(n, 7.25, dtype=np.float64)
    c_sdfg = np.full(n, 7.25, dtype=np.float64)

    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, c=c_sdfg, n=n)

    # Reference  --  gfortran-compiled whole-array assign / zero-fill.
    b_ref = np.full(n, 7.25, order="F", dtype=np.float64)
    c_ref = np.full(n, 7.25, order="F", dtype=np.float64)
    mod.copy_and_memset(np.asfortranarray(a), b_ref, c_ref)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_sdfg, c_ref, rtol=1e-12, atol=1e-12)


def test_copy_and_memset_structure(tmp_path):
    """The SDFG should carry exactly one CopyLibraryNode (for ``b = a``)
    and one MemsetLibraryNode (for ``c = 0.0``), rather than tasklet-and-
    loop decompositions."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="copy_and_memset",
                      pipeline="hlfir-propagate-shapes").build()

    from dace.sdfg.state import LoopRegion, SDFGState
    from dace.libraries.standard.nodes import (CopyLibraryNode, MemsetLibraryNode)

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif isinstance(n, SDFGState):
                yield n

    nodes = [n for s in iter_states(sdfg) for n in s.nodes()]
    copies = [n for n in nodes if isinstance(n, CopyLibraryNode)]
    memsets = [n for n in nodes if isinstance(n, MemsetLibraryNode)]
    assert len(copies) == 1, f"expected 1 CopyLibraryNode, got {len(copies)}"
    assert len(memsets) == 1, f"expected 1 MemsetLibraryNode, got {len(memsets)}"
