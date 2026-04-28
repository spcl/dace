"""Struct-vs-flat numerical equivalence for the six E6 velocity-advection
representative loopnests.

Each ``loopnest_N.f90`` file is self-contained: a module with both a
struct-typed kernel and a flattened kernel, plus a driver that allocates
deterministic random inputs (fixed seed per file), calls both kernels,
and exits 0 iff their outputs match within 1e-12 (or bit-exactly for
the logical mask of loopnest 6).

The pytest here compiles each driver with ``gfortran`` (the only
Fortran compiler with a runtime on this box — ``flang-new-21`` lacks
``libflang_rt.runtime``), runs the binary, and asserts exit 0.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_GFORTRAN = shutil.which("gfortran")
_KERNELS = sorted(_HERE.glob("icon_loopnest_*.f90"))

pytestmark = pytest.mark.skipif(_GFORTRAN is None, reason="gfortran not on PATH")


@pytest.mark.parametrize("src", _KERNELS, ids=lambda p: p.stem)
def test_struct_flat_equivalence(src: Path, tmp_path: Path):
    """Struct-typed and flat versions of the E6 representative kernel
    produce bit-equal outputs on the same deterministic inputs."""
    exe = tmp_path / src.stem
    compile_cmd = [_GFORTRAN, "-O2", "-fcheck=bounds", str(src), "-o", str(exe)]
    subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, \
        f"{src.name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    # Keep the "OK max_err=..." line visible in -s mode for quick scanning.
    print(f"[{src.stem}] {result.stdout.strip()}")
