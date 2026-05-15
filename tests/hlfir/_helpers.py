"""Shared helpers for verbatim ports from ``f2dace/dev:tests/fortran/``.

These mirror the helpers used in the FaCe-native ports
(``baseline_*_test.py``) so a single canonical
implementation is reused across every ported file:

- ``_f2py(src, out_dir, mod_name)``    --  gfortran-backed reference build.
- ``_sdfg_call_args(sdfg, int_vals)``  --  route int args to scalar vs
  length-1-Array based on what ``sdfg.arglist()`` classifies them as.
- ``_xfail(reason, *, strict=True)``   --  uniform strict-xfail marker.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def f2py(src_text: str, out_dir: Path, mod_name: str):
    """Compile ``src_text`` as a Python extension via ``numpy.f2py`` and
    return the imported module.  Skips the test if gfortran or meson is
    not installed."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir / f"{mod_name}.f90"
    src_file.write_text(src_text)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(src_file), "-m", mod_name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each integer arg in ``int_values`` to either a plain int or
    a length-1 numpy int32 array, depending on whether the SDFG
    descriptor classifies it as a Scalar/symbol or a length-1 Array.
    Mirrors the helper in ``icon_loopnests/test_sdfg_equivalence.py``.
    """
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


def xfail(reason: str, *, strict: bool = True):
    """Uniform strict-xfail marker  --  any silent xpass should fire so
    flipped-green tests get a deliberate, visible un-marking."""
    return pytest.mark.xfail(strict=strict, reason=reason)
