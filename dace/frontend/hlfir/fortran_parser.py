"""Drop-in replacement for ``dace.frontend.fortran.fortran_parser`` that
drives the HLFIR-based SDFG builder.

The public surface mirrors the legacy frontend's so test files can swap
``from dace.frontend.fortran import fortran_parser`` for ``from
dace.frontend.hlfir import fortran_parser`` without further edits.
Unrecognised keyword arguments are silently accepted for interface
compatibility; only the subset of features the HLFIR frontend currently
lowers will actually produce correct SDFGs.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from dace import SDFG

from build_bridge import hb  # noqa: F401   --  ensures the bridge is built
from hlfir_to_sdfg import DEFAULT_PIPELINE, SDFGBuilder


def _find_flang() -> str:
    bin_ = shutil.which("flang-new-21") or shutil.which("flang-new-20")
    if bin_ is None:
        raise RuntimeError("flang-new-21 (or -20) not on PATH; install LLVM/Flang to use "
                           "the HLFIR frontend.")
    return bin_


def create_sdfg_from_string(source: str, name: str = "sdfg", pipeline: str = DEFAULT_PIPELINE, **_ignored) -> SDFG:
    """Compile Fortran source string -> HLFIR -> SDFG.

    :param source: Fortran source as a single string.  Program units that
                   mix a ``PROGRAM`` with a ``SUBROUTINE`` are not yet
                   lowered cleanly  --  prefer single-subroutine sources.
    :param name: basename used for the scratch ``.f90`` / ``.hlfir`` files.
    :param pipeline: MLIR pass pipeline to run before AST extraction.
    :param _ignored: legacy kwargs (``normalize_offsets``, ``use_explicit_cf``,
                     ``entry_point``, ...) accepted for drop-in compatibility.
    """
    flang = _find_flang()
    with tempfile.TemporaryDirectory(prefix=f"hlfir_{name}_") as td:
        tdp = Path(td)
        src = tdp / f"{name}.f90"
        src.write_text(source)
        hlfir = tdp / f"{name}.hlfir"
        subprocess.check_call([flang, "-fc1", "-emit-hlfir", str(src), "-o", str(hlfir)])
        return SDFGBuilder(str(hlfir), pipeline=pipeline).build()


def create_singular_sdfg_from_string(source, entry_point="main", **kw):
    """Legacy alias  --  ``entry_point`` is currently ignored (the HLFIR
    extractor emits the first func.func it finds)."""
    return create_sdfg_from_string(source, name=entry_point, **kw)
