"""Helpers for the HLFIR frontend test suite.

The HLFIR frontend takes a pre-lowered ``.hlfir`` file, not raw Fortran
source.  Tests stay readable by writing the Fortran inline and letting
this helper run ``flang-new-21 -fc1 -emit-hlfir`` behind the scenes.  If
no flang binary is found the helpers report it; the calling test module
should skip collection accordingly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HLFIR_DIR = _REPO_ROOT / "dace" / "frontend" / "hlfir"

# When this env var is set, every ``build_sdfg(...).build()`` call dumps
# its SDFG to the named directory for offline inspection.  Treating "1"
# / "true" / "yes" as the shorthand for a default path under /tmp keeps
# the common case ergonomic without hardcoding one.
_DUMP_ENV = "__DACE_HLFIR_GEN_TEST_SDFGS"
_DEFAULT_DUMP_DIR = Path("/tmp/hlfir_test_sdfgs")


def _dump_dir() -> Path | None:
    val = os.environ.get(_DUMP_ENV)
    if not val:
        return None
    if val.lower() in ("1", "true", "yes", "on"):
        return _DEFAULT_DUMP_DIR
    return Path(val)


# Prefer llvm-21 (matches build_bridge.py default); fall back to 20.
_FLANG = shutil.which("flang-new-21") or shutil.which("flang-new-20")


def have_flang() -> bool:
    return _FLANG is not None


def _ensure_on_path():
    p = str(_HLFIR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def compile_to_hlfir(source: str, out_dir: Path, name: str = "src") -> Path:
    """Write `source` as <out_dir>/<name>.f90, compile it to HLFIR, return the path."""
    assert _FLANG is not None, "flang-new-21 not available"
    src = out_dir / f"{name}.f90"
    src.write_text(source)
    hlfir = out_dir / f"{name}.hlfir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(src), "-o", str(hlfir)])
    return hlfir


class _DumpingBuilder:
    """Thin proxy around ``SDFGBuilder`` that dumps the built SDFG when
    ``__DACE_HLFIR_GEN_TEST_SDFGS`` is set.  Everything else flows through
    to the wrapped builder unchanged (``.arrays`` / ``.scalars`` / … still
    work the same way for tests that inspect them)."""

    def __init__(self, inner, name: str, dump_dir: Path):
        self._inner = inner
        self._name = name
        self._dump_dir = dump_dir

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def build(self):
        sdfg = self._inner.build()
        self._dump_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._dump_dir / f"{self._name}.sdfg"
        sdfg.save(str(out_path))
        return sdfg


def build_sdfg(source: str, out_dir: Path, name: str = "src", pipeline=None):
    """Compile inline Fortran to HLFIR and return a configured ``SDFGBuilder``.

    :param source: Fortran source as a string.
    :param out_dir: scratch directory (typically ``tmp_path`` from pytest).
    :param name: base filename for the .f90/.hlfir pair.
    :param pipeline: override the default MLIR pass pipeline.
    :return: ``SDFGBuilder`` with variables classified and the AST extracted.
    """
    _ensure_on_path()
    from hlfir_to_sdfg import SDFGBuilder, DEFAULT_PIPELINE
    hlfir = compile_to_hlfir(source, out_dir, name)
    builder = SDFGBuilder(str(hlfir), pipeline=(pipeline or DEFAULT_PIPELINE))
    dump = _dump_dir()
    if dump is not None:
        return _DumpingBuilder(builder, name, dump)
    return builder


def run_passes_dump(source: str, out_dir: Path, name: str = "src", pipeline: str = "builtin.module()") -> str:
    """Compile Fortran to HLFIR, run the given pipeline, return the IR dump.

    Use this when the test inspects post-pass MLIR directly rather than going
    through SDFG extraction — handy for passes whose downstream tracing is
    still being wired in.
    """
    _ensure_on_path()
    from build_bridge import hb
    hlfir = compile_to_hlfir(source, out_dir, name)
    mod = hb.HLFIRModule()
    if not mod.parse_file(str(hlfir)):
        raise RuntimeError(f"cannot parse {hlfir}")
    if pipeline:
        mod.run_passes(pipeline)
    return mod.dump()
