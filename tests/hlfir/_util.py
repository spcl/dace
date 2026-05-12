"""Helpers for the HLFIR frontend test suite.

The HLFIR frontend takes a pre-lowered ``.hlfir`` file, not raw Fortran
source.  Tests stay readable by writing the Fortran inline and letting
this helper run ``flang-new-21 -fc1 -emit-hlfir`` behind the scenes.  If
no flang binary is found the helpers report it; the calling test module
should skip collection accordingly.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def f2py_compile(src, out_dir: Path, mod_name: str):
    """Build the given Fortran source via gfortran/f2py and return the
    compiled module.  ``src`` may be a file path or an inline string —
    inline sources are written to ``<out_dir>/<mod_name>.f90`` first.

    Skips the calling test (via pytest.skip) when gfortran or meson is
    missing, so test files can call this unconditionally.

    Used by the e2e numerical tests to compare an SDFG's output against
    the same code compiled with gfortran (the reference implementation).
    Saved policy: HLFIR-frontend tests must compare against this kind of
    non-transformed reference — hand-tuned literal expectations are not
    a substitute.
    """
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_text = src if not isinstance(src, Path) else None
    if src_text is not None:
        src_file = out_dir / f"{mod_name}.f90"
        src_file.write_text(src_text)
    else:
        src_file = src
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c",
                           str(src_file), "-m", mod_name, "--quiet"],
                          cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def compile_to_hlfir(source: str, out_dir: Path, name: str = "src", *, preprocess: bool = False) -> Path:
    """Write `source` as <out_dir>/<name>.f90, compile it to HLFIR, return the path.

    When ``preprocess`` is true, run the optional Fortran source
    rewriter (``dace.frontend.hlfir.preprocess.preprocess_fortran``)
    before flang sees the source -- needed for legacy code that uses
    INTEGER flags as IF conditions (``IF (laericeauto)``), which
    flang-new-21 rejects.  Off by default so we don't paper over real
    issues in clean source; opt in per call site.
    """
    assert _FLANG is not None, "flang-new-21 not available"
    src = out_dir / f"{name}.f90"
    if preprocess:
        from dace.frontend.hlfir.preprocess import preprocess_fortran
        src.write_text(preprocess_fortran(source))
    else:
        src.write_text(source)
    hlfir = out_dir / f"{name}.hlfir"
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(src), "-o", str(hlfir)])
    return hlfir


def _per_test_suffix() -> str:
    """Return a readable, test-relevant suffix derived from
    ``PYTEST_CURRENT_TEST`` (e.g. ``_multi_target_reduction_mixed_reductions``).
    Empty when not running under pytest, so notebook / ad-hoc callers
    see unmodified SDFG names.

    Used by ``build_sdfg`` to ensure each test produces a uniquely-named
    SDFG.  Without this, multiple tests in the same file all produce an
    SDFG named e.g. ``main`` — under pytest-xdist they share the same
    ``.so`` filename within a worker, the OS dynamic loader returns a
    cached handle bound to the previous test's compiled symbols, and
    the second test silently runs stale code.

    Form: ``_<file-stem-minus-_test>_<test-name-minus-test_>``.  Both
    halves are sanitised so the resulting name is a valid C++ symbol
    (e.g. parametrised ``test_foo[3]`` becomes ``foo_3_``).
    """
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not raw or "::" not in raw:
        return ""
    nodeid = raw.rsplit(" ", 1)[0]
    file_part, _, test_part = nodeid.partition("::")
    stem = Path(file_part).stem
    if stem.endswith("_test"):
        stem = stem[:-len("_test")]
    if test_part.startswith("test_"):
        test_part = test_part[len("test_"):]
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", f"{stem}_{test_part}").strip("_")
    return f"_{sanitized}" if sanitized else ""


class _TestBuilder:
    """Thin proxy around ``SDFGBuilder`` for tests:

      * Renames the produced SDFG with a per-test hash suffix so two
        tests using ``name='main'`` end up with distinct ``.so`` files —
        a hard requirement for ``pytest-xdist`` parallel runs.
      * Optionally dumps the built SDFG to disk when
        ``__DACE_HLFIR_GEN_TEST_SDFGS`` is set.

    Everything else flows through to the wrapped builder unchanged
    (``.arrays`` / ``.scalars`` / … still work the same way for tests
    that inspect them).
    """

    def __init__(self, inner, name: str, suffix: str, dump_dir: Path | None):
        self._inner = inner
        self._name = name
        self._suffix = suffix
        self._dump_dir = dump_dir

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def build(self):
        sdfg = self._inner.build()
        if self._suffix:
            sdfg.name = f"{sdfg.name}{self._suffix}"
        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._dump_dir / f"{self._name}{self._suffix}.sdfgz"
            sdfg.save(str(out_path), compress=True)
        return sdfg


def build_sdfg(source: str, out_dir: Path, name: str = "src", pipeline=None, entry: str | None = None):
    """Compile inline Fortran to HLFIR and return a configured ``SDFGBuilder``.

    :param source: Fortran source as a string.
    :param out_dir: scratch directory (typically ``tmp_path`` from pytest).
    :param name: base filename for the .f90/.hlfir pair.
    :param pipeline: override the default MLIR pass pipeline.
    :param entry: mangled Flang symbol name of the subroutine the SDFG
                  should represent (e.g. ``_QPapply_delta``).  Needed
                  when the source declares additional public functions
                  in a module — those would otherwise leak their dummy
                  declares into the variable extraction.
    :return: ``SDFGBuilder`` with variables classified and the AST extracted.
    """
    _ensure_on_path()
    from hlfir_to_sdfg import SDFGBuilder, DEFAULT_PIPELINE
    hlfir = compile_to_hlfir(source, out_dir, name)
    builder = SDFGBuilder(str(hlfir), pipeline=(pipeline or DEFAULT_PIPELINE), entry=entry)
    suffix = _per_test_suffix()
    dump = _dump_dir()
    if suffix or dump is not None:
        return _TestBuilder(builder, name, suffix, dump)
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
