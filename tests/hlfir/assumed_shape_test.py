"""Fortran assumed-shape re-basing through inlining.

Edge case from https://fortran-lang.discourse.group/t/6923 (Federico's
explanation): an assumed-shape dummy argument always has ``lbound=1``
inside the callee regardless of the caller's custom lower bound.
When we inline such a callee into a caller whose actual argument has
``(-2:2)``-style bounds, the callee's body must still treat index 1
as "first storage element" — not as "first caller-bound element".

Our HLFIR bridge relies on Flang's own lowering for this: the caller
builds a ``fir.shape_shift %lb, %ext`` for the custom-bounded array,
and Flang's call site wraps the box such that the callee's
``hlfir.declare %arg0 dummy_scope %0`` (no shape operand) re-associates
to ``lbound=1``.  When ``hlfir-inline-all`` splices the callee body
into the caller, each IR construct retains its own declare — the
inlined declare still carries the 1-based view of the box, distinct
from the caller's ``custom_array`` declare with its ``-2:2`` bounds.

These tests guard that behaviour end to end:

- ``test_inline_rebase_storage`` — build the SDFG through
  ``SDFGBuilder.from_files`` (multi-file → inline-all path), run it,
  and assert that ``arr(1) = 999`` inside the callee lands in the
  caller's first storage slot (the ``custom_array(-2)`` element).

- ``test_sdfg_matches_gfortran_reference`` — same source compiled with
  ``gfortran`` via ``f2py``; two outputs must match bit-exactly.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dace" / "frontend" / "hlfir"))
from hlfir_to_sdfg import SDFGBuilder  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_FLANG = "flang-new-21"

# Two files: ``callee`` takes an assumed-shape dummy; ``driver`` owns a
# custom-bounded local and calls the callee on it.  Splitting across
# files exercises the multi-file driver (parse_files → inline-all) the
# way ICON's cross-module kernels will.

_CALLEE_SRC = """
subroutine callee(arr)
  implicit none
  integer, intent(inout) :: arr(:)
  ! arr has lbound=1 inside. arr(1) lands in the first storage slot,
  ! which in the caller's x(-2:2) view is x(-2).
  arr(1) = 999
end subroutine
"""

_DRIVER_SRC = """
subroutine driver(x)
  implicit none
  integer, intent(inout) :: x(-2:2)
  interface
    subroutine callee(arr)
      integer, intent(inout) :: arr(:)
    end subroutine
  end interface
  ! The callee's arr(1) must land in x(-2) per assumed-shape semantics.
  call callee(x)
end subroutine
"""


def _hlfir(src: str, path: Path) -> Path:
    f90 = path.with_suffix(".f90")
    f90.write_text(src)
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(f90), "-o", str(path)])
    return path


def _f2py_build(srcs: list[str], out_dir: Path, mod_name: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{mod_name}.f90"
    combined.write_text("\n".join(srcs))
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(combined), "-m", mod_name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def test_inlined_hlfir_has_assumed_shape_alias_declare(tmp_path: Path):
    """Structural guard: after ``hlfir-inline-all + symbol-dce`` the
    IR contains exactly the alias declare pattern that drives the
    assumed-shape re-basing problem — a second ``hlfir.declare`` with
    no shape operand whose memref is a ``fir.convert`` of the caller's
    ``fir.shape_shift``-bounded declare.  This test asserts the shape
    of the IR that subsequent frontend work needs to handle; it does
    NOT yet require the SDFG to be correct."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dace" / "frontend" / "hlfir"))
    from build_bridge import hb  # noqa: E402

    callee_hlfir = _hlfir(_CALLEE_SRC, tmp_path / "callee.hlfir")
    driver_hlfir = _hlfir(_DRIVER_SRC, tmp_path / "driver.hlfir")

    m = hb.HLFIRModule()
    assert m.parse_files([str(driver_hlfir), str(callee_hlfir)])
    m.set_entry_symbol("_QPdriver")
    m.run_passes("hlfir-inline-all,symbol-dce")
    dump = m.dump()

    # Outer declare: shape_shift with lbound=-2 on the caller's x.
    assert "shape_shift" in dump and "-2" in dump
    # Inlined alias declare: its uniq_name is _QFcalleeEarr and its
    # memref is a fir.convert (extent erasure).
    assert '_QFcalleeEarr' in dump, \
        "expected the inlined callee's alias declare to survive inline+dce"
    # The fir.convert box<array<5xi32>> → box<array<?xi32>> is the
    # assumed-shape-alias signature we'll need to fold.
    assert "fir.convert" in dump


def test_inline_rebase_storage(tmp_path: Path):
    """After inlining, ``arr(1) = 999`` in the callee must land in
    ``x(-2)`` (the first storage slot)."""
    callee_hlfir = _hlfir(_CALLEE_SRC, tmp_path / "callee.hlfir")
    driver_hlfir = _hlfir(_DRIVER_SRC, tmp_path / "driver.hlfir")

    b = SDFGBuilder.from_files([str(driver_hlfir), str(callee_hlfir)], entry="_QPdriver")
    sdfg = b.build()

    x = np.asfortranarray(np.array([10, 20, 30, 40, 50], dtype=np.int32))
    sdfg(x=x)
    assert x.tolist() == [999, 20, 30, 40, 50], x.tolist()


def test_sdfg_matches_gfortran_reference(tmp_path: Path):
    """The SDFG built from the two-file setup must match a single
    f2py-compiled combined reference bit-for-bit."""
    callee_hlfir = _hlfir(_CALLEE_SRC, tmp_path / "callee.hlfir")
    driver_hlfir = _hlfir(_DRIVER_SRC, tmp_path / "driver.hlfir")

    ref = _f2py_build([_CALLEE_SRC, _DRIVER_SRC], tmp_path / "ref", "asref")
    b = SDFGBuilder.from_files([str(driver_hlfir), str(callee_hlfir)], entry="_QPdriver")
    sdfg = b.build()

    rng = np.random.default_rng(0)
    x_init = rng.integers(0, 1000, size=5, dtype=np.int32)

    x_ref = np.asfortranarray(x_init.copy())
    ref.driver(x_ref)

    x_sdfg = np.asfortranarray(x_init.copy())
    sdfg(x=x_sdfg)

    np.testing.assert_array_equal(x_sdfg, x_ref)
