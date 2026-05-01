"""Sympy-collision rename pass — direct coverage.

The bridge renames Fortran identifiers that match a sympy module-level
``LazyFunction`` attribute (``test``, ``doctest``) to ``program_<name>``
at the SDFG layer.  Without the rename, parsing any interstate-edge
expression that mentions the array (``test[i, j, k]``, etc.) crashes
sympy with ``SympifyError: cannot sympify object of type LazyFunction``.

These tests exercise the rename pass directly so a regression on the
collision set surfaces here, not via the wider integration tests.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_reserved_name_test_as_local_array(tmp_path: Path):
    """Local array named ``test`` — the indexed-load case that
    triggered the original ``LazyFunction`` SympifyError."""
    src = """
subroutine main(d)
  double precision d(4)
  integer test(3, 3, 3)
  integer indices(3, 3, 3)
  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_reserved_name_doctest_as_local_array(tmp_path: Path):
    """Same as above but for ``doctest`` (the other sympy
    ``LazyFunction`` collision).  Same shape of failure when the
    rename pass misses an entry; the rename map must cover the full
    reserved set."""
    src = """
subroutine main(d)
  double precision d(4)
  integer doctest(3, 3)
  doctest(1, 1) = 2
  doctest(2, 3) = 3
  d(doctest(1, 1)) = 5.5
  d(doctest(2, 3)) = 7.5
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 7.5, 42])


def test_reserved_name_is_renamed_in_sdfg(tmp_path: Path):
    """Sanity-check the rename: the SDFG should advertise the
    ``program_test`` array and the builder should record the
    ``test → program_test`` mapping for the binding emitter."""
    src = """
subroutine main(d)
  double precision d(2)
  integer test(2)
  test(1) = 1
  test(2) = 2
  d(1) = test(1)
  d(2) = test(2)
end subroutine main
"""
    from _util import build_sdfg as _bs
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = _bs(src, sdfg_dir, name="main")
    sdfg = builder.build()

    assert "program_test" in sdfg.arrays
    assert "test" not in sdfg.arrays
    assert builder.dace_name_map.get("test") == "program_test"


def test_reserved_name_test_as_dummy_argument(tmp_path: Path):
    """``test`` as a subroutine dummy argument — exercises the SDFG
    signature side and the FrozenSignature build that the binding
    emitter consumes.  The SDFG-level kwarg uses the renamed form
    (``program_test``) since that's what DaCe registers; the
    FrozenSignature maps it back to the original ``test`` for the
    binding wrapper."""
    src = """
subroutine main(d, test)
  double precision, intent(inout) :: d(2)
  integer, intent(in) :: test(2)
  d(1) = test(1)
  d(2) = test(2)
end subroutine main
"""
    from _util import build_sdfg as _bs
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = _bs(src, sdfg_dir, name="main")
    sdfg = builder.build()

    # SDFG-internal name is renamed; user-facing Fortran name preserved
    # on the FrozenSignature for the binding emitter.
    assert "program_test" in sdfg.arrays
    fs = sdfg._frozen_signature
    test_arg = next((a for a in fs.args if a.fortran_name == "test"), None)
    assert test_arg is not None, f"no 'test' arg in FrozenSignature: {[a.fortran_name for a in fs.args]}"
    assert test_arg.sdfg_name == "program_test"

    # Numerical: pass the array via the SDFG-internal kwarg and verify
    # the data flows through the renamed identifier end-to-end.
    d = np.zeros(2, dtype=np.float64)
    test_arr = np.array([7, 9], dtype=np.int32)
    sdfg(d=d, program_test=test_arr)
    assert np.allclose(d, [7.0, 9.0])


def test_reserved_name_binding_wrapper_emits(tmp_path: Path):
    """The binding emitter should consume the renamed FrozenSignature
    without crashing and produce a wrapper file.  This is the surface
    that downstream Fortran callers see — without it the rename pass
    has fixed the SDFG side but left the binding side broken."""
    from dace.frontend.hlfir.bindings import emit_bindings
    from dace.frontend.hlfir.bindings.frozen_signature import FrozenArg, FrozenSignature
    from dace.frontend.hlfir.bindings.fortran_interface import OriginalArg, OriginalInterface
    from dace.frontend.hlfir.bindings.flatten_plan import FlattenPlan

    fs = FrozenSignature(
        entry="main",
        mangled="_QPmain",
        args=(
            FrozenArg(fortran_name="d",
                      sdfg_name="d",
                      kind="array",
                      dtype="float64",
                      rank=1,
                      shape=("2", ),
                      intent="inout"),
            FrozenArg(fortran_name="test",
                      sdfg_name="program_test",
                      kind="array",
                      dtype="int32",
                      rank=1,
                      shape=("2", ),
                      intent="in"),
        ),
        free_symbols=(),
    )
    iface = OriginalInterface(
        entry="main",
        args=(
            OriginalArg(name="d", fortran_type="real(c_double)", rank=1, intent="inout"),
            OriginalArg(name="test", fortran_type="integer(c_int)", rank=1, intent="in"),
        ),
    )
    plan = FlattenPlan(entries=())
    out = tmp_path / "main_bindings.f90"
    emit_bindings(fs, iface, plan, str(out))
    text = out.read_text()
    # Wrapper exposes the original Fortran identifier ``test`` to the
    # caller; the renamed ``program_test`` is the bind(C) parameter
    # passed across to the SDFG.
    assert "test" in text
    assert "program_test" in text
