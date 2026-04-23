"""``emit_bindings`` end-to-end template-assembly test.

This covers the scaffold: a real ``FrozenSignature`` + ``OriginalInterface``
produces a well-formed Fortran module that:
- declares the three C entry points with ``bind(c)``,
- exposes the ``<entry>_dace`` wrapper subroutine preserving the
  caller's signature,
- emits ``c_f_pointer`` aliases for array args whose layout matches,
- computes each SDFG free symbol via ``size(...)``,
- guards the init with a ref-counted ``init_count``,
- invokes ``__program_<entry>`` via the module-save handle.

Concrete numerical end-to-end (compile + run) lives in a separate
integration test that depends on flang + gfortran + DaCe codegen.
"""
from __future__ import annotations

from pathlib import Path

from dace.frontend.hlfir.bindings import (
    FrozenArg,
    FrozenSignature,
    emit_bindings,
)
from dace.frontend.hlfir.bindings.fortran_interface import (
    OriginalArg,
    OriginalInterface,
)


def _demo(tmp_path: Path) -> Path:
    frozen = FrozenSignature(
        entry="compute",
        mangled="_QPcompute",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="a",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="in"),
            FrozenArg(fortran_name="b",
                      sdfg_name="b",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="compute",
        args=(
            OriginalArg(name="a", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="in"),
            OriginalArg(name="b", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="inout"),
        ),
    )
    out = tmp_path / "compute_bindings.f90"
    emit_bindings(frozen, iface, str(out))
    return out


def test_emits_module_with_expected_symbols(tmp_path: Path):
    src = _demo(tmp_path).read_text()

    # Module boilerplate.
    assert "module compute_dace_bindings" in src
    assert "use iso_c_binding" in src
    assert "public :: compute_dace, compute_dace_finalize" in src

    # bind(c) interface block to all three entry points.
    assert "bind(c, name='__dace_init_compute')" in src
    assert "bind(c, name='__program_compute')" in src
    assert "bind(c, name='__dace_exit_compute')" in src

    # Ref-counted init.
    assert "init_count" in src
    assert "c_null_ptr" in src


def test_emits_alias_for_matching_layouts(tmp_path: Path):
    src = _demo(tmp_path).read_text()

    # Same rank + same element type on both outer and frozen sides
    # ⇒ zero-copy c_f_pointer / c_loc alias.
    assert "call c_f_pointer(c_loc(a), a," in src
    assert "call c_f_pointer(c_loc(b), b," in src


def test_emits_symbol_population_from_size(tmp_path: Path):
    src = _demo(tmp_path).read_text()

    # Free symbols n, m → size(outer, dim=?) assignments.
    assert "n = int(size(a, dim=1), c_int)" in src or "n = int(size(b, dim=1), c_int)" in src
    assert "m = int(size(a, dim=2), c_int)" in src or "m = int(size(b, dim=2), c_int)" in src


def test_emits_sdfg_call_with_handle(tmp_path: Path):
    src = _demo(tmp_path).read_text()
    assert "call dace_program_compute(dace_handle," in src


def test_finalize_sub_present(tmp_path: Path):
    src = _demo(tmp_path).read_text()
    assert "subroutine compute_dace_finalize()" in src
    assert "err = dace_exit_compute(dace_handle)" in src
