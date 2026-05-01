"""End-to-end emitter tests — the scaffold consumes
``(FrozenSignature, OriginalInterface, FlattenPlan)`` triples and
writes a ``<entry>_bindings.f90`` file that mirrors what
``hlfir-flatten-structs`` recorded.

Each test spells out a realistic fixture plan and asserts on the
key Fortran shapes the wrapper should contain (or NOT contain, for
the zero-copy guarantees).  No numerical compile-and-run here —
those live in a separate integration test once the bridge side is
wired.
"""
from __future__ import annotations

from pathlib import Path

from dace.frontend.hlfir.bindings import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _two_real_array_struct(tmp_path: Path) -> str:
    """``type(t_fields)`` with two plain ``real(c_double)`` members —
    everything aliases."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(fortran_name="a",
                      sdfg_name="fld_a",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout",
                      from_struct_member="fld%a",
                      layout="same"),
            FrozenArg(fortran_name="b",
                      sdfg_name="fld_b",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout",
                      from_struct_member="fld%b",
                      layout="same"),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name="fld", fortran_type="type(t_fields)", rank=0, intent="inout", struct_type="t_fields"),
              OriginalArg(name="n", fortran_type="integer(c_int)", rank=0,
                          intent="in"), OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in")),
        used_modules={"mo_fields": ("t_fields", )},
    )
    plan = FlattenPlan(entries=(
        FlattenEntry(outer_expr="fld%a",
                     outer_type="real(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("fld_a", ),
                                          read_exprs=("fld%a($i1, $i2)", ),
                                          rank=2,
                                          shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
                                          aliasable=True)),
        FlattenEntry(outer_expr="fld%b",
                     outer_type="real(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("fld_b", ),
                                          read_exprs=("fld%b($i1, $i2)", ),
                                          rank=2,
                                          shape_exprs=("size(fld%b, dim=1)", "size(fld%b, dim=2)"),
                                          aliasable=True)),
    ))
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def _complex_split_struct(tmp_path: Path) -> str:
    """``st%z`` complex member + ``st%u`` plain real."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(fortran_name="z_re",
                      sdfg_name="st_z_re",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
            FrozenArg(fortran_name="z_im",
                      sdfg_name="st_z_im",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
            FrozenArg(fortran_name="u",
                      sdfg_name="st_u",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name="st", fortran_type="type(t_state)", rank=0, intent="inout", struct_type="t_state"),
              OriginalArg(name="n", fortran_type="integer(c_int)", rank=0,
                          intent="in"), OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in")),
        used_modules={"mo_state": ("t_state", )},
    )
    plan = FlattenPlan(entries=(
        FlattenEntry(outer_expr="st%z",
                     outer_type="complex(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("st_z_re", "st_z_im"),
                                          read_exprs=("real(st%z($i1,$i2), kind=c_double)", "aimag(st%z($i1,$i2))"),
                                          write_expr="cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)",
                                          rank=2,
                                          shape_exprs=("size(st%z, dim=1)", "size(st%z, dim=2)"),
                                          aliasable=False)),
        FlattenEntry(outer_expr="st%u",
                     outer_type="real(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("st_u", ),
                                          read_exprs=("st%u($i1, $i2)", ),
                                          rank=2,
                                          shape_exprs=("size(st%u, dim=1)", "size(st%u, dim=2)"),
                                          aliasable=True)),
    ))
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def _nested_struct(tmp_path: Path) -> str:
    """Two-level nested struct: ``st%a%v`` + ``st%b%v``, both aliased
    with full ``%``-paths in ``c_loc(...)``."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(fortran_name="a_v",
                      sdfg_name="st_a_v",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
            FrozenArg(fortran_name="b_v",
                      sdfg_name="st_b_v",
                      kind="array",
                      dtype="float64",
                      rank=2,
                      shape=("n", "m"),
                      intent="inout"),
        ),
        free_symbols=("m", "n"),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name="st", fortran_type="type(t_outer)", rank=0, intent="inout", struct_type="t_outer"),
              OriginalArg(name="n", fortran_type="integer(c_int)", rank=0,
                          intent="in"), OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in")),
        used_modules={"mo_types": ("t_outer", )},
    )
    plan = FlattenPlan(entries=(
        FlattenEntry(outer_expr="st%a%v",
                     outer_type="real(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("st_a_v", ),
                                          read_exprs=("st%a%v($i1, $i2)", ),
                                          rank=2,
                                          shape_exprs=("size(st%a%v, dim=1)", "size(st%a%v, dim=2)"),
                                          aliasable=True)),
        FlattenEntry(outer_expr="st%b%v",
                     outer_type="real(c_double)",
                     writeback_intent="inout",
                     recipe=FlattenRecipe(flat_names=("st_b_v", ),
                                          read_exprs=("st%b%v($i1, $i2)", ),
                                          rank=2,
                                          shape_exprs=("size(st%b%v, dim=1)", "size(st%b%v, dim=2)"),
                                          aliasable=True)),
    ))
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


# --------------------------------------------------------------------------
# Two-real-array struct — zero-copy guarantee
# --------------------------------------------------------------------------


def test_two_real_array_struct_all_aliased(tmp_path: Path):
    """User's invariant: when every struct member has a matching
    layout, the generated wrapper must NOT allocate scratch or
    emit copy loops — pointer aliasing only.
    """
    src = _two_real_array_struct(tmp_path)
    # No deep-copy artefacts.
    assert "allocate(" not in src
    assert "deallocate(" not in src
    assert "do i1 =" not in src and "do i2 =" not in src
    # Two c_f_pointer calls, one per member.
    assert src.count("call c_f_pointer(c_loc(fld%a)") == 1
    assert src.count("call c_f_pointer(c_loc(fld%b)") == 1
    # Flat pointer declarations.
    assert "real(c_double), pointer :: fld_a(:, :)" in src
    assert "real(c_double), pointer :: fld_b(:, :)" in src


def test_two_real_array_struct_module_boilerplate(tmp_path: Path):
    """Common boilerplate — bind(c), handle, finalize — still emitted."""
    src = _two_real_array_struct(tmp_path)
    assert "module kernel_dace_bindings" in src
    assert "use mo_fields, only: t_fields" in src
    assert "bind(c, name='__program_kernel')" in src
    assert "subroutine kernel_dace_finalize()" in src


# --------------------------------------------------------------------------
# Complex-split — alloc + do-loop + dealloc
# --------------------------------------------------------------------------


def test_complex_split_emits_copy_in_loop(tmp_path: Path):
    src = _complex_split_struct(tmp_path)
    assert "allocate(st_z_re(size(st%z, dim=1), size(st%z, dim=2)))" in src
    assert "allocate(st_z_im(size(st%z, dim=1), size(st%z, dim=2)))" in src
    assert "st_z_re(i1, i2) = real(st%z(i1,i2), kind=c_double)" in src
    assert "st_z_im(i1, i2) = aimag(st%z(i1,i2))" in src


def test_complex_split_emits_copy_out_loop(tmp_path: Path):
    src = _complex_split_struct(tmp_path)
    assert "st%z(i1, i2) = cmplx(st_z_re(i1,i2), st_z_im(i1,i2), kind=c_double)" in src
    assert "deallocate(st_z_re)" in src
    assert "deallocate(st_z_im)" in src


def test_complex_split_still_aliases_plain_member(tmp_path: Path):
    """The ``st%u`` member (plain real) must alias, not copy."""
    src = _complex_split_struct(tmp_path)
    assert "call c_f_pointer(c_loc(st%u)" in src
    # No second allocate for st_u — it's a pointer, not scratch.
    assert "allocate(st_u" not in src


# --------------------------------------------------------------------------
# Nested struct — full %-path in c_loc
# --------------------------------------------------------------------------


def test_nested_struct_uses_full_path_in_c_loc(tmp_path: Path):
    """``st%a%v`` is more than one-level nesting; the Fortran
    compiler handles the ``%`` chain directly."""
    src = _nested_struct(tmp_path)
    assert "call c_f_pointer(c_loc(st%a%v)" in src
    assert "call c_f_pointer(c_loc(st%b%v)" in src


def test_nested_struct_no_copy_overhead(tmp_path: Path):
    """Nested + aliasable ⇒ still zero-copy."""
    src = _nested_struct(tmp_path)
    assert "allocate(" not in src
    assert "do i1 =" not in src


# --------------------------------------------------------------------------
# LOGICAL → logical(c_bool) bridge
# --------------------------------------------------------------------------


def _logical_array_kernel(tmp_path: Path) -> str:
    """A kernel taking a top-level ``LOGICAL`` array dummy.  After the
    LOGICAL-to-bool migration the SDFG sees this as ``np.bool_`` (1 byte)
    while the caller-visible Fortran type stays default ``LOGICAL`` (4
    bytes).  The wrapper has to bridge the two with a
    ``logical(c_bool)`` scratch buffer + an intrinsic-cast copy on
    entry / exit."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(FrozenArg(fortran_name="mask",
                        sdfg_name="mask",
                        kind="array",
                        dtype="bool",
                        rank=1,
                        shape=("n", ),
                        intent="inout",
                        from_struct_member="",
                        layout="same"),
              FrozenArg(fortran_name="n",
                        sdfg_name="n",
                        kind="symbol",
                        dtype="int32",
                        rank=0,
                        shape=(),
                        intent="in",
                        from_struct_member="",
                        layout="same")),
        free_symbols=("n", ),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name="mask", fortran_type="logical", rank=1, shape=("n", ),
                          intent="inout"), OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in")),
    )
    plan = FlattenPlan(entries=())
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def test_logical_array_emits_cbool_scratch(tmp_path: Path):
    """The wrapper declares a ``logical(c_bool), allocatable, target``
    scratch buffer for any LOGICAL outer-dummy whose SDFG dtype is
    ``bool`` and whose outer Fortran type isn't already
    ``logical(c_bool)``."""
    src = _logical_array_kernel(tmp_path)
    assert "logical(c_bool), allocatable, target :: mask_cbool(:)" in src


def test_logical_array_emits_intrinsic_cast_on_entry(tmp_path: Path):
    """Wrapper body copies the outer dummy into the scratch via the
    Fortran intrinsic LOGICAL-kind-conversion (a whole-array assign)."""
    src = _logical_array_kernel(tmp_path)
    assert "allocate(mask_cbool(size(mask, dim=1)))" in src
    assert "mask_cbool = mask" in src


def test_logical_array_passes_scratch_to_sdfg(tmp_path: Path):
    """The SDFG-call argument list uses the scratch name, not the outer
    dummy — passing the outer would corrupt every other element."""
    src = _logical_array_kernel(tmp_path)
    # The call line should reference ``mask_cbool``, not bare ``mask``.
    call_block = src[src.index("call dace_program_kernel"):]
    assert "mask_cbool" in call_block.splitlines()[1] or any("mask_cbool" in l for l in call_block.splitlines()[:6])


def test_logical_array_emits_intrinsic_cast_on_exit(tmp_path: Path):
    """For ``intent(inout)`` the wrapper copies the scratch back into
    the outer dummy after the SDFG call, then deallocates."""
    src = _logical_array_kernel(tmp_path)
    assert "mask = mask_cbool" in src
    assert "deallocate(mask_cbool)" in src


# --------------------------------------------------------------------------
# Per-kind LOGICAL(N) bridge coverage
# --------------------------------------------------------------------------
#
# Default ``logical`` is ``LOGICAL(KIND=4)`` (4 bytes), but Fortran
# permits explicit kinds 1, 2, 4, 8.  The SDFG-internal storage is
# always 1-byte ``bool`` (the C ABI of ``logical(c_bool)``); the
# wrapper bridges by allocating a ``logical(c_bool)`` scratch and
# letting Fortran's intrinsic LOGICAL-kind-conversion handle the
# bit fiddling.  Only the explicit ``logical(c_bool)`` outer type
# needs no bridge — it already matches.
#
# These tests parametrise the outer Fortran type per kind to confirm
# the bridge fires for every non-c_bool flavour and is suppressed for
# c_bool.


def _logical_kernel_with_outer_type(tmp_path: Path, fortran_outer_type: str, arr_name: str = "flag") -> str:
    """Same shape as ``_logical_array_kernel`` but with the outer
    Fortran type configurable so the test can probe each LOGICAL kind."""
    frozen = FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(FrozenArg(fortran_name=arr_name,
                        sdfg_name=arr_name,
                        kind="array",
                        dtype="bool",
                        rank=1,
                        shape=("n", ),
                        intent="inout",
                        from_struct_member="",
                        layout="same"),
              FrozenArg(fortran_name="n",
                        sdfg_name="n",
                        kind="symbol",
                        dtype="int32",
                        rank=0,
                        shape=(),
                        intent="in",
                        from_struct_member="",
                        layout="same")),
        free_symbols=("n", ),
    )
    iface = OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name=arr_name, fortran_type=fortran_outer_type, rank=1, shape=("n", ),
                          intent="inout"), OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in")),
    )
    plan = FlattenPlan(entries=())
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(frozen, iface, plan, str(out))
    return out.read_text()


def test_logical_kind_1_emits_bridge(tmp_path: Path):
    """``LOGICAL(KIND=1)`` — even though the byte width matches
    ``c_bool``, Fortran treats them as distinct kinds; the wrapper
    plays it safe with an explicit intrinsic-cast bridge."""
    src = _logical_kernel_with_outer_type(tmp_path, "logical(1)")
    assert "logical(c_bool), allocatable, target :: flag_cbool(:)" in src
    assert "flag_cbool = flag" in src
    assert "flag = flag_cbool" in src
    assert "deallocate(flag_cbool)" in src


def test_logical_kind_2_emits_bridge(tmp_path: Path):
    """``LOGICAL(KIND=2)`` — 2-byte storage, must bridge."""
    src = _logical_kernel_with_outer_type(tmp_path, "logical(2)")
    assert "logical(c_bool), allocatable, target :: flag_cbool(:)" in src
    assert "flag_cbool = flag" in src
    assert "flag = flag_cbool" in src


def test_logical_kind_4_emits_bridge(tmp_path: Path):
    """``LOGICAL(KIND=4)`` — the default kind, 4 bytes.  Most ICON
    code lands here; the bridge is the hot path."""
    src = _logical_kernel_with_outer_type(tmp_path, "logical(4)")
    assert "logical(c_bool), allocatable, target :: flag_cbool(:)" in src
    assert "flag_cbool = flag" in src
    assert "flag = flag_cbool" in src


def test_logical_kind_8_emits_bridge(tmp_path: Path):
    """``LOGICAL(KIND=8)`` — 8-byte storage, must bridge."""
    src = _logical_kernel_with_outer_type(tmp_path, "logical(8)")
    assert "logical(c_bool), allocatable, target :: flag_cbool(:)" in src
    assert "flag_cbool = flag" in src
    assert "flag = flag_cbool" in src


def test_logical_cbool_passes_through_no_bridge(tmp_path: Path):
    """``logical(c_bool)`` already matches the SDFG's bool layout — no
    scratch buffer, no Fortran-intrinsic cast.  The outer dummy goes
    straight through to the SDFG."""
    src = _logical_kernel_with_outer_type(tmp_path, "logical(c_bool)")
    # No scratch declaration.
    assert "_cbool" not in src.replace("logical(c_bool)", "")
    # No copy-in / copy-out / deallocate.
    assert "flag_cbool" not in src
