"""Per-renderer tests for the alias / copy-in / copy-out primitives.

These are the only pieces of binding-code-generation logic, so we
test each shape in isolation: single recipe in, list of Fortran
lines out.
"""
from __future__ import annotations

import pytest

from dace.frontend.hlfir.bindings import FlattenRecipe
from dace.frontend.hlfir.bindings.loop_copy import (
    render_alias_calls,
    render_aos_alloc_pack_in,
    render_aos_alloc_pack_out,
    render_copy_in_loop,
    render_copy_out_loop,
)


def _alias_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("fld_a", ),
        read_exprs=("fld%a($i1, $i2)", ),
        rank=2,
        shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
        aliasable=True,
    )


def _complex_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("st_z_re", "st_z_im"),
        read_exprs=(
            "real(st%z($i1,$i2), kind=c_double)",
            "aimag(st%z($i1,$i2))",
        ),
        write_expr="cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)",
        rank=2,
        shape_exprs=("size(st%z, dim=1)", "size(st%z, dim=2)"),
        aliasable=False,
    )


# --------------------------------------------------------------------------
# render_alias_calls
# --------------------------------------------------------------------------


def test_render_alias_calls_single_flat():
    lines = render_alias_calls(_alias_recipe())
    assert len(lines) == 1
    assert "call c_f_pointer(c_loc(fld%a), fld_a, [" in lines[0]
    assert "size(fld%a, dim=1)" in lines[0]
    assert "size(fld%a, dim=2)" in lines[0]


def test_render_alias_calls_strips_index_placeholders():
    """``c_loc`` must see the base storage path, not the indexed
    expression — strip_index_args handles this."""
    lines = render_alias_calls(_alias_recipe())
    # ``c_loc(fld%a)`` with no ``($i1, $i2)`` suffix.
    assert "c_loc(fld%a)," in lines[0]
    assert "$i" not in "".join(lines)


def test_render_alias_calls_raises_on_non_aliasable():
    with pytest.raises(ValueError, match="non-aliasable"):
        render_alias_calls(_complex_recipe())


def test_render_alias_calls_nested_struct_path():
    recipe = FlattenRecipe(
        flat_names=("st_a_v", ),
        read_exprs=("st%a%v($i1, $i2)", ),
        rank=2,
        shape_exprs=("size(st%a%v, dim=1)", "size(st%a%v, dim=2)"),
        aliasable=True,
    )
    lines = render_alias_calls(recipe)
    assert "c_loc(st%a%v)" in lines[0]


# --------------------------------------------------------------------------
# render_copy_in_loop
# --------------------------------------------------------------------------


def test_render_copy_in_loop_allocates_each_flat():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    assert "allocate(st_z_re(size(st%z, dim=1), size(st%z, dim=2)))" in joined
    assert "allocate(st_z_im(size(st%z, dim=1), size(st%z, dim=2)))" in joined


def test_render_copy_in_loop_nested_do_nest():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    # Two loop heads (one per rank), two ``end do`` markers.
    assert joined.count("do i1 = 1,") == 1
    assert joined.count("do i2 = 1,") == 1
    assert joined.count("end do") == 2


def test_render_copy_in_loop_substitutes_index_names():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    assert "st_z_re(i1, i2) = real(st%z(i1,i2), kind=c_double)" in joined
    assert "st_z_im(i1, i2) = aimag(st%z(i1,i2))" in joined
    # No raw placeholders survive.
    assert "$i1" not in joined and "$i2" not in joined


def test_render_copy_in_loop_raises_on_aliasable():
    with pytest.raises(ValueError, match="aliasable"):
        render_copy_in_loop(_alias_recipe())


# --------------------------------------------------------------------------
# render_copy_out_loop
# --------------------------------------------------------------------------


def test_render_copy_out_loop_shape():
    lines = render_copy_out_loop(_complex_recipe(), outer_expr="st%z")
    joined = "\n".join(lines)
    # Reconstructs outer element, then deallocates each flat.
    assert "st%z(i1, i2) = cmplx(st_z_re(i1,i2), st_z_im(i1,i2), kind=c_double)" in joined
    assert "deallocate(st_z_re)" in joined
    assert "deallocate(st_z_im)" in joined


def test_render_copy_out_loop_raises_on_empty_write_expr():
    r = FlattenRecipe(
        flat_names=("fld_a", ),
        read_exprs=("fld%a($i1, $i2)", ),
        rank=2,
        shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
        aliasable=False,  # non-aliasable but no writeback
    )
    with pytest.raises(ValueError, match="empty write_expr"):
        render_copy_out_loop(r, outer_expr="fld%a")


# --------------------------------------------------------------------------
# render_aos_alloc_pack_in / render_aos_alloc_pack_out (Phase 5c-B boundary)
# --------------------------------------------------------------------------


def _aos_alloc_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("a_w", ),
        read_exprs=("a($i1)%w($i2)", ),
        rank=2,
        shape_exprs=("size(a, dim=1)", "cap_a_w"),
        aliasable=False,
        aos_alloc=True,
        cap_symbol="cap_a_w",
    )


def test_render_aos_alloc_pack_in_emits_cap_max_loop():
    lines = render_aos_alloc_pack_in(_aos_alloc_recipe(), outer_expr="a")
    joined = "\n".join(lines)
    # Cap is computed by max-ing per-instance ``size`` — guarded by
    # ``allocated()`` so unallocated rows don't poison the max.
    assert "cap_a_w = 0" in joined
    assert "if (allocated(a(i1)%w))" in joined
    assert "if (size(a(i1)%w) > cap_a_w) cap_a_w = size(a(i1)%w)" in joined
    # Empty-batch sentinel keeps the buffer non-degenerate.
    assert "if (cap_a_w == 0) cap_a_w = 1" in joined
    # Buffer allocate at (N, cap) shape, zero-init, then per-row copy.
    assert "allocate(a_w(size(a, dim=1), cap_a_w))" in joined
    assert "a_w = 0" in joined
    assert "a_w(i1, 1:size(a(i1)%w)) = a(i1)%w" in joined


def test_render_aos_alloc_pack_out_copies_back_live_region():
    lines = render_aos_alloc_pack_out(_aos_alloc_recipe(), outer_expr="a")
    joined = "\n".join(lines)
    # Per-row copy-back, guarded by ``allocated()`` (skips unalloc'd
    # entries — bindings policy is to leave their buffer rows zeroed).
    assert "if (allocated(a(i1)%w))" in joined
    assert "a(i1)%w = a_w(i1, 1:size(a(i1)%w))" in joined
    # Final deallocate releases the scratch.
    assert "deallocate(a_w)" in joined


def test_render_aos_alloc_pack_in_raises_on_non_aos_alloc():
    plain = FlattenRecipe(
        flat_names=("a_w", ),
        read_exprs=("a($i1)%w($i2)", ),
        rank=2,
        shape_exprs=("size(a, dim=1)", "cap_a_w"),
        aliasable=False,
    )
    with pytest.raises(ValueError, match="non-aos_alloc"):
        render_aos_alloc_pack_in(plain, outer_expr="a")
    with pytest.raises(ValueError, match="non-aos_alloc"):
        render_aos_alloc_pack_out(plain, outer_expr="a")


def test_kind_convert_recipe_rendering():
    """Demonstrate the kind-convert shape the user wants:
    ``real(kind=4)`` outer → ``real(kind=8)`` SDFG flat.  Uses the
    same machinery — no dedicated strategy needed."""
    recipe = FlattenRecipe(
        flat_names=("st_x_d", ),
        read_exprs=("real(st%x($i1), kind=c_double)", ),
        write_expr="real(st_x_d($i1), kind=c_float)",
        rank=1,
        shape_exprs=("size(st%x, dim=1)", ),
        aliasable=False,
        scratch_dtype="float64",
    )
    in_lines = render_copy_in_loop(recipe)
    out_lines = render_copy_out_loop(recipe, outer_expr="st%x")
    assert "st_x_d(i1) = real(st%x(i1), kind=c_double)" in "\n".join(in_lines)
    assert "st%x(i1) = real(st_x_d(i1), kind=c_float)" in "\n".join(out_lines)
