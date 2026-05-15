"""``FlattenPlan`` data model  --  JSON round-trip, index substitution,
strip helpers, recipe equality."""

from pathlib import Path

import pytest

from dace.frontend.hlfir.bindings import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)


def _real_alias_recipe() -> FlattenRecipe:
    """``fld%a`` -> one flat ``fld_a``, aliased."""
    return FlattenRecipe(
        flat_names=("fld_a", ),
        read_exprs=("fld%a($i1, $i2)", ),
        rank=2,
        shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
        aliasable=True,
    )


def _complex_split_recipe() -> FlattenRecipe:
    """``st%z`` -> two flats ``st_z_re`` / ``st_z_im`` via complex split."""
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


def test_json_roundtrip_single_recipe(tmp_path: Path):
    plan = FlattenPlan(entries=(FlattenEntry(
        outer_expr="fld%a",
        outer_type="real(c_double)",
        writeback_intent="",
        recipe=_real_alias_recipe(),
    ), ))
    p = tmp_path / "plan.json"
    plan.to_json(str(p))
    assert FlattenPlan.from_json(str(p)) == plan


def test_json_roundtrip_mixed_plan(tmp_path: Path):
    """Alias + complex-split + nested-struct recipes round-trip together."""
    nested_alias = FlattenRecipe(
        flat_names=("st_a_v", ),
        read_exprs=("st%a%v($i1, $i2)", ),
        rank=2,
        shape_exprs=("size(st%a%v, dim=1)", "size(st%a%v, dim=2)"),
        aliasable=True,
    )
    plan = FlattenPlan(entries=(
        FlattenEntry(outer_expr="st%a%v", outer_type="real(c_double)", writeback_intent="", recipe=nested_alias),
        FlattenEntry(
            outer_expr="st%z", outer_type="complex(c_double)", writeback_intent="inout",
            recipe=_complex_split_recipe()),
    ))
    p = tmp_path / "plan.json"
    plan.to_json(str(p))
    assert FlattenPlan.from_json(str(p)) == plan


def test_substitute_indices_replaces_placeholders():
    out = substitute_indices("st%a($i1, $i2)", ("i1", "i2"))
    assert out == "st%a(i1, i2)"


def test_substitute_indices_with_rename():
    """Index names are a tuple  --  the emitter could use ``k1``, ``k2``
    for some reason."""
    out = substitute_indices("real(st%z($i1,$i2), kind=c_double)", ("k1", "k2"))
    assert out == "real(st%z(k1,k2), kind=c_double)"


def test_substitute_indices_raises_on_out_of_range():
    with pytest.raises(IndexError):
        substitute_indices("a($i1,$i2,$i3)", ("i1", "i2"))


def test_strip_index_args_plain():
    assert strip_index_args("fld%a($i1, $i2)") == "fld%a"


def test_strip_index_args_nested_path():
    assert strip_index_args("st%a%b%c($i1, $i2, $i3)") == "st%a%b%c"


def test_strip_index_args_non_indexed_passthrough():
    """A recipe read_expr that's already a plain path shouldn't be
    mangled (used for symbols or scalars that get aliased as-is)."""
    assert strip_index_args("fld%a") == "fld%a"
    assert strip_index_args("real(st%z($i1,$i2), kind=c_double)") == \
           "real(st%z($i1,$i2), kind=c_double)"


def test_empty_plan_roundtrip(tmp_path: Path):
    """Kernels with no struct dummies produce an empty plan."""
    plan = FlattenPlan()
    p = tmp_path / "plan.json"
    plan.to_json(str(p))
    assert FlattenPlan.from_json(str(p)) == plan
