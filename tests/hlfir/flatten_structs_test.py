"""Verify the ``hlfir-flatten-structs`` pass rewrites derived-type data into
flat per-member companions (uniform case) or into a single ELLPACK-style
combined array (jagged case) before SDFG generation sees it."""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang, run_passes_dump

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC = (_HERE / "complex_struct.f90").read_text()
_VELOCITY_SRC = (_HERE / "velocity_struct.f90").read_text()
_JAGGED_SRC = (_HERE / "jagged_struct.f90").read_text()
_FLATTEN_ONLY = "hlfir-flatten-structs"


def _names(builder):
    return set(builder.arrays) | set(builder.scalars)


def test_flatten_structs_splits_members(tmp_path):
    """Array-of-struct (``type(complex_t) :: z(8)``): the pass must synthesise
    the per-member flat companions ``z_re`` and ``z_im``  --  that's what the
    downstream SDFG will build against."""
    b = build_sdfg(_SRC, tmp_path, name="complex_struct")

    names = _names(b)
    assert any(n.endswith("_re") for n in names), (f"missing re companion array in {sorted(names)}")
    assert any(n.endswith("_im") for n in names), (f"missing im companion array in {sorted(names)}")


# ----------------------------------------------------------------------------
# Struct-typed dummy argument: four uniformly-shaped 2-D array members
# become four individual 2-D array arguments; function is renamed.
# ----------------------------------------------------------------------------


def test_velocity_struct_arg_flattens_to_four_args(tmp_path):
    ir = run_passes_dump(_VELOCITY_SRC, tmp_path, name="velocity", pipeline=_FLATTEN_ONLY)

    # Function renamed.
    assert "_soa" in ir, f"function should be renamed to *_soa:\n{ir[:600]}"

    # All four member companions show up as new hlfir.declare uniq_names.
    for mem in ("_u", "_v", "_w", "_p"):
        needle = f"Est{mem}"
        assert needle in ir, (f"expected declare with uniq_name ending in {needle!r}; IR excerpt:"
                              f"\n{ir[:800]}")

    # The struct type itself should no longer appear on the function argument.
    assert "!fir.type<" not in ir.split("func.func")[1].splitlines()[0], (
        f"function signature should no longer reference a struct type:\n{ir[:400]}")


# ----------------------------------------------------------------------------
# Jagged struct: four 1-D array members of different extents pack into one
# 2-D companion of shape [numMembers x max(extents)] (ELLPACK layout).
# ----------------------------------------------------------------------------


def test_jagged_struct_arg_packs_into_2d(tmp_path):
    ir = run_passes_dump(_JAGGED_SRC, tmp_path, name="jagged", pipeline=_FLATTEN_ONLY)

    assert "_soa" in ir, f"function should be renamed to *_soa:\n{ir[:600]}"

    # The combined 2-D array: 4 rows (one per member), 20 cols (= max(10,20,15,5)).
    assert "!fir.array<4x20xf64>" in ir, (f"expected packed 4x20xf64 array in post-pass IR:\n{ir[:1500]}")

    # The four coordinate_of / convert pairs that alias each member into a
    # row of the combined array should be present.
    assert ir.count("fir.coordinate_of") >= 4, (
        f"expected four fir.coordinate_of ops (one per jagged member):\n{ir[:1500]}")
