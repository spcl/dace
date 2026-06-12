# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`classify_tile_access`.

One test per per-dim kind, one per whole-subset composition rule, one
per special composition (diagonal, transpose). The classifier is a pure
function; tests construct ``Range`` objects directly and assert on the
returned ``TileAccess`` record.
"""
import pytest
import sympy

import dace
from dace import symbolic
from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.tile_access import (
    PerDimKind,
    TileAccessKind,
    classify_tile_access,
)


def _R(*ranges):
    """Build a :class:`Range` from one ``(lo, hi)`` or ``(lo, hi, step)``
    tuple per dim. Step defaults to ``1``."""
    out = []
    for r in ranges:
        if len(r) == 2:
            lo, hi = r
            step = 1
        else:
            lo, hi, step = r
        out.append((symbolic.pystr_to_symbolic(str(lo)) if isinstance(lo, str) else lo,
                    symbolic.pystr_to_symbolic(str(hi)) if isinstance(hi, str) else hi,
                    symbolic.pystr_to_symbolic(str(step)) if isinstance(step, str) else step))
    return Range(out)


# ---- per-dim kind tests ----------------------------------------------


def test_per_dim_broadcast_constant():
    """Constant subset on every dim -> all BROADCAST."""
    r = _R((0, 0), (5, 5))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.BROADCAST, PerDimKind.BROADCAST)
    assert ta.dim_strides == (0, 0)
    assert ta.kind == TileAccessKind.BROADCAST


def test_per_dim_broadcast_outer_symbol():
    """Outer-scope symbol (non-iter-var) on every dim -> BROADCAST."""
    r = _R(("M", "M"), ("N + 1", "N + 1"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.BROADCAST, PerDimKind.BROADCAST)
    assert ta.kind == TileAccessKind.BROADCAST


def test_per_dim_structured_identity():
    """Direct iter-var with identity coeff -> STRUCTURED_1 (dim_stride=1)."""
    r = _R(("i", "i"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.STRUCTURED_1, PerDimKind.STRUCTURED_1)
    assert ta.dim_strides == (1, 1)
    assert ta.kind == TileAccessKind.STRUCTURED
    assert ta.diagonal == {}
    assert ta.transpose is None


def test_per_dim_structured_with_offset():
    """``iter_var + c`` is still STRUCTURED_1 (offset captured)."""
    r = _R(("i + 1", "i + 1"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.STRUCTURED_1, )
    assert ta.dim_strides == (1, )
    # offset is symbolic 1
    assert int(ta.dim_offset[0]) == 1


def test_per_dim_affine_strided():
    """Non-unit coefficient -> AFFINE (dim_stride=coeff)."""
    r = _R(("2*i", "2*i"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.AFFINE, )
    assert ta.dim_strides == (2, )
    assert ta.kind == TileAccessKind.AFFINE


def test_per_dim_affine_multi_var():
    """Multiple iter-vars on one dim -> AFFINE (no isolatable stride)."""
    r = _R(("i + j", "i + j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind[0] == PerDimKind.AFFINE
    assert ta.kind == TileAccessKind.AFFINE


def test_per_dim_gather_nested_subscript():
    """Iter-var inside a Subscript -> GATHER dim."""
    # ``idx[i]`` -- iter-var nested
    r = _R(("idx[i]", "idx[i]"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.GATHER, PerDimKind.STRUCTURED_1)
    assert ta.dim_strides == (None, 1)
    assert ta.kind == TileAccessKind.GATHER


def test_per_dim_gather_resolves_index_access_node():
    """When inner SDFG is provided, GATHER dim's index access node is found."""
    sdfg = dace.SDFG("g_test")
    sdfg.add_array("idx", [16], dace.int32)
    sdfg.add_array("arr", [16], dace.float64)
    st = sdfg.add_state()
    st.add_access("idx")  # the AccessNode for idx
    r = _R(("idx[i]", "idx[i]"))
    ta = classify_tile_access(r, iter_vars=("i", ), inner_sdfg=sdfg)
    assert ta.per_dim_kind == (PerDimKind.GATHER, )
    assert ta.gather_index_per_dim[0] is not None
    assert ta.gather_index_per_dim[0].data == "idx"


# ---- whole-subset composition tests ----------------------------------


def test_whole_subset_kind_gather_wins():
    """Any GATHER dim -> whole-subset GATHER (precedence over STRUCTURED/AFFINE)."""
    r = _R(("idx[i]", "idx[i]"), ("2*j", "2*j"), ("k", "k"))
    ta = classify_tile_access(r, iter_vars=("i", "j", "k"))
    assert ta.kind == TileAccessKind.GATHER


def test_whole_subset_kind_affine_when_no_gather():
    """No GATHER but AFFINE present -> whole-subset AFFINE."""
    r = _R(("2*i", "2*i"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.AFFINE


def test_whole_subset_kind_structured_when_no_affine_no_gather():
    """All STRUCTURED_1 dims (+ optional BROADCAST) -> STRUCTURED."""
    r = _R(("i", "i"), ("M", "M"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.STRUCTURED


def test_whole_subset_kind_broadcast_when_all_broadcast():
    """All BROADCAST dims -> BROADCAST."""
    r = _R(("M", "M"), (0, 0))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.BROADCAST


# ---- special-composition tests ---------------------------------------


def test_diagonal_iter_var_in_multiple_dims():
    """Same iter-var as direct symbol in >=2 dims -> diagonal flag set."""
    r = _R(("i", "i"), ("i", "i"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert "i" in ta.diagonal
    assert ta.diagonal["i"] == (0, 1)


def test_transpose_non_canonical_order():
    """Iter-vars in permuted order across dims -> transpose flag set."""
    # spec.iter_vars = ("i", "j") but subset is (j, i)
    r = _R(("j", "j"), ("i", "i"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.STRUCTURED
    # transpose permutation: dim 0 -> iter_var index 1 (j); dim 1 -> iter_var index 0 (i)
    assert ta.transpose == (1, 0)


def test_canonical_order_no_transpose_flag():
    """Iter-vars in canonical order -> transpose is None."""
    r = _R(("i", "i"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.STRUCTURED
    assert ta.transpose is None


# ---- K-rank transition (broadcast) tests -----------------------------


def test_k0_to_k2_full_splat():
    """No iter-var anywhere -> all BROADCAST dims -> K=0 to K=2 splat."""
    r = _R(("M", "M"), (0, 0))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.dim_strides == (0, 0)
    assert ta.kind == TileAccessKind.BROADCAST


def test_k1_to_k2_broadcast_on_dim_0():
    """Iter-var on dim 1 only, dim 0 invariant -> K=1 to K=2 (broadcast dim 0)."""
    r = _R(("M", "M"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.dim_strides == (0, 1)
    assert ta.per_dim_kind == (PerDimKind.BROADCAST, PerDimKind.STRUCTURED_1)
    # Whole-subset is STRUCTURED (any non-broadcast dim promotes from BROADCAST).
    assert ta.kind == TileAccessKind.STRUCTURED


def test_k1_to_k2_broadcast_on_dim_1():
    """Iter-var on dim 0 only -> K=1 to K=2 (broadcast dim 1)."""
    r = _R(("i", "i"), (0, 0))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.dim_strides == (1, 0)
    assert ta.per_dim_kind == (PerDimKind.STRUCTURED_1, PerDimKind.BROADCAST)


# ---- mixed-case / fallback tests -------------------------------------


def test_gather_with_structured_other_dim_stays_gather():
    """Composition rule: any GATHER -> whole-subset GATHER; structured
    dims contribute as affine sub-expressions in the gather index."""
    r = _R(("idx[i]", "idx[i]"), ("j", "j"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.kind == TileAccessKind.GATHER
    # The structured dim is still recorded
    assert ta.per_dim_kind[1] == PerDimKind.STRUCTURED_1
    assert ta.dim_strides[1] == 1


def test_non_affine_iter_var_expression_degrades_to_affine_no_stride():
    """A genuinely non-affine expression like ``i**2`` -> AFFINE with no
    int stride (emitter degrades to GATHER). ``int_floor`` / ``int_ceil``
    patterns are now recognised separately as REPLICATE."""
    r = _R(("i ** 2", "i ** 2"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind[0] == PerDimKind.AFFINE
    assert ta.dim_strides[0] is None


# ---- REPLICATE / replicate_factor spectrum --------------------------


def test_replicate_int_floor_factor_2():
    """``arr[i // 2]`` -> REPLICATE with factor=2 (within-dim group
    sharing: every 2 lanes read the same source element). The codegen
    loads W/2 elements and group-broadcasts each twice."""
    r = _R(("i // 2", "i // 2"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.REPLICATE, )
    assert ta.replicate_factor_per_dim == (2, )
    # REPLICATE shares the STRUCTURED whole-subset bucket: it's a
    # perfectly regular access, just with grouped lanes.
    assert ta.kind == TileAccessKind.STRUCTURED


def test_replicate_int_floor_factor_4():
    """``arr[i // 4]`` -> REPLICATE with factor=4."""
    r = _R(("i // 4", "i // 4"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.REPLICATE, )
    assert ta.replicate_factor_per_dim == (4, )


def test_replicate_int_floor_explicit_form():
    """``int_floor(i, 2)`` (the user-facing function form) -> same
    REPLICATE classification as the ``i // 2`` Python-operator form."""
    r = _R(("int_floor(i, 2)", "int_floor(i, 2)"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.REPLICATE, )
    assert ta.replicate_factor_per_dim == (2, )


def test_replicate_int_ceil():
    """``int_ceil(i, 2)`` -> REPLICATE with factor=2 (the dividend is
    the iter-var, divisor is the replicate factor)."""
    r = _R(("int_ceil(i, 2)", "int_ceil(i, 2)"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.REPLICATE, )
    assert ta.replicate_factor_per_dim == (2, )


def test_replicate_int_floor_affine_inner():
    """``arr[(i + 1) // 2]`` -> REPLICATE with factor=2; the affine
    offset on the iter-var doesn't change the replicate factor."""
    r = _R(("(i + 1) // 2", "(i + 1) // 2"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.REPLICATE, )
    assert ta.replicate_factor_per_dim == (2, )


def test_replicate_factor_recorded_on_structured_dims_too():
    """STRUCTURED_1 dims have replicate_factor = 1 (the contiguous-load
    endpoint of the spectrum); BROADCAST dims have replicate_factor =
    None (the full-broadcast endpoint -- all W lanes share)."""
    r = _R(("i", "i"), (0, 0))  # STRUCTURED_1 on dim 0, BROADCAST on dim 1
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.STRUCTURED_1, PerDimKind.BROADCAST)
    assert ta.replicate_factor_per_dim == (1, None)


def test_replicate_mixed_with_structured():
    """``arr[i, j // 2]`` -> STRUCTURED_1 on dim 0, REPLICATE on dim 1.
    Demonstrates the per-dim spectrum: one dim contiguous, the other
    group-broadcast. The whole-subset kind stays STRUCTURED (both are
    perfectly regular)."""
    r = _R(("i", "i"), ("j // 2", "j // 2"))
    ta = classify_tile_access(r, iter_vars=("i", "j"))
    assert ta.per_dim_kind == (PerDimKind.STRUCTURED_1, PerDimKind.REPLICATE)
    assert ta.replicate_factor_per_dim == (1, 2)
    assert ta.kind == TileAccessKind.STRUCTURED


def test_replicate_data_dependent_falls_to_gather():
    """``arr[int_floor(idx[i], 2)]`` -> GATHER (data-dependent index
    inside the floor; the gather machinery handles it). REPLICATE
    detection requires the dividend to be AFFINE in the iter-var."""
    r = _R(("int_floor(idx[i], 2)", "int_floor(idx[i], 2)"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind == (PerDimKind.GATHER, )


def test_replicate_symbolic_divisor_stays_replicate_with_runtime_check():
    """Symbolic divisor (``i // K`` with K symbolic) classifies as REPLICATE
    per design 2c7b88e26 (runtime check ``W % K == 0`` emitted at codegen,
    replacing the prior compile-time refusal). Floats are still refused
    outright -- access expressions are integer-valued by contract."""
    r = _R(("i // K", "i // K"))
    ta = classify_tile_access(r, iter_vars=("i", ))
    assert ta.per_dim_kind[0] == PerDimKind.REPLICATE
    assert ta.replicate_factor_per_dim[0] is not None


def test_replicate_float_divisor_refused():
    """Float divisor in an access expression is illegal -- the classifier
    refuses (no silent truncation to int) so the dim falls to AFFINE/GATHER."""
    import sympy as _sp
    from dace.transformation.passes.vectorization.utils.tile_access import _detect_replicate_factor
    # _detect_replicate_factor should refuse a float divisor.
    expr = _sp.Function("int_floor")(_sp.Symbol("i"), _sp.Float(2.5))
    assert _detect_replicate_factor(expr, "i") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
