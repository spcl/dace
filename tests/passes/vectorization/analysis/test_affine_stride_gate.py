# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for G6 -- AFFINE stride tile-invariance gate (design section 4.2)."""
import dace
from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.tile_access import (
    PerDimKind,
    classify_tile_access,
)


def _build_body(symbol_name, rhs):
    sdfg = dace.SDFG("body_g6")
    sdfg.add_symbol(symbol_name, dace.int64)
    s0 = sdfg.add_state("entry", is_start_block=True)
    s1 = sdfg.add_state("after")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={symbol_name: rhs}))
    return sdfg


def test_outer_constant_coefficient_stays_affine():
    """``a[N*i + 1]`` with ``N`` outer-constant -> AFFINE stride N."""
    sdfg = dace.SDFG("outer_N")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_state("s")
    subset = Range([(dace.symbolic.pystr_to_symbolic("N*i + 1"), dace.symbolic.pystr_to_symbolic("N*i + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.AFFINE, )


def test_tile_dependent_coefficient_forces_gather():
    """``a[sym*i + 1]`` with ``sym <- j`` (j a tile iter-var) -> GATHER."""
    sdfg = _build_body("syma", "j")
    subset = Range([(dace.symbolic.pystr_to_symbolic("syma*i + 1"), dace.symbolic.pystr_to_symbolic("syma*i + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", "j"), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.GATHER, )


def test_unit_coefficient_stays_linear_even_with_inner_sdfg():
    """``a[i + 3]`` -> LINEAR; G6 gate doesn't affect coeff=1 cases."""
    sdfg = dace.SDFG("linear_check")
    sdfg.add_state("s")
    subset = Range([(dace.symbolic.pystr_to_symbolic("i + 3"), dace.symbolic.pystr_to_symbolic("i + 3"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.LINEAR, )


def test_integer_coefficient_stays_affine():
    """``a[2*i + 1]`` -> AFFINE stride 2; integer coeff has no free symbols, gate is a no-op."""
    sdfg = dace.SDFG("int_coeff")
    sdfg.add_state("s")
    subset = Range([(dace.symbolic.pystr_to_symbolic("2*i + 1"), dace.symbolic.pystr_to_symbolic("2*i + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=sdfg)
    assert record.per_dim_kind == (PerDimKind.AFFINE, )
    assert record.dim_strides == (2, )


def test_no_inner_sdfg_skips_gate():
    """When inner_sdfg is None, the gate is a no-op (preserves backwards compat)."""
    subset = Range([(dace.symbolic.pystr_to_symbolic("N*i + 1"), dace.symbolic.pystr_to_symbolic("N*i + 1"), 1)])
    record = classify_tile_access(subset, iter_vars=("i", ), inner_sdfg=None)
    assert record.per_dim_kind == (PerDimKind.AFFINE, )
