# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`TileDimSpec`, :func:`classify_tile_access`
and the :class:`MarkTileDims` validation pass.
"""
import pytest

import dace
from dace import subsets
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    TileDimSpec,
    build_dim_index_map,
    classify_tile_access,
)


def _range(*begins):
    """Build a Range whose per-dim begin (==end, step 1) are ``begins``."""
    return subsets.Range([(dace.symbolic.pystr_to_symbolic(b),
                           dace.symbolic.pystr_to_symbolic(b), 1) for b in begins])


def test_build_dim_index_map_affine_and_structured():
    """The per-lane index map records affine coeffs and the structured flag."""
    # a[2*i, j]: dim0 affine coeff 2 in tile 0; dim1 affine coeff 1 in tile 1.
    dims = build_dim_index_map(_range("2*i", "j"), ("i", "j"))
    assert dims[0].dep == (0,) and dims[0].affine_coeffs == {0: 2} and not dims[0].structured
    assert dims[1].dep == (1,) and dims[1].affine_coeffs == {1: 1}
    # a[i // 2]: structured (int_floor of affine arg), not affine.
    sdims = build_dim_index_map(_range("int_floor(i, 2)"), ("i",))
    assert sdims[0].dep == (0,) and sdims[0].structured and 0 not in sdims[0].affine_coeffs


def test_classify_structured_int_floor():
    """``a[i // 2]`` (int_floor of an affine arg) classifies as STRUCTURED."""
    cls = classify_tile_access(_range("int_floor(i, 2)"), array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.STRUCTURED


def test_classify_structured_int_ceil():
    """``int_ceil`` of an affine arg is STRUCTURED."""
    cls = classify_tile_access(_range("int_ceil(i, 4)"), array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.STRUCTURED


@pytest.mark.parametrize("widths,iter_vars,global_ubs", [
    ((8,), ("i",), ("N",)),
    ((4, 8), ("i", "j"), ("M", "N")),
])
def test_tile_dim_spec_accepts_K_1_2(widths, iter_vars, global_ubs):
    """``TileDimSpec`` is the data structure for K = 1, 2 (K=3 supported
    but untested in MVP)."""
    spec = TileDimSpec(iter_vars=iter_vars, widths=widths, global_ubs=global_ubs)
    assert spec.K == len(widths)
    assert spec.widths == widths
    assert spec.iter_vars == iter_vars
    assert spec.global_ubs == global_ubs


def test_tile_dim_spec_rejects_K_outside_1_3():
    """K must be in ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="K must be in"):
        TileDimSpec(iter_vars=("i", "j", "k", "l"), widths=(2, 2, 2, 2),
                    global_ubs=("A", "B", "C", "D"))
    with pytest.raises(ValueError, match="K must be in"):
        TileDimSpec(iter_vars=(), widths=(), global_ubs=())


def test_tile_dim_spec_rejects_length_mismatch():
    """The three tuples must agree on length."""
    with pytest.raises(ValueError, match="lengths must agree"):
        TileDimSpec(iter_vars=("i", "j"), widths=(8,), global_ubs=("N",))


def test_classify_contiguous_1d():
    """``A[i_0:i_0+W]`` with stride-1 array is :attr:`CONTIGUOUS`."""
    sub = subsets.Range([(dace.symbolic.pystr_to_symbolic("i"),
                          dace.symbolic.pystr_to_symbolic("i+7"),
                          1)])
    cls = classify_tile_access(sub, array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.CONTIGUOUS
    assert cls.dim_strides == (1,)


def test_classify_strided_1d_with_array_stride():
    """``A[i]`` on a non-unit *memory*-stride dim is :attr:`STRIDED`. The
    coefficient is 1 (``dim_strides``); the non-unit stride lives in the
    array's stride on the matched dim, applied at lowering time."""
    sub = subsets.Range([(dace.symbolic.pystr_to_symbolic("i"),
                          dace.symbolic.pystr_to_symbolic("i+7"),
                          1)])
    cls = classify_tile_access(sub, array_strides=(2,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.STRIDED
    assert cls.dim_strides == (1,)
    assert cls.match_dims == (0,)


def test_classify_strided_1d_with_linear_coeff():
    """``A[2*i]`` on a unit-array-stride dim is :attr:`STRIDED` with coeff 2."""
    sub = subsets.Range([(dace.symbolic.pystr_to_symbolic("2*i"),
                          dace.symbolic.pystr_to_symbolic("2*i+14"),
                          1)])
    cls = classify_tile_access(sub, array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.STRIDED
    assert cls.dim_strides == (2,)


def test_classify_broadcast_symbol_1d():
    """``A[k]`` (where ``k`` is NOT in tile iter-vars) is :attr:`BROADCAST_SYMBOL`."""
    sub = subsets.Range([(dace.symbolic.pystr_to_symbolic("k"),
                          dace.symbolic.pystr_to_symbolic("k"),
                          1)])
    cls = classify_tile_access(sub, array_strides=(1,), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.BROADCAST_SYMBOL
    assert cls.dim_strides == (0,)


def test_classify_contiguous_2d():
    """``A[i, j]`` 2D contiguous (C-layout) under K=2 tiling: tile dims map
    to the last 2 array dims in order with unit coefficients and a unit
    innermost memory stride, so it is :attr:`CONTIGUOUS`. The outer dim's
    memory stride (8) is applied at lowering via ``match_dims``."""
    sub = subsets.Range([
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i+3"), 1),
        (dace.symbolic.pystr_to_symbolic("j"), dace.symbolic.pystr_to_symbolic("j+7"), 1),
    ])
    cls = classify_tile_access(sub, array_strides=(8, 1), tile_iter_vars=("i", "j"))
    assert cls.kind == TileAccessKind.CONTIGUOUS
    assert cls.dim_strides == (1, 1)
    assert cls.match_dims == (0, 1)


def test_classify_transposed_2d_is_strided():
    """``A[j, i]`` with tile dim ``j`` mapping to a 2-D array's FIRST dim is
    a perfect box but transposed: STRIDED, with ``match_dims=(0,)`` so the
    lowering steps along the column (stride 8), not the row."""
    sub = subsets.Range([
        (dace.symbolic.pystr_to_symbolic("j"), dace.symbolic.pystr_to_symbolic("j+7"), 1),
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i"), 1),
    ])
    cls = classify_tile_access(sub, array_strides=(8, 1), tile_iter_vars=("j",))
    assert cls.kind == TileAccessKind.STRIDED
    assert cls.dim_strides == (1,)
    assert cls.match_dims == (0,)


def test_classify_diagonal_is_gather():
    """``A[i, i]`` (tile var ``i`` in two dims) is NOT a perfect box, so it
    classifies as :attr:`GATHER` rather than a strided load."""
    sub = subsets.Range([
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i+7"), 1),
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i+7"), 1),
    ])
    cls = classify_tile_access(sub, array_strides=(8, 1), tile_iter_vars=("i",))
    assert cls.kind == TileAccessKind.GATHER


def test_classify_diagonal_refused_for_K_gt_1():
    """A diagonal (one tile var spanning >=2 dims) is a valid K=1 GATHER but is
    refused for K>1: a multi-dim register tile cannot fold one shared tile var
    across tile dimensions, so it classifies as :attr:`UNRECOGNIZED` (the
    orchestrator then skips it -- a diagonal is a K=1 pattern)."""
    # a[i, i, j]: tile var i spans dims 0,1 (diagonal); j in dim 2; both bound.
    sub = subsets.Range([
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i+7"), 1),
        (dace.symbolic.pystr_to_symbolic("i"), dace.symbolic.pystr_to_symbolic("i+7"), 1),
        (dace.symbolic.pystr_to_symbolic("j"), dace.symbolic.pystr_to_symbolic("j+7"), 1),
    ])
    cls = classify_tile_access(sub, array_strides=(64, 8, 1), tile_iter_vars=("i", "j"))
    assert cls.kind == TileAccessKind.UNRECOGNIZED


def test_classify_unrecognized_when_tile_var_missing():
    """If a tile iter-var is referenced in NO subset dim, classifier
    can't bind it — :attr:`UNRECOGNIZED`."""
    sub = subsets.Range([(dace.symbolic.pystr_to_symbolic("i"),
                          dace.symbolic.pystr_to_symbolic("i+7"),
                          1)])
    cls = classify_tile_access(sub, array_strides=(1,), tile_iter_vars=("i", "j"))
    assert cls.kind == TileAccessKind.UNRECOGNIZED


def _build_k2_sdfg():
    """Build a minimal 2D map SDFG for MarkTileDims tests."""
    sdfg = dace.SDFG("k2_outer")
    N = dace.symbol("N")
    M = dace.symbol("M")
    sdfg.add_array("A", (M, N), dace.float64)
    sdfg.add_array("C", (M, N), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "body",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]")},
        "_c = _a",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def test_mark_tile_dims_picks_K_innermost():
    """K=2 inner map yields a spec covering its 2 innermost params."""
    sdfg = _build_k2_sdfg()
    result = MarkTileDims(widths=(4, 8)).apply_pass(sdfg, {})
    assert result is not None
    assert len(result) == 1
    spec = next(iter(result.values()))
    assert spec.iter_vars == ("i", "j")
    assert spec.widths == (4, 8)
    assert spec.global_ubs == ("M", "N")


def test_mark_tile_dims_K1_collapse():
    """K=1 spec only takes the last param of the inner map."""
    sdfg = _build_k2_sdfg()
    result = MarkTileDims(widths=(8,)).apply_pass(sdfg, {})
    assert result is not None
    spec = next(iter(result.values()))
    assert spec.iter_vars == ("j",)
    assert spec.widths == (8,)
    assert spec.global_ubs == ("N",)


def test_mark_tile_dims_raises_on_too_few_params():
    """A 1D map under K=2 raises ``NotImplementedError`` by default."""
    sdfg = dace.SDFG("k1_outer")
    N = dace.symbol("N")
    sdfg.add_array("A", (N,), dace.float64)
    sdfg.add_array("C", (N,), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "body",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]")},
        "_c = _a",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    with pytest.raises(NotImplementedError, match="< K=2"):
        MarkTileDims(widths=(4, 8)).apply_pass(sdfg, {})


def test_mark_tile_dims_soft_skip_ineligible():
    """``skip_ineligible=True`` silently drops the ineligible map."""
    sdfg = dace.SDFG("k1_outer_skip")
    N = dace.symbol("N")
    sdfg.add_array("A", (N,), dace.float64)
    sdfg.add_array("C", (N,), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "body",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]")},
        "_c = _a",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    result = MarkTileDims(widths=(4, 8), skip_ineligible=True).apply_pass(sdfg, {})
    assert result is None


def test_mark_tile_dims_rejects_invalid_K_at_construction():
    """Constructor refuses ``widths`` length outside ``{1, 2, 3}``."""
    with pytest.raises(ValueError, match="length"):
        MarkTileDims(widths=(2, 2, 2, 2))
