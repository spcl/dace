# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile tile-rank normalization (bug 07 root cause).

For an array with ``ndim > K`` the aligned ``ct.load`` path used to return a
source-rank tile (``ct.load(..., shape=(1, W, W))``) while the SDFG tile
descriptor declared rank ``K`` (``(W, W)``). The rank-``ndim`` tile then flowed
through the whole kernel (elementwise ops broadcast silently) and was squeezed
back only at the store (``_squeeze_tile_to_widths``) — sink-side compensation
that left rank-sensitive intermediate ops (``TileReduce`` axis numbering,
``TileMMA``'s 2-D requirement) exposed to a ``(1, W, W)`` tile where the
descriptor promised ``(W, W)``.

The fix restores the invariant *runtime tile rank == declared descriptor rank
K* at the SOURCE: the aligned load wraps its result in ``ct.reshape(..., (W,
...))`` when ``ndim > K``, and the store raises loudly on a rank-mismatched
``_src`` descriptor instead of silently reshaping.

Codegen-assertion tests need no GPU; the end-to-end tests (marked ``gpu``) run
the full ``VectorizeCuTile`` pipeline and compare against NumPy.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileGather, TileScatter
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

L = dace.symbol("L")
M = dace.symbol("M")
N = dace.symbol("N")


def _expand_cutile_with_edges(lib_node, in_arrays=None, out_arrays=None, transients=()):
    """Expand a lib node's ``cutile`` implementation with wired edges.

    :param lib_node: The tileops library node to expand.
    :param in_arrays: Mapping of input connector to ``(name, shape, dtype)``.
    :param out_arrays: Mapping of output connector to ``(name, shape, dtype)``.
    :param transients: Array names declared transient.
    :returns: The tasklet body string.
    """
    sdfg = dace.SDFG(f"cutile_rank_{lib_node.label}")
    state = sdfg.add_state("main")
    state.add_node(lib_node)
    for mapping, is_input in ((in_arrays or {}, True), (out_arrays or {}, False)):
        for conn, (arr_name, shape, dtype) in mapping.items():
            if arr_name not in sdfg.arrays:
                sdfg.add_array(arr_name, shape, dtype, transient=arr_name in transients)
            acc = state.add_access(arr_name)
            mem = dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name])
            if is_input:
                state.add_edge(acc, None, lib_node, conn, mem)
            else:
                state.add_edge(lib_node, conn, acc, None, mem)
    cls = lib_node.implementations["cutile"]
    return cls.expansion(lib_node, state, sdfg).code.as_string


@dace.program
def _scale3d(A: dace.float64[L, M, N], B: dace.float64[L, M, N]):
    for i in range(L):
        for j in range(M):
            for k in range(N):
                B[i, j, k] = 2.0 * A[i, j, k]


def _lowered_code(prog, widths):
    """Lower ``prog`` through ``VectorizeCuTile`` and return the generated code."""
    sdfg = prog.to_sdfg(simplify=True)
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    return sdfg, "".join(c.clean_code for c in sdfg.generate_code())


# ---------------------------------------------------------------------------
# Generated-code assertions (no GPU required)
# ---------------------------------------------------------------------------


def test_load_ndim_gt_k_reshapes_at_load():
    """Aligned load of a 3-D array with K=2 widths must reshape the source-rank
    ``(1, 8, 8)`` load result to the declared rank-K ``(8, 8)`` tile."""
    body = _expand_cutile_with_edges(
        TileGather(name="L", widths=(8, 8)),
        in_arrays={"_src": ("src", (10, 100, 200), dace.float64)},
        out_arrays={"_dst": ("dst", (8, 8), dace.float64)},
        transients=("dst", ),
    )
    assert "shape=(1, 8, 8)" in body
    assert "ct.reshape(ct.load(_src" in body
    assert "(8, 8))" in body


def test_load_ndim_eq_k_has_no_reshape():
    """When ``ndim == K`` the aligned load is already rank-K: no reshape."""
    body = _expand_cutile_with_edges(
        TileGather(name="L", widths=(8, 8)),
        in_arrays={"_src": ("src", (100, 200), dace.float64)},
        out_arrays={"_dst": ("dst", (8, 8), dace.float64)},
        transients=("dst", ),
    )
    assert "ct.reshape" not in body


def test_pipeline_ndim_gt_k_tiles_are_rank_k():
    """Full pipeline on a 3-D program with K=2: the reshape is emitted at the
    load and the store consumes the K-dim ``_src`` directly (no sink-side
    ``ct.reshape(_src, ...)`` squeeze)."""
    sdfg, code = _lowered_code(_scale3d, widths=(8, 8))
    tiles = [desc for owner in sdfg.all_sdfgs_recursive() for desc in owner.arrays.values()
             if desc.storage == dace.StorageType.CuTile_Tile]
    assert tiles and all(tuple(desc.shape) == (8, 8) for desc in tiles)
    assert "ct.load(" in code or "ct.gather(" in code
    assert "ct.reshape(_src" not in code


def test_store_rank_mismatch_raises():
    """A store whose ``_src`` descriptor rank differs from K must fail loudly
    (the rank invariant is restored at the load; nothing may silently
    reshape at the sink)."""
    with pytest.raises(ValueError, match="rank"):
        _expand_cutile_with_edges(
            TileScatter(name="S", widths=(8, ), has_mask=True),
            in_arrays={
                "_src": ("src_tile", (1, 8), dace.float64),
                "_mask": ("mask_tile", (8, ), dace.bool_),
            },
            out_arrays={"_dst": ("dst", (10, 100), dace.float64)},
            transients=("src_tile", "mask_tile"),
        )


# ---------------------------------------------------------------------------
# End-to-end GPU tests (ndim > K through the full pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(3, 16, 24), (3, 13, 21)])
def test_ndim_gt_k_elementwise_matches_numpy(shape):
    """3-D elementwise scale tiled over the inner 2 dims (K=2), divisible and
    non-divisible sizes (remainder handling via the iteration mask)."""
    sdfg = _scale3d.to_sdfg(simplify=True)
    VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    rng = np.random.default_rng(sum(shape))
    A = rng.random(shape)
    B = np.zeros(shape)
    csdfg(A=A, B=B, L=shape[0], M=shape[1], N=shape[2])
    np.testing.assert_allclose(B, 2.0 * A, rtol=1e-14)


@pytest.mark.gpu
def test_ndim_gt_k_two_input_binop_matches_numpy():
    """3-D binop of two rank-normalized loads (both operands rank-K)."""

    @dace.program
    def add3d(A: dace.float64[L, M, N], B: dace.float64[L, M, N], C: dace.float64[L, M, N]):
        for i in range(L):
            for j in range(M):
                for k in range(N):
                    C[i, j, k] = A[i, j, k] + B[i, j, k]

    sdfg = add3d.to_sdfg(simplify=True)
    VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    rng = np.random.default_rng(11)
    shape = (2, 11, 19)
    A = rng.random(shape)
    B = rng.random(shape)
    C = np.zeros(shape)
    csdfg(A=A, B=B, C=C, L=shape[0], M=shape[1], N=shape[2])
    np.testing.assert_allclose(C, A + B, rtol=1e-14)


if __name__ == "__main__":
    test_load_ndim_gt_k_reshapes_at_load()
    test_load_ndim_eq_k_has_no_reshape()
    test_pipeline_ndim_gt_k_tiles_are_rank_k()
    test_store_rank_mismatch_raises()
    test_ndim_gt_k_elementwise_matches_numpy((3, 13, 21))
    test_ndim_gt_k_two_input_binop_matches_numpy()
    print("rank normalization tests passed")
