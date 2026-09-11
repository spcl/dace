# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=2 broadcast patterns inside an NSDFG body — descent contract.

The K-dim descent (``PromoteNSDFGBodyToTiles`` + ``EmitTileOps``) must
lower every read into a tile lib node, including broadcasts where the
source rank is lower than the tile rank. The patterns covered here:

1. **Scalar (0-D) -> (W_jk, W_jc) tile** — every lane reads the same
   element, ``dim_strides=(0, 0)`` on the ``TileLoad``.
2. **1-D column ``a[jk]`` -> 2-D tile** — each row of the tile gets a
   distinct ``a[jk + l0]`` value, broadcast across ``jc``,
   ``dim_strides=(1, 0)``.
3. **1-D row ``a[jc]`` -> 2-D tile** — each column of the tile gets a
   distinct ``a[jc + l1]`` value, broadcast across ``jk``,
   ``dim_strides=(0, 1)``.
4. **2-D contiguous ``a[jk, jc]`` -> tile** — the baseline,
   ``dim_strides=(1, 1)``.
5. **1-D column gather ``a[idx[jk]]``** — per-row data-dependent
   gather, broadcast across ``jc`` — lowers to a ``TileLoad`` (gather) whose
   index tile encodes the broadcast.
6. **1-D column structured ``a[jk // 2]``** — per-row structured
   gather (lane replication), broadcast across ``jc``.
7. **1-D row gather ``a[idx[jc]]``** — per-column data-dep gather,
   broadcast across ``jk``.
8. **1-D row structured ``a[jc // 2]``** — per-column structured
   gather, broadcast across ``jk``.

All tests assert the post-descent SDFG holds **zero raw Tasklet
nodes** at the K-dim layer (the contract: K-dim → tile ops only) and
that the expected per-tile-dim shape survives.
"""

import numpy as np
import pytest
import dace

from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import is_assumption_guard_block
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import _is_assign_tasklet
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy, BranchMode
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

NK = dace.symbol("NK")
NJ = dace.symbol("NJ")


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Number of NON-assign raw ``Tasklet`` nodes anywhere in ``sdfg``.

    Trivial ``_out = _in`` assigns are LEFT in place by the descent
    (``_promote_internal_assigns`` is a no-op per user directive --
    collapsing them into AN -> AN would silently drop the source-side
    coordinates). They are semantically fine and lower to a one-element
    copy at codegen, so the K-dim tile-only contract is preserved as
    "no NON-assign raw tasklets" rather than "zero tasklets total".

    The runtime nonnegativity guard's ``std::abort`` tasklets (symbolic-size
    kernels) are infrastructure, not compute, so they are excluded exactly as the
    ``tile_runtime`` divisibility trip guards are."""
    return sum(
        1 for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and not _is_assign_tasklet(n) and not is_assumption_guard_block(parent))


def _count_lib_nodes_by_type(sdfg: dace.SDFG, cls) -> int:
    """Number of lib nodes of ``cls`` anywhere in ``sdfg``."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls))


def _tile_loads(sdfg: dace.SDFG) -> list[TileLoad]:
    """Every ``TileLoad`` node anywhere in ``sdfg``, recursing into nested SDFGs."""
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad)]


def _vectorize_k2(sdfg: dace.SDFG) -> None:
    """Run the K=2 (8, 8) orchestrator, leaving tile lib nodes intact."""
    VectorizeCPUMultiDim(
        VectorizeConfig(
            widths=(8, 8),
            target_isa=ISA.SCALAR,
            remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
            branch_mode=BranchMode.MERGE,
            loop_to_map_permissive=False,
            scalar_remainder_emit="tile_k1",
            expand_tile_nodes=False,
        )).apply_pass(sdfg, {})


# ---------------------------------------------------------------- shapes


@dace.program
def _scalar_broadcast(a: dace.float64[1], c: dace.float64[NK, NJ]):
    """Scalar -> (W_jk, W_jc) tile broadcast.

    Every lane reads the same ``a[0]`` element; the descent must
    materialize this as one ``TileLoad`` with ``dim_strides=(0, 0)``.
    """
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[0]


@dace.program
def _col_broadcast(a: dace.float64[NK], c: dace.float64[NK, NJ]):
    """1-D column ``a[jk]`` -> 2-D tile, broadcast across jc.

    Each row of the tile gets a distinct ``a`` value; the ``jc`` lanes
    of the same row read the same value, so the inner-lane dim_stride
    is ``0``.
    """
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jk]


@dace.program
def _row_broadcast(a: dace.float64[NJ], c: dace.float64[NK, NJ]):
    """1-D row ``a[jc]`` -> 2-D tile, broadcast across jk.

    Each column of the tile gets a distinct ``a`` value; the ``jk``
    lanes of the same column read the same value, so the outer-lane
    dim_stride is ``0``.
    """
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jc]


@dace.program
def _full_2d_baseline(a: dace.float64[NK, NJ], c: dace.float64[NK, NJ]):
    """2-D contiguous ``a[jk, jc]`` -> tile — no broadcast; sanity row."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jk, jc]


@dace.program
def _col_gather(a: dace.float64[NK], idx: dace.int32[NK], c: dace.float64[NK, NJ]):
    """Per-row data-dependent gather, broadcast across jc."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[idx[jk]]


@dace.program
def _col_structured(a: dace.float64[NK], c: dace.float64[NK, NJ]):
    """Per-row structured (lane-replication) gather, broadcast across jc."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jk // 2]


@dace.program
def _row_gather(a: dace.float64[NJ], idx: dace.int32[NJ], c: dace.float64[NK, NJ]):
    """Per-column data-dependent gather, broadcast across jk."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[idx[jc]]


@dace.program
def _row_structured(a: dace.float64[NJ], c: dace.float64[NK, NJ]):
    """Per-column structured (lane-replication) gather, broadcast across jk."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jc // 2]


@dace.program
def _fully_structured_2d(a: dace.float64[NK, NJ], c: dace.float64[NK, NJ]):
    """Fully structured 2-D: ``a[jk // 2, jc]`` — lane-replication along jk + affine along jc."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jk // 2, jc]


@dace.program
def _fully_unstructured_separable(a: dace.float64[NK, NJ], idx_k: dace.int32[NK], idx_j: dace.int32[NJ],
                                  c: dace.float64[NK, NJ]):
    """Fully unstructured separable: ``a[idx_k[jk], idx_j[jc]]``.

    Both dims data-dependent, but the index sources factor cleanly: one
    1-D per-row index tile + one 1-D per-column index tile, the outer
    product of which addresses the 2-D source. No tile var spans
    multiple source dims.
    """
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[idx_k[jk], idx_j[jc]]


@dace.program
def _fully_unstructured_2d_index(a: dace.float64[NK], idx: dace.int32[NK, NJ], c: dace.float64[NK, NJ]):
    """1-D source with 2-D index source: ``a[idx[jk, jc]]``.

    Per-lane gather with a tile-shaped (8, 8) index tile read from the
    2-D ``idx`` array. The single source dim collapses both tile vars.
    """
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[idx[jk, jc]]


# ---------------------------------------------------------------- tests


def test_scalar_broadcast_descent_to_tile_only():
    """Scalar (size-1) broadcast lowers via a Scalar-kind TileLoad, not per-element compute."""
    sdfg = _scalar_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim scalar-broadcast must lower to tile-only"
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert loads[0].src_kind == "Scalar", f"a size-1 source must broadcast via src_kind=Scalar, got {loads[0]!r}"
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_col_broadcast_descent_to_tile_only():
    """1D-column broadcast (a[jk] across jc): stride 1 along jk, 0 (broadcast) along jc."""
    sdfg = _col_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim col-broadcast must lower to tile-only"
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [1, 0], f"expected dim_strides=(1, 0), got {loads[0].dim_strides}"
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_row_broadcast_descent_to_tile_only():
    """1D-row broadcast (a[jc] across jk): stride 0 (broadcast) along jk, 1 along jc."""
    sdfg = _row_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim row-broadcast must lower to tile-only"
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [0, 1], f"expected dim_strides=(0, 1), got {loads[0].dim_strides}"
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_full_2d_baseline_descent_to_tile_only():
    """2D contiguous load baseline: unit stride on both tile dims, no broadcast."""
    sdfg = _full_2d_baseline.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [1, 1], f"expected dim_strides=(1, 1), got {loads[0].dim_strides}"
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_col_gather_descent_to_tile_only():
    """Per-row data-dep gather (a[idx[jk]]): gather_dims must fire, not degrade to a plain load."""
    sdfg = _col_gather.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    gathers = [n for n in _tile_loads(sdfg) if n.gather_dims]
    assert len(gathers) == 1, f"expected exactly one gather TileLoad, got {gathers}"
    assert list(gathers[0].gather_dims) == [0], f"expected gather_dims=(0,), got {gathers[0].gather_dims}"


def test_col_structured_descent_to_tile_only():
    """Per-row structured ``a[jk // 2]`` broadcast across jc: lane-replication factor 2 on jk."""
    sdfg = _col_structured.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [1, 0], f"expected dim_strides=(1, 0), got {loads[0].dim_strides}"
    assert list(loads[0].replicate_factor_per_dim) == [
        2, 8
    ], f"a[jk // 2] must replicate factor 2 on jk, got {loads[0].replicate_factor_per_dim}"


def test_row_gather_descent_to_tile_only():
    """Per-column data-dep gather (a[idx[jc]]) broadcast across jk; gather_dims must fire."""
    sdfg = _row_gather.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    gathers = [n for n in _tile_loads(sdfg) if n.gather_dims]
    assert len(gathers) == 1, f"expected exactly one gather TileLoad, got {gathers}"
    assert list(gathers[0].gather_dims) == [0], f"expected gather_dims=(0,), got {gathers[0].gather_dims}"


def test_row_structured_descent_to_tile_only():
    """Per-column structured ``a[jc // 2]`` broadcast across jk: lane-replication factor 2 on jc."""
    sdfg = _row_structured.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [0, 1], f"expected dim_strides=(0, 1), got {loads[0].dim_strides}"
    assert list(loads[0].replicate_factor_per_dim) == [
        8, 2
    ], f"a[jc // 2] must replicate factor 2 on jc, got {loads[0].replicate_factor_per_dim}"


def test_fully_structured_2d_descent_to_tile_only():
    """``a[jk // 2, jc]`` — affine stride on both dims, lane-replication factor 2 on jk only."""
    sdfg = _fully_structured_2d.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    loads = _tile_loads(sdfg)
    assert len(loads) == 1
    assert list(loads[0].dim_strides) == [1, 1], f"expected dim_strides=(1, 1), got {loads[0].dim_strides}"
    assert list(loads[0].replicate_factor_per_dim) == [
        2, 1
    ], f"a[jk // 2, jc] must replicate factor 2 on jk only, got {loads[0].replicate_factor_per_dim}"


def test_fully_unstructured_separable_descent_to_tile_only():
    """``a[idx_k[jk], idx_j[jc]]``: both source dims must appear in gather_dims."""
    sdfg = _fully_unstructured_separable.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    gathers = [n for n in _tile_loads(sdfg) if n.gather_dims]
    assert len(gathers) == 1, f"expected exactly one gather TileLoad, got {gathers}"
    assert list(gathers[0].gather_dims) == [0, 1], f"expected gather_dims=(0, 1), got {gathers[0].gather_dims}"


def test_fully_unstructured_2d_index_descent_to_tile_only():
    """``a[idx[jk, jc]]``: a single (8, 8) index TileLoad feeds the gather, not two 1-D tiles."""
    sdfg = _fully_unstructured_2d_index.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    loads = _tile_loads(sdfg)
    gathers = [n for n in loads if n.gather_dims]
    idx_loads = [n for n in loads if not n.gather_dims]
    assert len(gathers) == 1, f"expected exactly one gather TileLoad, got {gathers}"
    assert list(gathers[0].gather_dims) == [0], f"expected gather_dims=(0,), got {gathers[0].gather_dims}"
    assert len(idx_loads) == 1 and list(
        idx_loads[0].widths) == [8, 8], f"a single 2-D index tile must feed the gather, got {idx_loads}"


def test_all_kdim_broadcast_shapes_match_numpy():
    """Compile-and-run every K=2 broadcast/gather shape against a plain-numpy reference."""
    # One test, not eleven: each shape needs its own native compile + run (expensive), the
    # grouped-Act exception for tests that would otherwise repeat a costly setup per case.
    n, m = 8, 8
    rng = np.random.default_rng(0)

    def run(prog, **arrays) -> np.ndarray:
        sdfg = prog.to_sdfg()
        _vectorize_k2(sdfg)
        csdfg = sdfg.compile()
        c = np.zeros((n, m), dtype=np.float64)
        csdfg(NK=n, NJ=m, c=c, **arrays)
        return c

    a0 = rng.random(1)
    np.testing.assert_allclose(run(_scalar_broadcast, a=a0), np.broadcast_to(a0[0], (n, m)))

    a_col = rng.random(n)
    np.testing.assert_allclose(run(_col_broadcast, a=a_col), np.broadcast_to(a_col[:, None], (n, m)))

    a_row = rng.random(m)
    np.testing.assert_allclose(run(_row_broadcast, a=a_row), np.broadcast_to(a_row[None, :], (n, m)))

    a_2d = rng.random((n, m))
    np.testing.assert_allclose(run(_full_2d_baseline, a=a_2d), a_2d)

    idx_k = rng.integers(0, n, size=n).astype(np.int32)
    np.testing.assert_allclose(run(_col_gather, a=a_col, idx=idx_k), np.broadcast_to(a_col[idx_k][:, None], (n, m)))

    np.testing.assert_allclose(run(_col_structured, a=a_col), np.broadcast_to(a_col[np.arange(n) // 2][:, None],
                                                                              (n, m)))

    idx_j = rng.integers(0, m, size=m).astype(np.int32)
    np.testing.assert_allclose(run(_row_gather, a=a_row, idx=idx_j), np.broadcast_to(a_row[idx_j][None, :], (n, m)))

    np.testing.assert_allclose(run(_row_structured, a=a_row), np.broadcast_to(a_row[np.arange(m) // 2][None, :],
                                                                              (n, m)))

    np.testing.assert_allclose(run(_fully_structured_2d, a=a_2d), a_2d[np.arange(n) // 2, :])

    idx_ik = rng.integers(0, n, size=n).astype(np.int32)
    idx_ij = rng.integers(0, m, size=m).astype(np.int32)
    np.testing.assert_allclose(run(_fully_unstructured_separable, a=a_2d, idx_k=idx_ik, idx_j=idx_ij),
                               a_2d[np.ix_(idx_ik, idx_ij)])

    idx_2d = rng.integers(0, n, size=(n, m)).astype(np.int32)
    np.testing.assert_allclose(run(_fully_unstructured_2d_index, a=a_col, idx=idx_2d), a_col[idx_2d])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
