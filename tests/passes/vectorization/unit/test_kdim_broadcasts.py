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
   gather, broadcast across ``jc`` — lowers to a ``TileGather`` whose
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
import dace
import pytest

from dace.libraries.tileops import TileGather, TileLoad, TileStore
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

NK = dace.symbol("NK")
NJ = dace.symbol("NJ")


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Number of raw ``Tasklet`` nodes anywhere in ``sdfg``."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def _count_lib_nodes_by_type(sdfg: dace.SDFG, cls) -> int:
    """Number of lib nodes of ``cls`` anywhere in ``sdfg``."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls))


def _vectorize_k2(sdfg: dace.SDFG) -> None:
    """Run the K=2 (8, 8) orchestrator, leaving tile lib nodes intact."""
    VectorizeCPUMultiDim(
        widths=(8, 8),
        target_isa="SCALAR",
        remainder_strategy="scalar_postamble",
        branch_mode="merge",
        loop_to_map_permissive=False,
        nest_map_bodies=True,
        insert_copies=True,
        fuse_overlapping_loads=False,
        scalar_remainder_emit="tile_k1",
        expand_tile_nodes=False,
    ).apply_pass(sdfg, {})


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


# ---------------------------------------------------------------- tests


def test_scalar_broadcast_descent_to_tile_only():
    """Scalar broadcast produces 0 raw tasklets + at least one TileLoad."""
    sdfg = _scalar_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim scalar-broadcast must lower to tile-only"
    assert _count_lib_nodes_by_type(sdfg, TileLoad) >= 1
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_col_broadcast_descent_to_tile_only():
    """1D-column broadcast (a[jk] across jc) produces 0 raw tasklets."""
    sdfg = _col_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim col-broadcast must lower to tile-only"
    assert _count_lib_nodes_by_type(sdfg, TileLoad) >= 1
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_row_broadcast_descent_to_tile_only():
    """1D-row broadcast (a[jc] across jk) produces 0 raw tasklets."""
    sdfg = _row_broadcast.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0, "K-dim row-broadcast must lower to tile-only"
    assert _count_lib_nodes_by_type(sdfg, TileLoad) >= 1
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_full_2d_baseline_descent_to_tile_only():
    """2D contiguous load baseline: 0 raw tasklets + TileLoad + TileStore."""
    sdfg = _full_2d_baseline.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    assert _count_lib_nodes_by_type(sdfg, TileLoad) >= 1
    assert _count_lib_nodes_by_type(sdfg, TileStore) >= 1


def test_col_gather_descent_to_tile_only():
    """Per-row data-dep gather (a[idx[jk]]) broadcast across jc.

    Lowers to a ``TileGather`` (data-dep index tile) whose result is
    then broadcast across ``jc``. No raw Tasklets at the K-dim layer.
    """
    sdfg = _col_gather.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    assert _count_lib_nodes_by_type(sdfg, TileGather) + _count_lib_nodes_by_type(sdfg, TileLoad) >= 1


def test_col_structured_descent_to_tile_only():
    """Per-row structured ``a[jk // 2]`` broadcast across jc."""
    sdfg = _col_structured.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    assert _count_lib_nodes_by_type(sdfg, TileGather) + _count_lib_nodes_by_type(sdfg, TileLoad) >= 1


def test_row_gather_descent_to_tile_only():
    """Per-column data-dep gather (a[idx[jc]]) broadcast across jk."""
    sdfg = _row_gather.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    assert _count_lib_nodes_by_type(sdfg, TileGather) + _count_lib_nodes_by_type(sdfg, TileLoad) >= 1


def test_row_structured_descent_to_tile_only():
    """Per-column structured ``a[jc // 2]`` broadcast across jk."""
    sdfg = _row_structured.to_sdfg()
    sdfg.validate()
    _vectorize_k2(sdfg)
    sdfg.validate()
    assert _count_tasklets(sdfg) == 0
    assert _count_lib_nodes_by_type(sdfg, TileGather) + _count_lib_nodes_by_type(sdfg, TileLoad) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
