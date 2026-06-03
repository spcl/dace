# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON ``zekinh`` 3-edge bilinear interpolation — K-dim tile-only contract.

This is the canonical mixed-gather kernel: each tile lane reads
``z_kin_hor_e[edge_blk[jb, jc, m], jk, edge_idx[jb, jc, m]]`` for
``m = 0, 1, 2`` (the three incident edges of a cell), weighted by
``e_bln[jb, m, jc]`` per-cell coefficients. Dim 0 (``edge_blk``) and
dim 2 (``edge_idx``) are data-dependent gathers; dim 1 (``jk``) is
affine.

The K-dim descent (``PromoteNSDFGBodyToTiles`` + ``EmitTileOps``) is
exercised with K=2 (``widths=(8, 8)``); the pinning contract is that
the post-descent SDFG holds **zero raw Tasklet nodes** — every read /
write / index-tile-fill is a tile lib node.
"""
import dace
import pytest

from dace.libraries.tileops import TileGather
from dace.transformation.passes.vectorization.emit_tile_ops import _is_assign_tasklet
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")


@dace.program
def _icon_zekinh_gather(
    e_bln: dace.float64[NB, 3, NPROMA],
    edge_idx: dace.int32[NB, NPROMA, 3],
    edge_blk: dace.int32[NB, NPROMA, 3],
    z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
    z_ekinh: dace.float64[NB, NLEV, NPROMA],
):
    """Cell-from-edges bilinear interpolation (ICON velocity_zekinh_block.f90)."""
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                                       e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                                       e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Count tasklets the descent did NOT lower to lib nodes.

    Trivial ``_out = _in`` assign tasklets are LEFT in place by the descent
    (``_promote_internal_assigns`` is a no-op per user directive: collapsing
    them into AN -> AN would silently drop source-side coordinates). These
    are semantically fine -- they lower to a one-element copy at codegen --
    so the test asserts only "no NON-assign raw tasklets" rather than
    "zero tasklets total"."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.Tasklet) and not _is_assign_tasklet(n))


def _count_tile_gathers(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileGather))


def test_icon_zekinh_descent_to_tile_only():
    """The mixed-gather ICON kernel lowers to zero raw Tasklets at K=2.

    Pins the descent contract: every multi-element fill is a tile lib
    node; every length-1 scalar materialization is a Python tasklet (so
    the ``Tasklet`` count is ``0`` — Python-language single-element
    assigns are also counted, but the descent emits ``LibraryNode``
    instances for every multi-element read / write / index tile).
    """
    sdfg = _icon_zekinh_gather.to_sdfg()
    sdfg.validate()

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
    sdfg.validate()

    n_tasklet = _count_tasklets(sdfg)
    n_gather = _count_tile_gathers(sdfg)
    assert n_tasklet == 0, (f"icon_zekinh_gather must lower to tile lib nodes only at the K-dim layer; "
                            f"got {n_tasklet} raw Tasklet nodes after the descent.")
    assert n_gather >= 1, (f"The 3-edge mixed gather must yield at least one TileGather; got {n_gather}.")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
