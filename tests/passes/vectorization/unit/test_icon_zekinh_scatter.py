# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON ``zekinh``-style SCATTER mirror — K-dim tile-only contract.

Symmetric to :file:`test_icon_zekinh_gather.py`: the destination is
``dst[edge_blk[jb, jc], jk, edge_idx[jb, jc]]`` with two data-dependent
scatter dims (0 and 2) and a tile-var-bound middle dim (1 = jk). The
source is fully bound at ``src[jb, jk, jc]``.

The K-dim descent (``PromoteNSDFGBodyToTiles`` + ``EmitTileOps``) at
K=2 (``widths=(8, 8)``) must lower the body to zero raw Tasklet nodes
and emit at least one :class:`TileStore` (scatter).
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import dace
import pytest

from dace.libraries.tileops import TileStore
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import _is_assign_tasklet
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, ISA, RemainderStrategy
from tests.passes.vectorization.helpers.tile_probe import tasklet_reads_or_writes_tile
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")

_WIDTHS = (8, 8)


@dace.program
def _icon_zekinh_scatter(
    e_bln: dace.float64[NB, NPROMA],
    edge_idx: dace.int32[NB, NPROMA],
    edge_blk: dace.int32[NB, NPROMA],
    src: dace.float64[NB, NLEV, NPROMA],
    dst: dace.float64[NB, NLEV, NPROMA],
):
    """Write-side mirror of zekinh: scatter ``src`` into a data-indexed destination."""
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                dst[edge_blk[jb, jc], jk, edge_idx[jb, jc]] = e_bln[jb, jc] * src[jb, jk, jc]


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Count raw tasklets that still touch TILE-shaped data after the descent.

    The K-dim tile-only invariant is "tile-shaped values flow only through tile
    lib nodes" -- so a tasklet is unlowered residue iff it reads/writes a tile
    (full ``widths`` or a ``ONE``-broadcast tile). Trivial assigns,
    ``tile_runtime_*`` trip guards, and the scalar ``__tile_k1_tail`` remainder
    (scalar-load -> scalar python tasklets, user direction 2026-06-15) touch no
    tile and are not counted.
    """
    return sum(1 for n, parent in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.Tasklet) and not _is_assign_tasklet(n)
               and not n.label.startswith("tile_runtime") and tasklet_reads_or_writes_tile(parent, n, _WIDTHS))


def _count_tile_scatters(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if (isinstance(n, TileStore) and tuple(n.gather_dims)))


def test_icon_zekinh_scatter_descent_to_tile_only():
    """Mixed-scatter ICON-style kernel lowers to zero raw Tasklets at K=2."""
    sdfg = _icon_zekinh_scatter.to_sdfg()
    sdfg.validate()

    VectorizeCPUMultiDim(
        VectorizeConfig(
            widths=(8, 8),
            target_isa=ISA.SCALAR,
            remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
            branch_mode=BranchMode.MERGE,
            loop_to_map_permissive=True,
            scalar_remainder_emit="tile_k1",
            expand_tile_nodes=False,
        )).apply_pass(sdfg, {})
    sdfg.validate()

    n_tasklet = _count_tasklets(sdfg)
    n_scatter = _count_tile_scatters(sdfg)
    assert n_tasklet == 0, (f"icon_zekinh_scatter must lower to tile lib nodes only at the K-dim layer; "
                            f"got {n_tasklet} raw Tasklet nodes after the descent.")
    assert n_scatter >= 1, (
        f"The mixed-scatter destination must yield at least one TileStore (scatter); got {n_scatter}.")

    # The scatter guard is HOISTED out of the loop's own single-block wrapper: leaving it there
    # makes that wrapper multi-state, a multi-state nested sdfg inside a map cannot be inlined, and
    # the surrounding nest then never collapses to the two params K=2 needs. Hoisting it costs the
    # key a dimension per map scope crossed, which is what this checks -- a key sized by the inner
    # trip alone would mean the guard went back inside.
    keys = [(name, desc) for sd in sdfg.all_sdfgs_recursive() for name, desc in sd.arrays.items()
            if name.startswith('_scatter_joint_key')]
    assert len(keys) == 1, f'expected the one joint key array; got {[n for n, _ in keys]}'
    # Symbols resolve out of the shape itself: re-minting one from its bare name gives a
    # different dtype instance, and the subtraction then refuses to fold.
    extent = keys[0][1].shape[0]
    syms = {str(sym): sym for sym in extent.free_symbols}
    assert dace.symbolic.simplify(extent - syms['NLEV'] * syms['NPROMA']) == 0, (
        f'the hoisted key must cover the crossed NLEV map as well as the NPROMA loop; got {extent}')


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
