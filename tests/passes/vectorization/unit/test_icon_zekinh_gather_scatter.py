# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON ``zekinh``-style COMBINED gather + scatter — K-dim tile-only contract.

Combines the patterns from :file:`test_icon_zekinh_gather.py` and
:file:`test_icon_zekinh_scatter.py`: the source reads through one
gather (``src[g_blk[jb,jc], jk, g_idx[jb,jc]]``) and the destination
writes through a *different* scatter (``dst[s_blk[jb,jc], jk,
s_idx[jb,jc]] = …``). Both directions have two data-dependent dims
(0 and 2) and a tile-var-bound middle dim (1 = jk).

The K-dim descent (``PromoteNSDFGBodyToTiles`` + ``EmitTileOps``) at
K=2 (``widths=(8, 8)``) must lower the body to zero raw Tasklet nodes
and emit at least one ``TileLoad`` (gather) AND at least one ``TileStore`` (scatter).
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import dace
import pytest

from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import _is_assign_tasklet
from dace.transformation.passes.vectorization.utils.tasklets import tasklet_reads_or_writes_tile
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")

_WIDTHS = (8, 8)


@dace.program
def _icon_zekinh_gather_scatter(
    coeff: dace.float64[NB, NPROMA],
    g_idx: dace.int32[NB, NPROMA],
    g_blk: dace.int32[NB, NPROMA],
    s_idx: dace.int32[NB, NPROMA],
    s_blk: dace.int32[NB, NPROMA],
    src: dace.float64[NB, NLEV, NPROMA],
    dst: dace.float64[NB, NLEV, NPROMA],
):
    """Combined-direction zekinh: gather from ``src`` via ``g_*``, scatter into
    ``dst`` via the (different) ``s_*`` index arrays."""
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                dst[s_blk[jb, jc], jk, s_idx[jb, jc]] = coeff[jb, jc] * src[g_blk[jb, jc], jk, g_idx[jb, jc]]


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Count raw tasklets that still touch TILE-shaped data after the descent.

    The K-dim tile-only invariant is "tile-shaped values flow only through tile
    lib nodes" -- so a tasklet is unlowered residue iff it reads/writes a tile
    (full ``widths`` or a ``ONE``-broadcast tile). Excluded as legitimate:

    * Trivial ``_out = _in`` assign tasklets (``_is_assign_tasklet``) -- the
      descent leaves these as one-element copies (collapsing into AN -> AN
      would drop source-side coordinates, per user directive).
    * ``tile_runtime_*`` trip-guard tasklets (SYMBOLIC-dim ``__builtin_trap``
      control, not a compute residual).
    * Scalar ``__tile_k1_tail`` remainder tasklets operating purely on ``(1,)``
      scalars -- scalar-load -> scalar chains stay python scalar tasklets (user
      direction 2026-06-15); they touch no tile, so are not counted.
    """
    return sum(1 for n, parent in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.Tasklet) and not _is_assign_tasklet(n)
               and not n.label.startswith("tile_runtime") and tasklet_reads_or_writes_tile(parent, n, _WIDTHS))


def _count_lib(sdfg: dace.SDFG, cls) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls))


def test_icon_zekinh_gather_scatter_descent_to_tile_only():
    """Mixed gather + scatter ICON-style kernel lowers to zero raw Tasklets at K=2."""
    sdfg = _icon_zekinh_gather_scatter.to_sdfg()
    sdfg.validate()

    VectorizeCPUMultiDim(
        widths=(8, 8),
        target_isa="SCALAR",
        remainder_strategy="scalar_postamble",
        branch_mode="merge",
        loop_to_map_permissive=True,
        nest_map_bodies=True,
        scalar_remainder_emit="tile_k1",
        expand_tile_nodes=False,
    ).apply_pass(sdfg, {})
    sdfg.validate()

    n_tasklet = _count_tasklets(sdfg)
    n_gather = _count_lib(sdfg, TileLoad)
    n_scatter = _count_lib(sdfg, TileStore)
    assert n_tasklet == 0, (f"icon_zekinh_gather_scatter must lower to tile lib nodes only at the K-dim "
                            f"layer; got {n_tasklet} raw Tasklet nodes after the descent.")
    assert n_gather >= 1, (f"The mixed-gather source must yield at least one TileLoad (gather); got {n_gather}.")
    assert n_scatter >= 1, (f"The mixed-scatter destination must yield at least one TileStore (scatter); "
                            f"got {n_scatter}.")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
