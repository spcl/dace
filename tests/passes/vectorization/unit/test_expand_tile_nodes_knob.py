# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeCPUMultiDim(expand_tile_nodes=...)`` knob coverage.

The orchestrator's default (``expand_tile_nodes=False``) leaves every emitted
``Tile*`` library node present so the caller can inspect / save / further-transform
the lib-node shape and lower it later (``sdfg.expand_library_nodes()`` / ``compile()``).
``expand_tile_nodes=True`` instead lowers every tile node to its per-ISA body before
returning, so the SDFG is ready to compile immediately.

This test pins both modes on a 2-D axpy kernel:

- ``expand_tile_nodes=False`` (default): at least one ``Tile*`` lib node remains.
- ``expand_tile_nodes=True``: no ``Tile*`` lib node survives.

The kernel exercises a divisible 2-D map so the run finishes cleanly
in both modes (no postamble interaction needed for this knob).
"""

import dace
import pytest

from dace.libraries.tileops import TileBinop, TileLoad, TileMaskGen, TileReduce, TileStore, TileUnop
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )

_TILE_LIB_NODE_TYPES = (
    TileBinop,
    TileLoad,
    TileMaskGen,
    TileReduce,
    TileStore,
    TileUnop,
)

KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")


@dace.program
def _axpy_2d(a: dace.float64[KLEV, KLON], b: dace.float64[KLEV, KLON], c: dace.float64[KLEV, KLON]):
    for jk in range(KLEV):
        for jl in range(KLON):
            c[jk, jl] = a[jk, jl] + b[jk, jl] * 2.0


def _count_tile_lib_nodes(sdfg: dace.SDFG) -> int:
    """Total number of ``Tile*`` library nodes anywhere in ``sdfg``."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, _TILE_LIB_NODE_TYPES))


def _vectorize(sdfg: dace.SDFG, *, expand_tile_nodes: bool) -> None:
    VectorizeCPUMultiDim(
        VectorizeConfig(
            widths=(8, 8),
            target_isa=ISA.SCALAR,
            remainder_strategy=RemainderStrategy.FULL_MASK,
            branch_mode=BranchMode.MERGE,
            expand_tile_nodes=expand_tile_nodes,
        )).apply_pass(sdfg, {})


def test_expand_tile_nodes_true_leaves_no_tile_lib_nodes():
    """``expand_tile_nodes=True`` lowers every ``Tile*`` lib node so the
    returned SDFG is ready to compile (the default leaves them present)."""
    sdfg = _axpy_2d.to_sdfg()
    sdfg.name = "axpy_2d_expanded"
    sdfg.validate()
    _vectorize(sdfg, expand_tile_nodes=True)
    sdfg.validate()
    assert _count_tile_lib_nodes(sdfg) == 0, (f"with expand_tile_nodes=True every Tile* lib node must be lowered; "
                                              f"found {_count_tile_lib_nodes(sdfg)}")


def test_expand_tile_nodes_false_preserves_tile_lib_nodes():
    """``expand_tile_nodes=False`` leaves the tile lib nodes intact so the
    caller can inspect / save / further-transform the lib-node shape."""
    sdfg = _axpy_2d.to_sdfg()
    sdfg.name = "axpy_2d_deferred"
    sdfg.validate()
    _vectorize(sdfg, expand_tile_nodes=False)
    sdfg.validate()
    n_tile = _count_tile_lib_nodes(sdfg)
    assert n_tile > 0, (f"with expand_tile_nodes=False at least one Tile* lib node must "
                        f"survive; found {n_tile}")

    # Explicit expansion by the caller still works and reduces the count to 0.
    sdfg.expand_library_nodes()
    sdfg.validate()
    assert _count_tile_lib_nodes(sdfg) == 0, ("manual sdfg.expand_library_nodes() after deferred return must "
                                              "lower every Tile* lib node")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
