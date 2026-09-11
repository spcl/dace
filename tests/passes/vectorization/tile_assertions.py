# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``assert_tiled`` -- the guard that lets a value-only vectorization test fail on a refusal."""
from __future__ import annotations

import dace
from dace.ordered import OrderedSet
from dace.libraries.tileops import (TileBinop, TileFMA, TileIota, TileITE, TileLoad, TileMaskGen, TileMMA, TileReduce,
                                    TileStore, TileUnop)

# Spelled out, not imported from the pass: the assertion audits production code, not restates it.
TILE_NODE_TYPES = (TileBinop, TileFMA, TileIota, TileITE, TileLoad, TileMaskGen, TileMMA, TileReduce, TileStore,
                   TileUnop)

REFUSAL_HINT = ("VectorizeMultiDim.apply_pass catches VectorizeUnsupported, calls warnings.warn and "
                "restore_sdfg_in_place, then returns None -- a total refusal hands back the pristine "
                "un-tiled input, so every value-only comparison below would pass by comparing the "
                "reference to itself. Re-run with -W error::UserWarning to read the refusal reason.")


def tile_library_nodes(sdfg: dace.SDFG) -> list[dace.nodes.LibraryNode]:
    """Tile lib nodes anywhere in ``sdfg``, nested SDFGs included.

    :param sdfg: SDFG to scan.
    :returns: Every ``TileLoad`` / ``TileBinop`` / ``TileStore`` / ... node reachable from ``sdfg``.
    """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TILE_NODE_TYPES)]


def assert_tiled(vectorized: dace.SDFG, untransformed: dace.SDFG, what: str = "") -> None:
    """Assert the vectorizer tiled ``vectorized``, with ``untransformed`` as the empty-bracket control.

    :param vectorized: SDFG the pass ran on -- read after ``apply_pass``, before ``compile()`` expands it.
    :param untransformed: The same kernel with the pass NOT run -- the control that must read zero.
    :param what: Optional tag naming the case, for the failure message.
    """
    tag = f"{what}: " if what else ""
    control = tile_library_nodes(untransformed)
    assert not control, (
        f"{tag}empty-bracket control failed: untransformed {untransformed.name!r} already "
        f"holds {len(control)} tile lib node(s) {list(OrderedSet(type(n).__name__ for n in control))}. "
        f"This counter cannot read zero, so a non-empty count proves nothing.")
    emitted = tile_library_nodes(vectorized)
    assert emitted, f"{tag}the vectorizer emitted ZERO tile lib nodes into {vectorized.name!r}. {REFUSAL_HINT}"
