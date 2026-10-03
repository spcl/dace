# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``assert_tiled`` -- the guard that lets a value-only vectorization test fail on a refusal."""
from __future__ import annotations

import dace
from dace.ordered import OrderedSet
from dace.libraries.tileops import (MaskedCopyLibraryNode, TileBinop, TileFMA, TileIota, TileITE, TileGather,
                                    TileMaskGen, TileMMA, TileReduce, TileScatter, TileUnop)

# Spelled out, not imported from the pass: the assertion audits production code, not restates it.
TILE_NODE_TYPES = (MaskedCopyLibraryNode, TileBinop, TileFMA, TileIota, TileITE, TileGather, TileMaskGen, TileMMA,
                   TileReduce, TileScatter, TileUnop)

REFUSAL_HINT = ("VectorizeMultiDim.apply_pass catches VectorizeUnsupported, calls warnings.warn and "
                "restore_sdfg_in_place, then returns None -- a total refusal hands back the pristine "
                "un-tiled input, so every value-only comparison below would pass by comparing the "
                "reference to itself. Re-run with -W error::UserWarning to read the refusal reason.")


def tile_library_nodes(sdfg: dace.SDFG) -> list[dace.nodes.LibraryNode]:
    """Tile lib nodes anywhere in ``sdfg``, nested SDFGs included.

    :param sdfg: SDFG to scan.
    :returns: Every ``MaskedCopyLibraryNode`` / ``TileBinop`` / ``TileScatter`` / ... node reachable from ``sdfg``.
    """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TILE_NODE_TYPES)]


def masked_loads(state: dace.SDFGState) -> list[MaskedCopyLibraryNode]:
    """The masked copies of ``state`` that fill a tile from an array."""
    return [n for n in state.nodes() if isinstance(n, MaskedCopyLibraryNode) and not n.stores(state)]


def masked_stores(state: dace.SDFGState) -> list[MaskedCopyLibraryNode]:
    """The masked copies of ``state`` that write a tile into an array."""
    return [n for n in state.nodes() if isinstance(n, MaskedCopyLibraryNode) and n.stores(state)]


def sdfg_masked_loads(sdfg: dace.SDFG) -> list[MaskedCopyLibraryNode]:
    """The masked loads anywhere in ``sdfg``, nested SDFGs included."""
    return [
        n for n, state in sdfg.all_nodes_recursive() if isinstance(n, MaskedCopyLibraryNode) and not n.stores(state)
    ]


def sdfg_masked_stores(sdfg: dace.SDFG) -> list[MaskedCopyLibraryNode]:
    """The masked stores anywhere in ``sdfg``, nested SDFGs included."""
    return [n for n, state in sdfg.all_nodes_recursive() if isinstance(n, MaskedCopyLibraryNode) and n.stores(state)]


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


def assert_tiled_unless_pinned(vectorized: dace.SDFG, untransformed: dace.SDFG, kernel: str,
                               untiled: frozenset[str]) -> None:
    """Assert a corpus kernel tiled, or -- if ``kernel`` is pinned in ``untiled`` -- that it still did not.

    Pinned in both directions, so a corpus comparison can no longer pass on a refusal, and a kernel the
    vectorizer starts tiling is a failure until it leaves the pinned set.

    :param vectorized: SDFG the pass ran on -- read after ``apply_pass``, before ``compile()`` expands it.
    :param untransformed: The same kernel with the pass NOT run.
    :param kernel: Corpus kernel name, as the pinned set spells it.
    :param untiled: Kernels measured to come back with no tile lib node.
    """
    if kernel not in untiled:
        assert_tiled(vectorized, untransformed, kernel)
        return
    emitted = tile_library_nodes(vectorized)
    assert not emitted, (f"{kernel}: pinned as un-tiled, but the vectorizer now emits {len(emitted)} tile lib "
                         f"node(s). Remove it from the pinned set.")
