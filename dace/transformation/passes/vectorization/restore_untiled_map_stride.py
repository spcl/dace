# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RestoreUntiledMapStride`` -- give back step 1 to a map the tile emitters never tiled.

:class:`~dace.transformation.passes.vectorization.stride_map_by_tile_widths.StrideMapByTileWidths`
sets ``step = W`` on every map the shared candidate gate admits, and the emitters that turn the
body into ``W`` lanes select through a SECOND predicate of their own: the scope must be exactly one
body ``NestedSDFG``. Whenever the two disagree the map keeps ``step = W`` over a body that still
computes ONE element, so ``W - 1`` of every ``W`` iterations are simply never executed -- a silent
wrong answer, not a slow one.

That disagreement is reachable: every body predicate behind the gate reads
``SDFGState.all_nodes_between``, which returns an EMPTY set as soon as the body holds a node with
no out-edge (``dace/sdfg/graph.py:441``) -- routine for a dead-end transient. Read as "the body",
the empty set makes the gate answer vacuously and makes
:class:`~dace.transformation.passes.vectorization.nest_innermost_map_body.NestInnermostMapBodyIntoNSDFG`
skip the map as "empty", so the body NSDFG the emitters require is never built. CloudSC hits it: at
width 8 one kernel region came out strided with a scalar body and computed one column in eight.

Restoring the step is EXACT. The map's body was never rewritten, and a remainder split preserves
the union of the two ranges, so a step-1 interior plus a step-1 tail iterate precisely the original
range. A map that WAS tiled keeps everything: it is recognised by the tile library nodes in its
scope.

Fail-closed on the third case -- no tile op but a body that was already widened. Un-striding a
widened body would run per-lane buffers under a step-1 map, so the kernel is refused with
:class:`~dace.transformation.passes.vectorization.utils.errors.VectorizeUnsupported` and the
orchestrator hands the caller back their pristine input.
"""
from typing import Any

import dace
from dace import properties, subsets, symbolic
from dace.libraries.tileops.nodes import (TileBinop, TileFMA, TileIota, TileITE, TileLoad, TileMaskGen, TileMMA,
                                          TileReduce, TileStore, TileUnop)
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.errors import VectorizeUnsupported

#: Every tile library node the emit stage can leave behind. A map whose scope holds one of these
#: WAS tiled, whatever any predicate would say about its body now.
TILE_NODES = (TileBinop, TileFMA, TileIota, TileITE, TileLoad, TileMaskGen, TileMMA, TileReduce, TileStore, TileUnop)


@properties.make_properties
class RestoreUntiledMapStride(ppl.Pass):
    """Undo the tile stride on every map that came out of the emit stage untiled.

    Runs after the emitters, so "tiled" is answered by what is actually in the graph -- a tile
    library node in the map's scope -- rather than by re-deriving a predicate over a body the
    emitters have been rewriting.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: tuple[int, ...] = (8, )) -> None:
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"RestoreUntiledMapStride: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Rewrites map ranges.

        :returns: ``ppl.Modifies.Scopes``.
        """
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent -- the maps it repairs no longer match.

        :param modified: Earlier passes' modifications (unused).
        :returns: ``False``.
        """
        return False

    def steps_match_widths(self, map_entry: MapEntry) -> bool:
        """Do the last-K dims of ``map_entry`` step by ``widths``?

        :param map_entry: The map to inspect.
        :returns: ``True`` when every one of the K innermost steps equals its width.
        """
        ranges = list(map_entry.map.range.ranges)
        widths = tuple(self.widths)
        if len(ranges) < len(widths):
            return False
        return all(str(step) == str(width) for (_lb, _ub, step), width in zip(ranges[-len(widths):], widths))

    @staticmethod
    def scope_nodes(state: dace.SDFGState, map_entry: MapEntry) -> list[dace.nodes.Node]:
        """Every node in the map's scope, entry and exit excluded.

        Deliberately NOT ``all_nodes_between``: that answers with an empty set whenever the body
        holds a dead-end node, which is the very shape this pass exists to repair.

        :param state: The state holding ``map_entry``.
        :param map_entry: The map whose scope is wanted.
        :returns: The scope's nodes.
        """
        return list(state.scope_subgraph(map_entry, include_entry=False, include_exit=False).nodes())

    def body_was_widened(self, scope: list[dace.nodes.Node]) -> bool:
        """Did the widener already give this body per-lane buffers?

        A widened body carries the iteration mask, or a transient whose innermost extent is the
        tile width. Either one means the step is load-bearing and must not be taken away.

        :param scope: The map scope's nodes.
        :returns: ``True`` when a per-lane artifact is present.
        """
        width = self.widths[-1]
        for node in scope:
            if isinstance(node, TileMaskGen):
                return True
            if not isinstance(node, dace.nodes.NestedSDFG):
                continue
            for desc in node.sdfg.arrays.values():
                if len(desc.shape) >= 1 and str(desc.shape[-1]) == str(width):
                    return True
        return False

    def restore_step(self, map_entry: MapEntry) -> None:
        """Put the K innermost steps of ``map_entry`` back to 1.

        :param map_entry: The map to repair.
        """
        ranges = list(map_entry.map.range.ranges)
        K = len(self.widths)
        repaired = [(lb, ub, symbolic.SymExpr(1)) for (lb, ub, _step) in ranges[-K:]]
        map_entry.map.range = subsets.Range(ranges[:-K] + repaired)

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Repair every W-strided map the emitters left untiled.

        :param sdfg: SDFG to transform in place.
        :param _: Pipeline-results placeholder (unused).
        :returns: Number of maps repaired, or ``None`` when none needed it.
        :raises VectorizeUnsupported: When an untiled map's body was already widened, so the
            stride cannot be taken away and the kernel has to be refused whole.
        """
        repaired = 0
        for node, graph in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, MapEntry) or not isinstance(graph, dace.SDFGState):
                continue
            if not self.steps_match_widths(node):
                continue
            scope = self.scope_nodes(graph, node)
            if any(isinstance(n, TILE_NODES) for n in scope):
                continue
            if any(
                    isinstance(inner, TILE_NODES) for n in scope if isinstance(n, dace.nodes.NestedSDFG)
                    for inner, _ in n.sdfg.all_nodes_recursive()):
                continue
            if self.body_was_widened(scope):
                raise VectorizeUnsupported(f"map {node.map.label!r} steps by {tuple(self.widths)} over a body the "
                                           f"emitters widened but never lowered to tile ops; the stride cannot be "
                                           f"restored, so this kernel is refused")
            self.restore_step(node)
            repaired += 1
        return repaired or None
