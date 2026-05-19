# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tile each innermost map into ``num_cores`` contiguous core blocks.

First slice of the SVE-style lowering: the data-parallel innermost map
is tiled into an outer ``core`` block-distribution map plus an inner
per-core contiguous chunk. The chunk is what the existing masked
vectorize path and the later for-loop / while rewrite operate on.

The contiguous partition is produced via :class:`MapTiling` with a
per-core block size ``ceil(trip / num_cores)``. ``MapTiling`` delegates
to ``StripMining``'s *Normal* path (the heavily-used,
non-divisible-correct tiling with proper ``min``-clamped inner bounds),
so the partition has no overlap and stays in bounds for any trip count.
The ``StripMining`` ``NumberOfTiles`` path is *not* used — it emits
overlapping inclusive ranges for non-divisible trips.

The pass is inert at ``num_cores <= 1`` and when the trip is provably
smaller than ``num_cores`` (cannot form that many non-empty contiguous
blocks), and idempotent (a map already enclosed by a ``core`` block
map is left alone), so wiring it with the default keeps existing
pipelines byte-identical.
"""
from typing import Optional

import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

_CORE_PREFIX = "core"


@properties.make_properties
class TileMapByNumCores(ppl.Pass):
    """Tile every innermost map into ``num_cores`` contiguous core blocks.

    After the pass each targeted innermost map is enclosed by a
    ``core``-prefixed block-distribution map; the inner map iterates a
    single contiguous chunk. Numerically equivalent to the input
    (tiling only re-nests the iteration space).
    """

    CATEGORY: str = "Vectorization Preparation"

    num_cores = properties.Property(dtype=int,
                                    default=1,
                                    desc="Number of contiguous core blocks the outer map distributes. "
                                    "``<= 1`` makes the pass a no-op.")

    def __init__(self, num_cores: int = 1):
        super().__init__()
        self.num_cores = num_cores

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _enclosed_by_core_map(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
        """Whether ``map_entry`` already sits inside a ``core`` block map.

        ``MapTiling`` puts the new distribution parameter on the *outer*
        map, so idempotency must be decided by walking the scope parents,
        not the inner map's own parameters.

        :param state: The state containing ``map_entry``.
        :param map_entry: The candidate inner map entry.
        :returns: ``True`` if an enclosing ``core``-prefixed map exists.
        """
        scope = state.scope_dict()
        node = scope.get(map_entry)
        while node is not None:
            if isinstance(node, dace.nodes.MapEntry) and any(p.startswith(_CORE_PREFIX) for p in node.map.params):
                return True
            node = scope.get(node)
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Tile each eligible innermost map into ``num_cores`` blocks.

        Only single-parameter, step-1 innermost maps are handled (the
        SVE-style chunking model is one contiguous core-distribution
        dimension). A map whose trip count is provably smaller than
        ``num_cores`` is skipped (it cannot yield that many non-empty
        contiguous blocks).

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: The number of maps tiled, or ``None`` if none.
        """
        if self.num_cores <= 1:
            return None

        tiled = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                continue
            if not is_innermost_map(g, n):
                continue
            if len(n.map.params) != 1:
                continue
            lb, ub, step = n.map.range[0]
            if (step != 1) and (str(step) != "1"):
                continue
            if self._enclosed_by_core_map(g, n):
                continue
            trip = dace.symbolic.simplify(ub - lb + 1)
            try:
                if bool((trip < self.num_cores).simplify()):
                    continue
            except Exception:
                pass
            block = dace.symbolic.int_ceil(trip, self.num_cores)
            MapTiling.apply_to(g.sdfg,
                               options={
                                   "tile_sizes": (block, ),
                                   "prefix": _CORE_PREFIX,
                                   "divides_evenly": False,
                                   "tile_trivial": True,
                               },
                               verify=True,
                               save=False,
                               map_entry=n)
            tiled += 1
        return tiled or None
