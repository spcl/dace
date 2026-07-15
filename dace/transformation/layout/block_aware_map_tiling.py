# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BlockAwareMapTiling -- tile the schedule to a block factor so Block lowers cleanly.

Blocking an array by factor ``b`` rewrites an access ``A[i]`` to ``A[i//b, i%b]`` (emitting
``int_floor`` / ``%``). If the map iterating that dimension is TILED by ``b`` -- an outer loop
``tile: 0:N:b`` and an inner loop ``i: tile:tile+b`` -- then ``i//b`` is the tile index and
``i%b`` is the inner offset, so ``SplitDimensions`` takes its perfect-block-match path and emits
clean tile/offset indices with no residual ``%``/``int_floor``.

This pass performs that tiling: it applies ``MapTiling`` to the top-level maps whose dimensionality
matches ``tile_sizes``. Run it BEFORE ``SplitDimensions`` (block), or from the post-layout schedule
normalization, so the layout's index arithmetic aligns with the schedule.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling


@dataclass
class BlockAwareMapTiling(ppl.Pass):
    """Tile top-level maps by ``tile_sizes`` so a subsequent Block aligns with the schedule.

    :param tile_sizes: per-map-dimension tile size (the block factors). Applied to every top-level
                       map whose parameter count equals ``len(tile_sizes)``.
    :param divides_evenly: pass through to ``MapTiling`` -- ``True`` when the extents are known to be
                           multiples of the tile sizes (e.g. after ``PadDimensions``), giving a clean
                           ``0:N:b`` outer range with no remainder tile.
    """

    def __init__(self, tile_sizes: Tuple[int, ...], divides_evenly: bool = False):
        self._tile_sizes = tuple(tile_sizes)
        self._divides_evenly = divides_evenly

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        count = 0
        for state in sdfg.all_states():
            scope = state.scope_dict()
            top_maps = [
                n for n in state.nodes()
                if isinstance(n, dace.nodes.MapEntry) and scope[n] is None
                and len(n.map.params) == len(self._tile_sizes)
            ]
            for me in top_maps:
                MapTiling.apply_to(sdfg,
                                   options={
                                       'tile_sizes': self._tile_sizes,
                                       'divides_evenly': self._divides_evenly
                                   },
                                   map_entry=me)
                count += 1
        return count
