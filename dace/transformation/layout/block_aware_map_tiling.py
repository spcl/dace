# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BlockAwareMapTiling -- tile the schedule to a block factor so Block lowers cleanly.

Tiles top-level maps matching ``tile_sizes`` so SplitDimensions emits clean tile/offset indices, no residual %/int_floor.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import sympy

import dace
from dace import symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling


def provably_indivisible(map_entry: dace.nodes.MapEntry, tile_sizes: Tuple[int, ...]) -> bool:
    """True iff some tiled dim's extent is a known constant not divisible by its tile; symbolic extents return False."""
    for (begin, end, _), tile in zip(map_entry.map.range, tile_sizes):
        extent = symbolic.simplify(symbolic.pystr_to_symbolic(end) - symbolic.pystr_to_symbolic(begin) + 1)
        remainder = symbolic.simplify(sympy.Mod(extent, tile))
        if remainder.is_number and remainder != 0:
            return True
    return False


@dataclass
class BlockAwareMapTiling(ppl.Pass):
    """Tile top-level maps by ``tile_sizes`` so a subsequent Block aligns with the schedule."""

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
                # Never assert divides_evenly on a map that provably does not.
                divides = self._divides_evenly and not provably_indivisible(me, self._tile_sizes)
                MapTiling.apply_to(sdfg,
                                   options={
                                       'tile_sizes': self._tile_sizes,
                                       'divides_evenly': divides
                                   },
                                   map_entry=me)
                count += 1
        return count
