# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NormalizeScheduleForLayout -- re-tile each top-level map to the block width its operands are
laid out with, so the inner ``Mod(i, b)`` offset iterates contiguously. Run after applying a layout."""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import sympy

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.layout.block_aware_map_tiling import provably_indivisible


@dataclass
class NormalizeScheduleForLayout(ppl.Pass):
    """Tile each top-level map by the block width its operands are laid out with.

    :param divides_evenly: True when extents are known multiples of the block width (no remainder map).
    """

    def __init__(self, divides_evenly: bool = False):
        self._divides_evenly = divides_evenly

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _block_width(self, state, me) -> Optional[Dict[str, int]]:
        """Per-parameter inner block width of ``me``, or ``None`` unless every parameter has one."""
        param_syms = {p: dace.symbolic.pystr_to_symbolic(p) for p in me.map.params}
        widths: Dict[str, set] = {}
        for edge in state.scope_subgraph(me).edges():
            if edge.data is None or edge.data.subset is None:
                continue
            for begin, end, _ in edge.data.subset.ranges:
                # Only a POINT access (begin==end) is a genuine block offset; excludes propagated ranges for idempotence.
                if not isinstance(begin, sympy.Basic) or str(dace.symbolic.simplify(end - begin)) != '0':
                    continue
                for mod in begin.atoms(sympy.Mod):
                    base, modulus = mod.args
                    if not modulus.is_Integer:
                        continue
                    for p, sym in param_syms.items():
                        if base == sym:
                            widths.setdefault(p, set()).add(int(modulus))
        if len(widths) != len(me.map.params) or any(len(bs) != 1 for bs in widths.values()):
            return None
        return {p: next(iter(bs)) for p, bs in widths.items()}

    def _already_tiled(self, me, widths: Dict[str, int]) -> bool:
        """True if some parameter already iterates exactly a ``0:b`` block tile (idempotence)."""
        for (b, e, s), p in zip(me.map.range.ranges, me.map.params):
            if str(dace.symbolic.simplify(e - (widths[p] - 1))) == '0' and str(b) == '0' and str(s) == '1':
                return True
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        count = 0
        for state in sdfg.all_states():
            scope = state.scope_dict()
            top_maps = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and scope[n] is None]
            for me in top_maps:
                widths = self._block_width(state, me)
                if widths is None or self._already_tiled(me, widths):
                    continue
                tile_sizes = tuple(widths[p] for p in me.map.params)
                # Per-map override: divides_evenly can't hold globally if this map's extent isn't a multiple of its width.
                divides = self._divides_evenly and not provably_indivisible(me, tile_sizes)
                MapTiling.apply_to(sdfg,
                                   options={
                                       'tile_sizes': tile_sizes,
                                       'divides_evenly': divides
                                   },
                                   map_entry=me)
                count += 1
        return count


def normalize_schedule_for_layout(sdfg: dace.SDFG, divides_evenly: bool = False) -> int:
    """Re-tile every top-level map to the block width its operands are laid out with; returns maps tiled."""
    return NormalizeScheduleForLayout(divides_evenly=divides_evenly).apply_pass(sdfg, {})
