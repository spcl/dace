# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NormalizeScheduleForLayout -- re-tile the schedule to a layout's block width.

A layout change only moves DATA. After ``SplitDimensions`` blocks an array, an access ``A[i]``
becomes ``A[int_floor(i, b), Mod(i, b)]`` -- but if the map iterating ``i`` is NOT tiled by ``b``,
the inner offset ``Mod(i, b)`` does not correspond to a contiguous inner loop, so the schedule does
not exploit the block. This pass is the DUAL of :class:`BlockAwareMapTiling` (which tiles BEFORE a
Block): it reads the POST-layout access patterns, finds each top-level map whose dimensions are all
accessed with a common inner block width ``b`` (a ``Mod(param, b)`` in a memlet subset), and tiles
that map by ``b`` so the innermost loop iterates the block contiguously (``int_floor(i, b)`` becomes
the tile index, ``Mod(i, b)`` the inner offset).

Run AFTER applying a layout (``prepare_for_layout`` -> Block/... -> ``normalize_schedule_for_layout``).
Only maps whose EVERY parameter has a single detected block width are tiled (the paper's "all
dimensions accessed with an inner dimension of ``b`` -> block" case); a map already iterating a tile
(``0:b`` range) is skipped, so re-running is idempotent.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import sympy

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling


@dataclass
class NormalizeScheduleForLayout(ppl.Pass):
    """Tile each top-level map by the block width its operands are laid out with.

    :param divides_evenly: passed to ``MapTiling`` -- ``True`` when the extents are known multiples
                           of the block widths (e.g. after ``PadDimensions``), for a clean tiling
                           with no remainder map.
    """

    def __init__(self, divides_evenly: bool = False):
        self._divides_evenly = divides_evenly

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _block_width(self, state, me) -> Optional[Dict[str, int]]:
        """The per-parameter inner block width of ``me``, or ``None`` if not every parameter has a
        single detected width (only uniform, fully-blocked maps are tiled)."""
        param_syms = {p: dace.symbolic.pystr_to_symbolic(p) for p in me.map.params}
        widths: Dict[str, set] = {}
        for edge in state.scope_subgraph(me).edges():
            if edge.data is None or edge.data.subset is None:
                continue
            for begin, end, _ in edge.data.subset.ranges:
                # Only a POINT access ``Mod(param, b):Mod(param, b)`` is the genuine blocked inner
                # offset. A propagated RANGE reservoir memlet (begin != end) after tiling also
                # carries Mod(tile_param, b) -- excluding ranges keeps re-runs idempotent.
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
                MapTiling.apply_to(sdfg,
                                   options={
                                       'tile_sizes': tile_sizes,
                                       'divides_evenly': self._divides_evenly
                                   },
                                   map_entry=me)
                count += 1
        return count


def normalize_schedule_for_layout(sdfg: dace.SDFG, divides_evenly: bool = False) -> int:
    """Re-tile every top-level map to the block width its operands are laid out with. Returns the
    number of maps tiled. See :class:`NormalizeScheduleForLayout`."""
    return NormalizeScheduleForLayout(divides_evenly=divides_evenly).apply_pass(sdfg, {})
