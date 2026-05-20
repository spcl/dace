# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SVE-style finalize orchestrator (the ``sve_style='fixed'`` chain).

A single coordinating :class:`Pass` (the M3.2/M3.3 pattern — composition
order and arguments depend on runtime SDFG state, so a flat declarative
pipeline cannot express it). It owns the whole SVE-style finalize and,
critically, **captures the global trip bound once at tile time** (the
single source of truth) and threads it explicitly to the mask + Min-swap
passes, immune to any intervening pass mutating the ``core`` map range.

Chain (analyze-clean-then-Min):

1. For each eligible innermost single-param step-1 map: capture
   ``global_ub = ub + 1`` (the original exclusive trip bound) *now*,
   compute the clean block ``B = roundup(ceil(trip / num_cores), W)``,
   and ``MapTiling(divides_evenly=True)`` it into a ``core`` outer map
   plus a divisible per-core block. ``divides_evenly=True`` keeps every
   map range affine so downstream divisibility / subset analysis never
   sees a ``Min`` (avoids the ``SympifyError`` a ``Min`` in a map range
   triggers).
2. ``NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True)`` — the
   per-core block is divisible by design but still needs a NestedSDFG
   body for the mask.
3. ``GenerateIterationMask(mode='global', global_ub=<captured>)`` — the
   global-keyed ``_iter_mask`` (``i + l < global_ub``).
4. The shared ``Vectorize`` pass W-strides the divisible block and emits
   the masked (``_av_masked``) variants.
5. ``Detect{Gather,Scatter}`` (and the strided detectors when
   ``lower_to_intrinsics``) collapse per-lane fans to masked intrinsics
   — mandatory under masking (a per-lane scalar fan faults on inactive
   lanes).
6. ``MapToForLoop`` turns each W-strided per-core map into a
   :class:`LoopRegion`.
7. ``ForLoopToMaskedWhile(global_ub=<captured>)`` Min-swaps the loop
   condition and W-stride-normalizes the update; it re-derives the bound
   from the ``core`` map and **asserts it equals the captured value**
   (loud failure if any pass perturbed the range).

The bound is *captured at step 1*, where the original map is in hand,
and used in steps 3 and 7 — never re-derived from a secondary artifact
several passes later.
"""
from typing import List, Optional, Tuple

import dace
from dace import properties
from dace import symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.for_loop_to_masked_while import ForLoopToMaskedWhile
from dace.transformation.passes.vectorization.detect_gather import DetectGather
from dace.transformation.passes.vectorization.detect_scatter import DetectScatter
from dace.transformation.passes.vectorization.detect_strided_load import DetectStridedLoad
from dace.transformation.passes.vectorization.detect_strided_store import DetectStridedStore
from dace.transformation.passes.vectorization.detect_multi_dim_strided_load import DetectMultiDimStridedLoad
from dace.transformation.passes.vectorization.detect_multi_dim_strided_store import DetectMultiDimStridedStore
from dace.transformation.passes.vectorization.remove_vector_maps import RemoveVectorMaps
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

_CORE_PREFIX = "core"


@properties.make_properties
class SveStyleFinalize(ppl.Pass):
    """Drive the ``sve_style='fixed'`` chain with a tile-time-captured bound."""

    CATEGORY: str = "Vectorization Preparation"

    def __init__(self,
                 vectorizer,
                 vector_width: int,
                 num_cores: int,
                 lower_to_intrinsics: bool = True,
                 eliminate_trivial_vector_map: bool = True):
        super().__init__()
        self._vectorizer = vectorizer
        self._W = vector_width
        self._P = num_cores
        self._lower_to_intrinsics = lower_to_intrinsics
        self._eliminate_trivial_vector_map = eliminate_trivial_vector_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _lane_winner_param(map_entry: dace.nodes.MapEntry, state: dace.SDFGState) -> str:
        """Pick the map param that most often drives the unit-stride dim
        of an array access in this map's scope. That param is the best
        lane dim (W-stride across it touches contiguous memory); it
        should end up innermost so the vectorizer's W-strip-mining
        operates on the contiguous axis. Falls back to the last param
        when no analysis info is available (default-correct for
        C-layout where the last dim is already the contiguous one).

        :param map_entry: The multi-param map to analyze.
        :param state: The state containing ``map_entry``.
        :returns: The param name with the highest unit-stride-dim
            access count, or the last param as a safe fallback.
        """
        from collections import Counter
        counts: Counter = Counter()
        scope = state.scope_subgraph(map_entry)
        for node in scope.nodes():
            for e in list(scope.in_edges(node)) + list(scope.out_edges(node)):
                if e.data is None or e.data.data is None:
                    continue
                if e.data.data not in state.sdfg.arrays:
                    continue
                arr = state.sdfg.arrays[e.data.data]
                strides = getattr(arr, "strides", None)
                if not strides:
                    continue
                try:
                    unit_dim = [str(s) for s in strides].index("1")
                except ValueError:
                    continue
                if e.data.subset is None or unit_dim >= len(e.data.subset):
                    continue
                dim_expr = e.data.subset[unit_dim]
                begin = dim_expr[0] if isinstance(dim_expr, tuple) else dim_expr
                try:
                    free = {str(s) for s in begin.free_symbols}
                except Exception:
                    continue
                for p in map_entry.map.params:
                    if p in free:
                        counts[p] += 1
        if not counts:
            return map_entry.map.params[-1]
        return counts.most_common(1)[0][0]

    @staticmethod
    def _permute_winner_last(map_entry: dace.nodes.MapEntry, winner: str) -> bool:
        """Rearrange ``map_entry.params`` and ``range.ranges`` so
        ``winner`` is the last param (innermost after MapExpansion).
        In-place; semantically equivalent (the cartesian product is
        unchanged). Returns True iff a swap occurred.

        :param map_entry: Map whose params to permute.
        :param winner: Param to move to the last position.
        :returns: True if the order changed; False if winner was
            already last or absent.
        """
        params = list(map_entry.map.params)
        if winner not in params or params[-1] == winner:
            return False
        ranges = list(map_entry.map.range.ranges)
        idx = params.index(winner)
        params.append(params.pop(idx))
        ranges.append(ranges.pop(idx))
        map_entry.map.params = params
        map_entry.map.range = dace.subsets.Range(ranges)
        return True

    def _expand_multi_to_1d(self, sdfg: dace.SDFG) -> int:
        """Analyze-permute-expand every multi-param innermost map so the
        SVE chain sees only 1D maps. For each multi-param map: pick the
        lane winner (most unit-stride accesses), permute it to last,
        ``MapExpansion`` splits into nested 1D maps (the deepest is the
        winner). Mark every post-expansion deepest 1D map nested under
        another MapEntry ``Sequential`` — these come from multi-dim
        expansion and don't need ``num_cores`` core-tiling (the outer
        expanded maps already provide parallelism).

        :param sdfg: SDFG to transform in place.
        :returns: Number of multi-param maps expanded.
        """
        multi_targets = [
            (n, g) for n, g in list(sdfg.all_nodes_recursive())
            if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)
            and len(n.map.params) > 1 and not any(p.startswith(_CORE_PREFIX) for p in n.map.params)
        ]
        expanded = 0
        for n, g in multi_targets:
            winner = self._lane_winner_param(n, g)
            self._permute_winner_last(n, winner)
            MapExpansion.apply_to(g.sdfg, verify=True, save=False, map_entry=n)
            expanded += 1
        if expanded:
            # Mark deepest post-expansion 1D maps Sequential (those
            # nested under another MapEntry — i.e. came from an expanded
            # multi-param). True top-level 1D maps are unchanged.
            for n, g in list(sdfg.all_nodes_recursive()):
                if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                    continue
                if not is_innermost_map(g, n) or len(n.map.params) != 1:
                    continue
                if isinstance(g.scope_dict().get(n), dace.nodes.MapEntry):
                    n.map.schedule = dace.dtypes.ScheduleType.Sequential
        return expanded

    def _eligible_innermost_maps(self,
                                 sdfg: dace.SDFG) -> List[Tuple[dace.nodes.MapEntry, dace.SDFGState, str, object]]:
        """Collect ``(map_entry, state, global_ub, block)`` for every
        tileable innermost map, capturing the bound *before* tiling.

        :param sdfg: The SDFG to scan.
        :returns: One tuple per eligible single-param step-1 innermost map.
        """
        W, P = self._W, self._P
        out = []
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                continue
            if not is_innermost_map(g, n):
                continue
            if len(n.map.params) != 1:
                continue
            if any(p.startswith(_CORE_PREFIX) for p in n.map.params):
                continue
            lb, ub, step = n.map.range[-1]
            if (step != 1) and (str(step) != "1"):
                continue
            trip = symbolic.simplify(ub - lb + 1)
            global_ub = str(ub + 1)  # original exclusive bound, captured now
            block = symbolic.int_ceil(symbolic.int_ceil(trip, P), W) * W
            out.append((n, g, global_ub, block))
        return out

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Run the SVE-style finalize. Returns 1 if it fired, else ``None``.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :raises NotImplementedError: if eligible innermost maps have
            distinct global trips (first-cut supports one global bound).
        """
        W = self._W
        # 0. Multi-dim handling: permute lane winner to last + MapExpansion
        #    + mark expanded inner maps Sequential (skip core-tile for them).
        self._expand_multi_to_1d(sdfg)

        targets = self._eligible_innermost_maps(sdfg)
        if not targets:
            return None
        gubs = {gub for _, _, gub, _ in targets}
        if len(gubs) != 1:
            raise NotImplementedError(f"sve_style: eligible innermost maps have distinct global trips {sorted(gubs)}; "
                                      f"the first-cut SVE chain threads a single captured global_ub. Split the kernel "
                                      f"or restrict via apply_on_maps.")
        global_ub = gubs.pop()

        # 1. Tile only TRUE 1D maps (no MapEntry scope-parent) into
        #    a clean divisible per-core block. Multi-dim maps from
        #    expansion are skipped: their outer expanded maps already
        #    provide parallelism, the inner is Sequential — no need
        #    for num_cores partitioning on the inner.
        for n, g, _gub, block in targets:
            if isinstance(g.scope_dict().get(n), dace.nodes.MapEntry):
                continue  # multi-dim inner — skip core-tile
            MapTiling.apply_to(g.sdfg,
                               options={
                                   "tile_sizes": (block, ),
                                   "prefix": _CORE_PREFIX,
                                   "divides_evenly": True,
                                   "tile_trivial": True,
                               },
                               verify=True,
                               save=False,
                               map_entry=n)

        # 2-3. Nest the divisible block body + attach the global mask.
        NestInnermostMapBodyIntoNSDFG(vector_width=W, nest_provably_divisible=True).apply_pass(sdfg, {})
        GenerateIterationMask(vector_width=W, mode="global", global_ub=global_ub).apply_pass(sdfg, {})

        # 4. Vectorize the divisible block (emits the masked variants).
        self._vectorizer.apply_pass(sdfg, {})

        # 5. Collapse per-lane gather/scatter (and strided) fans to masked
        #    intrinsics — mandatory under masking.
        DetectGather().apply_pass(sdfg, {})
        DetectScatter().apply_pass(sdfg, {})
        if self._lower_to_intrinsics:
            DetectStridedLoad().apply_pass(sdfg, {})
            DetectStridedStore().apply_pass(sdfg, {})
            DetectMultiDimStridedLoad().apply_pass(sdfg, {})
            DetectMultiDimStridedStore().apply_pass(sdfg, {})

        # 6. Each W-strided per-core map -> a LoopRegion.
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)):
                continue
            if any(p.startswith(_CORE_PREFIX) for p in n.map.params):
                continue
            st = n.map.range[-1][2]
            if (st == W) or (str(st) == str(W)):
                MapToForLoop.apply_to(g.sdfg, verify=True, save=False, map_entry=n)

        # 7. for-loop -> masked while (Min-swap + tile-time-bound assert).
        ForLoopToMaskedWhile(vector_width=W, global_ub=global_ub).apply_pass(sdfg, {})

        if self._eliminate_trivial_vector_map:
            RemoveVectorMaps().apply_pass(sdfg, {})
        return 1
