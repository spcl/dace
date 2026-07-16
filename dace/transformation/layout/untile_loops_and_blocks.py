# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse a manually-tiled loop nest back to a unit-stride loop AND unblock the
matching materialized array dimension at the same time.

This is the layout-suite sibling of
:class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`. The plain
``UntileLoops`` collapses a two-level tile ``for i in range(0, N, K): for ii in range(0, K):``
back to a single ``for k in range(0, N)`` loop, but it operates on the SCHEDULE only: its
combined-access audit requires every memlet to address arrays via the fused ``i + ii``
expression, and it REFUSES any nest whose body reaches an array through SPLIT tile indices
such as ``A[int_floor(i, K), ii]`` (a physically blocked array of shape ``[N/K, K]``).

The layout framework's canonical starting point is packed-C arrays plus plain (un-tiled)
loops. When a source kernel arrives with BOTH a manually-tiled loop and an array that was
physically blocked to match that tile, returning to canonical form requires two coordinated
moves: untile the loop AND unblock the array. ``UntileLoopsAndBlocks`` performs both together.

Recognition (blocked pattern, case A)
=====================================

For a tile pair ``for i in range(0, N, K): for ii in range(0, K)`` (concrete tile ``K``), an
array ``A`` accessed in the body is a materialized block of a logical dimension when its
subset has the form::

    A[..., int_floor(i, K), ii]

that is: the LAST physical dimension is a point access by the inner tile variable ``ii`` and
has extent ``K`` (matching the tile), the SECOND-TO-LAST physical dimension is a point access
by the block index ``int_floor(i, K)``, and every remaining dimension references neither ``i``
nor ``ii``. This is exactly the shape ``SplitDimensions`` (the Block primitive) emits when
blocking the last logical dimension by ``K`` (block index in place, inner offset appended).

Because the outer loop steps ``i`` by ``K``, ``int_floor(i, K) * K == i`` and the block access
addresses logical flat element ``int_floor(i, K) * K + ii == i + ii`` -- the same element a
combined ``A[i + ii]`` access would. Unblocking merges ``(block, inner) -> block * K + inner``,
after which the standard case-A substitution ``i + ii -> k`` folds the access to ``A[k]``.

Rewrite
=======

For each detected blocked array the pass runs
:class:`~dace.transformation.layout.unblock_dimensions.UnblockDimensions` FIRST (rewriting the
split subset ``[int_floor(i, K), ii]`` to the combined ``[int_floor(i, K) * K + ii]`` and the
descriptor from ``[..., N/K, K]`` back to ``[..., N]``), then performs the identical loop-untile
rewrite as ``UntileLoops`` -- so the combined subset substitutes to ``k`` and the array plus
schedule land in packed-C at once.

The pure loop-only cases ``UntileLoops`` already handles are unchanged: when no array in the
body is blocked, the blocked-array set is empty, no ``UnblockDimensions`` runs, and the rewrite
is byte-for-byte the plain untile. The pass is therefore a strict superset of ``UntileLoops``
and a strict no-op on any nest that is not blocked+tiled.

The loop-untiling machinery (tile detection, inner-case classification, multi-dim ascent,
fixpoint, Map round-trip, splice) is REUSED from ``untile_loops`` rather than duplicated; only
the audit is extended (combined OR blocked) and an ``UnblockDimensions`` step is spliced in
before the substitution.
"""
import copy
from typing import Dict, List, Optional, Tuple

import sympy

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.state import LoopRegion, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.layout.unblock_dimensions import UnblockDimensions
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption
# Reuse the pure loop-untiling helpers from UntileLoops rather than duplicate them; the
# original module stays unchanged and pure. Only the audit + unblock step below are new.
from dace.transformation.passes.canonicalize.untile_loops import (_audit_combined_access, _diff_is_zero,
                                                                  _intermediate_chain_clean, _iter_candidate_inners,
                                                                  _match_inner_case, _next_id, _tile_size,
                                                                  _UNTILE_PREFIX)


@properties.make_properties
@xf.explicit_cf_compatible
class UntileLoopsAndBlocks(ppl.Pass):
    """Untile a manually-tiled loop nest and, when the body reaches an array through matching
    block/inner tile indices, unblock that array dimension in the same rewrite.

    A strict superset of :class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`:
    every nest ``UntileLoops`` collapses, this collapses identically (no blocked array -> no
    unblock); additionally it recognizes the case-A blocked access ``A[..., int_floor(i, K), ii]``
    (last dim extent ``K`` indexed by the inner tile variable, adjacent dim the block index) and
    calls :class:`~dace.transformation.layout.unblock_dimensions.UnblockDimensions` to merge the
    ``(block, inner)`` pair back before the untile substitution folds it to ``A[k]``.

    Blocked coordination requires a CONCRETE tile ``K`` (the block factor and the extent match
    are integer). A symbolic tile still untiles (as in ``UntileLoops``) but a split blocked access
    under a symbolic tile is refused -- ``UnblockDimensions`` needs an integer factor.
    """

    CATEGORY: str = 'Canonicalization'

    map_roundtrip = properties.Property(dtype=bool,
                                        default=False,
                                        desc='Lower Maps to LoopRegions before untile and re-lift after, so '
                                        'Map-tiled patterns are detected. Off by default: the canonicalize '
                                        'pipeline runs the lift downstream.')

    def __init__(self, map_roundtrip: bool = False):
        super().__init__()
        self.map_roundtrip = map_roundtrip

    def modifies(self) -> ppl.Modifies:
        # CFG/Symbols/Memlets from the untile; Descriptors/Nodes from the coordinated unblock.
        return (ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Memlets | ppl.Modifies.Descriptors
                | ppl.Modifies.Nodes)

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def _maps_to_loops(self, sdfg: SDFG) -> None:
        """Pre-round-trip: lower every Map to a LoopRegion so Map-tiled patterns enter the
        matcher as loops (mirrors ``UntileLoops._maps_to_loops``)."""
        from dace.transformation.dataflow.map_expansion import MapExpansion
        from dace.transformation.dataflow.map_for_loop import MapToForLoop
        from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
        from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(sdfg, {})
        for _ in range(16):
            before = sum(1 for n, _ in sdfg.all_nodes_recursive()
                         if isinstance(n, dace.nodes.NestedSDFG))
            PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
            PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {})
            after = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
            if after >= before:
                break

    def _loops_back_to_maps(self, sdfg: SDFG) -> None:
        """Post-round-trip: re-lift parallelizable LoopRegions to Maps (mirrors
        ``UntileLoops._loops_back_to_maps``)."""
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([MapCollapse()]).apply_pass(sdfg, {})

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Run the per-loop untile+unblock rewrite as a fixpoint over the SDFG (identical driver
        to ``UntileLoops.apply_pass``; each iteration collapses one tile pair)."""
        if self.map_roundtrip:
            self._maps_to_loops(sdfg)

        total = 0
        max_iters = 1 + sum(1 for sd in sdfg.all_sdfgs_recursive()
                            for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion))
        for _ in range(max_iters):
            rewritten_this_pass = 0
            for sd in sdfg.all_sdfgs_recursive():
                for cfg in list(sd.all_control_flow_regions()):
                    if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                        continue
                    if self._try_untile(cfg, sd):
                        rewritten_this_pass += 1
            if rewritten_this_pass == 0:
                break
            total += rewritten_this_pass

        if self.map_roundtrip:
            self._loops_back_to_maps(sdfg)
        if total:
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        return total or None

    # ------------------------------------------------------------------ #
    #  Blocked-access recognition (the new part).
    # ------------------------------------------------------------------ #
    def _free_syms(self, comp):
        return symbolic.pystr_to_symbolic(str(comp)).free_symbols

    def _dim_mentions(self, rng, sym) -> bool:
        """``True`` iff any component (lo/hi/stride) of range ``rng`` references ``sym``."""
        for comp in rng:
            if sym in self._free_syms(comp):
                return True
        return False

    def _is_point(self, rng, target) -> bool:
        """``True`` iff ``rng`` is a unit-stride point access equal to ``target``."""
        lo, hi, stp = rng
        return _diff_is_zero(lo, hi) and _diff_is_zero(stp, 1) and _diff_is_zero(lo, target)

    def _match_block_memlet(self, sdfg: SDFG, arr_name: str, ranges, outer_var: str, inner_var: str,
                            K_expr: symbolic.SymbolicType,
                            K_const: int) -> Optional[Tuple[List[bool], List[int]]]:
        """If ``ranges`` (a memlet subset on ``arr_name``) is the case-A blocked access
        ``[..., int_floor(i, K), ii]`` -- last dim a point ``ii`` of extent ``K``, second-to-last
        a point ``int_floor(i, K)``, all others independent of ``i``/``ii`` -- return the
        ``(masks, factors)`` (original-rank length) that ``UnblockDimensions`` needs to merge the
        last two physical dims back. Otherwise ``None``."""
        if arr_name not in sdfg.arrays:
            return None
        arr = sdfg.arrays[arr_name]
        rank = len(ranges)
        if rank < 2 or len(arr.shape) != rank:
            return None
        ii_sym = symbolic.pystr_to_symbolic(inner_var)
        i_sym = symbolic.pystr_to_symbolic(outer_var)
        # Last physical dim: point access by the inner tile variable, extent == tile K.
        if not self._is_point(ranges[rank - 1], ii_sym):
            return None
        if not _diff_is_zero(arr.shape[rank - 1], K_const):
            return None
        # Second-to-last physical dim: point access by the block index int_floor(i, K).
        block_target = symbolic.pystr_to_symbolic(f"int_floor({outer_var}, {symbolic.symstr(K_expr)})")
        if not self._is_point(ranges[rank - 2], block_target):
            return None
        # Every remaining dim must be independent of the tile variables.
        for d in range(rank - 2):
            if self._dim_mentions(ranges[d], i_sym) or self._dim_mentions(ranges[d], ii_sym):
                return None
        masks = [False] * (rank - 2) + [True]
        factors = [1] * (rank - 2) + [K_const]
        return (masks, factors)

    def _audit_case(self, inner: LoopRegion, outer_var: str, inner_var: str, case: str,
                    K_expr: symbolic.SymbolicType, K_const: Optional[int],
                    sdfg: SDFG) -> Optional[Dict[str, Tuple[List[bool], List[int]]]]:
        """Blocked-aware body audit. Returns ``None`` to refuse, or a (possibly empty) dict
        ``{array: (masks, factors)}`` of blocked arrays to unblock before the untile.

        Case B delegates to the plain combined audit (no blocked support -- the block/inner form
        differs and case A is the user-directed pattern). Case A accepts each memlet that is
        EITHER combined (every subset component mentioning ``i`` also mentions ``ii`` and
        vice-versa -- the classic ``UntileLoops`` criterion) OR a recognized block access. Any
        other access refuses."""
        if case == 'B':
            return {} if _audit_combined_access(inner, outer_var, inner_var, case) else None

        i_sym = symbolic.pystr_to_symbolic(outer_var)
        ii_sym = symbolic.pystr_to_symbolic(inner_var)
        blocked: Dict[str, Tuple[List[bool], List[int]]] = {}
        for st in inner.all_states():
            for e in st.edges():
                if e.data is None or e.data.is_empty() or e.data.subset is None:
                    continue
                ranges = e.data.subset.ranges
                combined = True
                mentions_tile = False
                for rng in ranges:
                    for comp in rng:
                        free = self._free_syms(comp)
                        has_i = i_sym in free
                        has_ii = ii_sym in free
                        if has_i or has_ii:
                            mentions_tile = True
                        if has_i != has_ii:
                            combined = False
                if not mentions_tile or combined:
                    # Array not indexed by the tile vars, or accessed in flat combined form:
                    # the standard case-A rewrite handles it, no unblock needed.
                    continue
                # A split access -- must be a recognized materialized block (concrete tile only).
                if K_const is None:
                    return None
                spec = self._match_block_memlet(sdfg, e.data.data, ranges, outer_var, inner_var, K_expr, K_const)
                if spec is None:
                    return None
                if e.data.data in blocked and blocked[e.data.data] != spec:
                    return None
                blocked[e.data.data] = spec
        return blocked

    def _unblock_is_safe(self, sdfg: SDFG, blocked_arrays: Dict[str, Tuple[List[bool], List[int]]]) -> bool:
        """Pre-validate that ``UnblockDimensions`` will not raise for any array in
        ``blocked_arrays`` -- so a failed coordination refuses (no-op) rather than leaving a
        half-mutated SDFG. Checks the descriptor rank, every memlet's subset rank and absence of
        ``other_subset``, and refuses any array that flows into a nested SDFG (not visible to this
        top-level pre-validation)."""
        for arr_name, (masks, factors) in blocked_arrays.items():
            if arr_name not in sdfg.arrays:
                return False
            arr = sdfg.arrays[arr_name]
            expected = len(masks) + sum(1 for m in masks if m)
            if len(arr.shape) != expected:
                return False
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        for e in state.in_edges(node) + state.out_edges(node):
                            if e.data is not None and e.data.data == arr_name:
                                return False
                for edge in state.edges():
                    if edge.data is not None and edge.data.data == arr_name:
                        if edge.data.other_subset is not None:
                            return False
                        if edge.data.subset is None or len(edge.data.subset.ranges) != expected:
                            return False
        return True

    # ------------------------------------------------------------------ #
    #  Per-loop rewrite (copied from UntileLoops, with the unblock step spliced in).
    # ------------------------------------------------------------------ #
    def _try_untile(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        outer_stride = loop_analysis.get_loop_stride(outer)
        outer_start = loop_analysis.get_init_assignment(outer)
        outer_end = loop_analysis.get_loop_end(outer)
        if outer_stride is None or outer_start is None or outer_end is None:
            return False
        tile = _tile_size(outer_stride)
        if tile is None:
            return False
        K_expr, K_const = tile
        outer_start_sym = symbolic.simplify(outer_start)

        case: Optional[str] = None
        inner_stride: symbolic.SymbolicType = None
        needs_div_assumption = False
        inner: Optional[LoopRegion] = None
        blocked_arrays: Optional[Dict[str, Tuple[List[bool], List[int]]]] = None
        for candidate in _iter_candidate_inners(outer):
            if not candidate.loop_variable:
                continue
            match = _match_inner_case(candidate, outer.loop_variable, K_expr, K_const)
            if match is None:
                continue
            cand_case, cand_stride, cand_needs_div = match
            audit = self._audit_case(candidate, outer.loop_variable, candidate.loop_variable, cand_case, K_expr,
                                     K_const, sdfg)
            if audit is None:
                continue
            if candidate is not outer and not _intermediate_chain_clean(outer, candidate, outer.loop_variable):
                continue
            inner = candidate
            case = cand_case
            inner_stride = cand_stride
            needs_div_assumption = cand_needs_div
            blocked_arrays = audit
            break
        if inner is None:
            return False

        # The case-A fold (ii -> k - i, then i -> 0) rewrites the unblocked element
        # ``int_floor(i, K) * K + ii`` to ``k``. That IS the original element only when
        # ``int_floor(i, K) * K == i`` -- i.e. when the outer start is a multiple of K, which is how
        # the outer loop steps by K. With a start such as 1 and K=4 the first blocked access reads
        # logical element int_floor(1,4)*4 + 0 = 0 while the untiled one reads k = 1: every element
        # shifts by (start mod K), silently. Refuse instead, and refuse BEFORE the unblock below
        # mutates anything. The pure-combined (unblocked) case-A path needs no such guard -- its
        # element genuinely is i + ii = k for any start.
        if blocked_arrays and case == 'A':
            if symbolic.simplify(sympy.Mod(outer_start_sym, K_expr)) != 0:
                return False

        # Coordinated unblock: merge each detected (block, inner) array dimension back to the flat
        # logical dimension BEFORE the untile substitution, so the resulting combined subset
        # ``int_floor(i, K) * K + ii`` folds to ``k`` alongside the schedule. Refuse (no-op) if the
        # unblock cannot be applied safely rather than crash the pipeline.
        if blocked_arrays:
            if not self._unblock_is_safe(sdfg, blocked_arrays):
                return False
            UnblockDimensions(unblock_map=blocked_arrays).apply_pass(sdfg, {})

        if needs_div_assumption:
            record_assumption(sdfg, sympy.Eq(sympy.Mod(K_expr, inner_stride), 0))

        k_var = f"{_UNTILE_PREFIX}{_next_id(sdfg)}"
        sdfg.add_symbol(k_var, sdfg.symbols.get(outer.loop_variable, dace.int64))
        stop_excl = symbolic.simplify(outer_end + 1)
        span = symbolic.simplify(stop_excl - outer_start_sym)
        N_excl = symbolic.simplify(outer_start_sym + symbolic.int_ceil(span, K_expr) * K_expr)

        i_sym = outer.loop_variable
        ii_sym = inner.loop_variable
        if case == 'A':
            inner.replace_dict({ii_sym: f"({k_var}) - ({i_sym})"})
            inner.replace_dict({i_sym: '0'})
        else:
            inner.replace_dict({ii_sym: k_var})

        parent_of_inner: ControlFlowRegion = inner.parent_graph
        inner_was_start = (parent_of_inner.start_block is inner)
        pred_edges = list(parent_of_inner.in_edges(inner))
        succ_edges = list(parent_of_inner.out_edges(inner))
        try:
            inner_start_block = inner.start_block if inner.number_of_nodes() > 0 else None
        except (NodeNotFoundError, ValueError):
            inner_start_block = None
        inner_sinks = inner.sink_nodes()
        child_blocks = list(inner.nodes())
        inner_edges = list(inner.edges())
        parent_of_inner.remove_node(inner)
        for child in child_blocks:
            inner.remove_node(child)
            child_is_start = inner_was_start and (child is inner_start_block)
            parent_of_inner.add_node(child, is_start_block=child_is_start, ensure_unique_name=True)
        for ie in inner_edges:
            parent_of_inner.add_edge(ie.src, ie.dst, ie.data)
        if inner_start_block is not None:
            for pe in pred_edges:
                parent_of_inner.add_edge(pe.src, inner_start_block, copy.deepcopy(pe.data))
        for se in succ_edges:
            for sink in inner_sinks:
                parent_of_inner.add_edge(sink, se.dst, copy.deepcopy(se.data))

        outer.loop_variable = k_var
        outer.init_statement = dace.properties.CodeBlock(f"{k_var} = {symbolic.symstr(outer_start_sym)}")
        outer.loop_condition = dace.properties.CodeBlock(f"{k_var} < ({symbolic.symstr(N_excl)})")
        outer.update_statement = dace.properties.CodeBlock(f"{k_var} = {k_var} + {symbolic.symstr(inner_stride)}")
        return True


__all__ = ['UntileLoopsAndBlocks']
