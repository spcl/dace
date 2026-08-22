# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseBranchedTailRemainder`` — GPU-only post-transform that fuses a
vectorized main-tiled map and its remainder map into ONE map whose body is a
``ConditionalBlock`` (``if`` full-tile -> mask-free vectorized tile ops /
``else`` -> the remainder body).

Two remainder strategies share this pass, differing only in what the ``else``
arm holds:

* ``branched_masked_tail`` (the GPU K=1 DEFAULT) -- the tail is the W-strided
  MASKED tile map :class:`SplitMapForTileRemainder` peels in ``tail_mode='masked_branch'``
  (marked ``__masked_tail``). Both arms are tile ops over the SAME tile start, so a
  full-tile iteration runs the widened mask-free body and NO scalar loop is emitted
  anywhere in the kernel; the partial tile takes the masked (per-element, never
  over-reading) arm, which is also where the alignment fact does not hold.
* ``branched_tail`` -- the tail is the step-1 ``__scalar_tail`` map
  (``tail_mode='scalar'``), reused verbatim inside a ``Sequential`` lane loop.

Either way the mechanism is a control-flow BRANCH inside ONE kernel rather than a
mask on every tile. It runs LAST in the GPU pipeline, after the split peeled a
provably-divisible interior + a tail, and after the tile prep/emit passes
vectorized the interior (marked ``__tile_main``) mask-free.

Motivation (the two-kernel problem): on GPU the interior and the tail are two
``GPU_Device`` maps -> two kernel launches. This pass fuses them into a single
``GPU_Device`` map over the whole tile range, so there is ONE kernel launch;
branch divergence happens only on the single partial (tail) tile.

Mechanism (the exact split-then-fuse the maintainer specified)::

    for each tile-start s over [lb : ub : W]:      # ONE GPU_Device map
        if (s + W - 1) <= ub:  <MASK-FREE tile body>        # __tile_main NSDFG, unchanged
        else:                  <MASKED tile body>           # __masked_tail NSDFG, unchanged
                          -or- for t in [s : ub]: <scalar body>   # __scalar_tail NSDFG

The vectorized tile ops (``TileLoad`` / ``TileBinop`` / ``TileStore`` / ...) are
reused UNCHANGED inside the ``if`` branch -- NOT flattened to an arithmetic
select (no ``LowerITEToFpFactor``). The masked tail NSDFG is likewise reused
unchanged: only ONE tile start ever reaches the ``else`` (the one partial tile),
and it is exactly the start the peeled slab map would have run, so its mask
``s + lane < ub + 1`` and its subsets are already right. The scalar tail NSDFG is
instead wrapped in a ``Sequential`` loop over the tail lanes, so the single thread
that takes the ``else`` processes every remaining lane.

Scope: K=1 (one tiled/innermost dim). The map may carry outer (prefix) params;
only the innermost dim is tiled + fused. A pair is fused only when both bodies
are a single nested SDFG whose boundary the fused map can carry (the canonical
post-pipeline shape); any other shape is left as two maps (correct, un-fused)
rather than mis-fused.
"""
import copy
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (MASKED_TAIL_MARKER,
                                                                                   SCALAR_TAIL_MARKER, TILE_MAIN_MARKER)

#: Tail markers this pass folds into the ``else`` arm. ``__masked_tail`` is a tile body placed as
#: is; ``__scalar_tail`` is a step-1 body that needs the lane loop around it.
_TAIL_MARKERS = (MASKED_TAIL_MARKER, SCALAR_TAIL_MARKER)


@properties.make_properties
class FuseBranchedTailRemainder(ppl.Pass):
    """Fuse each ``__tile_main`` map + its ``__masked_tail`` / ``__scalar_tail`` sibling into ONE
    map with an ``if(full-tile)/else(remainder)`` ``ConditionalBlock`` body (GPU only).

    Implements the ``branched_masked_tail`` (masked ``else``) and ``branched_tail`` (scalar
    ``else``) remainder strategies."""

    CATEGORY: str = "Vectorization"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last. This strategy supports K=1 (a single tiled dim).",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last. Only K=1 (one tiled dim) is
            supported; the innermost width is used.
        :raises ValueError: If ``widths`` is empty.
        """
        super().__init__()
        if len(widths) != 1:
            raise ValueError(f"FuseBranchedTailRemainder supports K=1 (one tiled dim); got widths={widths!r}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Pass replaces map bodies and removes the remainder scope.

        :returns: ``Nodes | States | Scopes | Symbols``.
        """
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Scopes | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Runs once at the end of the GPU pipeline.

        :param modified: Earlier passes' modifications (unused).
        :returns: ``False``.
        """
        return False

    @staticmethod
    def _base_label(label: str, marker: str) -> str:
        """Strip a tile-remainder marker suffix to recover the shared base label."""
        return label[:-len(marker)] if label.endswith(marker) else label

    @staticmethod
    def _tail_marker(label: str) -> Optional[str]:
        """The remainder marker ``label`` carries, or ``None`` if it is not a fusable tail."""
        return next((m for m in _TAIL_MARKERS if label.endswith(m)), None)

    @staticmethod
    def _is_split_sibling(main_entry: MapEntry, rem_entry: MapEntry) -> bool:
        """Is ``rem_entry`` the tail :class:`SplitMapForTileRemainder` peeled off ``main_entry``?

        The split leaves one structural signature: identical prefix dims, and an innermost tail
        range starting exactly one past the interior's end (interior ``[lb : main_end]``, tail
        ``[main_end + 1 : ub]``). Checked rather than assumed, because a map label is not unique --
        an SDFG routinely holds several ``_Mult__map`` scopes -- so a shared base label alone can
        name maps that were never split from each other.
        """
        main_ranges, rem_ranges = list(main_entry.map.range.ranges), list(rem_entry.map.range.ranges)
        if len(main_ranges) != len(rem_ranges) or not main_ranges:
            return False
        if main_entry.map.params != rem_entry.map.params:
            return False
        if any(str(a) != str(b) for a, b in zip(main_ranges[:-1], rem_ranges[:-1])):
            return False
        return symbolic.simplify(rem_ranges[-1][0] - main_ranges[-1][1] - 1) == 0

    def _find_pairs(self, state: dace.SDFGState) -> List[Tuple[MapEntry, MapEntry]]:
        """Pair every top-level ``__tile_main`` map with the remainder sibling it split from.

        A tail is consumed by at most one main, and a main with no structural sibling is left
        alone -- fusing it with some other map's tail would silently run that map's body over the
        wrong range.

        :param state: A dataflow state to scan.
        :returns: ``[(main_entry, remainder_entry), ...]`` for pairs in ``state``.
        """
        scope = state.scope_dict()
        mains: Dict[str, List[MapEntry]] = {}
        tails: Dict[str, List[MapEntry]] = {}
        for node in state.nodes():
            if not isinstance(node, MapEntry) or scope[node] is not None:
                continue  # only top-level (kernel-level) maps
            if node.map.label.endswith(TILE_MAIN_MARKER):
                mains.setdefault(self._base_label(node.map.label, TILE_MAIN_MARKER), []).append(node)
                continue
            marker = self._tail_marker(node.map.label)
            if marker is not None:
                tails.setdefault(self._base_label(node.map.label, marker), []).append(node)

        pairs: List[Tuple[MapEntry, MapEntry]] = []
        for base, main_entries in mains.items():
            available = list(tails.get(base, []))
            for main_entry in main_entries:
                sibling = next((t for t in available if self._is_split_sibling(main_entry, t)), None)
                if sibling is not None:
                    available.remove(sibling)
                    pairs.append((main_entry, sibling))
        return pairs

    @staticmethod
    def _sole_body_nsdfg(state: dace.SDFGState, entry: MapEntry) -> Optional[NestedSDFG]:
        """Return the single nested SDFG forming ``entry``'s body, or ``None`` if not that shape."""
        exit_node = state.exit_node(entry)
        body = [n for n in state.all_nodes_between(entry, exit_node)]
        nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
        if len(nsdfgs) == 1 and len(body) == 1:
            return nsdfgs[0]
        return None

    def _symbol_dtype(self, sdfg: dace.SDFG, sym: str):
        """Best-effort dtype for a symbol: the SDFG's declared type, else ``int64``."""
        declared = sdfg.symbols
        return declared[sym] if sym in declared else dace.int64

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Fuse every ``(__tile_main, tail)`` sibling pair in the SDFG.

        :param sdfg: SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of pairs fused, or ``None`` if none.
        """
        fused = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            for state in list(sd.states()):
                for main_entry, rem_entry in self._find_pairs(state):
                    if self._fuse_one(sd, state, main_entry, rem_entry):
                        fused += 1
        if fused:
            sdfg.reset_cfg_list()
        return fused or None

    def _fuse_one(self, sd: dace.SDFG, state: dace.SDFGState, main_entry: MapEntry, rem_entry: MapEntry) -> bool:
        """Fuse one main-tiled + remainder pair into a single conditional-body map.

        :param sd: The SDFG owning ``state``.
        :param state: The state holding both map scopes.
        :param main_entry: The ``__tile_main`` (mask-free vectorized) map entry.
        :param rem_entry: The ``__masked_tail`` / ``__scalar_tail`` map entry.
        :returns: ``True`` if fused; ``False`` if the pair was left un-fused (unhandled shape).
        """
        main_exit = state.exit_node(main_entry)
        rem_exit = state.exit_node(rem_entry)
        main_nsdfg = self._sole_body_nsdfg(state, main_entry)
        rem_nsdfg = self._sole_body_nsdfg(state, rem_entry)
        if main_nsdfg is None or rem_nsdfg is None:
            return False  # not the canonical single-NSDFG body shape -> leave as two maps
        # The fused map carries the MAIN map's boundary edges, so a tail that touches an array the
        # main body does not would end up in a body with no edge feeding it. Refuse the pair (two
        # correct maps) rather than build that.
        if not (set(rem_nsdfg.in_connectors) <= set(main_nsdfg.in_connectors)
                and set(rem_nsdfg.out_connectors) <= set(main_nsdfg.out_connectors)):
            return False

        W = int(self.widths[-1])
        tiled_param = main_entry.map.params[-1]
        main_ranges = list(main_entry.map.range.ranges)
        rem_ranges = list(rem_entry.map.range.ranges)
        tiled_lb = symbolic.simplify(main_ranges[-1][0])
        # The tail's innermost dim is [main_end+1 : ub : step]; ``ub`` (its inclusive end) is the
        # ORIGINAL extent's upper bound -- the clean value the fused map + branch condition reuse.
        tail_ub = symbolic.simplify(rem_ranges[-1][1])

        # Capture the main map's boundary memlets before we detach its body, keyed by the
        # nested-SDFG connector name so we can rewire them onto the fused body verbatim.
        in_edge_by_conn = {e.dst_conn: (e.src_conn, e.data) for e in state.out_edges(main_entry) if e.dst is main_nsdfg}
        out_edge_by_conn = {e.src_conn: (e.dst_conn, e.data) for e in state.in_edges(main_exit) if e.src is main_nsdfg}

        masked_tail = rem_entry.map.label.endswith(MASKED_TAIL_MARKER)
        fused_body = self._build_fused_body(sd, main_entry, main_nsdfg, rem_nsdfg, W, tiled_param, tail_ub, masked_tail)

        # Detach the two original bodies + the whole remainder scope from the state.
        state.remove_node(main_nsdfg)
        state.remove_node(rem_nsdfg)
        state.remove_node(rem_entry)
        state.remove_node(rem_exit)

        # Reuse the main map as the fused map, iterating the ORIGINAL element range strided by W:
        # extend the (already-original) innermost lower bound to the tail's upper bound ``ub`` so
        # the single strided map ``[lb : ub : W]`` covers the partial tile too. Same param, step W.
        new_ranges = main_ranges[:-1] + [(tiled_lb, tail_ub, symbolic.SymExpr(W))]
        main_entry.map.range = dace.subsets.Range(new_ranges)
        main_entry.map.label = self._base_label(main_entry.map.label, TILE_MAIN_MARKER)

        # Wire the fused body NSDFG under the (now fused) map, reusing the captured memlets.
        # Every symbol the body still needs from the outer scope (the fused-map param + N + ...) is
        # passed through by identity -- they all name the same symbol in the enclosing map scope.
        # Sorted, and the connector DICTS rather than sets: symbol order decides the emitted
        # declaration order, and a set would both randomise it and drop the connector types.
        symbol_mapping = {s: symbolic.pystr_to_symbolic(s) for s in sorted(str(s) for s in fused_body.free_symbols)}
        fused_nsdfg = state.add_nested_sdfg(fused_body,
                                            inputs=dict(main_nsdfg.in_connectors),
                                            outputs=dict(main_nsdfg.out_connectors),
                                            symbol_mapping=symbol_mapping)
        for conn, (src_conn, memlet) in in_edge_by_conn.items():
            state.add_edge(main_entry, src_conn, fused_nsdfg, conn, dace.Memlet.from_memlet(memlet))
        for conn, (dst_conn, memlet) in out_edge_by_conn.items():
            state.add_edge(fused_nsdfg, conn, main_exit, dst_conn, dace.Memlet.from_memlet(memlet))
        return True

    def _build_fused_body(self, sd: dace.SDFG, main_entry: MapEntry, main_nsdfg: NestedSDFG, rem_nsdfg: NestedSDFG,
                          W: int, tiled_param: str, tail_ub, masked_tail: bool) -> dace.SDFG:
        """Construct the fused-body SDFG: one ``ConditionalBlock`` over the two reused bodies.

        :param masked_tail: ``True`` when the tail is a ``__masked_tail`` TILE body (placed as is,
            it already runs over the same tile start), ``False`` for a ``__scalar_tail`` step-1
            body (wrapped in a ``Sequential`` lane loop).
        :returns: A fresh SDFG whose sole block is the ``if(full-tile)/else(tail)`` conditional.
        """
        base = self._base_label(main_entry.map.label, TILE_MAIN_MARKER)
        body = dace.SDFG(f"{base}_fused_remainder")

        # Boundary arrays (the nested-SDFG connectors) become non-transient descriptors. Ordered
        # dedup, not a set: this is the order the arrays are declared in, hence in the emitted code.
        conn_arrays = dict.fromkeys(list(main_nsdfg.in_connectors) + list(main_nsdfg.out_connectors))
        for name in conn_arrays:
            desc = copy.deepcopy(sd.arrays[name])
            desc.transient = False
            body.add_datadesc(name, desc)

        # Symbols the two reused bodies reference from the outer (fused-map) scope. Sorted per
        # mapping value, because ``free_symbols`` is an unordered sympy set and the insertion order
        # here is observable in the emitted declarations.
        outer_syms = dict.fromkeys(main_entry.map.params)  # every fused-map param is a body symbol
        for nsdfg in (main_nsdfg, rem_nsdfg):
            for value in nsdfg.symbol_mapping.values():
                free = symbolic.pystr_to_symbolic(str(value)).free_symbols
                outer_syms.update(dict.fromkeys(sorted(str(s) for s in free)))
        for s in outer_syms:
            if s not in body.symbols:
                body.add_symbol(s, self._symbol_dtype(sd, s))

        cond_block = ConditionalBlock(f"{base}_remainder_cond", sdfg=body)
        body.add_node(cond_block, ensure_unique_name=True)

        # if-branch: the vectorized tile body runs when the whole W-tile is inside the original
        # extent. The predicate collapses the split's inner/remainder boundary into ONE clean check
        # over the ORIGINAL extent end ``ub`` and width ``W`` -- a W-tile at start i is fully in
        # bounds iff its last lane ``i + W - 1 <= ub`` <=> ``i <= ub - W + 1`` (simplified, so the
        # emitted CUDA reads as a plain expression in N and W, not a residue of the split).
        full_cond = self._full_tile_condition(tiled_param, W, tail_ub)
        full_region = ControlFlowRegion(f"{cond_block.label}_full", sdfg=body)
        cond_block.add_branch(CodeBlock(full_cond), full_region)
        self._populate_tile_branch(body, full_region, main_nsdfg, "full_tile")

        # else-branch: either the MASKED tile body placed as is (the one partial tile reaching here
        # is exactly the tile start the peeled slab map ran, so its mask + subsets already hold, and
        # no scalar loop is emitted), or the scalar body under a Sequential loop over its lanes.
        rem_region = ControlFlowRegion(f"{cond_block.label}_remainder", sdfg=body)
        cond_block.add_branch(None, rem_region)
        if masked_tail:
            self._populate_tile_branch(body, rem_region, rem_nsdfg, "masked_tile")
        else:
            self._populate_scalar_branch(body, rem_region, rem_nsdfg, tiled_param, tail_ub)
        return body

    @staticmethod
    def _full_tile_condition(tiled_param: str, W: int, ub) -> str:
        """Clean ``if``-branch predicate: a W-tile at start ``i`` is fully inside the extent.

        ``i + W - 1 <= ub`` <=> ``i <= ub - W + 1``; the right-hand side is a pure expression in
        the original extent end ``ub`` and ``W``, so ``symbolic.simplify`` folds it to a tidy bound
        (e.g. ``i <= N - 9`` for ``ub = N - 2``, ``W = 8``) with no ``int_floor`` split residue.
        """
        bound = symbolic.simplify(symbolic.pystr_to_symbolic(str(ub)) - W + 1)
        return f"{tiled_param} <= {symbolic.symstr(bound)}"

    @staticmethod
    def _populate_tile_branch(body: dace.SDFG, region: ControlFlowRegion, tile_nsdfg: NestedSDFG, label: str) -> None:
        """Place a reused TILE body NSDFG (mask-free main, or masked tail) into a branch state.

        Both arms run over the same tile start -- the fused map's param -- so the body is reused
        with its own symbol mapping untouched.
        """
        st = region.add_state(label, is_start_block=True)
        node = st.add_nested_sdfg(tile_nsdfg.sdfg,
                                  inputs=dict(tile_nsdfg.in_connectors),
                                  outputs=dict(tile_nsdfg.out_connectors),
                                  symbol_mapping=dict(tile_nsdfg.symbol_mapping))
        for conn in tile_nsdfg.in_connectors:
            an = st.add_access(conn)
            st.add_edge(an, None, node, conn, dace.Memlet.from_array(conn, body.arrays[conn]))
        for conn in tile_nsdfg.out_connectors:
            an = st.add_access(conn)
            st.add_edge(node, conn, an, None, dace.Memlet.from_array(conn, body.arrays[conn]))

    @staticmethod
    def _populate_scalar_branch(body: dace.SDFG, region: ControlFlowRegion, rem_nsdfg: NestedSDFG, tiled_param: str,
                                tail_ub) -> None:
        """Place the reused scalar body (remainder NSDFG) into the ``else`` branch, wrapped in a
        Sequential loop over this partial tile's lanes ``[i : ub]`` of the innermost dim.

        The loop starts at the CURRENT tile start ``i`` (the fused-map param), not the split's
        ``main_end+1`` constant: only the single partial-tile thread (where ``i`` equals that split
        point) ever enters the ``else``, so ``[i : ub]`` is exactly that thread's tail lanes and the
        emitted loop bound is a clean expression (``i`` .. ``ub``) free of the ``int_floor`` split
        residue."""
        st = region.add_state("scalar_tail", is_start_block=True)
        loop_var = f"__rem_{tiled_param}"
        loop_lb = symbolic.pystr_to_symbolic(tiled_param)
        me, mx = st.add_map(f"{region.label}_loop",
                            {loop_var: dace.subsets.Range([(loop_lb, symbolic.simplify(tail_ub), 1)])},
                            schedule=dace.dtypes.ScheduleType.Sequential)
        # Remap the scalar body's tiled iter-symbol to this loop var; every other mapping is kept.
        sym_map = dict(rem_nsdfg.symbol_mapping)
        sym_map[tiled_param] = symbolic.pystr_to_symbolic(loop_var)
        node = st.add_nested_sdfg(rem_nsdfg.sdfg,
                                  inputs=dict(rem_nsdfg.in_connectors),
                                  outputs=dict(rem_nsdfg.out_connectors),
                                  symbol_mapping=sym_map)
        # No data edge to the entry/exit: an empty memlet keeps the node inside the scope.
        if not rem_nsdfg.in_connectors:
            st.add_nedge(me, node, dace.Memlet())
        if not rem_nsdfg.out_connectors:
            st.add_nedge(node, mx, dace.Memlet())
        for conn in rem_nsdfg.in_connectors:
            an = st.add_access(conn)
            st.add_memlet_path(an, me, node, dst_conn=conn, memlet=dace.Memlet.from_array(conn, body.arrays[conn]))
        for conn in rem_nsdfg.out_connectors:
            an = st.add_access(conn)
            st.add_memlet_path(node, mx, an, src_conn=conn, memlet=dace.Memlet.from_array(conn, body.arrays[conn]))
