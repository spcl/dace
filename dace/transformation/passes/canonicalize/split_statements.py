# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement fission: split a loop/map body into one perfect nest per
independent output *statement*, so ``LoopFission`` / ``MapFission`` can then
distribute and parallelize each statement independently.

This is the single statement-fission pass of the canonicalization design; it
handles the three shapes ordinary in-place fission cannot separate:

  * **statements inside an if** -- ``for i: if c: A[i]=..; B[i]=..`` -- the body
    is a ``NestedSDFG`` holding a ``ConditionalBlock`` (the branch cannot be
    split in place);
  * **indirect (gather/scatter) access** -- ``for i: A[i]=B[idx[i]]; C[i]=D[idx[i]]``
    -- where ``idx[i]`` is an iterator-dependent interstate-edge symbol
    assignment that ``MapFission`` cannot hoist; and
  * a **forward-read anti-dependence** -- ``for i: A[i]=..; d[i]=A[i]+A[i+1]``
    (TSVC s1244) -- the read-ahead ``A[i+1]`` binds two otherwise-independent
    statements.

For the first two the ``NestedSDFG`` is cloned once per independent output
group, deep-copying the shared condition / index-symbol interstate assignments
into every clone (this subsumes the former ``ConditionalComponentFission``).
For the third the array is snapshotted before the loop and only the read-ahead
accesses are redirected to the snapshot, reusing
:meth:`BreakAntiDependence._dep_class` (the direction-aware WAR/RAW oracle) and
:meth:`BreakAntiDependence._emit_positive_guard` (symbolic-offset soundness).

The actual distribution + parallelization is done by the passes that follow
(``LoopFission`` / ``MapFission`` / ``LoopToMap``); ``MapFusion`` re-fuses
whatever should recombine.
"""
import copy
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, Memlet, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.loop_fission import _single_compute_state


def _has_conditional(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` contains a ``ConditionalBlock`` (recursively)."""
    return any(isinstance(cfg, ConditionalBlock) for cfg in sdfg.all_control_flow_regions(recursive=True))


def _has_interstate_assignments(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` has any interstate edge carrying an assignment.

    By the canonicalization convention these encode indirect-access index
    symbols (``__sym = idx[i]``); ``MapFission`` refuses to split a map whose
    NestedSDFG body has such a map-iterator-dependent assignment, so the
    NestedSDFG must be replicated per output group first.
    """
    return any(e.data.assignments for e in sdfg.all_interstate_edges())


def _output_dependency(sdfg: SDFG, out_name: str, input_names: Set[str]) -> Set[str]:
    """Inner array names that feed ``out_name``, excluding pure shared inputs."""
    deps: Set[str] = set()
    for state in sdfg.all_states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen = set()
        stack = list(writers)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if isinstance(node, nodes.AccessNode):
                if node.data in input_names:
                    continue
                deps.add(node.data)
            for e in state.in_edges(node):
                stack.append(e.src)
    return deps


@transformation.explicit_cf_compatible
@properties.make_properties
class SplitStatements(ppl.Pass):
    """Split a loop/map body into one perfect nest per independent output
    statement -- including statements inside ifs and gather/scatter accesses
    (per-output NestedSDFG replication) and forward-read anti-dependences
    (snapshot rename). Subsumes ConditionalComponentFission.

    ``split_maps`` additionally fissions a STRAIGHT-LINE map that writes several
    global outputs into one map per output (the map analogue of the loop split);
    a shared local temp is recomputed in each, never materialized. It is OFF by
    default so the canonicalization pipeline (which lowers maps to loops and
    fissions there) is byte-identical; the nest-forge agent path turns it on to
    fission at map granularity without lowering."""

    CATEGORY: str = 'Canonicalization'

    split_maps = properties.Property(
        dtype=bool, default=False, desc="Also fission a straight-line multi-global-output map into one map per output.")

    def __init__(self, split_maps: bool = False) -> None:
        super().__init__()
        self.split_maps = split_maps

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        count = 0
        # (1) Statements inside ifs + gather/scatter: replicate the blocking
        #     NestedSDFG once per independent output group so MapFission splits it.
        count += self._replicate_components(sdfg)
        # (1b) Straight-line map with >=2 global outputs -> one map per output
        #      (opt-in; leaves the canon pipeline unchanged).
        if self.split_maps:
            count += self._split_map_bodies(sdfg)
        # (2) Forward-read anti-dependences: snapshot-rename the read-ahead so
        #     LoopFission can distribute the loop into independent statements.
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
        for loop in loops:
            count += self._snapshot_forward_reads(loop, sdfg)
        return count or None

    # ------------------------------------------------------------------
    # (1) Per-output replication of a MapFission-blocking NestedSDFG
    #     (conditional / indirection-symbol). Formerly ConditionalComponentFission.
    # ------------------------------------------------------------------

    def _replicate_components(self, sdfg: SDFG) -> int:
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(nsdfg.states()):
                for node in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    groups = self._independent_output_groups(state, node)
                    if groups is None or len(groups) < 2:
                        continue
                    self._split(nsdfg, state, node, groups, SimplifyPass)
                    count += 1
        return count

    @staticmethod
    def _independent_output_groups(state, node: nodes.NestedSDFG):
        """Partition ``node``'s output connectors into independent groups."""
        if not _has_conditional(node.sdfg) and not _has_interstate_assignments(node.sdfg):
            return None
        out_conns = [c for c in node.out_connectors]
        if len(out_conns) < 2:
            return None
        # No WCR on the boundary (it would not be replicable per group).
        for e in state.out_edges(node):
            if e.data is None or e.data.wcr is not None:
                return None
        in_names = set(node.in_connectors)
        dep = {oc: _output_dependency(node.sdfg, oc, in_names) for oc in out_conns}
        parent = {oc: oc for oc in out_conns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(out_conns):
            for b in out_conns[i + 1:]:
                if dep[a] & dep[b]:
                    parent[find(a)] = find(b)
        groups: Dict[str, Set[str]] = {}
        for oc in out_conns:
            groups.setdefault(find(oc), set()).add(oc)
        return list(groups.values())

    @staticmethod
    def _split(parent_sdfg: SDFG, state, node: nodes.NestedSDFG, groups, simplify_cls):
        """Clone ``node`` once per group, prune each, rewire, drop original."""
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        for grp in groups:
            clone_sdfg = copy.deepcopy(node.sdfg)
            # dicts/sorted, not sets: these become the clone's connector dicts, which are
            # observable in validation and codegen order.
            clone = state.add_nested_sdfg(clone_sdfg,
                                          inputs=dict.fromkeys(node.in_connectors),
                                          outputs=dict.fromkeys(sorted(grp)),
                                          symbol_mapping=dict(node.symbol_mapping))
            for e in in_edges:
                state.add_edge(e.src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                if e.src_conn in grp:
                    state.add_edge(clone, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
            for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in node.in_connectors]:
                desc = clone_sdfg.arrays[arr]
                if not desc.transient:
                    desc.transient = True
            simplify_cls().apply_pass(clone_sdfg, {})
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)

    # ------------------------------------------------------------------
    # (1b) Straight-line map statement fission (opt-in via split_maps).
    # ------------------------------------------------------------------

    def _split_map_bodies(self, sdfg: SDFG) -> int:
        """Map analogue of the loop statement split: fission a STRAIGHT-LINE map writing >=2 global
        outputs into one map per output, so each is a self-contained statement.

        Reuses :meth:`_split`: nest the flat body into a NestedSDFG so the map reads as
        ``MapEntry -> NestedSDFG(out = the global outputs) -> MapExit``, then clone that NestedSDFG once
        per output with STRICT per-output groups. ``SimplifyPass`` (inside ``_split``) drops each clone's
        dead compute while keeping a shared local producer that still feeds the kept output -- so a temp
        feeding two outputs is RECOMPUTED in each map, never materialized to an array (the over-split
        MapFission would otherwise promote it). The downstream ``MapFission`` then separates the cloned
        bodies into distinct maps.

        Only PLAIN leaf maps: a body holding a NestedSDFG (a conditional / gather-scatter index symbol)
        is left to :meth:`_replicate_components`, which already duplicates the guard / ``sym = idx[i]``
        assignment per output. A WCR output (reduction) is never split.
        """
        from dace.transformation.passes.simplify import SimplifyPass
        from dace.transformation import helpers
        from dace.transformation.interstate import InlineSDFG
        from dace.sdfg.graph import SubgraphView

        count = 0
        for cfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(cfg.states()):
                entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
                for entry in entries:
                    if entry not in state.nodes():  # a prior split in this state removed/replaced it
                        continue
                    if self._split_one_map(cfg, state, entry, SimplifyPass, helpers, InlineSDFG, SubgraphView):
                        count += 1
        return count

    @staticmethod
    def _split_one_map(cfg, state, entry, simplify_cls, helpers, inline_cls, subgraph_cls) -> bool:
        """Split one straight-line multi-output map into one FLAT map per output; return whether it fired.

        Nest the WHOLE scope (entry..exit) into a NestedSDFG, clone it per output (``_split`` duplicates a
        shared local + prunes the dead output), then inline each clone back so the result is flat maps --
        not maps buried in NestedSDFGs. The inline is confined to the clones this call made (captured by
        diff), so unrelated NestedSDFGs are untouched.
        """
        xit = state.exit_node(entry)
        # Global outputs = distinct arrays written through the exit; bail on a WCR (reduction) edge.
        out_names: List[str] = []
        for e in state.in_edges(xit):
            if e.data is None or e.data.data is None:
                continue
            if e.data.wcr is not None:
                return False
            if e.data.data not in out_names:
                out_names.append(e.data.data)
        if len(out_names) < 2:
            return False
        # In-place read-modify-write: a GLOBAL array read into the scope is also written out of it
        # (``t = A[i]; A[i] = t + B[i]; C[i] = t*2``). Cloning the scope per output would let one clone
        # recompute ``t = A[i]`` AFTER another clone overwrote ``A[i]`` (the read/modify/write order is
        # lost) and duplicate the write -- both silently wrong, and it still validates. Mirror the RMW
        # guard SplitTasklets uses (split_tasklets.py) and leave such a map unsplit. Global
        # (non-transient) arrays only: a shared local temp is meant to be recomputed per output.
        read_arrays = {
            e.data.data
            for e in state.in_edges(entry)
            if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient
        }
        write_arrays = {
            e.data.data
            for e in state.in_edges(xit)
            if e.data is not None and e.data.data is not None and not cfg.arrays[e.data.data].transient
        }
        if read_arrays & write_arrays:
            return False
        scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        inner = [n for n in scope.nodes() if n not in (entry, xit)]
        # A PLAIN dataflow NestedSDFG body (a dependent read-after-write the frontend wrapped, e.g.
        # ``A[i]=..; B[i]=A[i]*2``) is inlined FIRST, so the split sees flat tasklets and a shared local
        # is duplicated like any other. A CONDITIONAL / indirection-symbol body is NOT inlined -- it
        # cannot live directly in a map scope and is _replicate_components' job.
        if (len(inner) == 1 and isinstance(inner[0], nodes.NestedSDFG) and not _has_conditional(inner[0].sdfg)
                and not _has_interstate_assignments(inner[0].sdfg)):
            inline_cls.apply_to(cfg, nested_sdfg=inner[0], save=False, verify=False)
            scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            inner = [n for n in scope.nodes() if n not in (entry, xit)]
        # PLAIN leaf map only: no nested map / NestedSDFG in the body (those go to _replicate_components).
        if not inner or any(isinstance(n, (nodes.NestedSDFG, nodes.MapEntry, nodes.MapExit)) for n in inner):
            return False
        before = {n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)}
        nsdfg_node = helpers.nest_state_subgraph(cfg, state, subgraph_cls(state, list(scope.nodes())))
        groups = [{o} for o in nsdfg_node.out_connectors if o in out_names]
        if len(groups) < 2:  # nesting coalesced the outputs onto one connector -- nothing to split
            return False
        SplitStatements._split(cfg, state, nsdfg_node, groups, simplify_cls)
        for clone in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]:
            inline_cls.apply_to(cfg, nested_sdfg=clone, save=False, verify=False)
        return True

    # ------------------------------------------------------------------
    # (2) Forward-read anti-dependence snapshot (TSVC s1244).
    # ------------------------------------------------------------------

    def _snapshot_forward_reads(self, loop: LoopRegion, sdfg: SDFG) -> int:
        state = _single_compute_state(loop)
        if state is None:
            return 0
        ivar = loop.loop_variable
        oracle = BreakAntiDependence()
        internal_syms = oracle._loop_internal_symbols(loop)
        applied = 0

        written = sorted(
            {n.data
             for n in state.data_nodes() if state.in_degree(n) > 0 and not sdfg.arrays[n.data].transient})
        for arr in written:
            write_subsets = []
            for n in state.data_nodes():
                if n.data != arr:
                    continue
                for e in state.in_edges(n):
                    ws = e.data.get_dst_subset(e, state) if e.data is not None else None
                    if ws is not None:
                        write_subsets.append(ws)
            if not write_subsets:
                continue

            fwd_edges = []
            sym_guards = set()
            for n in list(state.data_nodes()):
                if n.data != arr:
                    continue
                for e in state.out_edges(n):
                    rs = e.data.get_src_subset(e, state) if e.data is not None else None
                    if rs is None:
                        continue
                    verdicts = [oracle._dep_class(rs, ws, ivar, loop=loop, sdfg=sdfg) for ws in write_subsets]
                    kinds = {v[0] for v in verdicts}
                    # Redirect to the pre-loop snapshot ONLY when EVERY verdict is a read-ahead
                    # (WAR / WAR_symbolic). A RAW/complex producer, OR a 'none' (offset-0, same-index
                    # producer THIS iteration), means the read consumes a value made within the sweep
                    # and must keep its live-array value -- moving it to the stale snapshot is a silent
                    # miscompile. (The old gate only skipped RAW/complex and required *some* WAR, so a
                    # read that was WAR vs one sibling write but 'none' vs another --
                    # ``A[i]=..; A[i+1]=..; d[i]=A[i+1]`` -- slipped through and read the stale value.)
                    if not (kinds and kinds <= {'WAR', 'WAR_symbolic'}):
                        continue
                    guards = {p for k, p in verdicts if k == 'WAR_symbolic'}
                    if any({str(s) for s in g.free_symbols} & internal_syms for g in guards):
                        continue
                    sym_guards |= guards
                    fwd_edges.append((n, e))
            if not fwd_edges:
                continue

            # Anti-dependence is allowed by default: snapshot the FULL array before the loop
            # and redirect only the read-ahead edges to the snapshot. With the swept sizes the
            # array already tracks the loop, and the snapshot copy lowers to a parallel memcpy,
            # so a whole-array copy is simple and cheap -- no footprint bookkeeping needed.
            desc = sdfg.arrays[arr]
            snap, _ = sdfg.add_transient(f'{arr}_split_snap',
                                         desc.shape,
                                         desc.dtype,
                                         storage=desc.storage,
                                         find_new_name=True)
            pre = loop.parent_graph.add_state_before(loop, label=f'{arr}_split_snapshot')
            pre.add_nedge(pre.add_read(arr), pre.add_write(snap), Memlet.from_array(arr, desc))
            # sorted: ``sym_guards`` is a set of sympy exprs (hashed via symbol-name strings). It is iterated
            # to EMIT tasklets into ``pre``, so its order fixes their node names/ids and the emitted C order.
            for expr in sorted(sym_guards, key=symbolic.symstr):
                # STRICT (>0) guard -- NOT the >=0 that BreakAntiDependence's whole-array
                # pure-WAR rename uses. There every read of ``arr`` moves to the snapshot and
                # a same-index read ``arr[i]`` equals the pre-loop original (only iteration i
                # writes ``arr[i]``), so a symbolic offset of 0 is sound. HERE the split is
                # the MIXED shape ``arr[i]=..; d[i]=arr[i]+arr[i+sym]``: a SIBLING statement
                # writes ``arr[i]`` earlier in the SAME iteration, so a read ``arr[i+sym]``
                # with ``sym == 0`` aliases that just-written live value and must NOT be
                # redirected to the stale snapshot. Trap unless ``sym >= 1`` (offsets are
                # integer, so ``sym - 1 >= 0`` is exactly the strict ``sym > 0``); ``sym == 0``
                # is then a loud runtime fault instead of a silent miscompile.
                oracle._emit_positive_guard(pre, expr - 1)

            for src, e in fwd_edges:
                snap_node = state.add_access(snap)
                new_mem = Memlet(data=snap, subset=e.data.get_src_subset(e, state))
                if isinstance(e.dst, nodes.AccessNode):
                    new_mem.other_subset = e.data.get_dst_subset(e, state)
                state.add_edge(snap_node, e.src_conn, e.dst, e.dst_conn, new_mem)
                state.remove_edge(e)
                if state.degree(src) == 0:
                    state.remove_node(src)
            applied += 1
        return applied


__all__ = ['SplitStatements']
