# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage every ``Tasklet -> Global -> Tasklet`` hop through fresh transient scalars.

A non-transient (global / argument) array access node bridging producer and
consumer tasklets inside a Map body forces a global-memory round-trip and a
false serialization that blocks vectorization / tiling. This pass rewrites
each such bridge into a per-distinct-subset scalar fan-out:

- One transient scalar per **distinct** array subset touched by the bridge.
  A subset that appears on both sides of the bridge (RMW) shares a single
  scalar; subsets that are only written or only read get their own.
- Write subsets drain out through every enclosing ``MapExit`` to the outer
  array access node via :meth:`SDFGState.add_memlet_path`.
- Read-only subsets source in through every enclosing ``MapEntry`` from the
  outer source access node, same routing.
- A **W x R cross-product of empty dependency edges** between every write
  scalar and every read-only scalar enforces all-writes-before-all-reads
  ordering (the safe approximation when sibling subsets cannot be proven
  disjoint -- the cloudsc ``zsolqa`` swap-subscript family is the
  motivating case).
- The bridge is dropped once isolated.

**Body restriction**: only Map bodies whose every node is a ``Tasklet`` or
``AccessNode`` are eligible. A NestedSDFG (or any other compound) in the
body disqualifies the whole Map -- the read / write classifier would have
to descend into opaque dataflow that the per-subset rewrite does not model.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG
from dace import dtypes
from dace.memlet import Memlet
from dace.transformation import pass_pipeline as ppl, transformation

#: Storage type used for the staged scalars (kept in registers, never spilled).
_STAGED_SCALAR_STORAGE = dtypes.StorageType.Register


@transformation.explicit_cf_compatible
class StageGlobalArrayThroughScalars(ppl.Pass):
    """Stage ``Tasklet -> global -> Tasklet`` hops through per-subset transient scalars.

    Tightened multi-subset variant of the historical Case-A / Case-B
    rewrite: every distinct subset touched by the bridge gets its own
    scalar, sibling writes / reads are joined by ``add_memlet_path``
    through the full enclosing-Map chain, and shared (RMW) subsets fold
    onto a single scalar.
    """

    #: This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        """Adds descriptors / access nodes and rewires memlets."""
        return ppl.Modifies.AccessNodes | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent: nothing left to stage after the first run."""
        return False

    def depends_on(self) -> Set[type]:
        """Standalone pass: no dependencies."""
        return set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _array_side_subset(edge, array_name: str):
        """Return the subset of ``edge`` that addresses ``array_name``."""
        mem = edge.data
        if mem is None:
            return None
        if mem.data == array_name:
            return mem.subset
        return mem.other_subset

    @staticmethod
    def _is_single_element(s) -> bool:
        """True iff ``s`` provably describes exactly one element."""
        if s is None:
            return False
        try:
            return bool(dace.symbolic.simplify(s.num_elements()) == 1)
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _subset_key(s) -> str:
        """Canonical hashable key (string form) for grouping by subset."""
        return str(s).strip()

    @staticmethod
    def _enclosing_map_chain(state: 'dace.SDFGState', node) -> List[dace.nodes.MapEntry]:
        """Walk every enclosing ``MapEntry`` from innermost to outermost."""
        chain: List[dace.nodes.MapEntry] = []
        cur = state.entry_node(node)
        while cur is not None:
            chain.append(cur)
            cur = state.entry_node(cur)
        return chain

    @staticmethod
    def _carries_data_read(edge, array_name: str) -> bool:
        """Whether ``edge`` carries a non-empty memlet referencing ``array_name``."""
        m = edge.data
        if m is None or m.is_empty():
            return False
        if m.data != array_name and m.other_subset is None:
            return False
        return True

    def _eligible_map_bodies(self, state: 'dace.SDFGState') -> Dict[dace.nodes.MapEntry, Set]:
        """Build ``MapEntry -> body-node-set`` for every Map whose body is
        composed only of tasklets and access nodes. ``MapExit`` is exempt
        (it is the body's boundary, not a body node).
        """
        eligible: Dict[dace.nodes.MapEntry, Set] = {}
        for entry in state.nodes():
            if not isinstance(entry, dace.nodes.MapEntry):
                continue
            exit_node = state.exit_node(entry)
            if exit_node is None:
                continue
            body = state.all_nodes_between(entry, exit_node) or set()
            if any(not isinstance(n, (dace.nodes.Tasklet, dace.nodes.AccessNode)) for n in body):
                continue
            eligible[entry] = body
        return eligible

    @staticmethod
    def _is_nsdfg_inside_map_body(sdfg: SDFG) -> bool:
        """Whether ``sdfg`` is an NSDFG whose parent state encloses it in a Map scope.

        Per the spec the staging pass recurses into ``NestedSDFG`` s whose
        body is restricted to states + interstate edges (no Maps, no
        nested NSDFGs) AND whose ``NestedSDFG`` instance sits inside a
        Map in the parent state. Used to enable flat-state staging
        within the NSDFG without having to find an enclosing Map there.
        """
        parent_node = sdfg.parent_nsdfg_node
        parent_state = sdfg.parent
        if parent_node is None or parent_state is None:
            return False
        if parent_state.entry_node(parent_node) is None:
            return False
        for s in sdfg.states():
            for n in s.nodes():
                if isinstance(n, (dace.nodes.NestedSDFG, dace.nodes.MapEntry, dace.nodes.MapExit)):
                    return False
        return True

    def _find_outer_drain(self, state: 'dace.SDFGState', bridge, outermost_exit) -> Optional[dace.nodes.AccessNode]:
        """Locate the outer ``AccessNode`` the bridge currently drains into.

        Preferred: the bridge's existing ``... -> outermost_exit -> outer_AN``
        path. Fallback: any top-level ``AccessNode`` for the same array.
        Returns ``None`` when neither exists -- the caller refuses the
        rewrite rather than synthesising a free-floating outer node.
        """
        array_name = bridge.data
        if outermost_exit is not None:
            for fe in state.out_edges(outermost_exit):
                if isinstance(fe.dst, dace.nodes.AccessNode) and fe.dst.data == array_name:
                    return fe.dst
        scope_dict = state.scope_dict()
        for n in state.data_nodes():
            if n is bridge or scope_dict.get(n) is not None or n.data != array_name:
                continue
            return n
        return None

    def _find_outer_source(self, state: 'dace.SDFGState', bridge, outermost_entry,
                           outer_drain) -> dace.nodes.AccessNode:
        """Locate the outer ``AccessNode`` to source read-only subsets from.

        Preferred: the bridge's existing ``outer_AN -> outermost_entry -> ...``
        source path (a separate node from ``outer_drain`` when the SDFG
        has both an in- and out- access node for the same array).
        Fallback: a top-level node distinct from ``outer_drain``;
        otherwise reuses ``outer_drain``.
        """
        array_name = bridge.data
        if outermost_entry is not None:
            for ie in state.in_edges(outermost_entry):
                if isinstance(ie.src, dace.nodes.AccessNode) and ie.src.data == array_name:
                    return ie.src
        scope_dict = state.scope_dict()
        for n in state.data_nodes():
            if n is bridge or n is outer_drain or scope_dict.get(n) is not None:
                continue
            if n.data == array_name:
                return n
        return outer_drain

    @staticmethod
    def _scalar_basename(array_name: str, tag: str) -> str:
        """Build a descriptive base name for a staged scalar."""
        return f"stage_{array_name}_{tag}"

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def _collect_occurrences(self, state: 'dace.SDFGState') -> List[Tuple[dace.nodes.AccessNode, List, List]]:
        """Enumerate every stageable bridge in ``state``.

        :returns: list of ``(bridge, producer_edges, consumer_edges)``.
        """
        sdfg = state.sdfg
        eligible = self._eligible_map_bodies(state)
        body_owner: Dict[dace.nodes.Node, dace.nodes.MapEntry] = {}
        for entry, body in eligible.items():
            for n in body:
                body_owner[n] = entry

        # For an NSDFG state inside a parent Map (states-only body), every
        # tasklet / access node IS "in body" -- accept all of them.
        nsdfg_internal = self._is_nsdfg_inside_map_body(sdfg)

        occurrences: List[Tuple[dace.nodes.AccessNode, List, List]] = []
        for node in state.data_nodes():
            desc = sdfg.arrays.get(node.data)
            if desc is None or desc.transient:
                continue
            owning_entry = body_owner.get(node)
            if owning_entry is None and not nsdfg_internal:
                continue
            # For nsdfg-internal flat states ``owning_entry`` is ``None``;
            # the body filter on producer / consumer tasklets is ``None ==
            # None`` instead of matching a real entry node.
            producers = [
                e for e in state.in_edges(node) if isinstance(e.src, dace.nodes.Tasklet)
                and body_owner.get(e.src) is owning_entry and self._carries_data_read(e, node.data)
            ]
            consumers = [
                e for e in state.out_edges(node) if isinstance(e.dst, dace.nodes.Tasklet)
                and body_owner.get(e.dst) is owning_entry and self._carries_data_read(e, node.data)
            ]
            if not producers or not consumers:
                continue
            if any(e.data.wcr is not None for e in producers + consumers):
                continue
            if any(not self._is_single_element(self._array_side_subset(e, node.data)) for e in producers + consumers):
                continue
            occurrences.append((node, producers, consumers))
        return occurrences

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------
    def _apply_multi(self, sdfg: SDFG, state: 'dace.SDFGState', bridge: dace.nodes.AccessNode, producers: List,
                     consumers: List) -> bool:
        """Replace the bridge with one scalar per distinct subset plus dep edges.

        Returns ``True`` iff the rewrite fired (else the caller treats
        this occurrence as refused / unchanged).
        """
        array_name = bridge.data
        dtype = sdfg.arrays[array_name].dtype

        entries = self._enclosing_map_chain(state, bridge)
        exits = [state.exit_node(e) for e in entries]
        if any(x is None for x in exits):
            return False
        if entries:
            outermost_entry, outermost_exit = entries[-1], exits[-1]
            outer_drain = self._find_outer_drain(state, bridge, outermost_exit)
            if outer_drain is None:
                return False
            outer_source = self._find_outer_source(state, bridge, outermost_entry, outer_drain)
        else:
            # NSDFG-internal flat state inside a parent Map: there is no
            # enclosing Map chain in this state, so the staging is a
            # straight ``t_i -> scalar -> t_{i+1}`` reroute. The chain's
            # data persistence is handled by the NSDFG's in / out
            # connector views in the parent state -- we don't need to
            # synthesise drain / source endpoints inside the inner state.
            outer_drain = None
            outer_source = None

        writes_by_key: Dict[str, Dict[str, Any]] = {}
        for e in producers:
            s = self._array_side_subset(e, array_name)
            writes_by_key.setdefault(self._subset_key(s), {"subset": s, "edges": []})["edges"].append(e)
        reads_by_key: Dict[str, Dict[str, Any]] = {}
        for e in consumers:
            s = self._array_side_subset(e, array_name)
            reads_by_key.setdefault(self._subset_key(s), {"subset": s, "edges": []})["edges"].append(e)

        write_keys = set(writes_by_key.keys())
        read_keys = set(reads_by_key.keys())
        shared_keys = write_keys & read_keys
        read_only_keys = read_keys - shared_keys

        # Allocate one scalar per distinct subset (shared if RMW).
        scalar_nodes: Dict[str, dace.nodes.AccessNode] = {}
        for idx, k in enumerate(sorted(write_keys | read_keys)):
            tag = "rmw" if k in shared_keys else ("w" if k in write_keys else "r")
            name, _ = sdfg.add_scalar(self._scalar_basename(array_name, f"{tag}{idx}"),
                                      dtype,
                                      storage=_STAGED_SCALAR_STORAGE,
                                      transient=True,
                                      find_new_name=True)
            scalar_nodes[k] = state.add_access(name)

        # Apply all mutations in one shot to avoid stranding nodes
        # mid-rewrite (DaCe's scope walker would misclassify a stranded
        # scalar / orphan-bridge as top-level and cascade through the
        # MapExit). For single-element subsets, the inner and outer
        # memlets along the drain / source chain are identical, so we
        # add edges manually with fresh ``IN_<data>_N`` / ``OUT_<data>_N``
        # connector names rather than going through
        # :meth:`SDFGState.add_memlet_path`, which triggers propagation
        # the moment its empty-edge skeleton is in place.
        for k in write_keys:
            sn = scalar_nodes[k]
            for e in writes_by_key[k]["edges"]:
                if e not in state.edges():
                    continue
                state.remove_edge(e)
                state.add_edge(e.src, e.src_conn, sn, None, Memlet(f"{sn.data}[0]"))

        for k in read_keys:
            sn = scalar_nodes[k]
            for e in reads_by_key[k]["edges"]:
                if e not in state.edges():
                    continue
                state.remove_edge(e)
                state.add_edge(sn, None, e.dst, e.dst_conn, Memlet(f"{sn.data}[0]"))

        for x in exits:
            self._purge_bridge_scope_edges(state, bridge, x, array_name, direction="drain")
        for n in entries:
            self._purge_bridge_scope_edges(state, bridge, n, array_name, direction="source")

        # Drain / source paths only fire when there is an enclosing Map
        # chain in this state. The flat NSDFG-internal case relies on
        # the parent state's Map for persistence: the bridge's outer
        # data lives behind the NSDFG's connectors, so we leave the
        # scalar reroute alone.
        if entries and outer_drain is not None:
            for k in write_keys:
                self._add_scoped_path(state,
                                      src=scalar_nodes[k],
                                      scope_nodes=exits,
                                      dst=outer_drain,
                                      array_name=array_name,
                                      subset=writes_by_key[k]["subset"])

            for k in read_only_keys:
                self._add_scoped_path(state,
                                      src=outer_source,
                                      scope_nodes=list(reversed(entries)),
                                      dst=scalar_nodes[k],
                                      array_name=array_name,
                                      subset=reads_by_key[k]["subset"])

        # W x R cross-product dep edges (empty memlets) -- enforce
        # all-writes-before-all-reads ordering for cross-subset hops.
        # Shared scalars don't need dep edges (natural serialization
        # through producer / consumer).
        for wk in sorted(write_keys):
            for rk in sorted(read_only_keys):
                state.add_edge(scalar_nodes[wk], None, scalar_nodes[rk], None, Memlet())

        if bridge in state.nodes() and state.in_degree(bridge) == 0 and state.out_degree(bridge) == 0:
            state.remove_node(bridge)
        # The bridge's source purge may have left a now-isolated outer
        # input access node for the same array (it only fed the bridge's
        # MapEntry-source path). Drop any such orphans so the SDFG
        # validates.
        for n in list(state.data_nodes()):
            if n.data == array_name and state.in_degree(n) == 0 and state.out_degree(n) == 0:
                state.remove_node(n)
        return True

    @staticmethod
    def _add_scoped_path(state: 'dace.SDFGState', *, src: dace.nodes.AccessNode, scope_nodes: List,
                         dst: dace.nodes.AccessNode, array_name: str, subset) -> None:
        """Add a memlet chain ``src -> scope_0 -> ... -> scope_N -> dst``.

        For single-element subsets the propagated outer memlet equals
        the inner memlet, so we add raw edges with one matched
        ``IN_<base>`` / ``OUT_<base>`` connector pair per scope node
        instead of calling :meth:`SDFGState.add_memlet_path` (which
        would trigger propagation while the surrounding rewrite is
        mid-flight). Each scope node uses the SAME ``<base>`` for its
        in and out connectors so the pairing is well-formed.

        :param src: First node of the chain.
        :param scope_nodes: Intermediate ``MapEntry`` (source path) or
            ``MapExit`` (drain path) nodes in traversal order.
        :param dst: Last node of the chain.
        :param array_name: Data name carried along the chain.
        :param subset: Single-element subset describing the access.
        """
        # Reserve a single ``base`` per scope so its IN_/OUT_ pair matches.
        scope_bases = [n.next_connector(array_name) for n in scope_nodes]
        for n, base in zip(scope_nodes, scope_bases):
            n.add_in_connector(f"IN_{base}")
            n.add_out_connector(f"OUT_{base}")
        chain = [src, *scope_nodes, dst]
        for idx, (u, v) in enumerate(zip(chain[:-1], chain[1:])):
            mem = Memlet(data=array_name, subset=copy.deepcopy(subset))
            src_conn = f"OUT_{scope_bases[idx - 1]}" if idx > 0 else None
            dst_conn = f"IN_{scope_bases[idx]}" if idx < len(scope_nodes) else None
            state.add_edge(u, src_conn, v, dst_conn, mem)

    @staticmethod
    def _purge_bridge_scope_edges(state: 'dace.SDFGState', bridge: dace.nodes.AccessNode, scope_node, array_name: str,
                                  *, direction: str) -> None:
        """Strip the bridge's redundant ``bridge <-> scope_node`` edges and
        their orphaned connectors after the new paths are in place.

        :param direction: ``"drain"`` strips ``bridge -> MapExit -> outer``
            chains; ``"source"`` strips ``outer -> MapEntry -> bridge``.
        """
        if direction == "drain":
            edges = [
                e for e in state.out_edges(bridge) if isinstance(e.dst, dace.nodes.MapExit) and e.dst is scope_node
                and e.data is not None and e.data.data == array_name
            ]
        else:
            edges = [
                e for e in state.in_edges(bridge) if isinstance(e.src, dace.nodes.MapEntry) and e.src is scope_node
                and e.data is not None and e.data.data == array_name
            ]
        for edge in edges:
            if direction == "drain":
                in_conn, prefix = edge.dst_conn, "IN_"
                out_conn = "OUT_" + in_conn[len("IN_"):] if in_conn and in_conn.startswith("IN_") else None
                state.remove_edge(edge)
                if in_conn and in_conn in scope_node.in_connectors:
                    scope_node.remove_in_connector(in_conn)
                if out_conn and out_conn in scope_node.out_connectors:
                    for outer in list(state.out_edges(scope_node)):
                        if outer.src_conn == out_conn:
                            state.remove_edge(outer)
                    scope_node.remove_out_connector(out_conn)
            else:
                out_conn = edge.src_conn
                in_conn = "IN_" + out_conn[len("OUT_"):] if out_conn and out_conn.startswith("OUT_") else None
                state.remove_edge(edge)
                if out_conn and out_conn in scope_node.out_connectors:
                    scope_node.remove_out_connector(out_conn)
                if in_conn and in_conn in scope_node.in_connectors:
                    for outer in list(state.in_edges(scope_node)):
                        if outer.dst_conn == in_conn:
                            state.remove_edge(outer)
                    scope_node.remove_in_connector(in_conn)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def _apply(self, sdfg: SDFG) -> int:
        """Stage every eligible bridge in ``sdfg`` and recurse into NSDFGs."""
        count = 0
        for state in sdfg.all_states():
            for bridge, producers, consumers in self._collect_occurrences(state):
                if bridge not in state.nodes():
                    continue
                if self._apply_multi(sdfg, state, bridge, producers, consumers):
                    count += 1
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    count += self._apply(node.sdfg)
        return count

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the staging pass over ``sdfg``.

        :returns: Number of bridges rewritten, or ``None`` when nothing changed.
        """
        count = self._apply(sdfg)
        return count if count > 0 else None
