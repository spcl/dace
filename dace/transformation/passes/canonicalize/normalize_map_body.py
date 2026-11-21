# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize a map body to a single NestedSDFG (or all tasklets)."""

import copy
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from dace import SDFG, nodes, properties, subsets
from dace.optionals import required
from dace.sdfg import InterstateEdge, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.replace import replace_datadesc_names
from dace.sdfg.sdfg import scope_bound_names
from dace.sdfg.state import SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import map_scope
from dace.utils import find_new_name


def _map_body_nsdfgs(
    state: SDFGState, map_entry: nodes.MapEntry, order_of: Callable[[SDFGState], dict[nodes.Node, int]]
) -> list[nodes.NestedSDFG]:
    """The NestedSDFG nodes in ``map_entry``'s OWN scope, in dependency (topological) order so a producer is always
    merged before its consumer."""
    # THIS scope only: ``scope_children`` lists exactly the nodes whose scope is ``map_entry``.
    nsdfgs = [n for n in state.scope_children()[map_entry] if isinstance(n, nodes.NestedSDFG)]
    if len(nsdfgs) < 2:
        return nsdfgs
    order = order_of(state)
    # Total key: a tie (a body node the topological sort did not cover) decides which sibling becomes the
    # merge base, so break it on the node id rather than on traversal order.
    return sorted(nsdfgs, key=lambda n: (order.get(n, len(order)), state.node_id(n)))


def _map_body_size(state: SDFGState, map_entry: nodes.MapEntry) -> int:
    """Body nodes excluding inner scope entries/exits."""
    return sum(
        1 for n in map_scope.map_body_nodes(state, map_entry) if not isinstance(n, (nodes.MapEntry, nodes.MapExit))
    )


def _uniquify_data_against(inner: SDFG, taken_data: Iterable[str]) -> dict[str, str]:
    """Rename ``inner``'s data descriptors that collide with ``taken_data`` to fresh names, so ``inner`` can be spliced
    into another SDFG without a clash."""
    drepl = {}
    for name in list(inner.arrays.keys()):
        if name in taken_data:
            drepl[name] = find_new_name(name, dict.fromkeys([*taken_data, *inner.arrays.keys(), *drepl.values()]))
    if drepl:
        replace_datadesc_names(inner, drepl)
    return drepl


def shared_carrier_connectors(
    state: SDFGState, keep: nodes.NestedSDFG, drop: nodes.NestedSDFG
) -> dict[str, tuple[str, nodes.AccessNode, bool]] | None:
    """Connectors of ``drop`` that read a container ``keep`` writes, or ``None`` to refuse the merge."""
    produced = {e.data.data: e.src_conn for e in state.out_edges(keep) if e.data.data is not None}
    consumed = {e.data.data for e in state.in_edges(keep) if e.data.data is not None}
    if any(e.data.data in consumed for e in state.out_edges(drop) if e.data.data is not None):
        return None

    forward: dict[str, tuple[str, nodes.AccessNode, bool]] = {}
    # One whole-SDFG count for all carriers, built on first need -- most edges never reach it.
    elsewhere: Counter | None = None
    for e in state.in_edges(drop):
        if e.data.data not in produced or not isinstance(e.src, nodes.AccessNode):
            continue
        carrier, name = e.src, e.data.data
        if any(ie.src is not keep for ie in state.in_edges(carrier)):
            return None
        private = all(oe.dst is drop for oe in state.out_edges(carrier))
        if private and elsewhere is None:
            elsewhere = Counter(n.data for st in state.sdfg.states() for n in st.data_nodes())
        if private and (required(elsewhere)[name] != 1 or not required(state.sdfg.arrays[required(name)]).transient):
            return None  # something else observes it: cannot be made internal
        forward[e.dst_conn] = (produced[name], carrier, private)
    return forward


def co_written_connectors(state: SDFGState, keep: nodes.NestedSDFG, drop: nodes.NestedSDFG) -> dict[str, str]:
    """Out-connectors of ``drop`` that write the access node ``keep`` writes too, each with ``keep``'s connector."""
    # Through an access node only: two writes through a map exit leave by connectors of their own
    plain = [
        e
        for e in state.out_edges(keep)
        if e.src_conn is not None and e.data.data and e.data.wcr is None and isinstance(e.dst, nodes.AccessNode)
    ]
    written = {(e.dst, e.data.data): e.src_conn for e in plain}
    return {
        e.src_conn: written[e.dst, e.data.data]
        for e in state.out_edges(drop)
        if e.src_conn is not None and e.data.wcr is None and (e.dst, e.data.data) in written
    }


def bound_apart(keep: nodes.NestedSDFG, drop: nodes.NestedSDFG) -> dict[str, str]:
    """New names for the symbols of ``drop`` that ``keep`` binds to another value or binds in a scope of its body."""
    keep_bound = scope_bound_names(keep.sdfg)
    drop_bound = scope_bound_names(drop.sdfg)
    taken = dict.fromkeys(
        [*keep.sdfg.symbols, *drop.sdfg.symbols, *keep.symbol_mapping, *drop.symbol_mapping, *keep_bound, *drop_bound]
    )
    repl: dict[str, str] = {}
    for name, value in drop.symbol_mapping.items():
        if name in keep.symbol_mapping and str(keep.symbol_mapping[name]) != str(value):
            repl[name] = find_new_name(name, taken)
        elif name in keep_bound:
            same_value = (
                other
                for other, other_value in keep.symbol_mapping.items()
                if str(other_value) == str(value)
                and other not in keep_bound | drop_bound
                and other not in drop.sdfg.symbols
                and other not in drop.sdfg.arrays
            )
            repl[name] = next(same_value, None) or find_new_name(name, taken)
        else:
            continue
        taken[repl[name]] = None
    return repl


def bound_apart_in_tail(base: SDFG, tail: SDFG) -> dict[str, str]:
    """Fresh names for the parameters a scope of ``tail`` binds that ``base`` declares as symbols: sequenced into
    ``base``, such a scope would bind a symbol of the merged body."""
    clash = base.symbols.keys() & scope_bound_names(tail)
    taken = dict.fromkeys(
        [*base.symbols, *base.arrays, *tail.symbols, *tail.arrays, *scope_bound_names(base), *scope_bound_names(tail)]
    )
    repl: dict[str, str] = {}
    for name in sorted(clash):
        repl[name] = find_new_name(name, taken)
        taken[repl[name]] = None
    return repl


def linked_through_other_nodes(
    state: SDFGState, keep: nodes.NestedSDFG, drop: nodes.NestedSDFG, carriers: set[nodes.Node]
) -> bool:
    """Whether a path inside their map scope leads from ``keep`` to ``drop`` other than through a carrier's own edge to
    ``drop``: the other readers of a published carrier are on such paths."""
    scope_exit = state.exit_node(state.entry_node(keep))
    stack = [e.dst for e in state.out_edges(keep) if e.dst not in carriers]
    stack += [e.dst for carrier in carriers for e in state.out_edges(carrier) if e.dst is not drop]
    seen: set[nodes.Node] = set()
    while stack:
        node = stack.pop()
        if node is drop:
            return True
        if node in seen or node is scope_exit:
            continue
        seen.add(node)
        stack.extend(e.dst for e in state.out_edges(node))
    return False


def _append_cfg(base: SDFG, tail: SDFG) -> None:
    """Splice ``tail``'s control-flow graph after ``base``'s sink, in place."""
    sink = base.sink_nodes()[0]
    tail_start = tail.start_block
    blocks = list(tail.nodes())
    edges = list(tail.edges())
    for b in blocks:
        tail.remove_node(b)
        base.add_node(b, ensure_unique_name=True)  # deepcopy siblings share labels; wired by object ref
    for e in edges:
        base.add_edge(e.src, e.dst, e.data)
    tail_start.is_start_block = False
    base.add_edge(sink, tail_start, InterstateEdge())


def _dedup_boundary_aliases(state: SDFGState, keep: nodes.NestedSDFG) -> None:
    """Fold duplicate boundary reads left by sibling consolidation."""
    base = keep.sdfg

    # Symbol-mapping dedup: group keys by the outer value they bind to, and collapse each >=2 group onto one canonical
    # key (prefer the key whose name already IS the bound value, e.g. the map parameter).
    assigned = {name for edge in base.all_interstate_edges() for name in edge.data.assignments}
    groups: dict[str, list[str]] = {}
    for k, v in keep.symbol_mapping.items():
        if k not in assigned:
            groups.setdefault(str(v), []).append(k)
    sym_repl: dict[str, str] = {}
    for value_str, keys in groups.items():
        if len(keys) < 2:
            continue
        canon = next((k for k in keys if k == value_str), sorted(keys)[0])
        for k in keys:
            if k != canon:
                sym_repl[k] = canon
    if sym_repl:
        # replace_keys=False: rewrite symbol USES (memlets, conditions, ranges) but not the descriptor/symbol dict keys
        base.replace_dict(sym_repl, replace_keys=False)
        for old in sym_repl:
            keep.symbol_mapping.pop(old, None)
            if old in base.symbols:
                base.symbol_repo.remove(old)

    # In-connector dedup: an in-edge whose outer memlet matches an earlier one reads identical data
    seen: dict[tuple, str] = {}
    data_repl: dict[str, str] = {}
    for e in list(state.in_edges(keep)):
        if e.dst_conn is None:
            continue
        sig = (id(e.src), e.src_conn, e.data.data, str(e.data.subset))
        canon = seen.get(sig)
        if canon is None:
            seen[sig] = e.dst_conn
            continue
        data_repl[e.dst_conn] = canon
        state.remove_edge(e)
        keep.remove_in_connector(e.dst_conn)
    if data_repl:
        replace_datadesc_names(base, data_repl)


@properties.make_properties
class NormalizeMapBody(ppl.Pass):
    """Consolidate each control-flow-bearing map body into a single NestedSDFG."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        merged = 0
        # One topological order per state and run: a state can hold many map entries, and each used
        # to sort the whole state again. A merge rewires its state, so that state's order is dropped.
        orders: dict[SDFGState, dict[nodes.Node, int]] = {}

        def order_of(state: SDFGState) -> dict[nodes.Node, int]:
            if state not in orders:
                orders[state] = {n: i for i, n in enumerate(sdutil.dfs_topological_sort(state))}
            return orders[state]

        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, nodes.MapEntry) and isinstance(g, SDFGState)):
                continue
            body_nsdfgs = _map_body_nsdfgs(g, n, order_of)
            # Only consolidate when at least one nested SDFG is present AND the
            # body is not already a single nested SDFG (nothing else beside it).
            if not body_nsdfgs:
                continue
            if len(body_nsdfgs) == 1 and _map_body_size(g, n) == 1:
                continue
            if len(body_nsdfgs) < 2:
                continue  # 1 nsdfg + tasklets: handled by wrapping (future); skip for now
            if self._merge_siblings(g, body_nsdfgs):
                merged += 1
                orders.pop(g, None)
        return merged or None

    def _merge_siblings(self, state: SDFGState, siblings: list[nodes.NestedSDFG]) -> bool:
        """Merge sibling NestedSDFGs in ``state`` into ``siblings[0]`` by sequencing their control-flow graphs."""
        keep = siblings[0]
        base = keep.sdfg
        merged = False
        for drop in siblings[1:]:
            carried = shared_carrier_connectors(state, keep, drop)
            if carried is None:
                break
            if linked_through_other_nodes(state, keep, drop, {carrier for _, carrier, _ in carried.values()}):
                # The merged node would feed the nodes between and read them back: a cycle
                continue
            merged = True
            tail = copy.deepcopy(drop.sdfg)
            # Same-named symbols are shared, unless the siblings bind them to different values (``OTFMapFusion``
            # replicates a producer per element it recomputes, each with its own ``_i0``): those are renamed apart
            symbol_repl = bound_apart(keep, drop)
            tail.replace_dict(symbol_repl)
            tail.replace_dict(bound_apart_in_tail(base, tail))
            # Rename tail's data that collides with any base identifier, tracking how drop's connectors were renamed so
            # we can rewire the outer edges.
            reserved = dict.fromkeys([*base.arrays.keys(), *base.symbols.keys(), *base.constants_prop.keys()])
            for bstate in base.states():
                for bnode in bstate.nodes():
                    reserved.update(dict.fromkeys([*bnode.in_connectors.keys(), *bnode.out_connectors.keys()]))
            drepl = _uniquify_data_against(tail, reserved)
            # A container keep WRITES and drop READS is one value, not two: point tail's copy at
            # keep's array so it crosses the sequencing edge ``_append_cfg`` adds.
            for drop_conn, (keep_conn, _carrier, _private) in carried.items():
                renamed = drepl.get(drop_conn, drop_conn)
                if renamed != keep_conn:
                    replace_datadesc_names(tail, {renamed: keep_conn})
                    drepl[drop_conn] = keep_conn
            # A container both write through one access node is one array of the merged node as well.
            co_written = co_written_connectors(state, keep, drop)
            # Asked before the rewiring below: half rewired, the scope is not walkable
            facts = SymbolResolver().facts_at(state, keep) if co_written else None
            for drop_conn, keep_conn in co_written.items():
                renamed = drepl.get(drop_conn, drop_conn)
                if renamed != keep_conn:
                    replace_datadesc_names(tail, {renamed: keep_conn})
                    drepl[drop_conn] = keep_conn
            # Carry tail's (now non-colliding) data descriptors + symbols into base.
            for name, desc in list(tail.arrays.items()):
                if name not in base.arrays:
                    base.add_datadesc(name, desc)
            for sym, stype in tail.symbols.items():
                if sym not in base.symbols:
                    base.add_symbol(sym, stype)
            for cname, cval in tail.constants_prop.items():
                if cname not in base.constants_prop:
                    base.add_constant(cname, cval[1], cval[0])
            _append_cfg(base, tail)

            # Re-point drop's boundary edges onto keep with (possibly renamed) connectors.
            for e in list(state.in_edges(drop)):
                if e.dst_conn in carried:
                    state.remove_edge(e)  # satisfied inside; re-adding it would close a cycle
                    continue
                if e.dst_conn is None:
                    # Ordering memlet: no connector to rename, and keep->keep would be a cycle.
                    if e.src is not keep:
                        state.add_edge(e.src, e.src_conn, keep, None, copy.deepcopy(e.data))
                    state.remove_edge(e)
                    continue

                conn = drepl.get(e.dst_conn, e.dst_conn)
                keep.add_in_connector(conn, force=True)
                state.add_edge(e.src, e.src_conn, keep, conn, copy.deepcopy(e.data))
                state.remove_edge(e)
            for e in list(state.out_edges(drop)):
                if e.src_conn is None:
                    if e.dst is not keep:
                        state.add_edge(keep, None, e.dst, e.dst_conn, copy.deepcopy(e.data))
                    state.remove_edge(e)
                    continue
                if e.src_conn in co_written:
                    (write,) = [w for w in state.out_edges_by_connector(keep, co_written[e.src_conn]) if w.dst is e.dst]
                    write.data.subset = subsets.union(write.data.subset, e.data.subset, facts)
                    state.remove_edge(e)
                    continue
                conn = drepl.get(e.src_conn, e.src_conn)
                keep.add_out_connector(conn, force=True)
                state.add_edge(keep, conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
                state.remove_edge(e)

            # A carrier only ``drop`` read is now produced AND consumed inside ``keep``, so its outer plumbing is dead.
            for keep_conn, carrier, alone in carried.values():
                if not alone:
                    continue
                for oe in list(state.out_edges(keep)):
                    if oe.dst is carrier:
                        state.remove_edge(oe)
                keep.remove_out_connector(keep_conn)
                state.remove_node(carrier)
                base.arrays[keep_conn].transient = True

            # Merge symbol mapping (rename keys that were renamed inside tail).
            for k, v in drop.symbol_mapping.items():
                keep.symbol_mapping.setdefault(symbol_repl.get(k, k), v)
            state.remove_node(drop)

        # Fold redundant boundary connectors / symbol-mapping aliases the merge introduced, so same-condition guards
        # from independent-but-identical reads (``g[i]`` in both siblings) become syntactically identical and the
        # follow-up ConditionFusion folds them instead of building a cartesian product.
        _dedup_boundary_aliases(state, keep)

        # ``base.add_node`` re-homed every nested SDFG the tail carried and spliced the CFG list in place.
        for blk in base.all_control_flow_blocks():
            blk.sdfg = base
        return merged
