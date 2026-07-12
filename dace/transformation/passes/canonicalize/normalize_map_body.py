# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize a map body to a single NestedSDFG (or all tasklets).

MapFusion co-locates independent computations under one map, but leaves each as
its own ``NestedSDFG`` -- ``map: { nsdfg1, nsdfg2 }``. That form blocks
downstream normalization: two same-condition guards trapped in separate nested
SDFGs are never merged (``ConditionFusion`` only matches consecutive
``ConditionalBlock``s in ONE control-flow graph), and ``MoveMapInvariantIfUp``
cannot hoist a guard split across two nested SDFGs.

This pass consolidates every map body that mixes control flow with siblings into
exactly one NestedSDFG whose control-flow graph *sequences* the merged bodies
(``map: { nsdfg{ body1; body2 } }``). A subsequent ``ConditionFusion`` then folds
the now-consecutive same-condition guards, and the guard can hoist. The canonical
target form is ``map_consists_of_single_nsdfg_or_no_nsdfg``: a map body is either
all tasklets (no control flow, left untouched) or exactly one NestedSDFG.
"""
import copy
from typing import Dict, List, Optional, Tuple

from dace import SDFG, data, nodes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFGState
from dace.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation import pass_pipeline as ppl


def _map_body_nsdfgs(state: SDFGState, map_entry: nodes.MapEntry) -> List[nodes.NestedSDFG]:
    """The NestedSDFG nodes inside ``map_entry``'s scope, in dependency
    (topological) order so a producer is always merged before its consumer."""
    body = set(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    order = {n: i for i, n in enumerate(sdutil.dfs_topological_sort(state))}
    return sorted((n for n in body if isinstance(n, nodes.NestedSDFG)), key=lambda n: order.get(n, 0))


def _map_body_size(state: SDFGState, map_entry: nodes.MapEntry) -> int:
    return sum(1 for n in state.all_nodes_between(map_entry, state.exit_node(map_entry))
               if not isinstance(n, (nodes.MapEntry, nodes.MapExit)))


def _uniquify_data_against(inner: SDFG, taken_data) -> dict:
    """Rename ``inner``'s data descriptors that collide with ``taken_data`` to
    fresh names, so ``inner`` can be spliced into another SDFG without a clash.

    Symbols are deliberately NOT renamed: the sibling bodies come from the same
    replicated NestedSDFG, so they already agree on symbol names (the map index
    and shape symbols), and renaming would sever that sharing. A collision on a
    connector name (e.g. both read ``a``) is resolved by giving the tail a fresh
    connector reading the same outer array -- redundant but correct; a later
    SimplifyPass dedups it.
    """
    drepl = {}
    for name in list(inner.arrays.keys()):
        if name in taken_data:
            drepl[name] = data.find_new_name(name, taken_data | set(inner.arrays.keys()) | set(drepl.values()))
    if drepl:
        replace_datadesc_names(inner, drepl)
    return drepl


def _append_cfg(base: SDFG, tail: SDFG) -> None:
    """Splice ``tail``'s control-flow graph after ``base``'s sink, in place.

    ``tail`` is a private deep copy; its blocks are moved into ``base`` and an
    unconditional edge links ``base``'s sink to ``tail``'s (former) start block.
    """
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
    """Fold the redundant boundary plumbing that sibling consolidation created.

    ``MapFusion`` co-locates independent computations that read the SAME outer
    array (``g``) at the SAME iterator; the sibling merge renamed the second copy
    to a fresh connector (``g_0``) and kept its own iterator symbol
    (``_loop_it_1``, bound to the same outer value as ``_loop_it_0``). The two
    now-consecutive guards then read ``g[_loop_it_0]`` vs ``g_0[_loop_it_1]`` --
    the same predicate, but syntactically distinct, so the follow-up
    ``ConditionFusion`` builds a cartesian product instead of folding them.

    This collapses each duplicate onto its canonical name so the guards become
    syntactically identical:

    * an in-connector whose incoming memlet is identical (same source node +
      source connector + data + subset) to an earlier one is redundant -- rename
      its inner descriptor to the earlier connector and drop the extra edge;
    * two symbol-mapping keys bound to the identical outer value collapse to one.

    Value-preserving: the merged names denote the same data / same value. Only
    reads (in-connectors) are deduplicated; writes are left untouched.
    """
    base = keep.sdfg

    # Symbol-mapping dedup: group keys by the outer value they bind to, and
    # collapse each >=2 group onto one canonical key (prefer the key whose name
    # already IS the bound value, e.g. the map parameter).
    groups: Dict[str, List[str]] = {}
    for k, v in keep.symbol_mapping.items():
        groups.setdefault(str(v), []).append(k)
    sym_repl: Dict[str, str] = {}
    for value_str, keys in groups.items():
        if len(keys) < 2:
            continue
        canon = next((k for k in keys if k == value_str), sorted(keys)[0])
        for k in keys:
            if k != canon:
                sym_repl[k] = canon
    if sym_repl:
        # replace_keys=False: rewrite symbol USES (memlets, conditions, ranges)
        # but not the descriptor/symbol dict keys -- the redundant keys are
        # dropped explicitly below.
        base.replace_dict(sym_repl, replace_keys=False)
        for old in sym_repl:
            keep.symbol_mapping.pop(old, None)
            base.symbols.pop(old, None)

    # In-connector dedup: an in-edge whose outer memlet matches an earlier one
    # reads identical data; rename its inner descriptor onto the earlier
    # connector and drop the duplicate edge + connector.
    seen: Dict[Tuple, str] = {}
    data_repl: Dict[str, str] = {}
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
    """Consolidate each control-flow-bearing map body into a single NestedSDFG.

    A map body of all tasklets (no nested SDFG) is left untouched. A body with a
    single NestedSDFG is already normalized. A body with >=2 NestedSDFGs, or one
    NestedSDFG plus sibling tasklets, is merged into one NestedSDFG that
    sequences the sibling bodies, exposing same-condition guards as consecutive
    ConditionalBlocks for ``ConditionFusion``.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        merged = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not (isinstance(n, nodes.MapEntry) and isinstance(g, SDFGState)):
                continue
            body_nsdfgs = _map_body_nsdfgs(g, n)
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
        return merged or None

    def _merge_siblings(self, state: SDFGState, siblings: List[nodes.NestedSDFG]) -> bool:
        """Merge sibling NestedSDFGs in ``state`` into ``siblings[0]`` by
        sequencing their control-flow graphs. Returns True on success."""
        keep = siblings[0]
        base = keep.sdfg
        for drop in siblings[1:]:
            tail = copy.deepcopy(drop.sdfg)
            # Rename tail's data that collides with any base identifier, tracking how
            # drop's connectors were renamed so we can rewire the outer edges. DaCe forbids
            # an array name that also names a symbol, constant, or tasklet connector -- so
            # the reserved set is base's arrays PLUS its symbols, constants, and every node
            # connector (e.g. symm's ``reset_tmp`` writes a connector ``tmp``; a tail array
            # ``tmp`` merged in unchecked would collide with it).
            reserved = set(base.arrays.keys()) | set(base.symbols.keys()) | set(base.constants_prop.keys())
            for bstate in base.all_states():
                for bnode in bstate.nodes():
                    reserved |= set(bnode.in_connectors.keys()) | set(bnode.out_connectors.keys())
            drepl = _uniquify_data_against(tail, reserved)
            # Carry tail's (now non-colliding) data descriptors + symbols into base.
            for name, desc in list(tail.arrays.items()):
                if name not in base.arrays:
                    base.add_datadesc(name, desc)
            for sym, stype in tail.symbols.items():
                if sym not in base.symbols:
                    base.add_symbol(sym, stype)
            for cname, cval in tail.constants_prop.items():
                if cname not in base.constants_prop:
                    base.add_constant(cname, cval)
            _append_cfg(base, tail)

            # Re-point drop's boundary edges onto keep with (possibly renamed) connectors.
            # ``force=True``: a carrier read by one sibling and written by another (an
            # in-place update like ``b`` in TSVC s212 -- ``a = a*b`` then ``b = b + ...``)
            # legitimately becomes BOTH an in- and an out-connector of the merged nested
            # SDFG. Without ``force`` the second ``add_*_connector`` silently no-ops
            # because the name already exists on the other side, leaving an edge that
            # references a nonexistent connector (invalid SDFG: "b written but only given
            # as an input connector").
            for e in list(state.in_edges(drop)):
                conn = drepl.get(e.dst_conn, e.dst_conn)
                keep.add_in_connector(conn, force=True)
                state.add_edge(e.src, e.src_conn, keep, conn, copy.deepcopy(e.data))
                state.remove_edge(e)
            for e in list(state.out_edges(drop)):
                conn = drepl.get(e.src_conn, e.src_conn)
                keep.add_out_connector(conn, force=True)
                state.add_edge(keep, conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
                state.remove_edge(e)

            # Merge symbol mapping (rename keys that were renamed inside tail).
            for k, v in drop.symbol_mapping.items():
                keep.symbol_mapping.setdefault(k, v)
            state.remove_node(drop)

        # Fold redundant boundary connectors / symbol-mapping aliases the merge
        # introduced, so same-condition guards from independent-but-identical
        # reads (``g[i]`` in both siblings) become syntactically identical and
        # the follow-up ConditionFusion folds them instead of building a
        # cartesian product.
        _dedup_boundary_aliases(state, keep)

        sdutil.set_nested_sdfg_parent_references(base)
        for blk in base.all_control_flow_blocks(recursive=True):
            blk.sdfg = base
        return True
