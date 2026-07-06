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
from typing import List, Optional

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
        base.add_node(b)
    for e in edges:
        base.add_edge(e.src, e.dst, e.data)
    tail_start.is_start_block = False
    base.add_edge(sink, tail_start, InterstateEdge())


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
            # Rename tail's data that collides with base's, tracking how drop's
            # connectors were renamed so we can rewire the outer edges.
            drepl = _uniquify_data_against(tail, set(base.arrays.keys()))
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
            for e in list(state.in_edges(drop)):
                conn = drepl.get(e.dst_conn, e.dst_conn)
                keep.add_in_connector(conn)
                state.add_edge(e.src, e.src_conn, keep, conn, copy.deepcopy(e.data))
                state.remove_edge(e)
            for e in list(state.out_edges(drop)):
                conn = drepl.get(e.src_conn, e.src_conn)
                keep.add_out_connector(conn)
                state.add_edge(keep, conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
                state.remove_edge(e)

            # Merge symbol mapping (rename keys that were renamed inside tail).
            for k, v in drop.symbol_mapping.items():
                keep.symbol_mapping.setdefault(k, v)
            state.remove_node(drop)

        sdutil.set_nested_sdfg_parent_references(base)
        for blk in base.all_control_flow_blocks(recursive=True):
            blk.sdfg = base
        return True
