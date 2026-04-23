# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Utilities that operate on constructed SDFGs.

This module currently hosts :func:`prune_unused_nsdfg_connectors` and its
recursive variant, which drop input/output connectors of a ``NestedSDFG``
whose associated arrays are never accessed inside the nested SDFG. Dead
connectors block ``InlineSDFG`` and balloon the ``symbol_mapping``; cleaning
them up is a common prerequisite before inlining.
"""
import re
from typing import Set

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion

_TOKEN_SPLIT_RE = re.compile(r'[()\[\]\s,+\-*/%<>!=&|^~?:]+')


def _tokens(expr: str) -> Set[str]:
    return {s.strip() for s in _TOKEN_SPLIT_RE.split(expr) if s.strip()}


def _array_is_used_in_the_sdfg(sdfg: dace.SDFG, arr_name: str) -> bool:
    """Returns True if ``arr_name`` appears as an access node, in any
    interstate-edge assignment (LHS or RHS), in any conditional-block branch
    condition, or in any loop init/condition/update statement inside
    ``sdfg``. Conservative: a substring-over-tokens check is used for
    expressions, so false positives are possible but false negatives are
    not."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == arr_name:
                return True

    for edge in sdfg.all_interstate_edges():
        if arr_name in edge.data.assignments:
            return True
        for v in edge.data.assignments.values():
            if arr_name in _tokens(str(v)):
                return True
        cond = edge.data.condition.as_string if edge.data.condition is not None else ""
        if cond and arr_name in _tokens(cond):
            return True

    for node in sdfg.all_control_flow_blocks():
        if isinstance(node, ConditionalBlock):
            for cond, _ in node.branches:
                if cond is None:
                    continue
                text = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
                if arr_name in _tokens(text):
                    return True

    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for attr in ("loop_condition", "update_statement", "init_statement"):
                code = getattr(region, attr, None)
                text = code.as_string if isinstance(code, CodeBlock) else (str(code) if code else "")
                if text and arr_name in _tokens(text):
                    return True

    return False


def _prune_memlet_path(state: dace.SDFGState, edge):
    """Remove ``edge`` together with the full memlet path it belongs to,
    cleaning up connectors on any intermediate map entries/exits and any
    access-node taps that become orphan as a result."""
    for e in list(state.memlet_path(edge)):
        if e not in state.edges():
            continue
        state.remove_edge(e)
        if e.src_conn is not None:
            try:
                e.src.remove_out_connector(e.src_conn)
            except (KeyError, ValueError):
                pass
        if e.dst_conn is not None:
            try:
                e.dst.remove_in_connector(e.dst_conn)
            except (KeyError, ValueError):
                pass
        for ep in (e.src, e.dst):
            if (isinstance(ep, dace.nodes.AccessNode) and ep in state.nodes() and state.degree(ep) == 0):
                state.remove_node(ep)


def prune_unused_nsdfg_connectors(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG) -> int:
    """Drop input/output connectors of ``nsdfg`` whose associated arrays are
    never accessed inside its inner SDFG.

    For each dead connector, the surrounding memlet path is walked outward
    and pruned: access-node taps outside any map scope are removed when they
    become orphan, and the corresponding ``IN_*`` / ``OUT_*`` connector pair
    on any enclosing map entry/exit is cleaned up. Inner arrays that become
    unused as a result are dropped from ``nsdfg.sdfg`` so validation doesn't
    trip on a non-transient array without a feeding connector.

    Returns the number of connector names removed.
    """
    inner = nsdfg.sdfg
    names = set(nsdfg.in_connectors) | set(nsdfg.out_connectors)
    removed = 0
    for name in names:
        if _array_is_used_in_the_sdfg(inner, name):
            continue

        for ie in list(state.in_edges_by_connector(nsdfg, name)):
            _prune_memlet_path(state, ie)
        for oe in list(state.out_edges_by_connector(nsdfg, name)):
            _prune_memlet_path(state, oe)

        if name in nsdfg.in_connectors:
            nsdfg.remove_in_connector(name)
        if name in nsdfg.out_connectors:
            nsdfg.remove_out_connector(name)

        if name in inner.arrays and not _array_is_used_in_the_sdfg(inner, name):
            try:
                inner.remove_data(name, validate=False)
            except Exception:
                inner.arrays.pop(name, None)
        removed += 1
    return removed


def prune_unused_nsdfg_connectors_recursive(sdfg: dace.SDFG) -> int:
    """Apply :func:`prune_unused_nsdfg_connectors` to every ``NestedSDFG`` in
    the SDFG hierarchy, bottom-up so that outer nested SDFGs see already
    cleaned inner ones."""
    total = 0
    for state in sdfg.all_states():
        for node in list(state.nodes()):
            if isinstance(node, dace.nodes.NestedSDFG):
                total += prune_unused_nsdfg_connectors_recursive(node.sdfg)
                total += prune_unused_nsdfg_connectors(state, node)
    return total
