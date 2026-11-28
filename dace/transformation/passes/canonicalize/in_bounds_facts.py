# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower bounds on extent symbols that the program's own accesses imply."""

from typing import Any

import sympy

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import BreakBlock, ContinueBlock, LoopRegion, ReturnBlock, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis


def map_runs(entry: nodes.EntryNode, facts: symbolic.Facts) -> bool:
    """Whether ``entry`` is a map every dimension of which provably holds a point (``0:M`` may hold none)."""
    if not isinstance(entry, nodes.MapEntry):
        return False
    for begin, end, step in entry.map.range:
        step = sympy.sympify(step)
        if not (step.is_number and step > 0 and symbolic.provably_le(sympy.sympify(begin), sympy.sympify(end), facts)):
            return False
    return True


def scope_runs(state: SDFGState, entry: nodes.EntryNode) -> bool:
    """Whether everything in ``entry``'s scope runs whenever ``state`` does: every map in it provably iterates, and
    no nested SDFG in it propagates the accesses of its branches as if they ran."""
    facts = state.sdfg.facts()
    for node in state.scope_subgraph(entry).nodes():
        if isinstance(node, nodes.NestedSDFG):
            return False
        if isinstance(node, nodes.EntryNode) and not map_runs(node, facts):
            return False
    return True


def always_runs(block) -> bool:
    """Whether ``block`` runs on every execution of the root SDFG: no conditional around it, only loops and maps that
    provably iterate, no conditional edge or early exit in what holds it, up through every nested SDFG."""
    region = block.parent_graph
    while region is not None:
        if isinstance(region, SDFG):
            if any(not e.data.is_unconditional() for e in region.edges()):
                return False
            if any(isinstance(b, ReturnBlock) for b in region.all_control_flow_blocks()):
                return False
            node = region.parent_nsdfg_node
            if node is None:
                return True
            state = region.parent
            scopes = state.scope_dict()
            entry = scopes[node]
            while entry is not None:
                if not map_runs(entry, state.sdfg.facts()):
                    return False
                entry = scopes[entry]
            region = state.parent_graph
            continue
        if not isinstance(region, LoopRegion) or not loop_analysis.loop_provably_at_least_one_iteration(region):
            return False
        if any(not e.data.is_unconditional() for e in region.edges()):
            return False
        if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in region.all_control_flow_blocks()):
            return False
        region = region.parent_graph
    return False


def single_symbol_bound(slack: sympy.Expr, names: frozenset[str]) -> tuple[str, int] | None:
    """``(s, k)`` when ``slack >= 0`` says ``s >= k`` for one root symbol ``s`` (``a * s + c`` with ``a > 0``)."""
    slack = sympy.expand(slack)
    free = slack.free_symbols
    if len(free) != 1:
        return None
    (sym,) = free
    if str(sym) not in names or not slack.is_polynomial(sym):
        return None
    poly = sympy.Poly(slack, sym)
    if poly.degree() != 1:
        return None
    a, c = poly.all_coeffs()
    if not (a.is_Integer and c.is_Integer and a > 0):
        return None
    return str(sym), int(sympy.ceiling(-c / a))


def declare_in_bounds_extents(sdfg: SDFG) -> dict[str, int]:
    """Declare on ``sdfg`` (the root) the single-symbol lower bounds its always-executed accesses imply; returns them."""
    names = frozenset(sdfg.free_symbols)
    bounds: dict[str, int] = {}
    for nested in sdfg.all_sdfgs_recursive():
        node = nested.parent_nsdfg_node
        if node is not None and any(str(v) != k for k, v in node.symbol_mapping.items() if k in names):
            continue
        for state in nested.states():
            if not always_runs(state):
                continue
            scopes = state.scope_dict()
            for an in state.data_nodes():
                if scopes[an] is not None:
                    continue
                shape = nested.arrays[an.data].shape
                for edge in state.all_edges(an):
                    memlet = edge.data
                    if memlet.is_empty() or memlet.dynamic or memlet.data != an.data:
                        continue
                    # A nested SDFG's subset is propagated over its branches; its own accesses are visited with it.
                    other = edge.dst if edge.src is an else edge.src
                    if isinstance(other, nodes.NestedSDFG):
                        continue
                    # A map's memlet is propagated over its range, which may be empty
                    if isinstance(other, nodes.ExitNode):
                        other = state.entry_node(other)
                    if isinstance(other, nodes.EntryNode) and not scope_runs(state, other):
                        continue
                    subset = memlet.src_subset if edge.src is an else memlet.dst_subset
                    subset = subset if subset is not None else memlet.subset
                    if subset is None or len(subset) != len(shape):
                        continue
                    for (begin, end, _), extent in zip(subset.ndrange(), shape):
                        for slack in (begin, extent - 1 - end):
                            found = single_symbol_bound(symbolic.pystr_to_symbolic(str(slack)), names)
                            if found is not None and found[1] > bounds.get(found[0], -(1 << 62)):
                                bounds[found[0]] = found[1]
    for name, low in sorted(bounds.items()):
        relation = symbolic.comparison_relation(sympy.Ge(symbolic.symbol(name), low))
        if relation is not None and low > 1:
            sdfg.add_symbol_relation(relation)
    return {name: low for name, low in bounds.items() if low > 1}


@transformation.explicit_cf_compatible
class DeclareInBoundsExtents(ppl.Pass):
    """:func:`declare_in_bounds_extents` as a recipe unit, for the accesses lifting and parallelization surface."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> None:
        declare_in_bounds_extents(sdfg)
