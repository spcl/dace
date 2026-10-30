# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lower bounds on extent symbols that the program's own accesses imply.

An access every execution makes must be in bounds: ``Psi[i, j, NZ - 2]`` in a state that always runs needs
``NZ >= 2``. Such a bound is often what a proof is missing (bout_elm_pb's ``ddt_P[..., 0]`` and
``ddt_P[..., NZ - 1]`` are disjoint only when ``NZ >= 2``). Only bounds on a single symbol are kept, the tightest per
symbol, and each is declared as an assumption, so it is checked when the program is called.
"""

import sympy

from dace import SDFG, symbolic
from dace.sdfg.state import BreakBlock, ContinueBlock, LoopRegion, ReturnBlock
from dace.transformation.passes.analysis import loop_analysis


def always_runs(block) -> bool:
    """Whether ``block`` runs on every execution of the root SDFG: no conditional around it, only loops that provably
    iterate, no conditional edge or early exit on the way, up through every nested SDFG."""
    region = block.parent_graph
    while region is not None:
        if isinstance(region, SDFG):
            if any(not e.data.is_unconditional() for e in region.edges()):
                return False
            if region.parent_nsdfg_node is None:
                return True
            region = region.parent.parent_graph
            continue
        if not isinstance(region, LoopRegion) or not loop_analysis.loop_provably_at_least_one_iteration(region):
            return False
        if any(not e.data.is_unconditional() for e in region.edges()):
            return False
        if any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in region.nodes()):
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
    """Declare on ``sdfg`` (the root) the single-symbol lower bounds its always-executed accesses imply; returns them.

    A nested SDFG contributes only accesses whose symbols its caller passes through unchanged.
    """
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
