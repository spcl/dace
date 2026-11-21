# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Re-express a Map whose accesses run BACKWARDS through memory so that they run forwards."""

from typing import Any

import sympy

from dace import SDFG, symbolic
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import SDFGState, SymbolResolver
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


def access_coefficients(
    state: SDFGState, entry: nodes.MapEntry, param: str, symbols: SymbolResolver | None = None
) -> list[int | None] | None:
    """Sign of ``param``'s coefficient in every data access inside ``entry``'s scope."""
    sym = symbolic.pystr_to_symbolic(param)
    # Only a symbolic coefficient needs the facts at ``entry``; a number's sign is its own
    facts = None
    signs: list[int | None] = []
    for edge in state.scope_subgraph(entry).edges():
        if edge.data is None or edge.data.is_empty():
            continue
        for subset in (edge.data.subset, edge.data.other_subset):
            if not isinstance(subset, Range):
                continue
            for rng in subset.ranges:
                for expr in rng:
                    if not isinstance(expr, sympy.Basic) or sym not in expr.free_symbols:
                        continue
                    coeff = as_expr(expr).coeff(as_expr(sym), 1)
                    # Affine iff removing the linear term leaves the parameter behind entirely.
                    if (
                        sym in required(coeff).free_symbols
                        or sym in as_basic(symbolic.simplify(expr - coeff * sym)).free_symbols
                    ):
                        return None
                    if coeff.is_Number:
                        if coeff == 0:
                            return None
                        if coeff > 0:
                            return [1]  # one ascending access keeps the map as it is, whatever the rest reads
                        signs.append(-1)
                        continue
                    if facts is None:
                        facts = (SymbolResolver() if symbols is None else symbols).facts_at(state, entry)
                    if (
                        symbolic.ask(symbolic.predicate_relation(symbolic.Predicate.NEGATIVE, required(coeff)), facts)
                        is symbolic.Truth.TRUE
                    ):
                        signs.append(-1)
                    elif (
                        symbolic.ask(symbolic.predicate_relation(symbolic.Predicate.POSITIVE, required(coeff)), facts)
                        is symbolic.Truth.TRUE
                    ):
                        return [1]
                    else:
                        return None  # a sign we cannot decide is not a direction we may flip
    return signs


def reverse_descending_maps(sdfg: SDFG) -> int | None:
    """Flip every map parameter whose accesses all descend, in place."""
    flipped = 0
    for g in sdfg.all_sdfgs_recursive():
        for state in g.states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                # The facts at ``entry`` hold around it
                symbols = SymbolResolver()
                for dim, param in enumerate(entry.map.params):
                    begin, end, step = entry.map.range[dim]
                    if symbolic.simplify(step - 1) != 0:
                        continue  # lo + hi - p only re-covers the range at unit stride
                    signs = access_coefficients(state, entry, param, symbols)
                    if not signs or any(required(s) > 0 for s in signs):
                        continue
                    state.scope_subgraph(entry).replace(param, f"({begin} + {end} - {param})")
                    flipped += 1
    return flipped or None


@xf.explicit_cf_compatible
class ReverseMapTraversal(ppl.Pass):
    """Re-express a Map whose data accesses all run backwards so that they run forwards."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        return reverse_descending_maps(sdfg)
