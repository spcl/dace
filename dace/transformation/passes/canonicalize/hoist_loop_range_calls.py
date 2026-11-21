# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bind a loop STEP that contains a call to a symbol, so codegen emits a plain name."""

import itertools
from collections.abc import Iterator
from typing import Any

import sympy

from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_map_entry
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import scopes

#: Prefix for a minted range symbol.
RANGE_SYMBOL_PREFIX = "__dace_rng_"


def contains_call(expr: symbolic.SymbolicType) -> bool:
    """Does ``expr`` render as anything other than names, numbers and operators?"""
    parsed = symbolic.pystr_to_symbolic(str(expr))
    if not isinstance(parsed, sympy.Basic):
        return False
    return any(isinstance(node, sympy.Function) for node in sympy.preorder_traversal(parsed))


@properties.make_properties
@transformation.explicit_cf_compatible
class HoistLoopRangeCalls(ppl.Pass):
    """Replace every call-bearing loop bound or step with a symbol assigned before the loop."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.InterstateEdges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many range components were bound to symbols, or ``None`` if none were."""
        counter = itertools.count(self._next_free_index(sdfg))
        # ONE resolver for the whole run: ``symbols_defined_at`` rebuilds the SDFG base and
        # ``scope_dict`` per call, and a call-bearing step is looked up once per candidate map.
        resolver = scopes.ScopedSymbolResolver()
        bound = 0
        for cfg in list(sdfg.states()):
            for node in list(cfg.nodes()):
                if not isinstance(node, nodes.MapEntry):
                    continue
                bound += self._bind_map_range(cfg, node, counter, resolver)
        return bound or None

    def _next_free_index(self, sdfg: SDFG) -> int:
        """One past the highest index any nested graph already used, so a name is never reused."""
        used = [
            int(name[len(RANGE_SYMBOL_PREFIX) :])
            for nested in sdfg.all_sdfgs_recursive()
            for name in nested.symbols
            if name.startswith(RANGE_SYMBOL_PREFIX) and name[len(RANGE_SYMBOL_PREFIX) :].isdigit()
        ]
        return max(used) + 1 if used else 0

    def step_dtype(
        self,
        state: SDFGState,
        entry: nodes.MapEntry,
        step: symbolic.SymbolicType,
        resolver: scopes.ScopedSymbolResolver,
    ) -> dtypes.typeclass:
        """The dtype the minted step symbol must be DECLARED with, so the declaration and the instance placed in the
        range are one symbol."""
        operands = [
            resolver.resolve_dtype(sym.name, state.sdfg, state, entry)
            for sym in sorted(symbolic.pystr_to_symbolic(str(step)).free_symbols, key=lambda s: s.name)
        ]
        if not operands:
            # A call over numbers folds before it reaches here (``int_ceil(7, 2)`` is 4).
            raise scopes.UndeterminedSymbolDType(str(step), state.sdfg.label)
        return dtypes.result_type_of(*operands)

    def _bind_map_range(
        self, state: SDFGState, entry: nodes.MapEntry, counter: Iterator[int], resolver: scopes.ScopedSymbolResolver
    ) -> int:
        sdfg = state.sdfg
        parent: ControlFlowRegion = state.parent_graph
        # An interstate assignment is evaluated at STATE scope, so a component may only be hoisted when every symbol it
        # names is live there.
        enclosing = set()
        scope = state.scope_dict()
        parent_entry = scope.get(entry)
        while parent_entry is not None:
            enclosing.update(as_map_entry(parent_entry).map.params)
            parent_entry = scope.get(parent_entry)
        assignments: list[tuple[str, str]] = []
        ranges = []
        for begin, end, step in entry.map.range:
            # ONLY the step.
            if not contains_call(step) or (
                {str(sym) for sym in symbolic.pystr_to_symbolic(str(step)).free_symbols} & enclosing
            ):
                ranges.append((begin, end, step))
                continue
            name = f"{RANGE_SYMBOL_PREFIX}{next(counter)}"
            dtype = self.step_dtype(state, entry, step, resolver)
            sdfg.add_symbol(name, dtype)
            assignments.append((name, symbolic.symstr(step)))
            # ``symbol(name, dtype)``, never ``pystr_to_symbolic(name)``: the latter mints the
            # default int32 and would disagree with the declaration above.
            ranges.append((begin, end, symbolic.symbol(name, dtype)))
        if not assignments:
            return 0
        entry.map.range = subsets.Range(ranges)

        # The value has to exist before the state runs, and an interstate edge is the only place a symbol may be
        # assigned.
        in_edges = parent.in_edges(state)
        if not in_edges:
            parent.add_state_before(state, label=f"{state.label}_range_bind")
            in_edges = parent.in_edges(state)
        for edge in in_edges:
            for name, value in assignments:
                edge.data.assignments[name] = value
        # A new SDFG symbol and a new interstate assignment both feed the per-SDFG base every scoped table starts from.
        resolver.invalidate_sdfg(sdfg)
        return len(assignments)
