# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bind a loop STEP that contains a call to a symbol, so codegen emits a plain name.

``int_ceil`` / ``int_floor`` / ``Min`` / ``Max`` carry exact integer semantics that the analysis
passes depend on -- a chunked map's stride IS ``int_ceil(extent, threads)``, and rewriting it to a
division would change what the pass proved. So they have to survive optimization.

They must not survive into an emitted loop's INCREMENT. Measured against gcc: OpenMP's canonical
loop form admits only a loop-invariant integer expression as the increment, and a call there is
rejected outright ("invalid increment expression") -- while the same call in either BOUND compiles,
under ``omp for`` and ``omp simd`` alike. So this binds the step and nothing else::

    for (i = Min(a, b); i < c; i += int_ceil(d, e))
    ->
    __dace_rng_0 = int_ceil(d, e)
    for (i = Min(a, b); i < c; i += __dace_rng_0)

Runs AFTER canonicalization as a cleanup: the analysis passes need the call form, codegen needs the
symbol form, and this is the seam between them. Names are minted from a counter that never reuses a
value, so two ranges can never collide on one symbol.
"""
import itertools
from typing import Any, Dict, List, Optional, Tuple

import sympy

from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Prefix for a minted range symbol. ``__dace`` keeps it out of the ABI (``SDFG.arglist`` drops the
#: prefix on both the scalar and free-symbol paths), which is what makes minting one free.
RANGE_SYMBOL_PREFIX = '__dace_rng_'


def contains_call(expr) -> bool:
    """Does ``expr`` render as anything other than names, numbers and operators?

    Keyed on the sympy tree rather than on the emitted string: a substring match would trip on a
    symbol whose NAME happens to contain a function's name, and miss a call sympy prints in an
    unexpected form.
    """
    parsed = symbolic.pystr_to_symbolic(str(expr))
    if not isinstance(parsed, sympy.Basic):
        return False
    return any(isinstance(node, sympy.Function) for node in sympy.preorder_traversal(parsed))


@properties.make_properties
@transformation.explicit_cf_compatible
class HoistLoopRangeCalls(ppl.Pass):
    """Replace every call-bearing loop bound or step with a symbol assigned before the loop."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.InterstateEdges | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """:returns: how many range components were bound to symbols, or ``None`` if none were."""
        counter = itertools.count(self._next_free_index(sdfg))
        bound = 0
        for cfg in list(sdfg.all_states()):
            for node in list(cfg.nodes()):
                if not isinstance(node, nodes.MapEntry):
                    continue
                bound += self._bind_map_range(cfg, node, counter)
        return bound or None

    def _next_free_index(self, sdfg: SDFG) -> int:
        """One past the highest index any nested graph already used, so a name is never reused."""
        used = [
            int(name[len(RANGE_SYMBOL_PREFIX):]) for nested in sdfg.all_sdfgs_recursive() for name in nested.symbols
            if name.startswith(RANGE_SYMBOL_PREFIX) and name[len(RANGE_SYMBOL_PREFIX):].isdigit()
        ]
        return max(used) + 1 if used else 0

    def _bind_map_range(self, state: SDFGState, entry: nodes.MapEntry, counter) -> int:
        sdfg = state.sdfg
        parent: ControlFlowRegion = state.parent_graph
        # An interstate assignment is evaluated at STATE scope, so a component may only be hoisted
        # when every symbol it names is live there. An inner map's bound routinely refers to the
        # ENCLOSING map's parameter (``Min(n, chunk_i + stride - 1)``), which exists only inside
        # that scope -- hoisting it would move the read outside the scope that defines it.
        enclosing = set()
        scope = state.scope_dict()
        parent_entry = scope.get(entry)
        while parent_entry is not None:
            enclosing.update(parent_entry.map.params)
            parent_entry = scope.get(parent_entry)
        assignments: List[Tuple[str, str]] = []
        ranges = []
        for begin, end, step in entry.map.range:
            # ONLY the step. Measured against gcc: a call in an ``omp for`` INCREMENT is rejected
            # ("invalid increment expression") because OpenMP's canonical form admits only a
            # loop-invariant integer expression there, while the same call in either BOUND compiles
            # in both ``omp for`` and ``omp simd``. So the begin and end are left exactly as the
            # analysis passes built them, and only the stride is bound to a name.
            if not contains_call(step) or ({str(sym)
                                            for sym in symbolic.pystr_to_symbolic(str(step)).free_symbols} & enclosing):
                ranges.append((begin, end, step))
                continue
            name = f'{RANGE_SYMBOL_PREFIX}{next(counter)}'
            sdfg.add_symbol(name, dtypes.int64)
            assignments.append((name, symbolic.symstr(step)))
            ranges.append((begin, end, symbolic.pystr_to_symbolic(name)))
        if not assignments:
            return 0
        entry.map.range = subsets.Range(ranges)

        # The value has to exist before the state runs, and an interstate edge is the only place a
        # symbol may be assigned. A state with no predecessor gets one, so the start block does not
        # silently miss the definition.
        in_edges = parent.in_edges(state)
        if not in_edges:
            pre = parent.add_state_before(state, label=f'{state.label}_range_bind')
            in_edges = parent.in_edges(state)
        for edge in in_edges:
            for name, value in assignments:
                edge.data.assignments[name] = value
        return len(assignments)
