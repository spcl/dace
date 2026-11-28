# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Emit a runtime guard that aborts if an assumption canonicalization relied on is violated by the values passed at the
SDFG boundary."""

from collections.abc import Iterable
from typing import Any

import sympy

from dace import SDFG, dtypes, symbolic
from dace.sdfg import SDFGState, nodes
from dace.sdfg import tasklet_utils as tutil
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import ControlFlowBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.canonicalize.tracked_assumptions import tracked_assumptions

#: A declared sign at least as strong as the contract's ``>= 0``; a contradicting one is left alone.
SIGN_PREDICATES = frozenset(
    {
        symbolic.Predicate.POSITIVE,
        symbolic.Predicate.NONNEGATIVE,
        symbolic.Predicate.NEGATIVE,
        symbolic.Predicate.NONPOSITIVE,
    }
)


#: Attribute under which the root SDFG records the symbols its caller knows to be nonnegative.
NONNEGATIVE_ATTR = "_declared_nonnegative_symbols"


def root_of(sdfg: SDFG) -> SDFG:
    while sdfg.parent_sdfg is not None:
        sdfg = sdfg.parent_sdfg
    return sdfg


def declare_nonnegative_symbols(sdfg: SDFG, names: Iterable[str]) -> None:
    """Record on the root of ``sdfg`` the symbols its caller knows to be nonnegative: the only ones canonicalization
    assumes nonnegative."""
    vars(root_of(sdfg))[NONNEGATIVE_ATTR] = frozenset(names)


def declared_nonnegative_symbols(sdfg: SDFG) -> frozenset[str]:
    """The names :func:`declare_nonnegative_symbols` recorded on the root of ``sdfg``; none when nothing was."""
    return vars(root_of(sdfg)).get(NONNEGATIVE_ATTR, frozenset())


def names_still_plain(sdfg: SDFG) -> dict[str, None]:
    """The signed-integer symbols ``sdfg`` declares with no sign predicate yet: the ones
    :func:`set_symbol_nonnegative_assumptions` would assume nonnegative (one level of ``sdfg``)."""
    predicates = sdfg.symbol_repo.params.predicates
    declared = declared_nonnegative_symbols(sdfg)
    return dict.fromkeys(
        name
        for name in sorted(sdfg.free_symbols)
        if name in declared
        and sdfg.symbols.get(name) in SIGNED_INTEGER_DTYPES
        and not predicates.get(name, frozenset()) & SIGN_PREDICATES
    )


def sized_symbols(sdfg: SDFG) -> list[sympy.Symbol]:
    """Every symbol in the sympy-typed properties ``replace_dict`` threads the assumption into:
    descriptor extents, map ranges, memlet subsets and volumes (one level of ``sdfg``)."""
    found: list[sympy.Symbol] = []

    def scan(*exprs: object) -> None:
        for expr in exprs:
            if isinstance(expr, sympy.Basic):
                found.extend(expr.free_symbols)

    for desc in sdfg.arrays.values():
        scan(*desc.shape, *desc.strides, desc.total_size, *desc.offset)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                scan(*(bound for rng in node.map.range.ndrange() for bound in rng))
        for edge in state.edges():
            scan(edge.data.volume)
            for subset in (edge.data.subset, edge.data.other_subset):
                if subset is not None:
                    scan(*(bound for rng in subset.ndrange() for bound in rng))
    return found


def passed_through(node: nodes.NestedSDFG, name: str) -> bool:
    """Whether ``node`` maps its symbol ``name`` to the parent's symbol of the same name."""
    value = node.symbol_mapping.get(name)
    return value is not None and str(value) == name


def set_symbol_nonnegative_assumptions(sdfg: SDFG) -> int | None:
    """Register as nonnegative, in the symbol repository of the root of ``sdfg``, every signed-integer symbol it
    declares that the caller named (:func:`declare_nonnegative_symbols`); no other symbol is assumed anything."""
    root = root_of(sdfg)
    predicates = root.symbol_repo.params.predicates
    updated = 0
    for name in names_still_plain(root):
        root.set_symbol_assumptions(name, predicates.get(name, frozenset()) | {symbolic.Predicate.NONNEGATIVE})
        updated += 1
    return updated or None


@xf.explicit_cf_compatible
class SetSymbolNonnegativeAssumptions(ppl.Pass):
    """Set the SymPy ``nonnegative=True`` assumption on the SDFG's signed-integer free symbols
    (:func:`set_symbol_nonnegative_assumptions`) -- the compile-time nonnegativity contract, WITHOUT the runtime guard
    state."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        return set_symbol_nonnegative_assumptions(sdfg)


#: Label of the guard state.
GUARD_STATE_LABEL = "_assume_nonneg_syms"

#: Symbols with these dtypes can be negative and so are worth guarding.
SIGNED_INTEGER_DTYPES: dict[dtypes.typeclass, None] = dict.fromkeys(
    [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
)


@xf.explicit_cf_compatible
class AssumeSymbolConstraints(ppl.Pass):
    """Prepend a start state that traps when an assumption canonicalization relied on is violated."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Single-shot: the emitted-trap dedup below makes a re-run a no-op anyway.
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        # Set the compile-time nonnegativity assumption on the symbols (so a downstream proof sees ``s >= 0``) AND emit
        # the runtime trap that checks it.
        assumed = set_symbol_nonnegative_assumptions(sdfg)
        guarded = insert_assumption_guards(sdfg)
        return (assumed or 0) + (guarded or 0) or None


def is_assumption_guard_block(block: ControlFlowBlock) -> bool:
    """True if ``block`` is the runtime assumption-guard state emitted by :func:`insert_assumption_guards` (label
    ``_assume_nonneg_syms``)."""
    return isinstance(block, SDFGState) and block.label == GUARD_STATE_LABEL


def _signed_integer_free_symbols(sdfg: SDFG) -> list[str]:
    """Sorted names of the SDFG's signed-integer argument symbols that canonicalization assumed nonnegative."""
    predicates = root_of(sdfg).symbol_repo.params.predicates
    declared = declared_nonnegative_symbols(sdfg)
    args = {
        name
        for name in sdfg.used_symbols(all_symbols=False)
        if name in declared
        or predicates.get(name, frozenset()) & {symbolic.Predicate.NONNEGATIVE, symbolic.Predicate.POSITIVE}
    }
    sized = sized_names(sdfg) | {
        s.name for relation in tracked_assumptions(sdfg) for s in as_basic(relation).free_symbols
    }
    return sorted(s for s in args if sdfg.symbols.get(s) in SIGNED_INTEGER_DTYPES and s in sized)


def sized_names(sdfg: SDFG) -> set[str]:
    """Names of :func:`sized_symbols` in ``sdfg``, plus the outer names a nested SDFG's sized
    symbols are bound to through its ``symbol_mapping``."""
    names = {str(s) for s in sized_symbols(sdfg)}
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG) and node.sdfg.parent_sdfg is sdfg:
            for inner in sized_names(node.sdfg) & node.symbol_mapping.keys():
                names.update(str(s) for s in symbolic.pystr_to_symbolic(str(node.symbol_mapping[inner])).free_symbols)
    return names


def collect_assumptions(sdfg: SDFG) -> list[symbolic.SymbolicType]:
    """The full list of relations that must hold at runtime for the SDFG to be correct, deduped and ordered for a stable
    guard."""
    free = dict.fromkeys(sdfg.used_symbols(all_symbols=False))
    # A symbol declared positive is reasoned about as ``> 0`` (an empty-range guard is dropped for it), so zero
    # violates its contract as much as a negative value does
    predicates = root_of(sdfg).symbol_repo.params.predicates
    assumptions: list = [
        as_expr(symbolic.pystr_to_symbolic(s)) > 0
        if symbolic.Predicate.POSITIVE in predicates.get(s, frozenset())
        else as_expr(symbolic.pystr_to_symbolic(s)) >= 0
        for s in _signed_integer_free_symbols(sdfg)
    ]
    for relation in tracked_assumptions(sdfg):
        if all(s.name in free for s in as_basic(relation).free_symbols) and relation not in assumptions:
            assumptions.append(relation)
    return assumptions


def lead_with_assumption_guard(sdfg: SDFG) -> bool:
    """Move the guard state to the head of the top-level block list; ``True`` if it moved."""
    blocks = sdfg.nodes()
    guard = next((b for b in blocks if is_assumption_guard_block(b)), None)
    if guard is None or blocks[0] is guard:
        return False
    sdfg.reorder_nodes([guard] + [b for b in blocks if b is not guard])
    return True


def insert_assumption_guards(sdfg: SDFG) -> int | None:
    """Prepend one guard state whose tasklets trap on any violated assumption; return ``1`` if emitted or repositioned,
    else ``None``."""
    # Dedup on the emitted TRAP CODE, not on the guard state's label: re-canonicalizing an already-canonicalized SDFG
    # (the vectorizer canonicalizes at its own entry) fuses the guard state into its successor, so the label is gone and
    # a label check re-emits every trap -- they accumulate one full copy per run.
    guards = {node.code.as_string for state in sdfg.states() for node in state.nodes() if tutil.is_abort_guard(node)}
    conditions = [symbolic.symstr(sympy.Not(a)) for a in collect_assumptions(sdfg)]
    checks = [c for c in conditions if tutil.abort_guard_code(c) not in guards]
    if not checks:
        if not lead_with_assumption_guard(sdfg):
            return None
        return 1

    # ``add_state_before`` prepends the guard before the current start and reconnects predecessors to it, so it
    # correctly becomes the new start block (``is_start_block=True``).
    guard_state = sdfg.add_state_before(sdfg.start_block, GUARD_STATE_LABEL, is_start_block=True)
    for i, condition in enumerate(checks):
        # Side-effecting, so DeadDataflowElimination keeps the output-less guard.
        tutil.add_abort_guard(guard_state, f"check_assumption_{i}", condition)
        # ``ordered_side_effects`` is left at its default (ordered): the trap has to run BEFORE the computation it
        # guards, and it is only a separate block that keeps it there.
    lead_with_assumption_guard(sdfg)
    return 1


#: Backwards-compatible aliases.
AssumeSymbolsNonnegative = AssumeSymbolConstraints
insert_symbol_nonnegative_guard = insert_assumption_guards

__all__ = [
    "AssumeSymbolConstraints",
    "AssumeSymbolsNonnegative",
    "insert_assumption_guards",
    "insert_symbol_nonnegative_guard",
    "is_assumption_guard_block",
    "lead_with_assumption_guard",
]
