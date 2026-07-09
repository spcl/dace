# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Emit a runtime guard that aborts if an assumption canonicalization relied on
is violated by the values passed at the SDFG boundary.

Two kinds of assumption are guarded, both with the same abort-only, no-recovery
discipline as the scatter no-conflict guard
(:mod:`dace.transformation.passes.scatter_conflict_guard`):

* **Nonnegative symbols.** Canonicalization builds every free symbol with
  ``nonnegative=True`` (see [[feedback_symbols_nonnegative_canonicalization]]):
  the offset-sign reasoning that lets ``LoopToMap`` / the subset-based transforms
  decide a write is in-bounds and race-free is only sound when the symbols
  supplied at the boundary are ``>= 0``. That contract is otherwise implicit --
  nothing checks it at runtime, so a caller who passes a negative size/offset
  gets silently wrong (or out-of-bounds) code from a pipeline that reasoned as if
  it could not happen.

* **Tracked relations.** A rewrite may be value-preserving only under a relation
  the compiler cannot prove -- e.g. the modular-wrap split needs the wrap offset
  below the modulus (``K < N``, see
  :mod:`dace.transformation.passes.canonicalize.tracked_assumptions`). Such a
  relation is recorded on the SDFG when the rewrite is applied and emitted here
  as a trap on its negation.

This pass makes the whole contract explicit and *checked*. It prepends a new
start state whose single side-effecting tasklet calls ``__builtin_trap()`` when
any guarded condition is violated. Only **signed** integer symbols are guarded
for nonnegativity (an unsigned symbol is nonnegative by construction, so
``x < 0`` is a tautology the guard would waste a comparison on). A tracked
relation is emitted only when all of its symbols are SDFG free symbols -- i.e.
in scope as function arguments at the entry state.

The tasklet has no connectors: it reads only symbols, which are in scope at the
SDFG entry. It is marked ``side_effects = True`` so the terminal
``SimplifyPass`` / ``DeadDataflowElimination`` does not prune a tasklet that has
no data outputs (the same drop that silently removed scatter guards before they
were marked side-effecting).
"""
from typing import List, Optional

import sympy

from dace import SDFG, dtypes, symbolic
from dace.sdfg import SDFGState
from dace.codegen.common import sym2cpp
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.canonicalize.tracked_assumptions import tracked_assumptions


def set_symbol_nonnegative_assumptions(sdfg: SDFG) -> Optional[int]:
    """Set the SymPy ``nonnegative=True`` assumption on every signed-integer free symbol of
    ``sdfg`` (and its nested SDFGs), in place.

    This is the compile-time half of the offset/size nonnegativity contract
    ([[feedback_symbols_nonnegative_canonicalization]]): DaCe size / offset symbols carry no
    sign, so a proof over them (``RelaxIntegerPowers`` deciding an integer power is ``>= 0``,
    the offset-sign reasoning ``LoopToMap`` relies on) cannot conclude ``s >= 0`` even though a
    size or a loop count always is. Rebuilding each such symbol with ``nonnegative=True`` and
    substituting it through the SDFG (via ``replace_dict``, which threads it into every
    descriptor shape / subset / interstate expression) makes those proofs go through.
    :func:`insert_assumption_guards` emits the matching runtime trap so the assumption is
    checked at the boundary rather than merely assumed.

    :param sdfg: the SDFG whose symbols are updated in place.
    :returns: the number of symbols updated, or ``None`` if none.
    """
    updated = 0
    for g in sdfg.all_sdfgs_recursive():
        repl = {}
        for name in g.free_symbols:
            dtype = g.symbols.get(name)
            if dtype not in _SIGNED_INTEGER_DTYPES:
                continue
            if symbolic.symbol(name, dtype=dtype).is_nonnegative:  # already nonnegative
                continue
            repl[name] = symbolic.symbol(name, dtype=dtype, nonnegative=True)
        if repl:
            g.replace_dict(repl)
            updated += len(repl)
    return updated or None


@xf.explicit_cf_compatible
class SetSymbolNonnegativeAssumptions(ppl.Pass):
    """Set the SymPy ``nonnegative=True`` assumption on the SDFG's signed-integer free symbols
    (:func:`set_symbol_nonnegative_assumptions`) -- the compile-time nonnegativity contract,
    WITHOUT the runtime guard state. Used where a later proof needs the assumption but the
    guard's start-block splice would be unsafe mid-pipeline (the vectorizer)."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.InterstateEdges

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        return set_symbol_nonnegative_assumptions(sdfg)


#: Label of the guard state; also the idempotence marker (re-running is a no-op
#: once a state with this label exists).
_GUARD_STATE_LABEL = '_assume_nonneg_syms'

#: Symbols with these dtypes can be negative and so are worth guarding. Unsigned
#: integer symbols are nonnegative by construction; float symbols are not part
#: of the offset/size nonnegativity contract.
_SIGNED_INTEGER_DTYPES = {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64}


@xf.explicit_cf_compatible
class AssumeSymbolConstraints(ppl.Pass):
    """Prepend a start state that traps when an assumption canonicalization relied on is violated.

    Guards both the nonnegativity of signed-integer free symbols and every
    relation recorded via
    :func:`~dace.transformation.passes.canonicalize.tracked_assumptions.record_assumption`.
    Runs at the very end of canonicalization (after the terminal ``SimplifyPass``,
    so nothing re-runs simplify over the new state) and splices its guard state in
    as the new start block, making it the first thing the generated program does.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        # Single-shot: the state-label marker below makes a re-run a no-op anyway.
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        # Set the compile-time nonnegativity assumption on the symbols (so a downstream proof
        # sees ``s >= 0``) AND emit the runtime trap that checks it. Both halves of the same
        # contract; the guard makes the assumption sound rather than merely asserted.
        assumed = set_symbol_nonnegative_assumptions(sdfg)
        guarded = insert_assumption_guards(sdfg)
        return (assumed or 0) + (guarded or 0) or None


def is_assumption_guard_block(block) -> bool:
    """True if ``block`` is the runtime assumption-guard state emitted by
    :func:`insert_assumption_guards` (label ``_assume_nonneg_syms``).

    Its ``__builtin_trap`` tasklets are infrastructure -- they read only symbols
    and touch no data -- so structural counts that verify a "tile-only descent"
    exclude them exactly as they already exclude the ``tile_runtime`` divisibility
    trip guards. Exposed so tests / audits recognize the guard without importing
    the private label constant.
    """
    return isinstance(block, SDFGState) and block.label == _GUARD_STATE_LABEL


def _signed_integer_free_symbols(sdfg: SDFG) -> List[str]:
    """Sorted names of the SDFG's signed-integer free symbols.

    ``free_symbols`` is exactly the externally-supplied set (symbols defined by
    interstate assignments / nested-SDFG mappings are excluded), i.e. the values
    a caller passes in -- the ones canonicalization assumed nonnegative.
    """
    return sorted(s for s in sdfg.free_symbols if sdfg.symbols.get(s) in _SIGNED_INTEGER_DTYPES)


def collect_assumptions(sdfg: SDFG) -> List:
    """The full list of relations that must hold at runtime for the SDFG to be
    correct, deduped and ordered for a stable guard.

    Two sources, unified into one list of sympy booleans:

    * **Auto-collected from the symbols** -- every signed-integer free symbol is
      assumed nonnegative (``s >= 0``), the offset-sign contract canonicalization
      reasons under (see [[feedback_symbols_nonnegative_canonicalization]]).
    * **Recorded by the passes** -- every relation a rewrite stashed via
      :func:`~dace.transformation.passes.canonicalize.tracked_assumptions.record_assumption`,
      kept only when all of its symbols are free symbols so it is evaluable at the
      entry state (a symbol defined by an interstate assignment is not in scope
      there). Parallelizations that keep a sequential fallback specialize with an
      ``if (cond) parallel else sequential`` branch instead of recording an
      assumption here, so only genuine no-fallback preconditions reach the trap.
    """
    free = set(sdfg.free_symbols)
    assumptions: List = [symbolic.pystr_to_symbolic(s) >= 0 for s in _signed_integer_free_symbols(sdfg)]
    for relation in tracked_assumptions(sdfg):
        if {s.name for s in relation.free_symbols} <= free and relation not in assumptions:
            assumptions.append(relation)
    return assumptions


def insert_assumption_guards(sdfg: SDFG) -> Optional[int]:
    """Prepend one guard state whose tasklets trap on any violated assumption;
    return ``1`` if emitted, else ``None`` (nothing to guard, or already present).

    Every assumption from :func:`collect_assumptions` becomes its OWN
    side-effecting ``__builtin_trap`` tasklet (``if (!assumption) trap()``) in a
    single new start state -- one tasklet per assumption so a fault points at the
    exact violated relation, all in one state so the guard is a single dominating
    block. ``sym2cpp`` prints the negation of a sympy relational directly
    (``Not(K < N)`` -> ``(K >= N)``, ``Not(s >= 0)`` -> ``(s < 0)``).
    """
    if any(b.label == _GUARD_STATE_LABEL for b in sdfg.nodes()):
        return None
    assumptions = collect_assumptions(sdfg)
    if not assumptions:
        return None

    # ``add_state_before`` prepends the guard before the current start and
    # reconnects predecessors to it, so it correctly becomes the new start block
    # (``is_start_block=True``). This pass runs LAST in canonicalization: a guard
    # prepended earlier is orphaned by any pass that builds its own entry state
    # (LoopToScan's scan-init, reduction init, ...) and resets the top-level
    # start, leaving the guard a disconnected source that dominator analyses
    # KeyError on. Running last -- nothing reshapes the start after -- is safe.
    guard_state = sdfg.add_state_before(sdfg.start_block, _GUARD_STATE_LABEL, is_start_block=True)
    for i, assumption in enumerate(assumptions):
        guard = guard_state.add_tasklet(
            f'check_assumption_{i}',
            {},
            {},
            f'if ({sym2cpp(sympy.Not(assumption))}) {{ __builtin_trap(); }}',
            language=dtypes.Language.CPP,
        )
        # ``__builtin_trap()`` is a real side effect with no data output, so
        # DeadDataflowElimination would otherwise prune this tasklet -- and with
        # it the guard -- as dead. Mark it side-effecting so simplify keeps it.
        guard.side_effects = True
    sdfg.reset_cfg_list()
    return 1


#: Backwards-compatible aliases. The pass historically guarded only symbol
#: nonnegativity; it now emits every tracked assumption, but existing pipelines
#: and tests import it under the old names.
AssumeSymbolsNonnegative = AssumeSymbolConstraints
insert_symbol_nonnegative_guard = insert_assumption_guards

__all__ = [
    'AssumeSymbolConstraints',
    'AssumeSymbolsNonnegative',
    'insert_assumption_guards',
    'insert_symbol_nonnegative_guard',
    'is_assumption_guard_block',
]
