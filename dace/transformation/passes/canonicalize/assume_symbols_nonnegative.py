# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Emit a runtime guard that aborts if a symbol canonicalization assumed
nonnegative is passed a negative value.

Canonicalization builds every free symbol with ``nonnegative=True`` (see
[[feedback_symbols_nonnegative_canonicalization]]): the offset-sign reasoning
that lets ``LoopToMap`` / the subset-based transforms decide a write is
in-bounds and race-free is only sound when the symbols supplied at the SDFG
boundary are ``>= 0``. That contract is otherwise implicit -- nothing checks it
at runtime, so a caller who passes a negative size/offset gets silently wrong
(or out-of-bounds) code from a pipeline that reasoned as if it could not happen.

This pass makes the contract explicit and *checked*. It prepends a new start
state whose single side-effecting tasklet calls ``__builtin_trap()`` when any
signed-integer free symbol is negative -- the same abort-only, no-recovery
discipline as the scatter no-conflict guard
(:mod:`dace.transformation.passes.scatter_conflict_guard`). Only **signed**
integer symbols are guarded (an unsigned symbol is nonnegative by construction,
so ``x < 0`` is a tautology the guard would waste a comparison on).

The tasklet has no connectors: it reads only symbols, which are in scope at the
SDFG entry as function arguments. It is marked ``side_effects = True`` so the
terminal ``SimplifyPass`` / ``DeadDataflowElimination`` does not prune a tasklet
that has no data outputs (the same drop that silently removed scatter guards
before they were marked side-effecting).
"""
from typing import List, Optional

from dace import SDFG, dtypes
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: Label of the guard state; also the idempotence marker (re-running is a no-op
#: once a state with this label exists).
_GUARD_STATE_LABEL = '_assume_nonneg_syms'

#: Symbols with these dtypes can be negative and so are worth guarding. Unsigned
#: integer symbols are nonnegative by construction; float symbols are not part
#: of the offset/size nonnegativity contract.
_SIGNED_INTEGER_DTYPES = {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64}


@xf.explicit_cf_compatible
class AssumeSymbolsNonnegative(ppl.Pass):
    """Prepend a start state that traps when a signed-integer free symbol is negative.

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
        return insert_symbol_nonnegative_guard(sdfg)


def _signed_integer_free_symbols(sdfg: SDFG) -> List[str]:
    """Sorted names of the SDFG's signed-integer free symbols.

    ``free_symbols`` is exactly the externally-supplied set (symbols defined by
    interstate assignments / nested-SDFG mappings are excluded), i.e. the values
    a caller passes in -- the ones canonicalization assumed nonnegative.
    """
    return sorted(s for s in sdfg.free_symbols if sdfg.symbols.get(s) in _SIGNED_INTEGER_DTYPES)


def insert_symbol_nonnegative_guard(sdfg: SDFG) -> Optional[int]:
    """Prepend the ``__builtin_trap`` guard state; return ``1`` if emitted, else ``None``.

    No-op (returns ``None``) when the SDFG has no signed-integer free symbols or a
    guard state already exists.
    """
    if any(b.label == _GUARD_STATE_LABEL for b in sdfg.nodes()):
        return None
    syms = _signed_integer_free_symbols(sdfg)
    if not syms:
        return None

    condition = ' || '.join(f'{s} < 0' for s in syms)
    # ``add_state_before`` prepends the guard before the current start and
    # reconnects predecessors to it, so it correctly becomes the new start block
    # (``is_start_block=True``). This pass runs LAST in canonicalization: a guard
    # prepended earlier is orphaned by any pass that builds its own entry state
    # (LoopToScan's scan-init, reduction init, ...) and resets the top-level
    # start, leaving the guard a disconnected source that dominator analyses
    # KeyError on. Running last -- nothing reshapes the start after -- is safe.
    guard_state = sdfg.add_state_before(sdfg.start_block, _GUARD_STATE_LABEL, is_start_block=True)
    guard = guard_state.add_tasklet(
        'assume_nonneg',
        {},
        {},
        f'if ({condition}) {{ __builtin_trap(); }}',
        language=dtypes.Language.CPP,
    )
    # ``__builtin_trap()`` is a real side effect with no data output, so
    # DeadDataflowElimination would otherwise prune this tasklet -- and with it
    # the whole guard -- as dead. Mark it side-effecting so simplify keeps it.
    guard.side_effects = True
    sdfg.reset_cfg_list()
    return 1


__all__ = ['AssumeSymbolsNonnegative', 'insert_symbol_nonnegative_guard']
