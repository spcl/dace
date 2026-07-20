# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Write-conflict policy for the lowering stage.

A write emitted inside a dataflow scope runs concurrently in every iteration of
the enclosing map(s). It is correct only if it either provably cannot collide
(distinct iterations touch distinct elements) or carries a conflict-resolution
function. This module is the single place that decides which; every write path
— assignments, replacement expansions, reductions, indirect writes — asks here
instead of re-deriving the rule, so a new write path cannot silently reopen the
race class that :func:`accumulation_wcr` closes.

The decision has two inputs, produced by two different stages:

- the *accumulation marker* (``augmented_op``), attached during
  canonicalization by ``canonical/passes.py`` to statements that are
  read-modify-writes of their own target, and
- the *collision test*, which needs the emission scope stack and therefore only
  exists here.

Canonicalization also marks self-referential writes it could **not** reduce to
an accumulation (``conflict_hazard``). Those cannot be expressed as a WCR at
all; :func:`report_unresolved` surfaces them as warnings rather than letting
them lower as silent races. That reporting channel is deliberate: a racing
read-modify-write still lowers to a callback-free schedule tree, so the
callback-discrepancy check cannot see it.
"""
import ast
import warnings
from typing import Optional

from dace.config import Config
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState

#: Binary operators that may become a conflict-resolution lambda. A write is
#: only safe to conflict-resolve when repeatedly folding the accumulator on the
#: left is order-independent — ``f(f(x, a), b) == f(f(x, b), a)`` — because WCR
#: applies the combiner in arbitrary thread order.
#:
#: This is classic's ``newast.py::augassign_ops`` table MINUS ``%``, which is
#: not order-independent (``(30 % 17) % 27 == 13`` but ``(30 % 27) % 17 == 3``)
#: and is therefore miscompiled by classic's WCR. ``/`` is kept: it reorders
#: only within floating-point rounding, the same tolerance WCR already accepts
#: for ``+`` and ``*``.
WCR_OPERATORS = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
}


def accumulation_wcr(statement: ast.stmt, state: LoweringState) -> Optional[str]:
    """
    The conflict-resolution lambda for an accumulation lowered inside a
    dataflow scope, or None when the write needs no conflict resolution.

    Mirrors the classic frontend (``newast.py:3799-3812``): inside a map, an
    accumulation is conflict-resolved unless ``frontend.avoid_wcr`` is set and
    the write provably cannot collide.

    :param statement: A canonical assignment; only statements carrying the
                      ``augmented_op`` marker attached by
                      ``canonical/passes.py`` qualify.
    """
    operator = getattr(statement, 'augmented_op', None)
    if operator is None or not state.emitter.in_dataflow_scope:
        return None
    symbol = WCR_OPERATORS.get(type(operator))
    if symbol is None:
        return None
    if Config.get_bool('frontend', 'avoid_wcr') and not writes_collide(statement.targets[0], state):
        return None
    return f'lambda x, y: x {symbol} y'


def writes_collide(target: ast.expr, state: LoweringState) -> bool:
    """
    Whether two iterations of the enclosing dataflow scopes may write the same
    element of ``target``.

    Only a subset that varies with *every* enclosing map parameter partitions
    the write across iterations. Anything that does not resolve to a plain
    subset (an indirect target, most notably) answers True: a collision cannot
    be ruled out.
    """
    params = state.emitter.enclosing_map_params
    if not params:
        return False
    try:
        access = resolve_access(target, state)
    except UnsupportedFeatureError:
        return True  # e.g. an indirect target: iterations may well collide
    if access is None:
        return True
    free_symbols = {str(symbol) for symbol in access.subset.free_symbols}
    return not all(param in free_symbols for param in params)


def report_unresolved(statement: ast.stmt, target: ast.expr, state: LoweringState) -> None:
    """
    Warn about a self-referential write inside a dataflow scope that
    canonicalization could not reduce to an accumulation, and that therefore
    lowers as a racing read-modify-write.

    No-op unless the statement carries the ``conflict_hazard`` marker and the
    write can actually collide. The classic frontend emits the same race
    silently; making it observable is the point of this function.
    """
    reason = getattr(statement, 'conflict_hazard', None)
    if reason is None or not state.emitter.in_dataflow_scope:
        return
    if not writes_collide(target, state):
        return
    location = f'{state.context.filename}:{getattr(statement, "lineno", "?")}'
    warnings.warn(
        f'Possible write conflict at {location}: "{astutils.unparse(target).strip()}" is both read '
        f'and written by a statement running concurrently in a map, and the update ({reason}) has no '
        f'conflict-resolution equivalent. The generated code contains a data race.', UserWarning)
