# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Write-conflict policy for the write paths a graph pass cannot decide.

Whether a write collides is a property of the dataflow under a parallel scope,
and :mod:`dace.transformation.passes.write_conflict_resolution` decides it
there, on the finished graph, for every write. This module is what remains for
the ONE case that pass cannot settle: an accumulation whose target element is
chosen at runtime (``hist[bin] += 1``). No analysis of the emitted dataflow can
say whether two iterations land on the same element, so the syntactic fact that
the statement is an accumulation is the only evidence available — and it is
only available here, before ANF hoists the self-reference out of the statement.

Everything else — plain accumulations, chained self-references, combiner calls
like ``b = max(b, x)``, and the reporting of updates that have no
conflict-resolution equivalent — moved to that pass, which sees the whole
region instead of one statement. Deciding those here was unsound in a way no
amount of marker refinement could fix: after ANF the self-read lives in a
different statement, so a marker-driven conflict resolution folded a stale
value and looked correct.
"""
import ast
from typing import Optional

from dace.config import Config
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

    Only consulted for a data-dependent target now (see the module docstring);
    the rule itself still mirrors the classic frontend
    (``newast.py:3799-3812``): inside a map, an accumulation is
    conflict-resolved unless ``frontend.avoid_wcr`` is set and the write
    provably cannot collide.

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
