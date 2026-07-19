# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering-rule registry and statement dispatcher.

Each Canonical Python AST statement type is handled by exactly one registered
:class:`LoweringRule`. Because the canonicalization stage is total, dispatch
failure is a frontend bug (a canonical node type without a rule), never a user
error.
"""
import ast
from typing import Callable, Dict, Type

from dace.frontend.python.nextgen.common import CanonicalViolationError, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.emitter import TreeEmitter
from dace.frontend.python.nextgen.semantics.context import ProgramContext
from dace.frontend.python.nextgen.semantics.inference import InferenceService

#: A lowering rule takes (statement, state) and emits schedule tree nodes.
LoweringRule = Callable[[ast.stmt, 'LoweringState'], None]

_RULES: Dict[Type[ast.stmt], LoweringRule] = {}


class LoweringState:
    """Bundles everything a lowering rule needs: context, inference, emitter."""

    def __init__(self, context: ProgramContext, emitter: TreeEmitter):
        self.context = context
        self.emitter = emitter
        self.inference = InferenceService(context)
        #: Optional progress feedback (dace.cli.progress.OptionalProgressBar),
        #: ticked once per lowered statement when set.
        self.progress = None

    def lower_body(self, body) -> None:
        """Lower a list of canonical statements into the current scope."""
        for statement in body:
            lower_statement(statement, self)


def rule(*statement_types: Type[ast.stmt]) -> Callable[[LoweringRule], LoweringRule]:
    """Register a lowering rule for one or more canonical statement types."""

    def decorator(function: LoweringRule) -> LoweringRule:
        for statement_type in statement_types:
            if statement_type in _RULES:
                raise ValueError(f'Duplicate lowering rule for {statement_type.__name__}')
            _RULES[statement_type] = function
        return function

    return decorator


def lower_statement(statement: ast.stmt, state: LoweringState) -> None:
    """
    Dispatch a canonical statement to its lowering rule.

    A :class:`UnsupportedFeatureError` escaping a rule is a semantic feature
    gap, not a failure: the partially emitted structure is rolled back and the
    whole statement falls back to the interpreter (totality safety net). Rules
    that can fall back more precisely catch the error themselves first.

    :raises CanonicalViolationError: If no rule exists for the statement type,
        which indicates that canonicalization and the rule registry are out of
        sync (a frontend bug).
    """
    handler = _RULES.get(type(statement))
    if handler is None:
        raise CanonicalViolationError(
            f'No lowering rule registered for canonical statement type {type(statement).__name__}',
            state.context.filename, statement)
    if state.progress is not None:
        state.progress.next()
    mark = state.emitter.checkpoint()
    saved_bindings = state.context.snapshot()
    try:
        handler(statement, state)
    except UnsupportedFeatureError as reason:
        from dace.frontend.python.nextgen.lowering import dispatch  # Deferred: dispatch imports this module
        state.emitter.rollback(mark)
        state.context.restore(saved_bindings)
        dispatch.fallback_to_callback(statement, state, str(reason))
