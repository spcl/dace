# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering-rule registry and statement dispatcher.

Each Canonical Python AST statement type is handled by exactly one registered
:class:`LoweringRule`. Because the canonicalization stage is total, dispatch
failure is a frontend bug (a canonical node type without a rule), never a user
error.
"""
import ast
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple, Type

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen import provenance
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
        #: Scalar read -> symbol promoted from it for a range bound
        #: (``lowering.access._promote_range_scalars``). A statement's target
        #: and operands resolve separately, so without this ``O[0:n] = A[0:n] +
        #: 1.0`` would define two symbols and give the write and the read
        #: subsets different extents. Reuse extends past the statement that
        #: defined the symbol for exactly as long as the value it holds is
        #: provably unchanged -- see :meth:`reusable_index_symbol` -- which is
        #: what lets the ANF temporaries hoisted out of ONE source expression
        #: (``numpy.add(Z[:n], C[:n])`` becomes two view statements and a call)
        #: agree on one extent.
        self.index_symbols: Dict[str, str] = {}
        #: Where each :attr:`index_symbols` entry was defined, as (emission
        #: scope, position in that scope right after the defining node, names
        #: the read expression reads). Reuse is only offered against this site.
        self.index_symbol_sites: Dict[str, Tuple[Any, int, FrozenSet[str]]] = {}
        #: Index expression -> container materialized for it
        #: (``lowering.dispatch.materialize_array_indices``), valid for the
        #: current statement only, so an array-valued index named on both sides
        #: of an accumulation is evaluated once.
        self.index_arrays: Dict[str, str] = {}
        #: Source names bound to a symbol for the current statement only
        #: (``lowering.dispatch._promote_scalar_arguments``), with the binding
        #: to put back afterwards. The symbol keeps the value the scalar had
        #: when the statement ran, which is what the statement needed; leaving
        #: the name pointing at it would make a LATER read miss a write that
        #: no assignment statement performed (a callee writing the scalar).
        self.promoted_scalars: List[Tuple[str, Any]] = []

    def lower_body(self, body) -> None:
        """Lower a list of canonical statements into the current scope."""
        for statement in body:
            lower_statement(statement, self)

    def record_index_symbol(self, expression: str, symbol_name: str, sources: Iterable[str]) -> None:
        """
        Register the symbol just defined from the scalar read ``expression``,
        together with the emission site that makes it reusable.

        Call this immediately after emitting the defining
        :class:`~dace.sdfg.analysis.schedule_tree.treenodes.AssignNode`.

        :param expression: The scalar read the symbol was defined from.
        :param symbol_name: The symbol defined from it.
        :param sources: Container names ``expression`` reads.
        """
        scope = self.emitter.current_scope
        self.index_symbols[expression] = symbol_name
        self.index_symbol_sites[expression] = (scope, len(scope.children), frozenset(sources))

    def reusable_index_symbol(self, expression: str) -> Optional[str]:
        """
        The symbol already defined from the scalar read ``expression``, when
        reusing it is guaranteed to mean the same thing as defining a new one,
        or None otherwise.

        A promoted symbol holds the value the scalar had WHERE IT WAS DEFINED,
        so reuse is only sound while nothing can have changed that value in
        between. Both halves of that are checked against the emission site:

        - the definition must sit in the scope being emitted into right now, so
          that "in between" is a straight run of siblings rather than a path
          that a loop or branch may re-enter (a definition outside a loop says
          nothing about the value on the loop's second iteration);
        - every sibling emitted since must be a node that cannot change a
          container's contents -- an interstate assignment (which defines a
          symbol) or a view binding of something else (which only aliases).
          Anything that runs code, copies, or calls invalidates the entry.

        The window this leaves open is the one that matters: the ANF temporaries
        hoisted out of a single source expression are emitted as exactly such a
        run of view bindings, and the extents of the operands and of the target
        have to agree for the expression to lower at all.
        """
        symbol_name = self.index_symbols.get(expression)
        site = self.index_symbol_sites.get(expression)
        if symbol_name is None or site is None:
            return None
        scope, position, sources = site
        if scope is not self.emitter.current_scope:
            return None
        for node in scope.children[position:]:
            if isinstance(node, tn.AssignNode):
                continue
            if isinstance(node, tn.ViewNode) and node.target not in sources:
                continue
            return None
        return symbol_name

    def forget_index_symbol(self, expression: str) -> None:
        """Drop a single :attr:`index_symbols` entry and its emission site."""
        self.index_symbols.pop(expression, None)
        self.index_symbol_sites.pop(expression, None)

    def forget_index_symbols_after(self, mark: int) -> None:
        """
        Drop every :attr:`index_symbols` entry whose defining node was just
        discarded by ``TreeEmitter.rollback(mark)``.
        """
        scope = self.emitter.current_scope
        for expression, (site_scope, position, _) in list(self.index_symbol_sites.items()):
            if site_scope is scope and position > mark:
                self.forget_index_symbol(expression)


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
    # What this statement's temporary stands for, for the diagnostics that hold
    # a container name rather than a node. Recorded before the rule runs, since
    # the rule itself may be the one that refuses something.
    provenance.adopt_statement_source(statement, state.context.expression_sources)
    mark = state.emitter.checkpoint()
    saved_bindings = state.context.snapshot()
    state.index_arrays = {}
    state.promoted_scalars = []
    try:
        handler(statement, state)
        for source_name, previous in state.promoted_scalars:
            if previous is None:
                state.context.bindings.pop(source_name, None)
            else:
                state.context.bindings[source_name] = previous
    except UnsupportedFeatureError as reason:
        from dace.frontend.python.nextgen.lowering import dispatch  # Deferred: dispatch imports this module
        state.emitter.rollback(mark)
        state.forget_index_symbols_after(mark)
        state.context.restore(saved_bindings)
        # An error reaching this net without a raise-site category has the
        # highest bug suspicion: no rule anticipated it.
        dispatch.fallback_to_callback(statement, state, reason, category='safety-net')
