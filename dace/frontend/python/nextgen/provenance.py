# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Provenance of generated names: recovering what the user actually wrote.

Canonicalization converts expressions to A-normal form, hoisting every compound
subexpression into a fresh single-assignment temporary (``return A[A > 15]``
becomes ``__anf0 = A > 15; __anf1 = A[__anf0]; return __anf1``). By the time a
lowering rule refuses something, the expression it holds is therefore a
generated name with no visible relation to the program text, and unparsing it
names the temporary rather than the user's expression.

This module records, at the moment of the hoist, the expression each temporary
replaced, and resolves a *chain* of such records back into source text for
diagnostics. Resolution is transitive on purpose: a temporary's recorded
expression is the already-flattened one (``A[__anf0]``), so only substituting
the inner records back in yields text the user recognizes (``A[A > 15]``).
Substitution happens on the AST rather than on strings, so the rendered text is
parenthesized correctly.

The record lives **on the AST node** the hoist produced, which is what makes it
survive everything the frontend does with the tree afterwards — including the
per-call-site deep copy of an inlined callee's body, where a name-keyed map
built by the caller's canonicalization would answer with the wrong expression.
Diagnostics that hold only a container name and no node read a name-keyed map
instead (:attr:`ProgramContext.expression_sources`), which lowering fills from
these same records as it walks the statements (see
:func:`adopt_statement_source`), so its entries are always the ones in scope.
"""
import ast
from typing import Mapping, MutableMapping, Optional, Union

from dace.frontend.python import astutils

#: Attribute under which a hoisted expression is recorded, both on the ``Name``
#: node substituted for it and on the target of the assignment that binds the
#: temporary.
EXPRESSION_SOURCE_ATTRIBUTE = 'dace_expression_source'

#: Maximum number of nested temporaries substituted while describing an
#: expression. A-normal form is single-assignment and a record can only mention
#: earlier temporaries, so the chain always terminates; the bound only keeps a
#: pathological program's diagnostic from expanding without limit.
_MAX_RESOLUTION_DEPTH = 16


def record_expression_source(expression: ast.expr, *nodes: ast.AST) -> ast.expr:
    """
    Record the expression that a generated temporary replaced.

    :param expression: The expression that was hoisted into the temporary.
    :param nodes: The nodes standing for it — the ``Name`` substituted for the
                  expression, and the target of the assignment binding it.
    :return: The snapshot stored on the nodes — a copy, so later passes that
             rewrite the hoisted assignment in place cannot alter what a
             diagnostic reports.
    """
    snapshot = astutils.copy_tree(expression)
    for node in nodes:
        setattr(node, EXPRESSION_SOURCE_ATTRIBUTE, snapshot)
    return snapshot


def adopt_statement_source(statement: ast.stmt, sources: MutableMapping[str, ast.AST]) -> None:
    """
    Record the provenance a statement carries in a name-keyed map, for the
    diagnostics that hold a container name rather than an AST node.

    Called as each statement is lowered, so the map only ever holds the
    temporaries of the function body being lowered — an inlined callee's
    ``__anf0`` never answers for its caller's.

    :param statement: The canonical statement about to be lowered.
    :param sources: The map to fill, usually
                    :attr:`ProgramContext.expression_sources`.
    """
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return
    target = statement.targets[0]
    recorded = getattr(target, EXPRESSION_SOURCE_ATTRIBUTE, None)
    if recorded is not None and isinstance(target, ast.Name):
        sources[target.id] = recorded


def describe_expression(value: Union[ast.AST, str], sources: Optional[Mapping[str, ast.AST]] = None) -> str:
    """
    Render an expression (or a container name) as the source text it came from,
    resolving generated temporaries back to what the user wrote.

    Falls back to the plain unparse (or to the name itself) when nothing was
    recorded, so this is always safe to call in an error path.

    :param value: An expression node, or the name of a container.
    :param sources: Recorded expression sources by generated name, usually
                    :attr:`ProgramContext.expression_sources`.
    :return: Single-line source text describing the expression.
    """
    recorded = sources or {}
    if isinstance(value, str):
        node = recorded.get(value)
        if node is None:
            return value
    else:
        node = value
    resolved = _Substituter(recorded, _MAX_RESOLUTION_DEPTH).visit(astutils.copy_tree(node))
    try:
        # ``ast.unparse`` rather than ``astutils.unparse``: this text is read by
        # a person, and the latter fully parenthesizes every operator
        # (``A[(A > 15)]``). Falls back to it for anything it cannot render.
        return ast.unparse(resolved).strip()
    except Exception:
        return astutils.unparse(resolved).strip()


class _Substituter(ast.NodeTransformer):
    """
    Rewrites a copied expression into the source text it came from: generated
    temporaries become the expressions they replaced, and a global that
    preprocessing resolved to the object itself becomes the name it was written
    under again (otherwise a nested ``@dace.program`` call unparses as the
    ``repr`` of a ``DaceProgram``).
    """

    def __init__(self, sources: Mapping[str, ast.AST], depth: int):
        self.sources = sources
        self.depth = depth

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.depth <= 0:
            return node
        recorded = getattr(node, EXPRESSION_SOURCE_ATTRIBUTE, None)
        if recorded is None:
            recorded = self.sources.get(node.id)
        if recorded is None:
            return node
        return _Substituter(self.sources, self.depth - 1).visit(astutils.copy_tree(recorded))

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (bool, int, float, complex, str, bytes, type(None))):
            return node  # An ordinary literal, which renders as itself
        # An object preprocessing resolved and embedded (a nested
        # ``@dace.program``, a module, a constant global), whose ``repr`` is no
        # use in a message. ``qualname`` records the source it came from --
        # for a resolved callee, the whole call text ("selector(A)"), whose
        # head is the name it was written under. The object's own name is the
        # last resort: a nested program renamed by its module ends up under a
        # name the program text never mentions.
        qualname = getattr(node, 'qualname', None)
        candidates = [qualname.split('(', 1)[0]] if isinstance(qualname, str) else []
        candidates += [getattr(node.value, '__name__', None), getattr(node.value, 'name', None)]
        for candidate in candidates:
            if isinstance(candidate, str) and all(part.isidentifier() for part in candidate.split('.')):
                return ast.copy_location(ast.Name(id=candidate, ctx=ast.Load()), node)
        return node
