# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The Canonical Python AST (CPA) contract.

The canonicalization stage reduces arbitrary preprocessed Python programs to
the restricted AST subset defined in this module. The subset is designed so
that the lowering stage can be *total*: every canonical statement type has
exactly one lowering rule, and no lowering rule ever encounters an unexpected
node shape.

Canonical statement grammar::

    stmt := Assign(target, flat)         # single target: Name | Attribute(Name) | Subscript(dataref, idx)
          | AnnAssign(Name, T)           # bare declaration (no value): descriptor hint
          | Expr(Call)                   # call statement (flat call)
          | If(atomexpr, [stmt], [stmt])
          | While(atomexpr, [stmt])      # test is atomic (complex tests are pre-hoisted)
          | For(Name, range(atom, atom, atom), [stmt])
          | For([Name...], dace.map[...], [stmt])
          | Return(atom | Tuple[atom] | None)
          | Break | Continue | Pass
          | ExplicitTasklet              # explicit-dataflow tasklet (opaque body)
          | ExplicitConsume              # explicit consume scope (stream pop)
          | OpaqueStmt                   # explicit interpreter-fallback marker

    flat  := atomexpr
           | Call(callable-ref, [atom...], keywords with atom values)

    atomexpr := operand
              | BinOp(operand, op, operand)
              | Compare(operand, [op], [operand])
              | BoolOp([operand...])

    operand := atom | Subscript(dataref, idx) | UnaryOp(op, operand)

    dataref := Name | Attribute(Name)    # container or single-level structure member
                                          # (nested structure members reach this
                                          # form via ANF hoisting, see
                                          # ANFTransform._as_dataref)

    atom  := Name | Constant | UnaryOp(op, atom) | Attribute-chain over Name

    idx     := idxexpr | Slice(idxexpr?, idxexpr?, idxexpr?) | Tuple([idx...])

    idxexpr := operand | BinOp(operand, op, operand)

Statements that cannot be reduced to this grammar become :class:`OpaqueStmt`
markers carrying the original statement and its input/output name sets, so
that the lowering stage can emit a fully-specified Python callback. This makes
"unhandled AST node" unreachable by construction.

The distinction between syntax and semantics is intentional: canonicalization
marks statements opaque when their *syntax* cannot be reduced (e.g.
``try``/``except``). A canonical ``Call`` whose target turns out to be an
unparseable Python function is resolved by the *call lowering rule*, which
falls back to the same callback outlining path.
"""
import ast
from typing import Iterable, List, Optional, Set, Tuple

from dace.frontend.python.nextgen.common import CanonicalViolationError


class OpaqueStmt(ast.stmt):
    """
    Explicit marker for one or more statements that must execute in the
    Python interpreter. Produced only by the canonicalization stage; adjacent
    markers are coalesced by the batching pass into a single marker carrying
    the whole statement run.

    :param original: The first original (non-canonical) statement.
    :param reason: Human-readable reason why the statement is opaque.
    :param inputs: Names read by the statement run (excluding names produced
                   earlier within the same run).
    :param outputs: Names written by the statement run.
    :param originals: All original statements of the run, in order. Defaults
                      to ``[original]``.
    """
    _fields = ()

    def __init__(self,
                 original: Optional[ast.stmt] = None,
                 reason: str = '',
                 inputs: Optional[Set[str]] = None,
                 outputs: Optional[Set[str]] = None,
                 originals: Optional[List[ast.stmt]] = None):
        # All parameters default so ``copy.deepcopy`` can reconstruct the node
        # (``ast.AST.__reduce__`` rebuilds with no arguments, then restores
        # the instance dictionary).
        super().__init__()
        self.original = original
        self.reason = reason
        self.inputs = inputs if inputs is not None else set()
        self.outputs = outputs if outputs is not None else set()
        self.originals = list(originals) if originals is not None else ([original] if original is not None else [])
        if original is not None:
            ast.copy_location(self, original)


class ExplicitTasklet(ast.stmt):
    """
    Canonical marker for explicit-dataflow tasklets (``with dace.tasklet:``
    blocks and ``@dace.tasklet``-decorated functions).

    The tasklet body is *tasklet code*, not program code: canonicalization
    passes must not desugar, flatten, or otherwise rewrite it. Memlet
    statements (``local << A[i]``, ``local >> B[i]``) are parsed during
    lowering, when data descriptors are available.

    :param label: Tasklet label (function name or generated from the line).
    :param statements: The raw tasklet body statements.
    :param language: Optional language name (e.g. ``'Python'``, ``'CPP'``).
    :param side_effects: Optional explicit side-effect flag.
    :param code_global: Global-scope code emitted alongside the tasklet.
    :param code_init: Initialization code emitted alongside the tasklet.
    :param code_exit: Finalization code emitted alongside the tasklet.
    :param original: The original (pre-canonicalization) statement, restored
                     when a surrounding region falls back to a callback. None
                     for synthesized tasklets with no standalone source form
                     (e.g. the body of a desugared ``@dace.map`` function).
    """
    _fields = ()

    def __init__(self,
                 label: str = '',
                 statements: Optional[List[ast.stmt]] = None,
                 language: Optional[str] = None,
                 side_effects: Optional[bool] = None,
                 code_global: str = '',
                 code_init: str = '',
                 code_exit: str = '',
                 location: Optional[ast.AST] = None,
                 original: Optional[ast.stmt] = None):
        # All parameters default so ``copy.deepcopy`` can reconstruct the node
        # (``ast.AST.__reduce__`` rebuilds with no arguments).
        super().__init__()
        self.label = label
        self.statements = statements if statements is not None else []
        self.language = language
        self.side_effects = side_effects
        self.code_global = code_global
        self.code_init = code_init
        self.code_exit = code_exit
        self.original = original
        if location is not None:
            ast.copy_location(self, location)


class ExplicitConsume(ast.stmt):
    """
    Canonical marker for explicit consume scopes (``@dace.consume(stream,
    num_pes[, condition, chunksize])`` and ``@dace.consumescope(...)``
    decorated functions, following the classic frontend's contract: the
    decorated function takes exactly two parameters — the popped stream
    element and the processing-element index).

    :param label: Scope label (the function name).
    :param stream: The stream expression from the decorator (an ``ast.expr``).
    :param num_pes_src: Source text of the number-of-processing-elements
                        argument.
    :param condition_src: Source text of the quiescence condition, or None
                          (run until the stream is empty).
    :param chunksize_src: Source text of the chunk size (default ``'1'``).
    :param element: Name of the popped-element parameter.
    :param pe_index: Name of the processing-element index parameter.
    :param statements: The function body. For the ``consume`` (tasklet) form
                       these are raw tasklet statements; for the
                       ``consumescope`` form they are canonicalized program
                       statements.
    :param scope_body: True for ``@dace.consumescope`` (statement body),
                       False for ``@dace.consume`` (tasklet body).
    :param original: The original decorated FunctionDef, restored when a
                     surrounding region falls back to a callback.
    """
    _fields = ()

    def __init__(self,
                 label: str = '',
                 stream: Optional[ast.expr] = None,
                 num_pes_src: str = '1',
                 condition_src: Optional[str] = None,
                 chunksize_src: str = '1',
                 element: str = '',
                 pe_index: str = '',
                 statements: Optional[List[ast.stmt]] = None,
                 scope_body: bool = False,
                 location: Optional[ast.AST] = None,
                 original: Optional[ast.stmt] = None):
        # All parameters default so ``copy.deepcopy`` can reconstruct the node
        # (``ast.AST.__reduce__`` rebuilds with no arguments).
        super().__init__()
        self.label = label
        self.stream = stream
        self.num_pes_src = num_pes_src
        self.condition_src = condition_src
        self.chunksize_src = chunksize_src
        self.element = element
        self.pe_index = pe_index
        self.statements = statements if statements is not None else []
        self.scope_body = scope_body
        self.original = original
        if location is not None:
            ast.copy_location(self, location)


#: Canonical statement markers whose contents no canonicalization pass may touch.
CANONICAL_LEAVES = (OpaqueStmt, ExplicitTasklet, ExplicitConsume)


def statement_io_sets(node: ast.stmt) -> Tuple[Set[str], Set[str]]:
    """
    Compute the sets of names a statement reads and writes.

    Nested function and lambda scopes contribute only their free-variable
    reads. Attribute and subscript stores count the base name as both read and
    written (the container object survives, only contents change).

    :return: A 2-tuple of (read names, written names).
    """
    reads: Set[str] = set()
    writes: Set[str] = set()

    class _IOVisitor(ast.NodeVisitor):

        def visit_OpaqueStmt(self, opaque_node: 'OpaqueStmt') -> None:
            # Opaque markers hide their contents from AST traversal
            # (_fields is empty) but carry precomputed I/O sets.
            reads.update(opaque_node.inputs)
            writes.update(opaque_node.outputs)

        def visit_Name(self, name_node: ast.Name) -> None:
            if isinstance(name_node.ctx, ast.Load):
                reads.add(name_node.id)
            else:
                writes.add(name_node.id)

        def visit_Attribute(self, attribute_node: ast.Attribute) -> None:
            if isinstance(attribute_node.ctx, (ast.Store, ast.Del)):
                base = attribute_node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    reads.add(base.id)
                    writes.add(base.id)
                    return
            self.generic_visit(attribute_node)

        def visit_Subscript(self, subscript_node: ast.Subscript) -> None:
            if isinstance(subscript_node.ctx, (ast.Store, ast.Del)):
                base = subscript_node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    reads.add(base.id)
                    writes.add(base.id)
                self.visit(subscript_node.slice)
                if not isinstance(subscript_node.value, ast.Name):
                    self.visit(subscript_node.value)
                return
            self.generic_visit(subscript_node)

    _IOVisitor().visit(node)
    return reads, writes


def is_atom(node: ast.AST) -> bool:
    """Check whether an expression is a canonical atom."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.UnaryOp):
        return is_atom(node.operand)
    if isinstance(node, ast.Attribute):
        return is_atom(node.value)
    return False


def is_dataref(node: ast.AST) -> bool:
    """Check whether an expression is a canonical data reference: a name or a
    single-level attribute over a name (a structure member)."""
    if isinstance(node, ast.Name):
        return True
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)


def is_operand(node: ast.AST) -> bool:
    """Check whether an expression is a canonical operator operand: an atom or a data subscript."""
    if is_atom(node):
        return True
    if isinstance(node, ast.Subscript):
        return is_dataref(node.value) and is_index(node.slice)
    if isinstance(node, ast.UnaryOp):
        return is_operand(node.operand)
    return False


def is_atomexpr(node: ast.AST) -> bool:
    """Check whether an expression is a canonical atomic expression (depth-1)."""
    if is_operand(node):
        return True
    if isinstance(node, ast.BinOp):
        return is_operand(node.left) and is_operand(node.right)
    if isinstance(node, ast.Compare):
        return is_operand(node.left) and all(is_operand(c) for c in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(is_operand(v) for v in node.values)
    return False


def is_index_expr(node: ast.AST) -> bool:
    """
    Check whether an expression is a canonical index expression (depth-1): an
    operand, or a binary operation of two operands (e.g. the ``i + 1`` in
    ``A[i + 1]``, or the ``A_col[j]`` data-read operand in ``x[A_col[j]]``).
    """
    if is_operand(node):
        return True
    if isinstance(node, ast.BinOp):
        return is_operand(node.left) and is_operand(node.right)
    return False


def is_index(node: ast.AST) -> bool:
    """Check whether an expression is a canonical subscript index."""
    if isinstance(node, ast.Slice):
        return all(part is None or is_index_expr(part) for part in (node.lower, node.upper, node.step))
    if isinstance(node, ast.Tuple):
        return all(is_index(element) for element in node.elts)
    return is_index_expr(node)


def is_flat(node: ast.AST) -> bool:
    """Check whether an expression is a canonical assignment right-hand side."""
    if is_atomexpr(node):
        return True
    if isinstance(node, (ast.List, ast.Tuple)) and not isinstance(getattr(node, 'ctx', ast.Load()), ast.Store):
        # Sequence literals of atoms are compile-time values in the semantic layer
        return all(is_atom(element) for element in node.elts)
    if isinstance(node, ast.Call):
        callable_ok = is_atom(node.func)
        args_ok = all(is_atom(a) for a in node.args)
        kwargs_ok = all(kw.arg is not None and is_atom(kw.value) for kw in node.keywords)
        return callable_ok and args_ok and kwargs_ok
    return False


def is_assign_target(node: ast.AST) -> bool:
    """Check whether an expression is a canonical assignment target."""
    if is_dataref(node):
        return True
    if isinstance(node, ast.Subscript):
        return is_dataref(node.value) and is_index(node.slice)
    return False


def is_dace_map_iterator(node: ast.AST) -> bool:
    """Check whether a for-loop iterator is a ``dace.map[...]`` subscript."""
    if not isinstance(node, ast.Subscript):
        return False
    try:
        from dace.frontend.python.astutils import rname
        return rname(node.value) in ('dace.map', 'map')
    except (AttributeError, TypeError):
        return False


def is_range_iterator(node: ast.AST) -> bool:
    """Check whether a for-loop iterator is a normalized 3-argument ``range`` call."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range'
            and len(node.args) == 3 and all(is_atom(a) for a in node.args) and not node.keywords)


def is_return_value(node: ast.AST) -> bool:
    """Check whether an expression is a canonical return value: an atom or a tuple of atoms."""
    if isinstance(node, ast.Tuple):
        return all(is_atom(element) for element in node.elts)
    return is_atom(node)


def _violations_in_statement(node: ast.stmt) -> Iterable[str]:
    """Yield CPA violations for a single statement (non-recursive)."""
    if isinstance(node, CANONICAL_LEAVES):
        return
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            yield 'Assign must have exactly one target'
        elif not is_assign_target(node.targets[0]):
            yield f'Non-canonical assignment target: {ast.dump(node.targets[0])[:80]}'
        if not is_flat(node.value):
            yield f'Non-canonical assignment value: {ast.dump(node.value)[:80]}'
    elif isinstance(node, ast.AnnAssign):
        # Only bare declarations survive desugaring (annotated assignments
        # become Assign nodes carrying the annotation as an attribute).
        if node.value is not None or not isinstance(node.target, ast.Name):
            yield f'Non-canonical annotated assignment: {ast.dump(node)[:80]}'
    elif isinstance(node, ast.If):
        if not is_atomexpr(node.test):
            yield f'Non-canonical if-test: {ast.dump(node.test)[:80]}'
    elif isinstance(node, ast.While):
        if not is_atomexpr(node.test):
            yield f'Non-canonical while-test: {ast.dump(node.test)[:80]}'
        if node.orelse:
            yield 'while-else must be desugared before verification'
    elif isinstance(node, ast.For):
        if is_range_iterator(node.iter):
            if not isinstance(node.target, ast.Name):
                yield 'range-loop target must be a single name'
        elif is_dace_map_iterator(node.iter):
            targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
            if not all(isinstance(t, ast.Name) for t in targets):
                yield 'dace.map loop targets must be names'
        else:
            yield f'Non-canonical for-iterator: {ast.dump(node.iter)[:80]}'
        if node.orelse:
            yield 'for-else must be desugared before verification'
    elif isinstance(node, ast.Return):
        if node.value is not None and not is_return_value(node.value):
            yield f'Non-canonical return value: {ast.dump(node.value)[:80]}'
    elif isinstance(node, ast.Expr):
        # Call statements are canonical (nested programs, library calls);
        # other bare expressions are not.
        if not (isinstance(node.value, ast.Call) and is_flat(node.value)):
            yield f'Non-canonical expression statement: {ast.dump(node.value)[:80]}'
    elif isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
        return
    else:
        yield f'Statement type {type(node).__name__} is not part of the canonical subset'


def verify_canonical(tree: ast.FunctionDef, filename: Optional[str] = None) -> None:
    """
    Verify the CPA postcondition of the canonicalization stage.

    :raises CanonicalViolationError: If any statement violates the contract.
        This indicates a canonicalization bug — the stage is required to be
        total over arbitrary input.
    """
    violations: List[str] = []

    def _walk(body: List[ast.stmt]) -> None:
        for statement in body:
            for violation in _violations_in_statement(statement):
                line = getattr(statement, 'lineno', '?')
                violations.append(f'line {line}: {violation}')
            if isinstance(statement, CANONICAL_LEAVES):
                continue
            for field in ('body', 'orelse'):
                child_body = getattr(statement, field, None)
                if child_body:
                    _walk(child_body)

    _walk(tree.body)
    if violations:
        raise CanonicalViolationError('Canonicalization produced non-canonical output (frontend bug):\n' +
                                      '\n'.join(violations),
                                      filename=filename)
