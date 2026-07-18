# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Canonicalization passes: reduce a preprocessed Python AST to the Canonical
Python AST (CPA) subset defined in :mod:`~dace.frontend.python.nextgen.canonical.cpa`.

Pass order matters and is fixed by :func:`default_passes`:

1. :class:`DesugarStatements` — multi-target/chained assignments, ``AugAssign``,
   ``AnnAssign``, loop ``else`` clauses, docstring removal.
2. :class:`NormalizeLoops` — ``range`` calls to 3-argument form; complex
   ``while`` tests to ``while True`` + conditional ``break`` (correct under
   ``break``/``continue`` because the test re-evaluates at the loop head).
3. :class:`ANFTransform` — A-normal form: every compound subexpression is
   hoisted into a fresh single-assignment temporary, so all remaining
   expressions are at most depth-1 ("flat").
4. :class:`MarkOpaque` — any statement still outside the CPA subset becomes an
   explicit :class:`~dace.frontend.python.nextgen.canonical.cpa.OpaqueStmt`
   with precomputed input/output sets. This pass makes the stage total.
"""
import ast
import copy
from typing import List, Optional, Tuple, Union

from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.canonical.cpa import CANONICAL_LEAVES, ExplicitTasklet, OpaqueStmt

_TERMINAL_STMTS = (ast.Break, ast.Continue, ast.Pass)


def _located(node: ast.AST, template: ast.AST) -> ast.AST:
    """Copy source location from a template node."""
    return ast.copy_location(node, template)


def _name_load(name: str, template: ast.AST) -> ast.Name:
    return _located(ast.Name(id=name, ctx=ast.Load()), template)


def _name_store(name: str, template: ast.AST) -> ast.Name:
    return _located(ast.Name(id=name, ctx=ast.Store()), template)


def _assign(target_name: str, value: ast.expr, template: ast.AST) -> ast.Assign:
    return _located(ast.Assign(targets=[_name_store(target_name, template)], value=value), template)


class _BodyTransformer:
    """
    Base class for passes that rewrite statement lists. Subclasses override
    :meth:`transform_statement` and return one or more replacement statements.
    """
    name = 'body-transform'

    def apply(self, tree: ast.FunctionDef, context) -> ast.FunctionDef:
        self.context = context
        tree.body = self._transform_body(tree.body)
        return tree

    def _transform_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        result: List[ast.stmt] = []
        for statement in body:
            replacement = self.transform_statement(statement)
            if replacement is None:
                continue
            if isinstance(replacement, list):
                result.extend(replacement)
            else:
                result.append(replacement)
        return result

    def _recurse(self, statement: ast.stmt) -> ast.stmt:
        """Transform nested statement bodies of a compound statement."""
        for field in ('body', 'orelse'):
            child_body = getattr(statement, field, None)
            if child_body:
                setattr(statement, field, self._transform_body(child_body))
        return statement

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES):
            return statement
        return self._recurse(statement)


class RecognizeExplicitDataflow(_BodyTransformer):
    """
    Recognize explicit-dataflow syntax before any other normalization, so
    tasklet bodies are preserved verbatim:

    - ``with dace.tasklet:`` and ``with dace.tasklet(language=...):`` blocks
      become :class:`ExplicitTasklet` markers.
    - ``@dace.tasklet``-decorated functions become :class:`ExplicitTasklet`.
    - ``@dace.map``-decorated functions become ``for ... in dace.map[...]``
      loops whose body is a single :class:`ExplicitTasklet` (map + tasklet,
      matching the stable frontend's explicit-dataflow semantics).
    - ``@dace.mapscope``-decorated functions become ``for ... in dace.map[...]``
      loops whose body is the function body itself, recursively canonicalized
      by this same pass (map + arbitrary nested dataflow, rather than a single
      tasklet).

    ``@dace.consume``/``@dace.consumescope`` are intentionally left
    unrecognized for now; such decorated functions fall through to the
    interpreter-callback path until a dedicated commit handles them.
    """
    name = 'recognize-explicit-dataflow'

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES):
            return statement
        if isinstance(statement, ast.With) and len(statement.items) == 1:
            context_expr = statement.items[0].context_expr
            callee = context_expr.func if isinstance(context_expr, ast.Call) else context_expr
            if _refers_to(callee, 'dace.tasklet', self.context.global_vars):
                return ExplicitTasklet(label=f'tasklet_{statement.lineno}',
                                       statements=statement.body,
                                       location=statement,
                                       original=statement,
                                       **_tasklet_arguments(context_expr))
        if isinstance(statement, ast.FunctionDef) and len(statement.decorator_list) == 1:
            decorator = statement.decorator_list[0]
            decorator_callee = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _refers_to(decorator_callee, 'dace.tasklet', self.context.global_vars):
                return ExplicitTasklet(label=statement.name,
                                       statements=statement.body,
                                       location=statement,
                                       original=statement,
                                       **_tasklet_arguments(decorator))
            if _refers_to(decorator_callee, 'dace.map', self.context.global_vars):
                return self._desugar_map_function(statement, decorator)
            if _refers_to(decorator_callee, 'dace.mapscope', self.context.global_vars):
                return self._desugar_mapscope_function(statement, decorator)
        return self._recurse(statement)

    def _extract_map_ranges(self, function: ast.FunctionDef,
                            decorator: ast.expr) -> Optional[Tuple[List[str], List[ast.expr]]]:
        """
        Extract per-dimension range slices and parameter names shared by
        ``@dace.map``/``@dace.mapscope``-decorated functions. Ranges come
        either from decorator arguments (``@dace.map(_[0:N, 0:M])``) or
        per-argument annotations (``def f(i: _[0:N], j: _[0:M])``). Returns
        ``None`` if the function is malformed (mismatched dimension count),
        leaving it for MarkOpaque.
        """
        dimension_slices: List[ast.expr] = []
        if isinstance(decorator, ast.Call) and decorator.args:
            range_argument = decorator.args[0]
            if isinstance(range_argument, ast.Subscript):
                index = range_argument.slice
                dimension_slices = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        else:
            for argument in function.args.args:
                if isinstance(argument.annotation, ast.Subscript):
                    dimension_slices.append(argument.annotation.slice)
        if len(dimension_slices) != len(function.args.args) or not dimension_slices:
            return None
        parameter_names = [argument.arg for argument in function.args.args]
        return parameter_names, dimension_slices

    def _build_map_for_loop(self, function: ast.FunctionDef, parameter_names: List[str],
                            dimension_slices: List[ast.expr], body: List[ast.stmt]) -> ast.For:
        """Build the ``for <params> in dace.map[<ranges>]:`` loop shape shared
        by ``@dace.map`` and ``@dace.mapscope`` desugaring."""
        if len(parameter_names) == 1:
            loop_target = _name_store(parameter_names[0], function)
        else:
            loop_target = _located(
                ast.Tuple(elts=[_name_store(parameter, function) for parameter in parameter_names], ctx=ast.Store()),
                function)
        map_attribute = _located(ast.Attribute(value=_name_load('dace', function), attr='map', ctx=ast.Load()),
                                 function)
        if len(dimension_slices) == 1:
            index: ast.expr = dimension_slices[0]
        else:
            index = _located(ast.Tuple(elts=dimension_slices, ctx=ast.Load()), function)
        iterator = _located(ast.Subscript(value=map_attribute, slice=index, ctx=ast.Load()), function)
        return _located(ast.For(target=loop_target, iter=iterator, body=body, orelse=[]), function)

    def _desugar_map_function(self, function: ast.FunctionDef, decorator: ast.expr) -> ast.stmt:
        """Turn ``@dace.map``-decorated functions into dace.map for-loops with a tasklet body."""
        extraction = self._extract_map_ranges(function, decorator)
        if extraction is None:
            # Malformed explicit map: leave for MarkOpaque
            return function
        parameter_names, dimension_slices = extraction
        tasklet = ExplicitTasklet(label=function.name, statements=function.body, location=function)
        return self._build_map_for_loop(function, parameter_names, dimension_slices, [tasklet])

    def _desugar_mapscope_function(self, function: ast.FunctionDef, decorator: ast.expr) -> ast.stmt:
        """
        Turn ``@dace.mapscope``-decorated functions into dace.map for-loops
        whose body is the function's own statements (rather than a single
        tasklet), recursively canonicalized by this same pass so nested
        ``@dace.map``/``@dace.tasklet``/``@dace.mapscope`` functions inside
        the body desugar too.
        """
        extraction = self._extract_map_ranges(function, decorator)
        if extraction is None:
            # Malformed explicit mapscope: leave for MarkOpaque
            return function
        parameter_names, dimension_slices = extraction
        body = self._transform_body(function.body)
        return self._build_map_for_loop(function, parameter_names, dimension_slices, body)


def _refers_to(node: ast.expr, qualified_name: str, global_vars: dict) -> bool:
    """
    Check whether an expression actually resolves to a dace built-in (e.g.
    ``dace.tasklet``), rather than merely sharing its name with one. Bare
    names (``from dace import tasklet``) must resolve to the built-in through
    the program globals; attribute accesses must go through the dace module.

    Some explicit-dataflow decorator spellings (``dace.mapscope``,
    ``dace.consumescope``) are not real attributes of the ``dace`` module —
    the classic frontend (``newast.py``) recognizes them purely syntactically
    by their dotted name, not by attribute lookup. When ``qualified_name``
    names no such attribute, recognition falls back to checking that the
    attribute access's root resolves to the ``dace`` module itself; bare
    names cannot be verified this way (there is no built-in object to import)
    and are rejected.
    """
    import dace  # Deferred to avoid an import cycle during package initialization
    attribute_name = qualified_name.split('.', 1)[1]
    builtin = getattr(dace, attribute_name, None)
    if isinstance(node, ast.Name):
        if builtin is None:
            return False
        return global_vars.get(node.id) is builtin
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.attr == attribute_name:
        root = global_vars.get(node.value.id)
        if root is None:
            # Preprocessing rewrites aliased module imports to the real module
            # name, which may be absent from the caller's globals.
            return node.value.id == 'dace'
        if builtin is None:
            return root is dace
        return getattr(root, attribute_name, None) is builtin
    return False


def _tasklet_arguments(node: ast.expr) -> dict:
    """Extract :class:`ExplicitTasklet` keyword arguments (language,
    side-effect flag, global/init/exit code) from a ``dace.tasklet(...)``
    call, if present."""
    arguments = {'language': None, 'side_effects': None, 'code_global': '', 'code_init': '', 'code_exit': ''}
    if isinstance(node, ast.Call):
        positional = [a for a in node.args if isinstance(a, ast.Constant)]
        if positional and isinstance(positional[0].value, str):
            arguments['language'] = positional[0].value
        for keyword in node.keywords:
            if keyword.arg in arguments and isinstance(keyword.value, ast.Constant):
                arguments[keyword.arg] = keyword.value.value
    return arguments


class DesugarStatements(_BodyTransformer):
    """
    Desugar statement forms that have direct canonical equivalents:

    - ``a = b = expr`` becomes ``a = expr; b = a``.
    - ``a, b = c, d`` becomes ``t0 = c; t1 = d; a = t0; b = t1`` (right-hand
      side evaluated fully before any target is assigned, per Python).
    - ``x += expr`` becomes ``x = x + expr``.
    - ``x: T = expr`` becomes ``x = expr``, carrying the declared type on the
      ``Assign`` node (``annotation`` attribute) as a descriptor hint for the
      lowering stage. Bare declarations (``x: T`` without a value) stay as
      canonical ``AnnAssign`` nodes so the hint (e.g. a Reference type) is
      applied when the name is first assigned.
    - ``for``/``while`` ``else`` clauses become explicit did-break flags.
    - Docstring expression statements are removed.
    """
    name = 'desugar-statements'

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES):
            return statement
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str):
            return None
        if isinstance(statement, ast.Assign) and len(statement.targets) > 1:
            first, *rest = statement.targets
            statements: List[ast.stmt] = [
                _located(ast.Assign(targets=[first], value=statement.value), statement),
            ]
            source = copy.deepcopy(first)
            for node in ast.walk(source):
                if hasattr(node, 'ctx'):
                    node.ctx = ast.Load()
            for target in rest:
                statements.append(_located(ast.Assign(targets=[target], value=copy.deepcopy(source)), statement))
            return self._transform_body(statements)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            unpacked = self._desugar_tuple_swap(statement)
            if unpacked is not None:
                return self._transform_body(unpacked)
        if isinstance(statement, ast.AugAssign):
            read_target = copy.deepcopy(statement.target)
            for node in ast.walk(read_target):
                if hasattr(node, 'ctx'):
                    node.ctx = ast.Load()
            value = _located(ast.BinOp(left=read_target, op=statement.op, right=statement.value), statement)
            return _located(ast.Assign(targets=[statement.target], value=value), statement)
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                # Bare declaration: canonical as-is (a descriptor hint for the
                # first assignment to the name, e.g. Reference declarations).
                return statement if isinstance(statement.target, ast.Name) else None
            assign = _located(ast.Assign(targets=[statement.target], value=statement.value), statement)
            # Keep the declared type as a descriptor hint for the lowering
            # stage (classic frontend semantics: annotations type the target,
            # including results inference cannot see, e.g. callback returns).
            assign.annotation = statement.annotation
            return assign
        if isinstance(statement, (ast.For, ast.While)) and statement.orelse:
            return self._desugar_loop_else(statement)
        return self._recurse(statement)

    def _desugar_tuple_swap(self, statement: ast.Assign) -> Optional[List[ast.stmt]]:
        """
        Unpack a tuple-to-tuple assignment (``a, b = c, d``) into temporaries
        followed by individual assignments, or None if the statement is not of
        that shape. Starred and nested-tuple targets are left untouched.
        """
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, (ast.Tuple, ast.List)) or not isinstance(value, (ast.Tuple, ast.List)):
            return None
        if len(target.elts) != len(value.elts):
            return None
        if any(isinstance(element, (ast.Starred, ast.Tuple, ast.List)) for element in target.elts + value.elts):
            return None
        statements: List[ast.stmt] = []
        temporaries: List[str] = []
        for element in value.elts:
            temp = self.context.fresh_name('__unpack')
            temporaries.append(temp)
            statements.append(_assign(temp, element, statement))
        for element, temp in zip(target.elts, temporaries):
            statements.append(_located(ast.Assign(targets=[element], value=_name_load(temp, statement)), statement))
        return statements

    def _desugar_loop_else(self, loop: Union[ast.For, ast.While]) -> List[ast.stmt]:
        """Rewrite a loop-else clause using an explicit did-break flag."""
        flag = self.context.fresh_name('__did_break')
        self._flag_breaks(loop.body, flag, loop)
        else_body = loop.orelse
        loop.orelse = []
        else_if = _located(
            ast.If(test=_located(ast.UnaryOp(op=ast.Not(), operand=_name_load(flag, loop)), loop),
                   body=else_body,
                   orelse=[]), loop)
        result: List[ast.stmt] = [
            _assign(flag, _located(ast.Constant(value=False), loop), loop),
            loop,
            else_if,
        ]
        return self._transform_body(result)

    def _flag_breaks(self, body: List[ast.stmt], flag: str, template: ast.AST) -> None:
        """Prefix every break belonging to this loop with a flag assignment."""
        for i, statement in enumerate(list(body)):
            if isinstance(statement, ast.Break):
                index = body.index(statement)
                body.insert(index, _assign(flag, _located(ast.Constant(value=True), template), template))
            elif isinstance(statement, (ast.For, ast.While)):
                continue  # Breaks inside nested loops belong to the nested loop
            elif isinstance(statement, ast.If):
                self._flag_breaks(statement.body, flag, template)
                self._flag_breaks(statement.orelse, flag, template)
            elif isinstance(statement, (ast.With, ast.Try)):
                for field in ('body', 'orelse', 'finalbody'):
                    child = getattr(statement, field, None)
                    if child:
                        self._flag_breaks(child, flag, template)


class NormalizeLoops(_BodyTransformer):
    """
    Normalize loop headers:

    - ``range(stop)`` / ``range(start, stop)`` become ``range(start, stop, step)``.
    - ``while`` loops with non-atomic tests become ``while True`` loops whose
      body starts with ``if not <test>: break``. The rewritten test re-evaluates
      at the loop head, so ``break`` and ``continue`` behave correctly, and the
      subsequent ANF pass may freely hoist temporaries for the test inside the
      loop body.
    """
    name = 'normalize-loops'

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES):
            return statement
        if isinstance(statement, ast.For) and isinstance(statement.iter, ast.Call) and isinstance(
                statement.iter.func, ast.Name) and statement.iter.func.id == 'range' and not statement.iter.keywords:
            args = statement.iter.args
            if len(args) == 1:
                zero = _located(ast.Constant(value=0), statement.iter)
                one = _located(ast.Constant(value=1), statement.iter)
                statement.iter.args = [zero, args[0], one]
            elif len(args) == 2:
                one = _located(ast.Constant(value=1), statement.iter)
                statement.iter.args = [args[0], args[1], one]
            return self._recurse(statement)
        if isinstance(statement, ast.While) and not cpa.is_atomexpr(statement.test):
            break_if = _located(
                ast.If(test=_located(ast.UnaryOp(op=ast.Not(), operand=statement.test), statement.test),
                       body=[_located(ast.Break(), statement)],
                       orelse=[]), statement)
            statement.test = _located(ast.Constant(value=True), statement)
            statement.body.insert(0, break_if)
            return self._recurse(statement)
        return self._recurse(statement)


class ANFTransform(_BodyTransformer):
    """
    Convert expressions to A-normal form: compound subexpressions are hoisted
    into fresh single-assignment temporaries so all remaining expressions match
    the canonical "flat" grammar.

    Short-circuit positions are treated conservatively: subexpressions that
    contain calls are never hoisted out of non-leading ``BoolOp`` operands or
    out of conditional-expression branches; such statements are left intact for
    :class:`MarkOpaque`.
    """
    name = 'anf'

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES) or isinstance(statement, _TERMINAL_STMTS):
            return statement
        hoisted: List[ast.stmt] = []
        try:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                statement.value = self._flatten(statement.value, hoisted, level='flat')
                statement.targets[0] = self._flatten_target(statement.targets[0], hoisted)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                statement.value = self._flatten(statement.value, hoisted, level='flat')
            elif isinstance(statement, ast.If):
                statement.test = self._flatten(statement.test, hoisted, level='atomexpr')
                self._recurse(statement)
            elif isinstance(statement, ast.While):
                # Tests are atomic after NormalizeLoops; nothing to hoist.
                self._recurse(statement)
            elif isinstance(statement, ast.For):
                if isinstance(statement.iter, ast.Call):
                    statement.iter.args = [self._flatten(a, hoisted, level='atom') for a in statement.iter.args]
                elif isinstance(statement.iter, ast.Subscript):
                    statement.iter.slice = self._flatten_index(statement.iter.slice, hoisted)
                self._recurse(statement)
            elif isinstance(statement, ast.Return) and statement.value is not None:
                if isinstance(statement.value, ast.Tuple):
                    statement.value.elts = [self._flatten(e, hoisted, level='atom') for e in statement.value.elts]
                else:
                    statement.value = self._flatten(statement.value, hoisted, level='atom')
            else:
                self._recurse(statement)
        except _ShortCircuitHazard:
            return statement  # Left non-canonical on purpose; MarkOpaque handles it
        return hoisted + [statement] if hoisted else statement

    def _hoist(self, expr: ast.expr, hoisted: List[ast.stmt]) -> ast.Name:
        """Assign an expression to a fresh temporary and return its name node."""
        temp = self.context.fresh_name('__anf')
        hoisted.append(_assign(temp, expr, expr))
        return _name_load(temp, expr)

    def _flatten(self, expr: ast.expr, hoisted: List[ast.stmt], level: str) -> ast.expr:
        """
        Flatten an expression to the requested canonical level (``'atom'``,
        ``'operand'``, ``'atomexpr'``, or ``'flat'``), hoisting temporaries as
        needed. Operands (operator arguments) may be data subscripts; plain
        atoms may not.
        """
        if isinstance(expr, ast.IfExp):
            raise _ShortCircuitHazard
        if isinstance(expr, ast.BoolOp):
            if any(_contains_call(v) for v in expr.values[1:]):
                raise _ShortCircuitHazard
            expr.values = [self._flatten(v, hoisted, level='operand') for v in expr.values]
            return expr if level not in ('atom', 'operand') else self._hoist(expr, hoisted)
        if isinstance(expr, ast.UnaryOp):
            expr.operand = self._flatten(expr.operand, hoisted, level='atom' if level == 'atom' else 'operand')
            return expr
        if isinstance(expr, ast.BinOp):
            expr.left = self._flatten(expr.left, hoisted, level='operand')
            expr.right = self._flatten(expr.right, hoisted, level='operand')
            return expr if level not in ('atom', 'operand') else self._hoist(expr, hoisted)
        if isinstance(expr, ast.Compare):
            expr.left = self._flatten(expr.left, hoisted, level='operand')
            expr.comparators = [self._flatten(c, hoisted, level='operand') for c in expr.comparators]
            return expr if level not in ('atom', 'operand') else self._hoist(expr, hoisted)
        if isinstance(expr, ast.Subscript) and isinstance(expr.ctx, ast.Load):
            if not cpa.is_dataref(expr.value):
                # Atom-level flattening also reduces nested structure-member
                # chains (``outer.inner.data``) to single-level datarefs.
                expr.value = self._flatten(expr.value, hoisted, level='atom')
            if not cpa.is_dataref(expr.value):
                raise _ShortCircuitHazard
            expr.slice = self._flatten_index(expr.slice, hoisted)
            return expr if level != 'atom' else self._hoist(expr, hoisted)
        if isinstance(expr, ast.Call):
            if not cpa.is_atom(expr.func):
                raise _ShortCircuitHazard
            expr.args = [self._flatten(a, hoisted, level='atom') for a in expr.args]
            for keyword in expr.keywords:
                if keyword.arg is None:
                    raise _ShortCircuitHazard
                keyword.value = self._flatten(keyword.value, hoisted, level='atom')
            return expr if level == 'flat' else self._hoist(expr, hoisted)
        if isinstance(expr, (ast.List, ast.Tuple)) and isinstance(getattr(expr, 'ctx', ast.Load()), ast.Load):
            # Sequence literals: flatten elements to atoms; the literal itself
            # is a compile-time value, so it may stay in assignment position
            # ('flat') and is hoisted to a named temporary elsewhere.
            expr.elts = [self._flatten(element, hoisted, level='atom') for element in expr.elts]
            return expr if level == 'flat' else self._hoist(expr, hoisted)
        if isinstance(expr, ast.Attribute) and not cpa.is_dataref(expr):
            # Nested structure-member chains (``outer.inner.leaf``) reduce to
            # single-level datarefs; chains not rooted at a Name stay atoms.
            expr = self._as_dataref(expr, hoisted)
        if cpa.is_atom(expr):
            return expr
        raise _ShortCircuitHazard

    def _flatten_target(self, target: ast.expr, hoisted: List[ast.stmt]) -> ast.expr:
        if isinstance(target, ast.Attribute) and not cpa.is_dataref(target):
            return self._as_dataref(target, hoisted)
        if isinstance(target, ast.Subscript):
            if not cpa.is_dataref(target.value):
                target.value = self._as_dataref(target.value, hoisted)
            target.slice = self._flatten_index(target.slice, hoisted)
        return target

    def _as_dataref(self, node: ast.expr, hoisted: List[ast.stmt]) -> ast.expr:
        """
        Reduce a chain of attribute accesses over a name (a nested structure
        member, e.g. ``outer.inner.leaf``) to the single-level dataref grammar
        (``Name`` or ``Attribute(Name)``), hoisting intermediate member
        accesses into fresh temporaries (``__anf0 = outer.inner``,
        ``__anf0.leaf``). This lets structures of structures lower through the
        same single-level member-access machinery as top-level members,
        without widening the dataref grammar itself.

        Chains not rooted at a ``Name`` (e.g. attribute access on an embedded
        constant) are returned unchanged: they remain legal atoms and either
        resolve semantically or fall back during lowering.
        """
        if cpa.is_dataref(node) or not isinstance(node, ast.Attribute):
            return node
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            return node
        base = self._as_dataref(node.value, hoisted)
        if not isinstance(base, ast.Name):
            base = self._hoist(base, hoisted)
        node.value = base
        return node

    def _flatten_index(self, index: ast.expr, hoisted: List[ast.stmt]) -> ast.expr:
        if isinstance(index, ast.Tuple):
            index.elts = [self._flatten_index(e, hoisted) for e in index.elts]
            return index
        if isinstance(index, ast.Slice):
            for field in ('lower', 'upper', 'step'):
                part = getattr(index, field)
                if part is not None:
                    setattr(index, field, self._flatten_index_expr(part, hoisted))
            return index
        return self._flatten_index_expr(index, hoisted)

    def _flatten_index_expr(self, node: ast.expr, hoisted: List[ast.stmt]) -> ast.expr:
        """
        Flatten a canonical index expression (see ``cpa.is_index_expr``): a
        ``BinOp`` of two operands is kept in place with its sides reduced to
        operand level; anything else goes through :meth:`_flatten_operand`.
        Unlike :meth:`_flatten`, the ``BinOp`` itself is never hoisted here —
        it is canonical directly in index position (e.g. ``A[i + 1]``).
        """
        if isinstance(node, ast.BinOp):
            node.left = self._flatten_operand(node.left, hoisted)
            node.right = self._flatten_operand(node.right, hoisted)
            return node
        return self._flatten_operand(node, hoisted)

    def _flatten_operand(self, node: ast.expr, hoisted: List[ast.stmt]) -> ast.expr:
        """
        Flatten an expression to canonical operand level (``cpa.is_operand``):
        atoms pass through unchanged; a unary op recurses into its operand; a
        data subscript (``A_col[j]``) is kept in place with its own index
        recursively flattened -- it is canonical directly in operand
        position, so it is never hoisted, unlike the general ``'operand'``
        level of :meth:`_flatten`. Anything else (calls, nested compound
        expressions, ...) is hoisted to a temporary via the existing
        atom-level machinery.
        """
        if isinstance(node, ast.Attribute) and not cpa.is_dataref(node):
            node = self._as_dataref(node, hoisted)
        if cpa.is_atom(node):
            return node
        if isinstance(node, ast.UnaryOp):
            node.operand = self._flatten_operand(node.operand, hoisted)
            return node
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and cpa.is_dataref(node.value):
            node.slice = self._flatten_index(node.slice, hoisted)
            return node
        return self._flatten(node, hoisted, level='atom')


class _ShortCircuitHazard(Exception):
    """Internal: expression cannot be hoisted without changing semantics."""
    pass


def _contains_call(node: ast.AST) -> bool:
    return any(isinstance(descendant, ast.Call) for descendant in ast.walk(node))


class MarkOpaque(_BodyTransformer):
    """
    Final canonicalization pass: wrap every statement that is still outside
    the CPA subset in an :class:`OpaqueStmt` marker with computed input/output
    sets. After this pass, canonicalization is total by construction.
    """
    name = 'mark-opaque'

    def transform_statement(self, statement: ast.stmt) -> Union[ast.stmt, List[ast.stmt], None]:
        if isinstance(statement, CANONICAL_LEAVES):
            return statement
        violations = list(cpa._violations_in_statement(statement))
        if violations:
            return self._wrap(statement, violations[0])
        return self._recurse(statement)

    def _wrap(self, statement: ast.stmt, reason: str) -> OpaqueStmt:
        reads, writes = cpa.statement_io_sets(statement)
        return OpaqueStmt(statement, reason, reads, writes)


class BatchOpaque(_BodyTransformer):
    """
    Coalesce maximal runs of adjacent :class:`OpaqueStmt` siblings into a
    single marker, so consecutive interpreter statements become one Python
    callback instead of many. Both statements already executed adjacently in
    the interpreter, so merging changes callback granularity, not semantics.

    The merged I/O sets chain dataflow through the run: a name produced by an
    earlier statement and consumed by a later one is not an input of the
    merged marker.
    """
    name = 'batch-opaque'

    def _transform_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        transformed = super()._transform_body(body)
        result: List[ast.stmt] = []
        for statement in transformed:
            if isinstance(statement, OpaqueStmt) and result and isinstance(result[-1], OpaqueStmt):
                result[-1] = _merge_opaque(result[-1], statement)
            else:
                result.append(statement)
        return result


def _merge_opaque(first: OpaqueStmt, second: OpaqueStmt) -> OpaqueStmt:
    """Merge two adjacent opaque markers into one, chaining their I/O sets."""
    inputs = set(first.inputs) | (set(second.inputs) - set(first.outputs))
    outputs = set(first.outputs) | set(second.outputs)
    reasons = list(dict.fromkeys([first.reason, second.reason]))  # Distinct, order-preserving
    return OpaqueStmt(first.original, '; '.join(reasons), inputs, outputs, originals=first.originals + second.originals)


def default_passes() -> List[_BodyTransformer]:
    """The default canonicalization pass order."""
    return [
        RecognizeExplicitDataflow(),
        DesugarStatements(),
        NormalizeLoops(),
        ANFTransform(),
        MarkOpaque(),
        BatchOpaque(),
    ]
