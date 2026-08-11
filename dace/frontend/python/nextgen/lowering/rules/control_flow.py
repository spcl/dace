# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical control flow: ``if``/``elif``/``else``,
``while``, range loops, ``dace.map`` loops, ``break``/``continue``/``pass``.

Loops and maps are emitted with *real* :class:`~dace.sdfg.state.LoopRegion`
and :class:`~dace.sdfg.nodes.MapEntry` objects (not stringly-typed frontend
metadata), so memlet propagation and downstream analysis behave identically
for frontend-produced and SDFG-derived schedule trees.
"""
import ast
import numbers
from typing import List, Optional, Tuple

import numpy

from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python import iterators
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access, resolve_symbol_names
from dace.frontend.python.nextgen.lowering.mechanisms import opaque_values, static_values
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.context import BindingSnapshot
from dace.frontend.python.nextgen.semantics.joins import merge_branches


@rule(ast.If)
def lower_if(statement: ast.If, state: LoweringState) -> None:
    """
    Lower an if/elif/else chain with branch-scoped bindings: each branch is
    lowered from the pre-chain binding state, and the branch-end states are
    merged at the join (see :mod:`~...semantics.joins`). If the join cannot be
    merged soundly, the whole chain is rolled back and re-lowered as a single
    Python callback.
    """
    from dace.frontend.python.nextgen.lowering import dispatch
    mark = state.emitter.checkpoint()
    before = state.context.snapshot()
    try:
        branch_scopes, branch_ends = _lower_if_chain(statement, before, state)
        state.context.restore(before)
        merge_branches(before, branch_ends, branch_scopes, statement, state)
    except UnsupportedFeatureError as reason:
        state.emitter.rollback(mark)
        state.context.restore(before)
        dispatch.fallback_to_callback(statement, state, reason)


def _lower_if_chain(statement: ast.If, before: BindingSnapshot,
                    state: LoweringState) -> Tuple[List[Optional[tn.ScheduleTreeScope]], List[BindingSnapshot]]:
    """
    Emit the scopes of an if/elif/else chain, lowering every branch from the
    ``before`` binding state, and collect (scope, end-state) per path. A chain
    without ``else`` contributes an implicit fall-through path (scope None,
    end state ``before``).
    """
    branch_scopes: List[Optional[tn.ScheduleTreeScope]] = []
    branch_ends: List[BindingSnapshot] = []

    def _lower_branch(scope: tn.ScheduleTreeScope, body: List[ast.stmt]) -> None:
        with state.emitter.scope(scope):
            state.lower_body(body)
        branch_scopes.append(scope)
        branch_ends.append(state.context.snapshot())
        state.context.restore(before)

    opaque_values.reject_opaque_condition(statement.test, statement, state)
    condition = CodeBlock(astutils.unparse(resolve_symbol_names(statement.test, state)))
    _lower_branch(tn.IfScope(condition=condition, children=[]), statement.body)

    orelse = statement.orelse
    while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        elif_statement = orelse[0]
        opaque_values.reject_opaque_condition(elif_statement.test, elif_statement, state)
        condition = CodeBlock(astutils.unparse(resolve_symbol_names(elif_statement.test, state)))
        _lower_branch(tn.ElifScope(condition=condition, children=[]), elif_statement.body)
        orelse = elif_statement.orelse

    if orelse:
        _lower_branch(tn.ElseScope(children=[]), orelse)
    else:
        branch_scopes.append(None)
        branch_ends.append(before)
    return branch_scopes, branch_ends


@rule(ast.While)
def lower_while(statement: ast.While, state: LoweringState) -> None:

    def _emit(state: LoweringState) -> None:
        opaque_values.reject_opaque_condition(statement.test, statement, state)
        condition = astutils.unparse(resolve_symbol_names(statement.test, state))
        loop = LoopRegion(f'while_{statement.lineno}', condition_expr=condition)
        with state.emitter.scope(tn.WhileScope(loop=loop, children=[])):
            state.lower_body(statement.body)

    _lower_loop_with_stability_check(statement, _emit, state)


@rule(ast.For)
def lower_for(statement: ast.For, state: LoweringState) -> None:
    if cpa.is_range_iterator(statement.iter, state.context.globals):
        # The iteration variable is the loop's OWN binding, not a carried one:
        # a sequential loop rebinds it to its symbol, and Python leaves that
        # value visible afterwards (``i = -1; for i in range(N): ...; if i >
        # 10``). A dace.map's parameter gets no such exemption -- it is scoped
        # to the map body, so a name that outlives the scope really would need
        # a merge.
        _lower_loop_with_stability_check(statement,
                                         lambda s: _lower_range_loop(statement, s),
                                         state,
                                         owned_names=(statement.target.id, ))
    elif cpa.is_dace_map_iterator(statement.iter, state.context.globals):
        _lower_loop_with_stability_check(statement, lambda s: _lower_map_loop(statement, s), state)
    else:
        raise UnsupportedFeatureError(
            f'Non-canonical for-iterator reached lowering: '
            f'{astutils.unparse(statement.iter)}', state.context.filename, statement)


def _lower_loop_with_stability_check(statement: ast.stmt,
                                     emit_loop,
                                     state: LoweringState,
                                     owned_names: Tuple[str, ...] = ()) -> None:
    """
    Lower a loop and enforce the loop-entry stability rule: any name bound
    before the loop whose binding the body changed (a different container,
    kind, or static value) would need a φ at the loop head, which the binding
    design intentionally avoids — the loop rolls back and re-lowers as a
    single Python callback instead. In-place rebinding through the same
    container (the common case for scalars) passes.

    One class of instability is repaired rather than rejected: a name bound to
    a COMPILE-TIME VALUE before the loop that the body materializes into a
    container (``i = numpy.int32(1)`` … ``while …: i += 1``). Nothing about
    that needs a merge — the name simply has to be a container by the time the
    loop starts — so the promotion is emitted ahead of the loop and the body
    re-lowered once against it (:func:`_promote_and_retry`).
    """
    from dace.frontend.python.nextgen.lowering import dispatch
    mark = state.emitter.checkpoint()
    before = state.context.snapshot()
    try:
        emit_loop(state)
    except UnsupportedFeatureError as reason:
        state.emitter.rollback(mark)
        state.context.restore(before)
        dispatch.fallback_to_callback(statement, state, reason)
        return
    reason = _loop_instability(before, state, owned_names)
    if reason is None:
        return
    promotable = _promotable_names(before, state)
    state.emitter.rollback(mark)
    state.context.restore(before)
    if promotable and _promote_and_retry(promotable, statement, emit_loop, state, owned_names):
        return
    state.emitter.rollback(mark)
    state.context.restore(before)
    dispatch.fallback_to_callback(statement, state, reason, category='loop-stability')


def _promotable_names(before: BindingSnapshot, state: LoweringState) -> List[str]:
    """
    Names the loop body turned from a compile-time VALUE into a container.

    These are the instabilities a promotion before the loop repairs: the value
    is the same either way, so binding it to a container up front costs only
    the constant-folding of its pre-loop reads — and the alternative is running
    the whole loop in the interpreter. Every other instability (a container
    rebound to a differently-shaped one, a name unbound) genuinely needs a
    merge and is left alone.
    """
    promotable = []
    for name, binding in before.bindings.items():
        current = state.context.bindings.get(name)
        if (binding.kind == 'constant' and current is not None and current.kind == 'container'
                and name in before.constant_values):
            promotable.append(name)
    return promotable


def _promote_and_retry(names: List[str],
                       statement: ast.stmt,
                       emit_loop,
                       state: LoweringState,
                       owned_names: Tuple[str, ...] = ()) -> bool:
    """
    Bind each name in ``names`` to a container holding its compile-time value,
    then lower the loop again. Returns True if the retry is stable; the caller
    rolls everything back and falls back otherwise.
    """
    from dace.frontend.python.nextgen.lowering.registry import lower_statement
    for name in names:
        value = state.context.constant_values[name]
        assignment = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(value=value)), statement)
        ast.fix_missing_locations(assignment)
        try:
            lower_statement(assignment, state)
        except UnsupportedFeatureError:
            return False
        if state.context.bindings[name].kind != 'container':
            return False  # Not representable as a container (e.g. an enum value)
    promoted = state.context.snapshot()
    try:
        emit_loop(state)
    except UnsupportedFeatureError:
        return False
    return _loop_instability(promoted, state, owned_names) is None


def _loop_instability(before: BindingSnapshot, state: LoweringState, owned_names: Tuple[str,
                                                                                        ...] = ()) -> Optional[str]:
    """The reason a loop body is binding-unstable, or None if it is stable.
    Names first bound inside the body are loop-local and always stable, and so
    are ``owned_names`` -- names the loop itself binds (its iteration
    variable)."""
    for name, binding in before.bindings.items():
        if name in owned_names:
            continue
        current = state.context.bindings.get(name)
        if current is None:
            return f'loop body unbinds "{name}"'
        if (current.kind, current.container) != (binding.kind, binding.container):
            return f'loop-carried rebinding of "{name}" requires a merge at the loop head'
        if binding.kind == 'static' and (state.context.static_values.get(name) is not before.static_values.get(name)):
            return f'loop-carried compile-time value change of "{name}"'
    return None


def _lower_range_loop(statement: ast.For, state: LoweringState) -> None:
    source_name = statement.target.id
    start, stop, step = (astutils.unparse(resolve_symbol_names(argument, state)) for argument in statement.iter.args)
    comparator = '<'
    try:
        if (symbolic.pystr_to_symbolic(step) < 0) == True:
            comparator = '>'
    except TypeError:
        pass

    # A loop variable that shadows a CONTAINER of the same name (``i = -1``
    # before the loop) needs a symbol name of its own: an SDFG rejects a symbol
    # and a data descriptor sharing one.
    loop_variable = source_name
    if source_name in state.context.containers:
        loop_variable = state.context.fresh_name(f'{source_name}_')
    state.context.bind_symbol(source_name, _index_dtype(statement.iter.args, state), symbol_name=loop_variable)
    loop = LoopRegion(f'for_{statement.lineno}',
                      condition_expr=f'{loop_variable} {comparator} {stop}',
                      loop_var=loop_variable,
                      initialize_expr=f'{loop_variable} = {start}',
                      update_expr=f'{loop_variable} = {loop_variable} + {step}')
    with state.emitter.scope(tn.ForScope(loop=loop, children=[])):
        state.lower_body(statement.body)


def _lower_map_loop(statement: ast.For, state: LoweringState) -> None:
    targets = statement.target.elts if isinstance(statement.target, ast.Tuple) else [statement.target]
    params = [target.id for target in targets]
    generator = iterators.iteration_object(statement.iter, state.context.globals)
    if generator is None:
        raise UnsupportedFeatureError(f'Could not resolve dace.map iterator: {astutils.unparse(statement.iter)}',
                                      state.context.filename,
                                      statement.iter,
                                      category='explicit-map')
    schedule = generator.schedule
    dynamic_inputs: List[tn.DynScopeCopyNode] = []
    ranges = _parse_map_ranges(generator.rng, statement.iter, state, dynamic_inputs)
    if len(params) != len(ranges):
        raise UnsupportedFeatureError('Number of dace.map indices does not match number of ranges',
                                      state.context.filename,
                                      statement,
                                      category='explicit-map')
    shadowed = {param: state.context.bindings.get(param) for param in params}
    for param in params:
        state.context.bind_symbol(param)
    # Dynamic-range inputs (data-dependent bounds) are emitted as siblings
    # immediately preceding the map scope, matching how SDFG-derived schedule
    # trees place them (see sdfg_to_tree.py) rather than as children inside
    # the scope: the scope emitter appends the scope node itself to the
    # *current* (enclosing) scope before entering it, so emitting these first
    # places them right before the map.
    for dynamic_input in dynamic_inputs:
        state.emitter.emit(dynamic_input)
    map_ = nodes.Map(f'map_{statement.lineno}', params, subsets.Range(ranges))
    if schedule is not None:
        map_.schedule = schedule
    map_node = nodes.MapEntry(map_)
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.lower_body(statement.body)
    # A map parameter is scoped to the map: it exists as a symbol only inside
    # the scope. Leaving the binding in place made the name look loop-carried
    # to a LATER scope that reuses it (``for i in dace.map[...]`` twice, the
    # second body assigning ``i = 1``), which rolled that whole scope back to a
    # callback. Restoring what the name meant before the map also un-shadows
    # any outer binding it hid.
    for param, previous in shadowed.items():
        if previous is None:
            state.context.bindings.pop(param, None)
        else:
            state.context.bindings[param] = previous


def _parse_map_ranges(rng: ast.expr, location: ast.expr, state: LoweringState,
                      dynamic_inputs: List[tn.DynScopeCopyNode]) -> List[Tuple]:
    """Parse the range of a ``dace.map[start:stop:step, ...]`` generator into inclusive-end symbolic ranges.

    :param rng: The generator's range, still an AST: evaluating the header
                (see :mod:`~dace.frontend.python.iterators`) leaves
                subscript indices unevaluated, precisely so that the symbolic
                and data-dependent bounds below survive to here.
    :param location: The header node, for error reporting.
    :param dynamic_inputs: Collects a :class:`~dace.sdfg.analysis.schedule_tree.treenodes.DynScopeCopyNode`
                           for every data-dependent bound encountered (see :func:`_dynamic_bound`).
    """
    dimensions = rng.elts if isinstance(rng, ast.Tuple) else [rng]
    ranges = []
    for dimension in dimensions:
        if not isinstance(dimension, ast.Slice):
            # A non-slice dimension (``dace.map[10]``) is an index, which the
            # classic frontend reads as the single-iteration range ``10:10:1``
            # (see ``newast.ProgramVisitor._parse_index_as_range``).
            index = _bound(dimension, None, state, dynamic_inputs)
            ranges.append((index, index, 1))
            continue
        start = _bound(dimension.lower, 0, state, dynamic_inputs)
        stop = _bound(dimension.upper, None, state, dynamic_inputs)
        step = _bound(dimension.step, 1, state, dynamic_inputs)
        if stop is None:
            raise UnsupportedFeatureError('dace.map dimensions require an upper bound',
                                          state.context.filename,
                                          location,
                                          category='explicit-map')
        ranges.append((start, stop - 1, step))
    return ranges


def _bound(node, default, state: LoweringState, dynamic_inputs: List[tn.DynScopeCopyNode]):
    """
    Resolve a single ``dace.map`` range bound to a symbolic expression.

    A bound that reads a data container — a scalar name, a scalar structure
    member, or a scalar array element like ``A_row[i]`` (index expressions
    are canonical in place, not hoisted) — becomes a fresh dynamic-map-range
    symbol fed by a :class:`DynScopeCopyNode`: see :func:`_dynamic_bound`.
    Purely symbolic expressions (``i + 1``) resolve symbolically. Compound
    expressions that mix data reads with arithmetic (``tmp + 1``) have each
    read promoted first, by :func:`_promote_bound_reads`.
    """
    if node is None:
        return default
    # ``A.shape[0]`` as a bound is a compile-time value, not a data read: fold
    # it before anything resolves ``A.shape`` as the container ``A``.
    node = static_values.fold_descriptor_properties(node, state)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        # An ANF temporary that HOLDS a compile-time value (``__anf0 =
        # ceiling(N / 32)``) is a symbolic bound, not a data-dependent one:
        # its materialized scalar is an artifact of hoisting the expression
        # out of the header, and reading it back at runtime would turn a
        # static range into a dynamic one.
        inferred = state.inference.infer(node)
        if inferred.kind == 'symbolic':
            return inferred.value
        # May raise UnsupportedFeatureError (e.g. a data-dependent index
        # inside the bound itself), falling the whole loop back to a callback.
        access = resolve_access(node, state)
        if access is not None:
            return _dynamic_bound(node, access, state, dynamic_inputs)
    node = _fold_symbolic_temporaries(node, state)
    node = _promote_bound_reads(node, state, dynamic_inputs)
    expression = astutils.unparse(resolve_symbol_names(node, state))
    try:
        return symbolic.pystr_to_symbolic(expression)
    except Exception:
        # Not symbolizable (e.g. references a value only the interpreter
        # knows); the loop falls back to a callback.
        raise UnsupportedFeatureError(f'Cannot parse loop bound "{expression}" symbolically',
                                      state.context.filename,
                                      node,
                                      category='dynamic-bound')


def _fold_symbolic_temporaries(node: ast.expr, state: LoweringState) -> ast.expr:
    """
    Replace every name INSIDE a compound ``dace.map`` bound that stands for a
    compile-time symbolic value with that value.

    :func:`_bound` already does this for a bound that *is* such a name. But
    canonicalization also hoists PARTS of a bound: ``dace.map[0:M * N * K]``
    becomes ``__anf0 = M * N`` followed by the range ``0:2 * __anf0``, and there
    the temporary sits inside a ``BinOp``. Nothing then substitutes it, so
    :func:`~dace.symbolic.pystr_to_symbolic` turns ``__anf0`` into a free symbol
    that no container and no symbol declares. It survives into the finished
    SDFG, where ``SDFG.arglist`` looks it up in ``symbols`` and raises
    ``KeyError: '__anf0'``.

    A name that resolves to data is left alone -- :func:`_promote_bound_reads`
    runs next and turns it into a dynamic-map-range symbol, which is the correct
    treatment for a bound the program computes at run time.
    """
    result = astutils.copy_tree(node)

    class _Folder(ast.NodeTransformer):

        def visit_Name(self, name_node: ast.Name) -> ast.AST:
            try:
                inferred = state.inference.infer(name_node)
            except Exception:
                return name_node  # Not inferable here; later stages report it
            if inferred is None or inferred.kind not in ('symbolic', 'constant'):
                return name_node
            value = inferred.value
            # A bound is a symbolic expression or an integer. A bool is neither
            # (Python makes it an integer, but nothing that reads a bound wants
            # ``True``), and a float would silently change the iteration space.
            if not symbolic.issymbolic(value):
                if not isinstance(value, numbers.Integral) or isinstance(value, (bool, numpy.bool_)):
                    return name_node
            try:
                return ast.parse(str(value), mode='eval').body
            except SyntaxError:
                # A value with no Python surface syntax: leave the name for the
                # ordinary paths to resolve or reject.
                return name_node

    return ast.fix_missing_locations(_Folder().visit(result))


def _promote_bound_reads(node: ast.expr, state: LoweringState, dynamic_inputs: List[tn.DynScopeCopyNode]) -> ast.expr:
    """
    Replace every data read inside a compound ``dace.map`` bound (``tmp + 1``,
    ``A_row[i] * 2``) with the dynamic-map-range symbol carrying its value.

    Without this the read survives into
    :func:`~dace.symbolic.pystr_to_symbolic`, which turns any identifier into a
    free symbol -- including a container's name, which nothing declares. The
    range then generated a loop over an undeclared variable, and the C++
    compiler was the first thing to notice.

    Reads that resolve to the same element share one symbol and one copy, so a
    bound like ``tmp:tmp + 1`` reads ``tmp`` once.
    """

    class _Promoter(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.expr:
            return self._promote(node)

        def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
            return self._promote(node)

        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            return self._promote(node)

        def _promote(self, node: ast.expr) -> ast.expr:
            # Left whole, deliberately: a data read is promoted as one access,
            # never by descending into the index expression of a subscript.
            if state.inference.infer(node).kind == 'symbolic':
                return node
            try:
                access = resolve_access(node, state)
            except UnsupportedFeatureError:
                return node
            if access is None:
                return node
            symbol = _dynamic_bound(node, access, state, dynamic_inputs)
            return ast.copy_location(ast.Name(id=symbol.name, ctx=ast.Load()), node)

    return _Promoter().visit(astutils.copy_tree(node))


def _dynamic_bound(node: ast.expr, access: DataAccess, state: LoweringState,
                   dynamic_inputs: List[tn.DynScopeCopyNode]) -> symbolic.symbol:
    """
    Turn a data-dependent ``dace.map`` bound (a scalar integer data access —
    a scalar container or a single array element like ``A_row[i]``) into a
    fresh symbol fed by a dynamic map-range input, recording a
    :class:`~dace.sdfg.analysis.schedule_tree.treenodes.DynScopeCopyNode`
    for the caller to emit right before the map scope.

    :raises UnsupportedFeatureError: If the access is not a single integer
        element (e.g. a whole array, a sub-range, or a floating-point
        scalar) -- those forms are not supported as dynamic map-range inputs.
    """
    if not access.is_scalar_access or access.descriptor.dtype not in dtypes.INTEGER_TYPES:
        raise UnsupportedFeatureError(
            f'Data-dependent dace.map bound "{astutils.unparse(node)}" must be a scalar integer '
            f'element (got subset {access.subset} of {access.descriptor})',
            state.context.filename,
            node,
            category='dynamic-bound')
    memlet = Memlet(data=access.container, subset=access.subset)
    # The same element read twice in one map header (``tmp:tmp + 1``) is one
    # value: reuse the symbol rather than copying it in again.
    for existing in dynamic_inputs:
        if existing.memlet.data == memlet.data and existing.memlet.subset == memlet.subset:
            return symbolic.symbol(existing.target, access.descriptor.dtype)
    symbol_name = state.context.fresh_name('__dyn')
    # A repository-only symbol: registered directly in the symbol table (which
    # *is* the tree root's symbol table), without a source-level name binding.
    state.context.symbols[symbol_name] = symbolic.symbol(symbol_name, access.descriptor.dtype)
    dynamic_inputs.append(tn.DynScopeCopyNode(target=symbol_name, memlet=memlet))
    return symbolic.symbol(symbol_name, access.descriptor.dtype)


def _index_dtype(bounds: List[ast.expr], state: LoweringState) -> dtypes.typeclass:
    return dtypes.int64


def _reject_in_dataflow_scope(keyword: str, statement: ast.stmt, state: LoweringState) -> None:
    """
    Refuse ``break``/``continue`` whose nearest enclosing scope is a dataflow
    scope (``dace.map``, ``dace.consume``) rather than a loop.

    A map's iterations are independent and may run in any order or in
    parallel, so there is no "rest of the loop" to abandon. Emitted anyway,
    the node had no loop to bind to and was simply dropped: the map ran its
    full range, and every iteration the guard was meant to skip executed. That
    silently computes the wrong answer whenever the writes stay in bounds --
    the stable frontend rejects the same program.
    """
    scope = state.emitter.current_scope
    while scope is not None:
        if isinstance(scope, tn.LoopScope):
            return
        if isinstance(scope, tn.DataflowScope):
            construct = 'dace.consume' if isinstance(scope, tn.ConsumeScope) else 'dace.map'
            raise DaceSyntaxError(
                None, statement, f'"{keyword}" is not allowed inside a {construct} scope: its iterations are '
                f'independent and may run in parallel, so there is no sequential order to {keyword} out of. '
                'Use a sequential "for" loop, or guard the body with an "if" instead.')
        scope = scope.parent


@rule(ast.Break)
def lower_break(statement: ast.Break, state: LoweringState) -> None:
    _reject_in_dataflow_scope('break', statement, state)
    state.emitter.emit(tn.BreakNode())


@rule(ast.Continue)
def lower_continue(statement: ast.Continue, state: LoweringState) -> None:
    _reject_in_dataflow_scope('continue', statement, state)
    state.emitter.emit(tn.ContinueNode())


@rule(ast.Pass)
def lower_pass(statement: ast.Pass, state: LoweringState) -> None:
    pass


@rule(cpa.NamedRegionStmt)
def lower_named_region(statement: cpa.NamedRegionStmt, state: LoweringState) -> None:
    """
    Lower ``with dace.named("label"):`` to a
    :class:`~...treenodes.NamedRegionScope`.

    The label groups statements without changing what they mean, so the body
    lowers into the scope exactly as it would outside one -- no binding scope
    of its own (Python's ``with`` introduces none either), and names assigned
    inside stay visible after it.
    """
    with state.emitter.scope(tn.NamedRegionScope(label=statement.label, children=[])):
        state.lower_body(statement.body)
