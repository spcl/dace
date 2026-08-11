# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for explicit-dataflow tasklets: ``with dace.tasklet:`` blocks,
``@dace.tasklet`` functions, and the tasklet bodies of ``@dace.map`` functions
(recognized during canonicalization as
:class:`~dace.frontend.python.nextgen.canonical.cpa.ExplicitTasklet` markers).

The memlet syntax (``local << A[i]`` for inputs, ``local >> B[i]`` and
``local >> B(1, lambda a, b: a + b)[i]`` for outputs) is parsed with the
shared :func:`~dace.frontend.python.memlet_parser.parse_memlet`, porting the
semantics of the stable frontend's ``TaskletTransformer``. A top-level string
statement provides intrinsic (C++) tasklet code.

Unlike general Python statements, malformed explicit-dataflow *structure*
(duplicate connectors, multiple intrinsic bodies, unknown languages) raises
frontend errors instead of falling back to callbacks: this is dace-specific
syntax, so a violation is a user error (matching the stable frontend's
``TaskletTransformer`` contract), not an unsupported-language feature. Memlet
*parse* failures fall back, however: the referenced name may be unavailable
only because its producer fell back to an interpreter callback, and the
replayed ``with dace.tasklet:`` block re-raises genuine syntax errors.

Global-scope, initialization, and finalization code attach to the tasklet
through ``with dace.tasklet(code_global=..., code_init=..., code_exit=...)``
keyword arguments and land on the emitted :class:`~dace.sdfg.nodes.Tasklet`'s
``code_global``/``code_init``/``code_exit`` properties.
"""
import ast
import copy
from typing import Dict, List, Optional

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.memlet_parser import parse_memlet
from dace.frontend.python.nextgen.canonical.cpa import ExplicitConsume, ExplicitTasklet
from dace.frontend.python.nextgen.common import FrontendError, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import (indirect_index_reads, promote_index_reads, resolve_access,
                                                          substitute_index_reads)
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.inference import _LocationShim


def _shim(state: LoweringState) -> _LocationShim:
    return _LocationShim(state.context.filename)


@rule(ExplicitTasklet)
def lower_explicit_tasklet(statement: ExplicitTasklet,
                           state: LoweringState,
                           extra_inputs: Optional[Dict[str, Memlet]] = None) -> None:
    """
    :param extra_inputs: Additional input connectors injected by an enclosing
                         construct (e.g. the popped stream element of a
                         consume scope), merged into the tasklet's inputs.
    """
    in_memlets: Dict[str, Memlet] = dict(extra_inputs or {})
    out_memlets: Dict[str, Memlet] = {}
    code_statements: List[ast.stmt] = []
    prelude: List[str] = []  # Indirection reads, prepended to the tasklet code
    epilogue: List[str] = []  # Indirection writes, appended to the tasklet code
    intrinsic_code: Optional[str] = None
    defined = state.context.defined_view()

    for body_statement in statement.statements:
        binop = _memlet_binop(body_statement)
        if binop is not None:
            indirect = indirect_index_reads(binop.right, state)
            source = binop.right
            if not indirect and isinstance(source, ast.Subscript):
                # A data read in a slice BOUND (``x[rows[i]:rows[i] + 2]``) is
                # a range bound, not indirection, so it belongs on the symbol
                # side -- the same rule the assignment paths apply through
                # ``resolve_access``. Left to the memlet parser it becomes an
                # un-evaluated ``rows(i)`` inside the subset.
                source = promote_index_reads(source, state)
                if source is not binop.right:
                    defined = state.context.defined_view()  # The promotion added symbols
            if isinstance(binop.op, ast.LShift):  # local << A[...]
                if indirect:
                    _lower_indirect_memlet(binop.left, binop.right, indirect, in_memlets, out_memlets, prelude,
                                           epilogue, True, state, body_statement)
                    continue
                connector, memlet = _parse_tasklet_memlet(source, binop.left, defined, state, body_statement)
                _check_connector(connector, in_memlets, out_memlets, state, body_statement)
                in_memlets[connector] = _to_repository(memlet, state, body_statement)
            else:  # local >> A[...]
                _reject_symbol_memlet_target(source, state, body_statement)
                if indirect:
                    _lower_indirect_memlet(binop.left, binop.right, indirect, in_memlets, out_memlets, prelude,
                                           epilogue, False, state, body_statement)
                    continue
                connector, memlet = _parse_tasklet_memlet(binop.left, source, defined, state, body_statement)
                _check_connector(connector, in_memlets, out_memlets, state, body_statement)
                out_memlets[connector] = _to_repository(memlet, state, body_statement)
            continue
        if (isinstance(body_statement, ast.Expr) and isinstance(body_statement.value, ast.Constant)
                and isinstance(body_statement.value.value, str)):
            # Intrinsic implementation (defaults to C++, as in the stable frontend)
            if intrinsic_code is not None:
                raise FrontendError('Cannot provide more than one intrinsic implementation for a tasklet',
                                    state.context.filename, body_statement)
            intrinsic_code = body_statement.value.value
            continue
        code_statements.append(body_statement)

    language = _language(statement, intrinsic_code, state)
    if intrinsic_code is not None:
        if prelude or epilogue:
            raise UnsupportedFeatureError('Indirect memlets are not supported with intrinsic tasklet code',
                                          state.context.filename,
                                          statement,
                                          category='indirect-memlet')
        code = intrinsic_code
    else:
        code_statements = [_restore_tasklet_callbacks(s, in_memlets, out_memlets, state) for s in code_statements]
        _validate_tasklet_body(code_statements, in_memlets, out_memlets, state)
        code = '\n'.join(prelude + [astutils.unparse(s) for s in code_statements] + epilogue)

    tasklet = nodes.Tasklet(statement.label,
                            set(in_memlets.keys()),
                            set(out_memlets.keys()),
                            code,
                            language=language,
                            code_global=statement.code_global,
                            code_init=statement.code_init,
                            code_exit=statement.code_exit)
    if statement.side_effects is not None:
        tasklet.side_effects = statement.side_effects
    state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _restore_tasklet_callbacks(code_statement: ast.stmt, in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet],
                               state: LoweringState) -> ast.stmt:
    """
    Rewrite calls to detected Python functions inside explicit tasklet code
    back into calls through the callback ABI.

    Preprocessing replaces a call to a plain Python function with the resolved
    object itself, embedded as a constant. Outside a tasklet the callback rules
    put the source form back; inside one the code goes straight to the code
    generator, where the object's ``repr`` is not valid syntax. Here the callee
    becomes a named callback symbol, which code generation emits as an ordinary
    call and the runtime binds to the original function.

    A tasklet's connectors carry the types the callback ABI needs, so only the
    directly typeable form is handled: ``<out connector> = f(<args>)``, whose
    arguments are input connectors or literals. Anything else raises rather
    than emitting an untyped callback, whose C signature would be wrong.
    """
    calls = [
        node for node in ast.walk(code_statement) if isinstance(node, ast.Call) and _callback_name(node.func, state)
    ]
    if not calls:
        return code_statement

    if not (isinstance(code_statement, ast.Assign) and len(code_statement.targets) == 1
            and isinstance(code_statement.targets[0], ast.Name) and code_statement.targets[0].id in out_memlets
            and isinstance(code_statement.value, ast.Call) and len(calls) == 1):
        raise UnsupportedFeatureError(
            'A Python callback inside a tasklet must be the whole right-hand side of an assignment to an output '
            f'connector ("b = f(a)"), so that its argument and return types can be read off the connectors. '
            f'Cannot lower "{astutils.unparse(code_statement).strip()}".',
            state.context.filename,
            code_statement,
            category='callback-in-tasklet')

    call = code_statement.value
    name = _callback_name(call.func, state)
    return_type = _connector_dtype(out_memlets[code_statement.targets[0].id], state)
    argument_types = []
    for argument in call.args:
        if isinstance(argument, ast.Name) and argument.id in in_memlets:
            argument_types.append(_connector_dtype(in_memlets[argument.id], state))
        elif isinstance(argument, ast.Constant) and isinstance(argument.value, (bool, int, float, complex)):
            argument_types.append(dtypes.dtype_to_typeclass(type(argument.value)))
        else:
            raise UnsupportedFeatureError(
                f'Argument "{astutils.unparse(argument)}" of the tasklet callback "{name}" is neither an input '
                'connector nor a literal, so its type cannot be determined.',
                state.context.filename,
                code_statement,
                category='callback-in-tasklet')
    if call.keywords:
        raise UnsupportedFeatureError(f'Tasklet callback "{name}" cannot be called with keyword arguments.',
                                      state.context.filename,
                                      code_statement,
                                      category='callback-in-tasklet')

    state.context.symbols[name] = dtypes.callback(return_type, *argument_types)
    state.emitter.root.callback_mapping.setdefault(name, name)

    restored = astutils.copy_tree(code_statement)
    restored.value.func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), call.func)
    return ast.fix_missing_locations(restored)


def _callback_name(callee: ast.expr, state: LoweringState) -> Optional[str]:
    """The callback symbol a resolved-object callee stands for, or None."""
    if not isinstance(callee, ast.Constant):
        return None
    qualname = getattr(callee, 'qualname', None)
    if qualname is None:
        return None
    # A callable detected in callee position carries the whole call expression
    # as its qualified name ("sq(a)"), not just the callee.
    try:
        expression = ast.parse(qualname, mode='eval').body
    except SyntaxError:
        return None
    if isinstance(expression, ast.Call):
        expression = expression.func
    if not isinstance(expression, ast.Name) or expression.id not in state.context.callback_callables:
        return None
    return expression.id


def _connector_dtype(memlet: Memlet, state: LoweringState) -> dtypes.typeclass:
    """The element type a tasklet connector carries."""
    return state.context.containers[memlet.data].dtype


def _lower_indirect_memlet(connector_expression: ast.expr, array_expression: ast.Subscript, reads: List[ast.expr],
                           in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet], prelude: List[str],
                           epilogue: List[str], is_input: bool, state: LoweringState, statement: ast.stmt) -> None:
    """
    Lower a tasklet memlet with a data-dependent index (``in_x << x[A_col[j]]``)
    as an indirection: each inner data read becomes a synthetic ``__ind<N>``
    input connector, the outer array becomes a full-array connector
    ``<conn>__arr``, and the actual element access moves into the tasklet code
    (``in_x = in_x__arr[__ind0]`` prepended for inputs, the mirrored store
    appended for outputs). The original connector name turns into a plain
    tasklet local. Write-conflict-resolution forms (``b(1, lambda ...)[...]``)
    cannot be indirected this way and fall back.
    """
    if not isinstance(connector_expression, ast.Name):
        raise UnsupportedFeatureError('Indirect memlets require a plain connector name',
                                      state.context.filename,
                                      statement,
                                      category='indirect-memlet')
    if not isinstance(array_expression.value, (ast.Name, ast.Attribute)):
        # e.g. a WCR/volume call form b(1, lambda a, b: ...)[...] — the
        # write-conflict semantics cannot move into the tasklet code.
        raise UnsupportedFeatureError('Indirect memlets do not support write-conflict or volume annotations',
                                      state.context.filename,
                                      statement,
                                      category='indirect-memlet')
    array_access = resolve_access(array_expression.value, state)
    if array_access is None:
        raise UnsupportedFeatureError(
            f'Indirect memlet references unknown container "{astutils.unparse(array_expression.value)}"',
            state.context.filename,
            statement,
            category='indirect-memlet')
    if isinstance(array_access.descriptor.dtype, dtypes.pyobject):
        raise UnsupportedFeatureError(
            f'Indirect memlet references interpreter-only container "{astutils.unparse(array_expression.value)}"',
            state.context.filename,
            statement,
            category='indirect-memlet')

    connector = connector_expression.id
    _check_connector(connector + '__arr', in_memlets, out_memlets, state, statement)

    # Synthetic affine input connectors for the inner reads
    index_names: Dict[str, str] = {}
    for read in reads:
        access = resolve_access(read, state)
        if isinstance(access.descriptor.dtype, dtypes.pyobject):
            raise UnsupportedFeatureError('Indirect memlet index references an interpreter-only container',
                                          state.context.filename,
                                          statement,
                                          category='indirect-memlet')
        synthetic = _fresh_connector('__ind', in_memlets, out_memlets)
        _check_connector(synthetic, in_memlets, out_memlets, state, statement)
        in_memlets[synthetic] = Memlet(data=access.container, subset=access.subset)
        index_names[astutils.unparse(read)] = synthetic

    # The element access moves into the tasklet code over a full-array connector
    index_code = astutils.unparse(substitute_index_reads(array_expression.slice, index_names))
    if is_input:
        in_memlets[connector + '__arr'] = Memlet(data=array_access.container, subset=array_access.subset)
        prelude.append(f'{connector} = {connector}__arr[{index_code}]')
    else:
        out_memlets[connector + '__arr'] = Memlet(data=array_access.container, subset=array_access.subset)
        epilogue.append(f'{connector}__arr[{index_code}] = {connector}')


def _fresh_connector(prefix: str, in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet]) -> str:
    index = 0
    while f'{prefix}{index}' in in_memlets or f'{prefix}{index}' in out_memlets:
        index += 1
    return f'{prefix}{index}'


def _parse_tasklet_memlet(memlet_expression: ast.expr, connector_expression: ast.expr, defined: Dict,
                          state: LoweringState, statement: ast.stmt):
    """
    Parse a tasklet memlet statement through the shared memlet parser. Parse
    failures become :class:`UnsupportedFeatureError` (falling back to a
    callback) rather than hard errors: the referenced name may be unavailable
    only because an earlier statement fell back (e.g. its producer became an
    interpreter callback), and genuinely malformed memlets surface when the
    callback replays the ``with dace.tasklet:`` block in the interpreter.
    """
    try:
        return parse_memlet(_shim(state), memlet_expression, connector_expression, defined)
    except UnsupportedFeatureError:
        raise
    except Exception as error:
        raise UnsupportedFeatureError(f'Cannot parse tasklet memlet: {error}',
                                      state.context.filename,
                                      statement,
                                      category='memlet-parse')


def _names_a_symbol(name: str, state: LoweringState) -> bool:
    """
    Whether a source-level name denotes a symbol rather than storage.

    A map parameter and a name rebound to a symbol carry a binding, but a
    PROGRAM symbol (the ``N`` of ``A: dace.int64[N]``) is registered straight
    into the symbol repository when the container's free symbols are collected
    and never gets one -- so the binding table alone misses exactly the case
    the tasklet rules need to reject.
    """
    binding = state.context.resolve(name)
    if binding is not None:
        return binding.kind == 'symbol'
    return name in state.context.symbols or name in state.context.symbol_aliases


def _reject_symbol_memlet_target(target: ast.expr, state: LoweringState, statement: ast.stmt) -> None:
    """
    Reject ``local >> N`` where ``N`` is a symbol rather than a container.

    A symbol is a compile-time quantity with no storage behind it, so there is
    nowhere for the memlet to write. The stable frontend refuses this in
    ``TaskletTransformer`` with the same wording.
    """
    name = target
    while isinstance(name, ast.Subscript):
        name = name.value
    if not isinstance(name, ast.Name):
        return
    if _names_a_symbol(name.id, state):
        raise DaceSyntaxError(None, statement, 'Symbolic variables cannot be used as memlet targets')


def _validate_tasklet_body(code_statements: List[ast.stmt], in_memlets: Dict[str, Memlet],
                           out_memlets: Dict[str, Memlet], state: LoweringState) -> None:
    """
    Reject a tasklet body that names a data container directly, or that assigns
    to a symbol.

    Inside an explicit tasklet every data access must arrive through a memlet
    (``local << A[i]``). A bare container name has no connector behind it, so
    the emitted tasklet reads a name nothing defines: the container becomes a
    free symbol of the tasklet and the generated code either fails to compile
    or reads a raw pointer where an element is expected. Assigning to a symbol
    is refused for the same reason :func:`_reject_symbol_memlet_target` refuses
    it as a memlet target -- a symbol has no storage.

    Reading a symbol is fine and stays legal: map parameters and program symbols
    are exactly what a tasklet body is expected to compute with.

    Both rules come from the stable frontend's ``TaskletTransformer``
    (``newast.py``), which nextgen otherwise ports in full.
    """
    connectors = set(in_memlets) | set(out_memlets)

    for statement in code_statements:
        for node in ast.walk(statement):
            targets: List[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if not isinstance(name, ast.Name) or name.id in connectors:
                        continue
                    if _names_a_symbol(name.id, state):
                        raise DaceSyntaxError(
                            None, statement, f'Cannot store into symbol "{name.id}" from a tasklet: a symbol is a '
                            'compile-time value and has no storage to write to')

    for statement in code_statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or node.id in connectors:
                continue
            binding = state.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                raise DaceSyntaxError(
                    None, statement, f'"{node.id}" is a data container and cannot be accessed directly in tasklet '
                    f'code: route it through a memlet ("{node.id}_local << {node.id}[...]") and use the '
                    'connector name instead')


def _memlet_binop(statement: ast.stmt) -> Optional[ast.BinOp]:
    """Return the shift binop of a memlet statement, or None."""
    if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.BinOp)
            and isinstance(statement.value.op, (ast.LShift, ast.RShift))):
        return statement.value
    return None


def _check_connector(connector: Optional[str], in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet],
                     state: LoweringState, statement: ast.stmt) -> None:
    if connector is None:
        raise UnsupportedFeatureError('Memlet statements require a local connector name',
                                      state.context.filename,
                                      statement,
                                      category='memlet-parse')
    if connector in in_memlets or connector in out_memlets:
        raise FrontendError(f'Local variable "{connector}" is already a tasklet input or output',
                            state.context.filename, statement)


def _to_repository(memlet: Memlet, state: LoweringState, statement: ast.stmt) -> Memlet:
    """Rewrite a parsed memlet to reference the repository container name."""
    binding = state.context.resolve(memlet.data)
    if binding is None or binding.kind != 'container':
        raise UnsupportedFeatureError(f'Tasklet memlet references unknown container "{memlet.data}"',
                                      state.context.filename,
                                      statement,
                                      category='memlet-parse')
    if isinstance(state.context.containers[binding.container].dtype, dtypes.pyobject):
        # The producer of this name fell back to the interpreter; its typed
        # form is unavailable, so the tasklet must replay there too.
        raise UnsupportedFeatureError(f'Tasklet memlet references interpreter-only container "{memlet.data}"',
                                      state.context.filename,
                                      statement,
                                      category='pyobject-propagation')
    memlet.data = binding.container
    return memlet


@rule(ExplicitConsume)
def lower_explicit_consume(statement: ExplicitConsume, state: LoweringState) -> None:
    """
    Lower an explicit consume scope to a :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ConsumeScope`
    with a real :class:`~dace.sdfg.nodes.ConsumeEntry`. The popped stream
    element enters the body as a dynamic (volume ``-1``) read of the stream:
    directly as a tasklet input connector for the ``@dace.consume`` (tasklet)
    form, or through a scalar element container written by a leading pop
    tasklet for the ``@dace.consumescope`` (statement-body) form. The
    processing-element index binds as a symbol.

    NOTE: ``tree_to_sdfg`` does not lower ``ConsumeScope`` yet — consume
    programs build correct schedule trees but cannot convert to SDFGs.
    """
    stream_access = resolve_access(statement.stream, state) if statement.stream is not None else None
    if stream_access is None or not isinstance(stream_access.descriptor, data.Stream):
        raise UnsupportedFeatureError(
            f'Consume scope requires a stream input (got "{astutils.unparse(statement.stream)}")',
            state.context.filename,
            statement,
            category='explicit-consume')
    try:
        num_pes = symbolic.pystr_to_symbolic(statement.num_pes_src)
    except Exception:
        raise UnsupportedFeatureError(f'Cannot parse consume processing-element count "{statement.num_pes_src}"',
                                      state.context.filename,
                                      statement,
                                      category='explicit-consume')
    chunksize = state.inference.constant_int(ast.parse(statement.chunksize_src, mode='eval').body) or 1
    condition = CodeBlock(statement.condition_src) if statement.condition_src is not None else None

    state.context.bind_symbol(statement.pe_index)
    consume_node = nodes.Consume(statement.label, (statement.pe_index, num_pes), condition, chunksize=chunksize)
    element_memlet = Memlet(data=stream_access.container, subset=subsets.Range([(0, 0, 1)]))
    element_memlet.dynamic = True
    element_memlet.volume = -1

    with state.emitter.scope(tn.ConsumeScope(node=nodes.ConsumeEntry(consume_node), children=[])):
        if statement.scope_body:
            # Statement body: the popped element materializes in a scalar
            # container written by a leading pop tasklet.
            element_descriptor = data.Scalar(stream_access.descriptor.dtype)
            element_container = state.context.add_container(statement.element, element_descriptor)
            state.context.bind(statement.element, element_container)
            pop = nodes.Tasklet(f'{statement.label}_pop', {'__stream'}, {'__out'}, '__out = __stream')
            state.emitter.emit(
                tn.TaskletNode(node=pop,
                               in_memlets={'__stream': copy.deepcopy(element_memlet)},
                               out_memlets={'__out': Memlet(data=element_container, subset='0')}))
            state.lower_body(statement.statements)
        else:
            # Tasklet body: the element is a direct tasklet input connector.
            tasklet = ExplicitTasklet(label=statement.label, statements=statement.statements, location=statement)
            lower_explicit_tasklet(tasklet, state, extra_inputs={statement.element: element_memlet})


def _language(statement: ExplicitTasklet, intrinsic_code: Optional[str], state: LoweringState) -> dtypes.Language:
    if statement.language is not None:
        try:
            return dtypes.Language[statement.language]
        except KeyError:
            raise FrontendError(f'Unknown tasklet language "{statement.language}"', state.context.filename, statement)
    return dtypes.Language.CPP if intrinsic_code is not None else dtypes.Language.Python
