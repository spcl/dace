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

The same memlet syntax written *outside* a tasklet is a copy between two
containers (``ostream >> out``), recognized as an
:class:`~dace.frontend.python.nextgen.canonical.cpa.ExplicitMemlet` marker and
lowered here too, since it shares the memlet-expression parsing.

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
from dace.frontend.python.memlet_parser import ParseMemlet, parse_memlet
from dace.frontend.python.nextgen.canonical.cpa import ExplicitConsume, ExplicitMemlet, ExplicitTasklet
from dace.frontend.python.nextgen.common import FrontendError, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import (indirect_index_reads, nondegenerate_shape,
                                                          promote_index_reads, resolve_access, substitute_index_reads)
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
    memlet.data = _repository_container(memlet.data, state, statement, 'Tasklet memlet')
    return memlet


def _repository_container(name: str, state: LoweringState, statement: ast.stmt, subject: str) -> str:
    """The repository container name a parsed memlet's data name is bound to."""
    binding = state.context.resolve(name)
    if binding is None or binding.kind != 'container':
        raise UnsupportedFeatureError(f'{subject} references unknown container "{name}"',
                                      state.context.filename,
                                      statement,
                                      category='memlet-parse')
    if isinstance(state.context.containers[binding.container].dtype, dtypes.pyobject):
        # The producer of this name fell back to the interpreter; its typed
        # form is unavailable, so the statement must replay there too.
        raise UnsupportedFeatureError(f'{subject} references interpreter-only container "{name}"',
                                      state.context.filename,
                                      statement,
                                      category='pyobject-propagation')
    return binding.container


@rule(ExplicitMemlet)
def lower_explicit_memlet(statement: ExplicitMemlet, state: LoweringState) -> None:
    """
    Lower a program-level memlet expression (``ostream >> out``) to a
    :class:`~dace.sdfg.analysis.schedule_tree.treenodes.CopyNode` between the
    two containers, porting ``newast.ProgramVisitor.visit_TopLevelExpr``.

    A copy node is anchored at its *source*: ``memlet.data`` names what is
    read and ``target`` what is written, so both sides' annotations meet on
    one memlet. The volume comes from whichever side declared one — a dynamic
    write (``A >> ostream(-1)``) annotates the destination — and the
    destination's subset is carried as ``other_subset`` only when the two
    sides move the same shape. Copying a stream into an array is the
    exception the syntax exists for: the stream is a single element with a
    buffer behind it, so the sides' shapes differ by design and the
    destination subset stays implicit (the whole target), which is what the
    stream-pop code generator expects.
    """
    defined = state.context.defined_view()
    source = _parse_program_memlet(statement.source, defined, state)
    destination = _parse_program_memlet(statement.destination, defined, state)
    if source.arrdims or destination.arrdims:
        raise UnsupportedFeatureError('Memlet expressions cannot copy with array indices',
                                      state.context.filename,
                                      statement,
                                      category='memlet-parse')
    source_container = _repository_container(source.name, state, statement, 'Memlet expression')
    target_container = _repository_container(destination.name, state, statement, 'Memlet expression')

    source_subset = _as_range(source.subset)
    target_subset = _as_range(destination.subset)
    accesses = destination.accesses if _has_volume_annotation(statement.destination) else source.accesses
    memlet = Memlet.simple(source_container, source_subset, num_accesses=accesses, wcr_str=destination.wcr)
    if nondegenerate_shape(source_subset) == nondegenerate_shape(target_subset):
        memlet.other_subset = target_subset
    state.emitter.emit(tn.CopyNode(target=target_container, memlet=memlet))


def _parse_program_memlet(expression: ast.expr, defined: Dict, state: LoweringState):
    """
    Parse one side of a program-level memlet expression. As with tasklet
    memlets, a parse failure falls back to the interpreter rather than
    erroring: the name may be missing only because its producer did.
    """
    try:
        return ParseMemlet(_shim(state), defined, expression)
    except UnsupportedFeatureError:
        raise
    except Exception as error:
        raise UnsupportedFeatureError(f'Cannot parse memlet expression: {error}',
                                      state.context.filename,
                                      expression,
                                      category='memlet-parse')


def _as_range(subset: subsets.Subset) -> subsets.Range:
    """A parsed subset as a range (an index form covers one element of it)."""
    return subsets.Range.from_indices(subset) if isinstance(subset, subsets.Indices) else subset


def _has_volume_annotation(expression: ast.expr) -> bool:
    """Whether a memlet expression declares its own volume/write-conflict
    resolution: ``B(-1)``, ``B(1, lambda x, y: x + y)[i]``."""
    if isinstance(expression, ast.Subscript):
        expression = expression.value
    return isinstance(expression, ast.Call)


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
