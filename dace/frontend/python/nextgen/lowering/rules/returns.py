# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical ``return`` statements.

Return values are materialized into the conventional non-transient
``__return`` containers (``__return_<index>`` for tuples), followed by an
explicit :class:`ReturnNode` naming them.

Inside an inlined nested ``@dace.program``
(:attr:`ProgramContext.return_prefix` is non-empty), return containers are
prefixed transients instead and their names are recorded in
:attr:`ProgramContext.return_names` for the call rule to bind. The
:class:`ReturnNode` is emitted uniformly; inside a
:class:`FunctionCallScope` it means "exit the inlined callee", and the call
rule strips a trailing one (a tail return falls off the scope end).
"""
import ast
from typing import List

from dace import data, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.nextgen.common import FrontendError, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering import dispatch
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(ast.Return)
def lower_return(statement: ast.Return, state: LoweringState) -> None:
    prefix = state.context.return_prefix
    if statement.value is None:
        state.emitter.emit(tn.ReturnNode())
        return

    # Indexed ``__return_<i>`` containers are what tells the calling convention
    # to hand back a tuple, so the choice follows the SYNTAX of the returned
    # expression rather than how many values it has: ``return (x, )`` returns a
    # one-element tuple in Python, and naming its single value ``__return``
    # would flatten it to a bare array.
    is_tuple = isinstance(statement.value, (ast.Tuple, ast.List))
    values = statement.value.elts if is_tuple else [statement.value]
    names: List[str] = []
    for index, value in enumerate(values):
        base_name = f'__return_{index}' if is_tuple else '__return'
        names.append(_materialize_return_value(f'{prefix}{base_name}', value, statement, state))
    state.context.return_names.extend(names)
    state.emitter.emit(tn.ReturnNode(values=names))


def _reject_deferred_size(shape, value: ast.expr, statement: ast.Return, state: LoweringState) -> None:
    """
    Refuse to return a container whose size is only known while the program
    runs -- the result of a boolean-mask gather (``B = A[mask]``), or a shape
    the program computed from data (``np.zeros((counts[0] + 1, ))``, whose size
    is read back into a symbol by ``dispatch._promote_scalar_arguments``).

    A compiled SDFG's calling convention needs every argument's memory --
    including the return value's -- to exist before the call, so a size
    computed mid-call cannot cross the boundary however it is produced. That is
    a property of the boundary, not of the producer: the same container is
    fully usable inside the program (``np.sum(A[mask])`` lowers and runs).

    Raised as a hard :class:`FrontendError` rather than an
    :class:`UnsupportedFeatureError`, deliberately: the usual degradation to a
    Python callback cannot help here either, since the callback would still
    have to write a ``__return`` of a size the caller could not allocate. What
    this replaces is a ``TypeError`` about an unevaluatable shape, raised from
    the runtime argument marshalling long after any connection to the source
    was lost. Supporting the form needs a count-then-fill two-kernel dispatch,
    which is future work.

    Both halves of the message are stated in the user's terms: the returned
    expression through :meth:`ProgramContext.describe_expression` (after
    A-normal form the node here is usually a generated ``__anf`` temporary),
    and the size through the description recorded with the deferred symbol
    (its generated name says nothing to a reader).
    """
    deferred = sorted({
        symbol.name
        for size in shape if hasattr(size, 'free_symbols') for symbol in size.free_symbols
        if symbol.name in state.context.deferred_symbols
    })
    if not deferred:
        return
    sizes = list(dict.fromkeys(state.context.deferred_symbols[name] or name for name in deferred))
    raise FrontendError(
        f'Cannot return "{state.context.describe_expression(value)}": data-dependent-shaped return values are not '
        f'supported. Its size ({", ".join(sizes)}) is only known while the program runs, and a compiled program\'s '
        'caller must allocate the return value before the call. Reduce it inside the program '
        '(``return np.sum(A[mask])``), or write into a caller-allocated output.', state.context.filename, statement)


def _reject_disagreeing_shape(return_name: str, shape, value: ast.expr, statement: ast.Return, state: LoweringState):
    """
    Refuse a second ``return`` that hands back a different shape from the first
    (``if c: return <20 elements>`` against ``else: return <30 elements>``).

    A compiled program has ONE return container, and its caller allocates the
    memory before the call -- so a size chosen at run time has nowhere to live.
    Without this the later ``return`` simply copied its own extent into the
    container the first one sized, writing past the end of it.

    Raised as :class:`~dace.frontend.python.common.DaceSyntaxError`, the class
    the stable frontend raises for the same program and the one callers already
    catch, rather than this frontend's own hierarchy.
    """
    existing = state.context.containers.get(return_name)
    if existing is None or list(existing.shape) == list(shape):
        return
    raise DaceSyntaxError(
        None, statement, f'Return value "{state.context.describe_expression(value)}" has shape {tuple(shape)}, but an '
        f'earlier return in the same program has shape {tuple(existing.shape)}. A compiled program returns one '
        'container, whose memory its caller allocates before the call, so every return must agree on the shape.')


def _materialize_return_value(return_name: str, value: ast.expr, statement: ast.Return, state: LoweringState) -> str:
    # Compile-time Python sequences materialize as constant containers first
    if isinstance(value, ast.Name):
        sequence = state.context.static_value_of(value.id)
        if sequence is not None:
            from dace.frontend.python.nextgen.lowering.mechanisms import static_values
            access = static_values.materialize(sequence, state)
            value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), value)

    # Top-level returns are non-transient program outputs; inlined-callee
    # returns are internal temporaries.
    transient = bool(state.context.return_prefix)

    access = None
    if isinstance(value, (ast.Name, ast.Subscript)):
        access = resolve_access(value, state)
    elif isinstance(value, ast.Attribute):
        # Registry ATTRIBUTE-family reads (``return A.T``/``.real``/
        # ``.imag``/``.flat``): ``resolve_access`` only resolves structure
        # members for a bare attribute, so materialize through the same
        # dedicated path ``rules.assign.lower_attribute_assign`` uses.
        base = resolve_access(value.value, state)
        if base is not None:
            access = dispatch.resolve_attribute_data(base, value.attr, state)
    if access is not None:
        # The returned container takes the access's NumPy result shape, so a
        # size-1 dimension the subscript asked for survives (``return A[0:1]``
        # hands back shape (1, N), as NumPy does) and a ``newaxis`` is not
        # dropped on the way out.
        shape = access.numpy_shape or [1]
        if not transient:
            _reject_deferred_size(shape, value, statement, state)
        _reject_disagreeing_shape(return_name, shape, value, statement, state)
        if return_name not in state.context.containers:
            source = access.descriptor
            if isinstance(source, data.Array) and list(source.shape) == list(shape):
                # A whole-array return keeps the source's LAYOUT: ``return A``
                # hands back A itself in Python, strides included, so a
                # default-layout copy would change what the caller observes.
                descriptor = data.Array(source.dtype, shape, strides=list(source.strides))
            else:
                descriptor = data.Array(access.descriptor.dtype, shape)
            return_name = state.context.add_container(return_name, descriptor, transient=transient)
        descriptor = state.context.containers[return_name]
        state.emitter.emit(
            tn.CopyNode(target=return_name,
                        memlet=Memlet(data=access.container,
                                      subset=access.subset,
                                      other_subset=subsets.Range.from_array(descriptor))))
        return return_name

    inferred = state.inference.infer(value)
    if inferred.is_data:
        # A DATA-valued expression that is not a plain access: ``return -A``,
        # since ``UnaryOp(op, atom)`` is a canonical atom and nothing hoisted
        # it out of the return. The scalar tasklet below would type the result
        # (1,) whatever the operand's shape and leave ``A`` in the code as a
        # bare name -- a raw pointer where the tasklet expects an element.
        # Compute it into the return container instead, which is exactly what
        # the equivalent ``C = -A; return C`` already does.
        source = inferred.descriptor
        shape = list(source.shape) if isinstance(source, data.Array) else [1]
        if not transient:
            _reject_deferred_size(shape, value, statement, state)
        _reject_disagreeing_shape(return_name, shape, value, statement, state)
        if return_name not in state.context.containers:
            dtype = state.context.return_dtype or source.dtype
            return_name = state.context.add_container(return_name, data.Array(dtype, shape), transient=transient)
        descriptor = state.context.containers[return_name]
        dispatch.lower_computation(DataAccess(return_name, subsets.Range.from_array(descriptor), descriptor), value,
                                   statement, state)
        return return_name

    dtype = state.inference.dtype_of(inferred)
    if dtype is None:
        raise UnsupportedFeatureError(f'Cannot determine return value type: {state.context.describe_expression(value)}',
                                      state.context.filename,
                                      statement,
                                      category='type-inference')
    _reject_disagreeing_shape(return_name, [1], value, statement, state)
    if return_name not in state.context.containers:
        # An inlined callee's ``-> dtype`` annotation names the type it hands
        # back, over the one inferred from the returned expression: ``return 5``
        # under ``-> dace.int32`` is an int32, not the literal's own type. The
        # tasklet below performs the conversion.
        return_name = state.context.add_container(return_name,
                                                  data.Array(state.context.return_dtype or dtype, [1]),
                                                  transient=transient)
    # A name that reaches here resolved to no container (``resolve_access``
    # declined above), so unparsing it would leave the tasklet reading a name
    # nothing defines. That is what canonicalization hands us for a hoisted
    # return: ``return len(A)`` becomes ``__anf0 = len(A); return __anf0``, and
    # the temporary binds to a compile-time value rather than to storage. The
    # dangling name survives as a free symbol of the finished SDFG, where
    # ``SDFG.arglist`` looks it up in ``symbols`` and raises ``KeyError``.
    # Emit the value the name stands for instead.
    expression = astutils.unparse(value)
    if isinstance(value, ast.Name) and inferred.kind in ('symbolic', 'constant') and inferred.value is not None:
        expression = str(inferred.value)
    tasklet = nodes.Tasklet(f'return_{statement.lineno}', set(), {'__out'}, f'__out = {expression}')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={},
                       out_memlets={'__out': Memlet(data=return_name, subset=subsets.Range([(0, 0, 1)]))}))
    return return_name
