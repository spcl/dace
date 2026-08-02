# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The type-directed dispatch seam between syntax rules and lowering mechanisms.

Syntax rules (one per canonical statement form) delegate computations here;
this module inspects *operand types* and routes to the appropriate mechanism:

- any pyobject operand: the computation must run in the interpreter
  (callback mechanism),
- static Python sequences mixed with data: materialize, then elementwise,
- data/symbolic/constant operands: elementwise map or tasklet.

Semantic feature gaps (:class:`UnsupportedFeatureError`) also fall back to the
callback mechanism, preserving totality: no user program fails to lower merely
because a construct has no dedicated mechanism yet. Future registry entries
(NumPy functions, library nodes, user dunders) plug in here rather than as
separate per-library rule sets.

Callback provenance: every fallback reason carries a stable kebab-case
``[category]`` prefix (set at the raise site via
``UnsupportedFeatureError(category=...)`` or at the fallback call site), so
discrepancy checks and gap reports can aggregate interpreter fallbacks by
cause. The taxonomy in use:

- ``detected-callback`` — the callee was already wrapped as a Python callback
  by preprocessing (intended interpreter work, mirrored in the classic
  frontend's ``callback_mapping``),
- ``unknown-call:<qualname>`` — a call with no lowering (the
  missing-replacements worklist),
- ``opaque-syntax:<stmt-type>`` — statement outside the CPA subset (marked
  during canonicalization),
- ``pyobject-propagation`` — a consumed operand/callee is an opaque Python
  object produced by an earlier callback,
- ``inline-fallback:<subreason>`` — a nested ``@dace.program`` call that
  could not be inlined,
- ``memlet-parse`` / ``indirect-memlet`` — explicit-tasklet memlet gaps,
- ``data-dependent-subscript``, ``dynamic-bound``, ``broadcast``,
  ``explicit-map``, ``explicit-consume``, ``join-merge``, ``loop-stability``,
  ``type-inference``, ``undefined-name``, ``static-sequence``, ``ufunc``,
  ``array-creation``, ``reduction``, ``reference-set``, ``structure-member``,
  ``assign-target``, ``reshape`` — per-feature semantic gaps,
- ``safety-net`` — an uncategorized error reaching the totality net in
  ``registry.lower_statement`` (highest bug suspicion),
- ``uncategorized`` — a fallback site with no assigned category yet.
"""
import ast
import copy
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.frontend.python import astutils
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt, statement_io_sets
from dace.frontend.python.nextgen.common import (UnsupportedFeatureError, normalize_qualname, registry_argument_value,
                                                 supported_data_attribute)
from dace.frontend.python.nextgen.lowering.access import (DataAccess, nondegenerate_shape, resolve_access,
                                                          scalar_read_expression)
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.lowering.mechanisms import creation, elementwise, reduction, static_values, streams
from dace.frontend.python.nextgen.semantics.indexing import array_index_slots, index_slots, substitute_slots
from dace.frontend.python.nextgen.semantics.values import StaticSequence

#: Full-reduction calls by registry-qualified name, mapped to their WCR ufunc.
_REDUCTION_CALLS = {
    'numpy.sum': 'add',
    'numpy.prod': 'multiply',
    'numpy.max': 'maximum',
    'numpy.amax': 'maximum',
    'numpy.min': 'minimum',
    'numpy.amin': 'minimum',
}

#: Array-method reductions (``a.sum()``), mapped to their WCR ufunc.
_REDUCTION_METHODS = {
    'sum': 'add',
    'prod': 'multiply',
    'max': 'maximum',
    'min': 'minimum',
}

#: Builtin functions over SCALAR operands that are elementwise ufuncs, mapped
#: to the ufunc implementing them, with the argument count that form takes.
_BUILTIN_ELEMENTWISE = {'min': 'minimum', 'max': 'maximum', 'abs': 'absolute'}
_BUILTIN_ELEMENTWISE_ARITY = {'min': 2, 'max': 2, 'abs': 1}


def _replacement_registered(name: str, require_descriptor_inference: bool = True) -> bool:
    """
    Whether a call is eligible for deferred replacement expansion
    (``tree_to_sdfg.visit_ReplacementCallNode``): it must have BOTH a
    registered replacement (the expansion body) and a descriptor inference
    entry (the frontend must type the result to allocate the target
    container). Replacements needing ``ProgramVisitor`` machinery beyond the
    expansion shim's surface fail loudly at expansion time.

    :param require_descriptor_inference: Whether an inference entry is
        required. Only DEFERRED expansion needs one, to allocate the result
        container before the replacement runs; the frontend view path
        (:func:`_lower_view_call`) takes its descriptor from the trial run
        itself and passes False.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get(name) is None:
        return False
    return not require_descriptor_inference or oprepo.Replacements.get_descriptor_inference(name) is not None


@dataclass
class _HandleProducer:
    """
    The replacement call that produced an opaque handle container (see
    :attr:`ProgramContext.replacement_handles`), recorded so a later trial
    expansion consuming the handle can replay it into its scratch SDFG —
    the handle's meaning lives in registry state the producer installs
    (``sdfg.subarrays``), which a copied descriptor does not carry.
    """
    target: str
    name: str
    arguments: List[Any]
    keyword_arguments: Dict[str, Any]
    data_arguments: Set[str]
    receiver: Optional[str]
    receiver_object: Any
    #: Creation order, so a replay runs producers in the order the program did.
    order: int


def _method_registered(receiver: Any, method_name: str, require_descriptor_inference: bool = True) -> bool:
    """
    Whether a bound-method call is eligible for deferred replacement
    expansion, mirroring :func:`_replacement_registered` for the ``_method_rep``
    keyspace (:meth:`Replacements.get_method` /
    :meth:`Replacements.get_method_descriptor_inference`).

    :param receiver: The receiver the method is called on — a data descriptor,
        or the compile-time object itself for an object receiver. Either way
        the registry is keyed on its CLASS.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get_method(type(receiver), method_name) is None:
        return False
    return (not require_descriptor_inference
            or oprepo.Replacements.get_method_descriptor_inference(type(receiver), method_name) is not None)


def lower_computation(target: DataAccess,
                      value: ast.expr,
                      statement: ast.stmt,
                      state: LoweringState,
                      wcr: Optional[str] = None) -> None:
    """
    Lower a canonical flat expression into a target access, dispatching on
    operand types.

    :param wcr: Conflict-resolution lambda for the write, when the statement is
                an accumulation inside a dataflow scope.
    """
    try:
        value = static_values.fold_static_subscripts(value, state)
        if _consumes_pyobject(value, state):
            fallback_to_callback(statement,
                                 state,
                                 'operates on an opaque Python object',
                                 category='pyobject-propagation')
            return
        # NumPy advanced indexing gathers through its own map: the result index
        # space comes from the broadcast index arrays, not from the subset, so
        # the elementwise mechanism cannot express it.
        if _lower_advanced_index(target, value, statement, state):
            return
        # Before dispatching on the operator: a registry implementation resolves
        # its operands as plain accesses, and an array-valued index is not one
        # (``vals @ x[cols]`` in spmv). Gathering first leaves it a container.
        value = _materialize_advanced_indices(value, statement, state)
        # Operators the elementwise mechanism cannot express (``@``, whose
        # result is a contraction rather than a broadcast) go to the registry
        # implementation that builds the real dataflow.
        if _lower_registry_operator(target, value, statement, state):
            return
        value = _materialize_attribute_reads(value, state)
        rewritten = static_values.materialize_operands(value, state)
        elementwise.emit_computation(target, rewritten, statement, state, wcr=wcr)
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, reason)


def _materialize_attribute_reads(value: ast.expr, state: LoweringState) -> ast.expr:
    """
    Replace registry ATTRIBUTE-family reads (``A.T``/``.real``/``.imag``/
    ``.flat``) inside an expression with the container they produce.

    ``rules.assign.lower_attribute_assign`` covers the one position where such
    a read is a whole right-hand side bound to a NAME. Anywhere else
    (``c[:, :] = a.T``, ``c[:] = a.T + b``) the read reaches the elementwise
    mechanism, which has no resolution for an attribute and substitutes it
    verbatim into the generated tasklet (``__out = __in0.T``) -- a wrong
    program, and one that only fails at C++ compile time. Materializing here
    reduces those forms to the ``__attr0 = a.T; c[:, :] = __attr0`` shape that
    already lowers.

    :raises UnsupportedFeatureError: Inside a dataflow scope, where the
        materialization (a deferred replacement call, or a view binding) is
        not a legal node -- the statement degrades to a callback instead.
    """
    reads = [
        node for node in ast.walk(value) if isinstance(node, ast.Attribute)
        and isinstance(getattr(node, 'ctx', ast.Load()), ast.Load) and supported_data_attribute(node.attr)
    ]
    if not reads:
        return value
    materialized: dict = {}
    for read in reads:
        base = resolve_access(read.value, state)
        if base is None:
            continue
        if state.emitter.in_dataflow_scope:
            raise UnsupportedFeatureError(f'Attribute "{astutils.unparse(read)}" inside a dataflow scope',
                                          state.context.filename,
                                          read,
                                          category='attribute')
        access = resolve_attribute_data(base, read.attr, state)
        if access is not None:
            materialized[ast.dump(read)] = access.container
    if not materialized:
        return value

    class _Substituter(ast.NodeTransformer):

        def visit_Attribute(self, attribute_node: ast.Attribute) -> ast.AST:
            container = materialized.get(ast.dump(attribute_node))
            if container is None:
                return self.generic_visit(attribute_node)
            return ast.copy_location(ast.Name(id=container, ctx=ast.Load()), attribute_node)

    return ast.fix_missing_locations(_Substituter().visit(astutils.copy_tree(value)))


def _lower_registry_operator(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> bool:
    """
    Emit a deferred :class:`~...treenodes.ReplacementCallNode` for an operator
    the replacement registry OVERRIDES for these operand types.

    Python operators are dunder methods, and the registry is where DaCe records
    what one means for a given pair of operand classes. An implementation that
    is not the stock elementwise one (see
    ``op_repository.ELEMENTWISE_OPERATOR_ATTRIBUTE``) may do anything —
    contract (``@``), move storage (``A @ StorageType.GPU_Global``), reduce —
    so the elementwise mechanism must not touch it. Lowering ``@``
    elementwise, which is what happened before this check existed, reads both
    operands at the RESULT's index space: for ``(24, 12) @ (12, 48)`` that is
    out of bounds on both, silently.

    Returns False when the registry does not override the operator here, so the
    caller falls through to the ordinary elementwise path.

    :raises UnsupportedFeatureError: If the operator must be lowered here but
        cannot be — the caller turns that into a callback, correct if slow.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
    from dace.frontend.python.nextgen.semantics.inference import (operator_lookup_arguments, registry_operator_operands)

    if not isinstance(value, (ast.BinOp, ast.UnaryOp)):
        return False
    expressions = [value.left, value.right] if isinstance(value, ast.BinOp) else [value.operand]
    try:
        inferred_operands = [state.inference.infer(expression) for expression in expressions]
    except UnsupportedFeatureError:
        return False
    resolved = registry_operator_operands(value, inferred_operands)
    if resolved is None:
        return False
    optype, values = resolved
    implementation = oprepo.Replacements.getop(*operator_lookup_arguments(optype, values))
    if implementation is None or oprepo.is_elementwise_operator(implementation):
        return False

    # From here the registry owns the operator: anything that stops us is a
    # callback, never a fall-through to the elementwise mechanism.
    arguments: List[Any] = []
    data_arguments = set()
    for expression, operand in zip(expressions, inferred_operands):
        if not operand.is_data:
            arguments.append(operand.value)
            continue
        access = _whole_container_operand(expression, statement, state)
        if access is None:
            raise UnsupportedFeatureError(f'Operand of "{optype}" does not resolve to a container',
                                          state.context.filename,
                                          statement,
                                          category='registry-operator')
        arguments.append(access.container)
        data_arguments.add(access.container)
    qualname = oprepo.operator_qualname(optype)
    if not _expansion_viable(qualname, arguments, {}, data_arguments, state):
        raise UnsupportedFeatureError(f'Deferred expansion of "{optype}" is not viable here',
                                      state.context.filename,
                                      statement,
                                      category='registry-operator')
    target_container, copy_out = _replacement_write_target(target, state.inference.infer(value), state)
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=qualname,
                               target=target_container,
                               arguments=arguments,
                               keyword_arguments={},
                               data_arguments=data_arguments))
    if copy_out:
        _emit_replacement_copy_out(target_container, target, statement, state)
    return True


def _operator_lookup(optype: str, arguments: List[Any], data_arguments: set, containers) -> Tuple[Any, ...]:
    """
    The :meth:`Replacements.getop` arguments for a deferred OPERATOR-family
    call, resolving each recorded argument back to its operand: a container
    name to that container's descriptor, anything else to itself.

    Shared by the viability trial here and the real expansion in
    ``tree_to_sdfg``, so the two look the same implementation up.
    """
    from dace.frontend.python.nextgen.semantics.inference import operator_lookup_arguments
    values = [containers[argument] if argument in data_arguments else argument for argument in arguments]
    return operator_lookup_arguments(optype, values)


def _whole_container_operand(operand: ast.expr, statement: ast.stmt, state: LoweringState) -> Optional[DataAccess]:
    """
    Resolve one operand of a registry operator to a WHOLE container, which is
    the only thing the registry implementations accept (they take container
    names and read ``sdfg.arrays[name]``).

    Four forms get there: a plain name, a registry attribute that produces
    data (``a.T``), a partial access (``A[i]``, a view), which is copied
    into a temporary first — passing its base container instead would silently
    operate on all of ``A`` — and a compound data-valued expression
    (``-tmp[i] @ B``), which is computed into a temporary.

    :return: The access, or None when the operand is not data at all.
    """
    access = None
    if isinstance(operand, (ast.Name, ast.Attribute, ast.Subscript)):
        access = resolve_access(operand, state)
    if access is None and isinstance(operand, ast.Attribute):
        base = resolve_access(operand.value, state)
        if base is not None:
            access = resolve_attribute_data(base, operand.attr, state)
    if access is None:
        return _materialize_data_operand(operand, statement, state)
    descriptor = state.context.containers.get(access.container)
    if descriptor is not None and str(access.subset) == str(
            subsets.Range.from_array(descriptor)) and not isinstance(descriptor, data.View):
        return access
    # A partial access or a view: stage it in a whole container of its own.
    staged_descriptor = data.Array(access.descriptor.dtype, nondegenerate_shape(access.subset) or [1])
    staged_descriptor.transient = True
    staged = state.context.add_container('__operand', staged_descriptor)
    state.emitter.emit(
        tn.CopyNode(target=staged,
                    memlet=Memlet(data=access.container,
                                  subset=access.subset,
                                  other_subset=subsets.Range.from_array(staged_descriptor))))
    return DataAccess(staged, subsets.Range.from_array(staged_descriptor), staged_descriptor)


def _materialize_data_operand(operand: ast.expr, statement: ast.stmt, state: LoweringState) -> Optional[DataAccess]:
    """
    Compute a compound data-valued operand of a registry operator (``-tmp[i]``
    in ``-tmp[i] @ A.upper[i]``) into a whole container of its own.

    Canonicalization only hoists what it must, so an operand can still be an
    expression rather than an access; the registry implementations take
    container NAMES, so an expression has to become one here.

    :return: The access holding the computed operand, or None when the operand
             is not data-valued at all.
    """
    inferred = state.inference.infer(operand)
    if not inferred.is_data:
        return None
    descriptor = copy.deepcopy(inferred.descriptor)
    container = state.context.add_container('__operand', descriptor)
    access = DataAccess(container, subsets.Range.from_array(descriptor), descriptor)
    # Mirrors the tail of :func:`lower_computation` (minus its callback
    # fallback, which belongs to the whole statement): a nested registry
    # operator (``-(a @ b)``) lowers through the registry too, and recursion
    # terminates because each operand is strictly smaller.
    if not _lower_registry_operator(access, operand, statement, state):
        rewritten = _materialize_attribute_reads(operand, state)
        rewritten = static_values.materialize_operands(rewritten, state)
        elementwise.emit_computation(access, rewritten, statement, state)
    return access


def _lower_advanced_index(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> bool:
    """
    Emit a bare advanced-indexing read (``b = A[indices]``) straight into the
    target as a gather. Returns False when the value is not such an access, so
    the caller falls through to the ordinary computation paths.

    Nested occurrences (``A[indices] + B``) are handled by
    :func:`_materialize_advanced_indices` instead, which gathers each one into a
    temporary; only the top-level form can write the target directly.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing
    access = _advanced_index_access(value, state)
    if access is None:
        return False
    advanced_indexing.emit_gather(target, access, statement, state)
    return True


def _advanced_index_access(value: ast.expr, state: LoweringState):
    """The resolved advanced-indexing access an expression performs, or None if
    it is not an array-valued subscript of a container."""
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing
    if not isinstance(value, ast.Subscript) or not isinstance(value.value, (ast.Name, ast.Attribute)):
        return None
    base = resolve_access(value.value, state)
    if base is None:
        return None
    value = materialize_array_indices(value, state)
    expr = state.inference.parse_access(value)
    if not expr.arrdims:
        return None
    return advanced_indexing.analyze(value, expr, base.container, base.descriptor, state.context, state.inference)


def materialize_array_indices(node: ast.Subscript, state: LoweringState) -> ast.Subscript:
    """
    Rewrite array-valued index EXPRESSIONS into containers bound to plain
    names, returning the rewritten subscript.

    The shared memlet parser recognizes an advanced (array-valued) index only
    when it is written as a bare name whose descriptor it can look up
    (``memlet_parser._fill_missing_slices``). Anything else -- ``A[ind[0]]``,
    ``A[2:4, ind[1], 3]``, ``A[ind[0] + 1]`` -- is not a name, so the parser
    took it for a symbolic expression, produced a subset referring to a
    function of runtime data, and the access fell back to the interpreter.
    That is a spelling restriction, not a feature boundary: the same access
    written through a temporary already lowered.

    Materializing here reduces every spelling to the one that works. Which
    slots need it is decided by the shared classification
    (:func:`~dace.frontend.python.nextgen.semantics.indexing.array_index_slots`),
    the same one inference uses to give those slots typing placeholders, so
    the two stages cannot disagree about what an index array is.

    :raises UnsupportedFeatureError: If an index expression is array-valued but
        of a dtype advanced indexing does not take (only integer and boolean
        arrays index; a float array is a user error the parser reports).
    """
    replacements = array_index_slots(node.slice, lambda element: _array_index_container(element, node, state))
    return substitute_slots(
        node, {
            position: ast.copy_location(ast.Name(id=container, ctx=ast.Load()), node)
            for position, container in replacements.items()
        })


def _array_index_container(element: ast.expr, node: ast.Subscript, state: LoweringState) -> Optional[str]:
    """
    The container holding the value of one array-valued index expression,
    materializing it on first use, or None when the expression is not an
    array-valued index (a scalar, a symbol, or an unresolvable form that the
    ordinary paths report on).
    """
    if isinstance(element, ast.Subscript):
        # ``A[ind.T[0]]``: the index is a subscript of a registry ATTRIBUTE
        # read, which is not a data name either. Materializing the attribute
        # first gives it one, after which it resolves like any other index.
        element = rewrite_attribute_subscript_base(element, state, writable=False)
    try:
        access = resolve_access(element, state)
    except UnsupportedFeatureError:
        access = None  # e.g. a nested indirection; not our case to rewrite

    if access is not None:
        shape = access.numpy_shape
        dtype = access.descriptor.dtype
    else:
        inferred = state.inference.infer(element)
        if not inferred.is_data or inferred.descriptor is None:
            return None
        shape = [size for size in inferred.descriptor.shape if size != 1]
        dtype = inferred.descriptor.dtype
    if not shape:
        return None  # A scalar index is indirection, lowered by the tasklet paths
    if not isinstance(dtype, dtypes.typeclass) or (dtype not in dtypes.INTEGER_TYPES
                                                   and dtype not in (dtypes.bool, dtypes.bool_)):
        raise UnsupportedFeatureError(
            f'Index expression "{astutils.unparse(element)}" is an array of {dtype}, '
            'which cannot index (only integer and boolean arrays can)',
            state.context.filename,
            node,
            category='advanced-index')

    # A value-numbering key, not a classification one: within a single
    # statement ANF guarantees that the same source text denotes the same
    # value, so an index written twice -- as an accumulation's read and write
    # sides always do -- is evaluated once.
    key = astutils.unparse(element)
    cached = state.index_arrays.get(key)
    if cached is not None:
        return cached

    descriptor = data.Array(dtype, shape)
    container = state.context.add_container('__idxarr', descriptor)
    # Bound to itself so the substituted name resolves as an ordinary container.
    state.context.bind(container, container)
    target = DataAccess(container, subsets.Range.from_array(descriptor), descriptor)
    if access is not None:
        state.emitter.emit(tn.CopyNode(target=container, memlet=Memlet(data=access.container, subset=access.subset)))
    else:
        # A computed index (``ind[0] + 1``): evaluate it into the container.
        elementwise.emit_computation(target, element, node, state)
    state.index_arrays[key] = container
    return container


def materialize_boolean_gathers(value: ast.expr, statement: ast.stmt, state: LoweringState) -> ast.expr:
    """
    Gather every boolean-mask read inside an expression into its own container,
    returning the expression rewritten to read those containers.

    A mask read has a data-dependent result size, which is why it cannot simply
    be typed and then lowered like any other access: inference has no correct
    answer for it before the selected element count is computed. The bare
    top-level form (``B = A[mask]``) solves that by typing and lowering
    together (``rules.assign._lower_boolean_gather_assign``); this does the
    same for every OTHER position -- nested in a computation (``A[mask] +
    1.0``), passed to a call (``np.sum(A[mask])``), written into a subset --
    by performing the gather FIRST. What the rest of the statement then sees is
    an ordinary container whose shape is the symbol the gather minted, so
    inference and lowering proceed from there with nothing special about it.

    Returns the expression unchanged when it contains no mask read, which is
    the overwhelmingly common case.
    """
    replacements: List[Tuple[ast.Subscript, ast.Name]] = []
    for node in ast.walk(value):
        if not isinstance(node, ast.Subscript):
            continue
        container = _boolean_gather_container(node, statement, state)
        if container is not None:
            replacements.append((node, ast.copy_location(ast.Name(id=container, ctx=ast.Load()), node)))
    if not replacements:
        return value
    return _substitute_subscripts(value, replacements)


def _mentions_a_boolean_container(slice_node: ast.expr, state: LoweringState) -> bool:
    """
    Whether any index slot reads a boolean container -- a necessary condition
    for the subscript to be a mask read.

    Purely a cheap pre-filter, and deliberately so: it is dictionary lookups
    only, and it runs on every subscript of every assignment. Confirming an
    actual mask read costs a ``resolve_access`` plus a parse, which is far too
    much to spend on the ``A[i]`` that most subscripts are.
    """
    for element in index_slots(slice_node):
        base = element
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if not isinstance(base, ast.Name):
            continue
        binding = state.context.resolve(base.id)
        if binding is None or binding.kind != 'container':
            continue
        descriptor = state.context.containers.get(binding.container)
        if descriptor is not None and descriptor.dtype in (dtypes.bool, dtypes.bool_):
            return True
    return False


def _boolean_gather_container(node: ast.Subscript, statement: ast.stmt, state: LoweringState) -> Optional[str]:
    """
    Emit the gather for one boolean-mask read and return the container holding
    its result, or None when the subscript is not a mask read at all.

    A mask this frontend cannot express (combined with an integer index array,
    more than one mask, partial coverage) is left alone rather than reported
    here: the ordinary paths reach the same refusal with the message that fits
    the position the access is used in.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if not isinstance(node.value, (ast.Name, ast.Attribute)):
        return None
    if not _mentions_a_boolean_container(node.slice, state):
        return None
    try:
        base = resolve_access(node.value, state)
        if base is None:
            return None
        expr = state.inference.parse_access(materialize_array_indices(node, state))
        if not expr.arrdims or not advanced_indexing.has_boolean_index(expr, state.context):
            return None
        mask_container = advanced_indexing.resolve_single_boolean_mask(node, expr, base.container, state.context,
                                                                       state.inference)
    except UnsupportedFeatureError:
        return None
    access = advanced_indexing.emit_boolean_gather('__maskread',
                                                   base.container,
                                                   base.descriptor,
                                                   mask_container,
                                                   statement,
                                                   state,
                                                   base_subset=expr.subset)
    # Bound to itself so the substituted name resolves as an ordinary container
    # read when the enclosing expression is lowered.
    state.context.bind(access.container, access.container)
    return access.container


def _substitute_subscripts(value: ast.expr, replacements: List[Tuple[ast.Subscript, ast.Name]]) -> ast.expr:
    """Replace specific subscript NODES (matched by identity, not by text) with
    the names of the containers they were gathered into."""

    class _Substituter(ast.NodeTransformer):

        def visit_Subscript(self, subscript: ast.Subscript) -> ast.AST:
            for original, replacement in replacements:
                if original is subscript:
                    return replacement
            return self.generic_visit(subscript)

    return _Substituter().visit(value)


def materialize_advanced_index_reads(value: ast.expr, statement: ast.stmt, state: LoweringState) -> ast.expr:
    """
    Gather EVERY advanced-indexing read in an expression into a temporary,
    including a top-level one, returning the expression rewritten to read the
    temporaries.

    Used where the value cannot be written straight into the target because the
    target is itself an advanced index (``A[i1] = A[i2]``, a gather and a
    scatter in one statement). Both halves already lower on their own; what
    does not work is doing them at once, because the scatter emits the value
    inside its own map, whose index space is the target's -- not the source's.
    Gathering first is exactly the ``t = A[i2]; A[i1] = t`` spelling that
    always worked, and it also gives NumPy's evaluation order for free: the
    right-hand side is fully read before any element of the target is written,
    which matters when the two overlap.
    """
    return _materialize_advanced_indices(value, statement, state, include_top_level=True)


def _materialize_advanced_indices(value: ast.expr,
                                  statement: ast.stmt,
                                  state: LoweringState,
                                  include_top_level: bool = False) -> ast.expr:
    """
    Gather every advanced-indexing subscript nested inside an expression into a
    temporary container, returning the expression rewritten to read the
    temporaries.

    ANF leaves these in operand position -- a data subscript is a legal operand,
    and canonicalization has no type information with which to tell
    ``A[scalar_i]`` from ``A[index_array]`` -- so the split has to happen here,
    where the index's descriptor is known.

    :param include_top_level: Also gather the expression when it is ITSELF such
                              an access. Off by default: an ordinary assignment
                              writes a top-level access straight into its
                              target, and staging it through a temporary would
                              be a pointless copy.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if isinstance(value, ast.Subscript) and not include_top_level:
        return value  # A top-level access writes the real target directly

    replacements: List[Tuple[ast.Subscript, ast.Name]] = []
    for node in ast.walk(value):
        if not isinstance(node, ast.Subscript):
            continue
        access = _advanced_index_access(node, state)
        if access is None:
            continue
        shape = [size for size in access.output_shape if size != 1]
        descriptor = (data.Array(access.descriptor.dtype, shape) if shape else data.Scalar(access.descriptor.dtype))
        container = state.context.add_container('__advidx', descriptor)
        # Bound to itself so the substituted name resolves as an ordinary
        # container read when the enclosing expression is lowered.
        state.context.bind(container, container)
        temporary = DataAccess(container, subsets.Range.from_array(descriptor), descriptor)
        advanced_indexing.emit_gather(temporary, access, statement, state)
        replacements.append((node, ast.copy_location(ast.Name(id=container, ctx=ast.Load()), node)))

    if not replacements:
        return value
    return _substitute_subscripts(value, replacements)


def lower_call(target: Optional[ast.expr], call: ast.Call, statement: ast.stmt, state: LoweringState) -> None:
    """
    Lower a canonical call, routing by callee identity and operand types:
    nested ``@dace.program``/SDFG-convertible callees go to the inlining rule,
    registry-known NumPy calls to their mechanisms (elementwise ufuncs, array
    creation, WCR reductions), and everything else to the callback fallback.

    :param target: The assignment target expression, or None for a bare call.
    """
    from dace.frontend.python.nextgen.lowering.rules import calls  # Deferred: rules import this module
    qualname, callee = state.inference.resolve_callee(call.func)
    if calls.is_sdfg_convertible(callee):
        calls.lower_nested_call(target, call, callee, statement, state)
        return
    if target is not None and _lower_reshape_call(target, call, qualname, state):
        return
    if target is not None and _lower_view_call(target, call, qualname, state):
        return
    try:
        if _lower_registry_call(target, call, qualname, callee, statement, state):
            return
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, reason)
        return
    category = _call_gap_category(call, qualname, state)
    if category == 'detected-callback':
        message = f'call to Python callback "{qualname}"'
    else:
        message = f'no lowering for call "{qualname}"'
    fallback_to_callback(statement, state, message, category=category)


def _lower_reshape_call(target: ast.expr, call: ast.Call, qualname: str, state: LoweringState) -> bool:
    """
    Lower ``<data>.reshape(<shape>)`` / ``numpy.reshape(<data>, <shape>)`` as
    a view binding.

    DaCe views carry a shape independent of their source subset, so a total
    -size-preserving reshape reduces to reinterpreting the resolved source
    access under a fresh, explicitly-shaped view container — the "frontend
    view path" that view-producing registry replacements (like the classic
    ``reshape`` replacement) explicitly defer to, since
    :func:`_expansion_viable` rejects any replacement whose RESULT is a view
    (a view binding is frontend-visible state, not something a deferred
    ``ReplacementCallNode`` can represent).

    ``<data>.ravel()`` is included: it is ``reshape(-1)`` whenever NumPy would
    return a view, i.e. on a contiguous source. On a non-contiguous source
    NumPy copies, and so does the registry implementation — that case is left
    to the deferred expansion by :func:`_reshape_operands`.

    A non-name target (``out[:] = numpy.reshape(A, [3, 3])``) is not a
    rebinding, so it takes the same view but copies through it into the
    target's own subset.

    Returns False when the call is not a recognized/resolvable reshape form
    (target not a data access, base not a data access, shape not resolvable,
    or element count mismatch); the caller then falls through to the normal
    registry-call dispatch and, ultimately, a callback.
    """
    base_expr, shape_args = _reshape_operands(call, qualname, state)
    if base_expr is None:
        return False
    access = resolve_access(base_expr, state)
    if access is None:
        return False
    if _is_ravel_call(call) and not (isinstance(access.descriptor, data.Array)
                                     and _is_contiguous_flat(access.descriptor)):
        return False  # Non-contiguous ravel copies; the registry implementation does that
    shape = _reshape_shape(shape_args, access.subset, state)
    if shape is None:
        return False
    target_access: Optional[DataAccess] = None
    if not isinstance(target, ast.Name):
        target_access = resolve_access(target, state)
        if target_access is None:
            return False
    view_descriptor = data.ArrayView(access.descriptor.dtype, shape)
    view_name = state.context.add_container(target.id if target_access is None else '__reshape', view_descriptor)
    if target_access is None:
        state.context.bind(target.id, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=access.container,
                    memlet=Memlet(data=access.container, subset=access.subset),
                    src_desc=access.descriptor,
                    view_desc=view_descriptor))
    if target_access is not None:
        # Writing the reshaped data into an existing container: copy out of
        # the view, which carries the reshaped dimensions.
        state.emitter.emit(
            tn.CopyNode(target=target_access.container,
                        memlet=Memlet(data=view_name,
                                      subset=subsets.Range.from_array(view_descriptor),
                                      other_subset=target_access.subset)))
    return True


def _lower_view_call(target: ast.expr, call: ast.Call, qualname: str, state: LoweringState) -> bool:
    """
    Lower any registry call whose RESULT is a view of one of its own data
    arguments — ``A.view(dace.int32)``, the dtype-reinterpreting view — as a
    frontend view binding.

    This is the general form of the path :func:`_lower_reshape_call`
    implements for one shape-changing call, and it exists for the same reason:
    a view binding is frontend-visible state that a deferred
    ``ReplacementCallNode`` cannot represent (:func:`_expansion_viable`
    rejects a view-valued result outright), so a replacement that returns one
    lowers here or not at all. Nothing about the call is recognized
    syntactically — the replacement is looked up like any other, trial-run on
    a scratch SDFG, and taken only if what it produced was exactly one view of
    an input and no dataflow — so a newly registered view-returning
    replacement needs no change here.

    The emitted descriptor is the one the TRIAL computed rather than one
    re-derived from inference: the replacement is the authority on the
    reinterpreted shape and strides (``view()`` divides them with
    ``symbolic.int_floor``, which prints correctly for a symbolic stride where
    ``//`` does not).

    ``reshape`` keeps its dedicated path ahead of this one because it resolves
    a ``-1`` dimension and checks the element count, neither of which the
    registry implementation does; this path only sees the forms that one
    declines.
    """
    # No descriptor-inference entry is required: the view descriptor comes
    # from the trial run below, not from the frontend typing the result first.
    name, receiver, receiver_object = _resolve_replacement_name(call,
                                                                qualname,
                                                                state,
                                                                require_descriptor_inference=False)
    if name is None or receiver_object is not None:
        # An object receiver produces a view of nothing this path can bind.
        return False
    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if receiver is not None:
        arguments = [receiver] + arguments
        data_arguments = data_arguments | {receiver}
    function = _replacement_implementation(name, receiver, arguments, data_arguments, state)
    if function is None:
        return False
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.is_program_dependent(function):
        # A replacement that reads the SDFG built so far is not a view producer,
        # and running it on a scratch answers nothing (see
        # :func:`_expansion_viable`): leave it to the deferred-call path.
        return False
    trial = _run_view_trial(function, arguments, keywords, data_arguments, state)
    if trial is None:
        return False
    view_descriptor, source, memlet = trial
    if not _view_matches_window(view_descriptor, state.context.containers[source], memlet):
        return False
    view_descriptor = copy.deepcopy(view_descriptor)
    target_access: Optional[DataAccess] = None
    if not isinstance(target, ast.Name):
        target_access = resolve_access(target, state)
        if target_access is None:
            return False
    view_name = state.context.add_container(target.id if target_access is None else '__view', view_descriptor)
    if target_access is None:
        state.context.bind(target.id, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=source,
                    memlet=copy.deepcopy(memlet),
                    src_desc=state.context.containers[source],
                    view_desc=view_descriptor))
    if target_access is not None:
        # Writing the viewed data into an existing container, as in
        # ``_lower_reshape_call``: copy out of the view, which carries the
        # reinterpreted dimensions.
        state.emitter.emit(
            tn.CopyNode(target=target_access.container,
                        memlet=Memlet(data=view_name,
                                      subset=subsets.Range.from_array(view_descriptor),
                                      other_subset=target_access.subset)))
    return True


def _run_view_trial(function, arguments: List, keywords: dict, data_arguments: set,
                    state: LoweringState) -> Optional[Tuple[data.Data, str, Memlet]]:
    """
    Trial-run a replacement on a scratch SDFG (the same trial-before-commit
    shape as :func:`_expansion_viable`) and report the view binding it
    recorded.

    :return: (view descriptor, source container, viewing memlet) when the
             result is exactly one view of one of the call's own data
             arguments and the replacement emitted no dataflow of its own —
             all a :class:`~...treenodes.ViewNode` can express — otherwise
             None, leaving the call to the ordinary dispatch below.
    """
    scratch, scratch_state, shim = _replacement_trial_scratch(data_arguments, state)
    try:
        result = function(shim, scratch, scratch_state, *arguments, **keywords)
    except Exception:
        return None
    result = _unwrap_nested_call(result)
    if not isinstance(result, str) or result not in shim.views:
        return None  # Not a view-valued result: an ordinary replacement
    if len(shim.views) != 1 or scratch_state.number_of_nodes() > 0:
        return None  # Additional bindings or dataflow a single ViewNode would drop
    source, memlet = shim.views[result]
    if source not in data_arguments or source not in state.context.containers:
        return None  # A view of the replacement's own temporary, not of an argument
    return scratch.arrays[result], source, memlet


def _view_matches_window(view_descriptor: data.Data, source_descriptor: data.Data, memlet: Memlet) -> bool:
    """
    Whether a view descriptor reinterprets exactly the window its viewing
    memlet names, measured in BYTES (the dtypes may differ — that is the point
    of ``A.view(dtype)``).

    A view is an aliasing reinterpretation, not an allocation, so it holds
    neither more nor fewer bytes than the subset it aliases: more reads out of
    bounds, fewer means part of the named window is unreachable through it.
    The registry implementations do not all check this — ``reshape`` builds a
    ``(2, 4)`` view of a 6-element array, where NumPy raises — so a binding
    that provably disagrees with its own window is refused here and the call
    falls through to a path that preserves Python's own error (ultimately the
    callback).

    The view side is measured by its SHAPE, not its ``total_size``: a view
    inherits the source's total size (``reshape`` passes it through verbatim),
    so it is the shape that says how far indexing it reaches. Only a PROVABLE
    disagreement is refused; symbolic sizes that do not compare are accepted,
    since the alternative is refusing every symbolically-shaped view.
    """
    view_bytes = data._prod(view_descriptor.shape) * view_descriptor.dtype.bytes
    window_bytes = memlet.subset.num_elements() * source_descriptor.dtype.bytes
    difference = symbolic.simplify(view_bytes - window_bytes)
    return not (difference.is_number and difference != 0)


def _resolve_replacement_name(call: ast.Call,
                              qualname: str,
                              state: LoweringState,
                              require_descriptor_inference: bool = True) -> Tuple[Optional[str], Optional[str], Any]:
    """
    The registry key a call lowers through, as (name, receiver, receiver
    object).

    A free function is keyed by its registry-normalized qualname; a method
    (``A.copy()``, ``A.view(dtype)``) by its bare method name plus the
    container its receiver resolves to, which is also the replacement's first
    positional argument (the classic frontend's convention, ``newast.py``'s
    Call visitor). A method on a compile-time OBJECT (``commworld.Bcast(A)``,
    where the receiver is an ``mpi4py`` communicator from the closure rather
    than a container) is keyed the same way but against the object's class,
    and the object travels alongside so the expansion can publish it under the
    receiver name. Returns (None, None, None) when the call is registered
    under none of them.

    :param require_descriptor_inference: See :func:`_replacement_registered`.
    """
    if _replacement_registered(qualname, require_descriptor_inference):
        return qualname, None, None
    fallback = normalize_qualname(getattr(call.func, 'qualname', None) or astutils.rname(call.func))
    if _replacement_registered(fallback, require_descriptor_inference):
        return fallback, None, None
    receiver_access = _method_call_receiver(call, state)
    if (receiver_access is not None
            and _method_registered(receiver_access.descriptor, call.func.attr, require_descriptor_inference)):
        return call.func.attr, receiver_access.container, None
    receiver_object = _object_method_receiver(call, state)
    if (receiver_object is not None
            and _method_registered(receiver_object, call.func.attr, require_descriptor_inference)):
        return call.func.attr, astutils.rname(call.func.value), receiver_object
    return None, None, None


def _object_method_receiver(call: ast.Call, state: LoweringState) -> Optional[Any]:
    """
    The compile-time Python object a method call's receiver is, when it is not
    a repository container (``commworld.Bcast(A)``), or None.
    """
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return None
    return state.inference.constant_object(call.func.value)


def _replacement_implementation(name: str,
                                receiver: Optional[str],
                                arguments: List,
                                data_arguments: set,
                                state: LoweringState,
                                receiver_object: Any = None):
    """
    The registry implementation a resolved replacement name refers to: a bound
    method on the receiver's descriptor (or object) class, an operator resolved
    from its operand classes (which keeps ``getop``'s class-hierarchy
    resolution), or a free function.
    """
    from dace.frontend.common import op_repository as oprepo

    if receiver_object is not None:
        return oprepo.Replacements.get_method(type(receiver_object), name)
    if receiver is not None:
        return oprepo.Replacements.get_method(type(state.context.containers[receiver]), name)
    if name.startswith(oprepo.OPERATOR_QUALNAME_MARKER):
        return oprepo.Replacements.getop(*_operator_lookup(oprepo.decode_operator_qualname(name), arguments,
                                                           data_arguments, state.context.containers))
    return oprepo.Replacements.get(name)


def _reshape_operands(call: ast.Call, qualname: str, state: LoweringState) -> Tuple[Optional[ast.expr], List[ast.expr]]:
    """
    The (base array expression, shape argument expressions) of a reshape call
    form, or (None, []) if the call is not one of the recognized forms:
    ``x.reshape(shape)``, ``x.reshape(d0, d1, ...)``, or
    ``numpy.reshape(x, shape)``. The shape argument expressions are returned
    as-is (not flattened) — ANF hoists a literal shape tuple to a name bound
    to a static sequence, so :func:`_reshape_shape` resolves through
    inference rather than requiring an inline ``ast.Tuple``/``ast.List``.

    Both forms are ``Attribute`` calls ending in ``.reshape``
    (``numpy.reshape(A, s)`` is ``Attribute(value=Name('numpy'),
    attr='reshape')``), so the method form is recognized by what its base
    resolves to — a registered container — rather than by its syntax; the
    module base of the function form resolves to no data and cannot be
    misread as the array being reshaped.
    """
    if qualname == 'numpy.reshape' and not call.keywords and call.args:
        return call.args[0], call.args[1:]
    if (isinstance(call.func, ast.Attribute) and call.func.attr == 'reshape' and not call.keywords
            and resolve_access(call.func.value, state) is not None):
        return call.func.value, call.args
    if _is_ravel_call(call):
        # ``A.ravel()`` is ``A.reshape(-1)``; the single ``-1`` dimension is
        # filled from the source's element count by ``_reshape_shape``.
        return call.func.value, [ast.Constant(value=-1)]
    return None, []


def _is_ravel_call(call: ast.Call) -> bool:
    """Whether the call is the no-argument ``<data>.ravel()`` method form."""
    return (isinstance(call.func, ast.Attribute) and call.func.attr == 'ravel' and not call.args and not call.keywords)


def _reshape_shape(shape_args: List[ast.expr], source_subset, state: LoweringState) -> Optional[List]:
    """
    Resolve reshape target-shape arguments to dimension sizes, filling in at
    most one ``-1`` placeholder dimension from the source's total element count
    (NumPy semantics). Dimensions may be compile-time integers or symbolic
    expressions (``A.reshape([1, N * N])``) — a view descriptor carries either.
    None when a dimension resolves to neither, or when the requested shape's
    element count does not match the source's (the caller degrades to a
    callback).
    """
    dims: List[Any] = []
    if len(shape_args) == 1:
        # A single shape argument: either a literal tuple/list (canonical
        # 'static' per Inferred.infer) or an ANF-hoisted name bound to one
        # (List/Tuple literals of atoms are canonical 'flat' static values —
        # ANF hoists them out of operand positions like this call argument).
        try:
            inferred = state.inference.infer(shape_args[0])
        except UnsupportedFeatureError:
            inferred = None
        if inferred is not None and inferred.kind == 'static':
            for element in inferred.value.elements:
                dimension = _reshape_dimension(element, state)
                if dimension is None:
                    return None
                dims.append(dimension)
    if not dims:
        for arg in shape_args:
            dimension = _reshape_dimension(arg, state)
            if dimension is None:
                return None
            dims.append(dimension)
    if not dims:
        return None
    total = source_subset.num_elements()
    try:
        known_product = 1
        placeholder = None
        for index, dim in enumerate(dims):
            if isinstance(dim, int) and dim == -1:
                if placeholder is not None:
                    return None
                placeholder = index
            else:
                known_product = known_product * dim
        if placeholder is not None:
            if known_product == 0 or bool(symbolic.simplify(total % known_product) != 0):
                return None
            dims[placeholder] = total // known_product
        elif not _same_element_count(known_product, total):
            return None
    except Exception:
        return None
    return dims


def _reshape_dimension(expression: ast.expr, state: LoweringState) -> Optional[Any]:
    """
    Resolve one reshape dimension to a compile-time integer or a symbolic
    expression, or None when it is neither.
    """
    value = state.inference.constant_int(expression)
    if value is not None:
        return value
    try:
        inferred = state.inference.infer(expression)
    except UnsupportedFeatureError:
        return None
    if inferred.kind == 'symbolic':
        return inferred.value
    return None


def _same_element_count(requested: Any, total: Any) -> bool:
    """Whether a requested shape's element count matches the source's, over
    integers or symbolic expressions."""
    if requested == total:
        return True
    return bool(symbolic.simplify(requested - total) == 0)


def resolve_attribute_data(base: DataAccess, attr_name: str, state: LoweringState) -> Optional[DataAccess]:
    """
    Materialize a registered data-descriptor ATTRIBUTE access (``A.T``,
    ``A.real``, ``A.imag``, ``A.flat``) of ``base`` into a fresh transient
    container.

    Attribute access is an expression, not a call, so it cannot go through
    ``_lower_registry_call``/``_lower_replacement_call`` (which only trigger
    from ``ast.Call`` nodes): this is the attribute-family counterpart,
    called from dedicated frontend entry points (``lower_attribute_assign``
    below, and the ``ast.Attribute`` branch of
    ``rules.returns._materialize_return_value``) tried before generic
    dispatch -- the same placement principle as ``_lower_reshape_call``.

    A CONTIGUOUS ``.flat`` binds a :class:`~...treenodes.ViewNode`: NumPy's
    flatiter is, for DaCe's purposes, an aliasing flattened view of a
    contiguous source, so writes through it must reach the original array --
    the same "view bindings are frontend-visible state a deferred
    ``ReplacementCallNode`` cannot represent" reasoning that makes
    ``_lower_reshape_call`` a dedicated view-binding path instead of a
    deferred call. Everything else (``.T``, ``.real``, ``.imag``, and
    non-contiguous ``.flat``, which the registry implementation copies
    through an explicit map) computes a fresh array and is safe to defer
    through a :class:`~...treenodes.ReplacementCallNode` running the exact
    classic ATTRIBUTE registry implementation.

    :return: The resolved access, or None when ``attr_name`` has no lowering
             path (:func:`~dace.frontend.python.nextgen.common.supported_data_attribute`),
             or the registry call is not viable here (the caller falls back
             to a callback).
    """
    if not supported_data_attribute(attr_name):
        return None
    if attribute_aliases_source(base, attr_name):
        return _materialize_flat_view(base, state)
    return _materialize_attribute_replacement(base, attr_name, state)


def _is_contiguous_flat(descriptor: data.Array) -> bool:
    """Whether ``descriptor`` is contiguous in the sense NumPy's ``.flat``
    requires to be representable as a plain reshape-to-1D view (mirrors the
    check in the classic ``flat()`` replacement,
    ``replacements/array_manipulation.py``)."""
    shape = descriptor.shape
    total = data._prod(shape)
    contiguous_strides = tuple(data._prod(shape[i + 1:]) for i in range(len(shape)))
    return bool(descriptor.total_size == total) and tuple(descriptor.strides) == contiguous_strides


def _materialize_flat_view(base: DataAccess, state: LoweringState) -> DataAccess:
    """Bind ``.flat`` of a contiguous array as a 1-D view, the same mechanism
    ``_lower_reshape_call`` uses for a single-dimension reshape."""
    total = data._prod(base.descriptor.shape) if isinstance(base.descriptor, data.Array) else 1
    view_descriptor = data.ArrayView(base.descriptor.dtype, [total])
    view_name = state.context.add_container('__flat', view_descriptor)
    # Self-bound so a caller that substitutes this name back into an
    # expression (e.g. rewriting a ``.flat[...]`` subscript's base, see
    # ``rewrite_flat_subscript_base``) resolves it as an ordinary container.
    state.context.bind(view_name, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=base.container,
                    memlet=Memlet(data=base.container, subset=base.subset),
                    src_desc=base.descriptor,
                    view_desc=view_descriptor))
    return DataAccess(view_name, subsets.Range.from_array(view_descriptor), view_descriptor)


def _materialize_attribute_replacement(base: DataAccess, attr_name: str, state: LoweringState) -> Optional[DataAccess]:
    """Defer an ATTRIBUTE-family replacement that computes a fresh array
    (everything :func:`resolve_attribute_data` does not bind as a view) to a
    :class:`~...treenodes.ReplacementCallNode`, after checking on a scratch
    SDFG that the deferred expansion will actually succeed (the same
    trial-before-commit shape as ``_expansion_viable``, generalized in
    :func:`_run_attribute_trial` to look the implementation up by
    ``(classname, attr_name)`` instead of a free-function qualname).

    A PROGRAM-DEPENDENT attribute skips the trial for the reason
    :func:`_expansion_viable` documents, and takes its result descriptor from
    the registry's own inference instead: ``ParameterArray.grad`` names the
    gradient buffer an earlier ``torch.autograd.backward`` expansion created,
    which no scratch SDFG holds."""
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    typename = type(base.descriptor).__name__
    function = oprepo.Replacements.get_attribute(typename, attr_name)
    if function is None:
        return None
    if oprepo.is_program_dependent(function):
        result_descriptor = _inferred_attribute_descriptor(base.descriptor, attr_name)
    else:
        result_descriptor = _run_attribute_trial(function, base.container, state)
    if result_descriptor is None:
        return None
    descriptor = copy.deepcopy(result_descriptor)
    container = state.context.add_container('__attr', descriptor)
    state.context.bind(container, container)  # Self-bound, see _materialize_flat_view
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=oprepo.attribute_qualname(typename, attr_name),
                               target=container,
                               arguments=[base.container],
                               keyword_arguments={},
                               data_arguments={base.container}))
    return DataAccess(container, subsets.Range.from_array(descriptor), descriptor)


def _unwrap_nested_call(result: Any) -> Any:
    """
    The value a replacement returned, with the state-chaining wrapper removed.

    Replacements that need several states return a ``(NestedCall, result)``
    pair, the first element being a bookkeeping object for the caller's state
    machine rather than part of the result.
    """
    from dace.frontend.python.nested_call import NestedCall  # Deferred to keep rule import light
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], NestedCall):
        return result[1]
    return result


def _inferred_attribute_descriptor(descriptor: data.Data, attr_name: str) -> Optional[data.Data]:
    """
    The result descriptor of an ATTRIBUTE-family replacement according to the
    registry's own inference (``infers_attribute_descriptor``), for the
    program-dependent attributes :func:`_run_attribute_trial` cannot answer
    for. Returns None when the registry types no result, which leaves the
    caller on the callback path.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    infer_fn = oprepo.Replacements.get_attribute_descriptor_inference(type(descriptor), attr_name)
    if infer_fn is None:
        return None
    try:
        result = infer_fn(descriptor)
    except Exception:
        return None
    if isinstance(result, (tuple, list)) and len(result) == 1:
        result = result[0]
    return result if isinstance(result, data.Data) else None


def _run_attribute_trial(function, container: str, state: LoweringState) -> Optional[data.Data]:
    """
    Run an ATTRIBUTE-family replacement on a scratch SDFG with a single data
    argument (every registered attribute implementation takes exactly the
    base container, plus attribute-specific defaulted keyword arguments) and
    return the resulting descriptor when deferred expansion will succeed at
    SDFG-build time (``tree_to_sdfg.visit_ReplacementCallNode`` runs the exact
    same call): no recorded view bindings, and a single named result. Returns
    None otherwise, including on any exception the trial call raises.

    Mirrors ``_expansion_viable``, specialized to the attribute calling
    convention and returning the scratch-computed descriptor directly (rather
    than a separate inference call) so the emitted container's shape is
    guaranteed consistent with what expansion will actually produce.
    """
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import ReplacementVisitorShim
    from dace.sdfg.sdfg import SDFG

    scratch = SDFG('__attribute_viability')
    descriptor = copy.deepcopy(state.context.containers[container])
    descriptor.transient = False
    scratch.add_datadesc(container, descriptor)
    scratch_state = scratch.add_state()
    shim = ReplacementVisitorShim(scratch, scratch_state, '__viability_target')
    try:
        result = function(shim, scratch, scratch_state, container)
    except Exception:
        return None
    result = _unwrap_nested_call(result)
    if not isinstance(result, str) or result not in scratch.arrays:
        return None
    if result in shim.views:
        return None  # A view-valued result belongs on the frontend view path (see _expansion_viable)
    return scratch.arrays[result]


def lower_attribute_assign(target: ast.Name, value: ast.Attribute, state: LoweringState) -> bool:
    """
    Lower ``<name> = <data>.<attr>`` through :func:`resolve_attribute_data`,
    tried before the generic Name/Attribute aliasing path in
    ``rules.assign._lower_name_assign`` -- the same placement principle as
    ``_lower_reshape_call`` being tried before ``_lower_registry_call``.

    Returns False when the value's base is not a resolvable data access or
    the attribute has no viable lowering (the caller falls through to the
    ordinary assignment paths, ultimately a callback).
    """
    base = resolve_access(value.value, state)
    if base is None:
        return False
    access = resolve_attribute_data(base, value.attr, state)
    if access is None:
        return False
    state.context.bind(target.id, access.container)
    return True


def attribute_aliases_source(base: DataAccess, attr_name: str) -> bool:
    """
    Whether a registered data attribute ALIASES its source, so a write through
    it reaches the source array -- as opposed to computing a fresh array, where
    a write would be discarded. This is the distinction
    :func:`resolve_attribute_data` acts on, asked ahead of materializing.

    A contiguous ``.flat`` is NumPy's flatiter over the same storage; a
    non-contiguous one, and ``.T``/``.real``/``.imag``, compute a copy.
    """
    return attr_name == 'flat' and isinstance(base.descriptor, data.Array) and _is_contiguous_flat(base.descriptor)


def rewrite_attribute_subscript_base(subscript: ast.Subscript, state: LoweringState, writable: bool) -> ast.Subscript:
    """
    Rewrite a subscript over a registry ATTRIBUTE-family read (``A.flat[10:15]``,
    ``A.T[0:2]``) to reference the materialized container directly, so the
    ordinary subscript machinery (``rules.assign``/``access.resolve_access``,
    neither of which knows about the ATTRIBUTE registry) can resolve it.
    Without this the access reaches the shared memlet parser with ``A.flat`` as
    a data NAME (*"Use of undefined data"*) and the statement degrades to a
    callback.

    :param writable: Whether the subscript is a WRITE target. A write may only
        be rewritten onto an attribute that aliases its source
        (:func:`attribute_aliases_source`); rewriting a computed one would
        silently discard the write, so those are left alone and degrade to a
        callback instead of miscompiling. A read of a computed attribute is
        just a read of the copy, and is always safe.

    Returns ``subscript`` unchanged when its base is not a rewritable
    attribute read.
    """
    if not isinstance(subscript.value, ast.Attribute) or not supported_data_attribute(subscript.value.attr):
        return subscript
    base = resolve_access(subscript.value.value, state)
    if base is None:
        return subscript
    aliases = attribute_aliases_source(base, subscript.value.attr)
    if writable and not aliases:
        return subscript
    if not aliases and state.emitter.in_dataflow_scope:
        # Materializing a computed attribute emits a deferred replacement call,
        # which is not a legal node inside a scope (see
        # :func:`_materialize_attribute_reads`).
        return subscript
    access = resolve_attribute_data(base, subscript.value.attr, state)
    if access is None:
        return subscript
    rewritten = copy.copy(subscript)
    rewritten.value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), subscript.value)
    return rewritten


def _call_gap_category(call: ast.Call, qualname: str, state: LoweringState) -> str:
    """
    The provenance category of a call with no lowering: calls whose callee
    preprocessing already wrapped as a Python callback are *intended*
    interpreter work (``detected-callback``, mirrored in the classic
    frontend's ``callback_mapping``); callees that are themselves opaque
    Python objects are downstream of an earlier gap
    (``pyobject-propagation``); everything else is a genuine missing-lowering
    gap, recorded with its qualified name (``unknown-call:<qualname>``).
    """
    callback_mapping = state.emitter.root.callback_mapping
    # Preprocessing rewrites detected-callable call sites to the callback
    # name and records it on the Call node's qualname (or leaves the name as
    # the callee for coroutine/decorated callables).
    detected_names = {astutils.rname(call.func), getattr(call.func, 'qualname', None), getattr(call, 'qualname', None)}
    if detected_names & set(callback_mapping):
        return 'detected-callback'
    if isinstance(call.func, ast.Name):
        binding = state.context.resolve(call.func.id)
        if binding is not None and binding.kind == 'container':
            descriptor = state.context.containers[binding.container]
            if isinstance(descriptor.dtype, dtypes.pyobject):
                return 'pyobject-propagation'
    return f'unknown-call:{qualname}'


def _promote_scalar_arguments(call: ast.Call, state: LoweringState) -> Optional['Inferred']:
    """
    Bind the integer-SCALAR temporaries a call's arguments name as SYMBOLS,
    defined from their containers by an injected interstate assignment
    (:class:`~...treenodes.AssignNode`), and re-run the call's descriptor
    inference against them.

    Shapes, bounds and axes are symbol-side quantities (see
    ``doc/extensions/symbolic.rst``, "Symbolic types vs. scalars"): the
    registry's descriptor inference reads such an argument as a number, and a
    container name is not one. Canonicalization is what puts a container
    there — hoisting ``numpy.zeros((A.shape[0], A.shape[1] + 2))`` leaves each
    element in a temporary of its own, and a temporary whose value is not
    *already* symbolic materializes as a scalar. Promoting it here is the
    general repair, and unlike folding a known value it works just as well
    when the scalar is only known at run time (``n = counts[0]`` …
    ``numpy.zeros((n, ))``), which is the same mechanism the classic frontend
    uses (``newast.ProgramVisitor._add_state`` plus a symbol assignment) and
    the one already used for dynamic map bounds and hoisted indices.

    The binding lasts for this statement only (``LoweringState.
    promoted_scalars``), which is exactly what the definition claims: the
    symbol holds the value the scalar had HERE. The per-statement cache the
    range bounds use (``access._index_symbol``) is shared, so a scalar named by
    both an argument and a subset in one statement defines one symbol.

    Attempted only after the un-promoted call fails to type, and rolled back
    (bindings, symbols, and emitted nodes) when promotion does not make it
    type — so an argument the registry is happy to take as a container is
    never disturbed.

    :return: The now-successful inference result, or None if promotion did not
             apply or did not help (nothing is left emitted in that case).
    """
    if state.emitter.in_dataflow_scope:
        # An interstate assignment is not a legal node inside a dataflow scope.
        return None
    candidates = _promotable_scalar_arguments(call, state)
    if not candidates:
        return None

    mark = state.emitter.checkpoint()
    before = state.context.snapshot()
    restore_from = len(state.promoted_scalars)
    defined: List[Tuple[str, str]] = []
    promoted = False
    for name, access in candidates:
        read = scalar_read_expression(access)
        if read is None:
            continue
        symbol_name = state.index_symbols.get(read)
        if symbol_name is None:
            symbol_name = state.context.fresh_name(f'__sym_{name.lstrip("_")}_')
            state.context.symbols[symbol_name] = symbolic.symbol(symbol_name, access.descriptor.dtype)
            # Its value is read out of a container while the program runs, so a
            # container sized by it cannot cross the program boundary -- see
            # ``rules.returns._reject_deferred_size``.
            state.context.deferred_symbols.add(symbol_name)
            state.emitter.emit(
                tn.AssignNode(name=symbol_name,
                              value=CodeBlock(read),
                              edge=InterstateEdge(assignments={symbol_name: read})))
            state.index_symbols[read] = symbol_name
            defined.append((read, symbol_name))
        state.promoted_scalars.append((name, state.context.bindings.get(name)))
        state.context.bind_symbol(name, access.descriptor.dtype, symbol_name=symbol_name)
        promoted = True

    inferred = state.inference.infer_call(call) if promoted else None
    if inferred is not None and not _undefined_result_symbols(inferred, state):
        return inferred
    state.emitter.rollback(mark)
    state.context.restore(before)
    del state.promoted_scalars[restore_from:]
    for read, symbol_name in defined:
        state.index_symbols.pop(read, None)
        state.context.symbols.pop(symbol_name, None)
        state.context.symbol_aliases.pop(symbol_name, None)
        state.context.deferred_symbols.discard(symbol_name)
    return None


def _undefined_result_symbols(inferred: 'Inferred', state: LoweringState) -> Set[str]:
    """
    The symbols a call's result descriptors are sized by that NOTHING defines.

    A registry implementation may invent a symbol for a scalar argument it
    needs as a number — ``numpy.arange(stop)`` names its result's single
    dimension after the container holding ``stop``
    (``array_creation.arange_promoted_symbol_name``). Inventing the name is
    only half of it: unless the program also assigns the symbol, the result is
    sized by a symbol the SDFG can only ask its caller for, and a shape the
    program itself computed becomes a required program argument. Treated
    exactly like a failure to type, so :func:`_promote_scalar_arguments`
    defines the symbols for real and the same inference is asked again.
    """
    descriptors: Tuple[data.Data, ...] = ()
    if inferred.is_data_tuple:
        descriptors = inferred.descriptors or ()
    elif inferred.is_data and inferred.descriptor is not None:
        descriptors = (inferred.descriptor, )
    undefined: Set[str] = set()
    for descriptor in descriptors:
        for size in descriptor.shape:
            undefined |= {name for name in symbolic.symlist(size) if name not in state.context.symbols}
    return undefined


def _promotable_scalar_arguments(call: ast.Call, state: LoweringState) -> List[Tuple[str, DataAccess]]:
    """
    The names among a call's arguments that hold an integer scalar container,
    as (source name, access) pairs, deduplicated and in argument order.
    Static-sequence arguments are looked through, since a hoisted shape tuple is
    exactly where these appear.
    """
    found: Dict[str, DataAccess] = {}

    def collect(expression: ast.expr) -> None:
        if not isinstance(expression, ast.Name):
            return
        binding = state.context.resolve(expression.id)
        if binding is None:
            return
        if binding.kind == 'static':
            for element in state.context.static_values[expression.id].elements:
                collect(element)
            return
        if binding.kind != 'container' or expression.id in found:
            return
        if binding.container in state.context.symbolic_scalar_values:
            # Already resolvable as a compile-time expression; inference reads
            # the value straight off the name, so there is nothing to promote.
            return
        descriptor = state.context.containers[binding.container]
        if not isinstance(descriptor, data.Scalar):
            return
        try:
            integral = numpy.issubdtype(descriptor.dtype.type, numpy.integer)
        except TypeError:
            return
        if not integral:
            return  # A non-integer scalar is not a symbol; the ordinary paths report it
        found[expression.id] = DataAccess(binding.container, subsets.Range.from_array(descriptor), descriptor)

    for argument in call.args:
        collect(argument)
    for keyword in call.keywords:
        collect(keyword.value)
    return list(found.items())


def _lower_registry_call(target: Optional[ast.expr], call: ast.Call, qualname: str, callee: object, statement: ast.stmt,
                         state: LoweringState) -> bool:
    """
    Try to lower a call through the descriptor-inference registry and the
    NumPy mechanisms. Returns False when no mechanism applies (the caller
    falls back to a callback).
    """
    inferred = state.inference.infer_call(call)
    if inferred is None or _undefined_result_symbols(inferred, state):
        # An argument the registry reads as a NUMBER but the program holds in a
        # container is a spelling problem, not a missing feature: promote it to
        # a symbol and ask again. A result sized by a symbol nothing defines
        # counts as untyped here — the promotion is what makes such a symbol
        # real, and without it the size would be unanswerable at run time.
        inferred = _promote_scalar_arguments(call, state)
    if inferred is None:
        return False  # No registry entry: the interpreter fallback preserves semantics.
    out_access: Optional[DataAccess] = None
    if inferred.is_none_output:
        if target is not None:
            # A call typed as zero-output but assigned to a target (e.g.
            # ``b = A.fill(0)``) isn't a form any mechanism below produces;
            # the interpreter fallback preserves Python's own semantics
            # (assigning ``None``).
            return False
    elif inferred.is_data_tuple:
        # Several result containers (``numpy.split``, ``numpy.divmod``). Every
        # mechanism below writes ONE target, so this takes its own path.
        return _lower_multi_output_call(target, call, qualname, inferred, statement, state)
    elif not inferred.is_data:
        # Not a data-valued result at all: the interpreter fallback preserves
        # semantics.
        return False
    elif target is None:
        # A bare-statement data-valued call. If its result goes to an ``out=``
        # container (``numpy.add(A, B, out=C)``, ``numpy.concatenate([a, b],
        # out=c)``), that container IS the target and the statement lowers;
        # otherwise the result is unused and the interpreter fallback
        # preserves semantics.
        out_access = _out_keyword_access(call, state)
        if out_access is None:
            return False

    # Python's builtin min/max/abs over scalar operands. These ARE elementwise
    # ufuncs: the classic replacements implement min/max as a pairwise chain of
    # two-scalar comparisons (never an iterable reduction -- see
    # ``reduction.py::_pymin``) and abs as a single-operand call, and their
    # descriptor inference only types scalar operands, so an array operand
    # never reaches here. Routing them through the elementwise mechanism rather
    # than deferred expansion is what makes them work INSIDE dataflow scopes,
    # where expansion cannot run at all (it needs state machinery).
    if target is not None and not call.keywords:
        builtin_ufunc = _BUILTIN_ELEMENTWISE.get(qualname)
        if builtin_ufunc is not None and len(call.args) == _BUILTIN_ELEMENTWISE_ARITY[qualname]:
            # A self-referential ``b = max(b, x)`` needs no special handling
            # here: it lowers as the plain read-modify-write it is, and
            # ``ResolveWriteConflicts`` turns it into a max conflict resolution
            # if the write turns out to collide.
            target_access = _call_target_access(target, inferred, statement, state)
            elementwise.emit_ufunc(target_access, builtin_ufunc, call.args, statement, state)
            return True

        # Datatype conversions INSIDE a dataflow scope. Outside one the
        # deferred expansion runs the registry converter itself (and keeps
        # classic's exact behavior); inside, expansion is unavailable, and a
        # cast is just a single-operand tasklet.
        if len(call.args) == 1 and state.emitter.in_dataflow_scope:
            cast_dtype = _converter_dtype(qualname)
            if cast_dtype is not None:
                target_access = _call_target_access(target, inferred, statement, state)
                elementwise.emit_cast(target_access, cast_dtype, call.args[0], statement, state)
                return True

    # NumPy universal functions, direct (numpy.add(...)) or through one of
    # their reduce/accumulate/outer methods (numpy.add.reduce(...)). A plain
    # elementwise call with no keywords lowers through the lightweight
    # elementwise mechanism (a single tasklet expression); everything that
    # mechanism cannot express -- reduce/accumulate/outer (which need real
    # reduction/scan/broadcast dataflow) or any keyword argument (out=/
    # where=/axis=/keepdims=/initial=, which the elementwise mechanism has no
    # way to honor) -- defers to the actual registry ufunc implementation
    # through the same deferred-expansion mechanism used for other
    # replacements below. A zero-output call (e.g. ``A.fill(0)``) is never a
    # ufunc match, so it falls through untouched to deferred replacement
    # expansion at the end of this function.
    ufunc_form = state.inference.resolve_ufunc_call(call)
    if ufunc_form is not None:
        ufunc, ufunc_method = ufunc_form
        if ufunc_method is None and not call.keywords:
            # Positional outputs (``numpy.add(A, B, C)``) are not operands:
            # only the ufunc's first ``nin`` arguments are inputs, and the
            # output container is already resolved in ``out_access``.
            target_access = out_access or _call_target_access(target, inferred, statement, state)
            elementwise.emit_ufunc(target_access, ufunc.__name__, call.args[:ufunc.nin], statement, state)
            return True
        return _lower_ufunc_replacement_call(target,
                                             call,
                                             ufunc.__name__,
                                             ufunc_method,
                                             inferred,
                                             statement,
                                             state,
                                             target_access=out_access)

    # Array/stream creation. ``qualname`` is already registry-normalized by
    # ``resolve_callee`` (e.g. ``dace.define_stream``, whose real module is
    # ``dace.frontend.python.wrappers``, normalizes back to the ``dace.``
    # form CREATION_CALLS is keyed on).
    creation_name = qualname
    if creation_name in creation.CREATION_CALLS:
        # Keywords this mechanism can honor: either they only affect the
        # descriptor (which inference derived, and the caller already
        # allocated from) or the contents this mechanism writes. Anything else
        # would be silently ignored, so it falls back instead.
        if any(keyword.arg not in ('dtype', 'fill_value', 'shape', 'strides', 'storage', 'lifetime', 'buffer_size')
               for keyword in call.keywords):
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        creation.lower_creation(creation_name, target_access, call, statement, state)
        return True

    # Reductions (full or per-axis with a compile-time scalar axis). Forms the
    # WCR-map mechanism does not cover (e.g. tuple axes) fall through to the
    # deferred replacement expansion below.
    matched = _match_reduction(call, qualname)
    if matched is not None:
        reduction_ufunc, source_expr, axis_expr = matched
        axis: Optional[int] = None
        supported = True
        if axis_expr is not None and not (isinstance(axis_expr, ast.Constant) and axis_expr.value is None):
            axis = state.inference.constant_int(axis_expr)
            supported = axis is not None
        source = resolve_access(source_expr, state) if supported else None
        if supported and source is not None:
            target_access = _call_target_access(target, inferred, statement, state)
            reduction.emit_reduction(target_access, reduction_ufunc, source, statement, state, axis=axis)
            return True

    # Stream pushes inside a dataflow scope, which the deferred replacement
    # expansion below cannot reach (it adds state machinery).
    if streams.lower_stream_push(call, statement, state):
        return True

    # Deferred replacement expansion: the descriptor inference typed the call,
    # so vetted registry functions emit a ReplacementCallNode that
    # tree_to_sdfg expands through the classic replacement implementation.
    if _lower_replacement_call(target, call, qualname, inferred, statement, state, target_access=out_access):
        return True
    return False


def _converter_dtype(qualname: str) -> Optional[dtypes.typeclass]:
    """
    The target dtype of a datatype-conversion call (``dace.int64``,
    ``numpy.float32``, the builtin ``int``/``float``/``bool``), or None when
    the call is not one.
    """
    from dace.frontend.python.replacements.array_manipulation import _resolve_converter_dtype
    name = qualname.rsplit('.', 1)[-1]
    if name not in dtypes.TYPECLASS_STRINGS:
        return None
    try:
        return _resolve_converter_dtype(name)
    except Exception:
        return None


def _out_keyword_access(call: ast.Call, state: LoweringState) -> Optional[DataAccess]:
    """
    The container a call's output argument writes into, or None when the call
    has none or it does not resolve to a data access.

    Covers both spellings: an explicit ``out=`` keyword, and — for a direct
    NumPy ufunc call — the positional output NumPy allows after the ufunc's
    ``nin`` inputs (``numpy.add(A, B, C)``).
    """
    for keyword in call.keywords:
        if keyword.arg == 'out':
            return resolve_access(keyword.value, state)
    ufunc_form = state.inference.resolve_ufunc_call(call)
    if ufunc_form is not None:
        ufunc, ufunc_method = ufunc_form
        if ufunc_method is None and len(call.args) > ufunc.nin:
            return resolve_access(call.args[ufunc.nin], state)
    return None


def _match_reduction(call: ast.Call, qualname: str) -> Optional[Tuple[str, ast.expr, Optional[ast.expr]]]:
    """
    Match a reduction call form (``np.sum(A[, axis])`` or ``A.sum([axis])``,
    ``axis=`` keyword allowed).

    :return: (WCR ufunc name, source expression, axis expression or None), or
             None if the call is not a lowerable reduction form (extra
             arguments like ``out=``/``initial=`` change semantics).
    """
    if qualname in _REDUCTION_CALLS:
        if not call.args or len(call.args) > 2:
            return None
        ufunc_name = _REDUCTION_CALLS[qualname]
        source_expr = call.args[0]
        axis_expr = call.args[1] if len(call.args) == 2 else None
    elif isinstance(call.func, ast.Attribute) and call.func.attr in _REDUCTION_METHODS:
        if len(call.args) > 1:
            return None
        ufunc_name = _REDUCTION_METHODS[call.func.attr]
        source_expr = call.func.value
        axis_expr = call.args[0] if call.args else None
    else:
        return None
    for keyword in call.keywords:
        if keyword.arg != 'axis' or axis_expr is not None:
            return None
        axis_expr = keyword.value
    return ufunc_name, source_expr, axis_expr


def _method_call_receiver(call: ast.Call, state: LoweringState) -> Optional[DataAccess]:
    """
    The bound receiver of a method-call form (``<base>.<method>(...)``), or
    None when the call is not a method call on a whole registered container.

    Restricted to ``Name`` and structure-member ``Attribute`` bases (both
    resolve through :func:`resolve_access` to the FULL array, never a
    subset) — a ``_method_rep`` replacement receives the whole container by
    name (e.g. ``arr: str`` in ``_ndarray_copy``), so a ``Subscript`` base
    like ``A[0].sum()`` cannot be passed as a receiver without silently
    operating on all of ``A`` instead of the indexed element.
    """
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, (ast.Name, ast.Attribute)):
        return None
    return resolve_access(call.func.value, state)


def _lower_multi_output_call(target: Optional[ast.expr], call: ast.Call, qualname: str, inferred, statement: ast.stmt,
                             state: LoweringState) -> bool:
    """
    Lower a call whose result is SEVERAL containers (``numpy.split``,
    ``numpy.divmod``, ``numpy.frexp``, ...) as one deferred replacement call
    with one result container per output.

    The target binds to a static sequence of container references, which is the
    same representation an inlined multi-return nested program produces
    (``rules.calls._bind_call_results``), so the element reads canonicalization
    already emits for unpacking (``p = __unpack0[0]``) fold to direct accesses
    with nothing further to build.

    Returns False when the call cannot be lowered this way (no assignment
    target, unresolvable arguments, or a trial expansion that does not produce
    exactly the inferred number of containers), so the caller falls back to a
    callback.
    """
    if not isinstance(target, ast.Name):
        # A bare statement discards every result, and a subscript target would
        # need a copy per output into subsets the call knows nothing about.
        return False
    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted

    ufunc_form = state.inference.resolve_ufunc_call(call)
    ufunc_name: Optional[str] = None
    ufunc_method: Optional[str] = None
    name = qualname
    if ufunc_form is not None:
        ufunc, ufunc_method = ufunc_form
        ufunc_name = ufunc.__name__
        if ufunc_method is not None:
            return False  # reduce/accumulate/outer are single-output by construction
        unsupported = {keyword.arg for keyword in call.keywords} - _SUPPORTED_UFUNC_KEYWORDS
        if unsupported:
            return False
        name = f'numpy.{ufunc_name}'
        viable = _multi_output_viable(None, ufunc_name, None, arguments, keywords, data_arguments, state,
                                      len(inferred.descriptors))
    else:
        if not _replacement_registered(name):
            name = normalize_qualname(getattr(call.func, 'qualname', None) or astutils.rname(call.func))
            if not _replacement_registered(name):
                return False
        viable = _multi_output_viable(name, None, None, arguments, keywords, data_arguments, state,
                                      len(inferred.descriptors))
    if not viable:
        return False

    containers = []
    for descriptor in inferred.descriptors:
        result_descriptor = copy.deepcopy(descriptor)
        result_descriptor.transient = True
        containers.append(state.context.add_container(f'{target.id}_out', result_descriptor))
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=name,
                               target=containers[0],
                               extra_targets=containers[1:],
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments,
                               ufunc_name=ufunc_name,
                               ufunc_method=ufunc_method))
    for container in containers:
        state.context.bind(container, container)
    elements = [ast.copy_location(ast.Name(id=container, ctx=ast.Load()), statement) for container in containers]
    state.context.bind_static(target.id, StaticSequence(elements=elements, kind='tuple'))
    return True


def _multi_output_viable(name: Optional[str], ufunc_name: Optional[str], receiver: Optional[str], arguments: List,
                         keywords: dict, data_arguments: set, state: LoweringState, expected: int) -> bool:
    """
    Trial-run a multi-output replacement on a scratch SDFG, the same
    commit-after-trial check :func:`_expansion_viable` performs, but requiring
    exactly ``expected`` result containers rather than one.

    A result that is a view of an input is rejected for the same reason it is
    there: a view binding is frontend-visible state that a deferred call cannot
    represent.
    """
    from dace.frontend.common import op_repository as oprepo

    scratch, scratch_state, shim = _replacement_trial_scratch(data_arguments, state)
    try:
        if ufunc_name is not None:
            function = oprepo.Replacements.get_ufunc(None)
            if function is None:
                return False
            result = function(shim, None, scratch, scratch_state, ufunc_name, list(arguments), dict(keywords))
        else:
            function = oprepo.Replacements.get(name)
            if function is None:
                return False
            result = function(shim, scratch, scratch_state, *arguments, **keywords)
    except Exception:
        return False
    result = _unwrap_nested_call(result)
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        return False
    if any(not isinstance(element, str) or element not in scratch.arrays for element in result):
        return False
    return not any(element in shim.views for element in result)


def _lower_replacement_call(target: Optional[ast.expr],
                            call: ast.Call,
                            qualname: str,
                            inferred,
                            statement: ast.stmt,
                            state: LoweringState,
                            target_access: Optional[DataAccess] = None) -> bool:
    """
    Emit a deferred :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`
    for a vetted registry replacement (free function or bound method).
    Returns False when the call is not vetted or an argument cannot be
    resolved (the caller falls back).

    :param target: The assignment target, or None for a bare-statement,
        zero-output call (``inferred.is_none_output``) — resolved through the
        same method-family lookup (:func:`_method_call_receiver`,
        :func:`_method_registered`) as a targeted method call.
    :param target_access: An already-resolved write target, used instead of
        ``target`` when the call's output goes to an ``out=`` keyword argument.
    """
    # The method family (``_method_rep``, e.g. ``A.copy()``/``A.fill(0)``)
    # covers both a targeted, data-valued method call and a bare-statement,
    # zero-output one; that doesn't depend on ``target``/
    # ``inferred.is_none_output`` because the caller's own gates above already
    # guarantee the two are paired correctly (targeted only when data-valued,
    # untargeted only when zero-output) before execution ever reaches here.
    name, receiver, receiver_object = _resolve_replacement_name(call, qualname, state)
    if name is None:
        return False
    # NOTE: expansion inside a dataflow scope is allowed. The expansion adds
    # state machinery, which a map body cannot hold directly -- but
    # ``tree_to_sdfg`` emits a map body containing a state boundary as a NESTED
    # SDFG, which can, and ``insert_state_boundaries`` forces that boundary
    # wherever a ReplacementCallNode sits inside a scope.

    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if receiver is not None:
        # Mirror the classic frontend's convention (newast.py's Call
        # visitor): the receiver is the replacement's first positional
        # argument. An OBJECT receiver passes by name only -- it names no
        # container, so it is not a data argument; the implementation resolves
        # it through the visitor's globals.
        arguments = [receiver] + arguments
        if receiver_object is None:
            data_arguments = data_arguments | {receiver}
    if _bind_compile_time_result(target, name, arguments, keywords, data_arguments, state, receiver, receiver_object):
        return True
    if not _expansion_viable(
            name, arguments, keywords, data_arguments, state, receiver=receiver, receiver_object=receiver_object):
        return False
    written: Optional[DataAccess] = target_access
    copy_out = False
    if target_access is None and target is None:
        # No frontend-declared target container to write into (a bare
        # statement): the schedule tree still requires a registered
        # container reference, so use the call's own data operand (the
        # method receiver, or the first data-valued argument for a
        # zero-output free function like ``dace.comm.Bcast``) as a stand-in.
        # ``visit_ReplacementCallNode`` never dereferences it as a copy
        # destination for a genuinely zero-output replacement (its result is
        # ``None``/``[]``).
        #
        # Keyword values count: a library-node replacement takes ALL of its
        # operands by keyword (``donnx.ONNXGather(data=inp, output=out,
        # axis=1)``), so restricting the search to positional arguments left
        # those calls with no stand-in and dropped them to a callback.
        operands = list(arguments) + list(keywords.values())
        target_container = (receiver if receiver_object is None else None) or next(
            (value for value in operands if isinstance(value, str) and value in data_arguments), None)
        if target_container is None:
            return False
    else:
        if written is None:
            written = _call_target_access(target, inferred, statement, state)
        target_container, copy_out = _replacement_write_target(written, inferred, state)
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=name,
                               target=target_container,
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments,
                               receiver=receiver,
                               receiver_object=receiver_object,
                               target_preexisting=_writes_a_preexisting_target(target, copy_out)))
    _apply_self_descriptor_side_effect(name, receiver, receiver_object, arguments, keywords, state)
    if target_container in state.context.containers and isinstance(state.context.containers[target_container].dtype,
                                                                   dtypes.pyobject):
        # An opaque HANDLE this replacement produced, which the next one
        # consumes by name (``dace.comm.Subarray`` into
        # ``dace.comm.Redistribute``): see ``ProgramContext.replacement_handles``.
        state.context.replacement_handles[target_container] = _HandleProducer(target=target_container,
                                                                              name=name,
                                                                              arguments=list(arguments),
                                                                              keyword_arguments=dict(keywords),
                                                                              data_arguments=set(data_arguments),
                                                                              receiver=receiver,
                                                                              receiver_object=receiver_object,
                                                                              order=len(
                                                                                  state.context.replacement_handles))
    if written is not None and copy_out:
        _emit_replacement_copy_out(target_container, written, statement, state)
    return True


def _apply_self_descriptor_side_effect(name: str, receiver: Optional[str], receiver_object: Any, arguments: List,
                                       keywords: dict, state: LoweringState) -> None:
    """
    Apply a method replacement's effect on its RECEIVER's descriptor, as the
    registry reports it through ``infers_method_self_descriptor``.

    Some method replacements change what their receiver *is* instead of
    producing a result: ``x.requires_grad_()`` converts ``x`` into a
    :class:`~dace.data.ml.ParameterArray`, and everything that follows depends
    on the repository agreeing — ``x.grad`` is only typeable, and only
    lowerable, against the converted descriptor. The classic frontend performs
    the conversion inside the replacement itself (on ``sdfg.arrays``), which a
    deferred expansion cannot do in time for the rest of the frontend to see
    it; the inference entry exists precisely so the descriptor change can be
    made here, at lowering time, on the container repository.

    A no-op for a free function, an object receiver (no repository container to
    retype), or a method the registry declares no self-inference for.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if receiver is None or receiver_object is not None or receiver not in state.context.containers:
        return
    descriptor = state.context.containers[receiver]
    infer_fn = oprepo.Replacements.get_method_self_descriptor_inference(type(descriptor), name)
    if infer_fn is None:
        return
    try:
        # The receiver is the replacement's first positional argument (see the
        # caller); the inference entry takes its DESCRIPTOR there instead.
        updated = infer_fn(descriptor, *arguments[1:], **keywords)
    except Exception:
        return  # A registry entry that cannot type this call leaves the receiver alone
    if isinstance(updated, (tuple, list)) and len(updated) == 1:
        updated = updated[0]
    if isinstance(updated, data.Data):
        state.context.retype_container(receiver, updated)


def _fills_subset(descriptor: Optional[data.Data], subset: subsets.Range) -> bool:
    """
    Whether a replacement's RESULT descriptor covers a write subset
    elementwise — that is, whether an expansion filling the whole result
    container performs exactly the write the statement asked for.

    A result of a *different* shape is a broadcast (``a[:] = numpy.mean(b)``
    fills ten elements from one), which no expansion can perform: it is handed
    a container name and nothing else, so it writes element 0 and leaves the
    rest of the target untouched.

    An unknown descriptor answers True, leaving such a call on whatever path
    it took before this check existed.
    """
    if descriptor is None:
        return True
    result_shape = [size for size in getattr(descriptor, 'shape', ()) if size != 1]
    return [str(size) for size in result_shape] == [str(size) for size in nondegenerate_shape(subset)]


def _writes_a_preexisting_target(target: Optional[ast.expr], copy_out: bool) -> bool:
    """
    Whether a replacement call writes its result into a container the program
    already had, rather than into one introduced to hold the result.

    A subscript or member target (``out[:] = stream.pop()``) names storage
    that existed before the statement; a bare name (``r = stream.pop()``)
    binds the result itself, so the container is the call's own. A result
    routed through a temporary (``copy_out``) never writes the target
    directly at all.

    See :attr:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode.target_preexisting`,
    which carries this to the expansion.
    """
    return not copy_out and isinstance(target, (ast.Subscript, ast.Attribute))


def _replacement_write_target(access: DataAccess, inferred, state: LoweringState) -> Tuple[str, bool]:
    """
    The container a deferred replacement expansion should write into.

    ``ReplacementCallNode.target`` names a whole container — it carries no
    subset, and no shape of its own — so anything but a write that fills the
    target container exactly has to go through a temporary that is copied into
    the target subset afterwards: a write into PART of one (``out[i] =
    numpy.mean(...)`` inside a map), or one that BROADCASTS a smaller result
    across it (``out[:] = numpy.mean(...)``). Without this the expansion
    writes the container's element 0 — silently overwritten by every map
    iteration in the first case, and leaving the rest of the target at its
    previous value in the second.

    :return: (container to expand into, whether a copy-out is needed).
    """
    descriptor = state.context.containers.get(access.container)
    if (descriptor is not None and str(access.subset) == str(subsets.Range.from_array(descriptor))
            and _fills_subset(inferred.descriptor, access.subset)):
        return access.container, False
    result_descriptor = copy.deepcopy(inferred.descriptor)
    result_descriptor.transient = True
    return state.context.add_container('__replacement', result_descriptor), True


def _emit_replacement_copy_out(source: str, target: DataAccess, statement: ast.stmt, state: LoweringState) -> None:
    """
    Write a replacement's whole-container result into the target subset it was
    declared to write (see :func:`_replacement_write_target`).

    A result that fills the subset elementwise is a plain memlet copy. One that
    does not is a broadcast, which a memlet cannot express (its two subsets
    would cover different element counts), so it takes the elementwise
    mechanism's map instead — the same lowering an ordinary ``a[:] = scalar``
    assignment gets.
    """
    descriptor = state.context.containers[source]
    access = DataAccess(source, subsets.Range.from_array(descriptor), descriptor)
    if not _fills_subset(descriptor, target.subset):
        elementwise.emit_elementwise(target, '__in', [('__in', access)], statement, state)
        return
    state.emitter.emit(
        tn.CopyNode(target=target.container,
                    memlet=Memlet(data=source, subset=access.subset, other_subset=target.subset)))


def _bind_compile_time_result(target: Optional[ast.expr],
                              name: str,
                              arguments: List,
                              keywords: dict,
                              data_arguments: set,
                              state: LoweringState,
                              receiver: Optional[str],
                              receiver_object: Any = None) -> bool:
    """
    Bind the target name to a replacement's COMPILE-TIME result, for
    replacements that return a Python value rather than a container: ``len(A)``
    yields the array's leading dimension (an int or a symbol), ``slice(0, 10)``
    a Python ``slice``. There is nothing to emit for these — the value lives in
    the frontend's binding repository and folds into whatever consumes it.

    The value comes from running the replacement on a scratch SDFG, the same
    trial :func:`_expansion_viable` performs; a result that names a scratch
    container (the ordinary case) is not a compile-time value and returns
    False, leaving the call to deferred expansion.

    :return: True when the target was bound to a compile-time value.
    """

    if not isinstance(target, ast.Name):
        return False
    function = _replacement_implementation(name, receiver, arguments, data_arguments, state, receiver_object)
    if function is None:
        return False
    scratch, scratch_state, shim = _replacement_trial_scratch(
        data_arguments, state, {receiver: receiver_object} if receiver_object is not None else None)
    try:
        result = function(shim, scratch, scratch_state, *arguments, **keywords)
    except Exception:
        return False
    if not _is_compile_time_value(result) or shim.views or scratch_state.nodes():
        # Either not a plain value, or the replacement also emitted dataflow
        # for it — in which case the value is not the whole result.
        return False
    state.context.bind_constant(target.id, result)
    return True


def _is_compile_time_value(result: Any) -> bool:
    """Whether a replacement's result is a compile-time Python value (as
    opposed to a container name, ``None``, or a list of names)."""
    if isinstance(result, (bool, numbers.Number, slice)):
        return True
    if symbolic.issymbolic(result):
        return True
    if isinstance(result, tuple) and result:
        return all(_is_compile_time_value(element) for element in result)
    return False


def _replacement_trial_scratch(data_arguments: set, state: LoweringState, globals_: Optional[dict] = None):
    """
    Build the scratch SDFG/state/shim triple shared by the build-time
    replacement-viability trials (:func:`_expansion_viable` and
    :func:`_ufunc_expansion_viable`): a standalone SDFG carrying copies of
    just the data arguments a replacement call touches, on which the
    replacement can be trial-run without mutating the real program.
    """
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import ReplacementVisitorShim
    from dace.sdfg.sdfg import SDFG

    producers = _handle_producers(data_arguments, state)
    scratch = SDFG('__replacement_viability')
    needed = set(data_arguments)
    for producer in producers:
        needed |= {name for name in producer.data_arguments if name in state.context.containers}
    # A replayed producer installs its handle's real descriptor under the
    # handle's own name; pre-declaring the frontend's placeholder there would
    # take the name and force the replay to uniquify away from it (the real
    # expansion releases the same declaration, see
    # ``tree_to_sdfg._release_declared_descriptor``).
    needed -= {producer.target for producer in producers}
    for data_name in needed:
        descriptor = copy.deepcopy(state.context.containers[data_name])
        descriptor.transient = False
        if isinstance(descriptor, data.DistributedDescriptor) and not descriptor.name:
            # A DECLARED communicator (the MPI replacements' inference types the
            # name a grid-creating call binds; the grid itself is installed when
            # that call's own expansion runs). By the time THIS replacement runs
            # for real, the declaration has been replaced by the installed
            # descriptor, so the scratch has to show it installed too -- a
            # sub-grid of an unnamed parent does not even validate.
            descriptor.name = data_name
        scratch.add_datadesc(data_name, descriptor)
    scratch_state = scratch.add_state()
    shim = ReplacementVisitorShim(scratch, scratch_state, '__viability_target')
    shim.globals.update(globals_ or {})
    _replay_handle_producers(producers, scratch, scratch_state, shim, state)
    return scratch, scratch_state, shim


def _handle_producers(data_arguments: set, state: LoweringState) -> List['_HandleProducer']:
    """
    The recorded producer calls of every opaque HANDLE among ``data_arguments``,
    transitively (a subarray's producer needs its process grid), in creation
    order.
    """
    producers: Dict[str, '_HandleProducer'] = {}
    pending = list(data_arguments)
    while pending:
        name = pending.pop()
        producer = state.context.replacement_handles.get(name)
        if producer is None or name in producers:
            continue
        producers[name] = producer
        pending.extend(producer.data_arguments)
    return sorted(producers.values(), key=lambda producer: producer.order)


def _replay_handle_producers(producers: List['_HandleProducer'], scratch, scratch_state, shim,
                             state: LoweringState) -> None:
    """
    Re-run the producers of the opaque handles a trial's arguments name, so the
    scratch SDFG carries the registry state they install (``sdfg.subarrays``
    for ``dace.comm.Subarray``) and not merely a descriptor for the name.

    Their own viability was already established when they were lowered, so a
    failure here can only mean the scratch is missing something the real
    program has; the trial then simply reports the consumer as non-viable,
    which is the same conservative answer as before this replay existed.
    """
    for producer in producers:
        function = _replacement_implementation(producer.name, producer.receiver, producer.arguments,
                                               producer.data_arguments, state, producer.receiver_object)
        if function is None:
            continue
        shim._target_name = producer.target
        if producer.receiver_object is not None:
            shim.globals[producer.receiver] = producer.receiver_object
        try:
            function(shim, scratch, scratch_state, *producer.arguments, **producer.keyword_arguments)
        except Exception:
            pass  # The consumer's own trial reports the shortfall
    shim._target_name = '__viability_target'


def _expansion_viable(name: str,
                      arguments: List,
                      keywords: dict,
                      data_arguments: set,
                      state: LoweringState,
                      receiver: Optional[str] = None,
                      receiver_object: Any = None) -> bool:
    """
    Trial-run a replacement on a scratch SDFG to decide, at tree-build time
    (where the graceful fallback is a callback), whether deferred expansion
    will succeed at tree-to-SDFG time (where failure would be a hard error).
    Runs the exact code path of ``tree_to_sdfg.visit_ReplacementCallNode``:
    non-viable outcomes are exceptions, recorded view bindings, and
    unsupported return forms.

    A PROGRAM-DEPENDENT replacement (``oprepo.is_program_dependent``) is
    exempt: it reads the SDFG built so far — ``torch.autograd.backward``
    differentiates everything parsed up to its call site — and the scratch
    holds only the containers this one call names, so the trial would report
    a failure the real expansion never has.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    function = _replacement_implementation(name, receiver, arguments, data_arguments, state, receiver_object)
    if oprepo.is_program_dependent(function):
        return True
    scratch, scratch_state, shim = _replacement_trial_scratch(
        data_arguments, state, {receiver: receiver_object} if receiver_object is not None else None)
    try:
        result = function(shim, scratch, scratch_state, *arguments, **keywords)
    except Exception:
        return False
    result = _unwrap_nested_call(result)
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], str):
        # Single-element list of output datanames: the convention ufunc
        # implementations and the dtype-cast replacements (``dace.int64(A)``)
        # use. ``visit_ReplacementCallNode`` normalizes it to the bare-string
        # form, so accepting it here keeps the two checks in agreement.
        result = result[0]
    if isinstance(result, str) and result in shim.views:
        # The RESULT is a view of an input (``reshape``/``ravel`` on contiguous
        # data): binding that as a Python name is frontend state, so it belongs
        # on the frontend view path (:func:`_lower_reshape_call`), not here.
        # Views recorded on the way to a freshly computed result are fine --
        # ``visit_ReplacementCallNode`` materializes those.
        return False
    if isinstance(result, str):
        return result in scratch.arrays
    return result is None or result == []


#: Keyword arguments the registry ufunc implementations
#: (:mod:`dace.frontend.python.replacements.ufunc`) understand, across all of
#: the direct-call/reduce/accumulate/outer forms. Anything else changes
#: semantics in a way the registry does not implement, so it is rejected
#: rather than silently ignored.
_SUPPORTED_UFUNC_KEYWORDS = frozenset({'out', 'where', 'axis', 'keepdims', 'initial', 'dtype'})


def _lower_ufunc_replacement_call(target: Optional[ast.expr],
                                  call: ast.Call,
                                  ufunc_name: str,
                                  ufunc_method: Optional[str],
                                  inferred,
                                  statement: ast.stmt,
                                  state: LoweringState,
                                  target_access: Optional[DataAccess] = None) -> bool:
    """
    Emit a deferred ``ReplacementCallNode`` for a NumPy ufunc call that the
    lightweight elementwise mechanism cannot express: ``reduce``/
    ``accumulate``/``outer`` (which need real reduction/scan/broadcast
    dataflow, not a single tasklet expression), or any keyword argument.
    Reuses the same deferred-expansion machinery as :func:`_lower_replacement_call`,
    but through the ufunc registry keyspace (``get_ufunc``/``get_ufunc``
    calling convention) rather than the free-function one. Returns False when
    the call is not viable (an argument cannot be resolved, an unsupported
    keyword is present, or the trial expansion fails), matching
    :func:`_lower_replacement_call`'s contract.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get_ufunc(ufunc_method) is None:
        return False
    unsupported = {keyword.arg for keyword in call.keywords} - _SUPPORTED_UFUNC_KEYWORDS
    if unsupported:
        return False  # Keywords the registry ufunc implementation does not accept
    # NOTE: expansion inside a dataflow scope is allowed. The expansion adds
    # state machinery, which a map body cannot hold directly -- but
    # ``tree_to_sdfg`` emits a map body containing a state boundary as a NESTED
    # SDFG, which can, and ``insert_state_boundaries`` forces that boundary
    # wherever a ReplacementCallNode sits inside a scope.

    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if not _ufunc_expansion_viable(ufunc_name, ufunc_method, arguments, keywords, data_arguments, state):
        return False
    if target_access is None:
        target_access = _call_target_access(target, inferred, statement, state)
    # Same subset/broadcast routing as the free-function path
    # (:func:`_lower_replacement_call`): the node's target is a bare container
    # name, so a partial or broadcasting write expands into a temporary first.
    target_container, copy_out = _replacement_write_target(target_access, inferred, state)
    display_name = f'numpy.{ufunc_name}' + (f'.{ufunc_method}' if ufunc_method else '')
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=display_name,
                               target=target_container,
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments,
                               ufunc_name=ufunc_name,
                               ufunc_method=ufunc_method,
                               target_preexisting=_writes_a_preexisting_target(target, copy_out)))
    if copy_out:
        _emit_replacement_copy_out(target_container, target_access, statement, state)
    return True


def _ufunc_expansion_viable(ufunc_name: str, ufunc_method: Optional[str], arguments: List, keywords: dict,
                            data_arguments: set, state: LoweringState) -> bool:
    """
    Trial-run a ufunc replacement on a scratch SDFG, mirroring
    :func:`_expansion_viable` but for the ufunc calling convention (a single
    ``(ast_node, ufunc_name, args, kwargs)`` positional group instead of
    ``*args``/``**kwargs``) and its ``List[UfuncOutput]`` return form (a
    single-element list of the output dataname, rather than a bare string).
    """
    from dace.frontend.common import op_repository as oprepo

    function = oprepo.Replacements.get_ufunc(ufunc_method)
    scratch, scratch_state, shim = _replacement_trial_scratch(data_arguments, state)
    try:
        result = function(shim, None, scratch, scratch_state, ufunc_name, list(arguments), dict(keywords))
    except Exception:
        return False
    result = _unwrap_nested_call(result)
    if shim.views:
        return False  # View bindings are frontend state; expansion cannot defer them
    return isinstance(result, list) and len(result) == 1 and isinstance(result[0], str) and result[0] in scratch.arrays


def _replacement_attribute_argument(expression: ast.expr, state: LoweringState) -> Optional[str]:
    """
    The repository container an ATTRIBUTE argument of a replacement call names
    — a structure member (``numpy.dot(A.left, A.right)``) outright, or a
    descriptor attribute (``numpy.dot(x, w.T)``) once materialized.

    Registry implementations take container NAMES, so an attribute read left
    as written has no way through them; the same read bound to a name first
    (``t = w.T; numpy.dot(x, t)``) already lowered.

    :return: The container name, or None when the expression is not an
             attribute that names one.
    """
    if not isinstance(expression, ast.Attribute):
        return None
    access = resolve_access(expression, state)
    if access is None and supported_data_attribute(expression.attr):
        if state.emitter.in_dataflow_scope:
            # Materializing the read is a deferred replacement call, which is
            # not a legal node inside a scope (see
            # :func:`_materialize_attribute_reads`).
            return None
        base = resolve_access(expression.value, state)
        if base is not None:
            access = resolve_attribute_data(base, expression.attr, state)
    if access is None:
        return None
    descriptor = state.context.containers.get(access.container)
    if descriptor is None or isinstance(descriptor.dtype, dtypes.pyobject):
        return None
    if str(access.subset) != str(subsets.Range.from_array(descriptor)):
        return None  # A partial access is not the container the registry would read
    return access.container


def _replacement_arguments(call: ast.Call, state: LoweringState) -> Optional[Tuple[List, dict, set]]:
    """
    Resolve call arguments to the replacement invocation convention: data
    operands pass as repository container names, everything else as
    compile-time Python values.

    :return: (positional arguments, keyword arguments, data argument names),
             or None if any argument cannot be represented.
    """
    data_arguments = set()

    def convert(expression: ast.expr):
        if isinstance(expression, ast.Lambda):
            # Reduction combiners pass as source text, the convention both
            # frontends use (see semantics.inference.call_arguments).
            return True, astutils.unparse(expression)
        if isinstance(expression, ast.Name):
            binding = state.context.resolve(expression.id)
            if binding is not None and binding.kind == 'container':
                # Compile-time-valued temps (symbolic aliases) pass by value
                symbolic_value = state.context.symbolic_scalar_values.get(binding.container)
                if symbolic_value is not None:
                    return True, registry_argument_value(symbolic_value)
                descriptor = state.context.containers[binding.container]
                if (isinstance(descriptor.dtype, dtypes.pyobject)
                        and binding.container not in state.context.replacement_handles):
                    return False, None
                data_arguments.add(binding.container)
                return True, binding.container
        container = _replacement_attribute_argument(expression, state)
        if container is not None:
            data_arguments.add(container)
            return True, container
        try:
            value = state.inference.infer(expression)
        except UnsupportedFeatureError:
            return False, None
        if value.kind in ('constant', 'symbolic'):
            return True, registry_argument_value(value.value)
        if value.kind == 'static':
            # A static sequence's elements may themselves be data containers
            # (e.g. the ``(A, B)`` in ``numpy.concatenate((A, B))``): resolve
            # each element the same way as a top-level argument, so a
            # container name is recorded in ``data_arguments`` instead of
            # being rejected outright.
            elements = []
            for element in value.value.elements:
                ok, element_value = convert(element)
                if not ok:
                    return False, None
                elements.append(element_value)
            return True, tuple(elements) if value.value.kind == 'tuple' else elements
        return False, None  # Data-valued compound expressions need a name

    arguments = []
    for argument in call.args:
        ok, value = convert(argument)
        if not ok:
            return None
        arguments.append(value)
    keywords = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        ok, value = convert(keyword.value)
        if not ok:
            return None
        keywords[keyword.arg] = value
    return arguments, keywords, data_arguments


def _call_target_access(target: ast.expr, inferred, statement: ast.stmt, state: LoweringState) -> DataAccess:
    """Prepare the write target of a call result (allocating a container for
    fresh names from the registry-inferred descriptor)."""
    # Deferred import: rules.assign imports this module at load time
    from dace.frontend.python.nextgen.lowering.rules.assign import prepare_name_target
    if isinstance(target, ast.Name):
        return prepare_name_target(target, inferred, state, statement)
    access = resolve_access(target, state)
    if access is None:
        raise UnsupportedFeatureError('Unsupported call assignment target',
                                      state.context.filename,
                                      statement,
                                      category='assign-target')
    return access


def fallback_to_callback(statement: ast.stmt,
                         state: LoweringState,
                         reason: Union[str, Exception],
                         category: Optional[str] = None) -> None:
    """
    Wrap a statement in a fully specified Python callback.

    :param reason: Why the statement runs in the interpreter — either a plain
        string or the :class:`UnsupportedFeatureError` that triggered the
        fallback.
    :param category: Stable kebab-case gap category for callback provenance,
        rendered as a ``[category]`` prefix on the callback reason. A category
        carried by the ``reason`` exception (set at the raise site) takes
        precedence; without either, the reason is ``[uncategorized]``.
    """
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    resolved = getattr(reason, 'category', None) or category or 'uncategorized'
    reason_text = f'[{resolved}] {reason}'
    if isinstance(statement, ast.Return):
        _fallback_return(statement, state, reason_text)
        return
    reads, writes = statement_io_sets(statement)
    lower_opaque(OpaqueStmt(statement, reason_text, reads, writes), state)


def _fallback_return(statement: ast.Return, state: LoweringState, reason: str) -> None:
    """
    Fall back a ``return`` whose value cannot be lowered: a ``return`` cannot
    execute inside a Python callback, so the value computation runs in the
    interpreter as an assignment to the conventional return container(s),
    followed by a regular :class:`ReturnNode` naming them.
    """
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    if statement.value is None:
        state.emitter.emit(tn.ReturnNode())
        return

    prefix = state.context.return_prefix
    values = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
    names: List[str] = []
    for index, value in enumerate(values):
        base_name = '__return' if len(values) == 1 else f'__return_{index}'
        target_name = f'{prefix}{base_name}'
        assign = ast.copy_location(ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())], value=value),
                                   statement)
        ast.fix_missing_locations(assign)
        reads, writes = statement_io_sets(assign)
        lower_opaque(OpaqueStmt(assign, reason, reads, writes), state)
        names.append(state.context.resolve(target_name).container)
    state.context.return_names.extend(names)
    state.emitter.emit(tn.ReturnNode(values=names))


def _consumes_pyobject(value: ast.expr, state: LoweringState) -> bool:
    """Check whether any operand of a flat expression is an opaque Python object."""
    # A pyobject name that only appears as the BASE of a member read whose
    # member is real data (``holder.data[i]`` on a ``PythonClass``) is not an
    # operand: the member resolves to an ordinary container, and the object
    # itself is never read as a value. Flagging it here sent every access to
    # an analyzable Python object's fields to the interpreter.
    typed_member_bases = set()
    for node in ast.walk(value):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        try:
            access = resolve_access(node, state)
        except UnsupportedFeatureError:
            continue
        if access is not None and not isinstance(access.descriptor.dtype, dtypes.pyobject):
            typed_member_bases.add(id(node.value))

    for node in ast.walk(value):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in typed_member_bases:
            binding = state.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return True
    return False
