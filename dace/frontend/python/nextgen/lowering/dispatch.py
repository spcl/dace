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
from typing import List, Optional, Tuple, Union

from dace import data, dtypes, subsets
from dace.memlet import Memlet
from dace.frontend.python import astutils
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt, statement_io_sets
from dace.frontend.python.nextgen.common import SUPPORTED_DATA_ATTRIBUTES, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.lowering.mechanisms import creation, elementwise, reduction, static_values

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


def _replacement_registered(name: str) -> bool:
    """
    Whether a call is eligible for deferred replacement expansion
    (``tree_to_sdfg.visit_ReplacementCallNode``): it must have BOTH a
    registered replacement (the expansion body) and a descriptor inference
    entry (the frontend must type the result to allocate the target
    container). Replacements needing ``ProgramVisitor`` machinery beyond the
    expansion shim's surface fail loudly at expansion time.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get(name) is None:
        return False
    return oprepo.Replacements.get_descriptor_inference(name) is not None


def _method_registered(descriptor: data.Data, method_name: str) -> bool:
    """
    Whether a bound-method call is eligible for deferred replacement
    expansion, mirroring :func:`_replacement_registered` for the ``_method_rep``
    keyspace (:meth:`Replacements.get_method` /
    :meth:`Replacements.get_method_descriptor_inference`).
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get_method(type(descriptor), method_name) is None:
        return False
    return oprepo.Replacements.get_method_descriptor_inference(type(descriptor), method_name) is not None


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
        value = _materialize_advanced_indices(value, statement, state)
        rewritten = static_values.materialize_operands(value, state)
        elementwise.emit_computation(target, rewritten, statement, state, wcr=wcr)
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, reason)


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
    expr = state.inference.parse_access(value)
    if not expr.arrdims:
        return None
    return advanced_indexing.analyze(value, expr, base.container, base.descriptor, state.context, state.inference)


def _materialize_advanced_indices(value: ast.expr, statement: ast.stmt, state: LoweringState) -> ast.expr:
    """
    Gather every advanced-indexing subscript nested inside an expression into a
    temporary container, returning the expression rewritten to read the
    temporaries.

    ANF leaves these in operand position -- a data subscript is a legal operand,
    and canonicalization has no type information with which to tell
    ``A[scalar_i]`` from ``A[index_array]`` -- so the split has to happen here,
    where the index's descriptor is known.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if isinstance(value, ast.Subscript):
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

    class _Substituter(ast.NodeTransformer):

        def visit_Subscript(self, subscript: ast.Subscript) -> ast.AST:
            for original, replacement in replacements:
                if original is subscript:
                    return replacement
            return self.generic_visit(subscript)

    return _Substituter().visit(value)


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
    :func:`_expansion_viable` rejects any replacement that records a view (a
    view binding is frontend-visible state, not something a deferred
    ``ReplacementCallNode`` can represent).

    Returns False when the call is not a recognized/resolvable reshape form
    (target not a name, base not a data access, shape not compile-time
    integers, or element count mismatch); the caller then falls through to
    the normal registry-call dispatch and, ultimately, a callback.
    """
    if not isinstance(target, ast.Name):
        return False
    base_expr, shape_args = _reshape_operands(call, qualname)
    if base_expr is None:
        return False
    access = resolve_access(base_expr, state)
    if access is None:
        return False
    shape = _reshape_shape(shape_args, access.subset, state)
    if shape is None:
        return False
    view_descriptor = data.ArrayView(access.descriptor.dtype, shape)
    view_name = state.context.add_container(target.id, view_descriptor)
    state.context.bind(target.id, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=access.container,
                    memlet=Memlet(data=access.container, subset=access.subset),
                    src_desc=access.descriptor,
                    view_desc=view_descriptor))
    return True


def _reshape_operands(call: ast.Call, qualname: str) -> Tuple[Optional[ast.expr], List[ast.expr]]:
    """
    The (base array expression, shape argument expressions) of a reshape call
    form, or (None, []) if the call is not one of the recognized forms:
    ``x.reshape(shape)``, ``x.reshape(d0, d1, ...)``, or
    ``numpy.reshape(x, shape)``. The shape argument expressions are returned
    as-is (not flattened) — ANF hoists a literal shape tuple to a name bound
    to a static sequence, so :func:`_reshape_shape` resolves through
    inference rather than requiring an inline ``ast.Tuple``/``ast.List``.
    """
    if isinstance(call.func, ast.Attribute) and call.func.attr == 'reshape' and not call.keywords:
        return call.func.value, call.args
    if qualname == 'numpy.reshape' and not call.keywords and call.args:
        return call.args[0], call.args[1:]
    return None, []


def _reshape_shape(shape_args: List[ast.expr], source_subset, state: LoweringState) -> Optional[List]:
    """
    Resolve reshape target-shape arguments to concrete dimension sizes,
    filling in at most one ``-1`` placeholder dimension from the source's
    total element count (NumPy semantics). None when a dimension is not a
    compile-time integer or the requested shape's element count does not
    match the source's (the caller degrades to a callback).
    """
    dims: List[int] = []
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
            try:
                dims = [int(element) for element in state.inference.sequence_constants(inferred.value)]
            except (UnsupportedFeatureError, TypeError, ValueError):
                return None
    if not dims:
        for arg in shape_args:
            value = state.inference.constant_int(arg)
            if value is None:
                return None
            dims.append(value)
    if not dims:
        return None
    total = source_subset.num_elements()
    try:
        known_product = 1
        placeholder = None
        for index, dim in enumerate(dims):
            if dim == -1:
                if placeholder is not None:
                    return None
                placeholder = index
            else:
                known_product *= dim
        if placeholder is not None:
            if known_product == 0 or bool(total % known_product != 0):
                return None
            dims[placeholder] = total // known_product
        elif bool(known_product != total):
            return None
    except Exception:
        return None
    return dims


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

    :return: The resolved access, or None when ``attr_name`` is not in
             :data:`~dace.frontend.python.nextgen.common.SUPPORTED_DATA_ATTRIBUTES`,
             or the registry call is not viable here (the caller falls back
             to a callback).
    """
    if attr_name not in SUPPORTED_DATA_ATTRIBUTES:
        return None
    if attr_name == 'flat' and isinstance(base.descriptor, data.Array) and _is_contiguous_flat(base.descriptor):
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
    ``(classname, attr_name)`` instead of a free-function qualname)."""
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    typename = type(base.descriptor).__name__
    function = oprepo.Replacements.get_attribute(typename, attr_name)
    if function is None:
        return None
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
    if isinstance(result, tuple) and len(result) == 2 and type(result[0]).__name__ == 'NestedCall':
        result = result[1]
    if shim.views or not isinstance(result, str) or result not in scratch.arrays:
        return None
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


def rewrite_flat_subscript_base(target: ast.Subscript, state: LoweringState) -> ast.Subscript:
    """
    Rewrite a ``A.flat[...]`` subscript target to reference the materialized
    flat-view container directly, so the ordinary subscript-assignment
    machinery (``rules.assign._lower_subscript_assign``/``access.resolve_access``,
    neither of which know about the ATTRIBUTE registry) can resolve it.

    Only the CONTIGUOUS case is rewritten: NumPy's flatiter is an aliasing
    view there (see :func:`resolve_attribute_data`), so a write through it
    must reach the source array. A non-contiguous ``.flat`` computes a
    disposable copy -- subscripting it as a write target would silently
    discard the write, so it is deliberately left unrewritten and falls
    through to the ordinary (and, for this form, unsupported) paths, which
    degrade to a callback instead of miscompiling.

    Returns ``target`` unchanged when it is not a rewritable ``.flat`` base.
    """
    if not isinstance(target.value, ast.Attribute) or target.value.attr != 'flat':
        return target
    base = resolve_access(target.value.value, state)
    if base is None or not isinstance(base.descriptor, data.Array) or not _is_contiguous_flat(base.descriptor):
        return target
    access = resolve_attribute_data(base, 'flat', state)
    if access is None:
        return target
    rewritten = copy.copy(target)
    rewritten.value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), target.value)
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


def _lower_registry_call(target: Optional[ast.expr], call: ast.Call, qualname: str, callee: object, statement: ast.stmt,
                         state: LoweringState) -> bool:
    """
    Try to lower a call through the descriptor-inference registry and the
    NumPy mechanisms. Returns False when no mechanism applies (the caller
    falls back to a callback).
    """
    inferred = state.inference.infer_call(call)
    if inferred is None or not inferred.is_data or target is None:
        # No registry entry, a multi-output result, or an unused result:
        # the interpreter fallback preserves semantics in all three cases.
        return False

    # NumPy universal functions, direct (numpy.add(...)) or through one of
    # their reduce/accumulate/outer methods (numpy.add.reduce(...)). A direct
    # call with no method lowers through the lightweight elementwise
    # mechanism (a single tasklet expression); reduce/accumulate/outer need
    # real reduction/scan/broadcast dataflow, which the elementwise mechanism
    # cannot express, so they defer to the actual registry ufunc
    # implementation through the same deferred-expansion mechanism used for
    # other replacements below.
    ufunc_form = state.inference.resolve_ufunc_call(call)
    if ufunc_form is not None:
        ufunc, ufunc_method = ufunc_form
        if ufunc_method is None:
            if call.keywords:
                return False  # out=/where=/dtype= change semantics; run in the interpreter
            target_access = _call_target_access(target, inferred, statement, state)
            elementwise.emit_ufunc(target_access, ufunc.__name__, call.args, statement, state)
            return True
        return _lower_ufunc_replacement_call(target, call, ufunc.__name__, ufunc_method, inferred, statement, state)

    # Array/stream creation. Resolution may yield the callee's real module
    # path (dace.define_stream lives in dace.frontend.python.wrappers), so
    # fall back to the source-level name: the qualname preprocessing attaches
    # to embedded callee constants, or the textual call name.
    creation_name = qualname
    if creation_name not in creation.CREATION_CALLS:
        creation_name = getattr(call.func, 'qualname', None) or astutils.rname(call.func)
    if creation_name in creation.CREATION_CALLS:
        if any(keyword.arg not in ('dtype', 'fill_value', 'shape', 'storage', 'lifetime', 'buffer_size')
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

    # Deferred replacement expansion: the descriptor inference typed the call,
    # so vetted registry functions emit a ReplacementCallNode that
    # tree_to_sdfg expands through the classic replacement implementation.
    if _lower_replacement_call(target, call, qualname, inferred, statement, state):
        return True

    return False


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


def _lower_replacement_call(target: ast.expr, call: ast.Call, qualname: str, inferred, statement: ast.stmt,
                            state: LoweringState) -> bool:
    """
    Emit a deferred :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`
    for a vetted registry replacement (free function or bound method).
    Returns False when the call is not vetted or an argument cannot be
    resolved (the caller falls back).
    """
    name = qualname
    receiver: Optional[str] = None
    if not _replacement_registered(name):
        name = getattr(call.func, 'qualname', None) or astutils.rname(call.func)
        if not _replacement_registered(name):
            name = None
            # Not a registered free function: try the method family
            # (``_method_rep``), e.g. ``A.copy()``/``A.fill(0)``.
            receiver_access = _method_call_receiver(call, state)
            if receiver_access is not None and _method_registered(receiver_access.descriptor, call.func.attr):
                receiver = receiver_access.container
                name = call.func.attr
            if name is None:
                return False
    if state.emitter.in_dataflow_scope:
        return False  # Expansion adds state machinery; no CFG inside dataflow scopes

    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if receiver is not None:
        # Mirror the classic frontend's convention (newast.py's Call
        # visitor): the receiver is the replacement's first positional
        # argument.
        arguments = [receiver] + arguments
        data_arguments = data_arguments | {receiver}
    if not _expansion_viable(name, arguments, keywords, data_arguments, state, receiver=receiver):
        return False
    target_access = _call_target_access(target, inferred, statement, state)
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=name,
                               target=target_access.container,
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments,
                               receiver=receiver))
    return True


def _replacement_trial_scratch(data_arguments: set, state: LoweringState):
    """
    Build the scratch SDFG/state/shim triple shared by the build-time
    replacement-viability trials (:func:`_expansion_viable` and
    :func:`_ufunc_expansion_viable`): a standalone SDFG carrying copies of
    just the data arguments a replacement call touches, on which the
    replacement can be trial-run without mutating the real program.
    """
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import ReplacementVisitorShim
    from dace.sdfg.sdfg import SDFG

    scratch = SDFG('__replacement_viability')
    for data_name in data_arguments:
        descriptor = copy.deepcopy(state.context.containers[data_name])
        descriptor.transient = False
        scratch.add_datadesc(data_name, descriptor)
    scratch_state = scratch.add_state()
    shim = ReplacementVisitorShim(scratch, scratch_state, '__viability_target')
    return scratch, scratch_state, shim


def _expansion_viable(name: str,
                      arguments: List,
                      keywords: dict,
                      data_arguments: set,
                      state: LoweringState,
                      receiver: Optional[str] = None) -> bool:
    """
    Trial-run a replacement on a scratch SDFG to decide, at tree-build time
    (where the graceful fallback is a callback), whether deferred expansion
    will succeed at tree-to-SDFG time (where failure would be a hard error).
    Runs the exact code path of ``tree_to_sdfg.visit_ReplacementCallNode``:
    non-viable outcomes are exceptions, recorded view bindings, and
    unsupported return forms.
    """
    from dace.frontend.common import op_repository as oprepo

    if receiver is not None:
        function = oprepo.Replacements.get_method(type(state.context.containers[receiver]), name)
    else:
        function = oprepo.Replacements.get(name)
    scratch, scratch_state, shim = _replacement_trial_scratch(data_arguments, state)
    try:
        result = function(shim, scratch, scratch_state, *copy.deepcopy(arguments), **copy.deepcopy(keywords))
    except Exception:
        return False
    if isinstance(result, tuple) and len(result) == 2 and type(result[0]).__name__ == 'NestedCall':
        result = result[1]
    if shim.views:
        return False  # View bindings are frontend state; expansion cannot defer them
    if isinstance(result, str):
        return result in scratch.arrays
    return result is None or result == []


#: Keyword arguments the registry ufunc implementations
#: (:mod:`dace.frontend.python.replacements.ufunc`) understand, across all of
#: the direct-call/reduce/accumulate/outer forms. Anything else changes
#: semantics in a way the registry does not implement, so it is rejected
#: rather than silently ignored.
_SUPPORTED_UFUNC_KEYWORDS = frozenset({'out', 'where', 'axis', 'keepdims', 'initial', 'dtype'})


def _lower_ufunc_replacement_call(target: ast.expr, call: ast.Call, ufunc_name: str, ufunc_method: Optional[str],
                                  inferred, statement: ast.stmt, state: LoweringState) -> bool:
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
    if state.emitter.in_dataflow_scope:
        return False  # Expansion adds state machinery; no CFG inside dataflow scopes

    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if not _ufunc_expansion_viable(ufunc_name, ufunc_method, arguments, keywords, data_arguments, state):
        return False
    target_access = _call_target_access(target, inferred, statement, state)
    display_name = f'numpy.{ufunc_name}' + (f'.{ufunc_method}' if ufunc_method else '')
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=display_name,
                               target=target_access.container,
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments,
                               ufunc_name=ufunc_name,
                               ufunc_method=ufunc_method))
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
        result = function(shim, None, scratch, scratch_state, ufunc_name, copy.deepcopy(list(arguments)),
                          copy.deepcopy(dict(keywords)))
    except Exception:
        return False
    if isinstance(result, tuple) and len(result) == 2 and type(result[0]).__name__ == 'NestedCall':
        result = result[1]
    if shim.views:
        return False  # View bindings are frontend state; expansion cannot defer them
    return isinstance(result, list) and len(result) == 1 and isinstance(result[0], str) and result[0] in scratch.arrays


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
        if isinstance(expression, ast.Name):
            binding = state.context.resolve(expression.id)
            if binding is not None and binding.kind == 'container':
                # Compile-time-valued temps (symbolic aliases) pass by value
                symbolic_value = state.context.symbolic_scalar_values.get(binding.container)
                if symbolic_value is not None:
                    return True, symbolic_value
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return False, None
                data_arguments.add(binding.container)
                return True, binding.container
        try:
            value = state.inference.infer(expression)
        except UnsupportedFeatureError:
            return False, None
        if value.kind in ('constant', 'symbolic'):
            return True, value.value
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
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            binding = state.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return True
    return False
