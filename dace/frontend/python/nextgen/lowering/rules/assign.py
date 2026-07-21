# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for canonical assignments.

The rule owns *binding semantics* — the decisions that follow Python and NumPy
assignment rules rather than computation:

- compile-time Python sequences (``a = [1, 2, x]``, concatenation, repetition,
  static indexing/slicing) stay in the value domain and emit no nodes,
- whole-array aliasing (``b = A``) is a binding update (arrays are mutable),
- slice reads (``b = A[1:4]``) bind as :class:`ViewNode` (NumPy basic-indexing
  view semantics),
- subset-to-subset copies (``B[2:4] = A[0:2]``) become :class:`CopyNode`,
- names declared as :class:`~dace.data.Reference` (via annotation) are runtime
  aliases: every assignment to them emits :class:`RefSetNode` instead of a
  binding update, and assigning *from* a reference-bound name materializes the
  target as a fresh reference (a pointer copy — a compile-time binding to a
  reference container would break when that reference is later re-set, e.g.
  in a double-buffering swap).

Everything computational is delegated to the type-directed dispatch seam
(:mod:`~dace.frontend.python.nextgen.lowering.dispatch`), which selects a
mechanism by operand types (elementwise, materialization, callback).

Scalar Python names are deliberately materialized as scalar containers with
in-place writes (not interstate symbols), which keeps loop-carried updates
correct without SSA φ-resolution. Symbol promotion of compile-time scalars is
a planned optimization pass, not a correctness requirement.
"""
import ast
import copy
from typing import Optional, Tuple

from dace import data, dtypes, subsets
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.semantics import structures as structure_support
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering import dispatch
from dace.frontend.python.nextgen.lowering.access import DataAccess, nondegenerate_shape, resolve_access
from dace.frontend.python.nextgen.lowering.mechanisms import conflict, static_values
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.inference import Inferred


@rule(ast.Assign)
def lower_assign(statement: ast.Assign, state: LoweringState) -> None:
    target = statement.targets[0]
    value = statement.value

    # Declared-type hint from a desugared annotated assignment: pre-bind the
    # target container so every lowering path (including callbacks whose
    # results inference cannot see) types it.
    if isinstance(target, ast.Name):
        apply_annotation_hint(target.id, statement, state)
        if isinstance(value, ast.Subscript) and _lower_boolean_gather_assign(target, value, statement, state):
            return

    if isinstance(value, ast.Call):
        from dace.frontend.python.nextgen.lowering.rules import calls
        calls.lower_call_assign(statement, state)
        return

    # Accumulation inside a dataflow scope: several iterations write the same
    # element, so the write carries conflict resolution and the tasklet drops
    # the self-read (``b[0] += x`` becomes ``b[0] (CR: Sum) = tasklet(x)``).
    wcr = conflict.accumulation_wcr(statement, state)
    if wcr is not None and _lower_accumulation(target, statement, wcr, state):
        return
    # A self-referential write canonicalization could not reduce to an
    # accumulation races here with no way to express the update as a WCR.
    conflict.report_unresolved(statement, target, state)

    # Value-domain handling: sequence literals and operations on them that
    # fold at compile time bind statically without emitting nodes.
    try:
        inferred = state.inference.infer(value)
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return
    if inferred.kind == 'static':
        if isinstance(target, ast.Name) and _reference_binding(target.id, state) is None:
            state.context.bind_static(target.id, inferred.value)
            return
        # Static value written into a container subset: materialize first
        access = static_values.materialize(inferred.value, state)
        value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), value)

    if isinstance(target, ast.Name):
        _lower_name_assign(target, value, inferred, statement, state)
    elif isinstance(target, ast.Attribute):
        _lower_member_assign(target, value, inferred, statement, state)
    elif isinstance(target, ast.Subscript):
        _lower_subscript_assign(target, value, statement, state)
    else:
        raise UnsupportedFeatureError(f'Unsupported assignment target: {astutils.unparse(target)}',
                                      state.context.filename,
                                      statement,
                                      category='assign-target')


def _lower_accumulation(target: ast.expr, statement: ast.Assign, wcr: str, state: LoweringState) -> bool:
    """
    Lower an accumulation as a conflict-resolved write of the accumulated value
    alone, dropping the self-read that the WCR subsumes. Returns False when the
    target does not resolve to a data access, so the caller falls through to the
    ordinary assignment paths.

    Falling through is what keeps the commuted form (``b = x + b``, detected by
    ``canonical/passes.py::DetectAccumulations``) safe for non-numeric values:
    swapping the operands of ``+`` would reverse a Python sequence
    concatenation, but a compile-time list or string never resolves to a
    container here, so the swap only ever happens on numeric data.
    """
    if not isinstance(target, (ast.Name, ast.Subscript, ast.Attribute)):
        return False
    if not isinstance(statement.value, ast.BinOp):
        return False
    try:
        target_access = resolve_access(target, state)
    except UnsupportedFeatureError:
        return False
    if target_access is None:
        return False
    # The accumulated value is whichever operand is not the self-read. A
    # desugared ``AugAssign`` carries no side marker and is always left-folded.
    accumulated = (statement.value.left
                   if getattr(statement, 'accumulator_side', 'left') == 'right' else statement.value.right)
    dispatch.lower_computation(target_access, accumulated, statement, state, wcr=wcr)
    return True


def _lower_name_assign(target: ast.Name, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                       state: LoweringState) -> None:
    # Names bound to Reference containers re-point at runtime: assignments are
    # reference sets, never binding updates.
    reference = _reference_binding(target.id, state)
    if reference is not None:
        _lower_reference_set(target.id, reference, value, inferred, statement, state)
        return

    # Registry ATTRIBUTE-family reads (``b = A.T``/``.real``/``.imag``/
    # ``.flat``) that need an actual data operation: tried before the
    # generic Name/Attribute aliasing path below (which only recognizes
    # structure members), the same placement principle as
    # ``dispatch._lower_reshape_call`` being tried before generic call
    # dispatch.
    if isinstance(value, ast.Attribute) and dispatch.lower_attribute_assign(target, value, state):
        return

    if isinstance(value, (ast.Name, ast.Attribute)):
        source_access = resolve_access(value, state)
        # Assigning from a reference-bound name materializes a fresh reference
        # (pointer copy): the source reference may later be re-set, so a
        # compile-time alias to its container would be unsound.
        if source_access is not None and isinstance(source_access.descriptor, data.Reference):
            reference_descriptor = copy.deepcopy(source_access.descriptor)
            container = state.context.add_container(target.id, reference_descriptor)
            state.context.bind(target.id, container)
            state.emitter.emit(
                tn.RefSetNode(target=container,
                              memlet=Memlet(data=source_access.container, subset=source_access.subset),
                              src_desc=source_access.descriptor,
                              ref_desc=copy.deepcopy(reference_descriptor)))
            return
        # Whole-structure aliasing (x = outer.inner, including the ANF temps
        # that reduce nested member chains to single-level datarefs): a
        # compile-time binding to the dotted member path. Sound because
        # structure members are embedded, never re-pointed.
        if source_access is not None and structure_support.supports_member_access(source_access.descriptor):
            state.context.bind(target.id, source_access.container)
            return
        # Whole-array aliasing: rebind the name. Arrays are mutable in Python,
        # so both names must observe subsequent writes; a binding update
        # models that.
        if source_access is not None and isinstance(source_access.descriptor, data.Array):
            state.context.bind(target.id, source_access.container)
            return

    # Slice reads bind as views so writes through the new name reach the
    # original container (NumPy basic-indexing semantics). The result shape
    # follows NumPy indexing: slice-formed dimensions survive (a[0:20, 1:2]
    # is (20, 1)), integer-indexed dimensions are dropped (a[0:20, 1] is
    # (20,)); a fully integer-indexed access is a scalar element read and
    # lowers as a computation instead.
    if isinstance(value, ast.Subscript):
        try:
            access = resolve_access(value, state)
        except UnsupportedFeatureError:
            # Not a directly resolvable access (e.g. an indirect index like
            # ``x[A_col[j]]``): falls through to the computation path below,
            # which lowers indirection through the elementwise mechanism.
            access = None
        if access is not None and access.numpy_shape:
            _lower_view_binding(target, access, state)
            return

    # Constants with no C representation (enum classes, type objects, ...)
    # cannot materialize as containers; they bind as compile-time values so
    # downstream attribute reads and call arguments resolve at compile time.
    if inferred.kind == 'constant' and not _representable_constant(inferred.value):
        state.context.bind_constant(target.id, inferred.value)
        return

    target_access = prepare_name_target(target, inferred, state, statement)
    # Pure-symbolic values of ANF temporaries stay visible to inference (e.g.
    # as computed shape arguments) alongside the materialized scalar. ANF
    # temps are single-assignment, so the recorded value cannot go stale.
    if inferred.kind == 'symbolic' and target.id.startswith('__anf'):
        state.context.symbolic_scalar_values[target_access.container] = inferred.value
    dispatch.lower_computation(target_access, value, statement, state)


def _lower_boolean_gather_assign(target: ast.Name, value: ast.Subscript, statement: ast.Assign,
                                 state: LoweringState) -> bool:
    """
    Lower ``B = A[mask]`` (NumPy boolean-mask indexing on the RHS). Returns
    False when ``value`` is not exactly this shape, so the caller falls
    through to the ordinary assignment paths -- this includes any OTHER
    boolean-mask use (nested in an expression, combined with other indices):
    only the bare top-level assignment of the whole access is implemented,
    everything else still reaches the existing ``analyze()``/
    ``has_boolean_index`` rejection unchanged.

    Tried BEFORE generic inference (``state.inference.infer(value)`` in
    :func:`lower_assign`) rather than after it, deliberately: inference has no
    correct way to type ``B`` before the element count is computed (it would
    otherwise silently answer with the mask's OWN shape, treating it like an
    ordinary same-shape index array), so this performs typing and lowering
    together instead, in :func:`advanced_indexing.emit_boolean_gather`.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if not isinstance(value.value, (ast.Name, ast.Attribute)):
        return False
    try:
        base = resolve_access(value.value, state)
        if base is None:
            return False
        expr = state.inference.parse_access(value)
        if not expr.arrdims or not advanced_indexing.has_boolean_index(expr, state.context):
            return False
        mask_container = advanced_indexing.resolve_single_boolean_mask(value, expr, base.container, state.context,
                                                                       state.inference)
        access = advanced_indexing.emit_boolean_gather(target.id, base.container, base.descriptor, mask_container,
                                                       statement, state)
        state.context.bind(target.id, access.container)
        return True
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return True


def _lower_member_assign(target: ast.Attribute, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                         state: LoweringState) -> None:
    """
    Lower an assignment to a whole structure member (``tracers.vapor = ...``).

    An EXISTING :class:`~dace.data.Reference` member can be assigned as a
    whole — the assignment re-points the member (a :class:`RefSetNode`),
    matching SDFG reference semantics. Rebinding an existing non-reference
    member has no dataflow equivalent (write into a subset instead) and
    degrades to the interpreter. A member that does not exist yet on a
    :class:`~dace.data.PythonClass` base is created dynamically (mirroring
    the classic frontend's ``_ensure_pythonclass_member``, see
    :func:`_create_pythonclass_member`); attributes that are neither an
    existing member nor a creatable one are a feature gap.
    """
    label = astutils.unparse(target)
    access = resolve_access(target, state)
    if access is not None:
        if not isinstance(access.descriptor, data.Reference):
            raise UnsupportedFeatureError(
                f'Cannot rebind non-reference structure member "{label}" (write into a subset instead)',
                state.context.filename,
                statement,
                category='structure-member')
        _lower_reference_set(label, (access.container, access.descriptor), value, inferred, statement, state)
        return

    created = _create_pythonclass_member(target, inferred, state)
    if created is None:
        raise UnsupportedFeatureError(f'Assignment to unsupported attribute "{label}"',
                                      state.context.filename,
                                      statement,
                                      category='structure-member')
    path, member_descriptor = created
    if isinstance(member_descriptor, data.Reference):
        _lower_reference_set(label, (path, member_descriptor), value, inferred, statement, state)
        return
    dispatch.lower_computation(DataAccess(path, subsets.Range.from_array(member_descriptor), member_descriptor), value,
                               statement, state)


def _create_pythonclass_member(target: ast.Attribute, inferred: Inferred,
                               state: LoweringState) -> Optional[Tuple[str, data.Data]]:
    """
    Dynamically create a new field on a :class:`~dace.data.PythonClass`
    structure, mirroring the classic frontend's dynamic attribute creation
    (``_ensure_pythonclass_member``): an array-valued assignment creates a
    :class:`~dace.data.Reference` member (set via :class:`RefSetNode`, a
    pointer copy on every subsequent assignment); any other value creates a
    plain member of the value's own type (written by a normal
    copy/computation). Members are non-transient (their storage belongs to
    the enclosing object, not an SDFG-scoped temporary). Other structure
    kinds (plain :class:`~dace.data.Structure`) have a fixed member set and
    do not support this.

    :return: The (dotted member path, registered descriptor) pair, or None if
             the target's base is not a name bound to a ``PythonClass`` or the
             value's descriptor cannot be determined.
    """
    if not isinstance(target.value, ast.Name):
        return None
    binding = state.context.resolve(target.value.id)
    if binding is None or binding.kind != 'container':
        return None
    base_container = binding.container
    base_descriptor = state.context.containers.get(base_container)
    if not isinstance(base_descriptor, data.PythonClass):
        return None
    member = target.attr
    if member in base_descriptor.members:
        return None  # Already a member: the normal resolve_access path handles it

    if inferred.is_data:
        member_descriptor = (data.Reference.view(inferred.descriptor)
                             if isinstance(inferred.descriptor, data.Array) else copy.deepcopy(inferred.descriptor))
    else:
        dtype = state.inference.dtype_of(inferred)
        if dtype is None:
            try:
                dtype = dtypes.typeclass(type(inferred.value))
            except (KeyError, TypeError):
                return None
        member_descriptor = data.Scalar(dtype)
    member_descriptor.transient = False
    base_descriptor.members[member] = member_descriptor
    return structure_support.structure_member_path(base_container, member), member_descriptor


def _lower_subscript_assign(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                            state: LoweringState) -> None:
    # ``A.flat[idx] = ...`` on a contiguous array: rewrite the base to the
    # materialized flat-view container (see
    # ``dispatch.rewrite_flat_subscript_base``) so the ordinary machinery
    # below, which only resolves a plain-name or structure-member base,
    # can write through it -- NumPy's flatiter aliases the source, so this
    # must reach ``A``, not a disposable copy.
    target = dispatch.rewrite_flat_subscript_base(target, state)
    if _lower_advanced_index_write(target, value, statement, state):
        return
    try:
        target_access = resolve_access(target, state)
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return
    if target_access is None:
        raise UnsupportedFeatureError(f'Assignment to unknown container "{astutils.unparse(target)}"',
                                      state.context.filename,
                                      statement,
                                      category='undefined-name')

    # Subset-to-subset copy
    if isinstance(value, (ast.Name, ast.Subscript)):
        source_access = resolve_access(value, state)
        if (source_access is not None and not source_access.is_scalar_access and not target_access.is_scalar_access
                and nondegenerate_shape(source_access.subset) == nondegenerate_shape(target_access.subset)):
            state.emitter.emit(
                tn.CopyNode(target=target_access.container,
                            memlet=Memlet(data=source_access.container,
                                          subset=source_access.subset,
                                          other_subset=target_access.subset)))
            return

    dispatch.lower_computation(target_access, value, statement, state)


def _lower_advanced_index_write(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                                state: LoweringState) -> bool:
    """
    Lower a write through NumPy advanced indexing (``A[indices] = B``) as a
    scatter. Returns False when the target uses no array-valued index, so the
    caller falls through to the ordinary subscript-assignment paths.

    An accumulation always takes conflict resolution here regardless of scope:
    the index array may name the same element twice (``A[[0, 0]] += 1``), so the
    map iterations collide on data the frontend cannot inspect. That is a
    stronger rule than :func:`conflict.accumulation_wcr` applies elsewhere,
    where collision is decided from the subset.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if not isinstance(target.value, (ast.Name, ast.Attribute)):
        return False
    try:
        base = resolve_access(target.value, state)
        if base is None:
            return False
        expr = state.inference.parse_access(target)
        if not expr.arrdims:
            return False
        # An accumulation reaches lowering desugared into ``t = <t read> op v``;
        # both write paths re-apply the operator themselves, so strip the
        # self-read down to the accumulated operand first.
        operator = getattr(statement, 'augmented_op', None)
        symbol = conflict.WCR_OPERATORS.get(type(operator)) if operator is not None else None
        accumulated = value
        if symbol is not None and isinstance(value, ast.BinOp):
            accumulated = (value.left if getattr(statement, 'accumulator_side', 'left') == 'right' else value.right)

        if advanced_indexing.has_boolean_index(expr, state.context):
            # A mask writes in place through a guarded update, so it needs
            # neither a gather nor conflict resolution.
            advanced_indexing.emit_masked_write(target, expr, base.container, base.descriptor, accumulated, statement,
                                                state)
            return True
        access = advanced_indexing.analyze(target, expr, base.container, base.descriptor, state.context,
                                           state.inference)
        wcr = f'lambda x, y: x {symbol} y' if symbol is not None else None
        advanced_indexing.emit_scatter(access, accumulated, statement, state, wcr=wcr)
        return True
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return True


def _lower_view_binding(target: ast.Name, access: DataAccess, state: LoweringState) -> None:
    """Bind a slice read as a view container with its NumPy result shape."""
    shape = access.numpy_shape
    selected = access.kept_dims if access.kept_dims is not None else [size != 1 for size in access.subset.size()]
    strides = [
        access.descriptor.strides[i] * step
        for i, (keep, (_, _, step)) in enumerate(zip(selected, access.subset.ranges)) if keep
    ] if isinstance(access.descriptor, data.Array) else None
    view_descriptor = data.ArrayView(access.descriptor.dtype, shape, strides=strides)
    view_name = state.context.add_container(target.id, view_descriptor)
    state.context.bind(target.id, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=access.container,
                    memlet=Memlet(data=access.container, subset=access.subset),
                    src_desc=access.descriptor,
                    view_desc=view_descriptor))


@rule(ast.AnnAssign)
def lower_declaration(statement: ast.AnnAssign, state: LoweringState) -> None:
    """
    Lower a bare declaration (``x: T`` without a value): register the declared
    descriptor so the first assignment to the name adopts it (the primary use
    is Reference declarations, e.g. ``tmp: dace.data.ArrayReference(...)``).
    Unresolvable annotations are no-ops, matching the classic frontend.
    """
    apply_annotation_hint(statement.target.id, statement, state)


def _reference_binding(name: str, state: LoweringState) -> Optional[Tuple[str, data.Data]]:
    """The (container, descriptor) pair of a name bound to a Reference
    container, or None."""
    binding = state.context.resolve(name)
    if binding is None or binding.kind != 'container':
        return None
    descriptor = state.context.containers[binding.container]
    if isinstance(descriptor, data.Reference):
        return binding.container, descriptor
    return None


def _lower_reference_set(label: str, reference: Tuple[str, data.Data], value: ast.expr, inferred: Inferred,
                         statement: ast.Assign, state: LoweringState) -> None:
    """
    Emit a :class:`RefSetNode` re-pointing a reference container (a
    reference-bound name or a reference-typed structure member). Direct
    container sources re-point in place; computed values materialize into a
    fresh container first and the reference is set to it.

    :param label: Source-level label of the reference (for diagnostics and
                  value-container name hints).
    """
    container, reference_descriptor = reference
    hint = f"__{label.replace('.', '_')}_value"

    access = None
    if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
        access = resolve_access(value, state)
    if access is None:
        if inferred.kind == 'static':
            access = static_values.materialize(inferred.value, state, name_hint=hint)
        elif inferred.is_data:
            descriptor = _result_descriptor(inferred, state, statement)
            if not isinstance(descriptor, data.Array):
                raise UnsupportedFeatureError(f'Cannot set reference "{label}" to a scalar value',
                                              state.context.filename,
                                              statement,
                                              category='reference-set')
            value_container = state.context.add_container(hint, descriptor)
            value_access = DataAccess(value_container, subsets.Range.from_array(descriptor), descriptor)
            dispatch.lower_computation(value_access, value, statement, state)
            access = value_access
        else:
            raise UnsupportedFeatureError(
                f'Cannot set reference "{label}" to non-data value '
                f'"{astutils.unparse(value)}"',
                state.context.filename,
                statement,
                category='reference-set')

    if access.is_scalar_access:
        raise UnsupportedFeatureError(f'Cannot set reference "{label}" to a scalar element',
                                      state.context.filename,
                                      statement,
                                      category='reference-set')
    if list(nondegenerate_shape(access.subset)) != [s for s in reference_descriptor.shape if s != 1]:
        raise UnsupportedFeatureError(
            f'Reference "{label}" of shape {tuple(reference_descriptor.shape)} cannot be set '
            f'to source of shape {tuple(nondegenerate_shape(access.subset))}',
            state.context.filename,
            statement,
            category='reference-set')

    state.emitter.emit(
        tn.RefSetNode(target=container,
                      memlet=Memlet(data=access.container, subset=access.subset),
                      src_desc=access.descriptor,
                      ref_desc=copy.deepcopy(reference_descriptor)))


def apply_annotation_hint(target_name: str, statement: ast.stmt, state: LoweringState) -> None:
    """
    Pre-register an annotated assignment target (``y: dace.float64 = ...``)
    with its declared descriptor. Applies only to names with no container
    binding yet; unresolvable annotations are ignored (inference decides).
    """
    descriptor = annotation_descriptor(getattr(statement, 'annotation', None), state)
    if descriptor is None:
        return
    binding = state.context.resolve(target_name)
    if binding is not None and binding.kind == 'container':
        return
    container = state.context.add_container(target_name, descriptor)
    state.context.bind(target_name, container)


def annotation_descriptor(annotation: Optional[ast.expr], state: LoweringState) -> Optional[data.Data]:
    """The data descriptor a type annotation declares, or None if the
    annotation is absent or does not resolve to a typed descriptor."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant):
        value = annotation.value
    else:
        try:
            value = astutils.evalnode(annotation, _annotation_environment(state))
        except Exception:
            return None
    try:
        descriptor = data.create_datadescriptor(value)
    except Exception:
        return None
    if not isinstance(descriptor, data.Data) or descriptor.dtype is None or descriptor.dtype.type is None:
        return None
    if isinstance(descriptor.dtype, dtypes.pyobject):
        return None
    descriptor = copy.deepcopy(descriptor)
    descriptor.transient = True
    return descriptor


def _annotation_environment(state: LoweringState) -> dict:
    """The evaluation environment for type annotations: program globals plus
    the descriptors of bound containers, so annotations may reference argument
    properties (``dace.data.ArrayReference(A.dtype, A.shape)``)."""
    environment = dict(state.context.globals)
    for name, binding in state.context.bindings.items():
        if binding.kind == 'container' and binding.container in state.context.containers:
            environment[name] = state.context.containers[binding.container]
    return environment


def prepare_name_target(target: ast.Name, inferred: Inferred, state: LoweringState,
                        statement: ast.Assign) -> DataAccess:
    """
    Return the target access for a name assignment, reusing the currently
    bound container when it is compatible (in-place update) or registering a
    new one otherwise.
    """
    result_descriptor = _result_descriptor(inferred, state, statement)

    binding = state.context.resolve(target.id)
    if binding is not None and binding.kind == 'container':
        existing = state.context.containers[binding.container]
        if _compatible(existing, result_descriptor):
            return DataAccess(binding.container, subsets.Range.from_array(existing), existing)

    container_name = state.context.add_container(target.id, result_descriptor)
    state.context.bind(target.id, container_name)
    return DataAccess(container_name, subsets.Range.from_array(result_descriptor), result_descriptor)


def _result_descriptor(inferred: Inferred, state: LoweringState, statement: ast.Assign) -> data.Data:
    if inferred.kind == 'static':
        return state.inference.sequence_descriptor(inferred.value)
    if inferred.is_data:
        descriptor = inferred.descriptor
        if isinstance(descriptor, data.Stream):
            # Streams keep their full descriptor (buffer size etc.)
            return copy.deepcopy(descriptor)
        if isinstance(descriptor, data.Array):
            return data.Array(descriptor.dtype,
                              list(descriptor.shape),
                              storage=descriptor.storage,
                              lifetime=descriptor.lifetime)
        return data.Scalar(descriptor.dtype, storage=descriptor.storage, lifetime=descriptor.lifetime)
    dtype = state.inference.dtype_of(inferred)
    if dtype is None:
        if inferred.kind == 'symbolic':
            dtype = dtypes.int64
        else:
            try:
                dtype = dtypes.typeclass(type(inferred.value))
            except (KeyError, TypeError):
                # Constants of non-C-representable types (enum classes, ...)
                raise UnsupportedFeatureError(
                    f'Cannot represent constant of type {type(inferred.value).__name__} in a container',
                    node=statement,
                    category='type-inference')
    return data.Scalar(dtype)


def _representable_constant(value) -> bool:
    """Whether a constant value's type has a C container representation."""
    try:
        dtypes.typeclass(type(value))
        return True
    except (KeyError, TypeError):
        return False


def _compatible(existing: data.Data, new: data.Data) -> bool:
    """In-place updates require same dtype and shape (scalars always match on dtype)."""
    if existing.dtype != new.dtype:
        return False
    if isinstance(existing, data.Scalar) and isinstance(new, data.Scalar):
        return True
    if isinstance(existing, data.Array) and isinstance(new, data.Array):
        return tuple(existing.shape) == tuple(new.shape) and not isinstance(existing, data.View)
    return False
