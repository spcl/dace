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
from dace.frontend.python.schedule_tree import structure_support
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering import dispatch
from dace.frontend.python.nextgen.lowering.access import DataAccess, nondegenerate_shape, resolve_access
from dace.frontend.python.nextgen.lowering.mechanisms import static_values
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

    if isinstance(value, ast.Call):
        from dace.frontend.python.nextgen.lowering.rules import calls
        calls.lower_call_assign(statement, state)
        return

    # Value-domain handling: sequence literals and operations on them that
    # fold at compile time bind statically without emitting nodes.
    try:
        inferred = state.inference.infer(value)
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, str(reason))
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
                                      state.context.filename, statement)


def _lower_name_assign(target: ast.Name, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                       state: LoweringState) -> None:
    # Names bound to Reference containers re-point at runtime: assignments are
    # reference sets, never binding updates.
    reference = _reference_binding(target.id, state)
    if reference is not None:
        _lower_reference_set(target.id, reference, value, inferred, statement, state)
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
        access = resolve_access(value, state)
        if access is not None and access.numpy_shape:
            _lower_view_binding(target, access, state)
            return

    target_access = prepare_name_target(target, inferred, state, statement)
    dispatch.lower_computation(target_access, value, statement, state)


def _lower_member_assign(target: ast.Attribute, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                         state: LoweringState) -> None:
    """
    Lower an assignment to a whole structure member (``tracers.vapor = ...``).

    Only :class:`~dace.data.Reference` members can be assigned as a whole —
    the assignment re-points the member (a :class:`RefSetNode`), matching SDFG
    reference semantics. Rebinding a non-reference member has no dataflow
    equivalent (write into a subset instead), and attributes that are not
    structure members are a feature gap; both degrade to the interpreter.
    """
    label = astutils.unparse(target)
    access = resolve_access(target, state)
    if access is None:
        raise UnsupportedFeatureError(f'Assignment to unsupported attribute "{label}"', state.context.filename,
                                      statement)
    if not isinstance(access.descriptor, data.Reference):
        raise UnsupportedFeatureError(
            f'Cannot rebind non-reference structure member "{label}" (write into a subset instead)',
            state.context.filename, statement)
    _lower_reference_set(label, (access.container, access.descriptor), value, inferred, statement, state)


def _lower_subscript_assign(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                            state: LoweringState) -> None:
    try:
        target_access = resolve_access(target, state)
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, str(reason))
        return
    if target_access is None:
        raise UnsupportedFeatureError(f'Assignment to unknown container "{astutils.unparse(target)}"',
                                      state.context.filename, statement)

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
                                              state.context.filename, statement)
            value_container = state.context.add_container(hint, descriptor)
            value_access = DataAccess(value_container, subsets.Range.from_array(descriptor), descriptor)
            dispatch.lower_computation(value_access, value, statement, state)
            access = value_access
        else:
            raise UnsupportedFeatureError(
                f'Cannot set reference "{label}" to non-data value "{astutils.unparse(value)}"', state.context.filename,
                statement)

    if access.is_scalar_access:
        raise UnsupportedFeatureError(f'Cannot set reference "{label}" to a scalar element', state.context.filename,
                                      statement)
    if list(nondegenerate_shape(access.subset)) != [s for s in reference_descriptor.shape if s != 1]:
        raise UnsupportedFeatureError(
            f'Reference "{label}" of shape {tuple(reference_descriptor.shape)} cannot be set '
            f'to source of shape {tuple(nondegenerate_shape(access.subset))}', state.context.filename, statement)

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
        if isinstance(descriptor, data.Array):
            return data.Array(descriptor.dtype, list(descriptor.shape))
        return data.Scalar(descriptor.dtype)
    dtype = inferred.dtype
    if dtype is None:
        dtype = dtypes.int64 if inferred.kind == 'symbolic' else dtypes.typeclass(type(inferred.value))
    return data.Scalar(dtype)


def _compatible(existing: data.Data, new: data.Data) -> bool:
    """In-place updates require same dtype and shape (scalars always match on dtype)."""
    if existing.dtype != new.dtype:
        return False
    if isinstance(existing, data.Scalar) and isinstance(new, data.Scalar):
        return True
    if isinstance(existing, data.Array) and isinstance(new, data.Array):
        return tuple(existing.shape) == tuple(new.shape) and not isinstance(existing, data.View)
    return False
