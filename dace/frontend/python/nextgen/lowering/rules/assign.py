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
- subset-to-subset copies (``B[2:4] = A[0:2]``) become :class:`CopyNode`.

Everything computational is delegated to the type-directed dispatch seam
(:mod:`~dace.frontend.python.nextgen.lowering.dispatch`), which selects a
mechanism by operand types (elementwise, materialization, callback).

Scalar Python names are deliberately materialized as scalar containers with
in-place writes (not interstate symbols), which keeps loop-carried updates
correct without SSA φ-resolution. Symbol promotion of compile-time scalars is
a planned optimization pass, not a correctness requirement.
"""
import ast

from dace import data, dtypes, subsets
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
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
        if isinstance(target, ast.Name):
            state.context.bind_static(target.id, inferred.value)
            return
        # Static value written into a container subset: materialize first
        access = static_values.materialize(inferred.value, state)
        value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), value)

    if isinstance(target, ast.Name):
        _lower_name_assign(target, value, inferred, statement, state)
    elif isinstance(target, ast.Subscript):
        _lower_subscript_assign(target, value, statement, state)
    else:
        raise UnsupportedFeatureError(f'Unsupported assignment target: {astutils.unparse(target)}',
                                      state.context.filename, statement)


def _lower_name_assign(target: ast.Name, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                       state: LoweringState) -> None:
    # Whole-array aliasing: rebind the name. Arrays are mutable in Python, so
    # both names must observe subsequent writes; a binding update models that.
    if isinstance(value, ast.Name):
        source_access = resolve_access(value, state)
        if source_access is not None and isinstance(source_access.descriptor, data.Array):
            state.context.bind(target.id, source_access.container)
            return

    # Slice reads bind as views so writes through the new name reach the
    # original container (NumPy basic-indexing semantics).
    if isinstance(value, ast.Subscript):
        access = resolve_access(value, state)
        if access is not None and not access.is_scalar_access:
            _lower_view_binding(target, access, state)
            return

    target_access = _prepare_name_target(target, inferred, state, statement)
    dispatch.lower_computation(target_access, value, statement, state)


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
    """Bind a slice read as a view container."""
    shape = nondegenerate_shape(access.subset)
    strides = [
        access.descriptor.strides[i] * step
        for i, (size, (_, _, step)) in enumerate(zip(access.subset.size(), access.subset.ranges)) if size != 1
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


def _prepare_name_target(target: ast.Name, inferred: Inferred, state: LoweringState,
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
