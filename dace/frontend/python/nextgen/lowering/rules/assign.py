# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical assignments.

Handles the canonical ``Assign`` forms produced by the ANF pass:

- whole-array aliasing (``b = A``) as binding updates,
- slice reads (``b = A[1:4]``) as :class:`ViewNode`,
- subset-to-subset copies (``B[2:4] = A[0:2]``) as :class:`CopyNode`,
- everything else (scalar loads/stores, flat operator expressions) as
  :class:`TaskletNode`, wrapped in a :class:`MapScope` for array-shaped
  results.

Scalar Python names are deliberately materialized as scalar containers with
in-place writes (not interstate symbols), which keeps loop-carried updates
correct without SSA φ-resolution. Symbol promotion of compile-time scalars is
a planned optimization pass, not a correctness requirement.
"""
import ast
from typing import List

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import (DataAccess, indexed_subset, resolve_access,
                                                          substitute_data_operands)
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

    if isinstance(target, ast.Name):
        _lower_name_assign(target, value, statement, state)
    elif isinstance(target, ast.Subscript):
        _lower_subscript_assign(target, value, statement, state)
    else:
        raise UnsupportedFeatureError(f'Unsupported assignment target: {astutils.unparse(target)}',
                                      state.context.filename, statement)


def _lower_name_assign(target: ast.Name, value: ast.expr, statement: ast.Assign, state: LoweringState) -> None:
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

    inferred = state.inference.infer(value)
    target_access = _prepare_name_target(target, inferred, state, statement)
    _emit_computation(target_access, value, statement, state)


def _lower_subscript_assign(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                            state: LoweringState) -> None:
    target_access = resolve_access(target, state)
    if target_access is None:
        raise UnsupportedFeatureError(f'Assignment to unknown container "{astutils.unparse(target)}"',
                                      state.context.filename, statement)

    # Subset-to-subset copy
    if isinstance(value, (ast.Name, ast.Subscript)):
        source_access = resolve_access(value, state)
        if (source_access is not None and not source_access.is_scalar_access and not target_access.is_scalar_access
                and _nondegenerate_shape(source_access.subset) == _nondegenerate_shape(target_access.subset)):
            state.emitter.emit(
                tn.CopyNode(target=target_access.container,
                            memlet=Memlet(data=source_access.container,
                                          subset=source_access.subset,
                                          other_subset=target_access.subset)))
            return

    _emit_computation(target_access, value, statement, state)


def _lower_view_binding(target: ast.Name, access: DataAccess, state: LoweringState) -> None:
    """Bind a slice read as a view container."""
    shape = _nondegenerate_shape(access.subset)
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


def _nondegenerate_shape(subset: subsets.Range) -> List:
    return [s for s in subset.size() if s != 1]


def _emit_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes a canonical flat expression into the target access.
    """
    code, operands = substitute_data_operands(value, state)
    line = getattr(statement, 'lineno', 0)
    result_shape = _nondegenerate_shape(target.subset)

    if not result_shape:
        # Scalar result: single tasklet
        tasklet = nodes.Tasklet(f'assign_{line}', {connector
                                                   for connector, _ in operands}, {'__out'}, f'__out = {code}')
        in_memlets = {connector: Memlet(data=access.container, subset=access.subset) for connector, access in operands}
        out_memlets = {'__out': Memlet(data=target.container, subset=target.subset)}
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
        return

    # Array result: elementwise map
    params = [f'__i{i}' for i in range(len(result_shape))]
    map_range = subsets.Range([(0, size - 1, 1) for size in result_shape])
    map_node = nodes.MapEntry(nodes.Map(f'map_{line}', params, map_range))
    tasklet = nodes.Tasklet(f'assign_{line}', {connector for connector, _ in operands}, {'__out'}, f'__out = {code}')

    in_memlets = {}
    for connector, access in operands:
        if access.is_scalar_access:
            in_memlets[connector] = Memlet(data=access.container, subset=access.subset)
        else:
            in_memlets[connector] = Memlet(data=access.container, subset=indexed_subset(access, params, result_shape))
    out_memlets = {'__out': Memlet(data=target.container, subset=_target_indexed_subset(target.subset, params))}

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _target_indexed_subset(subset: subsets.Range, params: List[str]) -> subsets.Range:
    """
    Index the non-degenerate dimensions of a write subset with map parameters,
    keeping degenerate dimensions pinned to their start.
    """
    ranges = []
    param_iterator = iter(params)
    for size, (start, _, step) in zip(subset.size(), subset.ranges):
        if size == 1:
            ranges.append((start, start, 1))
        else:
            param = symbolic.pystr_to_symbolic(next(param_iterator))
            index = start + param * step
            ranges.append((index, index, 1))
    return subsets.Range(ranges)
