# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Array-literal inference and lowering helpers for the direct frontend."""

import ast
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from dace import data, dtypes
from dace.frontend.python import astutils
from dace.frontend.python.replacements.array_creation_dace import infer_array_creation_descriptor
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

DescriptorInferer = Callable[[ast.AST], Optional[data.Data]]
ScalarDescriptorInferer = Callable[[ast.AST, Optional[data.Data]], Optional[data.Data]]
EvaluationContextFactory = Callable[[], Dict[str, Any]]
OutputTargetResolver = Callable[[ast.AST, ast.AST, Optional[data.Data]], Optional[Tuple[str, Memlet, data.Data]]]
DataAccessResolver = Callable[[ast.AST], Optional[Tuple[str, Memlet, data.Data, Optional[data.Data]]]]
CallableNameResolver = Callable[[ast.AST], str]
TaskletNameFactory = Callable[[ast.AST], str]
ArrayConstructorNameFactory = Callable[[], str]


@dataclass(frozen=True)
class ArrayLiteralContext:
    infer_descriptor: DescriptorInferer
    infer_scalar_descriptor: ScalarDescriptorInferer
    evaluation_context: EvaluationContextFactory
    resolve_output_target: OutputTargetResolver
    resolve_data_access: DataAccessResolver
    resolve_callable_name: CallableNameResolver
    tasklet_name: TaskletNameFactory
    array_constructor_name: ArrayConstructorNameFactory


class ArrayLiteralSupportLibrary:
    """Descriptor inference and lowering for array-valued literals."""

    def infer_expression_descriptor(self, context: ArrayLiteralContext, node: ast.AST) -> Optional[data.Data]:
        return infer_array_literal_descriptor(node,
                                              context.infer_descriptor,
                                              context.infer_scalar_descriptor,
                                              context.evaluation_context,
                                              callable_name_resolver=context.resolve_callable_name)

    def lower_assignment(self, context: ArrayLiteralContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        descriptor = self.infer_expression_descriptor(context, value)
        if descriptor is None or isinstance(descriptor, data.Scalar):
            return None

        output = context.resolve_output_target(target, value, annotated_descriptor)
        if output is None:
            return None

        _, out_memlet, _ = output
        rewritten, input_memlets = _rewrite_with_connectors(value, context.resolve_data_access)
        lowered_value = _lowered_array_expression(rewritten, value, context)
        tasklet = tn.FrontendTasklet(
            name=context.tasklet_name(target),
            code=CodeBlock(f'out = {astutils.unparse(ast.fix_missing_locations(lowered_value))}'))
        return tn.TaskletNode(node=tasklet, in_memlets=input_memlets, out_memlets={'out': copy.deepcopy(out_memlet)})


def infer_array_literal_descriptor(
        node: ast.AST,
        infer_descriptor: DescriptorInferer,
        infer_scalar_descriptor: ScalarDescriptorInferer,
        evaluation_context: EvaluationContextFactory,
        *,
        callable_name_resolver: Optional[CallableNameResolver] = None) -> Optional[data.Data]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return _infer_sequence_descriptor(node, infer_descriptor, infer_scalar_descriptor, evaluation_context)

    if isinstance(node, ast.Call) and _is_array_constructor_call(node, evaluation_context, callable_name_resolver):
        dtype = _parse_dtype_argument(node, evaluation_context)
        ndmin = _parse_ndmin_argument(node, evaluation_context)
        if ndmin is None:
            return None
        obj = _call_argument(node, 0, 'obj')
        if obj is None:
            return None
        return _infer_array_call_object_descriptor(obj,
                                                   infer_descriptor,
                                                   infer_scalar_descriptor,
                                                   evaluation_context,
                                                   dtype=dtype,
                                                   ndmin=ndmin)

    return None


def _infer_array_call_object_descriptor(obj: ast.AST, infer_descriptor: DescriptorInferer,
                                        infer_scalar_descriptor: ScalarDescriptorInferer,
                                        evaluation_context: EvaluationContextFactory, *,
                                        dtype: Optional[dtypes.typeclass], ndmin: int) -> Optional[data.Data]:
    if isinstance(obj, (ast.List, ast.Tuple)):
        descriptor = _infer_sequence_descriptor(obj, infer_descriptor, infer_scalar_descriptor, evaluation_context)
        if descriptor is None:
            return None
        return _apply_array_coercions(descriptor, dtype=dtype, ndmin=ndmin)

    descriptor = infer_descriptor(obj)
    if descriptor is not None and not isinstance(descriptor, data.Scalar):
        return _apply_array_coercions(descriptor, dtype=dtype, ndmin=ndmin)

    static_value = try_resolve_static_value(obj, evaluation_context())
    if static_value is UNRESOLVED:
        return None

    descriptor = infer_array_creation_descriptor(static_value, dtype=dtype, ndmin=ndmin)
    if descriptor is None:
        return None
    descriptor.transient = True
    return descriptor


def _infer_sequence_descriptor(node: ast.AST, infer_descriptor: DescriptorInferer,
                               infer_scalar_descriptor: ScalarDescriptorInferer,
                               evaluation_context: EvaluationContextFactory) -> Optional[data.Data]:
    static_value = try_resolve_static_value(node, evaluation_context())
    if static_value is not UNRESOLVED:
        descriptor = infer_array_creation_descriptor(static_value)
        if descriptor is not None:
            descriptor.transient = True
            return descriptor

    shape, dtype = _infer_sequence_shape_dtype(node, infer_descriptor, infer_scalar_descriptor)
    if shape is None or dtype is None:
        return None
    return data.Array(dtype, list(shape), transient=True)


def _infer_sequence_shape_dtype(
        node: ast.AST, infer_descriptor: DescriptorInferer, infer_scalar_descriptor: ScalarDescriptorInferer
) -> Tuple[Optional[Tuple[int, ...]], Optional[dtypes.typeclass]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        descriptor = infer_descriptor(node) or infer_scalar_descriptor(node, None)
        if not isinstance(descriptor, data.Scalar):
            return (None, None)
        return (tuple(), descriptor.dtype)

    child_shapes: list[Tuple[int, ...]] = []
    child_dtype: Optional[dtypes.typeclass] = None
    for element in node.elts:
        element_shape, element_dtype = _infer_sequence_shape_dtype(element, infer_descriptor, infer_scalar_descriptor)
        if element_shape is None or element_dtype is None:
            return (None, None)
        child_shapes.append(element_shape)
        child_dtype = element_dtype if child_dtype is None else dtypes.result_type_of(child_dtype, element_dtype)

    if not child_shapes:
        return ((0, ), dtypes.float64)

    first_shape = child_shapes[0]
    if any(shape != first_shape for shape in child_shapes[1:]):
        return (None, None)

    return ((len(node.elts), ) + first_shape, child_dtype)


def _apply_array_coercions(descriptor: data.Data, *, dtype: Optional[dtypes.typeclass],
                           ndmin: int) -> Optional[data.Data]:
    result = copy.deepcopy(descriptor)
    if dtype is not None:
        result.dtype = dtype

    shape = list(getattr(result, 'shape', ()))
    if isinstance(result, data.Scalar):
        if ndmin <= 0:
            return data.Scalar(result.dtype, transient=True)
        shape = [1] * ndmin
    elif len(shape) < ndmin:
        shape = [1] * (ndmin - len(shape)) + shape

    if isinstance(result, data.Scalar):
        return data.Array(result.dtype, shape, transient=True)

    if hasattr(result, 'set_shape'):
        result.set_shape(shape)
    result.transient = True
    return result


def _call_argument(node: ast.Call, position: int, keyword: str) -> Optional[ast.AST]:
    if len(node.args) > position:
        return node.args[position]
    for kw in node.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _parse_dtype_argument(node: ast.Call, evaluation_context: EvaluationContextFactory) -> Optional[dtypes.typeclass]:
    dtype_node = _call_argument(node, 1, 'dtype')
    if dtype_node is None:
        return None
    dtype_value = try_resolve_static_value(dtype_node, evaluation_context())
    if dtype_value is UNRESOLVED:
        return None
    try:
        return dtype_value if isinstance(dtype_value, dtypes.typeclass) else dtypes.typeclass(dtype_value)
    except TypeError:
        return None


def _parse_ndmin_argument(node: ast.Call, evaluation_context: EvaluationContextFactory) -> Optional[int]:
    ndmin_node = _call_argument(node, 4, 'ndmin')
    if ndmin_node is None:
        return 0
    ndmin_value = try_resolve_static_value(ndmin_node, evaluation_context())
    if not isinstance(ndmin_value, int):
        return None
    return ndmin_value


def _is_array_constructor_call(node: ast.Call, evaluation_context: EvaluationContextFactory,
                               callable_name_resolver: Optional[CallableNameResolver]) -> bool:
    if callable_name_resolver is not None:
        call_name = callable_name_resolver(node.func)
    else:
        call_name = astutils.rname(node.func)

    if call_name == 'numpy.array':
        return True

    resolved = try_resolve_static_value(node.func, evaluation_context())
    module_name = getattr(resolved, '__module__', None) if resolved is not UNRESOLVED else None
    callable_name = getattr(resolved, '__name__', None) if resolved is not UNRESOLVED else None
    return callable_name == 'array' and module_name == 'numpy'


def _rewrite_with_connectors(node: ast.AST,
                             resolve_data_access: DataAccessResolver) -> Tuple[ast.AST, Dict[str, Memlet]]:
    rewritten = copy.deepcopy(node)
    input_memlets: Dict[str, Memlet] = {}
    connector_names: Dict[Tuple[str, str, str], str] = {}

    class _AccessRewriter(ast.NodeTransformer):

        def visit(self, current: ast.AST) -> ast.AST:
            access = resolve_data_access(current)
            if access is not None:
                name, memlet, _descriptor, _view_descriptor = access
                key = (name, str(memlet.subset), str(memlet.other_subset) if memlet.other_subset is not None else '')
                connector = connector_names.get(key)
                if connector is None:
                    connector = f'in{len(connector_names)}'
                    connector_names[key] = connector
                    input_memlets[connector] = copy.deepcopy(memlet)
                return ast.copy_location(ast.Name(id=connector, ctx=ast.Load()), current)
            return super().visit(current)

    return (_AccessRewriter().visit(rewritten), input_memlets)


def _lowered_array_expression(node: ast.AST, original: ast.AST, context: ArrayLiteralContext) -> ast.AST:
    if isinstance(original, ast.Call):
        return node

    constructor = _dotted_name_ast(context.array_constructor_name())
    return ast.Call(func=constructor, args=[node], keywords=[])


def _dotted_name_ast(name: str) -> ast.AST:
    parts = name.split('.')
    expr: ast.AST = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        expr = ast.Attribute(value=expr, attr=part, ctx=ast.Load())
    return expr
