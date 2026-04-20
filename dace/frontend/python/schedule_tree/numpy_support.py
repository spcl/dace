# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NumPy-oriented lowering helpers for the direct schedule-tree frontend."""

import ast
import copy
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dace import data, dtypes, subsets, symbolic
from dace.data.pydata import PythonDict, PythonList, PythonTuple
from dace.frontend.python import astutils, memlet_parser
from dace.frontend.python.replacements.utils import broadcast_to, broadcast_together
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.frontend.python.schedule_tree.type_inference import _Binding
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.type_inference import infer_expr_type

OutputTargetResolver = Callable[[ast.AST, ast.AST, Optional[data.Data]], Optional[Tuple[str, Memlet, data.Data]]]
TaskletNameFactory = Callable[[ast.AST], str]
EvaluationContextFactory = Callable[[], Dict[str, Any]]
FreshSymbolFactory = Callable[[str], symbolic.symbol]


@dataclass(frozen=True)
class NumpyLoweringContext:
    bindings: Dict[str, _Binding]
    evaluation_context: EvaluationContextFactory
    resolve_output_target: OutputTargetResolver
    tasklet_name: TaskletNameFactory
    fresh_symbol: FreshSymbolFactory


@dataclass(frozen=True)
class _AdvancedIndexBlueprint:
    output_shape: Tuple[Any, ...]
    output_subset: subsets.Range
    source_memlet: Memlet
    index_memlets: Tuple[Memlet, ...]


@dataclass(frozen=True)
class _ResolvedAccess:
    node: ast.AST
    name: str
    descriptor: data.Data
    subset: subsets.Range
    logical_ranges: Tuple[Tuple[Any, Any, Any], ...]
    array_connector: str
    index_connectors: Tuple[str, ...]
    output_shape: Tuple[Any, ...]
    new_axes: Tuple[int, ...] = tuple()
    scalar_dims: Tuple[int, ...] = tuple()
    blueprint: Optional[_AdvancedIndexBlueprint] = None


@dataclass(frozen=True)
class _BooleanGatherPlan:
    source_name: str
    source_descriptor: data.Data
    result_descriptor: data.Array
    input_memlets: Dict[str, Memlet]
    nnz_symbol: symbolic.symbol
    mask_expr: Optional[ast.AST] = None


@dataclass(frozen=True)
class _ExpressionAnalysis:
    tasklet_value: ast.AST
    typing_value: ast.AST
    accesses: Tuple[_ResolvedAccess, ...]
    result_shape: Tuple[Any, ...]
    result_dtype: dtypes.typeclass


@dataclass(frozen=True)
class _IterationPlan:
    original_subset: subsets.Range
    squeezed_subset: subsets.Range
    non_singleton_dims: Tuple[int, ...]
    params: Tuple[str, ...]
    ranges: Tuple[Tuple[str, str, str], ...]


@dataclass(frozen=True)
class _AdvancedTarget:
    name: str
    output_shape: Tuple[Any, ...]
    output_memlet: Memlet
    target_expr: ast.AST
    input_memlets: Dict[str, Memlet]
    guard_expr: Optional[ast.AST] = None


class NumpySupportLibrary:
    """Ordered NumPy-specific lowering and inference helpers."""

    def __init__(self) -> None:
        self.assignment_passes = (_BooleanMaskReadPass(), _ElementwiseAssignmentPass())

    def lower_assignment(self, context: NumpyLoweringContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        for lowering_pass in self.assignment_passes:
            lowered = lowering_pass.lower_assignment(context, target, value, annotated_descriptor)
            if lowered is not None:
                return lowered
        return None

    def infer_expression_descriptor(self, context: NumpyLoweringContext, value: ast.AST) -> Optional[data.Data]:
        for lowering_pass in self.assignment_passes:
            descriptor = lowering_pass.infer_expression_descriptor(context, value)
            if descriptor is not None:
                return descriptor
        return None


class _BooleanMaskReadPass:
    """Lower RHS boolean-mask gathers as frontend library calls."""

    _LIBRARY_NAME = 'boolean_mask_gather'

    def lower_assignment(self, context: NumpyLoweringContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        output = context.resolve_output_target(target, value, annotated_descriptor)
        if output is None:
            return None

        _, output_memlet, output_descriptor = output
        plan = _resolve_boolean_gather(value, context, output_descriptor)
        if plan is None:
            return None

        properties: Dict[str, Any] = {
            'nnz_symbol': str(plan.nnz_symbol),
            'upper_bound': str(plan.result_descriptor.total_size),
        }
        if plan.mask_expr is not None:
            properties['mask_expr'] = _unparse(plan.mask_expr)

        return tn.LibraryCall(node=tn.FrontendLibrary(name=self._LIBRARY_NAME, properties=properties),
                              in_memlets=plan.input_memlets,
                              out_memlets={'out': output_memlet})

    def infer_expression_descriptor(self, context: NumpyLoweringContext, value: ast.AST) -> Optional[data.Data]:
        plan = _resolve_boolean_gather(value, context, None)
        if plan is None:
            return None
        return plan.result_descriptor


class _ElementwiseAssignmentPass:
    """Lower NumPy-style elementwise assignments to explicit map scopes."""

    def lower_assignment(self, context: NumpyLoweringContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        boolean_target = _resolve_boolean_target(context, target, value)
        if boolean_target is not None:
            return self._lower_boolean_target_assignment(context, boolean_target, target, value)

        advanced_target = _resolve_integer_target(context, target)

        analysis = _ElementwiseExpressionAnalyzer(context).analyze(value)
        scalar_only_value = analysis is None and _is_trivial_scalar(value, context)
        if analysis is None and not scalar_only_value:
            return None

        if advanced_target is not None:
            if isinstance(value, ast.BinOp) and _ast_equivalent(value.left, target):
                return self._lower_integer_target_augassign(context, advanced_target, target, value)
            if analysis is not None and analysis.result_shape and not _is_shape_compatible_shape(
                    advanced_target.output_shape, analysis.result_shape):
                return None
            iteration_plan = _build_iteration_plan_from_shape(advanced_target.output_shape)
            if iteration_plan is None:
                return None

            input_memlets: Dict[str, Memlet] = {}
            if analysis is not None:
                for access in analysis.accesses:
                    access_memlets = _build_input_memlets(access, iteration_plan)
                    if access_memlets is None:
                        return None
                    input_memlets.update(access_memlets)
            input_memlets.update(advanced_target.input_memlets)

            tasklet_value = analysis.tasklet_value if analysis is not None else copy.deepcopy(value)
            tasklet = tn.FrontendTasklet(
                name=context.tasklet_name(target),
                code=CodeBlock(f'{_unparse(advanced_target.target_expr)} = {_unparse(tasklet_value)}'))
            tasklet_node = tn.TaskletNode(node=tasklet,
                                          in_memlets=input_memlets,
                                          out_memlets={'out': advanced_target.output_memlet})
            map_scope = tn.MapScope(node=tn.FrontendMap(params=list(iteration_plan.params),
                                                        ranges=list(iteration_plan.ranges)),
                                    children=[])
            for param in iteration_plan.params:
                map_scope.symbols[param] = symbolic.symbol(param, dtypes.int64)
            tasklet_node.parent = map_scope
            map_scope.children.append(tasklet_node)
            return map_scope

        output = context.resolve_output_target(target, value, annotated_descriptor)
        if output is None:
            return None

        target_name, target_memlet, _ = output
        if not isinstance(target_memlet.subset, subsets.Range):
            return None
        if analysis is not None and analysis.result_shape and not _is_shape_compatible(
                target_memlet.subset, analysis.result_shape):
            return None

        if isinstance(target, ast.Subscript) and target_memlet.subset.num_elements() == 1:
            input_memlets: Dict[str, Memlet] = {}
            if analysis is not None:
                if analysis.result_shape:
                    return None
                for access in analysis.accesses:
                    access_memlet = _build_scalar_input_memlet(access)
                    if access_memlet is None:
                        return None
                    input_memlets.update(access_memlet)

            tasklet_value = analysis.tasklet_value if analysis is not None else copy.deepcopy(value)
            tasklet = tn.FrontendTasklet(name=context.tasklet_name(target),
                                         code=CodeBlock(f'out = {_unparse(tasklet_value)}'))
            return tn.TaskletNode(
                node=tasklet,
                in_memlets=input_memlets,
                out_memlets={'out': Memlet(data=target_name, subset=copy.deepcopy(target_memlet.subset))})

        iteration_plan = _build_iteration_plan(target_memlet.subset)
        if iteration_plan is None:
            return None

        input_memlets: Dict[str, Memlet] = {}
        if analysis is not None:
            for access in analysis.accesses:
                access_memlets = _build_input_memlets(access, iteration_plan)
                if access_memlets is None:
                    return None
                input_memlets.update(access_memlets)

        output_memlet = _build_output_memlet(target_name, iteration_plan)
        tasklet_value = analysis.tasklet_value if analysis is not None else copy.deepcopy(value)
        tasklet = tn.FrontendTasklet(name=context.tasklet_name(target),
                                     code=CodeBlock(f'out = {_unparse(tasklet_value)}'))
        tasklet_node = tn.TaskletNode(node=tasklet, in_memlets=input_memlets, out_memlets={'out': output_memlet})

        map_scope = tn.MapScope(node=tn.FrontendMap(params=list(iteration_plan.params),
                                                    ranges=list(iteration_plan.ranges)),
                                children=[])
        for param in iteration_plan.params:
            map_scope.symbols[param] = symbolic.symbol(param, dtypes.int64)
        tasklet_node.parent = map_scope
        map_scope.children.append(tasklet_node)
        return map_scope

    def _lower_integer_target_augassign(self, context: NumpyLoweringContext, advanced_target: _AdvancedTarget,
                                        target: ast.AST, value: ast.BinOp) -> Optional[tn.ScheduleTreeNode]:
        rhs_analysis = _ElementwiseExpressionAnalyzer(context).analyze(value.right)
        if rhs_analysis is None and not _is_trivial_scalar(value.right, context):
            return None

        iteration_plan = _build_iteration_plan_from_shape(advanced_target.output_shape)
        if iteration_plan is None:
            return None

        input_memlets: Dict[str, Memlet] = dict(advanced_target.input_memlets)
        if rhs_analysis is not None:
            if rhs_analysis.result_shape and not _is_shape_compatible_shape(advanced_target.output_shape,
                                                                            rhs_analysis.result_shape):
                return None
            for access in rhs_analysis.accesses:
                access_memlets = _build_input_memlets(access, iteration_plan)
                if access_memlets is None:
                    return None
                input_memlets.update(access_memlets)
            rhs_tasklet = rhs_analysis.tasklet_value
        else:
            rhs_tasklet = copy.deepcopy(value.right)

        input_memlets['cur'] = Memlet(data=advanced_target.name,
                                      subset=copy.deepcopy(advanced_target.output_memlet.subset),
                                      volume=advanced_target.output_memlet.volume)
        tasklet_value = ast.copy_location(
            ast.BinOp(left=ast.Name(id='cur', ctx=ast.Load()), op=copy.deepcopy(value.op), right=rhs_tasklet), value)
        tasklet = tn.FrontendTasklet(
            name=context.tasklet_name(target),
            code=CodeBlock(f'{_unparse(advanced_target.target_expr)} = {_unparse(tasklet_value)}'))
        tasklet_node = tn.TaskletNode(node=tasklet,
                                      in_memlets=input_memlets,
                                      out_memlets={'out': advanced_target.output_memlet})
        map_scope = tn.MapScope(node=tn.FrontendMap(params=list(iteration_plan.params),
                                                    ranges=list(iteration_plan.ranges)),
                                children=[])
        for param in iteration_plan.params:
            map_scope.symbols[param] = symbolic.symbol(param, dtypes.int64)
        tasklet_node.parent = map_scope
        map_scope.children.append(tasklet_node)
        return map_scope

    def _lower_boolean_target_assignment(self, context: NumpyLoweringContext, boolean_target: _AdvancedTarget,
                                         target: ast.AST, value: ast.AST) -> Optional[tn.ScheduleTreeNode]:
        is_augassign = isinstance(value, ast.BinOp) and _ast_equivalent(value.left, target)
        rhs_node = value.right if is_augassign else value
        rhs_analysis = _ElementwiseExpressionAnalyzer(context).analyze(rhs_node)
        if rhs_analysis is None and not _is_trivial_scalar(rhs_node, context):
            return None

        iteration_plan = _build_iteration_plan_from_shape(boolean_target.output_shape)
        if iteration_plan is None:
            return None

        input_memlets: Dict[str, Memlet] = dict(boolean_target.input_memlets)
        rhs_tasklet = ast.copy_location(ast.Constant(value=None), rhs_node)
        if rhs_analysis is not None:
            if rhs_analysis.result_shape and not _is_shape_compatible_shape(boolean_target.output_shape,
                                                                            rhs_analysis.result_shape):
                return None
            rhs_tasklet = rhs_analysis.tasklet_value
            for access in rhs_analysis.accesses:
                access_memlets = _build_input_memlets(access, iteration_plan)
                if access_memlets is None:
                    return None
                input_memlets.update(access_memlets)
        else:
            rhs_tasklet = copy.deepcopy(rhs_node)

        if is_augassign:
            current_memlet = Memlet(data=boolean_target.name, subset=copy.deepcopy(boolean_target.output_memlet.subset))
            input_memlets['cur'] = current_memlet
            rhs_tasklet = ast.copy_location(
                ast.BinOp(left=ast.Name(id='cur', ctx=ast.Load()), op=copy.deepcopy(value.op), right=rhs_tasklet),
                value)

        if boolean_target.guard_expr is None:
            return None
        code = f'if {_unparse(boolean_target.guard_expr)}:\n    out = {_unparse(rhs_tasklet)}'
        output_memlet = copy.deepcopy(boolean_target.output_memlet)
        output_memlet.dynamic = True
        tasklet = tn.FrontendTasklet(name=context.tasklet_name(target), code=CodeBlock(code))
        tasklet_node = tn.TaskletNode(node=tasklet, in_memlets=input_memlets, out_memlets={'out': output_memlet})
        map_scope = tn.MapScope(node=tn.FrontendMap(params=list(iteration_plan.params),
                                                    ranges=list(iteration_plan.ranges)),
                                children=[])
        for param in iteration_plan.params:
            map_scope.symbols[param] = symbolic.symbol(param, dtypes.int64)
        tasklet_node.parent = map_scope
        map_scope.children.append(tasklet_node)
        return map_scope

    def infer_expression_descriptor(self, context: NumpyLoweringContext, value: ast.AST) -> Optional[data.Data]:
        analysis = _ElementwiseExpressionAnalyzer(context).analyze(value)
        if analysis is None:
            return None
        if not analysis.result_shape:
            return data.Scalar(analysis.result_dtype, transient=True)
        return data.Array(analysis.result_dtype, list(analysis.result_shape), transient=True)


class _ElementwiseExpressionAnalyzer:
    """Recognizes scalarized NumPy expressions over array accesses."""

    def __init__(self, context: NumpyLoweringContext, start_index: int = 0) -> None:
        self.context = context
        self.start_index = start_index
        self.accesses: List[_ResolvedAccess] = []
        self.access_map: Dict[Tuple[str, str, Tuple[str, ...]], _ResolvedAccess] = {}

    def analyze(self, node: ast.AST) -> Optional[_ExpressionAnalysis]:
        rewritten = self._rewrite(copy.deepcopy(node))
        if rewritten is None or not self.accesses:
            return None

        tasklet_value, typing_value = rewritten
        result_shape = _broadcast_shape(tuple(access.output_shape for access in self.accesses))
        if result_shape is None:
            return None

        scalar_types = _scalar_type_environment(self.context, self.accesses)
        try:
            result_dtype = infer_expr_type(_unparse(typing_value), scalar_types)
        except Exception:
            result_dtype = self.accesses[0].descriptor.dtype

        return _ExpressionAnalysis(tasklet_value=ast.fix_missing_locations(tasklet_value),
                                   typing_value=ast.fix_missing_locations(typing_value),
                                   accesses=tuple(self.accesses),
                                   result_shape=result_shape,
                                   result_dtype=result_dtype)

    def _rewrite(self, node: ast.AST) -> Optional[Tuple[ast.AST, ast.AST]]:
        access = self._resolve_array_access(node)
        if access is not None:
            return (_tasklet_expr_for_access(access),
                    ast.copy_location(ast.Name(id=access.array_connector, ctx=ast.Load()), node))

        if isinstance(node, ast.Constant):
            copied = copy.deepcopy(node)
            return (copied, copy.deepcopy(copied))

        if isinstance(node, ast.Name):
            if not _is_scalar_leaf(node, self.context):
                return None
            copied = copy.deepcopy(node)
            return (copied, copy.deepcopy(copied))

        if isinstance(node, ast.Attribute):
            if not _is_scalar_leaf(node, self.context):
                return None
            copied = copy.deepcopy(node)
            return (copied, copy.deepcopy(copied))

        if isinstance(node, ast.Subscript):
            if not _is_scalar_leaf(node, self.context):
                return None
            copied = copy.deepcopy(node)
            return (copied, copy.deepcopy(copied))

        if isinstance(node, ast.BinOp):
            left = self._rewrite(node.left)
            right = self._rewrite(node.right)
            if left is None or right is None:
                return None
            return (ast.copy_location(ast.BinOp(left=left[0], op=copy.deepcopy(node.op), right=right[0]), node),
                    ast.copy_location(ast.BinOp(left=left[1], op=copy.deepcopy(node.op), right=right[1]), node))

        if isinstance(node, ast.UnaryOp):
            operand = self._rewrite(node.operand)
            if operand is None:
                return None
            return (ast.copy_location(ast.UnaryOp(op=copy.deepcopy(node.op), operand=operand[0]), node),
                    ast.copy_location(ast.UnaryOp(op=copy.deepcopy(node.op), operand=operand[1]), node))

        if isinstance(node, ast.BoolOp):
            values = [self._rewrite(value) for value in node.values]
            if any(value is None for value in values):
                return None
            return (ast.copy_location(ast.BoolOp(op=copy.deepcopy(node.op), values=[value[0] for value in values]),
                                      node),
                    ast.copy_location(ast.BoolOp(op=copy.deepcopy(node.op), values=[value[1] for value in values]),
                                      node))

        if isinstance(node, ast.Compare):
            left = self._rewrite(node.left)
            comparators = [self._rewrite(comp) for comp in node.comparators]
            if left is None or any(comp is None for comp in comparators):
                return None
            return (ast.copy_location(
                ast.Compare(left=left[0], ops=copy.deepcopy(node.ops), comparators=[comp[0] for comp in comparators]),
                node),
                    ast.copy_location(
                        ast.Compare(left=left[1],
                                    ops=copy.deepcopy(node.ops),
                                    comparators=[comp[1] for comp in comparators]), node))

        if isinstance(node, ast.IfExp):
            test = self._rewrite(node.test)
            body = self._rewrite(node.body)
            orelse = self._rewrite(node.orelse)
            if test is None or body is None or orelse is None:
                return None
            return (ast.copy_location(ast.IfExp(test=test[0], body=body[0], orelse=orelse[0]), node),
                    ast.copy_location(ast.IfExp(test=test[1], body=body[1], orelse=orelse[1]), node))

        if isinstance(node, ast.Call):
            if not _is_supported_call(node, self.context):
                return None
            args = [self._rewrite(arg) for arg in node.args]
            if any(arg is None for arg in args):
                return None
            keywords: List[Tuple[ast.keyword, ast.keyword]] = []
            for keyword in node.keywords:
                rewritten_value = self._rewrite(keyword.value)
                if rewritten_value is None:
                    return None
                keywords.append(
                    (ast.keyword(arg=keyword.arg,
                                 value=rewritten_value[0]), ast.keyword(arg=keyword.arg, value=rewritten_value[1])))
            return (ast.copy_location(
                ast.Call(func=copy.deepcopy(node.func),
                         args=[arg[0] for arg in args],
                         keywords=[kw[0] for kw in keywords]), node),
                    ast.copy_location(
                        ast.Call(func=copy.deepcopy(node.func),
                                 args=[arg[1] for arg in args],
                                 keywords=[kw[1] for kw in keywords]), node))

        return None

    def _resolve_array_access(self, node: ast.AST) -> Optional[_ResolvedAccess]:
        if isinstance(node, ast.Name):
            binding = self.context.bindings.get(node.id)
            if binding is None or binding.descriptor is None or not _is_numpy_arraylike(binding.descriptor):
                return None
            subset = subsets.Range.from_array(binding.descriptor)
            return self._register_basic_access(node, node.id, binding.descriptor, subset)

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            binding = self.context.bindings.get(node.value.id)
            if binding is None or binding.descriptor is None or not _is_numpy_arraylike(binding.descriptor):
                return None
            try:
                subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(binding.descriptor, node,
                                                                              self.context.evaluation_context())
            except Exception:
                return None

            if arrdims:
                return self._register_advanced_access(node, node.value.id, binding.descriptor, subset, new_axes,
                                                      arrdims)
            return self._register_basic_access(node, node.value.id, binding.descriptor, subset, new_axes)

        return None

    def _register_basic_access(self,
                               node: ast.AST,
                               name: str,
                               descriptor: data.Data,
                               subset: subsets.Range,
                               new_axes: Sequence[int] = ()) -> _ResolvedAccess:
        axis_key = tuple(str(axis) for axis in new_axes)
        key = (name, str(subset), axis_key)
        existing = self.access_map.get(key)
        if existing is not None:
            return existing

        scalar_dims = _scalar_indexed_dims(node, len(descriptor.shape), self.context)
        logical_ranges = _logical_ranges_from_basic_access(node, subset, descriptor, self.context)
        output_shape = tuple(_shape_from_ranges(logical_ranges))
        if output_shape:
            logical_shape = list(output_shape)
            for axis in sorted(new_axes):
                logical_shape.insert(axis, 1)
            output_shape = tuple(logical_shape)
        access = _ResolvedAccess(node=node,
                                 name=name,
                                 descriptor=_clone_descriptor(descriptor),
                                 subset=copy.deepcopy(subset),
                                 logical_ranges=logical_ranges,
                                 array_connector=f'in{self.start_index + len(self.accesses)}',
                                 index_connectors=tuple(),
                                 output_shape=output_shape,
                                 new_axes=tuple(new_axes),
                                 scalar_dims=scalar_dims,
                                 blueprint=None)
        self.accesses.append(access)
        self.access_map[key] = access
        return access

    def _register_advanced_access(self, node: ast.AST, name: str, descriptor: data.Data, subset: subsets.Range,
                                  new_axes: Sequence[int], arrdims: Dict[int, Any]) -> Optional[_ResolvedAccess]:
        if any(_is_boolean_index(index_name, self.context) for index_name in arrdims.values()):
            return None

        blueprint = _build_advanced_blueprint(name, subset, new_axes, arrdims, self.context)
        if blueprint is None:
            return None

        key = (name, str(subset), tuple(str(index) for index in arrdims.values()))
        existing = self.access_map.get(key)
        if existing is not None:
            return existing

        access_index = self.start_index + len(self.accesses)
        index_connectors = tuple(f'idx{access_index}_{i}' for i in range(len(blueprint.index_memlets)))
        access = _ResolvedAccess(node=node,
                                 name=name,
                                 descriptor=_clone_descriptor(descriptor),
                                 subset=copy.deepcopy(subset),
                                 logical_ranges=tuple(copy.deepcopy(blueprint.source_memlet.subset).ranges),
                                 array_connector=f'in{access_index}',
                                 index_connectors=index_connectors,
                                 output_shape=blueprint.output_shape,
                                 blueprint=blueprint)
        self.accesses.append(access)
        self.access_map[key] = access
        return access


def _build_iteration_plan(target_subset: subsets.Range) -> Optional[_IterationPlan]:
    if target_subset.num_elements() == 1:
        return None

    squeezed_subset = copy.deepcopy(target_subset)
    non_singleton_dims = tuple(range(len(target_subset.ranges)))
    params = tuple(f'__i{i}' for i in range(len(target_subset.ranges)))
    ranges = tuple(_frontend_range_tuple(dim) for dim in target_subset.ranges)
    return _IterationPlan(original_subset=copy.deepcopy(target_subset),
                          squeezed_subset=squeezed_subset,
                          non_singleton_dims=non_singleton_dims,
                          params=params,
                          ranges=ranges)


def _build_iteration_plan_from_shape(shape: Sequence[Any]) -> Optional[_IterationPlan]:
    if not shape:
        return None
    return _build_iteration_plan(subsets.Range([(0, dim - 1, 1) for dim in shape]))


def _build_output_memlet(target_name: str, iteration_plan: _IterationPlan) -> Memlet:
    param_iter = iter(iteration_plan.params)
    indices: List[Any] = []
    for dim, (start, _, _) in enumerate(iteration_plan.original_subset.ranges):
        if dim in iteration_plan.non_singleton_dims:
            indices.append(symbolic.symbol(next(param_iter), dtypes.int64))
        else:
            indices.append(start)
    return Memlet(data=target_name, subset=subsets.Range.from_indices(indices))


def _build_input_memlets(access: _ResolvedAccess, iteration_plan: _IterationPlan) -> Optional[Dict[str, Memlet]]:
    if access.blueprint is not None:
        return _build_advanced_input_memlets(access, iteration_plan)

    if access.new_axes:
        return _build_newaxis_input_memlets(access, iteration_plan)

    if isinstance(access.node, ast.Subscript) and access.subset.num_elements() == 1:
        return {access.array_connector: Memlet(data=access.name, subset=copy.deepcopy(access.subset))}

    try:
        _, all_idx_tuples, _, _, inp_idx = broadcast_to(iteration_plan.squeezed_subset.size(), access.output_shape)
    except Exception:
        return None

    input_indices = [part.strip() for part in inp_idx.split(',')] if inp_idx else []
    missing_dimensions = list(iteration_plan.squeezed_subset.ranges[:len(all_idx_tuples) - len(input_indices)])
    fake_subset = subsets.Range(missing_dimensions + list(access.logical_ranges))

    offset_indices_to_ignore = set()
    for index, idx in enumerate(input_indices):
        if not symbolic.issymbolic(symbolic.pystr_to_symbolic(idx)):
            offset_indices_to_ignore.add(index)
    offset_indices = [index for index in range(len(fake_subset)) if index not in offset_indices_to_ignore]
    fake_subset.offset(iteration_plan.squeezed_subset, True, indices=offset_indices)

    idx_and_subset = reversed(list(zip(reversed(input_indices), reversed(fake_subset.ranges))))
    subset_indices = [_compose_input_index(idx, subset) for idx, subset in idx_and_subset]
    subset_indices = _reinsert_scalar_dims(access, subset_indices)
    return {access.array_connector: Memlet(data=access.name, subset=subsets.Range.from_indices(subset_indices))}


def _build_scalar_input_memlet(access: _ResolvedAccess) -> Optional[Dict[str, Memlet]]:
    if access.blueprint is not None or access.new_axes or access.output_shape:
        return None
    if not isinstance(access.subset, subsets.Range) or access.subset.num_elements() != 1:
        return None
    return {access.array_connector: Memlet(data=access.name, subset=copy.deepcopy(access.subset))}


def _build_newaxis_input_memlets(access: _ResolvedAccess,
                                 iteration_plan: _IterationPlan) -> Optional[Dict[str, Memlet]]:
    try:
        _, _, _, _, inp_idx = broadcast_to(iteration_plan.squeezed_subset.size(), access.output_shape)
    except Exception:
        return None

    input_indices = [part.strip() for part in inp_idx.split(',')] if inp_idx else []
    logical_ranges = list(copy.deepcopy(access.logical_ranges))
    for axis in sorted(access.new_axes):
        logical_ranges.insert(axis, (0, 0, 1))

    if len(input_indices) != len(logical_ranges):
        return None

    subset_indices = [
        _compose_input_index(idx, subset) for dim, (idx, subset) in enumerate(zip(input_indices, logical_ranges))
        if dim not in access.new_axes
    ]
    subset_indices = _reinsert_scalar_dims(access, subset_indices)
    return {access.array_connector: Memlet(data=access.name, subset=subsets.Range.from_indices(subset_indices))}


def _reinsert_scalar_dims(access: _ResolvedAccess, subset_indices: Sequence[Any]) -> List[Any]:
    if not access.scalar_dims:
        return list(subset_indices)

    scalar_dims = set(access.scalar_dims)
    subset_iter = iter(subset_indices)
    result: List[Any] = []
    for dim, rng in enumerate(access.subset.ranges):
        if dim in scalar_dims:
            result.append(rng[0])
        else:
            result.append(next(subset_iter))
    return result


def _build_advanced_input_memlets(access: _ResolvedAccess,
                                  iteration_plan: _IterationPlan) -> Optional[Dict[str, Memlet]]:
    if access.blueprint is None:
        return None

    mapping = _build_access_symbol_mapping(iteration_plan, access.blueprint.output_subset, access.output_shape)
    if mapping is None:
        return None

    result = {
        access.array_connector:
        Memlet(data=access.name,
               subset=_substitute_subset(access.blueprint.source_memlet.subset, mapping),
               volume=access.blueprint.source_memlet.volume)
    }
    for connector, memlet in zip(access.index_connectors, access.blueprint.index_memlets):
        result[connector] = Memlet(data=memlet.data,
                                   subset=_substitute_subset(memlet.subset, mapping),
                                   volume=memlet.volume)
    return result


def _build_access_symbol_mapping(iteration_plan: _IterationPlan, output_subset: subsets.Range,
                                 operand_shape: Tuple[Any, ...]) -> Optional[Dict[Any, Any]]:
    varying_dims = [
        index for index, (start, end, step) in enumerate(output_subset.ranges)
        if step == 1 and start == end and symbolic.issymbolic(start)
    ]
    if not varying_dims:
        return {}
    try:
        _, _, _, _, operand_idx = broadcast_to(iteration_plan.squeezed_subset.size(), operand_shape)
    except Exception:
        return None
    operand_indices = [part.strip() for part in operand_idx.split(',')] if operand_idx else []
    if len(operand_indices) != len(output_subset.ranges):
        return None
    varying_operand_indices = [operand_indices[index] for index in varying_dims]
    symbols = [str(output_subset.ranges[index][0]) for index in varying_dims]
    if len(symbols) != len(varying_operand_indices):
        return None
    return {
        symbolic.symbol(symbol): symbolic.pystr_to_symbolic(index)
        for symbol, index in zip(symbols, varying_operand_indices)
    }


def _build_advanced_blueprint(name: str, subset: subsets.Range, new_axes: Sequence[int], arrdims: Dict[int, Any],
                              context: NumpyLoweringContext) -> Optional[_AdvancedIndexBlueprint]:
    output_shape = _compute_output_shape_from_advanced_indexing(subset, new_axes, arrdims, context)
    if output_shape is None:
        return None

    ndrange = subset.ndrange()
    output_ndrange = [(symbolic.symbol(f'__i{i}', dtypes.int64), symbolic.symbol(f'__i{i}', dtypes.int64),
                       1) if rng[0] != rng[1] else (0, 0, 1) for i, rng in enumerate(ndrange)]
    input_subset = subsets.Range([(rb + ind * rs, rb + ind * rs, 1)
                                  for (rb, _, rs), (ind, _, _) in zip(ndrange, output_ndrange)])
    index_memlets: List[Memlet] = []

    output_shape_marks = [size if index not in arrdims else None for index, size in enumerate(subset.size())]
    output_shape_marks = [None if rng[0] == rng[1] else size for size, rng in zip(output_shape_marks, subset.ndrange())]
    output_ndrange_marks: List[Optional[Tuple[Any, Any, Any]]] = [
        None if output_shape_marks[i] is None else rng for i, rng in enumerate(output_ndrange)
    ]

    advanced_dims = [
        index for index, size in enumerate(output_shape_marks)
        if size is None and (index == 0 or output_shape_marks[index - 1] is not None)
    ]
    prefix_dims = len(advanced_dims) > 1
    if prefix_dims:
        output_shape_marks = [None] + [size for size in output_shape_marks if size is not None]
        output_ndrange_marks = [None] + [rng for rng in output_ndrange_marks if rng is not None]
        dim_position = 0
    else:
        dim_position = advanced_dims[0]

    for new_axis in reversed(new_axes):
        if prefix_dims:
            output_shape_marks.insert(new_axis + 1, 1)
            output_ndrange_marks.insert(new_axis + 1, (0, 0, 1))
        else:
            output_shape_marks.insert(new_axis, 1)
            output_ndrange_marks.insert(new_axis, (0, 0, 1))
            if new_axis <= dim_position:
                dim_position += 1

    output_shape_marks = [
        size for index, size in enumerate(output_shape_marks)
        if size is not None or index == 0 or output_shape_marks[index - 1] is not None
    ]
    output_ndrange_marks = [
        rng for index, rng in enumerate(output_ndrange_marks)
        if rng is not None or index == 0 or output_ndrange_marks[index - 1] is not None
    ]

    advidx_shape: Optional[Tuple[Any, ...]] = None
    out_idx: Optional[str] = None
    advidx_arrays: Dict[int, Tuple[str, Sequence[Any]]] = {}
    for index, idxarrname in arrdims.items():
        if not isinstance(idxarrname, str):
            return None
        descriptor = _index_descriptor(idxarrname, context)
        if descriptor is None:
            return None
        advidx_arrays[index] = (idxarrname, descriptor.shape)
        if advidx_shape is not None:
            advidx_shape, _, out_idx, *_ = broadcast_together(descriptor.shape, advidx_shape)
        else:
            advidx_shape = tuple(descriptor.shape)
            out_idx = ', '.join(f'__i{i}' for i in range(len(descriptor.shape)))
    if advidx_shape is None or out_idx is None:
        return None

    out_idx = out_idx.replace('__i', '__ind')
    advidx_index: List[Tuple[Any, Any, Any]] = []
    for index_name, size in zip((part.strip() for part in out_idx.split(',')), advidx_shape):
        sym = symbolic.symbol(index_name, dtypes.int64)
        advidx_index.append((sym, sym, 1))

    for dim, (idxarrname, shape) in advidx_arrays.items():
        _, _, _, arr_idx, _ = broadcast_together(shape, advidx_shape)
        arr_idx = arr_idx.replace('__i', '__ind').split(',')
        arr_subset = subsets.Range([(symbolic.symbol(index.strip(),
                                                     dtypes.int64), symbolic.symbol(index.strip(), dtypes.int64), 1)
                                    for index in arr_idx])
        index_memlets.append(Memlet(data=idxarrname, subset=arr_subset, volume=1))
        input_subset[dim] = ndrange[dim]

    output_ndrange_final = output_ndrange_marks[:dim_position] + advidx_index + output_ndrange_marks[dim_position + 1:]
    return _AdvancedIndexBlueprint(output_shape=tuple(output_shape),
                                   output_subset=subsets.Range(output_ndrange_final),
                                   source_memlet=Memlet(data=name, subset=input_subset, volume=1),
                                   index_memlets=tuple(index_memlets))


def _compute_output_shape_from_advanced_indexing(subset: subsets.Range, new_axes: Sequence[int], arrdims: Dict[int,
                                                                                                               Any],
                                                 context: NumpyLoweringContext) -> Optional[List[Any]]:
    output_shape = [size if index not in arrdims else None for index, size in enumerate(subset.size())]
    if arrdims:
        output_shape = [None if rng[0] == rng[1] else size for size, rng in zip(output_shape, subset.ndrange())]

    advanced_dims = [
        index for index, size in enumerate(output_shape)
        if size is None and (index == 0 or output_shape[index - 1] is not None)
    ]
    prefix_dims = len(advanced_dims) > 1
    if prefix_dims:
        output_shape = [None] + [size for size in output_shape if size is not None]
        dim_position = 0
    else:
        dim_position = advanced_dims[0]

    for new_axis in new_axes:
        if prefix_dims:
            output_shape.insert(new_axis + 1, 1)
        else:
            output_shape.insert(new_axis, 1)
            if new_axis <= dim_position:
                dim_position += 1

    output_shape = [
        size for index, size in enumerate(output_shape)
        if size is not None or index == 0 or output_shape[index - 1] is not None
    ]

    chunk_shape: Optional[Tuple[Any, ...]] = None
    for arrname in arrdims.values():
        if not isinstance(arrname, str):
            return None
        descriptor = _index_descriptor(arrname, context)
        if descriptor is None:
            return None
        if chunk_shape is None:
            chunk_shape = tuple(descriptor.shape)
        else:
            try:
                chunk_shape, *_ = broadcast_together(descriptor.shape, chunk_shape)
            except Exception:
                return None

    if chunk_shape is None:
        return None
    return output_shape[:dim_position] + list(chunk_shape) + output_shape[dim_position + 1:]


def _varying_subset_symbols(subset: subsets.Range) -> List[str]:
    result: List[str] = []
    for start, end, step in subset.ranges:
        if step == 1 and start == end and symbolic.issymbolic(start):
            result.append(str(start))
    return result


def _substitute_subset(subset: subsets.Range, mapping: Dict[Any, Any]) -> subsets.Range:
    replaced = []
    for start, end, step in subset.ranges:
        replaced.append((_substitute_expr(start, mapping), _substitute_expr(end,
                                                                            mapping), _substitute_expr(step, mapping)))
    return subsets.Range(replaced)


def _substitute_expr(expr: Any, mapping: Dict[Any, Any]) -> Any:
    if isinstance(expr, symbolic.SymExpr):
        return symbolic.SymExpr(expr.expr.subs(mapping, simultaneous=True), expr.approx.subs(mapping,
                                                                                             simultaneous=True))
    if hasattr(expr, 'subs'):
        return expr.subs(mapping, simultaneous=True)
    return expr


def _compose_input_index(index_expr: str, subset: Tuple[Any, Any, Any]) -> Any:
    start, _, _ = subset
    if index_expr == '0':
        return start
    symbolic_index = symbolic.pystr_to_symbolic(index_expr)
    if symbolic.pystr_to_symbolic(str(start)) == 0:
        return symbolic_index
    return symbolic_index + start


def _broadcast_shape(shapes: Sequence[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
    concrete_shapes = [shape for shape in shapes if shape]
    if not concrete_shapes:
        return tuple()
    result = concrete_shapes[0]
    for shape in concrete_shapes[1:]:
        try:
            result, _, _, _, _ = broadcast_together(result, shape)
        except Exception:
            return None
    return tuple(result)


def _is_shape_compatible(target_subset: subsets.Range, source_shape: Tuple[Any, ...]) -> bool:
    try:
        broadcast_to(target_subset.size(), source_shape)
    except Exception:
        return False
    return True


def _is_shape_compatible_shape(target_shape: Sequence[Any], source_shape: Tuple[Any, ...]) -> bool:
    try:
        broadcast_to(tuple(target_shape), source_shape)
    except Exception:
        return False
    return True


def _scalar_type_environment(context: NumpyLoweringContext,
                             accesses: Sequence[_ResolvedAccess]) -> Dict[str, dtypes.typeclass]:
    result = {access.array_connector: access.descriptor.dtype for access in accesses}
    for name, binding in context.bindings.items():
        if binding.descriptor is not None and isinstance(binding.descriptor, data.Scalar):
            result[name] = binding.descriptor.dtype
    for name, value in context.evaluation_context().items():
        if isinstance(value, symbolic.symbol):
            result[name] = value.dtype
    return result


def _tasklet_expr_for_access(access: _ResolvedAccess) -> ast.AST:
    base = ast.Name(id=access.array_connector, ctx=ast.Load())
    if not access.index_connectors:
        return base
    if len(access.index_connectors) == 1:
        return ast.Subscript(value=base, slice=ast.Name(id=access.index_connectors[0], ctx=ast.Load()), ctx=ast.Load())
    return ast.Subscript(value=base,
                         slice=ast.Tuple(
                             elts=[ast.Name(id=connector, ctx=ast.Load()) for connector in access.index_connectors],
                             ctx=ast.Load()),
                         ctx=ast.Load())


def _resolve_boolean_gather(value: ast.AST, context: NumpyLoweringContext,
                            output_descriptor: Optional[data.Data]) -> Optional[_BooleanGatherPlan]:
    if not isinstance(value, ast.Subscript) or not isinstance(value.value, ast.Name):
        return None

    source_name = value.value.id
    binding = context.bindings.get(source_name)
    if binding is None or binding.descriptor is None or not _is_numpy_arraylike(binding.descriptor):
        return None
    source_descriptor = _clone_descriptor(binding.descriptor)

    try:
        subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(source_descriptor, value,
                                                                      context.evaluation_context())
    except Exception:
        subset, new_axes, arrdims = None, [], {}

    if subset is not None and arrdims:
        bool_indices = [index_name for index_name in arrdims.values() if _is_boolean_index(index_name, context)]
        if len(bool_indices) != 1 or len(arrdims) != 1 or new_axes:
            return None
        mask_name = bool_indices[0]
        mask_descriptor = _index_descriptor(mask_name, context)
        if mask_descriptor is None or tuple(mask_descriptor.shape) != tuple(source_descriptor.shape):
            return None
        result_descriptor = _boolean_gather_descriptor(source_descriptor, output_descriptor, context)
        return _BooleanGatherPlan(source_name=source_name,
                                  source_descriptor=source_descriptor,
                                  result_descriptor=result_descriptor,
                                  input_memlets={
                                      'data': Memlet.from_array(source_name, source_descriptor),
                                      'mask': Memlet.from_array(mask_name, mask_descriptor),
                                  },
                                  nnz_symbol=result_descriptor.shape[0])

    mask_analysis = _ElementwiseExpressionAnalyzer(context, start_index=100).analyze(value.slice)
    if mask_analysis is None or mask_analysis.result_dtype != dtypes.bool:
        return None
    if tuple(mask_analysis.result_shape) != tuple(source_descriptor.shape):
        return None
    if any(access.blueprint is not None for access in mask_analysis.accesses):
        return None

    input_memlets = {'data': Memlet.from_array(source_name, source_descriptor)}
    for access in mask_analysis.accesses:
        input_memlets[access.array_connector] = Memlet(data=access.name, subset=copy.deepcopy(access.subset))

    result_descriptor = _boolean_gather_descriptor(source_descriptor, output_descriptor, context)
    return _BooleanGatherPlan(source_name=source_name,
                              source_descriptor=source_descriptor,
                              result_descriptor=result_descriptor,
                              input_memlets=input_memlets,
                              nnz_symbol=result_descriptor.shape[0],
                              mask_expr=mask_analysis.tasklet_value)


def _boolean_gather_descriptor(source_descriptor: data.Data, output_descriptor: Optional[data.Data],
                               context: NumpyLoweringContext) -> data.Array:
    nnz_symbol = _boolean_gather_symbol(output_descriptor, context)
    bound = _shape_product(source_descriptor.shape)
    return data.Array(source_descriptor.dtype, [nnz_symbol], transient=True, total_size=bound)


def _boolean_gather_symbol(output_descriptor: Optional[data.Data], context: NumpyLoweringContext) -> symbolic.symbol:
    if isinstance(output_descriptor, data.Array) and len(output_descriptor.shape) == 1 and symbolic.issymbolic(
            output_descriptor.shape[0]):
        return symbolic.pystr_to_symbolic(str(output_descriptor.shape[0]))
    return context.fresh_symbol('__stree_mask_nnz')


def _shape_product(shape: Sequence[Any]) -> Any:
    result: Any = 1
    for dim in shape:
        result = result * dim
    return result


def _resolve_integer_target(context: NumpyLoweringContext, target: ast.AST) -> Optional[_AdvancedTarget]:
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
        return None
    binding = context.bindings.get(target.value.id)
    if binding is None or binding.descriptor is None or not _is_numpy_arraylike(binding.descriptor):
        return None
    try:
        subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(binding.descriptor, target,
                                                                      context.evaluation_context())
    except Exception:
        return None
    if not arrdims or any(_is_boolean_index(index_name, context) for index_name in arrdims.values()):
        return None
    blueprint = _build_advanced_blueprint(target.value.id, subset, new_axes, arrdims, context)
    if blueprint is None:
        return None
    iteration_plan = _build_iteration_plan_from_shape(blueprint.output_shape)
    if iteration_plan is None:
        return None
    mapping = _build_access_symbol_mapping(iteration_plan, blueprint.output_subset, blueprint.output_shape)
    if mapping is None:
        return None
    access_index = 1000
    index_connectors = tuple(f'outidx_{i}' for i in range(len(blueprint.index_memlets)))
    input_memlets = {
        connector: Memlet(data=memlet.data, subset=_substitute_subset(memlet.subset, mapping), volume=memlet.volume)
        for connector, memlet in zip(index_connectors, blueprint.index_memlets)
    }
    output_memlet = Memlet(data=target.value.id,
                           subset=_substitute_subset(blueprint.source_memlet.subset, mapping),
                           volume=blueprint.source_memlet.volume)
    return _AdvancedTarget(name=target.value.id,
                           output_shape=blueprint.output_shape,
                           output_memlet=output_memlet,
                           target_expr=_subscript_expr('out', index_connectors),
                           input_memlets=input_memlets)


def _resolve_boolean_target(context: NumpyLoweringContext, target: ast.AST,
                            value: ast.AST) -> Optional[_AdvancedTarget]:
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
        return None
    binding = context.bindings.get(target.value.id)
    if binding is None or binding.descriptor is None or not _is_numpy_arraylike(binding.descriptor):
        return None

    subset: Optional[subsets.Range] = None
    guard_expr: Optional[ast.AST] = None
    input_memlets: Dict[str, Memlet] = {}
    target_shape: Optional[Tuple[Any, ...]] = None

    try:
        subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(binding.descriptor, target,
                                                                      context.evaluation_context())
    except Exception:
        subset, new_axes, arrdims = None, [], {}

    if subset is not None and arrdims:
        bool_indices = [index_name for index_name in arrdims.values() if _is_boolean_index(index_name, context)]
        if len(bool_indices) != 1 or len(arrdims) != 1 or new_axes:
            return None
        bool_name = bool_indices[0]
        mask_desc = _index_descriptor(bool_name, context)
        if mask_desc is None or tuple(mask_desc.shape) != tuple(binding.descriptor.shape):
            return None
        target_shape = tuple(_shape_from_basic_subset(subset))
        iteration_plan = _build_iteration_plan(subset)
        if iteration_plan is None:
            return None
        mask_subset = _build_output_memlet(bool_name, iteration_plan).subset
        input_memlets['mask'] = Memlet(data=bool_name, subset=mask_subset)
        guard_expr = ast.Name(id='mask', ctx=ast.Load())
        output_memlet = Memlet(data=target.value.id,
                               subset=_build_output_memlet(target.value.id, iteration_plan).subset)
        return _AdvancedTarget(name=target.value.id,
                               output_shape=target_shape,
                               output_memlet=output_memlet,
                               target_expr=ast.Name(id='out', ctx=ast.Store()),
                               input_memlets=input_memlets,
                               guard_expr=guard_expr)

    if isinstance(target.slice, ast.Compare):
        subset = subsets.Range.from_array(binding.descriptor)
        target_shape = tuple(binding.descriptor.shape)
        iteration_plan = _build_iteration_plan(subset)
        if iteration_plan is None:
            return None
        mask_analysis = _ElementwiseExpressionAnalyzer(context, start_index=100).analyze(target.slice)
        if mask_analysis is None:
            return None
        if mask_analysis.result_shape and not _is_shape_compatible(subset, mask_analysis.result_shape):
            return None
        for access in mask_analysis.accesses:
            access_memlets = _build_input_memlets(access, iteration_plan)
            if access_memlets is None:
                return None
            input_memlets.update(access_memlets)
        output_memlet = Memlet(data=target.value.id,
                               subset=_build_output_memlet(target.value.id, iteration_plan).subset)
        return _AdvancedTarget(name=target.value.id,
                               output_shape=target_shape,
                               output_memlet=output_memlet,
                               target_expr=ast.Name(id='out', ctx=ast.Store()),
                               input_memlets=input_memlets,
                               guard_expr=mask_analysis.tasklet_value)

    return None


def _subscript_expr(base_name: str, connectors: Sequence[str]) -> ast.AST:
    if len(connectors) == 1:
        return ast.Subscript(value=ast.Name(id=base_name, ctx=ast.Load()),
                             slice=ast.Name(id=connectors[0], ctx=ast.Load()),
                             ctx=ast.Store())
    return ast.Subscript(value=ast.Name(id=base_name, ctx=ast.Load()),
                         slice=ast.Tuple(elts=[ast.Name(id=connector, ctx=ast.Load()) for connector in connectors],
                                         ctx=ast.Load()),
                         ctx=ast.Store())


def _ast_equivalent(left: ast.AST, right: ast.AST) -> bool:
    return _unparse(left) == _unparse(right)


def _is_trivial_scalar(node: ast.AST, context: NumpyLoweringContext) -> bool:
    if isinstance(node, ast.Constant):
        return True
    return _is_scalar_leaf(node, context)


def _is_supported_call(node: ast.Call, context: NumpyLoweringContext) -> bool:
    if astutils.rname(node.func) == 'abs':
        return True
    value = try_resolve_static_value(node.func, context.evaluation_context())
    if value is UNRESOLVED:
        return False
    return isinstance(value, np.ufunc)


def _is_scalar_leaf(node: ast.AST, context: NumpyLoweringContext) -> bool:
    if isinstance(node, ast.Name):
        binding = context.bindings.get(node.id)
        if binding is not None and binding.descriptor is not None:
            return isinstance(binding.descriptor, data.Scalar)
    value = try_resolve_static_value(node, context.evaluation_context())
    if value is UNRESOLVED:
        return False
    if isinstance(value, symbolic.symbol):
        return True
    if isinstance(value, data.Data):
        return isinstance(value, data.Scalar)
    return isinstance(value, (numbers.Number, bool, str, bytes, type(None))) or symbolic.issymbolic(value)


def _shape_from_basic_subset(subset: subsets.Range) -> List[Any]:
    squeezed = copy.deepcopy(subset)
    squeezed.squeeze(offset=False)
    return list(squeezed.size())


def _shape_from_ranges(ranges: Sequence[Tuple[Any, Any, Any]]) -> List[Any]:
    if not ranges:
        return []
    return list(subsets.Range(list(ranges)).size())


def _logical_ranges_from_basic_access(node: ast.AST, subset: subsets.Range, descriptor: data.Data,
                                      context: NumpyLoweringContext) -> Tuple[Tuple[Any, Any, Any], ...]:
    if isinstance(node, ast.Name):
        return tuple(copy.deepcopy(subset).ranges)

    scalar_dims = set(_scalar_indexed_dims(node, len(descriptor.shape), context))
    return tuple(copy.deepcopy(rng) for index, rng in enumerate(subset.ranges) if index not in scalar_dims)


def _scalar_indexed_dims(node: ast.AST, rank: int, context: NumpyLoweringContext) -> Tuple[int, ...]:
    if not isinstance(node, ast.Subscript):
        return tuple()

    remaining_dims = list(range(rank))
    scalar_dims: List[int] = []
    for ast_ndslice in astutils.subscript_to_ast_slice_recursive(node):
        expanded_dims = _expand_basic_slice_dims(ast_ndslice, len(remaining_dims), context)
        next_remaining: List[int] = []
        remaining_iter = iter(remaining_dims)
        for kind in expanded_dims:
            if kind == 'newaxis':
                continue
            base_dim = next(remaining_iter)
            if kind == 'scalar':
                scalar_dims.append(base_dim)
            else:
                next_remaining.append(base_dim)
        next_remaining.extend(remaining_iter)
        remaining_dims = next_remaining

    return tuple(scalar_dims)


def _expand_basic_slice_dims(ast_ndslice: Sequence[Any], remaining_rank: int,
                             context: NumpyLoweringContext) -> List[str]:
    kinds = [_basic_dim_kind(dim, context) for dim in ast_ndslice]
    consumed_dims = sum(1 for kind in kinds if kind not in {'newaxis', 'ellipsis'})
    if 'ellipsis' not in kinds:
        kinds.extend(['slice'] * max(0, remaining_rank - consumed_dims))
        return kinds

    expanded: List[str] = []
    for kind in kinds:
        if kind != 'ellipsis':
            expanded.append(kind)
            continue
        ellipsis_dims = max(0, remaining_rank - consumed_dims)
        expanded.extend(['slice'] * ellipsis_dims)
    return expanded


def _basic_dim_kind(dim: Any, context: NumpyLoweringContext) -> str:
    if isinstance(dim, tuple):
        return 'slice'
    if dim is None or (isinstance(dim, ast.Constant) and dim.value is None):
        return 'newaxis'
    if dim is Ellipsis or (isinstance(dim, ast.Constant) and dim.value is Ellipsis):
        return 'ellipsis'

    resolved = try_resolve_static_value(dim, context.evaluation_context())
    if isinstance(resolved, slice):
        return 'slice'

    return 'scalar'


def _is_scalar_subscript(node: ast.AST, subset: subsets.Range) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if isinstance(node.slice, ast.Slice):
        return False
    for (start, end, step), tile in zip(subset.ranges, subset.tile_sizes):
        if tile != 1 or step != 1 or start != end:
            return False
    return True


def _is_boolean_index(index_name: Any, context: NumpyLoweringContext) -> bool:
    if not isinstance(index_name, str):
        return False
    descriptor = _index_descriptor(index_name, context)
    return descriptor is not None and descriptor.dtype == dtypes.bool


def _index_descriptor(index_name: str, context: NumpyLoweringContext) -> Optional[data.Data]:
    binding = context.bindings.get(index_name)
    if binding is not None and binding.descriptor is not None:
        return binding.descriptor
    value = context.evaluation_context().get(index_name)
    if isinstance(value, data.Data):
        return value
    return None


def _is_numpy_arraylike(descriptor: data.Data) -> bool:
    return not isinstance(descriptor,
                          (data.Scalar, PythonDict, PythonList, PythonTuple)) and hasattr(descriptor, 'shape')


def _frontend_range_tuple(dim: Tuple[Any, Any, Any]) -> Tuple[str, str, str]:
    start, end, step = dim
    offset = -1 if (step < 0) == True else 1
    return (str(start), str(end + offset), str(step))


def _clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def _unparse(node: ast.AST) -> str:
    return astutils.unparse(copy.deepcopy(node))
