# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generic expression planning helpers for the direct schedule-tree frontend.

Terminology
-----------
"Expression planning" is not meant as a standardized compiler term here.
Within the schedule-tree frontend it is an internal shorthand for the step
that decides:

1. which array-valued subexpressions should be materialized into temporaries,
2. in what order they should be materialized, and
3. which lowering path should handle each materialized step.

Conceptually this is closest to turning nested expressions into a restricted
3-address-code / A-normal-form style representation before schedule-tree
lowering. The goal is not to preserve a single opaque source expression, but to
expose the intermediate array operations that later passes can lower as maps,
library calls, or fallback tasklets.

Examples
--------
Nested call arguments:

    Source:
        inner(A + 1, B + 2)

    Planned form:
        __stree_tmp = A + 1
        __stree_tmp1 = B + 2
        inner(__stree_tmp, __stree_tmp1)

    Lowering effect:
        The two temporary assignments can each become explicit elementwise map
        scopes through the NumPy lowering layer, while the call itself sees only
        simple array arguments.

Array-valued returns:

    Source:
        return A + B

    Planned form:
        __stree_tmp = A + B
        return __stree_tmp

    Lowering effect:
        The returned expression is no longer an opaque return value; it becomes
        a normal assignment that can be lowered structurally before the final
        ReturnNode.

Chained matmul:

    Source:
        return A @ B @ C

    Planned form:
        __stree_tmp = A @ B
        __stree_tmp1 = __stree_tmp @ C
        return __stree_tmp1

    Lowering effect:
        Each matmul step can lower independently as a schedule-tree library call
        instead of treating the whole chain as one opaque expression.
"""

import ast
import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from dace import data, dtypes
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import astutils
from dace.frontend.python.replacements.utils import broadcast_together
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn

DescriptorInferer = Callable[[ast.AST], Optional[data.Data]]
ExpressionMaterializer = Callable[[ast.AST, data.Data], ast.AST]
DataAccessResolver = Callable[[ast.AST], Optional[Tuple[str, Memlet, data.Data, Optional[data.Data]]]]
InputMemletCollector = Callable[[ast.AST], Dict[str, Memlet]]
OutputTargetResolver = Callable[[ast.AST, ast.AST, Optional[data.Data]], Optional[Tuple[str, Memlet, data.Data]]]
CallableNameResolver = Callable[[ast.AST], str]
CallMaterializationPredicate = Callable[[ast.Call], bool]


@dataclass(frozen=True)
class ExpressionPlanningContext:
    """Callbacks needed by the expression planner.

    The planner is deliberately frontend-agnostic: it does not own descriptor
    repositories or emit schedule-tree nodes by itself. Instead it asks the
    surrounding builder how to infer descriptors, how to materialize a chosen
    subexpression, and how to resolve accesses/memlets for lowering passes.
    """

    infer_descriptor: DescriptorInferer
    materialize_expression: ExpressionMaterializer
    resolve_data_access: DataAccessResolver
    collect_input_memlets: InputMemletCollector
    resolve_output_target: OutputTargetResolver
    resolve_callable_name: Optional[CallableNameResolver] = None
    should_materialize_call: Optional[CallMaterializationPredicate] = None


class GenericExpressionSupportLibrary:
    """Planning and lowering helpers for non-trivial array-valued expressions.

    This module currently has two responsibilities:

    1. Rewrite nested array-valued expressions into a sequence of simpler
       expressions by materializing selected subexpressions into temporaries.
    2. Provide lowering hooks for expression forms that are better handled by a
       dedicated pass than by the generic NumPy/tasklet fallback.

    At the moment the dedicated lowering pass covers matmul. Other array-valued
    expressions are planned here and then handed back to the builder, which in
    turn routes them through the NumPy support layer or the generic fallback.
    """

    def __init__(self) -> None:
        self.assignment_passes = (_OperatorAssignmentPass(), )

    def plan_expression(self, context: ExpressionPlanningContext, node: ast.AST, *, materialize_root: bool) -> ast.AST:
        return _ExpressionPlanner(context).rewrite(node, materialize_root=materialize_root)

    def lower_assignment(self, context: ExpressionPlanningContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        for lowering_pass in self.assignment_passes:
            lowered = lowering_pass.lower_assignment(context, target, value, annotated_descriptor)
            if lowered is not None:
                return lowered
        return None

    def infer_expression_descriptor(self, context: ExpressionPlanningContext, node: ast.AST) -> Optional[data.Data]:
        for lowering_pass in self.assignment_passes:
            descriptor = lowering_pass.infer_expression_descriptor(context, node)
            if descriptor is not None:
                return descriptor
        return None


class _ExpressionPlanner:

    def __init__(self, context: ExpressionPlanningContext) -> None:
        self.context = context

    def rewrite(self, node: ast.AST, *, materialize_root: bool) -> ast.AST:
        """Rewrite an expression tree into a planned form.

        If ``materialize_root`` is true and the final expression is still a
        non-trivial array expression, the root expression is also turned into a
        temporary. This is used for contexts such as ``return A + B``, where the
        frontend wants to lower the array expression structurally before
        emitting the final ReturnNode.
        """

        rewritten = self._rewrite(copy.deepcopy(node))
        if materialize_root and self._should_materialize(rewritten):
            return self._materialize(rewritten)
        return rewritten

    def _rewrite(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.BinOp):
            return ast.copy_location(
                ast.BinOp(left=self._rewrite_binop_child(node.left, node.right),
                          op=copy.deepcopy(node.op),
                          right=self._rewrite_binop_child(node.right, node.left)), node)

        if isinstance(node, ast.UnaryOp):
            return ast.copy_location(ast.UnaryOp(op=copy.deepcopy(node.op), operand=self._rewrite_child(node.operand)),
                                     node)

        if isinstance(node, ast.BoolOp):
            return ast.copy_location(
                ast.BoolOp(op=copy.deepcopy(node.op), values=[self._rewrite_child(value) for value in node.values]),
                node)

        if isinstance(node, ast.Compare):
            return ast.copy_location(
                ast.Compare(left=self._rewrite_child(node.left),
                            ops=copy.deepcopy(node.ops),
                            comparators=[self._rewrite_child(comp) for comp in node.comparators]), node)

        if isinstance(node, ast.IfExp):
            return ast.copy_location(
                ast.IfExp(test=self._rewrite_child(node.test),
                          body=self._rewrite_child(node.body),
                          orelse=self._rewrite_child(node.orelse)), node)

        if isinstance(node, ast.Attribute):
            return ast.copy_location(ast.Attribute(value=self._rewrite_child(node.value), attr=node.attr, ctx=node.ctx),
                                     node)

        if isinstance(node, ast.Call):
            iterator_protocol_call = self._is_iterator_protocol_call(node)
            array_constructor_call = self._is_array_constructor_call(node)
            return ast.copy_location(
                ast.Call(func=self._rewrite_call_func(node.func, iterator_protocol_call=iterator_protocol_call),
                         args=[
                             self._rewrite_child(arg,
                                                 materialize_pyobject_call=iterator_protocol_call,
                                                 preserve_array_literal=array_constructor_call and index == 0)
                             for index, arg in enumerate(node.args)
                         ],
                         keywords=[
                             ast.keyword(arg=kw.arg,
                                         value=self._rewrite_child(kw.value,
                                                                   materialize_pyobject_call=iterator_protocol_call))
                             for kw in node.keywords
                         ]), node)

        if isinstance(node, ast.Tuple):
            return ast.copy_location(ast.Tuple(elts=[self._rewrite_child(elt) for elt in node.elts], ctx=node.ctx),
                                     node)

        if isinstance(node, ast.List):
            return ast.copy_location(ast.List(elts=[self._rewrite_child(elt) for elt in node.elts], ctx=node.ctx), node)

        return node

    def _rewrite_call_func(self, func: ast.AST, *, iterator_protocol_call: bool) -> ast.AST:
        if isinstance(func, ast.Attribute):
            return ast.copy_location(
                ast.Attribute(value=self._rewrite_child(func.value, materialize_pyobject_call=iterator_protocol_call),
                              attr=func.attr,
                              ctx=func.ctx), func)
        return copy.deepcopy(func)

    def _rewrite_child(self,
                       node: ast.AST,
                       *,
                       materialize_pyobject_call: bool = False,
                       preserve_array_literal: bool = False) -> ast.AST:
        rewritten = self._rewrite(copy.deepcopy(node))
        if preserve_array_literal:
            return rewritten
        if self._should_materialize(rewritten):
            return self._materialize(rewritten)
        if materialize_pyobject_call and self._should_materialize_pyobject_call(rewritten):
            return self._materialize(rewritten)
        return rewritten

    def _rewrite_binop_child(self, node: ast.AST, sibling: ast.AST) -> ast.AST:
        rewritten = self._rewrite(copy.deepcopy(node))
        array_literal_descriptor = None
        if isinstance(rewritten, (ast.List, ast.Tuple)):
            array_literal_descriptor = self.context.infer_descriptor(
                ast.Call(func=ast.Attribute(value=ast.Name(id='numpy', ctx=ast.Load()), attr='array', ctx=ast.Load()),
                         args=[copy.deepcopy(rewritten)],
                         keywords=[]))
            sibling_descriptor = self.context.infer_descriptor(copy.deepcopy(sibling))
            if (array_literal_descriptor is not None and sibling_descriptor is not None
                    and not isinstance(sibling_descriptor, data.Scalar)):
                return self.context.materialize_expression(rewritten, array_literal_descriptor)

        if self._should_materialize(rewritten):
            return self._materialize(rewritten)
        return rewritten

    def _should_materialize(self, node: ast.AST) -> bool:
        """Return whether ``node`` should become a temporary.

        The planner only materializes array-valued expressions that are not
        already simple data accesses. Scalars remain inline. Plain accesses such
        as ``A`` or ``A[i:j]`` stay inline as well; they are already representable
        without introducing extra storage.
        """

        if (isinstance(node, ast.Call) and self.context.should_materialize_call is not None
                and self.context.should_materialize_call(node)):
            return True

        descriptor = self.context.infer_descriptor(node)
        if descriptor is None:
            return False
        if isinstance(descriptor, data.Scalar):
            return isinstance(node, ast.Call)
        if self.context.resolve_data_access(node) is not None:
            return False
        return isinstance(node, (ast.Attribute, ast.BinOp, ast.BoolOp, ast.Call, ast.Compare, ast.IfExp, ast.UnaryOp))

    def _should_materialize_pyobject_call(self, node: ast.AST) -> bool:
        descriptor = self.context.infer_descriptor(node)
        return isinstance(node, ast.Call) and isinstance(descriptor, data.Scalar) and isinstance(
            descriptor.dtype, dtypes.pyobject)

    def _is_iterator_protocol_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            return node.func.id in {'iter', 'next', '__dace_iterator_init', '__dace_iterator_next'}
        return isinstance(node.func, ast.Attribute) and node.func.attr == '__next__'

    def _is_array_constructor_call(self, node: ast.Call) -> bool:
        if self.context.resolve_callable_name is not None:
            return self.context.resolve_callable_name(node.func) == 'numpy.array'
        return astutils.rname(node.func) == 'numpy.array'

    def _materialize(self, node: ast.AST) -> ast.AST:
        descriptor = self.context.infer_descriptor(node)
        if (descriptor is None and isinstance(node, ast.Call) and self.context.should_materialize_call is not None
                and self.context.should_materialize_call(node)):
            descriptor = data.Scalar(dtypes.pyobject(), transient=True)
        if descriptor is None:
            return node
        return self.context.materialize_expression(node, descriptor)


class _OperatorAssignmentPass:
    """Lower materialized binary operator assignments as frontend library calls.

    Uses the operator descriptor-inference registry to handle any binary
    operator that has a registered inference function (currently ``@`` / MatMult).
    The planner first linearizes chains such as ``A @ B @ C`` into temporary
    assignments, then each individual assignment is recognized here.
    """

    # Maps AST operator class -> (registry_name, library_name)
    _OP_MAP = {ast.MatMult: ('MatMult', 'MatMul')}

    def lower_assignment(self, context: ExpressionPlanningContext, target: ast.AST, value: ast.AST,
                         annotated_descriptor: Optional[data.Data]) -> Optional[tn.ScheduleTreeNode]:
        if not isinstance(value, ast.BinOp):
            return None
        entry = self._OP_MAP.get(type(value.op))
        if entry is None:
            return None
        registry_name, library_name = entry
        descriptor = self.infer_expression_descriptor(context, value)
        if descriptor is None or isinstance(descriptor, data.Scalar):
            return None
        output = context.resolve_output_target(target, value, annotated_descriptor)
        if output is None:
            return None
        _, out_memlet, _ = output
        in_memlets = context.collect_input_memlets(value)
        if len(in_memlets) != 2:
            return None
        return tn.LibraryCall(node=tn.FrontendLibrary(name=library_name,
                                                      properties=self._operator_properties(registry_name)),
                              in_memlets=in_memlets,
                              out_memlets={'out': out_memlet})

    def infer_expression_descriptor(self, context: ExpressionPlanningContext, node: ast.AST) -> Optional[data.Data]:
        if not isinstance(node, ast.BinOp):
            return None
        entry = self._OP_MAP.get(type(node.op))
        if entry is None:
            return None
        registry_name, _library_name = entry

        left_descriptor = context.infer_descriptor(node.left)
        right_descriptor = context.infer_descriptor(node.right)
        if left_descriptor is None or right_descriptor is None:
            return None

        infer_fn = oprepo.Replacements.get_operator_descriptor_inference(registry_name)
        if infer_fn is not None:
            try:
                result = infer_fn(left_descriptor, right_descriptor)
                if result is not None:
                    return result
            except Exception:
                pass

        return None

    @staticmethod
    def _operator_properties(registry_name: str) -> dict:
        if registry_name == 'MatMult':
            return {'alpha': 1, 'beta': 0}
        return {}


def _matmul_output_shape(left_shape: Tuple[object, ...], right_shape: Tuple[object,
                                                                            ...]) -> Optional[Tuple[object, ...]]:
    """Infer the result shape for NumPy-style matmul semantics.

    This mirrors the subset of ``numpy.matmul`` shape rules that the direct
    schedule-tree frontend currently lowers structurally: vector-vector,
    matrix-vector, vector-matrix, matrix-matrix, and batched matrix-matrix.
    """

    if len(left_shape) == 1 and len(right_shape) == 1:
        return tuple()

    if len(left_shape) == 2 and len(right_shape) == 1:
        return (left_shape[0], )

    if len(left_shape) == 1 and len(right_shape) == 2:
        return (right_shape[1], )

    if len(left_shape) < 2 or len(right_shape) < 2:
        return None

    batch_shape = _broadcast_prefix_shapes(left_shape[:-2], right_shape[:-2])
    if batch_shape is None:
        return None
    return batch_shape + (left_shape[-2], right_shape[-1])


def _broadcast_prefix_shapes(left_prefix: Tuple[object, ...], right_prefix: Tuple[object,
                                                                                  ...]) -> Optional[Tuple[object, ...]]:
    if not left_prefix:
        return tuple(right_prefix)
    if not right_prefix:
        return tuple(left_prefix)
    try:
        result, _, _, _, _ = broadcast_together(left_prefix, right_prefix)
    except Exception:
        return None
    return tuple(result)
