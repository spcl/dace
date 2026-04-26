# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python frontend entry point for building schedule trees directly from AST."""

import ast
import builtins as pybuiltins
import copy
import inspect
import numbers
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dace import data, dtypes, symbolic, subsets
from dace.config import Config
from dace.data.pydata import PythonClass, PythonDict, PythonList, PythonTuple
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python import astutils, memlet_parser, preprocessing
from dace.frontend.python.schedule_tree.array_literal_support import ArrayLiteralContext, ArrayLiteralSupportLibrary
from dace.frontend.python.schedule_tree.dict_support import DictSupportContext, DictSupportLibrary, StaticDictBinding
from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver
from dace.frontend.python.schedule_tree.structure_support import (
    descriptor_from_structure, direct_class_annotation_type, ensure_nested_member_descriptor, nested_direct_class_owner,
    python_class_requirement_for_member_assignment, resolve_member_access)
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.frontend.python.schedule_tree.match_support import UnsupportedMatchPatternError, lower_match_to_statements
from dace.frontend.python.schedule_tree import (AttributeRewriter, ExpressionPlanningContext, CallbackHandler,
                                                CallableArgumentSpecializer, CallableResolver,
                                                GenericExpressionSupportLibrary, NumpyLoweringContext,
                                                NumpySupportLibrary, ScheduleTreeTypeInference, _Binding,
                                                callback_reason, desugar_schedule_tree_expansions,
                                                promote_dynamic_scope_copies, resolve_function_calls)
from dace.frontend.python.schedule_tree.type_inference import _binding_from_inference_result, \
    _infer_static_subscript_descriptor, _resolve_ufunc_inference_target
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.type_inference import infer_expr_type

_INTERNAL_ITERATOR_HELPERS = {
    '__dace_iterator_init',
    '__dace_iterator_next',
}

_SUPPORTED_RAISE_BEHAVIORS = {'support', 'ignore_dynamic', 'ignore_all'}


def _normalize_raise_behavior(value: Any) -> str:
    normalized = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    if normalized in _SUPPORTED_RAISE_BEHAVIORS:
        return normalized
    return 'support'


def _clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def _clone_binding(binding: _Binding) -> _Binding:
    descriptor = _clone_descriptor(binding.descriptor) if binding.descriptor is not None else None
    return _Binding(descriptor=descriptor, kind=binding.kind, structure=copy.deepcopy(binding.structure))


def _unparse(node: ast.AST) -> str:
    try:
        working_node = astutils.copy_tree(node)
    except Exception:
        working_node = node
    sanitized = _sanitize_ast_for_unparse(working_node)
    return astutils.unparse(sanitized)


def _sanitize_ast_for_unparse(node: ast.AST) -> ast.AST:

    class _ConstantNormalizer(ast.NodeTransformer):

        def visit_Constant(self, constant: ast.Constant) -> ast.AST:
            replacement_name = _constant_source_name(constant.value)
            if replacement_name is None:
                return constant
            return ast.copy_location(ast.Name(id=replacement_name, ctx=ast.Load()), constant)

    sanitized = _ConstantNormalizer().visit(node)
    return ast.fix_missing_locations(sanitized)


def _constant_source_name(value: Any) -> Optional[str]:
    if isinstance(value, (str, bytes, numbers.Number, bool, type(None), type(Ellipsis))):
        return None

    candidate = None
    if hasattr(value, 'f') and hasattr(value.f, '__name__'):
        candidate = value.f.__name__
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, 'name', None)
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, '__name__', None)
    if not isinstance(candidate, str) or not candidate:
        candidate = getattr(value, '__qualname__', None)
        if isinstance(candidate, str) and candidate:
            candidate = candidate.split('.')[-1]

    if not isinstance(candidate, str) or not candidate:
        return None

    sanitized = re.sub(r'\W|^(?=\d)', '_', candidate)
    if not sanitized or sanitized in {'True', 'False', 'None'}:
        return None
    return sanitized


def _normalize_dtype(dtype: Any) -> Optional[dtypes.typeclass]:
    if isinstance(dtype, dtypes.typeclass):
        return dtype
    if isinstance(dtype, data.Data):
        return dtype.dtype
    if dtype in (int, float, complex, bool):
        return dtypes.typeclass(dtype)
    if dtype is str:
        return dtypes.string
    try:
        return dtypes.typeclass(dtype)
    except (KeyError, TypeError, ValueError):
        return None


def _pyobject_scalar_descriptor() -> data.Scalar:
    return data.Scalar(dtypes.pyobject(), transient=True)


def _is_pyobject_scalar_descriptor(descriptor: Optional[data.Data]) -> bool:
    return isinstance(descriptor, data.Scalar) and isinstance(descriptor.dtype, dtypes.pyobject)


def _is_iterator_next_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (astutils.rname(node.func) == 'next' or
                                           (isinstance(node.func, ast.Attribute) and node.func.attr == '__next__'))


def _is_iterator_protocol_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (astutils.rname(
        node.func) in {'iter', '__dace_iterator_init', '__dace_iterator_next'} or _is_iterator_next_call(node))


def _is_singleton_scalar_memlet(memlet: Memlet) -> bool:
    subset = memlet.subset
    if not isinstance(subset, subsets.Range):
        return False
    try:
        return subset.num_elements() == 1
    except Exception:
        return False


def _string_scalar_descriptor() -> data.Scalar:
    return data.Scalar(dtypes.string, transient=True)


def _binding_to_descriptor(value: Any) -> data.Data:
    if isinstance(value, data.Data):
        descriptor = _clone_descriptor(value)
    else:
        descriptor = _clone_descriptor(data.create_datadescriptor(value))

    if isinstance(descriptor, data.View):
        descriptor = descriptor.as_array()
    descriptor.transient = False
    return descriptor


def _binding_kind_for_descriptor(descriptor: data.Data) -> str:
    if isinstance(descriptor, data.Reference):
        return 'reference'
    if isinstance(descriptor, data.Scalar):
        if isinstance(descriptor.dtype, dtypes.callback):
            return 'callback'
        return 'scalar'
    return 'container'


def _collect_scope_declarations(node: ast.AST) -> Tuple[set[str], set[str]]:

    class _ScopeDeclarationCollector(ast.NodeVisitor):

        def __init__(self) -> None:
            self.global_names: set[str] = set()
            self.nonlocal_names: set[str] = set()

        def visit_Global(self, global_node: ast.Global) -> None:
            self.global_names.update(global_node.names)

        def visit_Nonlocal(self, nonlocal_node: ast.Nonlocal) -> None:
            self.nonlocal_names.update(nonlocal_node.names)

        def visit_FunctionDef(self, nested_node: ast.FunctionDef) -> None:
            if nested_node is node:
                for stmt in nested_node.body:
                    self.visit(stmt)

        def visit_AsyncFunctionDef(self, nested_node: ast.AsyncFunctionDef) -> None:
            if nested_node is node:
                for stmt in nested_node.body:
                    self.visit(stmt)

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            if lambda_node is node:
                self.generic_visit(lambda_node.body)

        def visit_ClassDef(self, _: ast.ClassDef) -> None:
            return

    collector = _ScopeDeclarationCollector()
    collector.visit(node)
    return collector.global_names, collector.nonlocal_names


def _function_signature_from_ast(node: ast.FunctionDef) -> inspect.Signature:
    parameters: List[inspect.Parameter] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    positional_defaults = list(node.args.defaults)
    positional_default_offset = len(positional) - len(positional_defaults)

    for index, arg in enumerate(node.args.posonlyargs):
        default = inspect._empty if index < positional_default_offset else object()
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY, default=default))

    for index, arg in enumerate(node.args.args, start=len(node.args.posonlyargs)):
        default = inspect._empty if index < positional_default_offset else object()
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))

    if node.args.vararg is not None:
        parameters.append(inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))

    for arg, default_value in zip(node.args.kwonlyargs, node.args.kw_defaults):
        default = inspect._empty if default_value is None else object()
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=default))

    if node.args.kwarg is not None:
        parameters.append(inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))

    return inspect.Signature(parameters)


class _NestedFunctionProgram:
    """AST-backed inline callee used for known nested FunctionDefs."""

    _schedule_tree_inline_callable = True

    def __init__(self, name: str, function_ast: ast.FunctionDef, *, program_globals: Dict[str, Any],
                 external_globals: Dict[str, Any], captured_names: set[str], constants: Dict[str, Tuple[data.Data,
                                                                                                        Any]],
                 callback_mapping: Dict[str, str], seed_bindings: Dict[str, _Binding],
                 lambda_bindings: Dict[str, ast.Lambda], callable_bindings: Dict[str, Any]) -> None:
        self.name = name
        self.function_ast = ast.fix_missing_locations(astutils.copy_tree(function_ast))
        self.program_globals = copy.copy(program_globals)
        self.external_globals = copy.copy(external_globals)
        self.captured_names = set(captured_names)
        self.constants = {key: (_clone_descriptor(desc), value) for key, (desc, value) in constants.items()}
        self.callback_mapping = dict(callback_mapping)
        self.seed_bindings = {key: _clone_binding(binding) for key, binding in seed_bindings.items()}
        self.lambda_bindings = {key: astutils.copy_tree(value) for key, value in lambda_bindings.items()}
        self.callable_bindings = dict(callable_bindings)
        self.signature = _function_signature_from_ast(function_ast)
        self.argnames = [parameter.name for parameter in self.signature.parameters.values()]

    def __descriptor__(self) -> data.Data:
        return data.Scalar(dtypes.callback(None))

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_NestedFunctionProgram':
        memo[id(self)] = self
        return self

    def _generate_schedule_tree(self,
                                args: Tuple[Any],
                                kwargs: Dict[str, Any],
                                *,
                                lambda_bindings: Optional[Dict[str, ast.Lambda]] = None,
                                callable_bindings: Optional[Dict[str, Any]] = None) -> tn.ScheduleTreeRoot:
        bound_args = self.signature.bind_partial(*args, **kwargs)
        argtypes = {name: _binding_to_descriptor(value) for name, value in bound_args.arguments.items()}

        active_lambda_bindings = {key: astutils.copy_tree(value) for key, value in self.lambda_bindings.items()}
        active_lambda_bindings.update({
            key: astutils.copy_tree(value)
            for key, value in (lambda_bindings or {}).items()
        })

        active_callable_bindings = dict(self.callable_bindings)
        active_callable_bindings.update(dict(callable_bindings or {}))

        seed_bindings = {
            key: _clone_binding(binding)
            for key, binding in self.seed_bindings.items() if key not in bound_args.arguments
        }

        parsed_ast = preprocessing.PreprocessedAST('<nested function>', getattr(self.function_ast, 'lineno', 0), '',
                                                   astutils.copy_tree(self.function_ast),
                                                   copy.copy(self.program_globals))
        return build_schedule_tree(self.name,
                                   parsed_ast,
                                   argtypes,
                                   constants={
                                       key: (_clone_descriptor(desc), value)
                                       for key, (desc, value) in self.constants.items()
                                   },
                                   callback_mapping=dict(self.callback_mapping),
                                   arg_names=[name for name in self.argnames if name in argtypes],
                                   lambda_bindings=active_lambda_bindings,
                                   callable_bindings=active_callable_bindings,
                                   seed_bindings=seed_bindings,
                                   external_globals=self.external_globals)


def _should_fallback_to_pyobject_scalar(node: ast.AST, value: Any = UNRESOLVED) -> bool:
    if value is None or isinstance(value, (str, bytes, numbers.Number, bool, type(Ellipsis))):
        return False
    return isinstance(node, (ast.Await, ast.Attribute, ast.BinOp, ast.BoolOp, ast.Call, ast.Compare, ast.FormattedValue,
                             ast.IfExp, ast.JoinedStr, ast.Name, ast.NamedExpr, ast.UnaryOp, ast.Yield, ast.YieldFrom))


def _requires_fstring_callback(node: ast.AST) -> bool:
    return isinstance(node, (ast.JoinedStr, ast.FormattedValue))


def build_schedule_tree(name: str,
                        parsed_ast: preprocessing.PreprocessedAST,
                        argtypes: Dict[str, data.Data],
                        *,
                        constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
                        callback_mapping: Optional[Dict[str, str]] = None,
                        arg_names: Optional[Sequence[str]] = None,
                        lambda_bindings: Optional[Dict[str, ast.Lambda]] = None,
                        callable_bindings: Optional[Dict[str, Any]] = None,
                        seed_bindings: Optional[Dict[str, _Binding]] = None,
                        external_globals: Optional[Dict[str, Any]] = None,
                        inline_calls: bool = True) -> tn.ScheduleTreeRoot:
    """
    Build a schedule tree directly from a preprocessed Python AST.

    :param name: Program name.
    :param parsed_ast: Preprocessed program AST and metadata.
    :param argtypes: Mapping from visible argument names to DaCe descriptors.
    :param inline_calls: If True (default), resolve and inline nested
        ``@dace.program`` calls after building the tree.
    :return: A schedule tree rooted at a top-level scope.
    """
    desugared_ast = preprocessing.PreprocessedAST(
        parsed_ast.filename,
        parsed_ast.src_line,
        parsed_ast.src,
        desugar_schedule_tree_expansions(parsed_ast.preprocessed_ast,
                                         filename=parsed_ast.filename,
                                         global_vars=parsed_ast.program_globals,
                                         known_descriptors=argtypes,
                                         seed_bindings=seed_bindings,
                                         callable_bindings=callable_bindings),
        parsed_ast.program_globals,
        resolved_arg_annotations=copy.deepcopy(parsed_ast.resolved_arg_annotations))
    builder = PythonScheduleTreeBuilder(name,
                                        desugared_ast,
                                        argtypes,
                                        constants=constants,
                                        callback_mapping=callback_mapping,
                                        arg_names=arg_names,
                                        lambda_bindings=lambda_bindings,
                                        callable_bindings=callable_bindings,
                                        seed_bindings=seed_bindings,
                                        external_globals=external_globals)
    root = builder.build()
    promote_dynamic_scope_copies(root)
    if inline_calls:
        resolve_function_calls(root)
    return root


class PythonScheduleTreeBuilder(ast.NodeVisitor):
    """Builds schedule trees from preprocessed Python ASTs."""

    def __init__(self,
                 name: str,
                 parsed_ast: preprocessing.PreprocessedAST,
                 argtypes: Dict[str, data.Data],
                 *,
                 constants: Optional[Dict[str, Tuple[data.Data, Any]]] = None,
                 callback_mapping: Optional[Dict[str, str]] = None,
                 arg_names: Optional[Sequence[str]] = None,
                 lambda_bindings: Optional[Dict[str, ast.Lambda]] = None,
                 callable_bindings: Optional[Dict[str, Any]] = None,
                 seed_bindings: Optional[Dict[str, _Binding]] = None,
                 external_globals: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.filename = parsed_ast.filename
        self.parsed_ast = parsed_ast
        self.argtypes = {k: _clone_descriptor(v) for k, v in argtypes.items()}
        self.globals = copy.copy(parsed_ast.program_globals)
        self.external_globals = copy.copy(parsed_ast.program_globals if external_globals is None else external_globals)
        self.root = tn.ScheduleTreeRoot(name=name,
                                        children=[],
                                        containers={},
                                        symbols={},
                                        constants=self._clone_constants(constants),
                                        callback_mapping=dict(callback_mapping or {}),
                                        arg_names=list(arg_names or []))
        self.scope_stack: List[tn.ScheduleTreeScope] = [self.root]
        self.bindings: Dict[str, _Binding] = {}
        self.annotated_descriptors: Dict[str, data.Data] = {}
        self.annotated_class_types: Dict[str, type[Any]] = {}
        self.explicit_structure_argument_names: set[str] = set()
        self.lambda_bindings: Dict[str, ast.Lambda] = {
            key: astutils.copy_tree(value)
            for key, value in (lambda_bindings or {}).items()
        }
        self.callable_bindings: Dict[str, Any] = dict(callable_bindings or {})
        self.seed_bindings = {key: _clone_binding(binding) for key, binding in (seed_bindings or {}).items()}
        self._declared_global_names: set[str] = set()
        self._declared_nonlocal_names: set[str] = set()
        self._callback_mutated_global_names: set[str] = set()
        self._raise_behavior = _normalize_raise_behavior(Config.get('frontend', 'raise_statements'))
        self._emit_external_reassign_nodes = isinstance(parsed_ast.preprocessed_ast, ast.Module)
        self._global_lambda_cache: Dict[str, Optional[ast.Lambda]] = {}
        self.expression_support = GenericExpressionSupportLibrary()
        self.array_literal_support = ArrayLiteralSupportLibrary()
        self.dict_support = DictSupportLibrary()
        self.numpy_support = NumpySupportLibrary()
        self.attribute_rewriter = AttributeRewriter(self._evaluation_context)
        self.lambda_resolver = LambdaResolver(self.globals,
                                              self.lambda_bindings,
                                              self.callable_bindings,
                                              cache=self._global_lambda_cache)
        self.callable_resolver = CallableResolver(callable_bindings=self.callable_bindings,
                                                  evaluation_context=self._evaluation_context)
        self.callable_specializer = CallableArgumentSpecializer(
            lambda_resolver=self.lambda_resolver,
            callable_resolver=self.callable_resolver,
            bindings=self.bindings,
            infer_descriptor=self._infer_plannable_expression_descriptor,
            resolve_data_access=self._resolve_data_access,
            is_callback_descriptor=self._is_callback_descriptor,
            callback_specialization_value=self._callback_specialization_value)

        def _raise_callback_syntax_error(node: ast.AST, message: str) -> None:
            raise DaceSyntaxError(self, node, message)

        self.callback_handler = CallbackHandler(bindings=self.bindings,
                                                callback_mutated_global_names=self._callback_mutated_global_names,
                                                callable_resolver=self.callable_resolver,
                                                evaluation_context=self._evaluation_context,
                                                append_node=self._append_node,
                                                register_binding=self._register_binding,
                                                fresh_callback_name=self._fresh_callback_name,
                                                fresh_transient_name=self._fresh_transient_name,
                                                render_callback_code=self._callback_code_text,
                                                collect_scope_declarations=_collect_scope_declarations,
                                                raise_syntax_error=_raise_callback_syntax_error,
                                                binding_kind_for_descriptor=_binding_kind_for_descriptor,
                                                pyobject_scalar_descriptor=_pyobject_scalar_descriptor,
                                                is_pyobject_scalar_descriptor=_is_pyobject_scalar_descriptor,
                                                is_iterator_protocol_call=_is_iterator_protocol_call,
                                                is_iterator_next_call=_is_iterator_next_call)
        self._terminate_body_stack: List[bool] = []

        self._initialize_root_scope()
        self._initialize_seed_bindings()
        self._initialize_direct_class_annotations()
        self.inferred_bindings = ScheduleTreeTypeInference(self.globals, self.argtypes).infer(self._program_node())
        for name, binding in self.inferred_bindings.items():
            if binding.descriptor is not None:
                self.root.containers.setdefault(name, _clone_descriptor(binding.descriptor))

    def build(self) -> tn.ScheduleTreeRoot:
        """Build the schedule tree for the program AST."""
        program = self._program_node()
        for stmt in program.body:
            self.visit(stmt)
        return self.root

    def visit_Assign(self, node: ast.Assign) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        for target in node.targets:
            self._handle_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        descriptor = self._evaluate_descriptor(node.annotation)
        class_type = self._evaluate_annotation_class_type(node.annotation)
        if descriptor is not None and isinstance(node.target, ast.Name):
            self.annotated_descriptors[node.target.id] = descriptor
            if class_type is not None:
                self.annotated_class_types[node.target.id] = class_type
            if node.value is None:
                existing = self.bindings.get(node.target.id)
                if existing is not None and isinstance(existing.descriptor, data.Reference):
                    return
                if not isinstance(descriptor, data.Reference):
                    self._register_binding(node.target.id, descriptor, kind=_binding_kind_for_descriptor(descriptor))
                return
        if node.value is not None:
            self._handle_assignment(node.target, node.value, annotated_descriptor=descriptor)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        value = ast.BinOp(left=astutils.copy_tree(node.target), op=node.op, right=astutils.copy_tree(node.value))
        self._handle_assignment(node.target, value)

    def visit_Expr(self, node: ast.Expr) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.callback_handler.reject_mutated_global_uses(node.value)
        value = self.lambda_resolver.inline_known_lambda_calls(node.value)
        if self.callable_resolver.is_dace_program_call(value):
            self._materialize_call_args(value)
            self._emit_function_call(value)
            return
        if self.callable_resolver.is_sdfg_call(value):
            self._materialize_call_args(value)
            if self._emit_sdfg_call(value):
                return
        planned_value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                                value,
                                                                materialize_root=False)
        if self._handle_expression(planned_value):
            self._apply_method_self_descriptor_side_effect(planned_value)
            return
        if _requires_fstring_callback(planned_value):
            callback_expr = ast.copy_location(ast.Expr(value=astutils.copy_tree(planned_value)), planned_value)
            self.callback_handler.wrap_node(callback_expr, 'f-string')
            return
        self._append_node(tn.StatementNode(code=CodeBlock(self._format_runtime_expression(planned_value))))

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            values: List[str] = []
        else:
            self.callback_handler.reject_mutated_global_uses(node.value)
            return_value = self.lambda_resolver.inline_known_lambda_calls(node.value)
            if self.callable_resolver.is_dace_program_call(return_value):
                # Materialize array-valued arguments, emit the function call;
                # the inlining pass will propagate the callee's return value.
                self._materialize_call_args(return_value)
                tmp = self._fresh_transient_name('__stree_retval')
                self._emit_function_call(return_value, return_targets=[tmp])
                self._append_node(tn.ReturnNode(values=[tmp]))
                return
            if self.callable_resolver.is_sdfg_call(return_value):
                self._materialize_call_args(return_value)
                tmp = self._fresh_transient_name('__stree_retval')
                self._register_binding(tmp, _pyobject_scalar_descriptor(), kind='scalar')
                if self._emit_sdfg_call(return_value, return_targets=[tmp]):
                    self._append_node(tn.ReturnNode(values=[tmp]))
                    return
            if isinstance(return_value, ast.Tuple):
                planned_values = [
                    self.expression_support.plan_expression(self._expression_planning_context(),
                                                            value,
                                                            materialize_root=True) for value in return_value.elts
                ]
                values = [self._materialize_return_value(v) for v in planned_values]
            else:
                planned_value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                                        return_value,
                                                                        materialize_root=True)
                values = [self._materialize_return_value(planned_value)]
        self._append_node(tn.ReturnNode(values=values))

    def _materialize_return_value(self, value: ast.AST) -> str:
        """Return the descriptor name backing a return-value expression.

        Non-descriptor expressions are materialized into fresh temporaries
        before returning so :class:`ReturnNode` only refers to descriptor names.
        """
        if _requires_fstring_callback(value):
            materialized = self.callback_handler.materialize_expression(value,
                                                                        'f-string',
                                                                        _string_scalar_descriptor(),
                                                                        prefix='__stree_retval')
            return materialized.id

        if isinstance(value, ast.Name) and self._resolve_data_access(value) is not None:
            return value.id

        descriptor = self._infer_plannable_expression_descriptor(value)
        if descriptor is None:
            descriptor = self._infer_scalar_descriptor(value, None)
        if descriptor is not None:
            name = self._fresh_transient_name('__stree_retval')
            kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
            self._register_binding(name, descriptor, kind=kind)
            target = ast.Name(id=name, ctx=ast.Store())
            if self._emit_computed_assignment(target, value, descriptor):
                return name

            output = self._resolve_output_target(target, value, descriptor)
            if output is not None:
                _, out_memlet, _ = output
                tasklet = tn.FrontendTasklet(name=self._tasklet_name(target),
                                             code=CodeBlock(f'{_unparse(target)} = {_unparse(value)}'))
                self._append_node(
                    tn.TaskletNode(node=tasklet,
                                   in_memlets=self._collect_input_memlets(value),
                                   out_memlets={'out': out_memlet}))
                return name

        materialized = self.callback_handler.materialize_expression(value,
                                                                    'return expression',
                                                                    _pyobject_scalar_descriptor(),
                                                                    prefix='__stree_retval')
        return materialized.id

    def visit_Pass(self, node: ast.Pass) -> None:
        del node

    def visit_Break(self, node: ast.Break) -> None:
        del node
        self._append_node(tn.BreakNode())

    def visit_Continue(self, node: ast.Continue) -> None:
        del node
        self._append_node(tn.ContinueNode())

    def visit_If(self, node: ast.If) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        self.callback_handler.reject_mutated_global_uses(node.test)
        self._emit_if_chain(node)

    def visit_For(self, node: ast.For) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        self.callback_handler.reject_mutated_global_uses(node.iter)
        loop_indices = self._parse_for_indices(node.target)
        iterator_kind, iterator_ranges = self._parse_for_iterator(node.iter)

        if iterator_kind == 'dace.map':
            map_scope = tn.MapScope(node=tn.FrontendMap(params=loop_indices, ranges=iterator_ranges), children=[])
            for index_name in loop_indices:
                map_scope.symbols[index_name] = symbolic.symbol(index_name, dtypes.int64)
            self._append_node(map_scope)
            self._visit_body(map_scope, node.body)
        elif iterator_kind == 'range':
            index_name = loop_indices[0]
            start, stop, step = iterator_ranges[0]
            comparator = '>' if stop.startswith('-') or step.startswith('-') else '<'
            loop_scope = tn.LoopScope(loop=tn.FrontendLoop(
                loop_condition=CodeBlock(f'{index_name} {comparator} {stop}'),
                init_statement=CodeBlock(f'{index_name} = {start}'),
                update_statement=CodeBlock(f'{index_name} = {index_name} + {step}'),
                loop_variable=index_name),
                                      children=[])
            loop_scope.symbols[index_name] = symbolic.symbol(index_name, dtypes.int64)
            self._append_node(loop_scope)
            self._visit_body(loop_scope, node.body)
        else:
            self._append_node(tn.StatementNode(code=CodeBlock(_unparse(node))))

        if node.orelse:
            else_scope = tn.ElseScope(children=[])
            self._append_node(else_scope)
            self._visit_body(else_scope, node.orelse)

    def visit_While(self, node: ast.While) -> None:
        reason = callback_reason(node)
        if reason is not None:
            self.callback_handler.wrap_node(node, reason)
            return
        self.callback_handler.reject_mutated_global_uses(node.test)
        loop_scope = tn.LoopScope(
            loop=tn.FrontendLoop(loop_condition=CodeBlock(self._format_runtime_expression(node.test))), children=[])
        self._append_node(loop_scope)
        self._visit_body(loop_scope, node.body)
        if node.orelse:
            else_scope = tn.ElseScope(children=[])
            self._append_node(else_scope)
            self._visit_body(else_scope, node.orelse)

    def generic_visit(self, node: ast.AST) -> None:
        import warnings
        warnings.warn(
            f'Schedule tree frontend: unhandled AST node {type(node).__name__} '
            f'at line {getattr(node, "lineno", "?")} — wrapping as callback',
            stacklevel=2)
        self.callback_handler.wrap_node(node, f'unhandled {type(node).__name__}')

    # ------------------------------------------------------------------ #
    #  PythonCallbackNode helpers                                         #
    # ------------------------------------------------------------------ #

    def _callback_code_text(self, node: ast.AST) -> str:
        """Return parseable source code for a callback-wrapped AST node."""
        if isinstance(node, ast.stmt):
            try:
                return ast.unparse(_sanitize_ast_for_unparse(astutils.copy_tree(node)))
            except Exception:
                try:
                    return astutils.unparse(_sanitize_ast_for_unparse(astutils.copy_tree(node)))
                except Exception:
                    return 'pass'

        try:
            return self._format_runtime_expression(node)
        except Exception:
            try:
                return _unparse(node)
            except Exception:
                try:
                    return ast.unparse(node)
                except Exception:
                    return 'None'

    # ------------------------------------------------------------------ #
    #  Category C visitors — always callback                              #
    # ------------------------------------------------------------------ #

    def visit_Try(self, node: ast.Try) -> None:
        self.callback_handler.wrap_node(node, 'try/except')

    # Python 3.11+ except* (TryStar)
    if hasattr(ast, 'TryStar'):
        visit_TryStar = visit_Try

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise DaceSyntaxError(
            self, node,
            'Nested class definitions are unsupported in @dace.program schedule-tree lowering because they cannot '
            'be outlined safely from compiled code')

    def visit_Import(self, node: ast.Import) -> None:
        self.callback_handler.wrap_node(node, 'import')

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.callback_handler.wrap_node(node, 'import')

    def visit_Yield(self, node: ast.AST) -> None:
        self.callback_handler.wrap_node(node, 'yield')

    def visit_YieldFrom(self, node: ast.AST) -> None:
        self.callback_handler.wrap_node(node, 'yield from')

    def visit_Await(self, node: ast.AST) -> None:
        self.callback_handler.wrap_node(node, 'await')

    def visit_Match(self, node: ast.AST) -> None:
        subject = self._match_subject_expression(node.subject)
        try:
            lowered = lower_match_to_statements(node, subject)
        except UnsupportedMatchPatternError:
            self.callback_handler.wrap_node(node, 'match/case')
            return

        for stmt in lowered:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        self.callback_handler.wrap_node(node, 'context manager')

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.callback_handler.wrap_node(node, 'context manager')

    # ------------------------------------------------------------------ #
    #  Category B visitors — try to lower, fall back to callback          #
    # ------------------------------------------------------------------ #

    def visit_Global(self, node: ast.Global) -> None:
        self._declared_global_names.update(node.names)
        for name in node.names:
            value = self._resolve_external_scope_value(name)
            if value is UNRESOLVED:
                self.callback_handler.wrap_node(node, 'global scope')
                return
            self._bind_external_scope_value(name, value)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._declared_nonlocal_names.update(node.names)
        for name in node.names:
            if name in self.bindings:
                continue

            value = self._resolve_external_scope_value(name)
            if value is UNRESOLVED:
                raise DaceSyntaxError(self, node, f'Could not resolve nonlocal name "{name}" in schedule-tree lowering')

            self._bind_external_scope_value(name, value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle nested function definitions.

        Known nested function definitions are lowered as inline call regions.
        When the target cannot be modeled safely, keep explicit callback
        fallback.
        """
        global_names, nonlocal_names = _collect_scope_declarations(node)
        inline_function = self._make_nested_function_program(node)
        if inline_function is not None:
            self.callable_bindings[node.name] = inline_function
            self.lambda_bindings.pop(node.name, None)
            self._register_binding(node.name, data.Scalar(dtypes.callback(None), transient=True), kind='callback')
            return
        conflicting_globals = self._enclosing_load_uses_outside(node, global_names)
        if conflicting_globals:
            conflicts = ', '.join(sorted(conflicting_globals))
            raise DaceSyntaxError(
                self, node,
                f'Nested callback functions cannot reassign global names that are used in the enclosing program: '
                f'{conflicts}')
        if nonlocal_names:
            raise DaceSyntaxError(
                self, node, 'Nested functions that use nonlocal declarations cannot fall back to callbacks during '
                'schedule-tree lowering')
        self.callback_handler.wrap_node(node, 'nested function')

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        global_names, nonlocal_names = _collect_scope_declarations(node)
        conflicting_globals = self._enclosing_load_uses_outside(node, global_names)
        if conflicting_globals:
            conflicts = ', '.join(sorted(conflicting_globals))
            raise DaceSyntaxError(
                self, node,
                f'Nested callback functions cannot reassign global names that are used in the enclosing program: '
                f'{conflicts}')
        if nonlocal_names:
            raise DaceSyntaxError(
                self, node, 'Nested functions that use nonlocal declarations cannot fall back to callbacks during '
                'schedule-tree lowering')
        self.callback_handler.wrap_node(node, 'async function')

    def visit_Delete(self, node: ast.Delete) -> None:
        # del of DaCe arrays is a no-op (runtime manages memory)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.bindings:
                continue  # No-op for known containers
            else:
                self.callback_handler.wrap_node(node, 'delete')
                return

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.cause is not None:
            raise DaceSyntaxError(
                self, node,
                'raise from is unsupported in @dace.program schedule-tree lowering because exceptional control flow '
                'cannot be represented safely')

        if self._raise_behavior == 'ignore_all':
            return

        raise_node = self._build_direct_raise_node(node)
        if raise_node is not None:
            self._append_node(raise_node)
            self._terminate_current_body()
            return

        if self._raise_behavior == 'ignore_dynamic':
            return

        self.callback_handler.wrap_node(node, 'raise')
        self._terminate_current_body()

    def _append_node(self, node: tn.ScheduleTreeNode) -> None:
        scope = self.scope_stack[-1]
        node.parent = scope
        scope.children.append(node)

    def _visit_body(self, scope: tn.ScheduleTreeScope, body: Sequence[ast.AST]) -> None:
        self.scope_stack.append(scope)
        self._terminate_body_stack.append(False)
        try:
            for stmt in body:
                self.visit(stmt)
                if self._terminate_body_stack[-1]:
                    break
        finally:
            self._terminate_body_stack.pop()
            self.scope_stack.pop()

    def _program_node(self) -> ast.FunctionDef:
        program_ast = self.parsed_ast.preprocessed_ast
        if isinstance(program_ast, ast.Module):
            node = program_ast.body[0]
        else:
            node = program_ast
        if not isinstance(node, ast.FunctionDef):
            raise TypeError('Expected a preprocessed FunctionDef as schedule-tree frontend input')
        return node

    def _enclosing_load_uses_outside(self, excluded_node: ast.AST, names: set[str]) -> set[str]:
        if not names or not self.parsed_ast.src:
            return set()

        source_ast = ast.parse(astutils._remove_outer_indentation(self.parsed_ast.src))
        ast.increment_lineno(source_ast, self.parsed_ast.src_line)

        source_program = source_ast.body[0] if source_ast.body else None
        if not isinstance(source_program, ast.FunctionDef):
            return set()

        excluded_end = (getattr(excluded_node, 'end_lineno', None)
                        or excluded_node.lineno, getattr(excluded_node, 'end_col_offset', None) or 0)

        class _LoadUseFinder(ast.NodeVisitor):

            def __init__(self, candidates: set[str], excluded_end_location: Tuple[int, int]) -> None:
                self.candidates = candidates
                self.excluded_end_location = excluded_end_location
                self.used: set[str] = set()

            def visit_Name(self, name_node: ast.Name) -> None:
                location = (getattr(name_node, 'lineno', 0), getattr(name_node, 'col_offset', 0))
                if (isinstance(name_node.ctx, ast.Load) and name_node.id in self.candidates
                        and location > self.excluded_end_location):
                    self.used.add(name_node.id)

        finder = _LoadUseFinder(names, excluded_end)
        finder.visit(source_program)
        return finder.used

    def _terminate_current_body(self) -> None:
        if self._terminate_body_stack:
            self._terminate_body_stack[-1] = True

    def _is_direct_exception_type(self, node: ast.AST) -> bool:
        value = try_resolve_static_value(node, self._evaluation_context())
        if value is UNRESOLVED:
            builtins_env = {
                **pybuiltins.__dict__,
                'builtins': pybuiltins,
                '__builtins__': pybuiltins.__dict__,
            }
            value = try_resolve_static_value(node, builtins_env)
        if value is UNRESOLVED:
            try:
                value = astutils.evalnode(node, builtins_env)
            except Exception:
                return False
        if not inspect.isclass(value):
            return False
        try:
            return issubclass(value, BaseException)
        except TypeError:
            return False

    def _build_direct_raise_node(self, node: ast.Raise) -> Optional[tn.RaiseNode]:
        if node.exc is None:
            return None

        exc_type = node.exc
        args: List[ast.AST] = []
        kwargs: Dict[str, ast.AST] = {}

        if isinstance(node.exc, ast.Call):
            if any(keyword.arg is None for keyword in node.exc.keywords):
                return None
            exc_type = node.exc.func
            args = list(node.exc.args)
            kwargs = {keyword.arg: keyword.value for keyword in node.exc.keywords if keyword.arg is not None}

        if not self._is_direct_exception_type(exc_type):
            return None

        return tn.RaiseNode(exception_type=CodeBlock(self._format_runtime_expression(exc_type)),
                            args=[CodeBlock(self._format_runtime_expression(argument)) for argument in args],
                            kwargs={
                                name: CodeBlock(self._format_runtime_expression(value))
                                for name, value in kwargs.items()
                            })

    def _match_subject_expression(self, subject: ast.AST) -> ast.AST:
        planned = self.expression_support.plan_expression(self._expression_planning_context(),
                                                          subject,
                                                          materialize_root=False)
        if isinstance(planned, (ast.Name, ast.Constant)):
            return planned

        descriptor = self._infer_compute_descriptor(planned)
        if descriptor is None:
            descriptor = self._infer_scalar_descriptor(planned, None)
        if descriptor is None:
            descriptor = _pyobject_scalar_descriptor()
        return self._materialize_temporary_expression(planned, descriptor)

    def _initialize_root_scope(self) -> None:
        for name, descriptor in self.argtypes.items():
            self.root.containers[name] = _clone_descriptor(descriptor)
            kind = 'callback' if self._is_callback_descriptor(descriptor) else 'container'
            self.bindings[name] = _Binding(descriptor=_clone_descriptor(descriptor), kind=kind)
            self.globals[name] = descriptor
            for free_symbol in descriptor.free_symbols:
                self.root.symbols[free_symbol.name] = free_symbol

        for name, value in self.globals.items():
            if isinstance(value, symbolic.symbol):
                self.root.symbols[name] = value

    def _initialize_seed_bindings(self) -> None:
        for name, binding in self.seed_bindings.items():
            self.bindings[name] = _clone_binding(binding)
            if binding.descriptor is not None:
                self.root.containers[name] = _clone_descriptor(binding.descriptor)
                self.globals.setdefault(name, _clone_descriptor(binding.descriptor))

    def _external_scope_kind(self, name: str) -> Optional[str]:
        if name in self._declared_nonlocal_names:
            return 'nonlocal'
        if name in self._declared_global_names:
            return 'global'
        return None

    def _should_emit_external_reassign(self, name: str) -> bool:
        return self._emit_external_reassign_nodes and self._external_scope_kind(name) is not None

    def _handle_assignment(self,
                           target: ast.AST,
                           value: ast.AST,
                           annotated_descriptor: Optional[data.Data] = None) -> None:
        self.callback_handler.reject_mutated_global_uses(value)
        if isinstance(target, ast.Name):
            self._update_callable_binding(target.id, value)
            self.lambda_resolver.update_binding(target.id, value)

        value = self.lambda_resolver.inline_known_lambda_calls(value)

        # Intercept nested @dace.program calls — materialize array-valued
        # arguments into temporaries first, then emit FunctionCallScope.
        if self.callable_resolver.is_dace_program_call(value):
            self._materialize_call_args(value)
            targets = [target.id] if isinstance(target, ast.Name) else [_unparse(target)]
            self._emit_function_call(value, return_targets=targets)
            return

        if self.callable_resolver.is_sdfg_call(value) and not isinstance(target, (ast.Tuple, ast.List)):
            self._materialize_call_args(value)
            if self._emit_sdfg_call_assignment(target, value, annotated_descriptor):
                return

        if isinstance(target, (ast.Tuple, ast.List)):
            self._seed_inferred_target_bindings(target)
            self._append_node(tn.StatementNode(code=CodeBlock(self._format_assignment_statement(target, value))))
            return

        value = self.expression_support.plan_expression(self._expression_planning_context(),
                                                        value,
                                                        materialize_root=False)
        if not self._ensure_pythonclass_for_direct_class_annotation(target):
            self._raise_or_warn_if_member_assignment_requires_python_class(target)
        self._update_dict_subscript_binding(target, value)

        source_access = self._resolve_data_access(value)
        self._ensure_pythonclass_member_target(target, value, source_access)
        if isinstance(target, ast.Name):
            self._handle_name_assignment(target.id, value, source_access, annotated_descriptor)
            return

        target_access = self._resolve_data_access(target)
        if source_access is not None and target_access is not None:
            _, source_memlet, source_desc, _ = source_access
            target_name, target_memlet, target_desc, _ = target_access
            memlet = copy.deepcopy(source_memlet)
            memlet.other_subset = copy.deepcopy(target_memlet.subset)
            if isinstance(target_desc, data.Reference):
                self._append_node(
                    tn.RefSetNode(target=target_name,
                                  memlet=memlet,
                                  src_desc=_clone_descriptor(source_desc),
                                  ref_desc=_clone_descriptor(target_desc)))
                return
            self._append_node(tn.CopyNode(target=target_name, memlet=memlet))
            return

        if self._emit_computed_assignment(target, value, annotated_descriptor):
            return

        self._append_node(tn.StatementNode(code=CodeBlock(self._format_assignment_statement(target, value))))

    def _raise_or_warn_if_member_assignment_requires_python_class(self, target: ast.AST) -> None:
        if not isinstance(target, ast.Attribute):
            return

        owner_access = self._resolve_data_access(target.value)
        if owner_access is None:
            return

        _, _, owner_descriptor, _ = owner_access
        message = python_class_requirement_for_member_assignment(owner_descriptor, target.attr)
        if message is None:
            return

        root_name = self._attribute_root_name(target.value)
        if root_name in self.explicit_structure_argument_names:
            raise DaceSyntaxError(self, target, message)

        warnings.warn(message, UserWarning, stacklevel=2)

    def _ensure_pythonclass_for_direct_class_annotation(self, target: ast.AST) -> bool:
        if not isinstance(target, ast.Attribute):
            return False

        owner_access = self._resolve_data_access(target.value)
        if owner_access is None:
            return False

        _, _, owner_descriptor, _ = owner_access
        if python_class_requirement_for_member_assignment(owner_descriptor, target.attr) is None:
            return False

        root = self._attribute_root_and_members(target.value)
        if root is None:
            return False
        root_name, member_names = root
        class_type = self.annotated_class_types.get(root_name)
        if class_type is None:
            return False
        if nested_direct_class_owner(class_type, member_names) is None:
            return False

        binding = self.bindings.get(root_name)
        if binding is None or binding.descriptor is None or isinstance(binding.descriptor, PythonClass):
            return False

        try:
            python_class_descriptor = PythonClass.from_class(class_type)
        except (TypeError, ValueError):
            return False

        self._store_binding(root_name, python_class_descriptor, kind=binding.kind)
        return True

    def _attribute_root_name(self, node: ast.AST) -> Optional[str]:
        root = self._attribute_root_and_members(node)
        return root[0] if root is not None else None

    def _attribute_root_and_members(self, node: ast.AST) -> Optional[Tuple[str, List[str]]]:
        current = node
        members: List[str] = []
        while isinstance(current, ast.Attribute):
            members.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            members.reverse()
            return current.id, members
        return None

    def _update_dict_subscript_binding(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            return
        binding = self.bindings.get(target.value.id)
        if binding is None or binding.descriptor is None:
            return
        dict_binding = binding.structure if isinstance(binding.structure, StaticDictBinding) else None
        updated = self.dict_support.infer_assignment_binding(self._dict_support_context(target.value.id),
                                                             binding.descriptor, dict_binding, target.slice, value)
        if updated is None:
            return
        updated_descriptor, updated_binding = updated
        self._store_binding(target.value.id,
                            updated_descriptor,
                            kind=binding.kind,
                            structure=updated_binding if updated_binding is not None else None)

    def _handle_name_assignment(self, name: str, value: ast.AST, source_access: Optional[Tuple[str, Memlet, data.Data,
                                                                                               Optional[data.Data]]],
                                annotated_descriptor: Optional[data.Data]) -> None:
        if self._is_internal_iterator_binding_name(name) or self._is_internal_iterator_helper_call(value):
            self._infer_internal_iterator_binding(name, value, annotated_descriptor)
            self._append_node(tn.AssignNode(name=name, value=CodeBlock(self._format_runtime_expression(value))))
            return

        if self._should_emit_external_reassign(name):
            self._handle_external_name_reassignment(name, value, source_access, annotated_descriptor)
            return

        if _requires_fstring_callback(value):
            self.callback_handler.emit_assignment(name, value, 'f-string', _string_scalar_descriptor())
            return

        existing = self.bindings.get(name)
        target_descriptor = annotated_descriptor or self.annotated_descriptors.get(name)

        if source_access is not None and isinstance(value, ast.Subscript) and _is_singleton_scalar_memlet(
                source_access[1]):
            if target_descriptor is None and existing is None:
                self._register_binding(name, data.Scalar(source_access[2].dtype, transient=True), kind='scalar')
                existing = self.bindings.get(name)
            source_access = None

        if source_access is not None:
            source_name, memlet, source_desc, view_desc = source_access

            if target_descriptor is not None and isinstance(target_descriptor, data.Reference):
                ref_desc = self._ensure_reference_binding(name, target_descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is not None and isinstance(existing.descriptor, data.Reference):
                ref_desc = self._ensure_reference_binding(name, existing.descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is None and self._should_bind_as_reference(value, source_desc):
                ref_desc = self._ensure_reference_binding(name, source_desc)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

            if existing is None and self._is_aliasable_descriptor(source_desc):
                new_view_desc = view_desc or self._make_view_descriptor(source_desc)
                self._register_binding(name, new_view_desc, kind='view')
                self._append_node(
                    tn.ViewNode(target=name,
                                source=source_name,
                                memlet=memlet,
                                src_desc=source_desc,
                                view_desc=new_view_desc))
                return

            if existing is not None and existing.descriptor is not None and self._can_promote_to_reference(
                    existing.descriptor, source_desc):
                ref_desc = self._ensure_reference_binding(name, existing.descriptor)
                self._append_node(tn.RefSetNode(target=name, memlet=memlet, src_desc=source_desc, ref_desc=ref_desc))
                return

        inferred_descriptor = self._infer_descriptor(value, name)
        if inferred_descriptor is not None:
            if existing is None and self._should_bind_expression_as_reference(value, inferred_descriptor):
                ref_desc = self._ensure_reference_binding(name, inferred_descriptor)
                self._append_node(
                    tn.RefSetNode(target=name,
                                  memlet=None,
                                  src_desc=inferred_descriptor,
                                  ref_desc=ref_desc,
                                  source_expr=self._format_runtime_expression(value)))
                return
            kind = 'reference' if isinstance(inferred_descriptor, data.Reference) else 'container'
            if self._is_callback_descriptor(inferred_descriptor):
                kind = 'callback'
            structure = self._runtime_container_structure(name, value, inferred_descriptor)
            self._store_binding(name, inferred_descriptor, kind=kind, structure=structure)
        else:
            scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
            if scalar_descriptor is not None:
                kind = 'callback' if self._is_callback_descriptor(scalar_descriptor) else 'scalar'
                self._register_binding(name, scalar_descriptor, kind=kind)

        scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
        if (isinstance(value, ast.Call) and _is_pyobject_scalar_descriptor(scalar_descriptor)
                and self.callback_handler.should_emit_pyobject_call_callback(value)):
            self.callback_handler.emit_assignment(name, value, 'pyobject call', scalar_descriptor)
            return

        if self.callable_specializer.is_callback_expression(value):
            self._append_node(tn.AssignNode(name=name, value=CodeBlock(self._format_runtime_expression(value))))
            return

        if self._should_emit_runtime_container_assignment(name, value):
            self._append_node(tn.StatementNode(code=CodeBlock(f'{name} = {self._format_runtime_expression(value)}')))
            return

        if self._emit_computed_assignment(ast.Name(id=name, ctx=ast.Store()), value, annotated_descriptor):
            return

        self._append_node(tn.AssignNode(name=name, value=CodeBlock(self._format_runtime_expression(value))))

    def _runtime_container_structure(self, name: str, value: ast.AST, inferred_descriptor: data.Data) -> Optional[Any]:
        if isinstance(inferred_descriptor, PythonDict) and isinstance(value, ast.Dict):
            return self.dict_support.infer_literal_binding(self._dict_support_context(name), value)
        return None

    def _should_emit_runtime_container_assignment(self, name: str, value: ast.AST) -> bool:
        binding_descriptor = self.bindings.get(name).descriptor if name in self.bindings else None
        return self._is_runtime_container_descriptor(binding_descriptor) or self._is_runtime_container_literal(value)

    def _is_runtime_container_descriptor(self, descriptor: Optional[data.Data]) -> bool:
        return isinstance(descriptor, (PythonDict, PythonList, PythonTuple))

    def _is_runtime_container_literal(self, value: ast.AST) -> bool:
        return isinstance(value, (ast.Dict, ast.List, ast.Tuple))

    def _handle_external_name_reassignment(self, name: str, value: ast.AST,
                                           source_access: Optional[Tuple[str, Memlet, data.Data, Optional[data.Data]]],
                                           annotated_descriptor: Optional[data.Data]) -> None:
        existing = self.bindings.get(name)
        target_descriptor = annotated_descriptor or self.annotated_descriptors.get(name)

        if source_access is not None and isinstance(value, ast.Subscript) and _is_singleton_scalar_memlet(
                source_access[1]):
            if target_descriptor is None and existing is None:
                self._register_binding(name, data.Scalar(source_access[2].dtype, transient=True), kind='scalar')
                existing = self.bindings.get(name)
            source_access = None

        if source_access is not None:
            _, _, source_desc, view_desc = source_access

            if target_descriptor is not None and isinstance(target_descriptor, data.Reference):
                self._ensure_reference_binding(name, target_descriptor)
            elif existing is not None and isinstance(existing.descriptor, data.Reference):
                self._ensure_reference_binding(name, existing.descriptor)
            elif existing is None and self._should_bind_as_reference(value, source_desc):
                self._ensure_reference_binding(name, source_desc)
            elif existing is None and self._is_aliasable_descriptor(source_desc):
                self._register_binding(name, view_desc or self._make_view_descriptor(source_desc), kind='view')
            elif existing is not None and existing.descriptor is not None and self._can_promote_to_reference(
                    existing.descriptor, source_desc):
                self._ensure_reference_binding(name, existing.descriptor)

        inferred_descriptor = self._infer_descriptor(value, name)
        if inferred_descriptor is not None:
            self._register_binding(name, inferred_descriptor, kind=_binding_kind_for_descriptor(inferred_descriptor))
        else:
            scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
            if scalar_descriptor is not None:
                self._register_binding(name, scalar_descriptor, kind=_binding_kind_for_descriptor(scalar_descriptor))

        scope_kind = self._external_scope_kind(name)
        if scope_kind is None:
            raise DaceSyntaxError(self, self._program_node(), f'Could not determine external scope kind for "{name}"')
        self._append_node(
            tn.ReassignExternalNode(name=name,
                                    value=CodeBlock(self._format_runtime_expression(value)),
                                    scope=scope_kind))

    def _handle_expression(self, value: ast.AST) -> bool:
        if not isinstance(value, ast.Call) or not self._should_lower_as_library_call(value):
            return False

        in_memlets = self._collect_input_memlets(value)
        if not in_memlets:
            return False

        self._append_node(
            tn.LibraryCall(node=tn.FrontendLibrary(name=astutils.rname(value.func),
                                                   properties=self._library_properties(value)),
                           in_memlets=in_memlets,
                           out_memlets=set()))
        return True

    def _emit_sdfg_call_assignment(self, target: ast.AST, value: ast.Call,
                                   annotated_descriptor: Optional[data.Data]) -> bool:
        if isinstance(target, ast.Name):
            descriptor = annotated_descriptor
            if descriptor is None:
                binding = self.bindings.get(target.id)
                if binding is not None:
                    descriptor = binding.descriptor
            if descriptor is None:
                descriptor = _pyobject_scalar_descriptor()
            self._register_binding(target.id, descriptor, kind=_binding_kind_for_descriptor(descriptor))
            return self._emit_sdfg_call(value, return_targets=[target.id])

        return self._emit_sdfg_call(value, return_targets=[_unparse(target)])

    def _emit_computed_assignment(self, target: ast.AST, value: ast.AST,
                                  annotated_descriptor: Optional[data.Data]) -> bool:
        output = self._resolve_output_target(target, value, annotated_descriptor)
        out_memlet = output[1] if output is not None else None

        lowered = self.array_literal_support.lower_assignment(self._array_literal_context(), target, value,
                                                              annotated_descriptor)
        if lowered is not None:
            self._append_node(lowered)
            return True

        lowered = self.numpy_support.lower_assignment(self._numpy_lowering_context(), target, value,
                                                      annotated_descriptor)
        if lowered is not None:
            self._append_node(lowered)
            return True

        if output is not None and isinstance(value, ast.Call):
            library_info = self._library_info_for_call(value)
            if library_info is not None:
                library_name, library_properties = library_info
                in_memlets = self._collect_input_memlets(value)
                if not in_memlets:
                    return False
                self._append_node(
                    tn.LibraryCall(node=tn.FrontendLibrary(name=library_name, properties=library_properties),
                                   in_memlets=in_memlets,
                                   out_memlets={'out': out_memlet}))
                return True

        if output is not None and isinstance(value, ast.Attribute):
            library_info = self._library_info_for_attribute(value)
            if library_info is not None:
                library_name, library_properties = library_info
                in_memlets = self._collect_input_memlets(value)
                if not in_memlets:
                    return False
                self._append_node(
                    tn.LibraryCall(node=tn.FrontendLibrary(name=library_name, properties=library_properties),
                                   in_memlets=in_memlets,
                                   out_memlets={'out': out_memlet}))
                return True

        lowered = self.expression_support.lower_assignment(self._expression_planning_context(), target, value,
                                                           annotated_descriptor)
        if lowered is not None:
            self._append_node(lowered)
            return True

        if output is None:
            return False

        _, out_memlet, _ = output

        if isinstance(value, ast.Call) and self._should_lower_as_library_call(value):
            in_memlets = self._collect_input_memlets(value)
            if not in_memlets:
                return False
            self._append_node(
                tn.LibraryCall(node=tn.FrontendLibrary(name=astutils.rname(value.func),
                                                       properties=self._library_properties(value)),
                               in_memlets=in_memlets,
                               out_memlets={'out': out_memlet}))
            return True

        in_memlets = self._collect_input_memlets(value)
        if not in_memlets and not isinstance(target, ast.Attribute):
            return False

        tasklet_code = f'out = {_unparse(value)}' if isinstance(
            target, ast.Attribute) else f'{_unparse(target)} = {_unparse(value)}'
        tasklet = tn.FrontendTasklet(name=self._tasklet_name(target), code=CodeBlock(tasklet_code))
        self._append_node(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets={'out': out_memlet}))
        return True

    def _ensure_pythonclass_member_target(
            self, target: ast.AST, value: ast.AST, source_access: Optional[Tuple[str, Memlet, data.Data,
                                                                                 Optional[data.Data]]]) -> None:
        if not isinstance(target, ast.Attribute):
            return

        root = self._attribute_root_and_members(target)
        if root is None:
            return
        root_name, member_names = root

        binding = self.bindings.get(root_name)
        if binding is None or binding.descriptor is None or not isinstance(binding.descriptor, PythonClass):
            return

        member_descriptor = self._infer_pythonclass_member_descriptor(value, source_access)
        if member_descriptor is None:
            return

        for descriptor in (binding.descriptor, self.root.containers.get(root_name), self.globals.get(root_name),
                           self.scope_stack[-1].containers.get(root_name) if self.scope_stack else None):
            if isinstance(descriptor, data.Data):
                ensure_nested_member_descriptor(descriptor, member_names, member_descriptor)

    def _infer_pythonclass_member_descriptor(
            self, value: ast.AST, source_access: Optional[Tuple[str, Memlet, data.Data,
                                                                Optional[data.Data]]]) -> Optional[data.Data]:
        if source_access is not None:
            _, memlet, source_desc, view_desc = source_access
            if _is_singleton_scalar_memlet(memlet):
                return data.Scalar(source_desc.dtype)
            return data.Reference.view(view_desc or source_desc)

        scalar_descriptor = self._infer_scalar_descriptor(value, None)
        if scalar_descriptor is not None:
            return scalar_descriptor

        computed_descriptor = self._infer_compute_descriptor(value)
        if isinstance(computed_descriptor, data.Scalar):
            return computed_descriptor

        return None

    def _format_runtime_expression(self, node: ast.AST) -> str:
        return _unparse(self.attribute_rewriter.rewrite_expression(node))

    def _format_assignment_statement(self, target: ast.AST, value: ast.AST) -> str:
        rewritten_call = self.attribute_rewriter.rewrite_assignment(target, value)
        if rewritten_call is not None:
            return _unparse(rewritten_call)
        return f'{_unparse(target)} = {self._format_runtime_expression(value)}'

    def _resolve_data_access(self, node: ast.AST) -> Optional[Tuple[str, Memlet, data.Data, Optional[data.Data]]]:
        if isinstance(node, ast.Name) and node.id in self.bindings and self.bindings[node.id].descriptor is not None:
            descriptor = _clone_descriptor(self.bindings[node.id].descriptor)
            if isinstance(descriptor, PythonDict):
                return None
            return (node.id, Memlet.from_array(node.id, descriptor), descriptor, _clone_descriptor(descriptor))

        if isinstance(node, ast.Attribute):
            owner_access = self._resolve_data_access(node.value)
            if owner_access is None:
                return None
            owner_name, _, owner_descriptor, _ = owner_access
            member_access = resolve_member_access(owner_name, owner_descriptor, node.attr)
            if member_access is None or isinstance(member_access.descriptor, PythonDict):
                return None
            descriptor = member_access.descriptor
            data_name = member_access.data_name
            return (data_name, Memlet.from_array(data_name, descriptor), descriptor, _clone_descriptor(descriptor))

        if isinstance(node, ast.Subscript):
            base_access = self._resolve_data_access(node.value)
            if base_access is None:
                return None
            base_name, _, descriptor, _ = base_access
            descriptor = _clone_descriptor(descriptor)
            if isinstance(descriptor, PythonDict):
                return None
            try:
                subset, new_axes, arrdims = memlet_parser.parse_memlet_subset(descriptor, node,
                                                                              self._evaluation_context())
            except Exception:
                return None
            if arrdims:
                return None
            memlet = Memlet(data=base_name, subset=subset)
            return (base_name, memlet, descriptor, self._make_view_descriptor(descriptor, subset.size(), new_axes))

        return None

    def _resolve_output_target(self, target: ast.AST, value: ast.AST,
                               annotated_descriptor: Optional[data.Data]) -> Optional[Tuple[str, Memlet, data.Data]]:
        if isinstance(target, ast.Name):
            if annotated_descriptor is not None and isinstance(annotated_descriptor, data.Reference):
                return None

            existing = self.bindings.get(target.id)
            if existing is not None and existing.descriptor is not None and not isinstance(
                    existing.descriptor, data.Reference):
                descriptor = _clone_descriptor(existing.descriptor)
                return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

            if annotated_descriptor is not None:
                kind = 'scalar' if isinstance(annotated_descriptor, data.Scalar) else 'container'
                self._register_binding(target.id, annotated_descriptor, kind=kind)
                descriptor = _clone_descriptor(self.bindings[target.id].descriptor)
                return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

            inferred_descriptor = self._infer_compute_descriptor(value)
            if inferred_descriptor is None:
                return None
            kind = 'scalar' if isinstance(inferred_descriptor, data.Scalar) else 'container'
            self._register_binding(target.id, inferred_descriptor, kind=kind)
            descriptor = _clone_descriptor(self.bindings[target.id].descriptor)
            return (target.id, Memlet.from_array(target.id, descriptor), descriptor)

        target_access = self._resolve_data_access(target)
        if target_access is None:
            return None
        target_name, target_memlet, descriptor, _ = target_access
        return (target_name, target_memlet, descriptor)

    def _infer_compute_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        inferred_descriptor = self._infer_plannable_expression_descriptor(node)
        if inferred_descriptor is not None:
            return inferred_descriptor

        access = self._resolve_data_access(node)
        if access is not None:
            _, _, descriptor, view_descriptor = access
            result = _clone_descriptor(view_descriptor or descriptor)
            result.transient = True
            return result

        for _, _, descriptor, _ in self._collect_expression_accesses(node):
            result = _clone_descriptor(descriptor)
            result.transient = True
            return result
        return None

    def _collect_expression_accesses(self, node: ast.AST) -> List[Tuple[str, Memlet, data.Data, Optional[data.Data]]]:
        accesses: List[Tuple[str, Memlet, data.Data, Optional[data.Data]]] = []
        seen = set()

        def _visit(current: ast.AST) -> None:
            access = self._resolve_data_access(current)
            if access is not None:
                name, memlet, descriptor, view_descriptor = access
                key = (name, str(memlet.subset), str(memlet.other_subset) if memlet.other_subset is not None else '')
                if key not in seen:
                    seen.add(key)
                    accesses.append((name, memlet, descriptor, view_descriptor))
                return

            for child in ast.iter_child_nodes(current):
                _visit(child)

        _visit(node)
        return accesses

    def _collect_input_memlets(self, node: ast.AST) -> Dict[str, Memlet]:
        result: Dict[str, Memlet] = {}
        for index, (_, memlet, _, _) in enumerate(self._collect_expression_accesses(node)):
            result[f'in{index}'] = memlet
        return result

    def _infer_descriptor(self, node: ast.AST, target_name: str) -> Optional[data.Data]:
        if isinstance(node, ast.Dict):
            return self.dict_support.infer_literal_descriptor(self._dict_support_context(target_name), node)

        if isinstance(node, ast.Call):
            inferred = self.array_literal_support.infer_expression_descriptor(self._array_literal_context(), node)
            if inferred is not None:
                return inferred

        if isinstance(node, (ast.List, ast.Tuple)):
            structure, _ = self._structure_from_ast(node)
            if structure is not None:
                return descriptor_from_structure(structure)

        if isinstance(node, ast.Attribute):
            access = self._resolve_data_access(node)
            if access is not None:
                _, _, descriptor, view_descriptor = access
                result = _clone_descriptor(view_descriptor or descriptor)
                result.transient = True
                return result

        def _infer_operator_operand(operand: ast.AST) -> Optional[Any]:
            descriptor = self._infer_descriptor(operand, target_name)
            if descriptor is not None:
                return descriptor

            value = try_resolve_static_value(operand, self._evaluation_context())
            if value is not UNRESOLVED:
                return value

            return self._infer_scalar_descriptor(operand, None)

        if isinstance(node, ast.BinOp):

            left_operand = _infer_operator_operand(node.left)
            right_operand = _infer_operator_operand(node.right)
            if left_operand is not None and right_operand is not None:
                infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
                    type(node.op).__name__, left_operand, right_operand)
                if infer_fn is not None:
                    try:
                        inferred = infer_fn(left_operand, right_operand)
                    except Exception:
                        inferred = None
                    if inferred is not None:
                        return inferred

        if isinstance(node, ast.UnaryOp):

            operand = _infer_operator_operand(node.operand)
            if operand is not None:
                infer_fn = oprepo.Replacements.get_operator_descriptor_inference(type(node.op).__name__, operand)
                if infer_fn is not None:
                    try:
                        inferred = infer_fn(operand)
                    except Exception:
                        inferred = None
                    if inferred is not None:
                        return inferred

        if isinstance(node, ast.BoolOp) and len(node.values) > 0:

            infer_fn = oprepo.Replacements.get_operator_descriptor_inference(type(node.op).__name__)
            current = _infer_operator_operand(node.values[0])
            if current is not None:
                for value in node.values[1:]:
                    next_operand = _infer_operator_operand(value)
                    if current is None or next_operand is None:
                        current = None
                        break
                    infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
                        type(node.op).__name__, current, next_operand)
                    if infer_fn is None:
                        current = None
                        break
                    try:
                        current = infer_fn(current, next_operand)
                    except Exception:
                        current = None
                        break
                if current is not None:
                    return current

        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:

            left_operand = _infer_operator_operand(node.left)
            right_operand = _infer_operator_operand(node.comparators[0])
            if left_operand is not None and right_operand is not None:
                infer_fn = oprepo.Replacements.get_operator_descriptor_inference(
                    type(node.ops[0]).__name__, left_operand, right_operand)
                if infer_fn is not None:
                    try:
                        inferred = infer_fn(left_operand, right_operand)
                    except Exception:
                        inferred = None
                    if inferred is not None:
                        return inferred

        if isinstance(node, ast.Subscript):
            base_descriptor: Optional[data.Data] = None
            base_access = self._resolve_data_access(node.value)
            if base_access is not None:
                _, _, base_descriptor, _ = base_access
            elif isinstance(node.value, ast.Name):
                binding = self.bindings.get(node.value.id)
                if binding is not None and binding.descriptor is not None:
                    base_descriptor = binding.descriptor
            elif isinstance(node.value, ast.Attribute):
                base_descriptor = self._infer_descriptor(node.value, target_name)

            if base_descriptor is not None:
                dict_binding = None
                if isinstance(node.value, ast.Name):
                    binding = self.bindings.get(node.value.id)
                    if binding is not None and isinstance(binding.structure, StaticDictBinding):
                        dict_binding = binding.structure
                dict_descriptor = self.dict_support.infer_subscript_descriptor(self._dict_support_context(target_name),
                                                                               base_descriptor, node.slice,
                                                                               dict_binding)
                if dict_descriptor is not None:
                    return dict_descriptor
                inferred = _infer_static_subscript_descriptor(base_descriptor, node, self._evaluation_context())
                if inferred is not None:
                    return inferred

        if isinstance(node, ast.Lambda):
            return data.Scalar(dtypes.callback(None), transient=True)

        if isinstance(node, ast.Call):
            # Try the method descriptor-inference registry first (a.sum(), a.reshape(), etc.)
            if isinstance(node.func, ast.Attribute):
                inferred = self._try_method_descriptor_inference(node)
                if inferred is not None:
                    return inferred

            inferred = self._try_ufunc_descriptor_inference(node)
            if inferred is not None:
                return inferred

            # Try the free-function descriptor-inference registry (numpy.sum(), etc.)
            inferred = self._try_descriptor_inference(node)
            if inferred is not None:
                return inferred

        # Attribute inference (a.T, a.flat, a.real, a.imag, etc.)
        if isinstance(node, ast.Attribute):
            inferred = self._try_attribute_descriptor_inference(node)
            if inferred is not None:
                return inferred

        value = try_resolve_static_value(node, self._evaluation_context())
        if value is not UNRESOLVED:
            try:
                descriptor = _clone_descriptor(data.create_datadescriptor(value))
                descriptor.transient = True
                return descriptor
            except Exception:
                pass

        return None

    def _try_descriptor_inference(self, node: ast.Call) -> Optional[data.Data]:
        """Query the descriptor-inference registry for a call node."""

        call_name = self._resolved_callable_name(node.func)
        infer_fn = oprepo.Replacements.get_descriptor_inference(call_name)
        if infer_fn is None:
            textual_name = astutils.rname(node.func)
            if textual_name != call_name:
                infer_fn = oprepo.Replacements.get_descriptor_inference(textual_name)
        if infer_fn is None:
            return None
        input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            result = infer_fn(input_descs, *args, **kwargs)
        except Exception:
            return None
        binding = _binding_from_inference_result(result)
        return None if binding is None else binding.descriptor

    def _try_method_descriptor_inference(self, node: ast.Call) -> Optional[data.Data]:
        """Query the method descriptor-inference registry for ``obj.method(...)`` calls."""

        if not isinstance(node.func, ast.Attribute):
            return None
        # Resolve the object (e.g. ``a`` in ``a.sum()``)
        obj_access = self._resolve_data_access(node.func.value)
        if obj_access is None:
            return None
        _, _, obj_desc, _ = obj_access
        classname = type(obj_desc).__name__  # 'Array', 'View', 'Scalar'
        method_name = node.func.attr
        infer_fn = oprepo.Replacements.get_method_descriptor_inference(classname, method_name)
        if infer_fn is None:
            return None
        # Resolve the remaining arguments (skip 'self')
        _input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            result = infer_fn(obj_desc, *args, **kwargs)
        except Exception:
            return None
        binding = _binding_from_inference_result(result)
        return None if binding is None else binding.descriptor

    def _apply_method_self_descriptor_side_effect(self, node: ast.AST) -> None:

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return
        if not isinstance(node.func.value, ast.Name):
            return

        obj_name = node.func.value.id
        obj_access = self._resolve_data_access(node.func.value)
        if obj_access is None:
            return
        _, _, obj_desc, _ = obj_access

        infer_fn = oprepo.Replacements.get_method_self_descriptor_inference(type(obj_desc).__name__, node.func.attr)
        if infer_fn is None:
            return

        _input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            updated_self = infer_fn(obj_desc, *args, **kwargs)
        except Exception:
            return
        if not isinstance(updated_self, data.Data):
            return

        kind = self.bindings.get(obj_name).kind if obj_name in self.bindings else _binding_kind_for_descriptor(
            updated_self)
        self._store_binding(obj_name, updated_self, kind=kind)

    def _try_attribute_descriptor_inference(self, node: ast.Attribute) -> Optional[data.Data]:
        """Query the attribute descriptor-inference registry for ``obj.attr`` accesses."""

        obj_access = self._resolve_data_access(node.value)
        if obj_access is None:
            return None
        _, _, obj_desc, _ = obj_access
        classname = type(obj_desc).__name__
        infer_fn = oprepo.Replacements.get_attribute_descriptor_inference(classname, node.attr)
        if infer_fn is None:
            return None
        try:
            result = infer_fn(obj_desc)
        except Exception:
            return None
        binding = _binding_from_inference_result(result)
        return None if binding is None else binding.descriptor

    def _try_ufunc_descriptor_inference(self, node: ast.Call) -> Optional[data.Data]:
        """Query the descriptor-inference registry for a NumPy ufunc call or ufunc method."""

        target = _resolve_ufunc_inference_target(node, self._evaluation_context())
        if target is None:
            return None

        method_name, ufunc_name = target
        infer_fn = oprepo.Replacements.get_ufunc_descriptor_inference(method_name)
        if infer_fn is None:
            return None

        input_descs, args, kwargs = self._resolve_call_inputs(node)
        try:
            result = infer_fn(input_descs, ufunc_name, *args, **kwargs)
        except Exception:
            return None
        binding = _binding_from_inference_result(result)
        return None if binding is None else binding.descriptor

    def _resolve_call_inputs(self, call_node: ast.Call) -> tuple:
        """Resolve call arguments to ``(input_descriptors, args, kwargs)``."""
        input_descs: Dict[str, data.Data] = {}
        args: list = []
        for arg in call_node.args:
            access = self._resolve_data_access(arg)
            if access is not None:
                name, memlet, desc, view_desc = access
                is_scalar_memlet = _is_singleton_scalar_memlet(memlet)
                if is_scalar_memlet:
                    base_descriptor = view_desc or desc
                    resolved_desc = data.Scalar(base_descriptor.dtype, transient=True)
                    key = _unparse(arg)
                else:
                    resolved_desc = _clone_descriptor(view_desc or desc)
                    key = name
                input_descs[key] = resolved_desc
                args.append(key)
            else:
                val = try_resolve_static_value(arg, self._evaluation_context())
                args.append(val if val is not UNRESOLVED else _unparse(arg))
        kwargs: dict = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            val = try_resolve_static_value(kw.value, self._evaluation_context())
            kwargs[kw.arg] = val if val is not UNRESOLVED else _unparse(kw.value)
        return input_descs, args, kwargs

    def _infer_scalar_descriptor(self, node: ast.AST, annotated_descriptor: Optional[data.Data]) -> Optional[data.Data]:
        if annotated_descriptor is not None and isinstance(annotated_descriptor, data.Scalar):
            return _clone_descriptor(annotated_descriptor)

        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            return _string_scalar_descriptor()

        scalar_types = {
            name: binding.descriptor.dtype
            for name, binding in self.bindings.items()
            if binding.descriptor is not None and isinstance(binding.descriptor, data.Scalar)
        }
        try:
            inferred_type = infer_expr_type(_unparse(node), scalar_types)
        except Exception:
            inferred_type = None
        if inferred_type is not None:
            return data.Scalar(inferred_type, transient=True)

        value = try_resolve_static_value(node, self._evaluation_context())
        if value is not UNRESOLVED and value is not None:
            try:
                descriptor = _clone_descriptor(data.create_datadescriptor(value))
            except Exception:
                descriptor = None
            if isinstance(descriptor, data.Scalar):
                descriptor.transient = True
                return descriptor

        if isinstance(value, numbers.Number) or isinstance(value, bool):
            dtype = _normalize_dtype(type(value))
            if dtype is not None:
                return data.Scalar(dtype, transient=True)

        if _should_fallback_to_pyobject_scalar(node, value):
            return _pyobject_scalar_descriptor()

        if value is UNRESOLVED:
            return None

        return None

    def _evaluate_descriptor(self, node: Optional[ast.AST]) -> Optional[data.Data]:
        if node is None:
            return None
        class_type = self._evaluate_annotation_class_type(node)
        if class_type is not None:
            try:
                return data.Structure.from_class(class_type)
            except (TypeError, ValueError):
                return None
        try:
            value = astutils.evalnode(node, self._evaluation_context())
        except Exception:
            return None
        if isinstance(value, data.Data):
            return _clone_descriptor(value)
        dtype = _normalize_dtype(value)
        if dtype is not None:
            return data.Scalar(dtype, transient=True)
        return None

    def _evaluate_annotation_class_type(self, node: Optional[ast.AST]) -> Optional[type[Any]]:
        if node is None:
            return None
        try:
            value = astutils.evalnode(node, self._evaluation_context())
        except Exception:
            return None
        return direct_class_annotation_type(value)

    def _initialize_direct_class_annotations(self) -> None:
        program = self._program_node()
        arguments = list(program.args.posonlyargs) + list(program.args.args) + list(program.args.kwonlyargs)
        if program.args.vararg is not None:
            arguments.append(program.args.vararg)
        if program.args.kwarg is not None:
            arguments.append(program.args.kwarg)

        resolved_arg_annotations = self.parsed_ast.resolved_arg_annotations or {}

        for argument in arguments:
            resolved_annotation = resolved_arg_annotations.get(argument.arg, None)
            descriptor = None
            class_type = None
            if resolved_annotation is not None:
                class_type = direct_class_annotation_type(resolved_annotation)
                if isinstance(resolved_annotation, data.Data):
                    descriptor = _clone_descriptor(resolved_annotation)

            if descriptor is None:
                descriptor = self._evaluate_descriptor(argument.annotation)
            if class_type is None:
                class_type = self._evaluate_annotation_class_type(argument.annotation)

            if class_type is not None:
                self.annotated_class_types[argument.arg] = class_type
            elif isinstance(descriptor, data.Structure):
                self.explicit_structure_argument_names.add(argument.arg)

    def _parse_shape(self, node: ast.AST) -> List[Any]:
        value = try_resolve_static_value(node, self._evaluation_context())
        if value is UNRESOLVED:
            value = None

        if isinstance(value, (list, tuple)):
            return [self._shape_dim(dim) for dim in value]
        if value is not None:
            return [self._shape_dim(value)]

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._shape_dim(symbolic.pystr_to_symbolic(_unparse(elem))) for elem in node.elts]
        return [self._shape_dim(symbolic.pystr_to_symbolic(_unparse(node)))]

    def _parse_dtype(self, node: Optional[ast.AST]) -> Optional[dtypes.typeclass]:
        if node is None:
            return None
        value = try_resolve_static_value(node, self._evaluation_context())
        if value is UNRESOLVED:
            return None
        return _normalize_dtype(value)

    def _shape_dim(self, value: Any) -> Any:
        if isinstance(value, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)):
            return value
        if isinstance(value, str):
            return symbolic.pystr_to_symbolic(value)
        return value

    def _call_argument(self, node: ast.Call, position: int, keyword: str) -> Optional[ast.AST]:
        if len(node.args) > position:
            return node.args[position]
        for kw in node.keywords:
            if kw.arg == keyword:
                return kw.value
        return None

    def _library_properties(self, node: ast.Call) -> Dict[str, Any]:
        return {kw.arg: _unparse(kw.value) for kw in node.keywords if kw.arg is not None}

    def _tasklet_name(self, target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return f'{target.id}_tasklet'
        if isinstance(target, ast.Subscript):
            return f'{_unparse(target.value)}_tasklet'
        return 'tasklet'

    def _fresh_transient_name(self, prefix: str = '__stree_tmp') -> str:
        index = 0
        candidate = prefix
        while candidate in self.bindings or candidate in self.root.containers or candidate in self.globals:
            index += 1
            candidate = f'{prefix}{index}'
        return candidate

    def _fresh_callback_name(self, prefix: str = '__stree_callback') -> str:
        index = 0
        candidate = prefix
        while (candidate in self.bindings or candidate in self.root.containers or candidate in self.globals
               or candidate in self.callable_bindings):
            index += 1
            candidate = f'{prefix}{index}'
        return candidate

    def _array_constructor_name(self) -> str:
        return 'numpy.array'

    def _materialize_temporary_expression(self, value: ast.AST, descriptor: data.Data) -> ast.AST:
        name = self._fresh_transient_name()
        kind = 'scalar' if isinstance(descriptor, data.Scalar) else 'container'
        self._register_binding(name, descriptor, kind=kind)
        target = ast.Name(id=name, ctx=ast.Store())

        if isinstance(value, ast.Call) and self.callable_resolver.is_dace_program_call(value):
            self._materialize_call_args(value)
            self._emit_function_call(value, return_targets=[name])
            return ast.Name(id=name, ctx=ast.Load())

        if isinstance(value, ast.Call) and self.callable_resolver.is_sdfg_call(value):
            self._materialize_call_args(value)
            if self._emit_sdfg_call(value, return_targets=[name]):
                return ast.Name(id=name, ctx=ast.Load())

        if isinstance(value, ast.Call) and _is_pyobject_scalar_descriptor(descriptor):
            self.callback_handler.emit_assignment(name, value, 'pyobject call', descriptor)
            return ast.Name(id=name, ctx=ast.Load())

        if self._emit_computed_assignment(target, value, descriptor):
            return ast.Name(id=name, ctx=ast.Load())

        if isinstance(descriptor, data.Scalar):
            self._append_node(tn.AssignNode(name=name, value=CodeBlock(_unparse(value))))
        else:
            self._append_node(tn.StatementNode(code=CodeBlock(f'{name} = {_unparse(value)}')))
        return ast.Name(id=name, ctx=ast.Load())

    def _register_binding(self, name: str, descriptor: data.Data, kind: str) -> None:
        self._store_binding(name, descriptor, kind=kind)

    def _store_binding(self,
                       name: str,
                       descriptor: Optional[data.Data],
                       *,
                       kind: str,
                       structure: Optional[Any] = None) -> None:
        cloned = _clone_descriptor(descriptor) if descriptor is not None else None
        stored_structure = copy.deepcopy(structure)
        self.bindings[name] = _Binding(descriptor=cloned, kind=kind, structure=stored_structure)
        if cloned is None:
            return
        self.root.containers[name] = _clone_descriptor(cloned)
        if self.scope_stack[-1] is not self.root:
            self.scope_stack[-1].containers[name] = _clone_descriptor(cloned)
        self.globals[name] = cloned
        for free_symbol in cloned.free_symbols:
            self.root.symbols[free_symbol.name] = free_symbol
            self.globals[free_symbol.name] = free_symbol

    def _clone_constants(self, constants: Optional[Dict[str, Tuple[data.Data,
                                                                   Any]]]) -> Dict[str, Tuple[data.Data, Any]]:
        if not constants:
            return {}
        return {name: (_clone_descriptor(descriptor), value) for name, (descriptor, value) in constants.items()}

    def _infer_internal_iterator_binding(self, name: str, value: ast.AST,
                                         annotated_descriptor: Optional[data.Data]) -> None:
        inferred_binding = self.inferred_bindings.get(name)
        if inferred_binding is not None:
            self._store_binding(name,
                                inferred_binding.descriptor,
                                kind=inferred_binding.kind,
                                structure=inferred_binding.structure)
            return

        scalar_descriptor = self._infer_scalar_descriptor(value, annotated_descriptor)
        if scalar_descriptor is not None:
            self._store_binding(name, scalar_descriptor, kind='iterator-index', structure=scalar_descriptor)

    def _seed_inferred_target_bindings(self, target: ast.AST) -> None:
        for child in ast.walk(target):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Store):
                continue
            inferred_binding = self.inferred_bindings.get(child.id)
            if inferred_binding is None or inferred_binding.descriptor is None:
                continue
            self._store_binding(child.id,
                                inferred_binding.descriptor,
                                kind=inferred_binding.kind,
                                structure=inferred_binding.structure)

    def _structure_from_ast(self, node: ast.AST) -> Tuple[Optional[Any], bool]:
        if isinstance(node, ast.Name):
            binding = self.bindings.get(node.id)
            if binding is None:
                return (None, False)
            structure = binding.structure if binding.structure is not None else binding.descriptor
            uses_internal = binding.kind.startswith('iterator') or self._is_internal_iterator_binding_name(node.id)
            return (copy.deepcopy(structure), uses_internal)

        if isinstance(node, (ast.Tuple, ast.List)):
            elements: List[Any] = []
            uses_internal = False
            for element in node.elts:
                substructure, sub_internal = self._structure_from_ast(element)
                if substructure is None:
                    return (None, False)
                elements.append(substructure)
                uses_internal = uses_internal or sub_internal
            if isinstance(node, ast.List):
                return (elements, uses_internal)
            return (tuple(elements), uses_internal)

        return (None, False)

    def _ensure_reference_binding(self, name: str, descriptor: data.Data) -> data.Data:
        existing = self.bindings.get(name)
        if existing is not None and existing.descriptor is not None and isinstance(existing.descriptor, data.Reference):
            return _clone_descriptor(existing.descriptor)
        ref_desc = data.Reference.view(descriptor)
        self._register_binding(name, ref_desc, kind='reference')
        return _clone_descriptor(ref_desc)

    def _make_view_descriptor(self,
                              descriptor: data.Data,
                              shape: Optional[Sequence[Any]] = None,
                              new_axes: Optional[Sequence[int]] = None) -> data.Data:
        view_desc = data.View.view(descriptor)
        if shape is None:
            shape = descriptor.shape
        shape_list = list(shape)
        if new_axes:
            for axis in sorted(new_axes):
                shape_list.insert(axis, 1)
        if hasattr(view_desc, 'set_shape'):
            view_desc.set_shape(shape_list)
        return view_desc

    def _is_aliasable_descriptor(self, descriptor: data.Data) -> bool:
        return not isinstance(descriptor, data.Scalar)

    def _should_bind_expression_as_reference(self, value: ast.AST, descriptor: data.Data) -> bool:
        if not self._is_aliasable_descriptor(descriptor):
            return False
        if isinstance(value, ast.Attribute) and self._library_info_for_attribute(value) is not None:
            return False
        return isinstance(value, ast.Attribute)

    # ------------------------------------------------------------------ #
    #  Function-call detection for nested @dace.program calls              #
    # ------------------------------------------------------------------ #

    def _emit_function_call(self, call_node: ast.Call, return_targets: Optional[List[str]] = None) -> None:
        """Create a :class:`FunctionCallScope` placeholder and append it."""
        callee = self.callable_resolver.resolve_callable_value(call_node.func)
        callee_name = self.callable_resolver.callable_name(callee)
        arguments = self.callable_resolver.extract_argument_mapping(call_node, self._format_runtime_expression)
        specialization_args, specialization_kwargs, lambda_bindings, callable_bindings = self.callable_specializer.extract_call_specialization(
            call_node, _unparse)

        scope = tn.FunctionCallScope(
            children=[],
            call=tn.FrontendFunctionCall(callee_name=callee_name, arguments=arguments),
        )
        # Transient metadata consumed by the inlining pass.
        scope._callee_program = callee
        scope._call_node = call_node
        scope._call_args = specialization_args
        scope._call_kwargs = specialization_kwargs
        scope._lambda_bindings = lambda_bindings
        scope._callable_bindings = callable_bindings
        scope._captured_names = set(getattr(callee, 'captured_names', set()))
        scope._return_targets = return_targets
        self._append_node(scope)

    def _resolve_sdfg_call(self, call_node: ast.Call) -> Optional[Any]:
        callee = self.callable_resolver.resolve_callable_value(call_node.func)
        from dace import SDFG

        if isinstance(callee, SDFG):
            return callee

        if not hasattr(callee, '__sdfg__') or hasattr(callee, '__schedule_tree__'):
            return None

        specialization_args, specialization_kwargs, _, _ = self.callable_specializer.extract_call_specialization(
            call_node, _unparse)
        try:
            sdfg = callee.__sdfg__(*specialization_args, **specialization_kwargs)
        except Exception:
            return None

        return sdfg if isinstance(sdfg, SDFG) else None

    def _emit_sdfg_call(self, call_node: ast.Call, return_targets: Optional[List[str]] = None) -> bool:
        sdfg = self._resolve_sdfg_call(call_node)
        if sdfg is None:
            return False

        callee = self.callable_resolver.resolve_callable_value(call_node.func)
        callee_name = self.callable_resolver.callable_name(callee)
        arguments = self.callable_resolver.extract_argument_mapping(call_node, self._format_runtime_expression)
        self._append_node(
            tn.SDFGCallNode(sdfg=sdfg,
                            call=tn.FrontendFunctionCall(callee_name=callee_name, arguments=arguments),
                            return_targets=list(return_targets or [])))
        return True

    def _materialize_call_args(self, call_node: ast.Call) -> None:
        """Materialize array-valued call arguments into temporaries in-place."""
        ctx = self._expression_planning_context()
        for i, arg in enumerate(call_node.args):
            call_node.args[i] = self.expression_support.plan_expression(
                ctx, self.lambda_resolver.inline_known_lambda_calls(arg), materialize_root=True)
        for kw in call_node.keywords:
            kw.value = self.expression_support.plan_expression(ctx,
                                                               self.lambda_resolver.inline_known_lambda_calls(kw.value),
                                                               materialize_root=True)

    def _should_lower_as_library_call(self, node: ast.Call) -> bool:
        return self._library_info_for_call(node) is not None

    def _fresh_symbol(self, prefix: str = '__stree_sym') -> symbolic.symbol:
        index = 0
        candidate = prefix
        while (candidate in self.bindings or candidate in self.root.containers or candidate in self.globals
               or candidate in self.root.symbols):
            index += 1
            candidate = f'{prefix}{index}'
        symbol_value = symbolic.symbol(candidate, dtypes.int64)
        self.root.symbols[candidate] = symbol_value
        self.globals[candidate] = symbol_value
        return symbol_value

    def _library_info_for_call(self, node: ast.Call) -> Optional[Tuple[str, Dict[str, Any]]]:

        call_name = self._resolved_callable_name(node.func)
        if call_name in _INTERNAL_ITERATOR_HELPERS or call_name in {'range', 'prange', 'parrange'}:
            return None

        if isinstance(node.func, ast.Attribute):
            obj_access = self._resolve_data_access(node.func.value)
            if obj_access is not None:
                _, _, obj_desc, _ = obj_access
                classname = type(obj_desc).__name__
                if (oprepo.Replacements.get_method(classname, node.func.attr) is not None
                        or oprepo.Replacements.get_method_descriptor_inference(classname, node.func.attr) is not None):
                    properties = self._library_properties(node)
                    properties['receiver_class'] = classname
                    properties['access_kind'] = 'method'
                    return (node.func.attr, properties)

        if oprepo.Replacements.get(call_name) is None and oprepo.Replacements.get_descriptor_inference(
                call_name) is None:
            return None
        return (call_name, self._library_properties(node))

    def _library_info_for_attribute(self, node: ast.Attribute) -> Optional[Tuple[str, Dict[str, Any]]]:

        obj_access = self._resolve_data_access(node.value)
        if obj_access is None:
            return None
        _, _, obj_desc, _ = obj_access
        classname = type(obj_desc).__name__
        if (oprepo.Replacements.get_attribute(classname, node.attr) is None
                and oprepo.Replacements.get_attribute_descriptor_inference(classname, node.attr) is None):
            return None
        return (node.attr, {'receiver_class': classname, 'access_kind': 'attribute'})

    def _resolved_callable_name(self, node: ast.AST) -> str:
        textual_name = astutils.rname(node)
        callee = self.callable_resolver.resolve_callable_value(node)
        if callee is None:
            resolved = try_resolve_static_value(node, self._evaluation_context())
            if resolved is not UNRESOLVED:
                callee = resolved
        if callee is not None:
            module_name = getattr(callee, '__module__', None)
            callable_name = getattr(callee, '__name__', None)
            if module_name and callable_name and module_name != 'builtins':
                if module_name.startswith('numpy.'):
                    module_name = 'numpy'
                return f'{module_name}.{callable_name}'
            resolved_name = self.callable_resolver.callable_name(callee)
            if resolved_name:
                return resolved_name
        if '.' in textual_name:
            root_name, suffix = textual_name.split('.', 1)
            root_value = try_resolve_static_value(ast.Name(id=root_name, ctx=ast.Load()), self._evaluation_context())
            module_name = getattr(root_value, '__name__', None) if root_value is not UNRESOLVED else None
            if module_name is not None:
                if module_name.startswith('numpy.'):
                    module_name = 'numpy'
                return f'{module_name}.{suffix}'
        return textual_name

    def _is_internal_iterator_helper_call(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and astutils.rname(node.func) in _INTERNAL_ITERATOR_HELPERS

    def _is_internal_iterator_binding_name(self, name: str) -> bool:
        return name.startswith('__dace_iter_')

    def _should_bind_as_reference(self, value: ast.AST, source: data.Data) -> bool:
        if isinstance(source, data.Scalar):
            return False
        return isinstance(value, ast.Name)

    def _can_promote_to_reference(self, existing: data.Data, source: data.Data) -> bool:
        if isinstance(existing, data.Scalar) or isinstance(source, data.Scalar):
            return False
        if hasattr(existing, 'is_equivalent'):
            return existing.is_equivalent(source)
        return type(existing) is type(source)

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(self.globals)
        context.update({
            name: binding.descriptor
            for name, binding in self.bindings.items() if binding.descriptor is not None
        })
        context.update(self.root.symbols)
        for scope in self.scope_stack:
            context.update(scope.symbols)
        return context

    def _numpy_lowering_context(self) -> NumpyLoweringContext:
        return NumpyLoweringContext(bindings=self.bindings,
                                    evaluation_context=self._evaluation_context,
                                    resolve_output_target=self._resolve_output_target,
                                    tasklet_name=self._tasklet_name,
                                    fresh_symbol=self._fresh_symbol,
                                    fresh_name=self._fresh_transient_name,
                                    append_node=self._append_node,
                                    register_binding=self._register_binding)

    def _array_literal_context(self) -> ArrayLiteralContext:
        return ArrayLiteralContext(infer_descriptor=lambda node: self._infer_descriptor(node, '__probe'),
                                   infer_scalar_descriptor=self._infer_scalar_descriptor,
                                   evaluation_context=self._evaluation_context,
                                   resolve_output_target=self._resolve_output_target,
                                   resolve_data_access=self._resolve_data_access,
                                   resolve_callable_name=self._resolved_callable_name,
                                   tasklet_name=self._tasklet_name,
                                   array_constructor_name=self._array_constructor_name)

    def _dict_support_context(self, target_name: str = '__probe') -> DictSupportContext:
        return DictSupportContext(infer_descriptor=lambda node: self._infer_descriptor(node, target_name),
                                  infer_scalar_descriptor=self._infer_scalar_descriptor,
                                  evaluation_context=self._evaluation_context)

    def _expression_planning_context(self) -> ExpressionPlanningContext:
        return ExpressionPlanningContext(infer_descriptor=self._infer_plannable_expression_descriptor,
                                         materialize_expression=self._materialize_temporary_expression,
                                         resolve_data_access=self._resolve_data_access,
                                         collect_input_memlets=self._collect_input_memlets,
                                         resolve_output_target=self._resolve_output_target,
                                         resolve_callable_name=self._resolved_callable_name,
                                         should_materialize_call=lambda node:
                                         (self.callable_resolver.is_dace_program_call(node) or self.callable_resolver.
                                          is_sdfg_call(node) or self._should_lower_as_library_call(node)))

    def _infer_plannable_expression_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        node = self.lambda_resolver.inline_known_lambda_calls(node)
        generic_descriptor = self.expression_support.infer_expression_descriptor(self._expression_planning_context(),
                                                                                 node)
        if generic_descriptor is not None:
            return generic_descriptor

        if isinstance(node, ast.Call):
            array_literal_descriptor = self.array_literal_support.infer_expression_descriptor(
                self._array_literal_context(), node)
            if array_literal_descriptor is not None:
                return array_literal_descriptor

        numpy_descriptor = self.numpy_support.infer_expression_descriptor(self._numpy_lowering_context(), node)
        if numpy_descriptor is not None:
            return numpy_descriptor

        inferred_descriptor = self._infer_descriptor(node, '__probe')
        if inferred_descriptor is not None:
            return inferred_descriptor

        scalar_descriptor = self._infer_scalar_descriptor(node, None)
        if scalar_descriptor is not None:
            return scalar_descriptor

        access = self._resolve_data_access(node)
        if access is not None:
            _, _, descriptor, view_descriptor = access
            result = _clone_descriptor(view_descriptor or descriptor)
            result.transient = True
            return result

        return None

    def _is_callback_descriptor(self, descriptor: Optional[data.Data]) -> bool:
        return isinstance(descriptor, data.Scalar) and isinstance(descriptor.dtype, dtypes.callback)

    def _callback_specialization_value(self) -> data.Scalar:
        return data.Scalar(dtypes.callback(None), transient=False)

    def _make_nested_function_program(self, node: ast.FunctionDef) -> Optional[_NestedFunctionProgram]:
        if node.decorator_list:
            return None

        global_names, nonlocal_names = _collect_scope_declarations(node)

        class _SelfCallDetector(ast.NodeVisitor):

            def __init__(self, name: str) -> None:
                self.name = name
                self.recursive = False

            def visit_Call(self, call_node: ast.Call) -> None:
                if astutils.rname(call_node.func) == self.name:
                    self.recursive = True
                    return
                self.generic_visit(call_node)

        detector = _SelfCallDetector(node.name)
        detector.visit(node)
        if detector.recursive:
            return None

        return _NestedFunctionProgram(node.name,
                                      node,
                                      program_globals=self.globals,
                                      external_globals=self.external_globals,
                                      captured_names=set(global_names) | set(nonlocal_names),
                                      constants=self.root.constants,
                                      callback_mapping=self.root.callback_mapping,
                                      seed_bindings=self.bindings,
                                      lambda_bindings=self.lambda_bindings,
                                      callable_bindings=self.callable_bindings)

    def _resolve_external_scope_value(self, name: str) -> Any:
        if name in self.external_globals:
            return self.external_globals[name]
        return UNRESOLVED

    def _bind_external_scope_value(self, name: str, value: Any) -> None:
        try:
            descriptor = _binding_to_descriptor(value)
        except Exception:
            descriptor = _pyobject_scalar_descriptor()

        self._store_binding(name, descriptor, kind=_binding_kind_for_descriptor(descriptor))
        self.globals[name] = value

        if callable(value) and self.lambda_resolver.resolve_global_lambda_node(value) is None:
            self.callable_bindings[name] = value

        self.lambda_resolver.bind_value(name, value)

    def _update_callable_binding(self, name: str, value: ast.AST) -> None:
        if self.lambda_resolver.resolve_known_lambda_node(value) is not None:
            self.callable_bindings.pop(name, None)
            return
        resolved = self.callable_resolver.resolve_known_callable(value)
        if resolved is None:
            self.callable_bindings.pop(name, None)
            return
        self.callable_bindings[name] = resolved

    def _emit_if_chain(self, node: ast.If) -> None:
        parent = self.scope_stack[-1]
        current = node
        if_scope = tn.IfScope(condition=CodeBlock(_unparse(current.test)), children=[])
        if_scope.parent = parent
        parent.children.append(if_scope)
        self._visit_body(if_scope, current.body)

        orelse = current.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If):
            current = orelse[0]
            elif_scope = tn.ElifScope(condition=CodeBlock(_unparse(current.test)), children=[])
            elif_scope.parent = parent
            parent.children.append(elif_scope)
            self._visit_body(elif_scope, current.body)
            orelse = current.orelse

        if orelse:
            else_scope = tn.ElseScope(children=[])
            else_scope.parent = parent
            parent.children.append(else_scope)
            self._visit_body(else_scope, orelse)

    def _parse_for_indices(self, node: ast.AST) -> List[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            names = []
            for elt in node.elts:
                if not isinstance(elt, ast.Name):
                    raise TypeError('Only identifier loop targets are supported in the schedule-tree frontend')
                names.append(elt.id)
            return names
        raise TypeError('Only identifier loop targets are supported in the schedule-tree frontend')

    def _parse_for_iterator(self, node: ast.AST) -> Tuple[str, List[Tuple[str, str, str]]]:
        schedule_target = node
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            schedule_target = node.left

        if isinstance(schedule_target, ast.Call):
            iterator = astutils.rname(schedule_target.func)
            if iterator not in {'range', 'prange', 'parrange'}:
                raise TypeError(f'Unsupported for-loop iterator {iterator!r}')

            args = schedule_target.args
            if len(args) == 1:
                return 'range', [('0', _unparse(args[0]), '1')]
            if len(args) == 2:
                return 'range', [(_unparse(args[0]), _unparse(args[1]), '1')]
            if len(args) == 3:
                return 'range', [(_unparse(args[0]), _unparse(args[1]), _unparse(args[2]))]
            raise TypeError(f'Invalid number of arguments for {iterator!r}')

        if isinstance(schedule_target, ast.Subscript):
            iterator = astutils.rname(schedule_target.value)
            if iterator != 'dace.map':
                raise TypeError(f'Unsupported for-loop iterator {iterator!r}')
            return 'dace.map', self._parse_map_ranges(schedule_target)

        raise TypeError('Unsupported for-loop iterator expression in schedule-tree frontend')

    def _parse_map_ranges(self, node: ast.Subscript) -> List[Tuple[str, str, str]]:
        slice_node = node.slice
        if isinstance(slice_node, ast.Tuple):
            dims = list(slice_node.elts)
        else:
            dims = [slice_node]

        ranges: List[Tuple[str, str, str]] = []
        for dim in dims:
            if isinstance(dim, ast.Slice):
                start = '0' if dim.lower is None else _unparse(dim.lower)
                stop = _unparse(dim.upper) if dim.upper is not None else ''
                step = '1' if dim.step is None else _unparse(dim.step)
                ranges.append((start, stop, step))
            else:
                expr = _unparse(dim)
                ranges.append((expr, expr, '1'))
        return ranges
