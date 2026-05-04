# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for callback-like expressions and nested call specialization.

Example:
    If ``f`` is known to be ``lambda a, b: a + b``, then specializing the call
    ``inner(A, f)`` marks ``f`` as callback-typed and records the recovered
    lambda AST so the nested schedule-tree build can inline it later.
"""

import ast
import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from dace import data

from dace.frontend.python import astutils, preprocessing
from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.frontend.python.schedule_tree.type_inference import _Binding


def _binding_to_descriptor(value: Any) -> data.Data:
    descriptor = data.create_datadescriptor(value)
    if isinstance(descriptor, data.View):
        descriptor = descriptor.as_array()
    descriptor.transient = False
    return descriptor


def _callable_module_name(value: Any) -> str:
    function = value.__func__ if inspect.ismethod(value) else value
    module_name = getattr(function, '__module__', None)
    if isinstance(module_name, str):
        return module_name
    return ''


def _is_user_parseable_callable(value: Any) -> bool:
    module_name = _callable_module_name(value)
    return not module_name.startswith(('dace.frontend.python', 'sympy', 'numpy'))


def _unwrap_inline_callable(value: Any) -> Any:
    if isinstance(value, _ASTInlineCallable):
        return value._callee
    return value


class _ASTInlineCallable:
    """AST-backed inline wrapper for parseable Python callables."""

    _schedule_tree_inline_callable = True

    def __init__(self, callee: Any) -> None:
        self._callee = callee
        source_callable = callee.__func__ if inspect.ismethod(callee) else callee
        src_ast, src_file, src_line, src = astutils.function_to_ast(source_callable)
        if not src_ast.body or not isinstance(src_ast.body[0], ast.FunctionDef):
            raise TypeError('Expected a FunctionDef when wrapping a Python callable for schedule-tree inlining')

        self.function_ast = ast.fix_missing_locations(astutils.copy_tree(src_ast.body[0]))
        self.filename = src_file
        self.src_line = src_line
        self.src = src
        self.name = self.function_ast.name
        self.signature = inspect.signature(callee)
        self.argnames = [parameter.name for parameter in self.signature.parameters.values()]
        self.program_globals = copy.copy(getattr(source_callable, '__globals__', {}))
        self.external_globals = copy.copy(self.program_globals)
        self.constants: Dict[str, Tuple[data.Data, Any]] = {}
        self.callback_mapping: Dict[str, str] = {}
        self.captured_names: set[str] = set()
        self._bound_self = callee.__self__ if inspect.ismethod(callee) else None
        self._self_parameter = self.function_ast.args.args[
            0].arg if self._bound_self is not None and self.function_ast.args.args else None

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_ASTInlineCallable':
        memo[id(self)] = self
        return self

    def _generate_schedule_tree(self,
                                args: Tuple[Any],
                                kwargs: Dict[str, Any],
                                *,
                                lambda_bindings: Optional[Dict[str, ast.Lambda]] = None,
                                callable_bindings: Optional[Dict[str, Any]] = None):
        from dace.frontend.python import schedule_tree_frontend
        from dace.data.core import infer_structured_object_members
        from dace.data.pydata import PythonClass

        bound_args = self.signature.bind_partial(*args, **kwargs)
        argtypes = {name: _binding_to_descriptor(value) for name, value in bound_args.arguments.items()}

        seed_bindings: Dict[str, _Binding] = {}
        program_globals = copy.copy(self.program_globals)
        external_globals = copy.copy(self.external_globals)

        if self._bound_self is not None and self._self_parameter is not None:
            program_globals[self._self_parameter] = self._bound_self
            external_globals[self._self_parameter] = self._bound_self
            try:
                self_descriptor = PythonClass(infer_structured_object_members(self._bound_self),
                                              name=type(self._bound_self).__name__)
            except (TypeError, ValueError):
                self_descriptor = PythonClass({}, name=type(self._bound_self).__name__)
            seed_bindings[self._self_parameter] = _Binding(descriptor=self_descriptor, kind='container')

        parsed_ast = preprocessing.PreprocessedAST(self.filename, self.src_line, self.src,
                                                   astutils.copy_tree(self.function_ast), program_globals)
        return schedule_tree_frontend.build_schedule_tree(
            self.name,
            parsed_ast,
            argtypes,
            constants=self.constants,
            callback_mapping=self.callback_mapping,
            arg_names=[name for name in self.argnames if name in argtypes],
            lambda_bindings=lambda_bindings,
            callable_bindings=callable_bindings,
            seed_bindings=seed_bindings,
            external_globals=external_globals)


class CallableResolver:
    """Resolve callable values and nested-call metadata from AST nodes.

    Example:
        If ``inner`` is visible in the current evaluation context, then
        resolving ``inner(A, B).func`` returns the live callable object and the
        helper can derive call classification, parameter binding, and callee
        naming information from it.
    """

    def __init__(self, *, callable_bindings: Dict[str, Any], evaluation_context: Callable[[], Dict[str, Any]]) -> None:
        self.callable_bindings = callable_bindings
        self.evaluation_context = evaluation_context

    def resolve_static_value(self, node: ast.AST) -> Any:
        return try_resolve_static_value(node, self.evaluation_context())

    def resolve_callable_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name) and node.id in self.callable_bindings:
            return self._wrap_parseable_callable(self.callable_bindings[node.id])
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute):
            value = self._resolve_static_attribute(node)
        else:
            value = self.resolve_static_value(node)
        return self._wrap_parseable_callable(value)

    def resolve_known_callable(self, node: ast.AST) -> Optional[Any]:
        value = self.resolve_callable_value(node)
        if value is UNRESOLVED:
            return None
        if getattr(value, '_schedule_tree_inline_callable', False):
            return value
        if hasattr(value, '__schedule_tree__'):
            return None
        if not callable(value):
            return None
        from dace import SDFG
        if hasattr(value, '__sdfg__') and not isinstance(value, SDFG):
            return None
        return value

    def is_dace_program_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        value = self.resolve_callable_value(node.func)
        if value is UNRESOLVED:
            return False
        if getattr(value, '_schedule_tree_inline_callable', False):
            return True
        return hasattr(value, '__schedule_tree__')

    def is_sdfg_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        value = self.resolve_callable_value(node.func)
        if value is UNRESOLVED or hasattr(value, '__schedule_tree__'):
            return False
        from dace import SDFG
        return isinstance(value, SDFG) or hasattr(value, '__sdfg__')

    def callable_signature(self, callee: Any) -> inspect.Signature:
        from dace import SDFG

        if isinstance(callee, SDFG):
            arg_names = list(callee.arg_names)
        elif hasattr(callee, 'signature') and isinstance(callee.signature, inspect.Signature):
            return callee.signature
        elif hasattr(callee, '__schedule_tree_signature__'):
            arg_names, _ = callee.__schedule_tree_signature__()
        elif hasattr(callee, '__sdfg_signature__'):
            arg_names, _ = callee.__sdfg_signature__()
        elif hasattr(callee, 'f'):
            return inspect.signature(callee.f)
        else:
            return inspect.signature(callee)

        return inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in arg_names])

    def callable_name(self, callee: Any) -> str:
        function_name = getattr(getattr(callee, 'f', None), '__name__', None)
        if isinstance(function_name, str) and function_name:
            return function_name
        function_name = getattr(callee, '__name__', None)
        if isinstance(function_name, str) and function_name:
            return function_name
        if hasattr(callee, 'name') and isinstance(callee.name, str):
            return callee.name
        return type(callee).__name__

    def extract_argument_mapping(self, call_node: ast.Call, format_runtime_expression: Callable[[ast.AST],
                                                                                                str]) -> Dict[str, str]:
        callee = self.resolve_callable_value(call_node.func)
        sig = self.callable_signature(callee)
        params = [param for param in sig.parameters.values() if param.name != 'self']

        mapping: Dict[str, str] = {}
        for index, arg in enumerate(call_node.args):
            if index < len(params):
                mapping[params[index].name] = format_runtime_expression(arg)
        for kw in call_node.keywords:
            mapping[kw.arg] = format_runtime_expression(kw.value)
        return mapping

    def call_parameter_nodes(self, call_node: ast.Call) -> Dict[str, ast.AST]:
        callee = self.resolve_callable_value(call_node.func)
        sig = self.callable_signature(callee)
        params = [param for param in sig.parameters.values() if param.name != 'self']
        keywords = {kw.arg: kw.value for kw in call_node.keywords if kw.arg is not None}
        try:
            bound = inspect.Signature(params).bind_partial(*call_node.args, **keywords)
        except TypeError:
            return {}
        return dict(bound.arguments)

    def _resolve_static_attribute(self, node: ast.Attribute) -> Any:
        owner = self.resolve_static_value(node.value)
        if owner is UNRESOLVED:
            return UNRESOLVED
        try:
            return getattr(owner, node.attr)
        except Exception:
            return UNRESOLVED

    def _wrap_parseable_callable(self, value: Any) -> Any:
        if value is UNRESOLVED:
            return value
        if getattr(value, '_schedule_tree_inline_callable', False):
            return value
        if hasattr(value, '__schedule_tree__') or hasattr(value, '__sdfg__'):
            return value
        if not callable(value):
            return value
        from dace import SDFG

        if isinstance(value, (SDFG, data.Data)) or inspect.isclass(value):
            return value
        if inspect.ismethod(value) and isinstance(getattr(value, '__self__', None), data.Data):
            return value
        if inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
            return value
        try:
            if inspect.ismethod(value) or inspect.isfunction(value):
                source_callable = value.__func__ if inspect.ismethod(value) else value
                if getattr(source_callable, '__name__', None) == '<lambda>':
                    return value
                if inspect.isgeneratorfunction(source_callable):
                    return value
                if not _is_user_parseable_callable(value):
                    return value
                return _ASTInlineCallable(value)

            bound_call = getattr(value, '__call__', None)
            if bound_call is None or inspect.isbuiltin(bound_call) or inspect.ismethoddescriptor(bound_call):
                return value
            if not _is_user_parseable_callable(bound_call):
                return value

            call_impl = getattr(type(value), '__call__', None)
            if call_impl in {None, object.__call__}:
                return value

            return _ASTInlineCallable(bound_call)
        except (TypeError, OSError):
            return value


class CallableArgumentSpecializer:
    """Recognize callback-like values and specialize nested call arguments.

    The helper keeps the schedule-tree builder focused on structure creation by
    isolating the rules for callback expressions, lambda argument bindings, and
    argument specialization for nested function calls.

    Example:
        Given ``f = lambda a, b: a + b``, specializing ``inner(A, f)`` returns
        a callback descriptor for ``f`` and records ``f`` in the lambda binding
        map for the nested call scope.
    """

    def __init__(self, *, lambda_resolver: LambdaResolver, callable_resolver: CallableResolver,
                 bindings: Dict[str, _Binding], infer_descriptor: Callable[[ast.AST], Optional[data.Data]],
                 resolve_data_access: Callable[[ast.AST], Optional[Tuple[str, Any, data.Data, Optional[data.Data]]]],
                 is_callback_descriptor: Callable[[Optional[data.Data]],
                                                  bool], callback_specialization_value: Callable[[],
                                                                                                 data.Scalar]) -> None:
        self.lambda_resolver = lambda_resolver
        self.callable_resolver = callable_resolver
        self.bindings = bindings
        self.infer_descriptor = infer_descriptor
        self.resolve_data_access = resolve_data_access
        self.is_callback_descriptor = is_callback_descriptor
        self.callback_specialization_value = callback_specialization_value

    def is_callback_expression(self, node: ast.AST) -> bool:
        """Return whether ``node`` should stay callback-typed in the tree."""
        if self.lambda_resolver.resolve_known_lambda_node(node) is not None:
            return True
        if self.callable_resolver.resolve_known_callable(node) is not None:
            return True
        if isinstance(node, ast.Name):
            binding = self.bindings.get(node.id)
            if binding is not None and self.is_callback_descriptor(binding.descriptor):
                return True
        access = self.resolve_data_access(node)
        if access is None:
            return False
        _, _, descriptor, view_descriptor = access
        return self.is_callback_descriptor(view_descriptor or descriptor)

    def specialize_argument(self, node: ast.AST) -> Any:
        """Return the specialization payload for one nested call argument."""
        lambda_node = self.lambda_resolver.resolve_known_lambda_node(node)
        if lambda_node is not None:
            return self.callback_specialization_value()

        callable_value = self.callable_resolver.resolve_known_callable(node)
        if callable_value is not None:
            return _unwrap_inline_callable(callable_value)

        descriptor = self.infer_descriptor(node)
        if descriptor is not None:
            specialized = copy.deepcopy(descriptor)
            specialized.transient = False
            return specialized

        value = self.callable_resolver.resolve_static_value(node)
        if value is not UNRESOLVED:
            return value

        return None

    def extract_call_specialization(
            self, call_node: ast.Call,
            unparse: Callable[[ast.AST],
                              str]) -> Tuple[List[Any], Dict[str, Any], Dict[str, ast.Lambda], Dict[str, Any]]:
        """Build specialization payloads and known callable bindings for ``call_node``."""
        parameter_nodes = self.callable_resolver.call_parameter_nodes(call_node)
        lambda_bindings: Dict[str, ast.Lambda] = {}
        callable_bindings: Dict[str, Any] = {}

        for param_name, argument_node in parameter_nodes.items():
            lambda_node = self.lambda_resolver.resolve_known_lambda_node(argument_node)
            if lambda_node is not None:
                lambda_bindings[param_name] = lambda_node
                continue

            callable_value = self.callable_resolver.resolve_known_callable(argument_node)
            if callable_value is not None:
                callable_bindings[param_name] = _unwrap_inline_callable(callable_value)

        args = [self._specialize_or_unparse(arg, unparse) for arg in call_node.args]
        kwargs = {
            kw.arg: self._specialize_or_unparse(kw.value, unparse)
            for kw in call_node.keywords if kw.arg is not None
        }
        return args, kwargs, lambda_bindings, callable_bindings

    def _specialize_or_unparse(self, node: ast.AST, unparse: Callable[[ast.AST], str]) -> Any:
        specialized = self.specialize_argument(node)
        if specialized is None:
            return unparse(node)
        return specialized
