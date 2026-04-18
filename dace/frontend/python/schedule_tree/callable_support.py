# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for callback-like expressions and nested call specialization.

Example:
    If ``f`` is known to be ``lambda a, b: a + b``, then specializing the call
    ``inner(A, f)`` marks ``f`` as callback-typed and records the recovered
    lambda AST so the nested schedule-tree build can inline it later.
"""

import ast
import copy
from typing import Any, Callable, Dict, Optional, Tuple

from dace import data

from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value
from dace.frontend.python.schedule_tree.type_inference import _Binding


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

    def __init__(self, *, lambda_resolver: LambdaResolver, bindings: Dict[str, _Binding],
                 resolve_known_callable: Callable[[ast.AST], Optional[Any]],
                 infer_descriptor: Callable[[ast.AST],
                                            Optional[data.Data]], evaluation_context: Callable[[], Dict[str, Any]],
                 resolve_data_access: Callable[[ast.AST], Optional[Tuple[str, Any, data.Data, Optional[data.Data]]]],
                 is_callback_descriptor: Callable[[Optional[data.Data]],
                                                  bool], callback_specialization_value: Callable[[],
                                                                                                 data.Scalar]) -> None:
        self.lambda_resolver = lambda_resolver
        self.bindings = bindings
        self.resolve_known_callable = resolve_known_callable
        self.infer_descriptor = infer_descriptor
        self.evaluation_context = evaluation_context
        self.resolve_data_access = resolve_data_access
        self.is_callback_descriptor = is_callback_descriptor
        self.callback_specialization_value = callback_specialization_value

    def is_callback_expression(self, node: ast.AST) -> bool:
        """Return whether ``node`` should stay callback-typed in the tree."""
        if self.lambda_resolver.resolve_known_lambda_node(node) is not None:
            return True
        if self.resolve_known_callable(node) is not None:
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

        callable_value = self.resolve_known_callable(node)
        if callable_value is not None:
            return callable_value

        descriptor = self.infer_descriptor(node)
        if descriptor is not None:
            specialized = copy.deepcopy(descriptor)
            specialized.transient = False
            return specialized

        value = try_resolve_static_value(node, self.evaluation_context())
        if value is not UNRESOLVED:
            return value

        return None

    def extract_call_specialization(
            self, call_node: ast.Call, parameter_nodes: Dict[str, ast.AST],
            unparse: Callable[[ast.AST],
                              str]) -> Tuple[List[Any], Dict[str, Any], Dict[str, ast.Lambda], Dict[str, Any]]:
        """Build specialization payloads and known callable bindings for ``call_node``."""
        lambda_bindings: Dict[str, ast.Lambda] = {}
        callable_bindings: Dict[str, Any] = {}

        for param_name, argument_node in parameter_nodes.items():
            lambda_node = self.lambda_resolver.resolve_known_lambda_node(argument_node)
            if lambda_node is not None:
                lambda_bindings[param_name] = lambda_node
                continue

            callable_value = self.resolve_known_callable(argument_node)
            if callable_value is not None:
                callable_bindings[param_name] = callable_value

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
