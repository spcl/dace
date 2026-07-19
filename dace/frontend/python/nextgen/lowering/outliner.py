# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Callback outlining scaffolding for Python-callback schedule tree nodes.

Example:
    Wrapping ``it = iter(generator)`` as a callback keeps the original code
    block while also producing an outlined scaffold such as::

        def __callback_0():
            it = iter(generator)
            return it

        it = __callback_0()

    The outlined scaffold is what callback lowering executes; the schedule
    tree preserves the original callback code text as well.

This module is the canonical home of the outliner; the legacy
``dace.frontend.python.schedule_tree.callback_support`` module re-exports it
for the old frontend until its removal.
"""
from __future__ import annotations

import ast
from typing import List, Sequence, Tuple, Union

from dace.properties import CodeBlock
from dace.frontend.python import astutils

CallbackBody = Union[ast.AST, Sequence[ast.stmt]]


class CallbackOutliner:
    """Build callback scaffolding and basic name-flow metadata.

    The helper accepts either a single AST node or a list of statements. This
    lets callers wrap individual callback statements while also providing an
    API that can outline larger statement groups.
    """

    @staticmethod
    def analyze_name_flow(body: CallbackBody) -> Tuple[set[str], set[str]]:
        """Return ``(load_names, store_names)`` for a callback body."""
        inputs: set[str] = set()
        outputs: set[str] = set()
        for node in CallbackOutliner._body_nodes(body):
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        outputs.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        inputs.add(child.id)
                elif isinstance(child, ast.alias):
                    if child.asname:
                        outputs.add(child.asname)
                    else:
                        outputs.add(child.name.split('.')[0])
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    outputs.add(child.name)
                elif isinstance(child, ast.ExceptHandler) and isinstance(child.name, str):
                    outputs.add(child.name)
        return inputs, outputs

    @staticmethod
    def code_block(body: CallbackBody) -> CodeBlock:
        """Return a ``CodeBlock`` for the original callback body."""
        return CodeBlock(CallbackOutliner._body_nodes(body))

    @staticmethod
    def outline(body: CallbackBody, *, callback_name: str, input_names: Sequence[str],
                output_names: Sequence[str]) -> Tuple[CodeBlock, CodeBlock]:
        """Build outlined function and call-site scaffolding for ``body``."""
        input_names = list(input_names)
        output_names = list(output_names)
        function_body = CallbackOutliner._body_nodes(body)
        if output_names:
            returned = ast.Name(id=output_names[0], ctx=ast.Load())
            if len(output_names) > 1:
                returned = ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load()) for name in output_names], ctx=ast.Load())
            function_body.append(ast.Return(value=returned))

        function_def = ast.FunctionDef(name=callback_name,
                                       args=ast.arguments(posonlyargs=[],
                                                          args=[ast.arg(arg=name) for name in input_names],
                                                          vararg=None,
                                                          kwonlyargs=[],
                                                          kw_defaults=[],
                                                          kwarg=None,
                                                          defaults=[]),
                                       body=function_body or [ast.Pass()],
                                       decorator_list=[])
        function_code = CodeBlock([ast.fix_missing_locations(function_def)])

        call_expr = ast.Call(func=ast.Name(id=callback_name, ctx=ast.Load()),
                             args=[ast.Name(id=name, ctx=ast.Load()) for name in input_names],
                             keywords=[])
        if not output_names:
            call_stmt: ast.stmt = ast.Expr(value=call_expr)
        elif len(output_names) == 1:
            call_stmt = ast.Assign(targets=[ast.Name(id=output_names[0], ctx=ast.Store())], value=call_expr)
        else:
            call_stmt = ast.Assign(targets=[
                ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Store()) for name in output_names], ctx=ast.Store())
            ],
                                   value=call_expr)
        call_code = CodeBlock([ast.fix_missing_locations(call_stmt)])
        return function_code, call_code

    @staticmethod
    def _body_nodes(body: CallbackBody) -> List[ast.stmt]:
        if isinstance(body, Sequence) and not isinstance(body, ast.AST):
            return [astutils.copy_tree(statement) for statement in body]
        if isinstance(body, ast.stmt):
            return [astutils.copy_tree(body)]
        if isinstance(body, ast.AST):
            return [ast.Expr(value=astutils.copy_tree(body))]
        return [ast.Pass()]
