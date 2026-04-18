# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver


def test_lambda_resolver_inlines_named_lambda_calls():
    lambda_bindings = {'f': ast.parse('lambda a, b: a + b', mode='eval').body}
    resolver = LambdaResolver({}, lambda_bindings, {})

    rewritten = resolver.inline_known_lambda_calls(ast.parse('f(A, B)', mode='eval').body)

    assert isinstance(rewritten, ast.BinOp)
    assert astutils.unparse(rewritten.left) == 'A'
    assert astutils.unparse(rewritten.right) == 'B'


def test_lambda_resolver_recovers_global_lambda_with_capture():
    offset = 3.0
    f = lambda a: a + offset
    resolver = LambdaResolver({'f': f}, {}, {})

    lambda_node = resolver.resolve_known_lambda_node(ast.Name(id='f', ctx=ast.Load()))

    assert lambda_node is not None
    assert isinstance(lambda_node.body, ast.BinOp)
    assert astutils.unparse(lambda_node.body.left) == 'a'
    assert astutils.unparse(lambda_node.body.right) == '3.0'


def test_lambda_resolver_exposes_callable_capture_through_globals():

    def callee(a, b):
        return a + b

    f = lambda a, b: callee(a, b)
    resolver = LambdaResolver({'f': f}, {}, {})

    lambda_node = resolver.resolve_known_lambda_node(ast.Name(id='f', ctx=ast.Load()))

    assert lambda_node is not None
    assert isinstance(lambda_node.body, ast.Call)
    assert astutils.unparse(lambda_node.body.func) == 'callee'
    assert resolver.globals['callee'] is callee
