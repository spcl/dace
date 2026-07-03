# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import pytest

from dace import data, dtypes
from dace.frontend.python.schedule_tree.callable_support import CallableArgumentSpecializer, CallableResolver
from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver
from dace.frontend.python.schedule_tree.type_inference import _Binding


def test_callable_specializer_detects_callback_expressions():
    callback_descriptor = data.Scalar(dtypes.callback(None), transient=False)
    callable_resolver = CallableResolver(callable_bindings={}, evaluation_context=lambda: {})
    helper = CallableArgumentSpecializer(
        lambda_resolver=LambdaResolver({}, {'f': ast.parse('lambda a: a', mode='eval').body}, {}),
        callable_resolver=callable_resolver,
        bindings={'cb': _Binding(descriptor=callback_descriptor, kind='callback', structure=None)},
        infer_descriptor=lambda node: None,
        resolve_data_access=lambda node: None,
        is_callback_descriptor=lambda descriptor: isinstance(descriptor, data.Scalar) and isinstance(
            descriptor.dtype, dtypes.callback),
        callback_specialization_value=lambda: callback_descriptor)

    assert helper.is_callback_expression(ast.Name(id='f', ctx=ast.Load()))
    assert helper.is_callback_expression(ast.Name(id='cb', ctx=ast.Load()))


def test_callable_specializer_extracts_lambda_and_callable_bindings():

    def cb(value):
        return value

    def inner(A, f, cb=None, literal=None):
        return A

    callback_descriptor = data.Scalar(dtypes.callback(None), transient=False)
    array_descriptor = data.Scalar(dtypes.float64, transient=True)
    callable_resolver = CallableResolver(callable_bindings={
        'inner': inner,
        'cb': cb
    },
                                         evaluation_context=lambda: {
                                             'inner': inner,
                                             'cb': cb
                                         })
    helper = CallableArgumentSpecializer(
        lambda_resolver=LambdaResolver({}, {'f': ast.parse('lambda a: a', mode='eval').body}, {'cb': cb}),
        callable_resolver=callable_resolver,
        bindings={},
        infer_descriptor=lambda node: array_descriptor if isinstance(node, ast.Name) and node.id == 'A' else None,
        resolve_data_access=lambda node: None,
        is_callback_descriptor=lambda descriptor: isinstance(descriptor, data.Scalar) and isinstance(
            descriptor.dtype, dtypes.callback),
        callback_specialization_value=lambda: callback_descriptor)

    call_node = ast.parse('inner(A, f, cb=cb, literal=5)', mode='eval').body

    args, kwargs, lambda_bindings, callable_bindings = helper.extract_call_specialization(call_node, ast.unparse)

    assert len(args) == 2
    assert isinstance(args[0], data.Scalar)
    assert args[0].dtype == dtypes.float64
    assert args[0].transient is False
    assert isinstance(args[1], data.Scalar)
    assert isinstance(args[1].dtype, dtypes.callback)
    assert kwargs['cb'] is cb
    assert kwargs['literal'] == 5
    assert 'f' in lambda_bindings
    assert callable_bindings == {'cb': cb}


if __name__ == '__main__':
    pytest.main([__file__])
