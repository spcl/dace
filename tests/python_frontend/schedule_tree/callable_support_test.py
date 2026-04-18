# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

from dace import data, dtypes
from dace.frontend.python.schedule_tree.callable_support import CallableArgumentSpecializer
from dace.frontend.python.schedule_tree.lambda_support import LambdaResolver
from dace.frontend.python.schedule_tree.type_inference import _Binding


def test_callable_specializer_detects_callback_expressions():
    callback_descriptor = data.Scalar(dtypes.callback(None), transient=False)
    helper = CallableArgumentSpecializer(
        lambda_resolver=LambdaResolver({}, {'f': ast.parse('lambda a: a', mode='eval').body}, {}),
        bindings={'cb': _Binding(descriptor=callback_descriptor, kind='callback', structure=None)},
        resolve_known_callable=lambda node: None,
        infer_descriptor=lambda node: None,
        evaluation_context=lambda: {},
        resolve_data_access=lambda node: None,
        is_callback_descriptor=lambda descriptor: isinstance(descriptor, data.Scalar) and isinstance(
            descriptor.dtype, dtypes.callback),
        callback_specialization_value=lambda: callback_descriptor)

    assert helper.is_callback_expression(ast.Name(id='f', ctx=ast.Load()))
    assert helper.is_callback_expression(ast.Name(id='cb', ctx=ast.Load()))


def test_callable_specializer_extracts_lambda_and_callable_bindings():

    def cb(value):
        return value

    callback_descriptor = data.Scalar(dtypes.callback(None), transient=False)
    array_descriptor = data.Scalar(dtypes.float64, transient=True)
    helper = CallableArgumentSpecializer(
        lambda_resolver=LambdaResolver({}, {'f': ast.parse('lambda a: a', mode='eval').body}, {'cb': cb}),
        bindings={},
        resolve_known_callable=lambda node: cb if isinstance(node, ast.Name) and node.id == 'cb' else None,
        infer_descriptor=lambda node: array_descriptor if isinstance(node, ast.Name) and node.id == 'A' else None,
        evaluation_context=lambda: {},
        resolve_data_access=lambda node: None,
        is_callback_descriptor=lambda descriptor: isinstance(descriptor, data.Scalar) and isinstance(
            descriptor.dtype, dtypes.callback),
        callback_specialization_value=lambda: callback_descriptor)

    call_node = ast.parse('inner(A, f, cb=cb, literal=5)', mode='eval').body
    parameter_nodes = {
        'A': ast.Name(id='A', ctx=ast.Load()),
        'f': ast.Name(id='f', ctx=ast.Load()),
        'cb': ast.Name(id='cb', ctx=ast.Load()),
        'literal': ast.Constant(value=5),
    }

    args, kwargs, lambda_bindings, callable_bindings = helper.extract_call_specialization(
        call_node, parameter_nodes, ast.unparse)

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
