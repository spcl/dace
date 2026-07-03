# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from types import SimpleNamespace

from dace import data, dtypes

from dace.frontend.python.schedule_tree.callable_support import CallableResolver
from dace.frontend.python.schedule_tree.callback_support import CallbackHandler, CallbackOutliner
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_callback_outliner_wraps_assignment_as_function_and_call():
    node = ast.parse('it = iter(generator)').body[0]

    function_code, call_code = CallbackOutliner.outline(node,
                                                        callback_name='__stree_callback',
                                                        input_names=[],
                                                        output_names=['it'])

    assert function_code.as_string.startswith('def __stree_callback():')
    assert 'it = iter(generator)' in function_code.as_string
    assert function_code.as_string.endswith('return it')
    assert call_code.as_string == 'it = __stree_callback()'


def test_callback_outliner_supports_statement_groups():
    body = ast.parse('x = a + 1\ny = x + 1').body

    function_code, call_code = CallbackOutliner.outline(body,
                                                        callback_name='__stree_callback',
                                                        input_names=['a'],
                                                        output_names=['x', 'y'])

    assert function_code.as_string.startswith('def __stree_callback(a):')
    assert 'x = (a + 1)' in function_code.as_string
    assert 'y = (x + 1)' in function_code.as_string
    assert function_code.as_string.endswith('return (x, y)')
    assert call_code.as_string == '(x, y) = __stree_callback(a)'


def test_callback_handler_wraps_node_and_registers_unknown_outputs():
    appended_nodes = []
    bindings = {'generator': SimpleNamespace(descriptor=data.Scalar(dtypes.int64, transient=True), kind='scalar')}

    def _register_binding(name, descriptor, kind):
        bindings[name] = SimpleNamespace(descriptor=descriptor, kind=kind)

    handler = CallbackHandler(bindings=bindings,
                              callback_mutated_global_names=set(),
                              callable_resolver=CallableResolver(callable_bindings={}, evaluation_context=lambda: {}),
                              evaluation_context=lambda: {},
                              append_node=appended_nodes.append,
                              register_binding=_register_binding,
                              fresh_callback_name=lambda: '__stree_callback',
                              fresh_transient_name=lambda prefix='__stree_tmp': prefix,
                              render_callback_code=ast.unparse,
                              collect_scope_declarations=lambda node: (set(), set()),
                              raise_syntax_error=lambda node, message: (_ for _ in ()).throw(AssertionError(message)),
                              binding_kind_for_descriptor=lambda descriptor: 'scalar',
                              pyobject_scalar_descriptor=lambda: data.Scalar(dtypes.pyobject(), transient=True),
                              is_pyobject_scalar_descriptor=lambda descriptor: isinstance(
                                  getattr(descriptor, 'dtype', None), dtypes.pyobject),
                              is_iterator_protocol_call=lambda value: False,
                              is_iterator_next_call=lambda value: False)

    handler.wrap_node(ast.parse('it = iter(generator)').body[0], 'pyobject call')

    assert len(appended_nodes) == 1
    callback_node = appended_nodes[0]
    assert isinstance(callback_node, tn.PythonCallbackNode)
    assert callback_node.reason == 'pyobject call'
    assert callback_node.input_names == ['generator']
    assert callback_node.output_names == ['it']
    assert callback_node.outlined_function_name == '__stree_callback'
    assert callback_node.outlined_function_code is not None
    assert callback_node.outlined_call_code is not None
    assert bindings['it'].kind == 'scalar'
    assert bindings['it'].descriptor.dtype == dtypes.pyobject()
