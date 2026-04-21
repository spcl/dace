# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import dace
from dace import data
from dace import dtypes
from dace.data.creation import create_datadescriptor
from dace.data.pydata import PythonDict
from dace.frontend.python.schedule_tree.dict_support import infer_dict_assignment_descriptor, infer_dict_literal_binding, \
    infer_dict_literal_descriptor, infer_dict_subscript_descriptor


def test_create_datadescriptor_infers_typed_python_dict():
    descriptor = create_datadescriptor({'a': 1.0, 'b': 2.0})

    assert isinstance(descriptor, PythonDict)
    assert isinstance(descriptor.key_type, data.Scalar)
    assert descriptor.key_type.dtype == dace.string
    assert isinstance(descriptor.value_type, data.Scalar)
    assert descriptor.value_type.dtype == dace.float64


def test_create_datadescriptor_infers_pyobject_for_heterogeneous_values():
    descriptor = create_datadescriptor({'a': 1.0, 'b': 'two'})

    assert isinstance(descriptor, PythonDict)
    assert isinstance(descriptor.key_type, data.Scalar)
    assert descriptor.key_type.dtype == dace.string
    assert isinstance(descriptor.value_type, data.Scalar)
    assert descriptor.value_type.dtype == dtypes.pyobject()


def test_infer_dict_literal_descriptor_uses_pyobject_for_unknown_value():
    dict_node = ast.parse("{'left': value, 'right': 2.0}", mode='eval').body

    descriptor = infer_dict_literal_descriptor(
        dict_node, lambda node: None, lambda node, annotated: data.Scalar(dace.float64, transient=True)
        if isinstance(node, ast.Constant) and isinstance(node.value, float) else None)

    assert isinstance(descriptor, PythonDict)
    assert descriptor.key_type.dtype == dtypes.pyobject()
    assert descriptor.value_type.dtype == dtypes.pyobject()


def test_infer_dict_literal_descriptor_falls_back_per_component():
    dict_node = ast.parse("{'left': value, 'right': 2.0}", mode='eval').body

    descriptor = infer_dict_literal_descriptor(
        dict_node, lambda node: data.Scalar(dace.string, transient=True)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) else None,
        lambda node, annotated: data.Scalar(dace.float64, transient=True)
        if isinstance(node, ast.Constant) and isinstance(node.value, float) else None)

    assert isinstance(descriptor, PythonDict)
    assert isinstance(descriptor.key_type, data.Scalar)
    assert descriptor.key_type.dtype == dace.string
    assert isinstance(descriptor.value_type, data.Scalar)
    assert descriptor.value_type.dtype == dtypes.pyobject()


def test_infer_dict_assignment_descriptor_widens_value_type():
    descriptor = PythonDict(data.Scalar(dace.string, transient=True),
                            data.Scalar(dace.float64, transient=True),
                            transient=True)
    target = ast.parse("mapping['left']", mode='eval').body
    value = ast.parse("'two'", mode='eval').body

    updated = infer_dict_assignment_descriptor(
        descriptor, target.slice, value, lambda node: None,
        lambda node, annotated: data.Scalar(dace.string, transient=True)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) else None, lambda: {})

    assert isinstance(updated, PythonDict)
    assert updated.key_type.dtype == dace.string
    assert updated.value_type.dtype == dtypes.pyobject()


def test_infer_dict_subscript_descriptor_uses_static_key_binding():
    descriptor = PythonDict(data.Scalar(dace.string, transient=True),
                            data.Scalar(dtypes.pyobject(), transient=True),
                            transient=True)
    node = ast.parse("{'left': 1.0, 'right': 'two'}", mode='eval').body
    binding = infer_dict_literal_binding(
        node, lambda current: None, lambda current, annotated: data.Scalar(dace.float64, transient=True)
        if isinstance(current, ast.Constant) and isinstance(current.value, float) else
        (data.Scalar(dace.string, transient=True)
         if isinstance(current, ast.Constant) and isinstance(current.value, str) else None), lambda: {})

    left = infer_dict_subscript_descriptor(descriptor, ast.parse("'left'", mode='eval').body, lambda: {}, binding)
    missing = infer_dict_subscript_descriptor(descriptor, ast.parse("'missing'", mode='eval').body, lambda: {}, binding)

    assert isinstance(left, data.Scalar)
    assert left.dtype == dace.float64
    assert missing is None
