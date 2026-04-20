# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import dace
from dace import data
from dace import dtypes
from dace.data.creation import create_datadescriptor
from dace.data.pydata import PythonDict
from dace.frontend.python.schedule_tree.dict_support import infer_dict_literal_descriptor


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
