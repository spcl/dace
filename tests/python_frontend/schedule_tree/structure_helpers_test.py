# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import dace
from dace import data
from dace.data.pydata import PythonList, PythonTuple
from dace.frontend.python.schedule_tree.structure_helpers import bind_target_structure, descriptor_from_structure


def test_descriptor_from_structure_preserves_python_container_kind():
    tuple_descriptor = descriptor_from_structure(
        (data.Scalar(dace.float64, transient=True), data.Scalar(dace.float64, transient=True)))
    list_descriptor = descriptor_from_structure([data.Scalar(dace.float64, transient=True)])

    assert isinstance(tuple_descriptor, PythonTuple)
    assert tuple_descriptor.dtype == dace.float64
    assert tuple(tuple_descriptor.shape) == (2, )
    assert isinstance(list_descriptor, PythonList)
    assert list_descriptor.dtype == dace.float64
    assert tuple(list_descriptor.shape) == (1, )


def test_bind_target_structure_visits_starred_targets():
    target = ast.parse('head, *tail, last = value').body[0].targets[0]
    seen = {}

    def _bind(name, structure):
        seen[name] = structure

    matched = bind_target_structure(target, ('A', 'B', 'C', 'D'), _bind)

    assert matched is True
    assert seen == {'head': 'A', 'tail': ['B', 'C'], 'last': 'D'}
