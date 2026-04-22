# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from dataclasses import dataclass

import dace
from dace import data
from dace.data.pydata import PythonClass, python_dataclass_descriptor
from dace.frontend.python.schedule_tree.structure_support import bind_target_structure, descriptor_from_structure, \
    resolve_member_access
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_descriptor_from_structure_preserves_python_container_kind():
    tuple_descriptor = descriptor_from_structure(
        (data.Scalar(dace.float64, transient=True), data.Scalar(dace.float64, transient=True)))
    list_descriptor = descriptor_from_structure([data.Scalar(dace.float64, transient=True)])

    assert isinstance(tuple_descriptor, dace.data.PythonTuple)
    assert tuple_descriptor.dtype == dace.float64
    assert tuple(tuple_descriptor.shape) == (2, )
    assert isinstance(list_descriptor, dace.data.PythonList)
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


def test_python_dataclass_descriptor_preserves_structure_vs_python_class_split():

    @dataclass
    class Inner:
        x: dace.int32

    @dataclass
    class Outer:
        inner: Inner
        y: dace.float64

    by_value = python_dataclass_descriptor(Outer, by_value=True)
    python_object = python_dataclass_descriptor(Outer, by_value=False)

    assert isinstance(by_value, data.Structure)
    assert by_value.name == 'Outer'
    assert isinstance(by_value.members['inner'], data.Structure)

    assert isinstance(python_object, PythonClass)
    assert python_object.name == 'Outer'
    assert isinstance(python_object.members['inner'], data.Structure)
    assert isinstance(python_object.members['y'], data.Scalar)


def test_plain_class_descriptor_preserves_structure_vs_python_class_split():

    class Inner:
        x: dace.int32

    class Outer:
        inner: Inner
        y: dace.float64

    by_value = data.Structure.from_class(Outer)
    python_object = PythonClass.from_class(Outer)

    assert isinstance(by_value, data.Structure)
    assert by_value.name == 'Outer'
    assert isinstance(by_value.members['inner'], data.Structure)

    assert isinstance(python_object, PythonClass)
    assert python_object.name == 'Outer'
    assert isinstance(python_object.members['inner'], data.Structure)
    assert isinstance(python_object.members['y'], data.Scalar)


def test_resolve_member_access_returns_named_member_path():
    Bundle = dace.data.Structure({'data': dace.float64[4]}, name='Bundle')

    access = resolve_member_access('bundle', Bundle, 'data')

    assert access is not None
    assert access.data_name == 'bundle.data'
    assert isinstance(access.descriptor, data.Array)


def test_schedule_tree_supports_structure_member_copy():
    Bundle = dace.data.Structure({'data': dace.float64[4]}, name='Bundle')

    @dace.program
    def copy_member(bundle: Bundle, out: dace.float64[4]):
        out[:] = bundle.data[:]

    stree = copy_member.to_schedule_tree()

    assert isinstance(stree.children[0], tn.CopyNode)
    assert stree.children[0].target == 'out'
    assert stree.children[0].memlet.data == 'bundle.data'


def test_schedule_tree_supports_nested_structure_member_copy():
    Outer = dace.data.Structure({'inner': dace.data.Structure({'data': dace.float64[4]}, name='Inner')}, name='Outer')

    @dace.program
    def copy_member(bundle: Outer, out: dace.float64[4]):
        out[:] = bundle.inner.data[:]

    stree = copy_member.to_schedule_tree()

    assert isinstance(stree.children[0], tn.CopyNode)
    assert stree.children[0].target == 'out'
    assert stree.children[0].memlet.data == 'bundle.inner.data'


def test_schedule_tree_supports_python_class_member_copy():

    @dataclass
    class Inner:
        data: dace.float64[4]

    @dataclass
    class Outer:
        inner: Inner

    PyOuter = python_dataclass_descriptor(Outer, by_value=False)

    @dace.program
    def copy_member(bundle: PyOuter, out: dace.float64[4]):
        out[:] = bundle.inner.data[:]

    stree = copy_member.to_schedule_tree()

    assert isinstance(stree.children[0], tn.CopyNode)
    assert stree.children[0].target == 'out'
    assert stree.children[0].memlet.data == 'bundle.inner.data'
