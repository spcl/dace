# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for native Python collection data descriptors.

Tests the PythonList, PythonTuple, PythonDict, PythonClass, and
PythonGenerator data descriptor classes defined in dace.data.pydata.
"""
import copy
import json
from collections import OrderedDict

import numpy as np
import pytest

import dace
from dace import dtypes, serialize
from dace.data import (Array, Data, Scalar, Stream, Structure, PythonList, PythonTuple, PythonDict, PythonClass,
                       PythonGenerator, create_datadescriptor)

# ============================================================================
# PythonList tests
# ============================================================================


class TestPythonList:

    def test_creation_basic(self):
        """Test basic PythonList construction."""
        desc = PythonList(dace.float64, 10)
        assert isinstance(desc, Array)
        assert desc.dtype == dace.float64
        assert desc.shape == (10, )
        assert desc.element_type == dace.float64

    def test_creation_empty(self):
        """Test creating an empty PythonList."""
        desc = PythonList(dace.int32, 0)
        assert desc.shape == (0, )
        assert desc.dtype == dace.int32

    def test_creation_transient(self):
        """Test creating a transient PythonList."""
        desc = PythonList(dace.float64, 5, transient=True)
        assert desc.transient is True

    def test_validate_must_be_1d(self):
        """Test that PythonList enforces 1D shape."""
        desc = PythonList(dace.float64, 5)
        # Force a non-1D shape and check validation fails
        desc._shape = (2, 3)
        with pytest.raises(TypeError, match='PythonList must always be 1D'):
            desc.validate()

    def test_repr(self):
        """Test string representation."""
        desc = PythonList(dace.float64, 3)
        assert 'PythonList' in repr(desc)

    def test_clone(self):
        """Test cloning a PythonList."""
        desc = PythonList(dace.float64, 10, transient=True)
        cloned = desc.clone()
        assert isinstance(cloned, PythonList)
        assert cloned.dtype == desc.dtype
        assert cloned.shape == desc.shape
        assert cloned.transient == desc.transient

    def test_as_arg_nanobind(self):
        """Test that as_arg generates nanobind type."""
        desc = PythonList(dace.float64, 5)
        arg = desc.as_arg(with_types=True, name='my_list')
        assert 'nb::list' in arg

    def test_as_arg_for_call(self):
        """Test that as_arg for call returns just the name."""
        desc = PythonList(dace.float64, 5)
        arg = desc.as_arg(with_types=True, for_call=True, name='my_list')
        assert arg == 'my_list'

    def test_from_python_list_float(self):
        """Test creating descriptor from a Python float list."""
        desc = PythonList.from_python_list([1.0, 2.0, 3.0])
        assert desc.dtype == dace.float64
        assert desc.shape == (3, )

    def test_from_python_list_int(self):
        """Test creating descriptor from a Python int list."""
        desc = PythonList.from_python_list([1, 2, 3, 4])
        assert desc.dtype == dace.int64
        assert desc.shape == (4, )

    def test_from_python_list_empty(self):
        """Test creating descriptor from an empty list."""
        desc = PythonList.from_python_list([])
        assert desc.dtype == dace.int32
        assert desc.shape == (0, )

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        desc = PythonList(dace.float64, 10, transient=True)
        json_obj = desc.to_json()
        assert json_obj['type'] == 'PythonList'

        restored = PythonList.from_json(json_obj)
        assert isinstance(restored, PythonList)
        assert restored.dtype == desc.dtype
        assert restored.shape == desc.shape
        assert restored.transient == desc.transient

    def test_add_to_sdfg(self):
        """Test adding a PythonList descriptor to an SDFG."""
        sdfg = dace.SDFG('test_list_sdfg')
        desc = PythonList(dace.float64, 5)
        sdfg.add_datadesc('my_list', desc)
        assert 'my_list' in sdfg.arrays
        assert isinstance(sdfg.arrays['my_list'], PythonList)

    def test_sdfg_json_round_trip(self):
        """Test that SDFG with PythonList survives serialization."""
        sdfg = dace.SDFG('test_list_sdfg_json')
        desc = PythonList(dace.float64, 5)
        sdfg.add_datadesc('my_list', desc)

        json_str = sdfg.to_json()
        restored_sdfg = dace.SDFG.from_json(json_str)
        assert 'my_list' in restored_sdfg.arrays
        assert isinstance(restored_sdfg.arrays['my_list'], PythonList)
        assert restored_sdfg.arrays['my_list'].dtype == dace.float64


# ============================================================================
# PythonTuple tests
# ============================================================================


class TestPythonTuple:

    def test_creation_basic(self):
        """Test basic PythonTuple construction."""
        desc = PythonTuple(dace.int32, 3)
        assert isinstance(desc, Array)
        assert desc.dtype == dace.int32
        assert desc.shape == (3, )
        assert desc.element_type == dace.int32

    def test_creation_transient(self):
        """Test creating a transient PythonTuple."""
        desc = PythonTuple(dace.float64, 2, transient=True)
        assert desc.transient is True

    def test_validate_must_be_1d(self):
        """Test that PythonTuple enforces 1D shape."""
        desc = PythonTuple(dace.float64, 5)
        desc._shape = (2, 3)
        with pytest.raises(TypeError, match='PythonTuple must always be 1D'):
            desc.validate()

    def test_repr(self):
        """Test string representation."""
        desc = PythonTuple(dace.int32, 3)
        assert 'PythonTuple' in repr(desc)

    def test_clone(self):
        """Test cloning a PythonTuple."""
        desc = PythonTuple(dace.float64, 5)
        cloned = desc.clone()
        assert isinstance(cloned, PythonTuple)
        assert cloned.dtype == desc.dtype
        assert cloned.shape == desc.shape

    def test_as_arg_nanobind(self):
        """Test that as_arg generates nanobind tuple type."""
        desc = PythonTuple(dace.float64, 3)
        arg = desc.as_arg(with_types=True, name='my_tuple')
        assert 'nb::tuple' in arg

    def test_from_python_tuple_float(self):
        """Test creating descriptor from a Python float tuple."""
        desc = PythonTuple.from_python_tuple((1.0, 2.0, 3.0))
        assert desc.dtype == dace.float64
        assert desc.shape == (3, )

    def test_from_python_tuple_int(self):
        """Test creating descriptor from a Python int tuple."""
        desc = PythonTuple.from_python_tuple((10, 20))
        assert desc.dtype == dace.int64
        assert desc.shape == (2, )

    def test_from_python_tuple_empty(self):
        """Test creating descriptor from an empty tuple."""
        desc = PythonTuple.from_python_tuple(())
        assert desc.shape == (0, )

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        desc = PythonTuple(dace.float64, 3, transient=True)
        json_obj = desc.to_json()
        assert json_obj['type'] == 'PythonTuple'

        restored = PythonTuple.from_json(json_obj)
        assert isinstance(restored, PythonTuple)
        assert restored.dtype == desc.dtype
        assert restored.shape == desc.shape

    def test_add_to_sdfg(self):
        """Test adding a PythonTuple descriptor to an SDFG."""
        sdfg = dace.SDFG('test_tuple_sdfg')
        desc = PythonTuple(dace.int32, 3)
        sdfg.add_datadesc('my_tuple', desc)
        assert 'my_tuple' in sdfg.arrays
        assert isinstance(sdfg.arrays['my_tuple'], PythonTuple)


# ============================================================================
# PythonDict tests
# ============================================================================


class TestPythonDict:

    def test_creation_basic(self):
        """Test basic PythonDict construction."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'x': dace.float64,
                              'y': dace.float64
                          },
                          name='PointDict')
        assert isinstance(desc, Structure)
        assert desc.key_type == dace.string
        assert desc.value_type == dace.float64
        assert 'x' in desc.members
        assert 'y' in desc.members

    def test_creation_empty(self):
        """Test creating an empty PythonDict."""
        desc = PythonDict()
        assert isinstance(desc, Structure)
        assert len(desc.members) == 0

    def test_creation_with_data_descriptors(self):
        """Test creating PythonDict with explicit Data descriptors."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'a': Scalar(dace.float64),
                              'b': Array(dace.float64, shape=(3, ))
                          },
                          name='MixedDict')
        assert isinstance(desc.members['a'], Scalar)
        assert isinstance(desc.members['b'], Array)

    def test_repr(self):
        """Test string representation."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={'x': dace.float64},
                          name='MyDict')
        r = repr(desc)
        assert 'PythonDict' in r
        assert 'x' in r

    def test_clone(self):
        """Test cloning a PythonDict."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'x': dace.float64,
                              'y': dace.float64
                          },
                          name='TestDict')
        cloned = desc.clone()
        assert isinstance(cloned, PythonDict)
        assert cloned.key_type == desc.key_type
        assert cloned.value_type == desc.value_type
        assert set(cloned.members.keys()) == set(desc.members.keys())

    def test_as_arg_nanobind(self):
        """Test that as_arg generates nanobind dict type."""
        desc = PythonDict(key_type=dace.string, value_type=dace.float64, keys_and_values={'x': dace.float64})
        arg = desc.as_arg(with_types=True, name='my_dict')
        assert 'nb::dict' in arg

    def test_from_python_dict(self):
        """Test creating descriptor from an actual Python dict."""
        desc = PythonDict.from_python_dict({'a': 1.0, 'b': 2.0, 'c': 3.0})
        assert desc.key_type == dace.string
        assert 'a' in desc.members
        assert 'b' in desc.members
        assert 'c' in desc.members

    def test_from_python_dict_int_values(self):
        """Test creating descriptor from dict with int values."""
        desc = PythonDict.from_python_dict({'count': 42, 'size': 100})
        assert 'count' in desc.members
        assert 'size' in desc.members

    def test_from_python_dict_non_string_keys_fails(self):
        """Test that non-string keys raise TypeError."""
        with pytest.raises(TypeError, match='keys must be strings'):
            PythonDict.from_python_dict({1: 'a', 2: 'b'})

    def test_from_python_dict_with_array_value(self):
        """Test creating descriptor from dict with numpy array values."""
        desc = PythonDict.from_python_dict({'data': np.array([1.0, 2.0, 3.0]), 'label': 42})
        assert isinstance(desc.members['data'], Array)
        assert isinstance(desc.members['label'], Scalar)

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'x': dace.float64,
                              'y': dace.float64
                          },
                          name='TestDict')
        json_obj = desc.to_json()
        assert json_obj['type'] == 'PythonDict'

        restored = PythonDict.from_json(json_obj)
        assert isinstance(restored, PythonDict)

    def test_add_to_sdfg(self):
        """Test adding a PythonDict descriptor to an SDFG."""
        sdfg = dace.SDFG('test_dict_sdfg')
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'x': dace.float64,
                              'y': dace.float64
                          },
                          name='PointDict')
        sdfg.add_datadesc('my_dict', desc)
        assert 'my_dict' in sdfg.arrays
        assert isinstance(sdfg.arrays['my_dict'], PythonDict)

    def test_dict_keys_as_connectors(self):
        """Test that dict keys are accessible via connectors."""
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'alpha': dace.float64,
                              'beta': dace.float64,
                              'gamma': dace.float64
                          })
        assert set(desc.members.keys()) == {'alpha', 'beta', 'gamma'}
        assert set(desc.keys()) >= {'alpha', 'beta', 'gamma'}


# ============================================================================
# PythonClass tests
# ============================================================================


class TestPythonClass:

    def test_creation_basic(self):
        """Test basic PythonClass construction."""
        desc = PythonClass(class_name='Point', fields={'x': dace.float64, 'y': dace.float64, 'z': dace.float64})
        assert isinstance(desc, Structure)
        assert desc.class_name == 'Point'
        assert 'x' in desc.members
        assert 'y' in desc.members
        assert 'z' in desc.members

    def test_creation_empty(self):
        """Test creating PythonClass with no fields."""
        desc = PythonClass(class_name='Empty')
        assert len(desc.members) == 0

    def test_creation_with_array_field(self):
        """Test PythonClass with array-typed fields."""
        desc = PythonClass(class_name='MyObj',
                           fields={
                               'value': dace.float64,
                               'data': Array(dace.float64, shape=(10, ))
                           })
        assert isinstance(desc.members['value'], Scalar)
        assert isinstance(desc.members['data'], Array)

    def test_repr(self):
        """Test string representation."""
        desc = PythonClass(class_name='Foo', fields={'bar': dace.int32})
        r = repr(desc)
        assert 'PythonClass' in r
        assert 'Foo' in r

    def test_clone(self):
        """Test cloning a PythonClass."""
        desc = PythonClass(class_name='Point', fields={'x': dace.float64, 'y': dace.float64})
        cloned = desc.clone()
        assert isinstance(cloned, PythonClass)
        assert cloned.class_name == desc.class_name
        assert set(cloned.members.keys()) == set(desc.members.keys())

    def test_as_arg_nanobind(self):
        """Test that as_arg generates nanobind object type."""
        desc = PythonClass(class_name='Obj', fields={'x': dace.float64})
        arg = desc.as_arg(with_types=True, name='my_obj')
        assert 'nb::object' in arg

    def test_from_python_class(self):
        """Test creating descriptor from a Python class with annotations."""

        class Point:
            x: float
            y: float
            z: int

        desc = PythonClass.from_python_class(Point)
        assert desc.class_name == 'Point'
        assert 'x' in desc.members
        assert 'y' in desc.members
        assert 'z' in desc.members

    def test_from_python_instance(self):
        """Test creating descriptor from a Python instance."""

        class Particle:
            x: float
            y: float
            mass: float

            def __init__(self, x, y, mass):
                self.x = x
                self.y = y
                self.mass = mass

        p = Particle(1.0, 2.0, 3.0)
        desc = PythonClass.from_python_class(p)
        assert desc.class_name == 'Particle'
        assert 'x' in desc.members
        assert 'y' in desc.members
        assert 'mass' in desc.members

    def test_from_python_class_with_overrides(self):
        """Test creating descriptor with field overrides."""

        class Vec:
            x: float
            y: float

        desc = PythonClass.from_python_class(Vec, x=dace.float32, y=dace.float32)
        assert isinstance(desc.members['x'], Scalar)
        assert isinstance(desc.members['y'], Scalar)

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        desc = PythonClass(class_name='MyClass', fields={'a': dace.float64, 'b': dace.int32})
        json_obj = desc.to_json()
        assert json_obj['type'] == 'PythonClass'

        restored = PythonClass.from_json(json_obj)
        assert isinstance(restored, PythonClass)

    def test_add_to_sdfg(self):
        """Test adding a PythonClass descriptor to an SDFG."""
        sdfg = dace.SDFG('test_class_sdfg')
        desc = PythonClass(class_name='Point', fields={'x': dace.float64, 'y': dace.float64})
        sdfg.add_datadesc('my_obj', desc)
        assert 'my_obj' in sdfg.arrays
        assert isinstance(sdfg.arrays['my_obj'], PythonClass)


# ============================================================================
# PythonGenerator tests
# ============================================================================


class TestPythonGenerator:

    def test_creation_basic(self):
        """Test basic PythonGenerator construction."""
        desc = PythonGenerator(dace.int64)
        assert isinstance(desc, Stream)
        assert desc.dtype == dace.int64
        assert desc.buffer_size == 1
        assert desc.shape == (1, )

    def test_creation_float(self):
        """Test PythonGenerator with float type."""
        desc = PythonGenerator(dace.float64)
        assert desc.dtype == dace.float64

    def test_creation_transient(self):
        """Test creating a transient PythonGenerator."""
        desc = PythonGenerator(dace.int32, transient=True)
        assert desc.transient is True

    def test_repr(self):
        """Test string representation."""
        desc = PythonGenerator(dace.int64)
        assert 'PythonGenerator' in repr(desc)

    def test_clone(self):
        """Test cloning a PythonGenerator."""
        desc = PythonGenerator(dace.float64, transient=True)
        cloned = desc.clone()
        assert isinstance(cloned, PythonGenerator)
        assert cloned.dtype == desc.dtype
        assert cloned.transient == desc.transient

    def test_as_arg_nanobind(self):
        """Test that as_arg generates nanobind object type."""
        desc = PythonGenerator(dace.int64)
        arg = desc.as_arg(with_types=True, name='my_gen')
        assert 'nb::object' in arg

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        desc = PythonGenerator(dace.float64, transient=True)
        json_obj = desc.to_json()
        assert json_obj['type'] == 'PythonGenerator'

        restored = PythonGenerator.from_json(json_obj)
        assert isinstance(restored, PythonGenerator)
        assert restored.dtype == desc.dtype

    def test_add_to_sdfg(self):
        """Test adding a PythonGenerator descriptor to an SDFG."""
        sdfg = dace.SDFG('test_gen_sdfg')
        desc = PythonGenerator(dace.int64)
        sdfg.add_datadesc('my_gen', desc)
        assert 'my_gen' in sdfg.arrays
        assert isinstance(sdfg.arrays['my_gen'], PythonGenerator)


# ============================================================================
# SDFG integration tests
# ============================================================================


class TestSDFGIntegration:

    def test_python_list_in_sdfg_with_state(self):
        """Test using PythonList as data in an SDFG with a state."""
        sdfg = dace.SDFG('list_sdfg')
        sdfg.add_datadesc('input_list', PythonList(dace.float64, 5))
        sdfg.add_datadesc('output_list', PythonList(dace.float64, 5, transient=True))

        state = sdfg.add_state('process')
        r = state.add_read('input_list')
        w = state.add_write('output_list')
        state.add_nedge(r, w, dace.Memlet('input_list[0:5]'))

        sdfg.validate()

    def test_python_dict_in_sdfg(self):
        """Test using PythonDict as data in an SDFG."""
        sdfg = dace.SDFG('dict_sdfg')
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'x': dace.float64,
                              'y': dace.float64
                          },
                          name='PointDict')
        sdfg.add_datadesc('point', desc)

        state = sdfg.add_state('init')
        assert 'point' in sdfg.arrays

    def test_python_class_in_sdfg(self):
        """Test using PythonClass as data in an SDFG."""
        sdfg = dace.SDFG('class_sdfg')
        desc = PythonClass(class_name='Particle', fields={'x': dace.float64, 'y': dace.float64, 'mass': dace.float64})
        sdfg.add_datadesc('particle', desc)

        state = sdfg.add_state('init')
        assert 'particle' in sdfg.arrays
        assert isinstance(sdfg.arrays['particle'], PythonClass)

    def test_python_list_sdfg_serialization(self):
        """Test full SDFG serialization with PythonList."""
        sdfg = dace.SDFG('list_serial')
        sdfg.add_datadesc('data', PythonList(dace.float64, 10))
        state = sdfg.add_state('s0')

        json_str = sdfg.to_json()
        restored = dace.SDFG.from_json(json_str)

        assert 'data' in restored.arrays
        assert isinstance(restored.arrays['data'], PythonList)
        assert restored.arrays['data'].shape == (10, )

    def test_python_dict_sdfg_serialization(self):
        """Test full SDFG serialization with PythonDict."""
        sdfg = dace.SDFG('dict_serial')
        desc = PythonDict(key_type=dace.string,
                          value_type=dace.float64,
                          keys_and_values={
                              'a': dace.float64,
                              'b': dace.float64
                          },
                          name='MyDict')
        sdfg.add_datadesc('cfg', desc)
        state = sdfg.add_state('s0')

        json_str = sdfg.to_json()
        restored = dace.SDFG.from_json(json_str)

        assert 'cfg' in restored.arrays
        assert isinstance(restored.arrays['cfg'], PythonDict)

    def test_python_generator_sdfg_serialization(self):
        """Test full SDFG serialization with PythonGenerator."""
        sdfg = dace.SDFG('gen_serial')
        sdfg.add_datadesc('gen', PythonGenerator(dace.int64))
        state = sdfg.add_state('s0')

        json_str = sdfg.to_json()
        restored = dace.SDFG.from_json(json_str)

        assert 'gen' in restored.arrays
        assert isinstance(restored.arrays['gen'], PythonGenerator)

    def test_multiple_collection_types_in_sdfg(self):
        """Test using multiple collection types together in an SDFG."""
        sdfg = dace.SDFG('multi_collection')

        sdfg.add_datadesc('my_list', PythonList(dace.float64, 10))
        sdfg.add_datadesc('my_tuple', PythonTuple(dace.int32, 3))
        sdfg.add_datadesc(
            'my_dict',
            PythonDict(key_type=dace.string,
                       value_type=dace.float64,
                       keys_and_values={'param': dace.float64},
                       name='Config'))
        sdfg.add_datadesc('my_obj', PythonClass(class_name='Obj', fields={
            'val': dace.float64,
        }))
        sdfg.add_datadesc('my_gen', PythonGenerator(dace.int64))

        state = sdfg.add_state('work')
        sdfg.validate()

        assert isinstance(sdfg.arrays['my_list'], PythonList)
        assert isinstance(sdfg.arrays['my_tuple'], PythonTuple)
        assert isinstance(sdfg.arrays['my_dict'], PythonDict)
        assert isinstance(sdfg.arrays['my_obj'], PythonClass)
        assert isinstance(sdfg.arrays['my_gen'], PythonGenerator)


# ============================================================================
# Python frontend integration tests
# ============================================================================


class TestFrontendIntegration:

    def test_create_datadescriptor_from_dict(self):
        """Test that create_datadescriptor creates PythonDict from a dict."""
        desc = create_datadescriptor({'x': 1.0, 'y': 2.0})
        assert isinstance(desc, PythonDict)
        assert 'x' in desc.members
        assert 'y' in desc.members

    def test_create_datadescriptor_from_dict_int_values(self):
        """Test create_datadescriptor with dict of int values."""
        desc = create_datadescriptor({'count': 42, 'size': 100})
        assert isinstance(desc, PythonDict)
        assert 'count' in desc.members

    def test_create_datadescriptor_from_dict_mixed_values(self):
        """Test create_datadescriptor with dict of mixed types."""
        desc = create_datadescriptor({'data': np.array([1.0, 2.0]), 'label': 42})
        assert isinstance(desc, PythonDict)
        assert isinstance(desc.members['data'], Array)
        assert isinstance(desc.members['label'], Scalar)


# ============================================================================
# Code generation integration tests
# ============================================================================


class TestCodegenIntegration:

    def test_python_list_codegen_arg_type(self):
        """Test that PythonList generates nb::list type."""
        desc = PythonList(dace.float64, 5)
        arg = desc.as_arg(with_types=True, name='my_list')
        assert arg == 'nb::list my_list'

    def test_python_tuple_codegen_arg_type(self):
        """Test that PythonTuple generates nb::tuple type."""
        desc = PythonTuple(dace.int32, 3)
        arg = desc.as_arg(with_types=True, name='my_tuple')
        assert arg == 'nb::tuple my_tuple'

    def test_python_dict_codegen_arg_type(self):
        """Test that PythonDict generates nb::dict type."""
        desc = PythonDict(key_type=dace.string, value_type=dace.float64)
        arg = desc.as_arg(with_types=True, name='my_dict')
        assert arg == 'nb::dict my_dict'

    def test_python_class_codegen_arg_type(self):
        """Test that PythonClass generates nb::object type."""
        desc = PythonClass(class_name='Obj', fields={'x': dace.float64})
        arg = desc.as_arg(with_types=True, name='my_obj')
        assert arg == 'nb::object my_obj'

    def test_python_generator_codegen_arg_type(self):
        """Test that PythonGenerator generates nb::object type."""
        desc = PythonGenerator(dace.int64)
        arg = desc.as_arg(with_types=True, name='my_gen')
        assert arg == 'nb::object my_gen'

    def test_codegen_sdfg_with_collections_validates(self):
        """Test that SDFG with Python collection descriptors validates."""
        sdfg = dace.SDFG('codegen_test')
        sdfg.add_datadesc('my_list', PythonList(dace.float64, 5))
        sdfg.add_datadesc(
            'my_dict',
            PythonDict(key_type=dace.string, value_type=dace.float64, keys_and_values={'x': dace.float64}, name='D'))
        state = sdfg.add_state('s0')
        sdfg.validate()


# ============================================================================
# Imports/exports test
# ============================================================================


class TestImports:

    def test_importable_from_dace_data(self):
        """Test that all classes are importable from dace.data."""
        from dace.data import (PythonList, PythonTuple, PythonDict, PythonClass, PythonGenerator)
        assert PythonList is not None
        assert PythonTuple is not None
        assert PythonDict is not None
        assert PythonClass is not None
        assert PythonGenerator is not None

    def test_in_all(self):
        """Test that all classes are in __all__."""
        import dace.data as dd
        assert 'PythonList' in dd.__all__
        assert 'PythonTuple' in dd.__all__
        assert 'PythonDict' in dd.__all__
        assert 'PythonClass' in dd.__all__
        assert 'PythonGenerator' in dd.__all__

    def test_inheritance(self):
        """Test that the class hierarchy is correct."""
        assert issubclass(PythonList, Array)
        assert issubclass(PythonList, Data)
        assert issubclass(PythonTuple, Array)
        assert issubclass(PythonTuple, Data)
        assert issubclass(PythonDict, Structure)
        assert issubclass(PythonDict, Data)
        assert issubclass(PythonClass, Structure)
        assert issubclass(PythonClass, Data)
        assert issubclass(PythonGenerator, Stream)
        assert issubclass(PythonGenerator, Data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
