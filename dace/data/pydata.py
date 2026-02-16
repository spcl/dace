# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Native Python collection data descriptors.

This module contains data descriptor classes for representing native Python
collections (lists, tuples, dictionaries, classes, generators) in SDFGs.
These descriptors enable DaCe to work with Python data structures natively,
preserving their semantics in generated code.

The data container classes are strongly typed. Code generation will emit
appropriate nanobind types (``nb::list``, ``nb::tuple``, ``nb::dict``,
``nb::object``) when nanobind is available.

.. note::
    This module is intended to create useful shims to/from existing Python
    codes, not to generate the highest-performing code. For performance-critical
    applications, prefer native DaCe arrays and structures.
"""
import copy as cp

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import sympy as sp

from dace import dtypes, serialize, symbolic
from dace.data.core import Array, Data, Scalar, Stream, Structure, _arrays_from_json, _arrays_to_json
from dace.properties import (EnumProperty, ListProperty, Property, ShapeProperty, SymbolicProperty, TypeClassProperty,
                             make_properties)
from dace.utils import prod as _prod


@make_properties
class PythonList(Array):
    """
    Data descriptor for a Python list.

    Represented as a 1D array with an element type, but code-generated as
    ``nb::list`` (nanobind) when used in Python-interoperable contexts.
    Supports dynamic-length lists at the SDFG level.

    :param element_type: The DaCe type of the list elements (e.g., ``dace.float64``).
    :param size: Number of elements (can be symbolic).
    :param transient: Whether this is a transient (temporary) data container.

    Example usage::

        # Create a PythonList descriptor for a list of 10 float64 values
        desc = PythonList(dace.float64, 10)

        # Add to SDFG
        sdfg.add_datadesc('my_list', desc)
    """

    def __init__(self,
                 dtype=dtypes.int32,
                 size=0,
                 transient=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):

        shape = (size, )
        super(PythonList, self).__init__(dtype=dtype,
                                         shape=shape,
                                         transient=transient,
                                         storage=storage,
                                         location=location,
                                         lifetime=lifetime,
                                         debuginfo=debuginfo)

    def __repr__(self):
        return 'PythonList (element_type=%s, size=%s)' % (self.dtype, self.shape[0])

    @property
    def element_type(self):
        """Returns the element type of the list."""
        return self.dtype

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonList':
            raise TypeError("Invalid data type")

        ret = PythonList.__new__(PythonList)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        # Default shape-related properties
        if not ret.offset:
            ret.offset = [0] * len(ret.shape)
        if not ret.strides:
            ret.strides = [_prod(ret.shape[i + 1:]) for i in range(len(ret.shape))]
        if ret.total_size == 0:
            ret.total_size = _prod(ret.shape)

        ret.validate()
        return ret

    def clone(self):
        return PythonList(self.dtype, self.shape[0], self.transient, self.storage, self.location, self.lifetime,
                          self.debuginfo)

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return 'nb::list %s' % name

    @staticmethod
    def from_python_list(lst: list) -> 'PythonList':
        """
        Creates a PythonList descriptor from an actual Python list.

        :param lst: The Python list to create a descriptor from.
        :return: A PythonList data descriptor.
        """
        import numpy as np
        if len(lst) == 0:
            return PythonList(dtypes.int32, 0)
        # Infer type from first element
        elem = lst[0]
        if isinstance(elem, float):
            dtype = dtypes.float64
        elif isinstance(elem, int):
            dtype = dtypes.int64
        elif isinstance(elem, complex):
            dtype = dtypes.complex128
        elif isinstance(elem, bool):
            dtype = dtypes.bool_
        elif isinstance(elem, np.generic):
            dtype = dtypes.typeclass(type(elem))
        else:
            dtype = dtypes.pyobject()
        return PythonList(dtype, len(lst))


@make_properties
class PythonTuple(Array):
    """
    Data descriptor for a Python tuple.

    Represented as a 1D array with an element type, but code-generated as
    ``nb::tuple`` (nanobind). Unlike PythonList, tuples are conceptually
    immutable. The size is always fixed.

    :param element_type: The DaCe type of the tuple elements (e.g., ``dace.float64``).
    :param size: Number of elements.
    :param transient: Whether this is a transient (temporary) data container.

    Example usage::

        # Create a PythonTuple descriptor for a tuple of 3 int32 values
        desc = PythonTuple(dace.int32, 3)

        # Add to SDFG
        sdfg.add_datadesc('my_tuple', desc)
    """

    def __init__(self,
                 dtype=dtypes.int32,
                 size=0,
                 transient=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):

        shape = (size, )
        super(PythonTuple, self).__init__(dtype=dtype,
                                          shape=shape,
                                          transient=transient,
                                          storage=storage,
                                          location=location,
                                          lifetime=lifetime,
                                          debuginfo=debuginfo)

    def __repr__(self):
        return 'PythonTuple (element_type=%s, size=%s)' % (self.dtype, self.shape[0])

    @property
    def element_type(self):
        """Returns the element type of the tuple."""
        return self.dtype

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonTuple':
            raise TypeError("Invalid data type")

        ret = PythonTuple.__new__(PythonTuple)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        if not ret.offset:
            ret.offset = [0] * len(ret.shape)
        if not ret.strides:
            ret.strides = [_prod(ret.shape[i + 1:]) for i in range(len(ret.shape))]
        if ret.total_size == 0:
            ret.total_size = _prod(ret.shape)

        ret.validate()
        return ret

    def clone(self):
        return PythonTuple(self.dtype, self.shape[0], self.transient, self.storage, self.location, self.lifetime,
                           self.debuginfo)

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return 'nb::tuple %s' % name

    @staticmethod
    def from_python_tuple(tpl: tuple) -> 'PythonTuple':
        """
        Creates a PythonTuple descriptor from an actual Python tuple.

        :param tpl: The Python tuple to create a descriptor from.
        :return: A PythonTuple data descriptor.
        """
        import numpy as np
        if len(tpl) == 0:
            return PythonTuple(dtypes.int32, 0)
        elem = tpl[0]
        if isinstance(elem, float):
            dtype = dtypes.float64
        elif isinstance(elem, int):
            dtype = dtypes.int64
        elif isinstance(elem, complex):
            dtype = dtypes.complex128
        elif isinstance(elem, bool):
            dtype = dtypes.bool_
        elif isinstance(elem, np.generic):
            dtype = dtypes.typeclass(type(elem))
        else:
            dtype = dtypes.pyobject()
        return PythonTuple(dtype, len(tpl))


@make_properties
class PythonDict(Structure):
    """
    Data descriptor for a Python dictionary.

    Extends :class:`Structure` to represent a dictionary where keys are known
    at compile time and encoded as structure members (connectors). The key
    and value types are tracked. Code-generated as ``nb::dict`` (nanobind).

    Each key in the dictionary becomes a member of the underlying structure.
    This means that keys must be known at SDFG construction time.

    :param key_type: The DaCe type for dictionary keys (e.g., ``dace.string``).
    :param value_type: The DaCe type for dictionary values (e.g., ``dace.float64``).
    :param keys_and_values: A dictionary mapping string keys to Data descriptors
        representing the values.
    :param name: Name for this dict type.
    :param transient: Whether this is a transient (temporary) data container.

    Example usage::

        # Create a PythonDict from known keys
        desc = PythonDict(
            key_type=dace.string,
            value_type=dace.float64,
            keys_and_values={'x': dace.float64, 'y': dace.float64},
            name='MyDict'
        )

        # Add to SDFG
        sdfg.add_datadesc('my_dict', desc)
    """

    key_type = TypeClassProperty(default=dtypes.string, desc="Type of dictionary keys")
    value_type = TypeClassProperty(default=dtypes.int32, desc="Type of dictionary values")

    def __init__(self,
                 key_type=dtypes.string,
                 value_type=dtypes.int32,
                 keys_and_values: Optional[Dict[str, Any]] = None,
                 name: str = 'PythonDict',
                 transient: bool = False,
                 storage: dtypes.StorageType = dtypes.StorageType.Default,
                 location: Optional[Dict[str, str]] = None,
                 lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope,
                 debuginfo: dtypes.DebugInfo = None):

        if keys_and_values is None:
            keys_and_values = {}

        # Convert raw type values to Scalar descriptors for Structure compatibility
        members = OrderedDict()
        for k, v in keys_and_values.items():
            if isinstance(v, Data):
                members[k] = v
            elif isinstance(v, dtypes.typeclass):
                members[k] = Scalar(v)
            else:
                members[k] = Scalar(v)

        self.key_type = key_type
        self.value_type = value_type

        super(PythonDict, self).__init__(members=members,
                                         name=name,
                                         transient=transient,
                                         storage=storage,
                                         location=location,
                                         lifetime=lifetime,
                                         debuginfo=debuginfo)

    def __repr__(self):
        return 'PythonDict (key_type=%s, value_type=%s, keys=[%s])' % (self.key_type, self.value_type, ', '.join(
            self.members.keys()))

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonDict':
            raise TypeError("Invalid data type")

        ret = PythonDict.__new__(PythonDict)
        serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def clone(self):
        return PythonDict(self.key_type, self.value_type, dict(self.members), self.name, self.transient, self.storage,
                          self.location, self.lifetime, self.debuginfo)

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return 'nb::dict %s' % name

    @staticmethod
    def from_python_dict(d: dict, name: str = 'PythonDict') -> 'PythonDict':
        """
        Creates a PythonDict descriptor from an actual Python dictionary.

        All keys must be strings. Value types are inferred from the values.

        :param d: The Python dictionary to create a descriptor from.
        :param name: Name for the resulting dict type.
        :return: A PythonDict data descriptor.
        """
        import numpy as np
        members = OrderedDict()
        key_type = dtypes.string
        value_type = dtypes.int32  # Default

        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f'PythonDict keys must be strings, got {type(k).__name__}')
            if isinstance(v, float):
                vtype = dtypes.float64
            elif isinstance(v, int):
                vtype = dtypes.int64
            elif isinstance(v, bool):
                vtype = dtypes.bool_
            elif isinstance(v, np.ndarray):
                members[k] = Array(dtypes.typeclass(v.dtype.type), shape=v.shape)
                continue
            else:
                vtype = dtypes.pyobject()
            members[k] = Scalar(vtype)
            value_type = vtype

        return PythonDict(key_type=key_type, value_type=value_type, keys_and_values=members, name=name)


@make_properties
class PythonClass(Structure):
    """
    Data descriptor for a general Python class.

    Extends :class:`Structure` to represent a Python class instance. Accessing
    fields in the class acts similarly to Structure fields - to access, simply
    add a connector with the field's name. Code-generated as ``nb::object``
    (nanobind).

    :param class_name: The Python class name.
    :param fields: Dictionary of field names to Data descriptors.
    :param transient: Whether this is a transient (temporary) data container.

    Example usage::

        # Create a PythonClass descriptor
        desc = PythonClass(
            class_name='Point',
            fields={'x': dace.float64, 'y': dace.float64, 'z': dace.float64}
        )
    """

    class_name = Property(dtype=str, default='', desc="Original Python class name")

    def __init__(self,
                 class_name: str = 'PythonClass',
                 fields: Optional[Dict[str, Any]] = None,
                 transient: bool = False,
                 storage: dtypes.StorageType = dtypes.StorageType.Default,
                 location: Optional[Dict[str, str]] = None,
                 lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope,
                 debuginfo: dtypes.DebugInfo = None):

        if fields is None:
            fields = {}

        # Convert raw types to Scalar descriptors for Structure compatibility
        members = OrderedDict()
        for k, v in fields.items():
            if isinstance(v, Data):
                members[k] = v
            elif isinstance(v, dtypes.typeclass):
                members[k] = Scalar(v)
            else:
                members[k] = Scalar(v)

        self.class_name = class_name

        super(PythonClass, self).__init__(members=members,
                                          name=class_name,
                                          transient=transient,
                                          storage=storage,
                                          location=location,
                                          lifetime=lifetime,
                                          debuginfo=debuginfo)

    def __repr__(self):
        return 'PythonClass %s (%s)' % (self.class_name, ', '.join([f'{k}: {v}' for k, v in self.members.items()]))

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonClass':
            raise TypeError("Invalid data type")

        ret = PythonClass.__new__(PythonClass)
        serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def clone(self):
        return PythonClass(self.class_name, dict(self.members), self.transient, self.storage, self.location,
                           self.lifetime, self.debuginfo)

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return 'nb::object %s' % name

    @staticmethod
    def from_python_class(cls_or_obj, **overrides) -> 'PythonClass':
        """
        Creates a PythonClass descriptor from a Python class or instance.

        :param cls_or_obj: A Python class or instance to convert.
        :param overrides: Optional overrides for fields.
        :return: A PythonClass data descriptor.
        """
        import dataclasses
        import numpy as np

        if isinstance(cls_or_obj, type):
            # It's a class - use annotations
            cls = cls_or_obj
            fields = {}
            annotations = getattr(cls, '__annotations__', {})
            for field_name, field_type in annotations.items():
                if isinstance(field_type, dtypes.typeclass):
                    fields[field_name] = field_type
                elif isinstance(field_type, Data):
                    fields[field_name] = field_type
                elif field_type is float or field_type == float:
                    fields[field_name] = dtypes.float64
                elif field_type is int or field_type == int:
                    fields[field_name] = dtypes.int64
                elif field_type is bool or field_type == bool:
                    fields[field_name] = dtypes.bool_
                else:
                    fields[field_name] = dtypes.pyobject()
            fields.update(overrides)
            return PythonClass(class_name=cls.__name__, fields=fields)
        else:
            # It's an instance - infer from attribute values
            obj = cls_or_obj
            cls_name = type(obj).__name__
            fields = {}
            annotations = getattr(type(obj), '__annotations__', {})
            for field_name in annotations:
                val = getattr(obj, field_name, None)
                if isinstance(val, (int, np.integer)):
                    fields[field_name] = dtypes.int64
                elif isinstance(val, (float, np.floating)):
                    fields[field_name] = dtypes.float64
                elif isinstance(val, np.ndarray):
                    fields[field_name] = Array(dtypes.typeclass(val.dtype.type), shape=val.shape)
                else:
                    fields[field_name] = dtypes.pyobject()
            fields.update(overrides)
            return PythonClass(class_name=cls_name, fields=fields)


@make_properties
class PythonGenerator(Stream):
    """
    Data descriptor for a Python generator.

    Represents general stateful memory that, upon accessing with a read memlet,
    will generate a different value every time. The semantics are similar to
    DaCe streams (FIFO queues).

    Edges that write into a generator (without a ``set`` connector) are
    disallowed.

    :param dtype: The type of values yielded by the generator.
    :param transient: Whether this is a transient (temporary) data container.

    Example usage::

        # Create a PythonGenerator descriptor
        desc = PythonGenerator(dace.int64)
    """

    def __init__(self,
                 dtype=dtypes.int32,
                 transient=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):

        # Generators have a buffer size of 1 (single-element stream)
        super(PythonGenerator, self).__init__(dtype=dtype,
                                              buffer_size=1,
                                              shape=(1, ),
                                              transient=transient,
                                              storage=storage,
                                              location=location,
                                              lifetime=lifetime,
                                              debuginfo=debuginfo)

    def __repr__(self):
        return 'PythonGenerator (dtype=%s)' % self.dtype

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonGenerator':
            raise TypeError("Invalid data type")

        ret = PythonGenerator.__new__(PythonGenerator)
        serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def clone(self):
        return PythonGenerator(self.dtype, self.transient, self.storage, self.location, self.lifetime, self.debuginfo)

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return 'nb::object %s' % name
