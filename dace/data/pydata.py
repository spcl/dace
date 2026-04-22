# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-native data descriptors.

These descriptors are primarily used by the direct schedule-tree Python
frontend. They intentionally distinguish between:

- ``Structure`` for classes with a fully known by-value field layout, and
- ``PythonClass`` for analyzable Python objects whose field information is
  useful to the frontend even when the value is not treated as a by-value
  structure.
"""
# TODO: These classes are incomplete, they require more support on the SDFG/connector level and in code generation
#       (bindings, C++ codegen, etc.). They are currently only used for the Schedule Tree-based Python frontend.

import copy
from dataclasses import is_dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Type

from dace import dtypes
from dace.data.core import Array, Data, Scalar, Structure, infer_structured_class_members
from dace.properties import NestedDataClassProperty, make_properties


def _clone_descriptor(descriptor: Data) -> Data:
    return copy.deepcopy(descriptor)


def _pyobject_descriptor(*, transient: bool) -> Scalar:
    return Scalar(dtypes.pyobject(), transient=transient)


def _normalize_descriptor(descriptor: Optional[Any], *, transient: bool) -> Data:
    if descriptor is None:
        return _pyobject_descriptor(transient=transient)
    if isinstance(descriptor, dtypes.typeclass):
        descriptor = Scalar(descriptor, transient=transient)
    elif not isinstance(descriptor, Data):
        raise TypeError(f'Unsupported nested data descriptor type: {type(descriptor)}')
    else:
        descriptor = _clone_descriptor(descriptor)
        descriptor.transient = transient
    return descriptor


def descriptors_equivalent(left: Data, right: Data) -> bool:
    if type(left) is not type(right):
        return False
    try:
        return left.is_equivalent(right)
    except Exception:
        return left == right


def merge_python_dict_component_descriptors(descriptors: Iterable[Optional[Data]], *, transient: bool) -> Data:
    merged: Optional[Data] = None
    for descriptor in descriptors:
        if descriptor is None:
            return _pyobject_descriptor(transient=transient)
        candidate = _normalize_descriptor(descriptor, transient=transient)
        if merged is None:
            merged = candidate
            continue
        if not descriptors_equivalent(merged, candidate):
            return _pyobject_descriptor(transient=transient)
    return merged or _pyobject_descriptor(transient=transient)


def infer_python_dict_descriptor_from_value(value: Mapping[Any, Any],
                                            descriptor_factory: Callable[[Any], Data],
                                            *,
                                            transient: bool = False) -> 'PythonDict':
    key_descriptors = []
    value_descriptors = []
    for key, mapped_value in value.items():
        try:
            key_descriptor = descriptor_factory(key)
        except Exception:
            key_descriptor = None
        try:
            value_descriptor = descriptor_factory(mapped_value)
        except Exception:
            value_descriptor = None
        key_descriptors.append(key_descriptor)
        value_descriptors.append(value_descriptor)
    return PythonDict(merge_python_dict_component_descriptors(key_descriptors, transient=transient),
                      merge_python_dict_component_descriptors(value_descriptors, transient=transient),
                      transient=transient)


def python_dataclass_descriptor(cls: Type[Any], *, by_value: bool = False, **overrides) -> Data:
    """Create the canonical descriptor for a Python dataclass.

    ``by_value=True`` selects ``Structure`` and is intended for classes whose
    full field layout is known and safe to pass by value. ``by_value=False``
    keeps the descriptor on the ``PythonClass`` path, which preserves analyzable
    member information without collapsing the object to a plain ``pyobject``.
    """
    if by_value:
        return Structure.from_dataclass(cls, **overrides)
    return PythonClass.from_dataclass(cls, **overrides)


@make_properties
class PythonList(Array):
    """Represents a native Python list argument."""

    def __init__(self, dtype: Any = dtypes.pyobject(), shape: Sequence[Any] = (1, ), **kwargs):
        super().__init__(dtype=dtype, shape=shape, **kwargs)

    def as_arg(self, with_types: bool = True, for_call: bool = False, name: str = None):
        if not with_types or for_call:
            return name
        return f'nb::list {name}'


@make_properties
class PythonTuple(Array):
    """Represents a native Python tuple argument."""

    def __init__(self, dtype: Any = dtypes.pyobject(), shape: Sequence[Any] = (1, ), **kwargs):
        super().__init__(dtype=dtype, shape=shape, **kwargs)

    def as_arg(self, with_types: bool = True, for_call: bool = False, name: str = None):
        if not with_types or for_call:
            return name
        return f'nb::tuple {name}'


@make_properties
class PythonDict(Data):
    """Represents a native Python dictionary with a uniform key and value type.

    This is intentionally a frontend-facing stub descriptor for now: it carries
    typed key/value metadata for analysis and schedule-tree lowering, but it is
    not a promise of full first-class mapping code generation support.
    """

    key_type = NestedDataClassProperty(default=None, allow_none=True)
    value_type = NestedDataClassProperty(default=None, allow_none=True)

    def _transient_setter(self, value):
        self._transient = value
        if self.key_type is not None:
            self.key_type.transient = value
        if self.value_type is not None:
            self.value_type.transient = value

    def __init__(self,
                 key_type: Optional[Data] = None,
                 value_type: Optional[Data] = None,
                 transient: bool = False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):
        self.key_type = _normalize_descriptor(key_type, transient=transient)
        self.value_type = _normalize_descriptor(value_type, transient=transient)
        super().__init__(dtypes.pyobject(), (1, ), transient, storage, location, lifetime, debuginfo)

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'PythonDict':
            raise TypeError('Invalid data type')

        ret = PythonDict()
        from dace import serialize
        serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __repr__(self):
        return f'PythonDict(key_type={self.key_type}, value_type={self.value_type})'

    def clone(self):
        return PythonDict(self.key_type,
                          self.value_type,
                          transient=self.transient,
                          storage=self.storage,
                          location=self.location,
                          lifetime=self.lifetime,
                          debuginfo=self.debuginfo)

    @property
    def strides(self):
        return [1]

    @property
    def total_size(self):
        return 1

    @property
    def offset(self):
        return [0]

    @property
    def start_offset(self):
        return 0

    @property
    def alignment(self):
        return 0

    @property
    def optional(self) -> bool:
        return False

    @property
    def pool(self) -> bool:
        return False

    @property
    def may_alias(self) -> bool:
        return False

    @property
    def free_symbols(self):
        result = set()
        result |= self.key_type.free_symbols
        result |= self.value_type.free_symbols
        return result

    def is_equivalent(self, other):
        return isinstance(other, PythonDict) and descriptors_equivalent(
            self.key_type, other.key_type) and descriptors_equivalent(self.value_type, other.value_type)

    def as_arg(self, with_types: bool = True, for_call: bool = False, name: str = None):
        if not with_types or for_call:
            return name
        return f'nb::dict {name}'

    def as_python_arg(self, with_types: bool = True, for_call: bool = False, name: str = None):
        if not with_types or for_call:
            return name
        return f'{name}: dict'


@make_properties
class PythonClass(Structure):
    """Represents an analyzable Python class with typed fields.

    Unlike ``Structure``, this descriptor keeps the value on the Python-object
    path even when member types are known.
    """

    def __init__(self, members, name: str = 'PythonClass', **kwargs):
        super().__init__(members=members, name=name, **kwargs)

    @staticmethod
    def from_json(json_obj, context=None):
        """Deserialize PythonClass from JSON, handling both 'PythonClass' and 'Structure' types."""
        if json_obj['type'] not in ('Structure', 'PythonClass'):
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = PythonClass({})
        from dace import serialize
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @classmethod
    def from_dataclass(cls, dataclass_type: Type[Any], **overrides) -> 'PythonClass':
        if not is_dataclass(dataclass_type):
            raise TypeError(f'{dataclass_type} is not a dataclass')
        return cls.from_class(dataclass_type, **overrides)

    @classmethod
    def from_class(cls, class_type: Type[Any], **overrides) -> 'PythonClass':
        members = infer_structured_class_members(class_type, **overrides)
        return cls(members, name=class_type.__name__)
