# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-native data descriptors."""
# TODO: These classes are incomplete, they require more support on the SDFG/connector level and in code generation
#       (bindings, C++ codegen, etc.). They are currently only used for the Schedule Tree-based Python frontend.

from dataclasses import fields, is_dataclass
from typing import Any, Sequence, Type

from dace import dtypes
from dace.data.core import Array, Structure
from dace.properties import make_properties


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
class PythonClass(Structure):
    """Represents a Python dataclass with typed fields."""

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

    @staticmethod
    def from_dataclass(cls: Type[Any], **overrides) -> 'PythonClass':
        if not is_dataclass(cls):
            raise TypeError(f'{cls} is not a dataclass')

        from dace.data.creation import create_datadescriptor  # Avoid import cycle
        members = {}
        for field in fields(cls):
            members[field.name] = create_datadescriptor(field.type)
        members.update(overrides)
        return PythonClass(members, name=cls.__name__)
