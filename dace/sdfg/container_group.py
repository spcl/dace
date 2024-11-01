# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import OrderedDict
from typing import Set, Union
from dace import data
from dace.data import Data
from dace import serialize, symbolic
from dace.properties import OrderedDictProperty, Property, make_properties
from enum import Enum

import numpy
import sympy

class ContainerGroupFlatteningMode(Enum):
    ArrayOfStructs = 1
    StructsOfArrays = 2

def _members_to_json(members):
    if members is None:
        return None
    return [(k, serialize.to_json(v)) for k, v in members.items()]


def _members_from_json(obj, context=None):
    if obj is None:
        return {}
    return OrderedDict((k, serialize.from_json(v, context)) for k, v in obj)


@make_properties
class ContainerGroup:
    name = Property(dtype=str, default="", allow_none=False)
    members = OrderedDictProperty(
        default=OrderedDict(),
        desc="Dictionary of structure members",
        from_json=_members_from_json,
        to_json=_members_to_json,
    )

    def __init__(self, name):
        self.name = name
        self.members = OrderedDict()
        self._validate()

    def add_member(self, name: str, member: Union[Data, "ContainerGroup"]):
        if name is None or name == "":
            name = len(self.members)
        self.members[name] = member

    @property
    def free_symbols(self) -> Set[symbolic.SymbolicType]:
        """Returns a set of undefined symbols in this data descriptor."""
        result = set()
        for k, v in self.members.items():
            result |= v.free_symbols
        return result

    def __call__(self):
        return self

    def validate(self):
        self._validate()

    def _validate(self):
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)
        retdict = {"type": type(self).__name__, "attributes": attrs}
        return retdict

    def is_equivalent(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return serialize.dumps(self) == serialize.dumps(other)

    def __hash__(self):
        return hash(serialize.dumps(self))

    def __repr__(self):
        members_repr = ", ".join(
            f"{k}: {v.__class__.__name__}" for k, v in self.members.items()
        )
        return f"DataGroup(name='{self.name}', members={{ {members_repr} }})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_struct(cls, name: str, structure: data.Structure, flattening_mode: ContainerGroupFlatteningMode) -> "ContainerGroup":
        dg = cls(name)

        for member_name, member in structure.members.items():
            if isinstance(member, data.Structure):
                # Recursively convert nested Structures
                dg.add_member(
                    name=f"{member_name}",
                    member=cls.from_struct(
                        name=f"{member_name}", structure=member
                    )
                )
            elif isinstance(member, (data.Array, data.Scalar)):
                # Directly add Arrays and Scalars
                dg.add_member(member_name, member)
            elif isinstance(
                member, (sympy.Basic, symbolic.SymExpr, int, numpy.integer)
            ):
                # Convert other types to Scalar
                dg.add_member(member_name, data.Scalar(symbolic.symtype(member)))
            else:
                raise TypeError(f"Unsupported member type in Structure: {type(member)}")

        return dg