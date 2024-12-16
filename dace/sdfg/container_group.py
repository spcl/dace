# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import OrderedDict
from typing import Set, Union
import typing
from dace import data
from dace.data import Data
from dace import serialize, symbolic
from dace.properties import OrderedDictProperty, Property, make_properties
from enum import Enum

import numpy
import sympy


class ContainerGroupFlatteningMode(Enum):
    ArrayOfStructs = 1
    StructOfArrays = 2


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
    is_cg = Property(dtype=bool, default=False, allow_none=False)
    is_ca = Property(dtype=bool, default=False, allow_none=False)

    def __init__(self, name, is_cg, is_ca):
        self.name = name
        self.members = OrderedDict()
        self.is_cg = is_cg
        self.is_ca = is_ca
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
            f"{k}: {v.__repr__()}" for k, v in self.members.items()
        )
        return f"ContainerGroup(name='{self.name}', is_cg={self.is_cg}, is_ca={self.is_ca}, members={{ {members_repr} }})"

    def __str__(self):
        return self.__repr__()

    def _soa_from_struct(self, name, structure, acc_shape):
        self._add_members(name, structure, acc_shape=None)

    @classmethod
    def from_struct(
        cls,
        name: str,
        struct_or_container_array: typing.Union[data.Structure, data.ContainerArray],
        is_cg: bool,
        is_ca: bool,
    ) -> "ContainerGroup":
        dg = cls(name=name, is_cg=is_cg, is_ca=is_ca)
        assert is_cg ^ is_ca

        if isinstance(struct_or_container_array, data.Structure):
            struct = struct_or_container_array
            for member_name, member in struct.members.items():
                new_member = None
                if isinstance(member, data.Structure):
                    new_member = cls.from_struct(
                                    name=member_name,
                                    struct_or_container_array=member,
                                    is_cg=True,
                                    is_ca=False)
                elif isinstance(member, data.ContainerArray):
                    new_member = cls.from_struct(name=member_name,
                                                 struct_or_container_array=member,
                                                 is_cg=False,
                                                 is_ca=True)
                elif isinstance(member, (data.Array, data.Scalar)):
                    new_member = member
                elif isinstance(
                    member, (sympy.Basic, symbolic.SymExpr, int, numpy.integer)
                ):
                    new_member = data.Scalar(symbolic.symtype(member))
                else:
                    raise TypeError(f"Unsupported member type in Structure: {type(member)}")

                dg.add_member(
                    name=f"{member_name}",
                    member=new_member
                )
        else:
            assert isinstance(struct_or_container_array, data.ContainerArray)
            container_array = struct_or_container_array
            member = container_array.stype
            member_name = None
            new_member = None

            if isinstance(member, data.Structure):
                # Recursively convert nested Structures
                member_name = member.name
                new_member = cls.from_struct(name=member.name,
                                             struct_or_container_array=member,
                                             is_cg=True,
                                             is_ca=False)
            elif isinstance(member, data.ContainerArray):
                raise Exception("Two container arrays in a row is currently not supported")
            elif isinstance(member, (data.Array, data.Scalar)):
                new_member = member
            elif isinstance(
                member, (sympy.Basic, symbolic.SymExpr, int, numpy.integer)
            ):
                new_member=data.Scalar(symbolic.symtype(member))
            else:
                raise TypeError(f"Unsupported member type in Structure: {type(member)}")
            dg.add_member(
                name=member_name if member_name is not None else "Leaf",
                member=new_member
            )

        return dg
