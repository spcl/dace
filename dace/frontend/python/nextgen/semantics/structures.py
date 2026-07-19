# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Structure member-access helpers for the next-generation frontend.

The frontend registers only base :class:`~dace.data.Structure` containers in
the repository; members are addressed by dotted data paths (``tracers.data``,
``outer.inner.data``) that resolve through the base structure's ``members``,
matching ``SDFG.arrays`` (a ``NestedDict``).

This module is the canonical home of these helpers; the legacy
``dace.frontend.python.schedule_tree.structure_support`` module re-exports
them for the old frontend until its removal.
"""
import copy
from dataclasses import dataclass
from typing import Mapping, Optional

from dace import data


def clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def structure_member_path(base_path: str, member_name: str) -> str:
    """The dotted repository data path of a structure member."""
    return f'{base_path}.{member_name}'


@dataclass(frozen=True)
class StructureMemberAccess:
    data_name: str
    descriptor: data.Data


def descriptor_members(descriptor: data.Data) -> Optional[Mapping[str, data.Data]]:
    """
    The member descriptors of a structure (or structure-view) descriptor, or
    None if the descriptor has no members.
    """
    if hasattr(descriptor, 'members'):
        return descriptor.members
    stype = getattr(descriptor, 'stype', None)
    if stype is not None and hasattr(stype, 'members'):
        return stype.members
    return None


def supports_member_access(descriptor: data.Data) -> bool:
    return descriptor_members(descriptor) is not None


def member_descriptor(descriptor: data.Data, member_name: str) -> Optional[data.Data]:
    """
    A clone of a structure member's descriptor carrying the parent's
    transience, or None if the descriptor has no such member. The registered
    (uncloned) member descriptor stays reachable through the repository's
    dotted-path resolution; the clone exists so inference and lowering can
    hold member descriptors without aliasing the structure's own.
    """
    members = descriptor_members(descriptor)
    if members is None or member_name not in members:
        return None
    result = clone_descriptor(members[member_name])
    result.transient = descriptor.transient
    return result


def resolve_member_access(base_name: str, descriptor: data.Data, member_name: str) -> Optional[StructureMemberAccess]:
    member = member_descriptor(descriptor, member_name)
    if member is None:
        return None
    return StructureMemberAccess(data_name=structure_member_path(base_name, member_name), descriptor=member)
