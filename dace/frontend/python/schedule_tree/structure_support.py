# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonical helpers for Python structure handling in schedule-tree lowering.

This module is the forward-looking home for schedule-tree structure support.
The older ``structure_helpers`` module remains as a compatibility shim while
callers migrate to this boundary.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Callable, Mapping, Optional, Sequence

from dace import data, dtypes
from dace.data.pydata import PythonList, PythonTuple


def clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


def structure_member_path(base_path: str, member_name: str) -> str:
    return f'{base_path}.{member_name}'


def descriptor_members(descriptor: data.Data) -> Optional[Mapping[str, data.Data]]:
    if hasattr(descriptor, 'members'):
        return descriptor.members
    stype = getattr(descriptor, 'stype', None)
    if stype is not None and hasattr(stype, 'members'):
        return stype.members
    return None


def supports_member_access(descriptor: data.Data) -> bool:
    return descriptor_members(descriptor) is not None


def member_descriptor(descriptor: data.Data, member_name: str) -> Optional[data.Data]:
    members = descriptor_members(descriptor)
    if members is None or member_name not in members:
        return None
    result = clone_descriptor(members[member_name])
    result.transient = descriptor.transient
    return result


def nested_member_descriptor(descriptor: data.Data, member_names: Sequence[str]) -> Optional[data.Data]:
    current = clone_descriptor(descriptor)
    for member_name in member_names:
        current = member_descriptor(current, member_name)
        if current is None:
            return None
    return current


def descriptor_from_structure(structure: Any) -> Optional[data.Data]:
    """Build a transient Python container descriptor for a tuple or list structure."""
    if isinstance(structure, data.Data):
        return clone_descriptor(structure)

    if not isinstance(structure, (list, tuple)):
        return None

    dtype = dtypes.pyobject()
    if structure:
        first = structure[0]
        if all(
                isinstance(element, data.Scalar) and isinstance(first, data.Scalar) and element.dtype == first.dtype
                for element in structure):
            dtype = first.dtype
        elif all(isinstance(element, data.Scalar) for element in structure):
            if first.dtype != dtypes.pyobject() and not any(element.dtype == dtypes.pyobject()
                                                            for element in structure[1:]):
                dtype = first.dtype
                for element in structure[1:]:
                    dtype = dtypes.result_type_of(dtype, element.dtype)

    descriptor_type = PythonList if isinstance(structure, list) else PythonTuple
    return descriptor_type(dtype=dtype, shape=(len(structure), ), transient=True)


def bind_target_structure(target: ast.AST, structure: Any, bind_name: Callable[[str, Any], None]) -> bool:
    """Walk a destructuring target and invoke *bind_name* for each bound name."""
    if isinstance(target, ast.Name):
        bind_name(target.id, structure)
        return True

    if isinstance(target, ast.Starred):
        if not isinstance(structure, list):
            structure = list(structure) if isinstance(structure, tuple) else [structure]
        return bind_target_structure(target.value, structure, bind_name)

    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(structure, (list, tuple)):
        starred_indices = [index for index, element in enumerate(target.elts) if isinstance(element, ast.Starred)]
        if len(starred_indices) > 1:
            return False
        if not starred_indices:
            if len(target.elts) != len(structure):
                return False
            return all(
                bind_target_structure(subtarget, substructure, bind_name)
                for subtarget, substructure in zip(target.elts, structure))

        starred_index = starred_indices[0]
        if len(structure) < len(target.elts) - 1:
            return False

        prefix_targets = target.elts[:starred_index]
        suffix_targets = target.elts[starred_index + 1:]
        prefix_structures = structure[:starred_index]
        suffix_structures = structure[len(structure) - len(suffix_targets):]
        middle_structure = list(structure[starred_index:len(structure) - len(suffix_targets)])

        return all(
            bind_target_structure(subtarget, substructure, bind_name)
            for subtarget, substructure in zip(prefix_targets, prefix_structures)) and bind_target_structure(
                target.elts[starred_index], middle_structure, bind_name) and all(
                    bind_target_structure(subtarget, substructure, bind_name)
                    for subtarget, substructure in zip(suffix_targets, suffix_structures))

    return False
