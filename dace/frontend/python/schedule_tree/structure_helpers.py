# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for tuple and list structure handling in schedule-tree lowering.

These helpers are used when the frontend needs to describe or traverse Python
structures such as ``(A, B)`` or ``head, *tail``.

Example:
    ``descriptor_from_structure((Scalar(float64), Scalar(float64)))`` returns a
    transient ``PythonTuple`` descriptor with two elements.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Callable, Optional

from dace import data, dtypes
from dace.data.pydata import PythonList, PythonTuple


def clone_descriptor(descriptor: data.Data) -> data.Data:
    return copy.deepcopy(descriptor)


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
    """Walk a destructuring target and invoke *bind_name* for each bound name.

    Returns ``True`` when *structure* is compatible with *target* and all names
    were visited. Returns ``False`` when the target shape does not match.
    """
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
