# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from dace import data, dtypes
from dace.data.creation import create_datadescriptor
from dace.data.pydata import PythonDict, merge_python_dict_component_descriptors
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

DescriptorInference = Callable[[ast.AST], Optional[data.Data]]
ScalarDescriptorInference = Callable[[ast.AST, Optional[data.Data]], Optional[data.Data]]
EvaluationContextFactory = Callable[[], Dict[str, Any]]


@dataclass
class StaticDictBinding:
    entries: Dict[Any, data.Data]


@dataclass(frozen=True)
class DictSupportContext:
    infer_descriptor: DescriptorInference
    infer_scalar_descriptor: ScalarDescriptorInference
    evaluation_context: EvaluationContextFactory


class DictSupportLibrary:
    """Shared dict descriptor and binding helpers for the direct frontend."""

    def infer_literal_descriptor(self, context: DictSupportContext, node: ast.Dict) -> PythonDict:
        return infer_dict_literal_descriptor(node, context.infer_descriptor, context.infer_scalar_descriptor)

    def infer_literal_binding(self, context: DictSupportContext, node: ast.Dict) -> Optional[StaticDictBinding]:
        return infer_dict_literal_binding(node, context.infer_descriptor, context.infer_scalar_descriptor,
                                          context.evaluation_context)

    def infer_subscript_descriptor(self,
                                   context: DictSupportContext,
                                   descriptor: data.Data,
                                   slice_node: ast.AST,
                                   binding: Optional[StaticDictBinding] = None) -> Optional[data.Data]:
        return infer_dict_subscript_descriptor(descriptor, slice_node, context.evaluation_context, binding)

    def infer_assignment_binding(self, context: DictSupportContext, descriptor: data.Data,
                                 binding: Optional[StaticDictBinding], slice_node: ast.AST,
                                 value_node: ast.AST) -> Optional[tuple[PythonDict, Optional[StaticDictBinding]]]:
        return infer_dict_assignment_binding(descriptor, binding, slice_node, value_node, context.infer_descriptor,
                                             context.infer_scalar_descriptor, context.evaluation_context)

    def infer_assignment_descriptor(self, context: DictSupportContext, descriptor: data.Data, slice_node: ast.AST,
                                    value_node: ast.AST) -> Optional[PythonDict]:
        return infer_dict_assignment_descriptor(descriptor, slice_node, value_node, context.infer_descriptor,
                                                context.infer_scalar_descriptor, context.evaluation_context)


def infer_dict_literal_descriptor(node: ast.Dict, infer_descriptor: DescriptorInference,
                                  infer_scalar_descriptor: ScalarDescriptorInference) -> PythonDict:
    key_descriptors = []
    value_descriptors = []
    for key, value in zip(node.keys, node.values):
        if key is None:
            return PythonDict(transient=True)
        key_descriptors.append(infer_descriptor(key) or infer_scalar_descriptor(key, None))
        value_descriptors.append(infer_descriptor(value) or infer_scalar_descriptor(value, None))
    return PythonDict(merge_python_dict_component_descriptors(key_descriptors, transient=True),
                      merge_python_dict_component_descriptors(value_descriptors, transient=True),
                      transient=True)


def infer_dict_literal_binding(node: ast.Dict, infer_descriptor: DescriptorInference,
                               infer_scalar_descriptor: ScalarDescriptorInference,
                               evaluation_context: EvaluationContextFactory) -> Optional[StaticDictBinding]:
    entries: Dict[Any, data.Data] = {}
    for key, value in zip(node.keys, node.values):
        if key is None:
            return None
        key_value = try_resolve_static_value(key, evaluation_context())
        if key_value is UNRESOLVED:
            return None
        try:
            hash(key_value)
        except Exception:
            return None
        entries[key_value] = _infer_value_descriptor(value, infer_descriptor, infer_scalar_descriptor)
    return StaticDictBinding(entries=entries)


def infer_dict_subscript_descriptor(descriptor: data.Data,
                                    slice_node: ast.AST,
                                    evaluation_context: EvaluationContextFactory,
                                    binding: Optional[StaticDictBinding] = None) -> Optional[data.Data]:
    if not isinstance(descriptor, PythonDict):
        return None
    key_value = try_resolve_static_value(slice_node, evaluation_context())
    if key_value is UNRESOLVED:
        return None
    if binding is not None:
        entry = binding.entries.get(key_value)
        if entry is None:
            return None
        result = copy.deepcopy(entry)
        result.transient = True
        return result
    result = copy.deepcopy(descriptor.value_type)
    result.transient = True
    return result


def infer_dict_assignment_binding(
        descriptor: data.Data, binding: Optional[StaticDictBinding], slice_node: ast.AST, value_node: ast.AST,
        infer_descriptor: DescriptorInference, infer_scalar_descriptor: ScalarDescriptorInference,
        evaluation_context: EvaluationContextFactory) -> Optional[tuple[PythonDict, Optional[StaticDictBinding]]]:
    if not isinstance(descriptor, PythonDict):
        return None

    key_descriptor = _descriptor_from_key(slice_node, infer_descriptor, infer_scalar_descriptor, evaluation_context)
    if key_descriptor is None:
        return None

    value_descriptor = _infer_value_descriptor(value_node, infer_descriptor, infer_scalar_descriptor)
    updated_descriptor = PythonDict(merge_python_dict_component_descriptors((descriptor.key_type, key_descriptor),
                                                                            transient=True),
                                    merge_python_dict_component_descriptors((descriptor.value_type, value_descriptor),
                                                                            transient=True),
                                    transient=True)

    key_value = try_resolve_static_value(slice_node, evaluation_context())
    if key_value is UNRESOLVED or binding is None:
        return (updated_descriptor, None if key_value is UNRESOLVED else binding)

    if key_value not in binding.entries:
        return (updated_descriptor, None)

    updated_binding = copy.deepcopy(binding)
    updated_binding.entries[key_value] = copy.deepcopy(value_descriptor)
    updated_binding.entries[key_value].transient = True
    return (updated_descriptor, updated_binding)


def infer_dict_assignment_descriptor(descriptor: data.Data, slice_node: ast.AST, value_node: ast.AST,
                                     infer_descriptor: DescriptorInference,
                                     infer_scalar_descriptor: ScalarDescriptorInference,
                                     evaluation_context: EvaluationContextFactory) -> Optional[PythonDict]:
    updated = infer_dict_assignment_binding(descriptor, None, slice_node, value_node, infer_descriptor,
                                            infer_scalar_descriptor, evaluation_context)
    return None if updated is None else updated[0]


def _infer_value_descriptor(value_node: ast.AST, infer_descriptor: DescriptorInference,
                            infer_scalar_descriptor: ScalarDescriptorInference) -> data.Data:
    descriptor = infer_descriptor(value_node) or infer_scalar_descriptor(value_node, None)
    if descriptor is None:
        return data.Scalar(dtypes.pyobject(), transient=True)
    descriptor = copy.deepcopy(descriptor)
    descriptor.transient = True
    return descriptor


def _descriptor_from_key(slice_node: ast.AST, infer_descriptor: DescriptorInference,
                         infer_scalar_descriptor: ScalarDescriptorInference,
                         evaluation_context: EvaluationContextFactory) -> Optional[data.Data]:
    descriptor = _descriptor_from_static_key(slice_node, evaluation_context)
    if descriptor is not None:
        return descriptor
    descriptor = infer_descriptor(slice_node) or infer_scalar_descriptor(slice_node, None)
    if descriptor is None:
        return None
    descriptor = copy.deepcopy(descriptor)
    descriptor.transient = True
    return descriptor


def _descriptor_from_static_key(slice_node: ast.AST,
                                evaluation_context: EvaluationContextFactory) -> Optional[data.Data]:
    key_value = try_resolve_static_value(slice_node, evaluation_context())
    if key_value is UNRESOLVED:
        return None
    try:
        descriptor = copy.deepcopy(create_datadescriptor(key_value))
    except Exception:
        return None
    descriptor.transient = True
    return descriptor
