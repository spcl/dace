# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import copy
from typing import Any, Callable, Dict, Optional

from dace import data
from dace.data.pydata import PythonDict, merge_python_dict_component_descriptors
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

DescriptorInference = Callable[[ast.AST], Optional[data.Data]]
ScalarDescriptorInference = Callable[[ast.AST, Optional[data.Data]], Optional[data.Data]]
EvaluationContextFactory = Callable[[], Dict[str, Any]]


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


def infer_dict_subscript_descriptor(descriptor: data.Data, slice_node: ast.AST,
                                    evaluation_context: EvaluationContextFactory) -> Optional[data.Data]:
    if not isinstance(descriptor, PythonDict):
        return None
    key_value = try_resolve_static_value(slice_node, evaluation_context())
    if key_value is UNRESOLVED:
        return None
    result = copy.deepcopy(descriptor.value_type)
    result.transient = True
    return result
