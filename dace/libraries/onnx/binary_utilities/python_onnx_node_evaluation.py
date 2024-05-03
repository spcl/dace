"""
A faster way of evaluating ONNX nodes directly using the ORT C API directly from python.

This is mainly used for constant folding
"""
import copy
from typing import Dict

import torch
from dace import dtypes

from dace.libraries.onnx.onnx_importer import create_output_array
from dace.libraries.ort_api import ORTCAPIInterface, KernelSession, ExecutableKernelContext
from dace.util import out_desc_with_name


def evaluate_node(sdfg, state, node,
                  inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """ Evaluate the given node and return the outputs produced.

        :param sdfg: the sdfg of the node.
        :param state: the state of the node.
        :param node: the node to evaluate.
        :param inputs: mapping from connector name to input value.
        :return: a mapping from node output connector to the result tensor.
    """
    with ORTCAPIInterface() as api, \
            KernelSession(api) as session,\
            ExecutableKernelContext(api, session, node.name, node.schema.name) as context:
        for attribute, onnx_attribute in node.schema.attributes.items():
            if hasattr(node, attribute):
                context.add_attribute(attribute, getattr(node, attribute),
                                      onnx_attribute.attribute_type)

        for edge, is_input in node.iter_edges(state):
            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if is_input:
                context.add_input(edge_dtype)
            else:
                context.add_output(edge_dtype)

        with context.try_create_kernel(0) as kernel:
            for i, edge in enumerate(node.iter_inputs_in_onnx_order(state)):
                kernel.add_input(inputs[edge.dst_conn].cpu().numpy(), i)

            outputs = {}

            for i, edge in enumerate(node.iter_outputs_in_onnx_order(state)):
                desc = copy.deepcopy(
                    out_desc_with_name(node, state, sdfg, edge.src_conn))

                if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore,
                                     desc.storage):
                    pass
                elif dtypes.can_access(dtypes.ScheduleType.GPU_Default,
                                       desc.storage):
                    # outputs should be on CPU
                    desc.storage = dtypes.StorageType.CPU_Heap
                else:
                    raise ValueError(f"Unsupported storage {desc.storage}")

                output_arr = create_output_array({}, desc, use_torch=False)
                outputs[edge.src_conn] = output_arr
                kernel.add_output(output_arr, i)

            kernel.compute()

            for name, value in outputs.items():
                tensor = torch.from_numpy(value)
                outputs[name] = tensor

            return outputs
