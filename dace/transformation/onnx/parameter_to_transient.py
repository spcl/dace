import operator
import copy
import logging
from typing import List, Tuple, Any

import dace
from dace import dtypes, nodes

from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.torch import dlpack
from dace.frontend.python.module import DaceModule

log = logging.getLogger(__name__)


def parameter_to_transient(dace_module: DaceModule, parameter_path: str):
    """ Convert the dace array for pytorch parameter found at parameter_path to a persistently allocated transient.

        :param dace_module: the module containing the weight to transform.
        :param weight_path: the dotted path to the weight
    """

    log.debug(f"Converting parameter {parameter_path} to a transient")

    pt_weight_name = parameter_path
    pt_tensor = operator.attrgetter(pt_weight_name)(dace_module.model)
    array_name = clean_onnx_name(pt_weight_name)
    dace_module.dace_model.inputs.remove(parameter_path)

    # the the access node for this array of this array
    cands = [(node, parent) for (node, parent) in dace_module.sdfg.all_nodes_recursive()
             if isinstance(node, nodes.AccessNode) and node.data == array_name]

    if len(cands) == 0:
        log.warning(f"Could not find access node with name '{array_name}', skipping parameter to transient", )
        return

    if len(cands) != 1:
        raise ValueError("parameter_to_transient does not work when the target array has multiple AccessNodes")

    if array_name not in dace_module.sdfg.arrays:
        raise ValueError(f"Could not find parameter {array_name} in sdfg.")

    if dace_module.sdfg.arrays[array_name].storage is dtypes.StorageType.GPU_Global:
        dace_module.sdfg.arrays[array_name].transient = True
        dace_module.sdfg.arrays[array_name].lifetime = dtypes.AllocationLifetime.Persistent
        gpu_array_name = array_name
    else:

        # find the GPU transient of this array
        state: dace.SDFGState
        cand, state = cands[0]
        if state.out_degree(cand) != 1:
            raise ValueError(f"expected one out edge coming out of {cand}, found {state.out_degree(cand)}")
        _, _, dst_node, _, _ = state.out_edges(cand)[0]
        if (not isinstance(dst_node, nodes.AccessNode)
                or dace_module.sdfg.arrays[dst_node.data].storage is not dtypes.StorageType.GPU_Global):
            raise ValueError(f"parameter_to_transient only works for arrays that are copied to GPU_Global arrays,"
                             f" but array {array_name} was connected to {dst_node}")

        gpu_array_name = dst_node.data

        # since it is parsable, proceed with the transformation
        dace_module.sdfg.arrays[gpu_array_name].transient = True
        dace_module.sdfg.arrays[gpu_array_name].lifetime = dtypes.AllocationLifetime.Persistent

        # remove the CPU node
        state.remove_node(cand)
        del dace_module.sdfg[array_name]

    def post_compile_hook(compiled_sdfg):

        struct = compiled_sdfg.get_state_struct()

        param_sdfg = compiled_sdfg.sdfg
        struct_entry_name = f'__{param_sdfg.sdfg_id}_{gpu_array_name}'

        if not hasattr(struct, struct_entry_name):
            raise ValueError(f"Could not parse parameter {gpu_array_name} from state_struct.")

        ptr = getattr(struct, struct_entry_name)
        # copy the data into the torch parameter tensor
        torch_tensor = dlpack.array_to_torch_tensor(ptr, param_sdfg.arrays[gpu_array_name])
        torch_tensor[:] = pt_tensor

    dace_module.post_compile_hooks["init_" + pt_weight_name] = post_compile_hook
