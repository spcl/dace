# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import dace
from dace import dtypes, properties, SDFG
from dace.codegen import common
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpustream.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpustream.insert_gpu_streams_to_sdfgs import InsertGPUStreamsToSDFGs


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreamsToKernels(ppl.Pass):
    """
    This Pass attaches GPU streams to kernels (i.e., dtypes.ScheduleType.GPU_Device scheduled maps).

    Adds GPU stream AccessNodes and connects them to kernel entry and exit nodes,
    indicating which GPU stream each kernel is assigned to. These assignments are e.g.
    used when launching the kernels.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreamsToSDFGs}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        # Retrieve the GPU stream array name and the prefix for individual stream variables
        stream_array_name, stream_var_name_prefix = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')

        # Retrieve GPU stream assignments for nodes
        stream_assignments: Dict[nodes.Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']

        # Link kernels to their assigned GPU streams
        for sub_sdfg in sdfg.all_sdfgs_recursive():

            for state in sub_sdfg.states():
                for node in state.nodes():

                    # Not a kernel entry - continue
                    if not (isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device):
                        continue

                    # Stream connector name and the used GPU Stream for the kernel
                    assigned_gpustream = stream_assignments[node]
                    gpu_stream_var_name = f"{stream_var_name_prefix}{assigned_gpustream}"
                    accessed_gpu_stream = f"{stream_array_name}[{assigned_gpustream}]"

                    # Assign the GPU stream to the kernel entry
                    kernel_entry = node
                    kernel_entry.add_in_connector(gpu_stream_var_name, dtypes.gpuStream_t)
                    stream_array_in = state.add_access(stream_array_name)
                    state.add_edge(stream_array_in, None, kernel_entry, gpu_stream_var_name,
                                   dace.Memlet(accessed_gpu_stream))

                    # Assign the GPU stream to the kernel exit
                    kernel_exit = state.exit_node(kernel_entry)
                    kernel_exit.add_out_connector(gpu_stream_var_name, dtypes.gpuStream_t)
                    stream_array_out = state.add_access(stream_array_name)
                    state.add_edge(kernel_exit, gpu_stream_var_name, stream_array_out, None,
                                   dace.Memlet(accessed_gpu_stream))

        return {}
