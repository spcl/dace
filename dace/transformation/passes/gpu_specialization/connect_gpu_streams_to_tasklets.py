# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import dace
from dace import dtypes, properties, SDFG
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_kernels import ConnectGPUStreamsToKernels

# Placeholder for the GPU stream variable used in tasklet code
STREAM_PLACEHOLDER = "__dace_current_stream"


@properties.make_properties
@transformation.explicit_cf_compatible
class ConnectGPUStreamsToTasklets(ppl.Pass):
    """
    This pass ensures that tasklets which require access to their assigned GPU stream
    are provided with it explicitly.

    Such tasklets typically originate from expanded LibraryNodes targeting GPUs.
    These nodes may reference the special placeholder variable `__dace_current_stream`,
    which is expected to be defined during unparsing in `cpp.py`.

    To avoid relying on this "hidden" mechanism, the pass rewrites tasklets to use
    the GPU stream AccessNode directly.

    Note that this pass is similar to `ConnectGPUStreamsToKernels`.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreams, ConnectGPUStreamsToKernels}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        # Retrieve the GPU stream's array name
        stream_array_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[0]

        # Retrieve GPU stream assignments for nodes
        stream_assignments: Dict[nodes.Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']

        # Find all tasklets which use the GPU stream variable (STREAM_PLACEHOLDER) in the code
        # and provide them the needed GPU stream explicitly
        for sub_sdfg in sdfg.all_sdfgs_recursive():

            for state in sub_sdfg.states():
                for node in state.nodes():

                    # Not a tasklet - continue
                    if not isinstance(node, nodes.Tasklet):
                        continue

                    # Tasklet does not need use its assigned GPU stream - continue
                    if not STREAM_PLACEHOLDER in node.code.as_string:
                        continue

                    # Stream connector name and the used GPU Stream for the kernel
                    assigned_gpustream = stream_assignments[node]
                    gpu_stream_conn = STREAM_PLACEHOLDER
                    accessed_gpu_stream = f"{stream_array_name}[{assigned_gpustream}]"

                    # Provide the GPU stream explicitly to the tasklet
                    stream_array_in = state.add_access(stream_array_name)
                    stream_array_out = state.add_access(stream_array_name)

                    node.add_in_connector(gpu_stream_conn, dtypes.gpuStream_t)
                    node.add_out_connector(gpu_stream_conn, dtypes.gpuStream_t, force=True)

                    state.add_edge(stream_array_in, None, node, gpu_stream_conn, dace.Memlet(accessed_gpu_stream))
                    state.add_edge(node, gpu_stream_conn, stream_array_out, None, dace.Memlet(accessed_gpu_stream))

        return {}
