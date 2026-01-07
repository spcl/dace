# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import dace
from dace import SDFG, dtypes, properties
from dace.config import Config
from dace.sdfg import is_devicelevel_gpu
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, Node, Tasklet
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name, get_gpu_stream_connector_name

STREAM_PLACEHOLDER = "__dace_current_stream"


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreams(ppl.Pass):
    """
    Inserts a GPU stream array into the top-level SDFG and propagates it to all
    nested SDFGs that require it, including intermediate SDFGs along the hierarchy.

    This pass guarantees that every relevant SDFG has the array defined, avoiding
    duplication and allowing subsequent passes in the GPU stream pipeline to rely
    on its presence without redefining it.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        """
        Ensure that a GPU stream array is available in all SDFGs that require it.

        The pass creates the array once at the top-level SDFG and propagates it
        down the hierarchy by inserting matching arrays in child SDFGs and wiring
        them through nested SDFG connectors. This way, all SDFGs share a consistent
        reference to the same GPU stream array.
        """

        # Extract stream array name and number of streams to allocate
        stream_array_name = get_gpu_stream_array_name()
        stream_assignments: Dict[Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']
        num_assigned_streams = max(stream_assignments.values(), default=0) + 1

        # Add the GPU stream array at the top level
        sdfg.add_transient(stream_array_name, (num_assigned_streams, ),
                           dtype=dace.dtypes.gpuStream_t,
                           storage=dace.dtypes.StorageType.Register)

        # Ensure GPU stream array is defined where required
        for child_sdfg in self.find_child_sdfgs_requiring_gpu_stream(sdfg):

            # Skip if this child already has the array (inserted higher up in the hierarchy)
            if stream_array_name in child_sdfg.arrays:
                continue

            # Add the array to the child SDFG
            inner_sdfg = child_sdfg
            inner_sdfg.add_array(stream_array_name, (num_assigned_streams, ),
                                 dtype=dace.dtypes.gpuStream_t,
                                 storage=dace.dtypes.StorageType.Register)

            # Walk up the hierarchy until the array is found, inserting it into each parent
            outer_sdfg = inner_sdfg.parent_sdfg
            while stream_array_name not in outer_sdfg.arrays:

                # Insert array in parent SDFG
                outer_sdfg.add_array(stream_array_name, (num_assigned_streams, ),
                                     dtype=dace.dtypes.gpuStream_t,
                                     storage=dace.dtypes.StorageType.Register)

                # Connect parent SDFG array to nested SDFG node
                inner_nsdfg_node = inner_sdfg.parent_nsdfg_node
                inner_parent_state = inner_sdfg.parent
                inner_nsdfg_node.add_in_connector(stream_array_name, dtypes.gpuStream_t)
                inp_gpu_stream: AccessNode = inner_parent_state.add_access(stream_array_name)
                inner_parent_state.add_edge(inp_gpu_stream, None, inner_nsdfg_node, stream_array_name,
                                            dace.Memlet(stream_array_name))

                # Continue climbing up the hierarchy
                inner_sdfg = outer_sdfg
                outer_sdfg = outer_sdfg.parent_sdfg

            # Ensure final connection from the first parent that had the array down to this SDFG
            inner_nsdfg_node = inner_sdfg.parent_nsdfg_node
            inner_parent_state = inner_sdfg.parent
            inner_nsdfg_node.add_in_connector(stream_array_name, dtypes.gpuStream_t)
            inp_gpu_stream: AccessNode = inner_parent_state.add_access(stream_array_name)
            inner_parent_state.add_edge(inp_gpu_stream, None, inner_nsdfg_node, stream_array_name,
                                        dace.Memlet(f"{stream_array_name}[0:{num_assigned_streams}]"))

            outer_sdfg = inner_sdfg.parent_sdfg

        return {}

    def find_child_sdfgs_requiring_gpu_stream(self, sdfg) -> Set[SDFG]:
        """
        Identify all child SDFGs that require a GPU stream array in their
        array descriptor store. A child SDFG requires a GPU stream if:

        - It launches GPU kernels (MapEntry/MapExit with GPU_Device schedule).
        - It contains special Tasklets (e.g., from library node expansion) that
          use the GPU stream they are assigned to in the code.
        - It accesses GPU global memory outside device-level GPU scopes, which
          implies memory copies or kernel data feeds.

        Parameters
        ----------
        sdfg : SDFG
            The root SDFG to inspect.

        Returns
        -------
        Set[SDFG]
            The set of child SDFGs that need a GPU stream array in their array descriptor
            store.
        """
        requiring_gpu_stream = set()
        for child_sdfg in sdfg.all_sdfgs_recursive():

            # Skip the root SDFG itself
            if child_sdfg is sdfg:
                continue

            for state in child_sdfg.states():
                for node in state.nodes():

                    # Case 1: Kernel launch nodes
                    if isinstance(node, (MapEntry, MapExit)) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                        requiring_gpu_stream.add(child_sdfg)
                        break

                    # Case 2: Tasklets that use GPU stream in their code
                    if isinstance(node, Tasklet) and STREAM_PLACEHOLDER in node.code.as_string:
                        requiring_gpu_stream.add(child_sdfg)
                        break

                    # Case 3: Accessing GPU global memory outside device-level scopes
                    if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Global
                            and not is_devicelevel_gpu(state.sdfg, state, node)):
                        requiring_gpu_stream.add(child_sdfg)
                        break

                # Stop scanning this SDFG once a reason is found
                if child_sdfg in requiring_gpu_stream:
                    break

        return requiring_gpu_stream
