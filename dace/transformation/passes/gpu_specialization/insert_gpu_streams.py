# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Type, Union

import dace
from dace import SDFG, dtypes, properties
from dace.sdfg import is_devicelevel_gpu
from dace.sdfg.nodes import AccessNode, MapExit, Node
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (add_gpu_stream_connector,
                                                                               get_gpu_stream_array_name,
                                                                               is_gpu_relevant_node)


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreams(ppl.Pass):
    """Insert the GPU stream array into the top-level SDFG and propagate it to nested SDFGs that need it.

    Every intermediate SDFG along the path to a user gets the array and a matching nested-SDFG
    in-connector, so subsequent passes can rely on its presence everywhere it's needed.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        """Create the GPU stream array at the top level and wire it through every nested SDFG that needs it.

        All relevant SDFGs end up sharing a consistent reference to the same stream array, so
        subsequent pipeline passes can rely on its presence without redefining it.
        """

        # Extract stream array name and number of streams to allocate
        stream_array_name = get_gpu_stream_array_name()
        stream_assignments: Dict[Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']
        num_assigned_streams = max(stream_assignments.values(), default=0) + 1

        # Add the GPU stream array at the top level. The pass may run a second
        # time after `expand_library_nodes` has surfaced new GPU library nodes
        # in nested SDFGs; in that case the root array already exists and we
        # only need to propagate it deeper.
        if stream_array_name not in sdfg.arrays:
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
                add_gpu_stream_connector(inner_nsdfg_node, stream_array_name, single_stream=False)
                inp_gpu_stream: AccessNode = inner_parent_state.add_access(stream_array_name)
                inner_parent_state.add_edge(inp_gpu_stream, None, inner_nsdfg_node, stream_array_name,
                                            dace.Memlet(stream_array_name))

                # Continue climbing up the hierarchy
                inner_sdfg = outer_sdfg
                outer_sdfg = outer_sdfg.parent_sdfg

            # Ensure final connection from the first parent that had the array down to this SDFG
            inner_nsdfg_node = inner_sdfg.parent_nsdfg_node
            inner_parent_state = inner_sdfg.parent
            add_gpu_stream_connector(inner_nsdfg_node, stream_array_name, single_stream=False)
            inp_gpu_stream: AccessNode = inner_parent_state.add_access(stream_array_name)
            inner_parent_state.add_edge(inp_gpu_stream, None, inner_nsdfg_node, stream_array_name,
                                        dace.Memlet(f"{stream_array_name}[0:{num_assigned_streams}]"))

            outer_sdfg = inner_sdfg.parent_sdfg

        return {}

    def find_child_sdfgs_requiring_gpu_stream(self, sdfg: SDFG) -> Set[SDFG]:
        """Identify all child SDFGs that need a GPU stream array in their array descriptor store.

        A child SDFG requires a GPU stream if it launches GPU kernels, contains a
        ``CopyLibraryNode`` / ``MemsetLibraryNode`` targeting GPU storage (these lower to
        stream-bound memcpy/kernel launches), or accesses GPU global memory outside a
        device-level scope.
        """
        requiring_gpu_stream = set()
        for child_sdfg in sdfg.all_sdfgs_recursive():

            # Skip the root SDFG itself
            if child_sdfg is sdfg:
                continue

            for state in child_sdfg.states():
                for node in state.nodes():
                    # MapExit also counts (it's the symmetric end of the
                    # kernel launch); ``is_gpu_relevant_node`` only matches
                    # MapEntry, so check it here.
                    if isinstance(node, MapExit) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                        requiring_gpu_stream.add(child_sdfg)
                        break
                    # Skip GPU AccessNodes that are inside a device scope
                    # (codegen handles them through register/local paths,
                    # no host-side stream needed).
                    if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Global
                            and is_devicelevel_gpu(state.sdfg, state, node)):
                        continue
                    if is_gpu_relevant_node(node, child_sdfg, state):
                        requiring_gpu_stream.add(child_sdfg)
                        break

                # Stop scanning this SDFG once a reason is found
                if child_sdfg in requiring_gpu_stream:
                    break

        return requiring_gpu_stream
