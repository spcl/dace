# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, List, Set, Tuple, Type, Union
import copy

import dace
from dace import dtypes, properties, SDFG, SDFGState
from dace.codegen import common
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import is_within_schedule_types
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name, get_gpu_stream_connector_name
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_kernels import ConnectGPUStreamsToKernels
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_tasklets import ConnectGPUStreamsToTasklets

STREAM_PLACEHOLDER = "__dace_current_stream"


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreamSyncTasklets(ppl.Pass):
    """
    Inserts GPU stream synchronization tasklets in an SDFG where needed.

    This pass uses a heuristic approach to find locations matching specific patterns
    that require synchronization. Additional locations can be added easily if new
    cases are discovered.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreams, ConnectGPUStreamsToKernels, ConnectGPUStreamsToTasklets}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        """
        Inserts GPU stream synchronization tasklets at required locations
        after certain nodes and at the end of a state, for GPU streams used in the state.
        """
        stream_assignments: Dict[nodes.Node, int] = pipeline_results['NaiveGPUStreamScheduler']

        # Get sync locations
        sync_state, sync_node = self._identify_sync_locations(sdfg, stream_assignments)

        # Synchronize after a node when required
        self._insert_gpu_stream_sync_after_node(sdfg, sync_node, stream_assignments)

        # Synchronize all used streams at the end of a state
        self._insert_gpu_stream_sync_at_state_end(sdfg, sync_state, stream_assignments)
        return {}

    def _identify_sync_locations(
            self, sdfg: SDFG,
            stream_assignments: Dict[nodes.Node, int]) -> Tuple[Dict[SDFGState, Set[int]], Dict[nodes.Node, SDFGState]]:
        """
        Heuristically identifies GPU stream synchronization points in an SDFG.

        Parameters
        ----------
        sdfg : SDFG
            The SDFG to analyze.
        stream_assignments : Dict[nodes.Node, int]
            Mapping of nodes to their assigned GPU stream ids.

        Returns
        -------
        Tuple[Dict[SDFGState, Set[int]], Dict[nodes.Node, SDFGState]]
            - **sync_state**: Maps each state to the set of stream IDs that should be
                              synchronized at the end of the state.
            - **sync_node**: The keys of this dictionary are nodes after which synchronization
                             is needed, and their corresponding value is the state they belong to.
        """

        # ------------------ Helper predicates -----------------------------

        def is_gpu_global_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(
                state.parent).storage == dtypes.StorageType.GPU_Global

        def is_nongpu_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(
                state.parent).storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES

        def is_kernel_exit(node):
            return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device

        def is_sink_node(node, state):
            return state.out_degree(node) == 0

        def edge_within_kernel(state, src, dst):
            gpu_schedules = dtypes.GPU_SCHEDULES + dtypes.EXPERIMENTAL_GPU_SCHEDULES
            src_in_kernel = is_within_schedule_types(state, src, gpu_schedules)
            dst_in_kernel = is_within_schedule_types(state, dst, gpu_schedules)
            return src_in_kernel and dst_in_kernel

        def is_tasklet_with_stream_use(src):
            return isinstance(src, nodes.Tasklet) and STREAM_PLACEHOLDER in src.code.as_string

        # ------------------ Sync detection logic -----------------------------

        sync_state: Dict[SDFGState, Set[int]] = {}
        sync_node: Dict[nodes.Node, SDFGState] = {}

        for edge, state in sdfg.all_edges_recursive():
            src, dst = edge.src, edge.dst

            # Ensure state is initialized in sync_state
            if state not in sync_state:
                sync_state[state] = set()

            # --- Heuristics for when to sync ---
            if (is_gpu_global_accessnode(src, state) and is_nongpu_accessnode(dst, state) and is_sink_node(dst, state)
                    and not edge_within_kernel(state, src, dst)):
                sync_state[state].add(stream_assignments[dst])

            elif (is_gpu_global_accessnode(src, state) and is_nongpu_accessnode(dst, state)
                  and not is_sink_node(dst, state) and not edge_within_kernel(state, src, dst)):
                sync_node[dst] = state
                sync_state[state].add(stream_assignments[dst])

            elif (is_nongpu_accessnode(src, state) and is_gpu_global_accessnode(dst, state)
                  and not edge_within_kernel(state, src, dst)):
                sync_state[state].add(stream_assignments[dst])

            elif (is_kernel_exit(src) and is_gpu_global_accessnode(dst, state) and not is_sink_node(dst, state)):
                sync_state[state].add(stream_assignments[src])
                sync_state[state].add(stream_assignments[src])

            elif (is_kernel_exit(src) and is_gpu_global_accessnode(dst, state) and is_sink_node(dst, state)):
                sync_state[state].add(stream_assignments[dst])

            elif is_tasklet_with_stream_use(src):
                sync_state[state].add(stream_assignments[src])

            else:
                continue

            # Check that state is indeed a SDFGState when added to the dictionary, to be on the safe side
            if not isinstance(state, SDFGState):
                raise NotImplementedError(f"Unexpected parent type '{type(state).__name__}' for edge '{edge}'. "
                                          "Expected 'SDFGState'. Please handle this case explicitly.")

        # Remove states with no syncs
        sync_state = {state: streams for state, streams in sync_state.items() if len(streams) > 0}

        return sync_state, sync_node

    def _insert_gpu_stream_sync_at_state_end(self, sdfg: SDFG, sync_state: Dict[SDFGState, Set[int]],
                                             stream_assignments: Dict[nodes.Node, int]) -> None:
        """
        Inserts GPU stream synchronization tasklets at the end of SDFG states.

        For each state that requires synchronization, this method:

        1. Generates a tasklet that synchronizes all assigned GPU streams using
           the appropriate backend (e.g., CUDA).
        2. Ensures all other operations in the state complete before synchronization
           by connecting all sink nodes to the tasklet.
        3. Guarantees that only a single GPU stream AccessNode connects to the sync
           tasklet, creating one if needed.

        Parameters
        ----------
        sdfg : SDFG
            The top level SDFG.
        sync_state : Dict[SDFGState, Set[int]
            Mapping of states to sets of stream IDs that require synchronization at the end of the state.
        stream_assignments : Dict[nodes.Node, int]
            Mapping of nodes to their assigned GPU stream IDs.
        """
        # Prepare GPU stream info and backend
        stream_array_name = get_gpu_stream_array_name()
        stream_var_name_prefix = get_gpu_stream_connector_name()
        backend: str = common.get_gpu_backend()

        for state, streams in sync_state.items():

            #----------------- Generate GPU stream synchronization Tasklet -----------------

            # Build synchronization calls for all streams used in this state
            sync_code_lines = []
            for stream in streams:
                gpu_stream_var_name = f"{stream_var_name_prefix}{stream}"
                sync_call = f"DACE_GPU_CHECK({backend}StreamSynchronize({gpu_stream_var_name}));"
                sync_code_lines.append(sync_call)
            sync_code = "\n".join(sync_code_lines)

            # Create the tasklet
            tasklet = state.add_tasklet(name=f"gpu_stream_{stream}_synchronization",
                                        inputs=set(),
                                        outputs=set(),
                                        code=sync_code,
                                        language=dtypes.Language.CPP)

            # ----------------- Connect sink nodes to the synchronization tasklet -----------------

            # 1. Seperate GPU stream sink nodes and other sink nodes
            stream_sink_nodes: List[nodes.AccessNode] = []
            non_stream_sink_nodes: List[nodes.Node] = []
            for sink_node in state.sink_nodes():
                if isinstance(sink_node, nodes.AccessNode) and sink_node.desc(state).dtype == dtypes.gpuStream_t:
                    stream_sink_nodes.append(sink_node)

                elif sink_node != tasklet:
                    non_stream_sink_nodes.append(sink_node)

            # 2. Connect non-stream sink nodes to the sync tasklet
            for sink_node in non_stream_sink_nodes:
                state.add_edge(sink_node, None, tasklet, None, dace.Memlet())

            # 3. Connect a single GPU stream sink node (create or merge if needed)
            if len(stream_sink_nodes) == 0:
                combined_stream_node = state.add_access(stream_array_name)

            else:
                combined_stream_node = stream_sink_nodes.pop()
                for stream_node in stream_sink_nodes:
                    for edge in state.in_edges(stream_node):
                        state.add_edge(edge.src, edge.src_conn, combined_stream_node, edge.dst_conn, edge.data)
                        state.remove_edge(edge)
                    state.remove_node(stream_node)

            # Connect back to output stream node
            output_stream_node = state.add_access(combined_stream_node.data)
            for stream in streams:
                accessed_gpu_stream = f"{stream_array_name}[{stream}]"
                conn = f"{stream_var_name_prefix}{stream}"  # Note: Same as "gpu_stream_var_name" from tasklet

                tasklet.add_in_connector(conn, dtypes.gpuStream_t)
                state.add_edge(combined_stream_node, None, tasklet, conn, dace.Memlet(accessed_gpu_stream))
                state.add_edge(tasklet, None, output_stream_node, None, dace.Memlet(None))

    def _insert_gpu_stream_sync_after_node(self, sdfg: SDFG, sync_node: Dict[nodes.Node, SDFGState],
                                           stream_assignments: Dict[nodes.Node, int]) -> None:
        """
        Insert a GPU stream synchronization tasklet immediately after specified nodes.

        Parameters
        ----------
        sdfg : SDFG
            The top level SDFG.
        sync_node : Dict[nodes.Node, SDFGState]
            Mapping of nodes to their parent state. After after the node a GPU stream synchronization should occur.
        stream_assignments : Dict[nodes.Node, int]
            Mapping of nodes to their assigned GPU stream IDs.
        """
        # Prepare GPU stream info and backend
        stream_array_name = get_gpu_stream_array_name()
        stream_var_name_prefix = get_gpu_stream_connector_name()
        backend: str = common.get_gpu_backend()

        for node, state in sync_node.items():

            #----------------- Generate GPU stream synchronization Tasklet -----------------

            # Get assigned GPU stream
            stream = stream_assignments.get(node, "nullptr")
            if stream == "nullptr":
                raise NotImplementedError("Using the default 'nullptr' gpu stream is not supported yet.")

            # Create the tasklet
            stream_var_name = f"{stream_var_name_prefix}{stream}"
            sync_call = f"DACE_GPU_CHECK({backend}StreamSynchronize({stream_var_name}));\n"
            tasklet = state.add_tasklet(name=f"gpu_stream_{stream}_synchronization",
                                        inputs=set(),
                                        outputs=set(),
                                        code=sync_call,
                                        language=dtypes.Language.CPP)

            #----------------- Place tasklet between node and successors, link GPU streams ----------------

            # 1. Put the tasklet between the node and its successors
            for succ in state.successors(node):
                state.add_edge(tasklet, None, succ, None, dace.Memlet())
            state.add_edge(node, None, tasklet, None, dace.Memlet())

            # 2. Connect tasklet to GPU stream AccessNodes
            in_stream = state.add_access(stream_array_name)
            out_stream = state.add_access(stream_array_name)
            accessed_stream = f"{stream_array_name}[{stream}]"
            state.add_edge(in_stream, None, tasklet, stream_var_name, dace.Memlet(accessed_stream))
            state.add_edge(tasklet, None, out_stream, None, dace.Memlet(None))
            tasklet.add_in_connector(stream_var_name, dtypes.gpuStream_t, force=True)
