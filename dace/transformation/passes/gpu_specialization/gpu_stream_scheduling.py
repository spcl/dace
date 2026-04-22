# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, List, Set, Type, Union

import dace
from dace import SDFG, SDFGState, properties
from dace.config import Config
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.sdfg import nodes
from dace.sdfg.graph import Graph, NodeT
from dace.transformation import pass_pipeline as ppl, transformation

# Storages whose copies are serviced by the GPU stream pipeline.
_GPU_SIDE_STORAGES = frozenset({
    dace.dtypes.StorageType.GPU_Global,
    dace.dtypes.StorageType.GPU_Shared,
    dace.dtypes.StorageType.CPU_Pinned,
})


def _is_gpu_copy_or_memset(node, state: SDFGState, sdfg: SDFG) -> bool:
    """``CopyLibraryNode`` / ``MemsetLibraryNode`` whose storage involves GPU
    global or pinned host memory -- i.e. the nodes the GPU stream pipeline
    needs to wire a stream handle to.
    """
    if isinstance(node, CopyLibraryNode):
        return (node.src_storage(state, sdfg) in _GPU_SIDE_STORAGES
                or node.dst_storage(state, sdfg) in _GPU_SIDE_STORAGES)
    if isinstance(node, MemsetLibraryNode):
        # Memset has a single output; inspect its descriptor via the parent
        # state at detection time (see ``_requires_gpu_stream``).
        return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(ppl.Pass):
    """Assign backend GPU streams (CUDA/HIP) to nodes via weakly-connected-component grouping.

    Strategy:

    - Nodes in the same weakly connected component share one stream.
    - Top-level states start each new component on a fresh stream (wrapping
      according to ``compiler.cuda.max_concurrent_streams``).
    - In nested SDFGs, all internal components inherit the parent component's stream.

    The streams here are backend GPU streams (CUDA/HIP), not DaCe symbolic streams.
    """

    def __init__(self):
        # Maximum number of concurrent streams allowed (from config).
        # Cached locally for frequent reuse.
        self._max_concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, int]:
        """Return a dict mapping each node to its assigned GPU stream."""
        stream_assignments: Dict[nodes.Node, int] = dict()
        for state in sdfg.states():
            self._assign_gpu_streams_in_state(sdfg, False, state, stream_assignments, 0)

        return stream_assignments

    def _assign_gpu_streams_in_state(self, sdfg: SDFG, in_nested_sdfg: bool, state: SDFGState,
                                     stream_assignments: Dict[nodes.Node, int], gpu_stream: int):
        """Assign GPU streams to nodes in a single state; updates ``stream_assignments`` in place.

        If inside a nested SDFG, components inherit the parent's stream. Otherwise each connected
        component gets a different stream. Nested SDFGs are processed recursively.

        :param sdfg: the SDFG containing the state.
        :param in_nested_sdfg: True when the state lives inside a nested SDFG.
        :param state: the state to process.
        :param stream_assignments: mapping updated in place.
        :param gpu_stream: current GPU stream ID.
        """
        components = self._get_weakly_connected_nodes(state)

        for component in components:

            if not self._requires_gpu_stream(state, component):
                continue

            nodes_assigned_before = len(stream_assignments)

            for node in component:
                stream_assignments[node] = gpu_stream
                if isinstance(node, nodes.NestedSDFG):
                    for nested_state in node.sdfg.states():
                        self._assign_gpu_streams_in_state(node.sdfg, True, nested_state, stream_assignments, gpu_stream)

            # Move to the next stream if we have assigned streams to any node in this component
            # (careful: if nested, states are in same component)
            if not in_nested_sdfg and len(stream_assignments) > nodes_assigned_before:
                gpu_stream = self._next_stream(gpu_stream)

    def _get_weakly_connected_nodes(self, graph: Graph) -> List[Set[NodeT]]:
        """Return the weakly connected components of ``graph`` (edge directions ignored).

        A weakly connected component is a maximal group of nodes such that each pair
        of nodes is connected by a path when ignoring edge directions.
        """
        visited: Set[NodeT] = set()
        components: List[Set[NodeT]] = []

        for node in graph.nodes():
            if node in visited:
                continue

            # Start a new weakly connected component
            component: Set[NodeT] = set()
            stack = [node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        return components

    def _next_stream(self, gpu_stream: int) -> int:
        """Return the next stream index per the concurrency configuration.

        With ``max_concurrent_streams`` == ``0`` streams are unlimited (increment);
        ``-1`` always returns stream 0 (no concurrency); otherwise cycle through
        ``[0, max_concurrent_streams)``.
        """
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        elif self._max_concurrent_streams == -1:
            return 0
        else:
            return (gpu_stream + 1) % self._max_concurrent_streams

    def _requires_gpu_stream(self, state: SDFGState, component: Set[NodeT]) -> bool:
        """Return True when ``component`` needs a GPU stream.

        A component requires a stream if it contains any of:

        - an ``AccessNode`` with ``GPU_Global`` storage,
        - a ``MapEntry`` scheduled on ``GPU_Device``,
        - a ``CopyLibraryNode`` / ``MemsetLibraryNode`` touching GPU storage
          (these lower to stream-bound memcpy / kernel launches).
        """

        def gpu_relevant(node, parent) -> bool:
            if (isinstance(node, nodes.AccessNode) and node.desc(parent).storage == dace.dtypes.StorageType.GPU_Global):
                return True

            elif (isinstance(node, nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device):
                return True

            elif _is_gpu_copy_or_memset(node, parent, parent.sdfg):
                return True

            return False

        for node in component:
            if isinstance(node, nodes.NestedSDFG):
                if any(gpu_relevant(node, parent) for node, parent in node.sdfg.all_nodes_recursive()):
                    return True

            else:
                if gpu_relevant(node, state):
                    return True

        return False
