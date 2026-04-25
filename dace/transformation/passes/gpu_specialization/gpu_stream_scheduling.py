# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, List, Optional, Set, Type, Union

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


def _is_gpu_copy_or_memset(node) -> bool:
    """``CopyLibraryNode`` / ``MemsetLibraryNode`` whose storage involves GPU
    global or pinned host memory -- i.e. the nodes the GPU stream pipeline
    needs to wire a stream handle to.
    """
    if isinstance(node, CopyLibraryNode):
        return (node.src_storage in _GPU_SIDE_STORAGES or node.dst_storage in _GPU_SIDE_STORAGES)
    if isinstance(node, MemsetLibraryNode):
        # Memset has a single output; inspect its descriptor via the parent
        # state at detection time (see ``_requires_gpu_stream``).
        return True
    return False


class GPUStreamSchedulingStrategy(ppl.Pass):
    """ Base class for stream-assignment strategies plugged into the GPU stream
        pipeline.

        Subclasses override ``assign(sdfg) -> Dict[nodes.Node, int]`` mapping
        each node to a backend stream id. The base class itself raises on
        ``assign`` -- it is a contract, not a default. The default fallback
        when nobody registers a custom strategy is :class:`NaiveGPUStreamScheduler`
        (see :func:`get_gpu_stream_scheduler`).
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, int]:
        return self.assign(sdfg)

    def assign(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        raise NotImplementedError(
            f"{type(self).__name__} did not implement assign(sdfg). "
            "Subclass GPUStreamSchedulingStrategy and override assign, or "
            "use NaiveGPUStreamScheduler -- the default for the GPU stream pipeline.")


# Process-wide override of the default scheduler. ``None`` means
# ``NaiveGPUStreamScheduler`` is used. Set via ``register_gpu_stream_scheduler``.
_REGISTERED_SCHEDULER: 'Optional[Type[GPUStreamSchedulingStrategy]]' = None


def register_gpu_stream_scheduler(strategy_cls: 'Type[GPUStreamSchedulingStrategy]') -> None:
    """ Register ``strategy_cls`` as the default stream-assignment strategy.

        Subsequent calls to :func:`get_gpu_stream_scheduler` (and any pipeline
        that uses it) will instantiate this class instead of
        :class:`NaiveGPUStreamScheduler`. Pass ``None`` to clear the
        registration and fall back to the naive default.

        :param strategy_cls: a subclass of :class:`GPUStreamSchedulingStrategy`.
    """
    global _REGISTERED_SCHEDULER
    if strategy_cls is not None and not issubclass(strategy_cls, GPUStreamSchedulingStrategy):
        raise TypeError(
            f"{strategy_cls.__name__} must subclass GPUStreamSchedulingStrategy "
            "(it doesn't, so the pipeline cannot use it as a stream scheduler).")
    _REGISTERED_SCHEDULER = strategy_cls


def get_gpu_stream_scheduler() -> 'GPUStreamSchedulingStrategy':
    """ Returns a fresh instance of the registered stream-scheduling strategy,
        or :class:`NaiveGPUStreamScheduler` if no strategy was registered.
    """
    cls = _REGISTERED_SCHEDULER if _REGISTERED_SCHEDULER is not None else NaiveGPUStreamScheduler
    return cls()


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(GPUStreamSchedulingStrategy):
    """
    Assigns GPU streams to nodes and stores the assignments in a dictionary.
    This can be useful for enabling asynchronous and parallel GPU computation using GPU streams.

    Strategy Overview:
    ------------------
    - GPU stream assignment is based on weakly connected components (WCCs) within each state.
    - Nodes in the same WCC are assigned to the same stream.
    - For top-level states (not within nested SDFGs), each new WCC starts on a new stream (starting from 0).
    - In nested SDFGs:
        * Stream assignment is inherited from the parent component,
        * All internal components share the parent's stream.
    - GPU stream IDs wrap around according to the `max_concurrent_streams` configuration.

    Example:
    --------
    A state with the following independent chains:
        K1 → K2
        K3 → K4 → K5
        K6

    would be scheduled as:
        K1, K2     → stream 0
        K3, K4, K5 → stream 1
        K6         → stream 2

    (assuming no limit on the number of concurrent streams)

    Note:
    -----
    These refer to **backend GPU streams** (e.g., CUDA or HIP), not DaCe symbolic streams.
    """

    def __init__(self):
        # Maximum number of concurrent streams allowed (from config).
        # Cached locally for frequent reuse.
        self._max_concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

    def assign(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        """ Returns a ``{node: stream_id}`` mapping per the WCC strategy in the
            class docstring. """
        stream_assignments: Dict[nodes.Node, int] = dict()
        for state in sdfg.states():
            self._assign_gpu_streams_in_state(sdfg, False, state, stream_assignments, 0)
        return stream_assignments

    def _assign_gpu_streams_in_state(self, sdfg: SDFG, in_nested_sdfg: bool, state: SDFGState,
                                     stream_assignments: Dict[nodes.Node, int], gpu_stream: int) -> None:
        """
        Assigns GPU streams to nodes in a single state.

        If inside a nested SDFG, components inherit the parent's stream.
        Otherwise, each connected component gets a different stream.
        Nested SDFGs are processed recursively.

        Parameters
        ----------
        sdfg : SDFG
            The SDFG containing the state.
        in_nested_sdfg : bool
            True if the state is in a nested SDFG.
        state : SDFGState
            The state to process.
        stream_assignments : Dict[nodes.Node, int]
            Mapping of nodes to assigned GPU streams (updated in-place).
        gpu_stream : int
            The current GPU stream ID.

        Returns
        -------
        None
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
        """
        Returns all weakly connected components in the given directed graph.

        A weakly connected component is a maximal group of nodes such that each pair
        of nodes is connected by a path when ignoring edge directions.

        Parameters
        ----------
        graph: Graph
            A directed graph instance.

        Returns
        -------
        List[Set[Node_T]]

            A list containing sets of nodes, with each set corresponding to a weakly
            connected component.
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
        """
        Compute the next CUDA stream index according to the concurrency configuration.

        Behavior depends on the configured max_concurrent_streams value:
        - If 0: unlimited streams allowed, so increment the stream index by one.
        - If -1: default setting, always return stream 0 (no concurrency).
        - Otherwise: cycle through stream indices from 0 up to max_concurrent_streams - 1.

        Parameters
        ----------
        gpu_stream : int
            The current CUDA stream index.

        Returns
        -------
        int
            The next CUDA stream index based on the concurrency policy.
        """
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        elif self._max_concurrent_streams == -1:
            return 0
        else:
            return (gpu_stream + 1) % self._max_concurrent_streams

    def _requires_gpu_stream(self, state: SDFGState, component: Set[NodeT]) -> bool:
        """
        Check whether a connected component in an SDFG state should be assigned
        a GPU stream.

        A component requires a GPU stream if it contains at least one of:
        - An AccessNode with GPU global memory storage,
        - A MapEntry scheduled on a GPU device,
        - A ``CopyLibraryNode`` or ``MemsetLibraryNode`` touching GPU
          storage (these lower to stream-bound memcpy/kernel launches).

        Parameters
        ----------
        state : SDFGState
            The state containing the component.
        component : Set[NodeT]
            The set of nodes that form the connected component.

        Returns
        -------
        bool
            True if the component requires a GPU stream, False otherwise.
        """

        def gpu_relevant(node, parent) -> bool:
            if (isinstance(node, nodes.AccessNode) and node.desc(parent).storage == dace.dtypes.StorageType.GPU_Global):
                return True

            elif (isinstance(node, nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device):
                return True

            elif _is_gpu_copy_or_memset(node):
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
