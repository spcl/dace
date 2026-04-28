# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, List, Optional, Set, Type, Union

from dace import SDFG, SDFGState, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.graph import Graph, NodeT
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_relevant_node


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
        # Stream scheduling assumes implicit AccessNode->AccessNode GPU copies
        # have already been lifted to ``CopyLibraryNode``s — those are the
        # nodes the scheduler attaches stream ids to. Without the lift, GPU
        # transfers are invisible to the scheduler and downstream wiring
        # silently misses them.
        from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
            InsertExplicitGPUGlobalMemoryCopies)
        return {InsertExplicitGPUGlobalMemoryCopies}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, int]:
        return self.assign(sdfg)

    def assign(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        raise NotImplementedError(f"{type(self).__name__} did not implement assign(sdfg). "
                                  "Subclass GPUStreamSchedulingStrategy and override assign, or "
                                  "use NaiveGPUStreamScheduler -- the default for the GPU stream pipeline.")


# Process-wide override of the default scheduler. ``None`` means
# ``NaiveGPUStreamScheduler`` is used. Set via ``register_gpu_stream_scheduler``.
_REGISTERED_SCHEDULER: 'Optional[Type[GPUStreamSchedulingStrategy]]' = None


def register_gpu_stream_scheduler(strategy_cls: 'Optional[Type[GPUStreamSchedulingStrategy]]') -> None:
    """ Register ``strategy_cls`` as the default stream-assignment strategy.
        Pass ``None`` to clear the registration and fall back to
        :class:`NaiveGPUStreamScheduler`.
    """
    global _REGISTERED_SCHEDULER
    if strategy_cls is not None and not issubclass(strategy_cls, GPUStreamSchedulingStrategy):
        raise TypeError(f"{strategy_cls.__name__} must subclass GPUStreamSchedulingStrategy "
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

    def assign(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        """ Returns a ``{node: stream_id}`` mapping per the WCC strategy in the
            class docstring. """
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

        A component requires a stream if it contains any GPU-relevant node
        (see ``is_gpu_relevant_node``) at any nesting depth — that helper is
        the single source of truth for "this node needs a stream wired to
        it" across the GPU specialization pipeline.
        """
        sdfg = state.parent
        for node in component:
            if isinstance(node, nodes.NestedSDFG):
                if any(is_gpu_relevant_node(n, parent.sdfg, parent) for n, parent in node.sdfg.all_nodes_recursive()):
                    return True
            elif is_gpu_relevant_node(node, sdfg, state):
                return True
        return False
