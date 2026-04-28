# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ReconnectWithinExpandedSDFGs`` — Phase 2 of the stream pipeline.

Runs after ``expand_library_nodes(recursive=True)``. For every NestedSDFG
that inherited a single ``stream`` connector from the LibraryNode it
replaced (see :func:`is_expanded_libnode_nsdfg`), wire every internal
GPU stream consumer to reuse that one stream. No fresh ``gpu_streams``
array is threaded into the body — all kernels and sub-libnodes inside
share the inherited stream.
"""
from typing import Any, Dict, List, Optional, Set, Type, Union

from dace import SDFG, SDFGState, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               add_gpu_stream_connector,
                                                                               has_stream_connector,
                                                                               is_expanded_libnode_nsdfg,
                                                                               is_gpu_stream_consumer)


def _enclosing_sequential_map_chain(state: SDFGState, node: nodes.Node) -> List[nodes.MapEntry]:
    """Return the chain of Sequential MapEntries enclosing ``node`` in
    ``state``, ordered outermost → innermost. Empty when ``node`` is at
    state top level."""
    chain = []
    scope = state.entry_node(node)
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.Sequential:
            chain.append(scope)
        scope = state.entry_node(scope)
    chain.reverse()  # outermost first
    return chain


@properties.make_properties
@transformation.explicit_cf_compatible
class ReconnectWithinExpandedSDFGs(ppl.Pass):
    """Wire internal GPU consumers of expanded-libnode NestedSDFGs to
    reuse the single ``stream`` connector inherited from the LibraryNode."""

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Return the number of inner GPU consumers wired to an inherited
        stream, or ``None`` if nothing changed."""
        wired = 0
        # Walk the hierarchy depth-first so we treat parents before children:
        # each ``is_expanded_libnode_nsdfg`` recursion bottoms out on bodies
        # whose internal libnodes have themselves been expanded.
        for nsdfg_node, parent_state, parent_sdfg in self._all_nested_sdfg_nodes(sdfg):
            if not is_expanded_libnode_nsdfg(nsdfg_node):
                continue
            wired += self._reconnect_one(nsdfg_node, parent_state, parent_sdfg)
        return wired if wired > 0 else None

    @staticmethod
    def _all_nested_sdfg_nodes(sdfg: SDFG):
        """Yield ``(nsdfg_node, parent_state, parent_sdfg)`` for every
        NestedSDFG anywhere in the hierarchy."""
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        yield node, state, nsdfg

    def _reconnect_one(self, nsdfg_node: nodes.NestedSDFG, parent_state: SDFGState, parent_sdfg: SDFG) -> int:
        """Set up the inner-side ``stream`` Scalar (matching the inherited
        connector) and wire every internal GPU stream consumer to read from
        it. Returns the number of consumers wired."""
        inner_sdfg = nsdfg_node.sdfg

        # Inner-side ``stream`` Scalar must exist and match the dtype the
        # outer connector binds (``gpuStream_t``). Add or fix in place.
        if COPY_MEMSET_STREAM_CONNECTOR not in inner_sdfg.arrays:
            inner_sdfg.add_scalar(COPY_MEMSET_STREAM_CONNECTOR,
                                  dtypes.gpuStream_t,
                                  storage=dtypes.StorageType.Register,
                                  transient=False)

        wired = 0
        for state in inner_sdfg.states():
            # Find an existing AccessNode for ``stream`` in this state, or
            # create one — every consumer wired in this state reads from the
            # same node so the codegen sees a single bound stream.
            stream_an: Optional[nodes.AccessNode] = None
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == COPY_MEMSET_STREAM_CONNECTOR:
                    stream_an = node
                    break

            for node in list(state.nodes()):
                if not is_gpu_stream_consumer(node, inner_sdfg, state):
                    continue
                if has_stream_connector(node):
                    continue
                if stream_an is None:
                    stream_an = state.add_access(COPY_MEMSET_STREAM_CONNECTOR)
                self._wire_stream_to_consumer(state, stream_an, node, inner_sdfg)
                wired += 1
        return wired

    @staticmethod
    def _wire_stream_to_consumer(state: SDFGState, stream_an: nodes.AccessNode, consumer, sdfg: SDFG):
        """Wire ``stream_an → consumer.stream`` with single-element memlets.
        If ``consumer`` sits inside one or more Sequential map scopes, route
        the stream through each map's ``IN_stream`` / ``OUT_stream``
        pass-through connectors instead of crossing scope boundaries with a
        direct edge."""
        # Add the stream input connector on the consumer.
        if isinstance(consumer, nodes.MapEntry):
            add_gpu_stream_connector(consumer, COPY_MEMSET_STREAM_CONNECTOR, single_stream=True)
        elif COPY_MEMSET_STREAM_CONNECTOR not in consumer.in_connectors:
            # LibraryNode / Tasklet — connector may already exist from a
            # prior expansion; add only if missing.
            add_gpu_stream_connector(consumer, COPY_MEMSET_STREAM_CONNECTOR, single_stream=True)

        chain = _enclosing_sequential_map_chain(state, consumer)
        if not chain:
            # Consumer at state top level — single direct edge.
            state.add_edge(stream_an, None, consumer, COPY_MEMSET_STREAM_CONNECTOR,
                           Memlet(COPY_MEMSET_STREAM_CONNECTOR))
            return

        # Thread through every enclosing Sequential map: outermost map's
        # IN_stream binds to the wrapper's stream AccessNode; inner maps
        # chain via OUT_stream → next IN_stream; final OUT_stream goes to
        # the consumer's connector.
        outermost = chain[0]
        in_conn = f"IN_{COPY_MEMSET_STREAM_CONNECTOR}"
        out_conn = f"OUT_{COPY_MEMSET_STREAM_CONNECTOR}"
        outermost.add_in_connector(in_conn)
        outermost.add_out_connector(out_conn)
        state.add_edge(stream_an, None, outermost, in_conn,
                       Memlet.from_array(COPY_MEMSET_STREAM_CONNECTOR, sdfg.arrays[COPY_MEMSET_STREAM_CONNECTOR]))

        for outer, inner in zip(chain, chain[1:]):
            inner.add_in_connector(in_conn)
            inner.add_out_connector(out_conn)
            state.add_edge(outer, out_conn, inner, in_conn,
                           Memlet.from_array(COPY_MEMSET_STREAM_CONNECTOR, sdfg.arrays[COPY_MEMSET_STREAM_CONNECTOR]))

        # Innermost map → consumer.
        state.add_edge(chain[-1], out_conn, consumer, COPY_MEMSET_STREAM_CONNECTOR,
                       Memlet.from_array(COPY_MEMSET_STREAM_CONNECTOR, sdfg.arrays[COPY_MEMSET_STREAM_CONNECTOR]))
