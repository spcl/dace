# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ReconnectWithinExpandedSDFGs`` — Phase 2 of the stream pipeline.

Runs after ``expand_library_nodes(recursive=True)``. For every NestedSDFG
that inherited a single ``stream`` connector from the LibraryNode it
replaced (see :func:`is_expanded_libnode_nsdfg`), wire every internal
GPU stream consumer to reuse that one stream. No fresh ``gpu_streams``
array is threaded into the body — all kernels and sub-libnodes inside
share the inherited stream.
"""
from typing import Any, Dict, Optional, Set, Type, Union

from dace import SDFG, SDFGState, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    STREAM_CONNECTOR, add_gpu_stream_connector, enclosing_map_chain, has_stream_connector, is_expanded_libnode_nsdfg,
    is_gpu_stream_consumer)


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
        if STREAM_CONNECTOR not in inner_sdfg.arrays:
            inner_sdfg.add_scalar(STREAM_CONNECTOR,
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
                if isinstance(node, nodes.AccessNode) and node.data == STREAM_CONNECTOR:
                    stream_an = node
                    break

            for node in list(state.nodes()):
                if not is_gpu_stream_consumer(node, inner_sdfg, state):
                    continue
                if has_stream_connector(node):
                    continue
                if stream_an is None:
                    stream_an = state.add_access(STREAM_CONNECTOR)
                self._wire_stream_to_consumer(state, stream_an, node, inner_sdfg)
                wired += 1
        return wired

    @staticmethod
    def _wire_stream_to_consumer(state: SDFGState, stream_an: nodes.AccessNode, consumer, sdfg: SDFG):
        """Wire ``stream_an → consumer.__stream``.

        Top-level consumer → single direct edge. Consumer inside one or
        more Sequential map scopes → reuse the shared seq-scope routing
        from :mod:`stream_lowering_helpers` so the same idempotent
        ``IN_<conn>`` / ``OUT_<conn>`` chain logic applies on both
        sides of the expand boundary.
        """
        from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (
            thread_stream_through_seq_scope)

        # Add the stream input connector on the consumer (if missing).
        if isinstance(consumer, nodes.MapEntry):
            add_gpu_stream_connector(consumer, STREAM_CONNECTOR, single_stream=True)
        elif STREAM_CONNECTOR not in consumer.in_connectors:
            add_gpu_stream_connector(consumer, STREAM_CONNECTOR, single_stream=True)

        chain = enclosing_map_chain(state, consumer, dtypes.ScheduleType.Sequential)
        if not chain:
            state.add_edge(stream_an, None, consumer, STREAM_CONNECTOR, Memlet(STREAM_CONNECTOR))
            return

        thread_stream_through_seq_scope(
            state,
            chain,
            consumer,
            STREAM_CONNECTOR,
            get_source_access=lambda: stream_an,
            memlet_factory=lambda: Memlet.from_array(STREAM_CONNECTOR, sdfg.arrays[STREAM_CONNECTOR]),
        )
