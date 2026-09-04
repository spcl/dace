# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move an input-free Tasklet that blocks Map fusion into the Map it feeds.

``StateFusionExtended`` sequences a standalone Tasklet against a preceding Map with an ordering
(empty) Memlet. That turns the Tasklet into an intermediate of the first Map's exit, and
``MapFusionVertical.partition_first_outputs`` rejects an empty Memlet there, so the shape

.. code-block:: text

    MapExit1 -> T -----------> MapEntry2      (data)
    MapExit1 ~~> tasklet -> S -> MapEntry2    (~~> is the ordering edge)

never fuses, while the same shape without the ordering edge does. Distributing the Tasklet into
the second Map takes it off the first Map's outputs and restores fusability.

Replicating the Tasklet once per iteration preserves values because it reads nothing: every
replica computes what the single original execution computed. Its output becomes Map-scoped, so
the replicas do not race. The ordering edge is dropped rather than rerouted: a data path already
sequences the second Map after the first, so the Tasklet still runs after the first Map once it
lives inside that Map.
"""
import copy
from typing import Any, Union

import dace
from dace import dtypes, properties, transformation
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.transformation.dataflow import map_fusion_helper as mfhelper


@properties.make_properties
class DistributeTaskletIntoMap(transformation.SingleStateTransformation):
    """Distribute a free Tasklet feeding the second Map of a fusion candidate into that Map."""

    first_map_exit = transformation.PatternNode(nodes.MapExit)
    tasklet = transformation.PatternNode(nodes.Tasklet)
    access = transformation.PatternNode(nodes.AccessNode)
    second_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        return [sdutil.node_path_graph(cls.first_map_exit, cls.tasklet, cls.access, cls.second_map_entry)]

    def can_be_applied(self,
                       graph: Union[SDFGState, SDFG],
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        tasklet, access, second_map_entry = self.tasklet, self.access, self.second_map_entry

        # Free Tasklet: nothing to read, one value to produce.
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return False
        if graph.out_degree(tasklet) != 1 or tasklet.has_side_effects(sdfg):
            return False
        # A data edge from the first Map would be a real input, which a replica could not recompute.
        if any(not in_edge.data.is_empty() for in_edge in graph.in_edges(tasklet)):
            return False

        desc = access.desc(sdfg)
        if not desc.transient or desc.lifetime != dtypes.AllocationLifetime.Scope:
            return False
        # A larger buffer would make the per-iteration private copy expensive.
        if desc.total_size != 1:
            return False
        if graph.in_degree(access) != 1 or graph.out_degree(access) != 1:
            return False
        # Scope allocation is only sound if nothing outside this Map's scope names the buffer.
        if any(node is not access and node.data == access.data for state in sdfg.states()
               for node in state.data_nodes()):
            return False
        if any(access.data in edge.data.free_symbols for edge in sdfg.all_interstate_edges()):
            return False

        consumer_edge = graph.out_edges(access)[0]
        if consumer_edge.dst_conn is None or not consumer_edge.dst_conn.startswith('IN_'):
            return False
        if len(list(graph.in_edges_by_connector(second_map_entry, consumer_edge.dst_conn))) != 1:
            return False

        # Only worth doing when the Tasklet is the sole obstacle, so the Maps must otherwise fuse.
        first_map_entry = graph.entry_node(self.first_map_exit)
        if mfhelper.can_topologically_be_fused(first_map_entry, second_map_entry, graph, sdfg) is None:
            return False
        # That data path is also what keeps the dropped ordering edge satisfied after the move.
        return mfhelper.is_node_reachable_from(graph=graph,
                                               begin=self.first_map_exit,
                                               end=second_map_entry,
                                               ignore_empty_edges=True)

    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        tasklet, access, second_map_entry = self.tasklet, self.access, self.second_map_entry
        consumer_edge = graph.out_edges(access)[0]
        in_conn = consumer_edge.dst_conn
        out_conn = 'OUT_' + in_conn[3:]

        for in_edge in list(graph.in_edges(tasklet)):
            graph.remove_edge(in_edge)
        graph.remove_edge(consumer_edge)
        for inner_edge in list(graph.out_edges_by_connector(second_map_entry, out_conn)):
            graph.add_edge(access, None, inner_edge.dst, inner_edge.dst_conn, copy.deepcopy(inner_edge.data))
            graph.remove_edge(inner_edge)
        second_map_entry.remove_in_connector(in_conn)
        second_map_entry.remove_out_connector(out_conn)
        # A node with no data input needs an explicit scope edge to stay inside the Map.
        graph.add_edge(second_map_entry, None, tasklet, None, dace.Memlet())


__all__ = ['DistributeTaskletIntoMap']
