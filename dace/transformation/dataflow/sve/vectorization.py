# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    SVE Vectorization: This module offers all functionality to vectorize an SDFG for the Arm SVE codegen.
"""
from dace import registry, symbolic, subsets
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes, SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
import dace.dtypes
import dace.sdfg.infer_types
import dace.transformation.dataflow
from dace.transformation.optimizer import Optimizer


@registry.autoregister_params(singlestate=True)
class SVEVectorization(transformation.Transformation):
    """ Implements the Arm SVE vectorization transform.

        Takes a map entry of a possibly multidimensional map and enforces a
        vectorization on the innermost param for the SVE codegen.
"""

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    @classmethod
    def can_be_applied(cls, graph, candidate, expr_index, sdfg, strict=False):
        entry_node = graph.node(candidate[cls.map_entry])
        exit_node = graph.exit_node(entry_node)
        nodes_between = graph.all_nodes_between(entry_node, exit_node)

        # Ensure only Tasklets are within the map
        # TODO: Add AccessNode support
        for node in nodes_between:
            if not isinstance(node, nodes.Tasklet):
                return False

        return True

    def apply(self, sdfg: SDFG):
        graph = sdfg.node(self.state_id)
        map_entry = self.map_entry(sdfg)
        map_exit = graph.exit_node(map_entry)
        current_map = map_entry.map

        # Expand the innermost map
        if len(current_map.params) > 1:
            # First expand all maps
            entries = dace.transformation.dataflow.MapExpansion.apply_to(
                sdfg, map_entry=map_entry)

            # Then collapse starting from the outermost map up to the
            # 2nd inner map (so that the innermost is still expanded)
            curr_entry = entries[0]
            for i in range(1, len(entries) - 1):
                curr_entry, _ = dace.transformation.dataflow.MapCollapse.apply_to(
                    sdfg,
                    _outer_map_entry=curr_entry,
                    _inner_map_entry=entries[i])

            # From now on only focus on the innermost map
            map_entry = entries[-1]
            map_exit = graph.exit_node(map_entry)
            current_map = map_entry.map

        # Set the schedule
        current_map.schedule = dace.dtypes.ScheduleType.SVE_Map

        # Infer all connector types
        dace.sdfg.infer_types.infer_connector_types(sdfg)

        # TODO: Vectorize
        