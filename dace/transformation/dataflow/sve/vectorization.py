# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    SVE Vectorization: This module offers all functionality to vectorize an SDFG for the Arm SVE codegen.
"""
from dace.sdfg.scope import ScopeSubgraphView
from dace.sdfg.state import SDFGState
from dace import registry, symbolic, subsets
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
import dace.dtypes
import dace.sdfg.infer_types
import dace.transformation.dataflow
from dace.transformation.optimizer import Optimizer
import dace.transformation.helpers
import copy
import dace.codegen.targets.sve as sve
import dace.codegen.targets.sve.util


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
    def can_be_applied(cls,
                       graph: SDFGState,
                       candidate,
                       expr_index,
                       sdfg: SDFG,
                       strict=False) -> bool:
        # Infer all connector types for later checks
        # FIXME: Currently the SDFG is copied, modify `infer_connector_types` to work on subgraphs too
        state_id = sdfg.node_id(graph)
        sdfg = copy.deepcopy(sdfg)
        dace.sdfg.infer_types.infer_connector_types(sdfg)
        graph = sdfg.nodes()[state_id]

        entry_node = graph.node(candidate[cls.map_entry])
        exit_node = graph.exit_node(entry_node)
        subgraph = graph.scope_subgraph(entry_node)
        subgraph_contents = graph.scope_subgraph(entry_node,
                                                 include_entry=False,
                                                 include_exit=False)

        ########################
        # Ensure only Tasklets are within the map
        # TODO: Add AccessNode support
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                return False

        ########################
        # Check memlets for unsupported strides
        # The only unsupported strides are the ones containing the innermost
        # loop param because they are not constant during a vector step
        loop_param = symbolic.symbol(entry_node.params[-1])
        for edge, _ in subgraph.all_edges_recursive():
            if loop_param in edge.data.get_stride(sdfg,
                                                  entry_node.map).free_symbols:
                return False

        ########################
        # Check for unsupported datatypes on the connectors (including on the Map itself)
        for node, _ in subgraph.all_nodes_recursive():
            for conn in node.in_connectors:
                if not node.in_connectors[conn].type in sve.util.TYPE_TO_SVE:
                    return False
            for conn in node.out_connectors:
                if not node.out_connectors[conn].type in sve.util.TYPE_TO_SVE:
                    return False

        # TODO: Check for WCR conflicts

        # TODO: Check for stream pushes/pops

        # TODO: Check using unparser whether Tasklets are actually generateable

        return True

    def apply(self, sdfg: SDFG):
        graph = sdfg.node(self.state_id)
        map_entry = self.map_entry(sdfg)
        map_exit = graph.exit_node(map_entry)
        current_map = map_entry.map

        # Expand the innermost map if multidimensional
        if len(current_map.params) > 1:
            ext, rem = dace.transformation.helpers.extract_map_dims(
                sdfg, map_entry, list(range(len(current_map.params) - 1)))
            map_entry = rem
            map_exit = graph.exit_node(map_entry)
            current_map = map_entry.map

        # Set the schedule
        current_map.schedule = dace.dtypes.ScheduleType.SVE_Map

        # Infer all connector types
        # TODO: Vectorize
