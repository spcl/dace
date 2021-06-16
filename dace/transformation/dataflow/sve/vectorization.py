# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    SVE Vectorization: This module offers all functionality to vectorize an SDFG for the Arm SVE codegen.
"""
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.codegen.targets.cpp import is_write_conflicted_with_reason
from dace.sdfg.scope import ScopeSubgraphView
from dace.sdfg.state import SDFGState
from dace import registry, symbolic, subsets
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes, SDFG, SDFGState
import dace.sdfg
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
import dace.frontend.operations
import dace.data
import dace.dtypes as dtypes


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

        map_entry = graph.node(candidate[cls.map_entry])
        map_exit = graph.exit_node(map_entry)
        current_map = map_entry.map
        subgraph = graph.scope_subgraph(map_entry)
        subgraph_contents = graph.scope_subgraph(map_entry,
                                                 include_entry=False,
                                                 include_exit=False)

        ########################
        # Ensure only Tasklets are within the map
        # TODO: Add AccessNode support
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
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

        ########################
        # Check for unsupported memlets
        param_name = current_map.params[-1]
        for edge, _ in subgraph.all_edges_recursive():
            # Check for unsupported strides
            # The only unsupported strides are the ones containing the innermost
            # loop param because they are not constant during a vector step
            param_sym = symbolic.symbol(current_map.params[-1])
            for edge, _ in subgraph.all_edges_recursive():
                if param_sym in edge.data.get_stride(
                        sdfg, map_entry.map).free_symbols:
                    return False

            # Check for unsupported WCR
            if edge.data.wcr is not None:
                if is_write_conflicted_with_reason(graph, edge) is None:
                    return False

                # Unsupported reduction type
                reduction_type = dace.frontend.operations.detect_reduction_type(
                    edge.data.wcr)
                if reduction_type not in sve.util.REDUCTION_TYPE_TO_SVE:
                    return False

                # Param in memlet during WCR is not supported
                if param_name in edge.data.free_symbols:
                    return False

        ########################
        # Check for invalid copies in the subgraph
        for node, _ in subgraph.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                continue

            for edge in graph.in_edges(node):
                # Check for valid copies from other tasklets and/or streams
                if edge.data.data is not None:
                    src_node = graph.memlet_path(edge)[0].src
                    if not isinstance(src_node,
                                      (nodes.Tasklet, nodes.AccessNode)):
                        # Make sure we only have Code->Code copies and from arrays
                        return False

                    if isinstance(src_node, nodes.AccessNode):
                        src_desc = src_node.desc(sdfg)
                        if isinstance(src_desc, dace.data.Stream):
                            # Stream pops are not implemented
                            return False

        ########################
        # Check if at least one first level connector can be vectorized.
        # Otherwise there is no way for any vector information to pass through (no vectorization required).

        any_vector = False
        for edge in graph.out_edges(map_entry):
            # Vectorize the first level memlets, if possible
            any_vector = any_vector or cls.try_vectorize_dst(
                graph, edge, param_name, modify=False)

        if not any_vector:
            # No first level vector connector
            return False

        # TODO: Check using unparser whether Tasklets are actually generateable

        # TODO: Check where else the codegen could fail (e.g. output into pointers after vectorize attempt)

        return True

    def try_vectorize_dst(graph: SDFGState,
                          edge: MultiConnectorEdge[Memlet],
                          param: str,
                          modify: bool = True) -> bool:
        """
            Tries to vectorize the destination connector of the edge.
            
            Returns True, if the vectorization was successful.

            :param graph: The graph where the memlet resides.
            :param edge: The edge where the memlet resides.
            :param param: The param of the vectorized map.
            :param modify: True, if the connector should be vectorized.
                           False, if only check for possible vectorization
                           (no effect on the connectors).
        """

        ####################
        # Check for possible vectorization

        conn = edge.dst_conn
        dst_node = edge.dst
        in_conn = dst_node.in_connectors[conn]

        if edge.data.data is None:
            # Empty memlets
            return False

        if edge.data.subset.num_elements() != 1:
            # More than one element in memlet
            print(edge.data.subset.num_elements())
            return False

        if param not in edge.data.subset.free_symbols:
            # Loop param does not occur in memlet
            return False

        if isinstance(in_conn.type, (dtypes.vector, dtypes.pointer)):
            # TODO: How to treat pointers on single element?
            return False

        if not modify:
            # All checks were successful
            return True

        ####################
        # Vectorize the connector

        dst_node.in_connectors[conn] = dtypes.vector(
            sve.util.get_base_type(in_conn), sve.util.SVE_LEN)

        # TODO: Modify subset
        #edge.data.subset

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
        # TODO: Infer only on subgraph, same issue in can_be_applied
        dace.sdfg.infer_types.infer_connector_types(sdfg)

        # Vectorize the first level connectors, if possible
        param_name = current_map.params[0]
        for edge in graph.out_edges(map_entry):
            # Vectorize the first level connectors
            SVEVectorization.try_vectorize_dst(graph, edge, param_name)

        # TODO: Propagate vector information
