# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    SVE Vectorization: This module offers all functionality to vectorize an SDFG for the Arm SVE codegen.
"""
import dace.codegen.tools.type_inference as type_inference
from sympy.codegen.ast import Scope
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
import dace.transformation.dataflow.sve.infer_types as infer_types
from collections import defaultdict
from dace.sdfg.utils import dfs_topological_sort


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
                       state: SDFGState,
                       candidate,
                       expr_index,
                       sdfg: SDFG,
                       strict=False) -> bool:
        map_entry = state.node(candidate[cls.map_entry])
        map_exit = state.exit_node(map_entry)
        current_map = map_entry.map
        subgraph = state.scope_subgraph(map_entry)
        subgraph_contents = state.scope_subgraph(map_entry,
                                                 include_entry=False,
                                                 include_exit=False)

        # Infer all connector types for later checks (without modifying the graph)
        inferred = infer_types.infer_connector_types(sdfg, state, subgraph)

        ########################
        # Ensure only Tasklets are within the map
        # TODO: Add AccessNode support
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                return False

        ########################
        # Check for unsupported datatypes on the connectors (including on the Map itself)
        for node, _ in subgraph.all_nodes_recursive():
            for conn in node.in_connectors:
                if not inferred[(node, conn,
                                 True)].type in sve.util.TYPE_TO_SVE:
                    return False
            for conn in node.out_connectors:
                if not inferred[(node, conn,
                                 False)].type in sve.util.TYPE_TO_SVE:
                    return False

        ########################
        # Check for unsupported memlets
        param_name = current_map.params[-1]
        for e, _ in subgraph.all_edges_recursive():
            # Check for unsupported strides
            # The only unsupported strides are the ones containing the innermost
            # loop param because they are not constant during a vector step
            param_sym = symbolic.symbol(current_map.params[-1])
            for e, _ in subgraph.all_edges_recursive():
                if param_sym in e.data.get_stride(sdfg,
                                                  map_entry.map).free_symbols:
                    return False

            # Check for unsupported WCR
            if e.data.wcr is not None:
                if is_write_conflicted_with_reason(state, e) is None:
                    return False

                # Unsupported reduction type
                reduction_type = dace.frontend.operations.detect_reduction_type(
                    e.data.wcr)
                if reduction_type not in sve.util.REDUCTION_TYPE_TO_SVE:
                    return False

                # Param in memlet during WCR is not supported
                if param_name in e.data.free_symbols:
                    return False

        ########################
        # Check for invalid copies in the subgraph
        for node, _ in subgraph.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                continue

            for e in state.in_edges(node):
                # Check for valid copies from other tasklets and/or streams
                if e.data.data is not None:
                    src_node = state.memlet_path(e)[0].src
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
        # Check if at least one connector can be vectorized, that is either from the entry or to the exit.
        # Otherwise there is no way for any vector information to pass through (no vectorization required).

        any_vector = False
        for e in state.out_edges(map_entry):
            any_vector = any_vector or cls.try_vectorize(
                state, e, param_name, True, inferred)

        for e in state.in_edges(map_exit):
            any_vector = any_vector or cls.try_vectorize(
                state, e, param_name, False, inferred)

        if not any_vector:
            # No connector can be vectorized, therefore no vector information can propagate
            return False

        cls.propagate_vector_information(sdfg, state, subgraph, inferred)

        # TODO: Check using unparser whether Tasklets are actually generateable

        # TODO: Check where else the codegen could fail (e.g. output into pointers after vectorize attempt)

        return True

    def apply(self, sdfg: SDFG):
        state = sdfg.node(self.state_id)
        map_entry = self.map_entry(sdfg)
        map_exit = state.exit_node(map_entry)
        current_map = map_entry.map

        # Expand the innermost map if multidimensional
        if len(current_map.params) > 1:
            ext, rem = dace.transformation.helpers.extract_map_dims(
                sdfg, map_entry, list(range(len(current_map.params) - 1)))
            map_entry = rem
            map_exit = state.exit_node(map_entry)
            current_map = map_entry.map

        subgraph = state.scope_subgraph(map_entry)
        subgraph_contents = state.scope_subgraph(map_entry,
                                                 include_entry=False,
                                                 include_exit=False)

        # Set the schedule
        current_map.schedule = dace.dtypes.ScheduleType.SVE_Map

        # Infer all connector types
        inferred = infer_types.infer_connector_types(sdfg, state, subgraph)

        inf_without_vecs = copy.copy(inferred)

        # Vectorize the first and last level connectors, if possible
        param_name = current_map.params[0]
        for edge in state.out_edges(map_entry):
            SVEVectorization.try_vectorize(state, edge, param_name, True,
                                           inferred)

        for edge in state.in_edges(map_exit):
            SVEVectorization.try_vectorize(state, edge, param_name, False,
                                           inferred)

        SVEVectorization.propagate_vector_information(sdfg, state, subgraph, inferred)
        infer_types.apply_connector_types(inferred)

        # TODO: Use this difference to find out which memlets to vectorize (subset)
        difference = set(inferred.items()) - set(inf_without_vecs.items())

    def try_vectorize(graph: SDFGState,
                      edge: MultiConnectorEdge[Memlet],
                      param: str,
                      dst: bool = True,
                      inferred: defaultdict = None) -> bool:
        """
            Tries to vectorize the connector of the edge.
            
            Returns True, if the vectorization was successful.

            :param graph: The graph where the memlet resides.
            :param edge: The edge where the memlet resides.
            :param param: The param of the vectorized map.
            :param dst: Whether the destination (True) or the source (False) should be vectorized.
            :param inferred: The dictionary of inferred connector types. If None, the connector in the SDFG is queried and modified.
        """

        ####################
        # Check for possible vectorization

        if edge.data.data is None:
            # Empty memlets
            return False

        if edge.data.subset.num_elements() != 1:
            # More than one element in memlet
            return False

        if param not in edge.data.subset.free_symbols:
            # Loop param does not occur in memlet
            return False

        # Determine the current type of the connector
        conn = None
        node = None
        if dst:
            conn = edge.dst_conn
            node = edge.dst
        else:
            conn = edge.src_conn
            node = edge.src

        conn_type = None
        if inferred is not None:
            conn_type = inferred[(node, conn, dst)]
        else:
            if dst:
                conn_type = node.in_connectors[conn]
            else:
                conn_type = node.out_connectors[conn]

        if isinstance(conn_type, dtypes.vector):
            # No need to vectorize anymore
            return True
        elif isinstance(conn_type, dtypes.pointer):
            # Can't be vectorized
            return False

        ####################
        # Vectorize the connector

        new_type = dtypes.vector(sve.util.get_base_type(conn_type),
                                 sve.util.SVE_LEN)
        if dst:
            if inferred is not None:
                inferred[(node, conn, True)] = new_type
            else:
                node.in_connectors[conn] = new_type
        else:
            if inferred is not None:
                inferred[(node, conn, False)] = new_type
            else:
                node.out_connectors[conn] = new_type

        return True

    def vectorize_memlet_subset(edge: MultiConnectorEdge[Memlet], param: str):
        """
            Vectorized the subset of a memlet given the vector param.
            It is assumed, that at least one of the connectors was successfully vectorized
            (using `try_vectorize`).
        """
        pass

    def propagate_vector_information(sdfg: SDFG, state: SDFGState,
                                     scope: ScopeSubgraphView,
                                     inferred: defaultdict):
        pass
