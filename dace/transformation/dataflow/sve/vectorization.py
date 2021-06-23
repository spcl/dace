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
import dace.codegen.targets.sve.util as util
import dace.frontend.operations
import dace.data as data
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
        """
        for e in state.out_edges(map_entry):
            any_vector = any_vector or cls.try_vectorize(
                state, e, param_name, True, inferred)

        for e in state.in_edges(map_exit):
            any_vector = any_vector or cls.try_vectorize(
                state, e, param_name, False, inferred)
        """

        for node in dfs_topological_sort(subgraph_contents):
            if not isinstance(node, nodes.Tasklet):
                continue

            for e in state.in_edges(node):
                any_vector = any_vector or SVEVectorization.try_vectorize(
                    state, e, param_name, True, inferred)
            for e in state.out_edges(node):
                any_vector = any_vector or SVEVectorization.try_vectorize(
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
        """
        for edge in state.out_edges(map_entry):
            SVEVectorization.try_vectorize(state, edge, param_name, True,
                                           inferred)

        for edge in state.in_edges(map_exit):
            SVEVectorization.try_vectorize(state, edge, param_name, False,
                                           inferred)
        """

        edges_to_vectorize = []

        for node in dfs_topological_sort(subgraph_contents):
            if not isinstance(node, nodes.Tasklet):
                continue

            for e in state.in_edges(node):
                if SVEVectorization.try_vectorize(state, e, param_name, True,
                                                  inferred):
                    edges_to_vectorize.append(e)

            for e in state.out_edges(node):
                if SVEVectorization.try_vectorize(state, e, param_name, False,
                                                  inferred):
                    edges_to_vectorize.append(e)

        edges_to_vectorize.extend(
            SVEVectorization.propagate_vector_information(
                sdfg, state, subgraph, inferred))

        for e in edges_to_vectorize:
            SVEVectorization.vectorize_memlet_subset(sdfg, e, current_map)

        # Apply the changes
        infer_types.apply_connector_types(inferred)

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

    def vectorize_memlet_subset(sdfg: SDFG, edge: MultiConnectorEdge[Memlet],
                                map: nodes.Map):
        """
            Vectorized the subset of a memlet given the vector param.
        """
        if edge.data.num_elements() != 1:
            # Subsets can't be vectorized
            return

        sve_dim = None
        if len(edge.data.subset) > 1:
            for dim, sub in enumerate(edge.data.subset):
                for expr in sub:
                    if map.params[-1] in symbolic.pystr_to_symbolic(
                            expr).free_symbols:
                        if sve_dim is not None:
                            # TODO: Param occurs in multiple dimensions
                            return
                        sve_dim = dim

        if sve_dim is None:
            sve_dim = -1

        sub = edge.data.subset[sve_dim]

        stride = edge.data.get_stride(sdfg, map)
        if stride == 0:
            # Scalar
            stride = 1

        edge.data.subset[sve_dim] = (sub[0], sub[1] + stride * util.SVE_LEN,
                                     stride)

    def propagate_vector_information(sdfg: SDFG, state: SDFGState,
                                     scope: ScopeSubgraphView,
                                     inferred: defaultdict) -> list:
        """
            Propagates the vector information through the scope given
            a dictionary of inferred types (will be modified).
            Returns a list of all newly vectorized edges.
        """

        inf = copy.copy(inferred)
        vec_edges = []

        # Forward propagation
        for node in dfs_topological_sort(scope):
            if not isinstance(node, nodes.Tasklet):
                continue

            ###################
            # Input connector inference from above
            # Check if we can copy vector information from some out connector above
            for edge in scope.in_edges(node):
                if isinstance(edge.src, nodes.AccessNode):
                    ###################
                    # Copying from some AccessNode

                    arr = edge.src.desc(sdfg)
                    # Only propagate if a Scalar with vector type
                    if not isinstance(arr, data.Scalar) or not isinstance(
                            arr.dtype, dtypes.vector):
                        continue

                    # Copy type from AccessNode
                    inf[(node, edge.dst_conn, True)] = arr.dtype
                    vec_edges.append(edge)
                else:
                    ###################
                    # Copying from some Tasklet
                    # Directly propagate the source out connector type

                    src_type = inf[(edge.src, edge.src_conn, False)]
                    dst_type = inf[(node, edge.dst_conn, True)]

                    if not isinstance(src_type, dtypes.vector):
                        # Nothing to propagate
                        continue

                    if isinstance(dst_type, (dtypes.vector, dtypes.pointer)):
                        # No need to further vectorize
                        continue

                    # Copy the vector information for the in connector
                    inf[(node, edge.dst_conn, True)] = src_type
                    vec_edges.append(edge)

            ###################
            # Infer the outputs again using the newly obtained vector input information

            # Drop all outputs and infer again
            for conn in node.out_connectors:
                del inf[(node, conn, False)]

            infer_types.infer_tasklet_connectors(sdfg, state, node, inf)

            # Check if anything fails and detect newly vectorized out connectors
            # We don't care about its base type, only whether it got vectorized (or even got scalar'd again)
            for conn in node.out_connectors:
                # Restore correct base type in case inferece changed it
                # (for example assigning 0.0 to a float32 output makes it float64)
                inf[(node, conn, False)].type = inferred[(node, conn,
                                                          False)].type

                if not isinstance(inferred[(node, conn, False)],
                                  dtypes.vector) and isinstance(
                                      inf[(node, conn, False)], dtypes.vector):
                    # Connector got vectorized, add all out edges
                    vec_edges.extend(state.out_edges_by_connector(node, conn))

                if isinstance(inferred[(node, conn, False)],
                              dtypes.vector) and not isinstance(
                                  inf[(node, conn, False)], dtypes.vector):
                    # Connector went from vector back to scalar
                    # Force it to become a vector (codegen takes care of scalar->vector)
                    inf[(node, conn, False)] = dtypes.vector(inf[(node, conn, False)], util.SVE_LEN)
                    # TODO: What else does this imply?

            ###################
            # Propagate information onto AccessNode, if any output connector is vector
            # FIXME: What if multiple Tasklets read from the same node below?
            # One node might not want it as a vector (then we should fail), but how do we find that out?

            for edge in scope.out_edges(node):
                if not isinstance(edge.dst, nodes.AccessNode):
                    continue

                out_type = inf[(node, edge.src_conn, False)]
                if not isinstance(out_type, dtypes.vector):
                    # Nothing to propagate
                    continue

                arr = edge.dst.desc(sdfg)

                # Only vectorize AccessNode if Scalar and not already vectorized
                if not isinstance(arr, data.Scalar) or isinstance(
                        arr.dtype, dtypes.vector):
                    continue

                arr.dtype = dtypes.vector(arr.dtype, util.SVE_LEN)

        inferred.update(inf)

        return vec_edges
