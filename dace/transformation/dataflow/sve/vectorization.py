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
from dace.properties import make_properties, Property, SymbolicProperty
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
import dace.sdfg.analysis.vector_inference as vector_inference


@registry.autoregister_params(singlestate=True)
@make_properties
class SVEVectorization(transformation.Transformation):
    """ Implements the Arm SVE vectorization transform.

        Takes a map entry of a possibly multidimensional map and enforces a
        vectorization on the innermost param for the SVE codegen.
"""

    map_entry = transformation.PatternNode(nodes.MapEntry)

    vec_len = SymbolicProperty(desc="Vector length", default=util.SVE_LEN)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state: SDFGState, candidate, expr_index, sdfg: SDFG, permissive=False) -> bool:
        map_entry = self.map_entry(sdfg)
        current_map = map_entry.map
        subgraph = state.scope_subgraph(map_entry)
        subgraph_contents = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)

        # Prevent infinite repeats
        if current_map.schedule == dace.dtypes.ScheduleType.SVE_Map:
            return False

        # Infer all connector types for later checks (without modifying the graph)
        inferred = infer_types.infer_connector_types(sdfg, state, subgraph)

        ########################
        # Ensure only Tasklets and AccessNodes are within the map
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                return False

        ########################
        # Check for unsupported datatypes on the connectors (including on the Map itself)
        bit_widths = set()
        for node, _ in subgraph.all_nodes_recursive():
            for conn in node.in_connectors:
                t = inferred[(node, conn, True)]
                bit_widths.add(util.get_base_type(t).bytes)
                if not t.type in sve.util.TYPE_TO_SVE:
                    return False
            for conn in node.out_connectors:
                t = inferred[(node, conn, False)]
                bit_widths.add(util.get_base_type(t).bytes)
                if not t.type in sve.util.TYPE_TO_SVE:
                    return False

        # Multiple different bit widths occuring (messes up the predicates)
        if len(bit_widths) > 1:
            return False

        ########################
        # Check for unsupported memlets
        param_name = current_map.params[-1]
        for e, _ in subgraph.all_edges_recursive():
            # Check for unsupported strides
            # The only unsupported strides are the ones containing the innermost
            # loop param because they are not constant during a vector step
            param_sym = symbolic.symbol(current_map.params[-1])

            if param_sym in e.data.get_stride(sdfg, map_entry.map).free_symbols:
                return False

            # Check for unsupported WCR
            if e.data.wcr is not None:
                # Unsupported reduction type
                reduction_type = dace.frontend.operations.detect_reduction_type(e.data.wcr)
                if reduction_type not in sve.util.REDUCTION_TYPE_TO_SVE:
                    return False

                # Param in memlet during WCR is not supported
                if param_name in e.data.subset.free_symbols and e.data.wcr_nonatomic:
                    return False

                # vreduce is not supported
                dst_node = state.memlet_path(e)[-1]
                if isinstance(dst_node, nodes.Tasklet):
                    if isinstance(dst_node.in_connectors[e.dst_conn], dtypes.vector):
                        return False
                elif isinstance(dst_node, nodes.AccessNode):
                    desc = dst_node.desc(sdfg)
                    if isinstance(desc, data.Scalar) and isinstance(desc.dtype, dtypes.vector):
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
                    if not isinstance(src_node, (nodes.Tasklet, nodes.AccessNode)):
                        # Make sure we only have Code->Code copies and from arrays
                        return False

                    if isinstance(src_node, nodes.AccessNode):
                        src_desc = src_node.desc(sdfg)
                        if isinstance(src_desc, dace.data.Stream):
                            # Stream pops are not implemented
                            return False

        # Run the vector inference algorithm to check if vectorization is feasible
        try:
            vector_inference.infer_vectors(sdfg,
                                           state,
                                           map_entry,
                                           self.vec_len,
                                           flags=vector_inference.VectorInferenceFlags.Allow_Stride,
                                           apply=False)
        except vector_inference.VectorInferenceException as ex:
            return False

        return True

    def apply(self, sdfg: SDFG):
        state = sdfg.node(self.state_id)
        map_entry = self.map_entry(sdfg)
        current_map = map_entry.map

        # Expand the innermost map if multidimensional
        if len(current_map.params) > 1:
            ext, rem = dace.transformation.helpers.extract_map_dims(sdfg, map_entry,
                                                                    list(range(len(current_map.params) - 1)))
            map_entry = rem
            current_map = map_entry.map

        subgraph = state.scope_subgraph(map_entry)

        # Set the schedule
        current_map.schedule = dace.dtypes.ScheduleType.SVE_Map

        # Infer all connector types and apply them
        inferred = infer_types.infer_connector_types(sdfg, state, subgraph)
        infer_types.apply_connector_types(inferred)

        # Infer vector connectors and AccessNodes and apply them
        vector_inference.infer_vectors(sdfg,
                                       state,
                                       map_entry,
                                       self.vec_len,
                                       flags=vector_inference.VectorInferenceFlags.Allow_Stride,
                                       apply=True)
