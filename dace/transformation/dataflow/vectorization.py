# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the vectorization transformation. """
from dace import data, dtypes, registry, symbolic, subsets
from dace.frontend.octave.lexer import raise_exception
from dace.sdfg import nodes, SDFG, SDFGState, propagation
from dace.sdfg import utils as sdutil
from dace.sdfg.scope import ScopeSubgraphView
from dace.transformation import transformation
from dace.transformation.helpers import replicate_scope
from dace.properties import Property, make_properties, EnumProperty, SymbolicProperty
import itertools
import dace.transformation.dataflow.vectorization_infer_types as infer_types
import dace.sdfg.analysis.vector_inference as vector_inference
import dace.codegen.targets.sve.util as sve_util
import dace.frontend.operations


@registry.autoregister_params(singlestate=True)
@make_properties
class Vectorization(transformation.Transformation):
    """ Implements the vectorization transformation.

        Vectorization matches when all the input and output memlets of a 
        tasklet inside a map access the inner-most loop variable in their last
        dimension. The transformation changes the step of the inner-most loop
        to be equal to the length of the vector and vectorizes the memlets.

        Possible targets: ARM SVE or Default.
        Note: ARM SVE is length agnostic. 

  """

    vector_len = Property(desc="Vector length", dtype=int, default=4)

    propagate_parent = Property(desc="Propagate vector length through "
                                "parent SDFGs",
                                dtype=bool,
                                default=False)

    strided_map = Property(desc="Use strided map range (jump by vector length)"
                           " instead of modifying memlets",
                           dtype=bool,
                           default=True)
    preamble = Property(
        dtype=bool,
        default=None,
        allow_none=True,
        desc='Force creation or skipping a preamble map without vectors')

    postamble = Property(
        dtype=bool,
        default=None,
        allow_none=True,
        desc='Force creation or skipping a postamble map without vectors')

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    target = EnumProperty(dtype=dtypes.ScheduleType,
                          desc='Set storage type for the newly-created stream',
                          default=dtypes.ScheduleType.Default)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(Vectorization._map_entry)]

    def can_be_applied(self,
                       state: SDFGState,
                       candidate,
                       expr_index,
                       sdfg: SDFG,
                       strict=False) -> bool:

        # Check if supported!
        supported_targets = [
            dtypes.ScheduleType.Default,
            dtypes.ScheduleType.SVE_Map,
        ]

        if self.target not in supported_targets:
            return False

        map_entry = self._map_entry(sdfg)
        subgraph = state.scope_subgraph(map_entry)
        subgraph_contents = state.scope_subgraph(map_entry,
                                                 include_entry=False,
                                                 include_exit=False)

        # Prevent infinite repeats
        if map_entry.map.schedule == dtypes.ScheduleType.SVE_Map:
            return False

        ########################
        # Ensure only Tasklets and AccessNodes are within the map
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                return False

        if self.target == dtypes.ScheduleType.SVE_Map:

            # Infer all connector types for later checks (without modifying the graph)
            inferred = infer_types.infer_connector_types(sdfg, state, subgraph)

            ########################
            # Check for unsupported datatypes on the connectors (including on the Map itself)
            bit_widths = set()
            for node, _ in subgraph.all_nodes_recursive():
                for conn in node.in_connectors:
                    t = inferred[(node, conn, True)]
                    bit_widths.add(sve_util.get_base_type(t).bytes)
                    if not t.type in sve_util.TYPE_TO_SVE:
                        return False
                for conn in node.out_connectors:
                    t = inferred[(node, conn, False)]
                    bit_widths.add(sve_util.get_base_type(t).bytes)
                    if not t.type in sve_util.TYPE_TO_SVE:
                        return False

            # Multiple different bit widths occuring (messes up the predicates)
            if len(bit_widths) > 1:
                return False

        ########################
        # Check for unsupported memlets
        param_sym = symbolic.symbol(map_entry.map.params[-1])
        param_name = map_entry.map.params[-1]
        for e, _ in subgraph.all_edges_recursive():

            # Cases that do not matter for vectorization
            if e.data.data is None:  # Empty memlets
                continue
            if isinstance(sdfg.arrays[e.data.data], data.Stream):  # Streams
                continue

            # Check for unsupported strides
            # The only unsupported strides are the ones containing the innermost
            # loop param because they are not constant during a vector step
            if param_sym in e.data.get_stride(sdfg, map_entry.map).free_symbols:
                return False

            # Check for unsupported WCR
            if e.data.wcr is not None:
                # Unsupported reduction type
                reduction_type = dace.frontend.operations.detect_reduction_type(
                    e.data.wcr)
                if reduction_type not in sve_util.REDUCTION_TYPE_TO_SVE and self.target == dtypes.ScheduleType.SVE_Map:
                    return False

                # Param in memlet during WCR is not supported
                if param_name in e.data.subset.free_symbols and e.data.wcr_nonatomic:
                    return False

                # vreduce is not supported
                dst_node = state.memlet_path(e)[-1]
                if isinstance(dst_node, nodes.Tasklet):
                    if isinstance(dst_node.in_connectors[e.dst_conn],
                                  dtypes.vector):
                        return False

                elif isinstance(dst_node, nodes.AccessNode):
                    desc = dst_node.desc(sdfg)
                    if isinstance(desc, data.Scalar) and isinstance(
                            desc.dtype, dtypes.vector):
                        return False

        ########################
        # Check for invalid copies in the subgraph
        for node, _ in subgraph.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet):
                continue

            for e, conntype in state.all_edges_and_connectors(node):
                # If already vectorized or a pointer, do not apply
                if isinstance(conntype, (dtypes.vector, dtypes.pointer)):
                    return False

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
                        if isinstance(
                                src_desc, data.Stream
                        ) and self.target == dtypes.ScheduleType.SVE_Map:
                            # Stream pops are not implemented
                            return False

            # Run the vector inference algorithm to check if vectorization is feasible
        try:
            vector_inference.infer_vectors(
                sdfg,
                state,
                map_entry,
                self.vector_len,
                flags=vector_inference.VectorInferenceFlags.Allow_Stride,
                strided_map=self.strided_map,
                apply=False)
        except vector_inference.VectorInferenceException as ex:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = candidate[Vectorization._map_entry]
        return str(map_entry)

    def apply(self, sdfg: SDFG):
        state = sdfg.node(self.state_id)
        map_entry = self._map_entry(sdfg)

        # Determine new range for vectorized map
        vector_size = self.vector_len
        dim_from, dim_to, dim_skip = map_entry.map.range[-1]
        if self.strided_map:
            new_range = [dim_from, dim_to - vector_size + 1, vector_size]
        else:
            new_range = [
                dim_from // vector_size, ((dim_to + 1) // vector_size) - 1,
                dim_skip
            ]

        if self.target == dtypes.ScheduleType.SVE_Map:
            new_range = [dim_from, dim_to, dim_skip]

        # Determine whether to create preamble or postamble maps
        if self.preamble is not None:
            create_preamble = self.preamble
        else:
            create_preamble = not ((dim_from % vector_size == 0) == True
                                   or dim_from == 0)
        if self.postamble is not None:
            create_postamble = self.postamble
        else:
            if isinstance(dim_to, symbolic.SymExpr):
                create_postamble = (((dim_to.approx + 1) %
                                     vector_size == 0) == False)
            else:
                create_postamble = (((dim_to + 1) % vector_size == 0) == False)

        graph = sdfg.nodes()[self.state_id]

        # Create preamble non-vectorized map (replacing the original map)
        if create_preamble:
            old_scope = graph.scope_subgraph(map_entry, True, True)
            new_scope: ScopeSubgraphView = replicate_scope(
                sdfg, graph, old_scope)
            new_begin = dim_from + (vector_size - (dim_from % vector_size))
            map_entry.map.range[-1] = (dim_from, new_begin - 1, dim_skip)
            # Replace map_entry with the replicated scope (so that the preamble
            # will usually come first in topological sort)
            map_entry = new_scope.entry
            # tasklet = new_scope.nodes()[old_scope.nodes().index(tasklet)]
            new_range[0] = new_begin

        # Create postamble non-vectorized map
        if create_postamble:
            new_scope: ScopeSubgraphView = replicate_scope(
                sdfg, graph, graph.scope_subgraph(map_entry, True, True))
            dim_to_ex = dim_to + 1
            new_scope.entry.map.range[-1] = (dim_to_ex -
                                             (dim_to_ex % vector_size), dim_to,
                                             dim_skip)

        # Change the step of the inner-most dimension.
        map_entry.map.range[-1] = tuple(new_range)

        # Expand the innermost map if multidimensional
        if len(map_entry.map.params) > 1:
            ext, rem = dace.transformation.helpers.extract_map_dims(
                sdfg, map_entry, list(range(len(map_entry.map.params) - 1)))
            map_entry = rem

        subgraph = state.scope_subgraph(map_entry)

        if self.target == dtypes.ScheduleType.SVE_Map:
            # Set the schedule
            map_entry.map.schedule = dace.dtypes.ScheduleType.SVE_Map

        # Infer all connector types and apply them
        inferred = infer_types.infer_connector_types(sdfg, state, subgraph)
        infer_types.apply_connector_types(inferred)

        # Infer vector connectors and AccessNodes and apply them
        vector_inference.infer_vectors(
            sdfg,
            state,
            map_entry,
            self.vector_len,
            flags=vector_inference.VectorInferenceFlags.Allow_Stride,
            strided_map=self.strided_map,
            apply=True,
        )

        # Vector length propagation using data descriptors, recursive traversal
        # outwards
        if self.propagate_parent:
            for edge, _ in subgraph.all_edges_recursive():

                desc = sdfg.arrays[edge.data.data]
                contigidx = desc.strides.index(1)

                newlist = []

                lastindex = edge.data.subset[contigidx]
                if isinstance(lastindex, tuple):
                    newlist = [(rb, re, rs) for rb, re, rs in edge.data.subset]
                else:
                    newlist = [(rb, rb, 1) for rb in edge.data.subset]

                # Modify memlet subset to match vector length
                rb = newlist[contigidx][0]
                if self.strided_map:
                    newlist[contigidx] = (rb / self.vector_len,
                                          rb / self.vector_len, 1)
                else:
                    newlist[contigidx] = (rb, rb, 1)

                edge.data.subset = subsets.Range(newlist)

                cursdfg = sdfg
                curedge = edge
                while cursdfg is not None:
                    arrname = curedge.data.data
                    dtype = cursdfg.arrays[arrname].dtype

                    # Change type and shape to vector
                    if not isinstance(dtype, dtypes.vector):
                        cursdfg.arrays[arrname].dtype = dtypes.vector(
                            dtype, vector_size)
                        new_shape = list(cursdfg.arrays[arrname].shape)
                        contigidx = cursdfg.arrays[arrname].strides.index(1)
                        new_shape[contigidx] /= vector_size
                        try:
                            new_shape[contigidx] = int(new_shape[contigidx])
                        except TypeError:
                            pass
                        cursdfg.arrays[arrname].shape = new_shape

                    propagation.propagate_memlets_sdfg(cursdfg)

                    # Find matching edge in parent
                    nsdfg = cursdfg.parent_nsdfg_node
                    if nsdfg is None:
                        break
                    tstate = cursdfg.parent
                    curedge = ([
                        e
                        for e in tstate.in_edges(nsdfg) if e.dst_conn == arrname
                    ] + [
                        e for e in tstate.out_edges(nsdfg)
                        if e.src_conn == arrname
                    ])[0]
                    cursdfg = cursdfg.parent_sdfg