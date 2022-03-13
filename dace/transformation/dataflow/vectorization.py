# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the vectorization transformation. """
from re import M
import re
from numpy import core
import sympy

from numpy.lib.arraysetops import isin
from dace import data, dtypes, registry, symbolic, subsets, symbol
from dace.frontend.octave.lexer import raise_exception
from dace.sdfg import nodes, SDFG, SDFGState, propagation, InterstateEdge
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


def get_post_state(sdfg: SDFG, state: SDFGState):
    """ 
    Returns the post state (the state that copies the data a back from the FGPA device) if there is one.
    """
    for s in sdfg.all_sdfgs_recursive():
        for post_state in s.states():

            if 'post_' + str(state) == str(post_state):
                return post_state

    return None


# Find the innermost state in which the node is
# There is probably a better solution than this
def get_innermost_state_for_node(sdfg: SDFG, node):
    for s in sdfg.states():
        for n, state in s.all_nodes_recursive():
            if n == node:
                return state

    raise Exception("State for the node {n} not found".format(n=node))


# Find the innermost sdfg in which the node is
# There is probably a better solution than this
def get_innermost_sdfg_for_node(sdfg: SDFG, node):
    for s in sdfg.all_sdfgs_recursive():  #sdfg
        for n in s.nodes():  # states
            for i in n.nodes():  # nodes
                if node == i:
                    return s
    raise Exception("sdfg for the node {n} not found".format(n=node))


# Find the innermost sdfg in which the array is
# There is probably a better solution than this
def get_innermost_sdfg_for_array(sdfg: SDFG, array):
    for s in sdfg.all_sdfgs_recursive():  #sdfg
        if array in s.arrays:
            return s
    raise Exception("sdfg for the array {n} not found".format(n=array))


def collect_maps_to_vectorize(sdfg: SDFG, state, map_entry):
    """
    Collect all maps and the corresponding data descriptors that have to be vectorized
    if target == FPGA.
    """
    # Collect all possible and maps
    all_maps_entries_exits = set()

    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, (nodes.MapEntry, nodes.MapExit)):
            all_maps_entries_exits.add(n)

    # Check which we have to vectorize
    data_descriptors_to_vectorize = set()
    maps_to_vectorize = set()
    entries_exits_to_vectorize = set()

    # Add current map
    maps_to_vectorize.add(map_entry.map)
    entries_exits_to_vectorize.add(map_entry)
    all_maps_entries_exits.remove(map_entry)

    # Add current in edges
    for e in state.in_edges(map_entry):
        data_descriptors_to_vectorize.add(e.data.data)

    # Check all the remaing maps
    collected_all = False

    while not collected_all:
        collected_all = True

        for n in all_maps_entries_exits:

            add_map = False

            # Get all out/in edges of the map
            correct_state = get_innermost_state_for_node(sdfg, n)

            possible_edges = set()
            for e in correct_state.all_edges(n):
                possible_edges.add(e)

            for e in possible_edges:
                if e.data.data in data_descriptors_to_vectorize or n.map in maps_to_vectorize:
                    add_map = True
                    collected_all = False
                    break

            if add_map:
                maps_to_vectorize.add(n.map)
                entries_exits_to_vectorize.add(n)
                all_maps_entries_exits.remove(n)

                for e in possible_edges:
                    if e.data.data is not None:
                        data_descriptors_to_vectorize.add(e.data.data)
                break

    # Only interested in map entries
    results = set()
    for m in entries_exits_to_vectorize:
        if isinstance(m, nodes.MapEntry):

            is_ok = True
            correct_state = get_innermost_state_for_node(sdfg, m)
            subgraph_contents = correct_state.scope_subgraph(m, include_entry=False, include_exit=False)

            # Ensure only Tasklets and AccessNodes are within the map
            for node, _ in subgraph_contents.all_nodes_recursive():
                if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                    is_ok = False

            if is_ok:
                results.add(m)

    return results, data_descriptors_to_vectorize


def is_int(i):
    return isinstance(i, int) or isinstance(i, sympy.core.numbers.Integer)


@make_properties
class Vectorization(transformation.SingleStateTransformation):
    """ Implements the vectorization transformation.

        Vectorization matches when all the input and output memlets of a 
        tasklet inside a map access the inner-most loop variable in their last
        dimension. The transformation changes the step of the inner-most loop
        to be equal to the length of the vector and vectorizes the memlets.

        Possible targets: ARM SVE: FPGA, Default.
        Note: ARM SVE is length agnostic. If target == FPGA, the data containers are vectorized. 

  """

    vector_len = Property(desc="Vector length", dtype=int, default=4)
    strided_map = Property(desc="Use strided map range (jump by vector length)"
                           " instead of modifying memlets",
                           dtype=bool,
                           default=True)
    preamble = Property(
        dtype=bool,
        default=None,
        allow_none=True,
        desc='Force creation or skipping a preamble map without vectors. Only available if target == Default')

    postamble = Property(
        dtype=bool,
        default=None,
        allow_none=True,
        desc='Force creation or skipping a postamble map without vectors. Only available if target == Default')

    map_entry = transformation.PatternNode(nodes.MapEntry)

    target = EnumProperty(dtype=dtypes.ScheduleType,
                          desc='Set storage type for the newly-created stream',
                          default=dtypes.ScheduleType.Default)

    _level = 0  # used to prevent infinite loops

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        # Check if supported!
        supported_targets = [dtypes.ScheduleType.Default, dtypes.ScheduleType.SVE_Map, dtypes.ScheduleType.FPGA_Device]

        if self.target not in supported_targets:
            return False

        # To support recursivity in the FPGA case, see below

        if self._level == 0:
            state = graph
        else:
            state = get_innermost_state_for_node(sdfg, map_entry)

        subgraph = state.scope_subgraph(map_entry)
        subgraph_contents = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)

        # Prevent infinite repeats
        if map_entry.map.schedule == dtypes.ScheduleType.SVE_Map:
            return False

        ########################
        # Ensure only Tasklets and AccessNodes are within the map
        for node, _ in subgraph_contents.all_nodes_recursive():
            if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                return False

        # Strided maps cannot be vectorized
        if map_entry.map.range[-1][2] != 1 \
                and self.strided_map \
                and self.target != dtypes.ScheduleType.SVE_Map:
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
                reduction_type = dace.frontend.operations.detect_reduction_type(e.data.wcr)
                if reduction_type not in sve_util.REDUCTION_TYPE_TO_SVE and self.target == dtypes.ScheduleType.SVE_Map:
                    return False

                # Param in memlet during WCR is not supported
                if param_name in e.data.subset.free_symbols and e.data.wcr_nonatomic:
                    return False

                # vreduce is not supported
                dst_node = state.memlet_path(e)[-1].dst
                if isinstance(dst_node, nodes.Tasklet):
                    if isinstance(dst_node.in_connectors[e.dst_conn], dtypes.vector):
                        return False

                elif isinstance(dst_node, nodes.AccessNode):
                    desc = dst_node.desc(sdfg)
                    if isinstance(desc.dtype, (dtypes.vector, dace.pointer)):
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

            if self.target != dtypes.ScheduleType.SVE_Map:
                subset = e.data.subset
                array = sdfg.arrays[e.data.data]
                param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])

                try:
                    for idx, expr in enumerate(subset):
                        if isinstance(expr, tuple):
                            for ex in expr:
                                ex = symbolic.pystr_to_symbolic(ex)
                                symbols = ex.free_symbols
                                if param in symbols:
                                    if array.strides[idx] != 1:
                                        return False
                        else:
                            expr = symbolic.pystr_to_symbolic(expr)
                            symbols = expr.free_symbols
                            if param in symbols:
                                if array.strides[idx] != 1:
                                    return False
                except TypeError:  # cannot determine truth value of Relational
                    return False

            for e in state.in_edges(node):
                # Check for valid copies from other tasklets and/or streams
                if e.data.data is not None:
                    src_node = state.memlet_path(e)[0].src
                    if not isinstance(src_node, (nodes.Tasklet, nodes.AccessNode)):
                        # Make sure we only have Code->Code copies and from arrays
                        return False

                    if isinstance(src_node, nodes.AccessNode):
                        src_desc = src_node.desc(sdfg)
                        if isinstance(src_desc, data.Stream) and self.target == dtypes.ScheduleType.SVE_Map:
                            # Stream pops are not implemented
                            return False

        # Check if it is possible to vectorize data container
        if self.target == dtypes.ScheduleType.FPGA_Device and self._level == 0:
            maps_to_vectorize, data_descriptors_to_vectorize = collect_maps_to_vectorize(sdfg, state, map_entry)

            old_map_entry = self.map_entry
            self._level = 1  # To prevent infinte loop

            for m in maps_to_vectorize:
                self.map_entry = m

                if not self.can_be_applied(get_innermost_state_for_node(sdfg, m), expr_index,
                                           get_innermost_sdfg_for_node(sdfg, m), permissive):
                    return False

            # Check alls strideds of the arrays
            for a in data_descriptors_to_vectorize:

                correct_sdfg = get_innermost_sdfg_for_array(sdfg, a)
                array = correct_sdfg.arrays[a]
                strides_list = list(array.strides)

                if strides_list[-1] != 1:
                    return False

                strides_list.pop()

                for i in strides_list:
                    if is_int(i) and i % self.vector_len != 0:
                        return False

            # Check all maps
            for m in maps_to_vectorize:
                real_map: nodes.Map = m.map
                ranges_list = list(real_map.range)

                if is_int(ranges_list[-1][1]) and (ranges_list[-1][1] + 1) % self.vector_len != 0:
                    return False

                if ranges_list[-1][2] != 1:
                    return False

            # Check all edges
            for m in maps_to_vectorize:

                correct_state = get_innermost_state_for_node(sdfg, m)

                edges = correct_state.all_edges(m)

                map_subset = m.map.params

                for e in edges:

                    if e is None or e.data is None or e.data.subset is None:
                        continue

                    edge_subset = [a_tuple[0] for a_tuple in list(e.data.subset)]

                    if isinstance(edge_subset[-1], symbol) and str(edge_subset[-1]) != map_subset[-1]:
                        return False

            # Not possible to handle interstate edges

            for e, _ in sdfg.all_edges_recursive():

                if isinstance(e.data, InterstateEdge):

                    if e.data.assignments != {} or e.data.condition.as_string != "1":
                        return False

            self.map_entry = old_map_entry
            self._level = 0

            # Run the vector inference algorithm to check if vectorization is feasible
        try:
            vector_inference.infer_vectors(sdfg,
                                           state,
                                           map_entry,
                                           self.vector_len,
                                           flags=vector_inference.VectorInferenceFlags.Allow_Stride,
                                           strided_map=self.strided_map
                                           or self.target == dtypes.ScheduleType.FPGA_Device,
                                           apply=False)
        except vector_inference.VectorInferenceException as ex:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry
        tasklet: nodes.Tasklet = graph.successors(map_entry)[0]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])

        # To support recursivity in the FPGA case, see below
        if self._level == 0:
            state = graph
        else:
            state = get_innermost_state_for_node(sdfg, map_entry)

        if self.target == dtypes.ScheduleType.FPGA_Device and self._level == 0:

            maps_to_vectorize, data_descriptors_to_vectorize = collect_maps_to_vectorize(sdfg, state, map_entry)

            old_map_entry = self.map_entry
            self._level = 1  # To prevent infinte loop

            for m in maps_to_vectorize:
                self.map_entry = m

                self.apply(get_innermost_state_for_node(sdfg, m), get_innermost_sdfg_for_node(sdfg, m))

            post_state = get_post_state(sdfg, graph)

            if post_state != None:

                for e in post_state.edges():
                    # Change subset in the post state such that the correct amount of memory is copied back from the device

                    if e.data.data in data_descriptors_to_vectorize:

                        desc = sdfg.arrays[e.data.data]
                        contigidx = desc.strides.index(1)
                        lastindex = e.data.subset[contigidx]
                        i, j, k = lastindex
                        e.data.subset[contigidx] = (i, (j + 1) / self.vector_len - 1, k)

            # Change strides of all arrays involved
            for a in data_descriptors_to_vectorize:

                correct_sdfg = get_innermost_sdfg_for_array(sdfg, a)
                array = correct_sdfg.arrays.get(a)
                new_strides = list(array.strides)

                for i in range(len(new_strides)):
                    if i == len(new_strides) - 1:  # Skip last dimension since it is always 1
                        continue
                    new_strides[i] = new_strides[i] / self.vector_len
                correct_sdfg.arrays[a].strides = new_strides

            self.map_entry = old_map_entry
            self._level = 0
            return

        # Determine new range for vectorized map
        vector_size = self.vector_len
        dim_from, dim_to, dim_skip = map_entry.map.range[-1]
        if self.strided_map:
            new_range = [dim_from, dim_to - vector_size + 1, vector_size]
        else:
            new_range = [dim_from // vector_size, ((dim_to + 1) // vector_size) - 1, dim_skip]

        if self.target == dtypes.ScheduleType.SVE_Map:
            new_range = [new_range[0], new_range[1], dim_skip]

        # Determine whether to create preamble or postamble maps
        if self.preamble is not None:
            create_preamble = self.preamble
        else:
            create_preamble = not ((dim_from % vector_size == 0) == True or dim_from == 0)
        if self.postamble is not None:
            create_postamble = self.postamble
        else:
            if isinstance(dim_to, symbolic.SymExpr):
                create_postamble = (((dim_to.approx + 1) % vector_size == 0) == False)
            else:
                create_postamble = (((dim_to + 1) % vector_size == 0) == False)

        graph = sdfg.nodes()[self.state_id]

        # Create preamble non-vectorized map (replacing the original map)
        if create_preamble and self.target == dtypes.ScheduleType.Default:
            old_scope = graph.scope_subgraph(map_entry, True, True)
            new_scope: ScopeSubgraphView = replicate_scope(sdfg, graph, old_scope)
            new_begin = dim_from + (vector_size - (dim_from % vector_size))
            map_entry.map.range[-1] = (dim_from, new_begin - 1, dim_skip)
            # Replace map_entry with the replicated scope (so that the preamble
            # will usually come first in topological sort)
            map_entry = new_scope.entry
            # tasklet = new_scope.nodes()[old_scope.nodes().index(tasklet)]
            new_range[0] = new_begin

        # Create postamble non-vectorized map
        if create_postamble and self.target == dtypes.ScheduleType.Default:
            new_scope: ScopeSubgraphView = replicate_scope(sdfg, graph, graph.scope_subgraph(map_entry, True, True))
            dim_to_ex = dim_to + 1
            new_scope.entry.map.range[-1] = (dim_to_ex - (dim_to_ex % vector_size), dim_to, dim_skip)

        # Change the step of the inner-most dimension.
        map_entry.map.range[-1] = tuple(new_range)

        # Expand the innermost map if multidimensional
        if len(map_entry.map.params) > 1:
            ext, rem = dace.transformation.helpers.extract_map_dims(sdfg, map_entry,
                                                                    list(range(len(map_entry.map.params) - 1)))
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
            strided_map=self.strided_map or self.target == dtypes.ScheduleType.FPGA_Device,
            apply=True,
        )

        # Vector length propagation using data descriptors, recursive traversal
        # outwards
        if self.target == dtypes.ScheduleType.FPGA_Device:

            for edge, _ in subgraph.all_edges_recursive():

                if edge is None or edge.data is None or edge.data.data is None:
                    continue

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
                    newlist[contigidx] = (rb / self.vector_len, rb / self.vector_len, 1)
                else:
                    newlist[contigidx] = (rb, rb, 1)

                edge.data.subset = subsets.Range(newlist)

                cursdfg = sdfg
                curedge = edge
                while cursdfg is not None:
                    arrname = curedge.data.data
                    arr = cursdfg.arrays[arrname]
                    dtype = arr.dtype

                    # Change type and shape to vector
                    if not isinstance(dtype, dtypes.vector):
                        arr.dtype = dtypes.vector(dtype, vector_size)
                        new_shape = list(arr.shape)
                        contigidx = arr.strides.index(1)
                        new_shape[contigidx] /= vector_size
                        try:
                            new_shape[contigidx] = int(new_shape[contigidx])
                        except TypeError:
                            pass
                        arr.shape = new_shape

                    propagation.propagate_memlets_sdfg(cursdfg)

                    # Find matching edge in parent
                    nsdfg = cursdfg.parent_nsdfg_node
                    if nsdfg is None:
                        break
                    tstate = cursdfg.parent
                    curedge = ([e for e in tstate.in_edges(nsdfg) if e.dst_conn == arrname] +
                               [e for e in tstate.out_edges(nsdfg) if e.src_conn == arrname])[0]
                    cursdfg = cursdfg.parent_sdfg
