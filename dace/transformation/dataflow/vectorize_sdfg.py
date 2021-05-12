# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the vectorization transformation. """
import dace.data
from dace.sdfg.state import SDFGState
from dace import data, dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, SDFG, propagation
from dace.sdfg import utils as sdutil
from dace.sdfg.scope import ScopeSubgraphView
from dace.transformation import transformation
from dace.transformation.helpers import replicate_scope
from dace.properties import Property, make_properties
import itertools
import functools
import operator
from dace.sdfg.replace import _replsym

from typing import List


class ParsedMap:

    def __init__(self, state, map_entry, input_accesses, nsdfg, output_accesses, map_exit):
        self.state: SDFGState = state
        self.map_entry: nodes.MapEntry = map_entry
        self.input_accesses: List[nodes.AccessNode] = input_accesses
        self.nsdfg: nodes.NestedSDFG = nsdfg
        self.output_accesses: List[nodes.AccessNode] = output_accesses
        self.map_exit: nodes.MapExit = map_exit

    @staticmethod
    def parse(state: SDFGState, map_entry: nodes.MapEntry):
        map_exit: nodes.MapExit = state.exit_node(map_entry)

        # if map_entry has inputs or map_exit has outputs, then it is not expected map format
        if state.in_edges(map_entry) or state.out_edges(map_exit):
            return None

        map_input_edges = state.out_edges(map_entry)

        no_inputs = False

        # special case, there are no inputs, so empty memlet connects map entry to nested sdfg
        if len(map_input_edges) == 1 and map_input_edges[0].data.is_empty() and isinstance(map_input_edges[0].dst, nodes.NestedSDFG):
            no_inputs = True

        input_accesses: List[nodes.AccessNode] = []
        if not no_inputs:
            # we expect that all edges connected to corresponding access nodes
            for e in map_input_edges:
                if not e.data.is_empty():
                    return None
                if not isinstance(e.dst, nodes.AccessNode):
                    return None
                input_accesses.append(e.dst)

        # find nested sdfg
        nested_sdfg: nodes.NestedSDFG
        if no_inputs:
            nested_sdfg = map_input_edges[0].dst
        else:
            nsdfg_input_edge = state.out_edges(input_accesses[0])
            if len(nsdfg_input_edge) != 1:
                return None
            nested_sdfg = nsdfg_input_edge[0].dst
            if not isinstance(nested_sdfg, nodes.NestedSDFG):
                return None

        # find output access nodes
        output_accesses: List[nodes.AccessNode] = []
        for nsdfg_output_edge in state.out_edges(nested_sdfg):
            an = nsdfg_output_edge.dst
            if not isinstance(an, nodes.AccessNode):
                return None
            output_accesses.append(an)

        # check that output access nodes connected directly to map_exit
        for out_edge in state.in_edges(map_exit):
            if out_edge.src not in output_accesses:
                return None

        return ParsedMap(state, map_entry, input_accesses, nested_sdfg, output_accesses, map_exit)


def vectorize_nsdfg(nsdfg: nodes.NestedSDFG, conn: str, orig_type, vec_size):

    # vectorize data descriptor
    array: dace.data.Array = nsdfg.sdfg.arrays[conn]
    vector_dim = array.strides.index(1)
    vec_type = dtypes.vector(orig_type, VectorizeSDFG.vector_size)
    array.dtype = vec_type

    # vectorize all memlets
    for state in nsdfg.sdfg.nodes():
        state: SDFGState
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            node: nodes.AccessNode
            if node.data != conn:
                continue
            for e in state.all_edges(node):
                # propagate vectorization to nodes that touch vectorized edge
                if e.dst != node:
                    if isinstance(e.dst, nodes.NestedSDFG):
                        e.dst.in_connectors[e.dst_conn] = vec_type
                        vectorize_nsdfg(e.dst, e.dst_conn, orig_type, vec_size)
                    if isinstance(e.dst, nodes.Tasklet):
                        e.dst.in_connectors[e.dst_conn] = vec_type
                    # propagate vectorization from input to output of tasklets
                    if isinstance(e.dst, nodes.Tasklet):
                        t: nodes.Tasklet = e.dst
                        for e in state.out_edges(t):
                            if not e.data.wcr:
                                # common case, just vectorize output
                                vectorize_nsdfg(nsdfg, e.data.data, orig_type, vec_size)
                            else:
                                # special case, WCR should be reduced
                                tmp_name = e.data.data + '_vec_wcr'
                                state.remove_node(e.dst)
                                nsdfg.sdfg.add_transient(name=tmp_name, shape=[1], dtype=vec_type)
                                tmp_access = state.add_access(tmp_name)
                                state.add_edge(e.src, e.src_conn, tmp_access, None,
                                               dace.Memlet(data=tmp_name, subset='0'))

                                # wcr nsdfg with map START
                                wcr_nsdfg = dace.SDFG(name='wcr_nsdfg')

                                state_with_map = wcr_nsdfg.add_state(label='wcr_from_vector_map_state',
                                                                     is_start_state=True)

                                wcr_nsdfg.add_array(name='in_vec', shape=[vec_size], dtype=orig_type)
                                wcr_nsdfg.add_array(name='out_scalar', shape=[1], dtype=orig_type)

                                map_entry, map_exit = state_with_map.add_map(name='wcr_vec_map',
                                                                             ndrange={'idx': f'0:{vec_size}'})

                                wcr_tasklet = state_with_map.add_tasklet(
                                    name='wcr_tasklet', inputs={'a': orig_type}, outputs={'b': orig_type},
                                    code='b = a;', language=dtypes.Language.CPP)

                                in_vec_access = state_with_map.add_access('in_vec')
                                out_scalar_access = state_with_map.add_access('out_scalar')

                                state_with_map.add_edge(map_entry, None, in_vec_access, None, dace.Memlet())
                                state_with_map.add_edge(in_vec_access, None, wcr_tasklet, 'a',
                                                        dace.Memlet(data='in_vec', subset='idx'))
                                state_with_map.add_edge(wcr_tasklet, 'b', out_scalar_access, None,
                                                        dace.Memlet(data='out_scalar', subset='0',
                                                                    wcr=e.data.wcr))
                                state_with_map.add_edge(out_scalar_access, None, map_exit, None, dace.Memlet())
                                # wcr nsdfg with map END

                                wcr_state = nsdfg.sdfg.add_state_after(state)

                                wcr_nsdfg_node = wcr_state.add_nested_sdfg(
                                    sdfg=wcr_nsdfg, parent='unused',
                                    inputs={'in_vec': dtypes.pointer(orig_type)},
                                    outputs={'out_scalar': dtypes.pointer(orig_type)})
                                wcr_in = wcr_state.add_access(tmp_name)
                                wcr_out = wcr_state.add_access(e.dst.data)

                                wcr_state.add_edge(wcr_in, None, wcr_nsdfg_node, 'in_vec',
                                                   dace.Memlet(data=tmp_name, subset='0'))
                                wcr_state.add_edge(wcr_nsdfg_node, 'out_scalar', wcr_out, None,
                                                   dace.Memlet(data=e.data.data, subset=e.data.subset, wcr=e.data.wcr))

                if e.src != node:
                    if isinstance(e.src, nodes.NestedSDFG):
                        e.src.out_connectors[e.src_conn] = vec_type
                        vectorize_nsdfg(e.src, e.src_conn, orig_type, vec_size)
                    if isinstance(e.src, nodes.Tasklet):
                        e.src.out_connectors[e.src_conn] = vec_type




@registry.autoregister_params(singlestate=True)
@make_properties
class VectorizeSDFG(transformation.Transformation):

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    out_access = transformation.PatternNode(nodes.AccessNode)
    map_exit = transformation.PatternNode(nodes.MapExit)

    vector_size = 4 # TODO should be replaced by property but how to access it from can_be_applied?

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                VectorizeSDFG.nested_sdfg,
                VectorizeSDFG.out_access,
                VectorizeSDFG.map_exit,
            )
        ]

    @staticmethod
    def can_be_applied(state: SDFGState, candidate, expr_index, sdfg, strict=False):
        nested_sdfg = state.nodes()[candidate[VectorizeSDFG.nested_sdfg]]
        out_access = state.nodes()[candidate[VectorizeSDFG.out_access]]
        map_exit = state.nodes()[candidate[VectorizeSDFG.map_exit]]

        # check that map can be removed and replaced by vector

        # we expect map with a single dimension
        map_range: subsets.Range = map_exit.map.range
        if map_range.dims() != 1:
            return False

        if map_range.size() != [VectorizeSDFG.vector_size]:
            return False

        range_start, range_end, range_step = map_range.ndrange()[0]

        if range_step != 1:
            return False

        pm = ParsedMap.parse(state, state.entry_node(map_exit))

        if not pm:
            return False

        return True

    def apply(self, sdfg: SDFG):

        state: SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph

        nested_sdfg = state.nodes()[candidate[VectorizeSDFG.nested_sdfg]]
        out_access = state.nodes()[candidate[VectorizeSDFG.out_access]]
        map_exit = state.nodes()[candidate[VectorizeSDFG.map_exit]]

        pm = ParsedMap.parse(state, state.entry_node(map_exit))

        # find map param name and range
        vectorization_param: str = pm.map_entry.map.params[0]
        map_first, map_last, map_step = pm.map_entry.map.range[0]

        # remove map
        state.remove_nodes_from([pm.map_entry, pm.map_exit])

        # iterate over all input/output edges and find use of map variables
        # (TODO: check in can_be_applied that all memlet indices point to Point, not Range)

        nsdfg_in_edges = state.in_edges(pm.nsdfg)
        nsdfg_out_edges = state.out_edges(pm.nsdfg)
        for e in itertools.chain(nsdfg_in_edges, nsdfg_out_edges):
            subset = e.data.subset
            print(type(subset), subset)
            total_size = functools.reduce(operator.mul, subset.size(), 1)
            assert total_size == 1  # TODO: should be checked in can_be_applied

            orig_type = sdfg.arrays[e.data.data].dtype
            vec_type = dtypes.vector(orig_type, VectorizeSDFG.vector_size)

            vectorized = False

            # find the location of index in data descriptor
            vectorized_range = []
            for start, end, step in subset:
                assert start == end
                assert step == 1
                if vectorization_param in map(str, start.free_symbols):
                    symrepl = { symbolic.symbol(vectorization_param): map_first }
                    new_start = _replsym(start, symrepl)

                    symrepl = { symbolic.symbol(vectorization_param): map_last }
                    new_end = _replsym(start, symrepl)

                    symrepl = { symbolic.symbol(vectorization_param): map_first + map_step }
                    new_step = _replsym(start, symrepl) - new_start

                    vectorized_range.append((new_start, new_end, new_step))

                    if e in nsdfg_in_edges:
                        pm.nsdfg.in_connectors[e.dst_conn] = vec_type

                        vectorize_nsdfg(pm.nsdfg, e.dst_conn, orig_type, VectorizeSDFG.vector_size)
                    else:
                        pm.nsdfg.out_connectors[e.src_conn] = vec_type

                        vectorize_nsdfg(pm.nsdfg, e.src_conn, orig_type, VectorizeSDFG.vector_size)

                    vectorized = True
                else:
                    vectorized_range.append((start, end, step))

            if vectorized:
                e.data.subset = subsets.Range(vectorized_range)
                #e.data.volume = functools.reduce(operator.mul, e.data.subset.size(), 1)  # TODO: should I update it manually?





        vectorizable_edges = [] # TODO

        # for each vectorizable edge,
        #   collect data descriptors that should be vectorized.
        data_descriptors = [] # TODO

        # create state before and after that load/store non-stride 1 elements in vectors
        # state_before = sdfg.add_state_before(state)
        # state_after = sdfg.add_state_after(state)

        # for each data descriptor to be vectorized
        #   check stride of data descriptor
        #   if stride is 1:
        #     replace data descriptor (TODO: consider later the case when data descriptor also used not as a vector)
        #   else:
        #     create new vector data desc, add load/store in state before or after
        #     replace memlet and access node by this new data descriptor

        # for each memlet to be vectorized
        # change its range and connector of nested sdfg

        # call method that will apply the same thing again for memlets inside nested sdfg that touch this memory location

        # when you get down to tasklet:
        # for remaining edges
        # if you see output edge with WCR, replace it by vector and move WCR code into state after
        # if you see normal edge
        #   if it is input edge
        #     create vector fill in state before
        #   if it is output edge
        #     create extraction of last vector elem in state after


        # TODO