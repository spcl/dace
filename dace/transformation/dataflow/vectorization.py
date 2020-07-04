""" Contains classes that implement the vectorization transformation. """
from dace import data, dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, SDFG
from dace.sdfg import utils as sdutil
from dace.sdfg.scope import ScopeSubgraphView
from dace.transformation import pattern_matching
from dace.transformation.helpers import replicate_scope
from dace.properties import Property, make_properties
import itertools


@registry.autoregister_params(singlestate=True)
@make_properties
class Vectorization(pattern_matching.Transformation):
    """ Implements the vectorization transformation.

        Vectorization matches when all the input and output memlets of a 
        tasklet inside a map access the inner-most loop variable in their last
        dimension. The transformation changes the step of the inner-most loop
        to be equal to the length of the vector and vectorizes the memlets.
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
    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(Vectorization._map_entry,
                                   Vectorization._tasklet,
                                   Vectorization._map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[Vectorization._map_entry]]
        tasklet = graph.nodes()[candidate[Vectorization._tasklet]]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])
        found = False

        # Strided maps cannot be vectorized
        if map_entry.map.range[-1][2] != 1:
            return False

        # Check if all edges, adjacent to the tasklet,
        # use the parameter in their contiguous dimension.
        for e, conntype in graph.all_edges_and_connectors(tasklet):

            # Cases that do not matter for vectorization
            if e.data.data is None:  # Empty memlets
                continue
            if isinstance(sdfg.arrays[e.data.data], data.Stream):  # Streams
                continue

            # Vectorization can not be applied in WCR
            if e.data.wcr is not None:
                return False

            subset = e.data.subset
            array = sdfg.arrays[e.data.data]

            # If already vectorized or a pointer, do not apply
            if isinstance(conntype, (dtypes.vector, dtypes.pointer)):
                return False

            try:
                for idx, expr in enumerate(subset):
                    if isinstance(expr, tuple):
                        for ex in expr:
                            ex = symbolic.pystr_to_symbolic(ex)
                            symbols = ex.free_symbols
                            if param in symbols:
                                if array.strides[idx] == 1:
                                    found = True
                                else:
                                    return False
                    else:
                        expr = symbolic.pystr_to_symbolic(expr)
                        symbols = expr.free_symbols
                        if param in symbols:
                            if array.strides[idx] == 1:
                                found = True
                            else:
                                return False
            except TypeError:  # cannot determine truth value of Relational
                return False

        return found

    @staticmethod
    def match_to_str(graph, candidate):

        map_entry = candidate[Vectorization._map_entry]
        tasklet = candidate[Vectorization._tasklet]
        map_exit = candidate[Vectorization._map_exit]

        return ' -> '.join(str(node) for node in [map_entry, tasklet, map_exit])

    def apply(self, sdfg: SDFG):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[Vectorization._map_entry]]
        tasklet = graph.nodes()[self.subgraph[Vectorization._tasklet]]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])

        # Create new vector size.
        vector_size = self.vector_len
        dim_from, dim_to, _ = map_entry.map.range[-1]

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

        # Determine new range for vectorized map
        if self.strided_map:
            new_range = [dim_from, dim_to - vector_size + 1, vector_size]
        else:
            new_range = [
                dim_from // vector_size, ((dim_to + 1) // vector_size) - 1, 1
            ]

        # Create preamble non-vectorized map (replacing the original map)
        if create_preamble:
            old_scope = graph.scope_subgraph(map_entry, True, True)
            new_scope: ScopeSubgraphView = replicate_scope(
                sdfg, graph, old_scope)
            new_begin = dim_from + (vector_size - (dim_from % vector_size))
            map_entry.map.range[-1] = (dim_from, new_begin - 1, 1)
            # Replace map_entry with the replicated scope (so that the preamble
            # will usually come first in topological sort)
            map_entry = new_scope.entry
            tasklet = new_scope.nodes()[old_scope.nodes().index(tasklet)]
            new_range[0] = new_begin

        # Create postamble non-vectorized map
        if create_postamble:
            new_scope: ScopeSubgraphView = replicate_scope(
                sdfg, graph, graph.scope_subgraph(map_entry, True, True))
            new_scope.entry.map.range[-1] = (dim_to - (dim_to % vector_size),
                                             dim_to, 1)

        # Change the step of the inner-most dimension.
        map_entry.map.range[-1] = tuple(new_range)

        # Vectorize connectors adjacent to the tasklet.
        processed_edges = set()
        for edge in graph.all_edges(tasklet):
            connectors = (tasklet.in_connectors
                          if edge.dst == tasklet else tasklet.out_connectors)
            conn = edge.dst_conn if edge.dst == tasklet else edge.src_conn

            if edge.data.data is None:  # Empty memlets
                continue
            desc = sdfg.arrays[edge.data.data]
            contigidx = desc.strides.index(1)

            newlist = []

            lastindex = edge.data.subset[contigidx]
            if isinstance(lastindex, tuple):
                newlist = [(rb, re, rs) for rb, re, rs in edge.data.subset]
                symbols = set()
                for indd in lastindex:
                    symbols.update(
                        symbolic.pystr_to_symbolic(indd).free_symbols)
            else:
                newlist = [(rb, rb, 1) for rb in edge.data.subset]
                symbols = symbolic.pystr_to_symbolic(lastindex).free_symbols

            if str(param) not in map(str, symbols):
                continue

            # Vectorize connector
            oldtype = connectors[conn]
            if oldtype is None or oldtype.type is None:
                oldtype = desc.dtype
            connectors[conn] = dtypes.vector(oldtype, vector_size)

            # Modify memlet subset to match vector length
            if self.strided_map:
                rb = newlist[contigidx][0]
                newlist[contigidx] = (rb, rb + self.vector_len - 1, 1)
            else:
                rb = newlist[contigidx][0]
                newlist[contigidx] = (self.vector_len * rb,
                                      self.vector_len * rb + self.vector_len -
                                      1, 1)
            edge.data.subset = subsets.Range(newlist)
            edge.data.volume = vector_size

            # TODO: Reinstate vector length propagation with data descriptors
            # try:
            #     # propagate vector length inside this SDFG
            #     for e in graph.memlet_tree(edge):
            #         e.data.veclen = vector_size
            #         if not self.strided_map and e not in processed_edges:
            #             e.data.subset.replace({param: vector_size * param})
            #             processed_edges.add(e)

            #     # propagate to the parent (TODO: handle multiple level of nestings)
            #     if self.propagate_parent and sdfg.parent is not None:
            #         source_edge = graph.memlet_path(edge)[0]
            #         sink_edge = graph.memlet_path(edge)[-1]

            #         # Find parent Nested SDFG node
            #         parent_node = sdfg.parent_nsdfg_node

            #         # continue in propagating the vector length following the
            #         # path that arrives to source_edge or starts from sink_edge
            #         for pe in sdfg.parent.all_edges(parent_node):
            #             if str(pe.dst_conn) == str(source_edge.src) or str(
            #                     pe.src_conn) == str(sink_edge.dst):
            #                 for ppe in sdfg.parent.memlet_tree(pe):
            #                     ppe.data.veclen = vector_size
            #                     if (not self.strided_map
            #                             and ppe not in processed_edges):
            #                         ppe.data.subset.replace(
            #                             {param: vector_size * param})
            #                         processed_edges.add(ppe)

            # except AttributeError:
            #     raise
