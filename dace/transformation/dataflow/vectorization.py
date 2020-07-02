""" Contains classes that implement the vectorization transformation. """
from dace import data, registry, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties


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
                           default=False)

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

        # Check if all edges, adjacent to the tasklet,
        # use the parameter in their last dimension.
        for _src, _, _dest, _, memlet in graph.all_edges(tasklet):

            # Cases that do not matter for vectorization
            if memlet.data is None:  # Empty memlets
                continue
            if isinstance(sdfg.arrays[memlet.data], data.Stream):  # Streams
                continue

            # Vectorization can not be applied in WCR
            if memlet.wcr is not None:
                return False

            try:
                subset = memlet.subset
                veclen = memlet.veclen
            except AttributeError:
                return False

            if subset is None:
                return False

            try:
                if veclen > symbolic.pystr_to_symbolic('1'):
                    return False

                for idx, expr in enumerate(subset):
                    if isinstance(expr, tuple):
                        for ex in expr:
                            ex = symbolic.pystr_to_symbolic(ex)
                            symbols = ex.free_symbols
                            if param in symbols:
                                if idx == subset.dims() - 1:
                                    found = True
                                else:
                                    return False
                    else:
                        expr = symbolic.pystr_to_symbolic(expr)
                        symbols = expr.free_symbols
                        if param in symbols:
                            if idx == subset.dims() - 1:
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

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[Vectorization._map_entry]]
        tasklet = graph.nodes()[self.subgraph[Vectorization._tasklet]]
        map_exit = graph.nodes()[self.subgraph[Vectorization._map_exit]]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])

        # Create new vector size.
        vector_size = self.vector_len

        # Change the step of the inner-most dimension.
        dim_from, dim_to, dim_step = map_entry.map.range[-1]
        if self.strided_map:
            map_entry.map.range[-1] = (dim_from, dim_to, vector_size)
        else:
            map_entry.map.range[-1] = (dim_from, (dim_to + 1) / vector_size - 1,
                                       dim_step)

        # TODO: Postamble and/or preamble non-vectorized map

        # Vectorize memlets adjacent to the tasklet.
        processed_edges = set()
        for edge in graph.all_edges(tasklet):
            _src, _, _dest, _, memlet = edge

            if memlet.data is None:  # Empty memlets
                continue

            lastindex = memlet.subset[-1]
            if isinstance(lastindex, tuple):
                symbols = set()
                for indd in lastindex:
                    symbols.update(
                        symbolic.pystr_to_symbolic(indd).free_symbols)
            else:
                symbols = symbolic.pystr_to_symbolic(
                    memlet.subset[-1]).free_symbols

            if param not in symbols:
                continue
            try:
                # propagate vector length inside this SDFG
                for e in graph.memlet_tree(edge):
                    e.data.veclen = vector_size
                    if not self.strided_map and e not in processed_edges:
                        e.data.subset.replace({param: vector_size * param})
                        processed_edges.add(e)

                # propagate to the parent (TODO: handle multiple level of nestings)
                if self.propagate_parent and sdfg.parent is not None:
                    source_edge = graph.memlet_path(edge)[0]
                    sink_edge = graph.memlet_path(edge)[-1]

                    # Find parent Nested SDFG node
                    parent_node = next(n for n in sdfg.parent.nodes()
                                       if isinstance(n, nodes.NestedSDFG)
                                       and n.sdfg.name == sdfg.name)

                    # continue in propagating the vector length following the
                    # path that arrives to source_edge or starts from sink_edge
                    for pe in sdfg.parent.all_edges(parent_node):
                        if str(pe.dst_conn) == str(source_edge.src) or str(
                                pe.src_conn) == str(sink_edge.dst):
                            for ppe in sdfg.parent.memlet_tree(pe):
                                ppe.data.veclen = vector_size
                                if (not self.strided_map
                                        and ppe not in processed_edges):
                                    ppe.data.subset.replace(
                                        {param: vector_size * param})
                                    processed_edges.add(ppe)

            except AttributeError:
                raise
        return
