""" Contains classes that implement the vectorization transformation. """
from dace import data, types, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties


@make_properties
class Vectorization(pattern_matching.Transformation):
    """ Implements the vectorization transformation.

        Vectorization matches when all the input and output memlets of a 
        tasklet inside a map access the inner-most loop variable in their last
        dimension. The transformation changes the step of the inner-most loop
        to be equal to the length of the vector and vectorizes the memlets.
  """

    vector_len = Property(desc="Vector length", dtype=int, default=4)

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(Vectorization._map_entry,
                                   Vectorization._tasklet,
                                   Vectorization._map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[Vectorization._map_entry]]
        tasklet = graph.nodes()[candidate[Vectorization._tasklet]]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])
        found = False
        dtype = None

        # Check if all edges, adjacent to the tasklet,
        # use the parameter in their last dimension.
        for _src, _, _dest, _, memlet in graph.all_edges(tasklet):

            # Cases that do not matter for vectorization
            if isinstance(sdfg.arrays[memlet.data], data.Stream):
                continue
            if memlet.wcr is not None:
                continue

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
                            symbolic.pystr_to_symbolic(ex)
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

        return ' -> '.join(
            str(node) for node in [map_entry, tasklet, map_exit])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[Vectorization._map_entry]]
        tasklet = graph.nodes()[self.subgraph[Vectorization._tasklet]]
        map_exit = graph.nodes()[self.subgraph[Vectorization._map_exit]]
        param = symbolic.pystr_to_symbolic(map_entry.map.params[-1])

        # Create new vector size.
        vector_size = self.vector_len

        # Change the step of the inner-most dimension.
        dim_from, dim_to, _dim_step = map_entry.map.range[-1]
        map_entry.map.range[-1] = (dim_from, dim_to, vector_size)

        # Vectorize memlets adjacent to the tasklet.
        for _src, _, _dest, _, memlet in graph.all_edges(tasklet):
            subset = memlet.subset
            lastindex = memlet.subset[-1]
            if isinstance(lastindex, tuple):
                symbols = set()
                for indd in lastindex:
                    symbols.update(
                        symbolic.pystr_to_symbolic(indd).free_symbols)
            else:
                symbols = symbolic.pystr_to_symbolic(
                    memlet.subset[-1]).free_symbols
            if param in symbols:
                try:
                    memlet.veclen = vector_size
                except AttributeError:
                    return

        # TODO: Create new map for non-vectorizable part.

        return

    def modifies_graph(self):
        return True


pattern_matching.Transformation.register_pattern(Vectorization)
