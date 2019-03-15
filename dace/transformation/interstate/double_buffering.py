"""Contains classes that implement the double buffering pattern. """

import copy
import itertools

import dace
from dace import data, types, sdfg as sd, subsets, symbolic
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching


class DoubleBuffering(pattern_matching.Transformation):
    """ Implements the double buffering pattern, which pipelines reading
        and processing data by creating a second copy of the memory. """

    _begin = sd.SDFGState()
    _guard = sd.SDFGState()
    _body = sd.SDFGState()
    _end = sd.SDFGState()

    @staticmethod
    def expressions():
        for_loop_graph = dace.graph.graph.OrderedDiGraph()
        for_loop_graph.add_nodes_from([
            DoubleBuffering._begin, DoubleBuffering._guard,
            DoubleBuffering._body, DoubleBuffering._end
        ])
        for_loop_graph.add_edge(DoubleBuffering._begin, DoubleBuffering._guard,
                                None)
        for_loop_graph.add_edge(DoubleBuffering._guard, DoubleBuffering._body,
                                None)
        for_loop_graph.add_edge(DoubleBuffering._body, DoubleBuffering._guard,
                                None)
        for_loop_graph.add_edge(DoubleBuffering._guard, DoubleBuffering._end,
                                None)

        return [for_loop_graph]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        begin = graph.nodes()[candidate[DoubleBuffering._begin]]
        guard = graph.nodes()[candidate[DoubleBuffering._guard]]
        body = graph.nodes()[candidate[DoubleBuffering._body]]
        end = graph.nodes()[candidate[DoubleBuffering._end]]

        if not begin.is_empty():
            return False
        if not guard.is_empty():
            return False
        if not end.is_empty():
            return False
        if body.is_empty():
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        begin = graph.nodes()[candidate[DoubleBuffering._begin]]
        guard = graph.nodes()[candidate[DoubleBuffering._guard]]
        body = graph.nodes()[candidate[DoubleBuffering._body]]
        end = graph.nodes()[candidate[DoubleBuffering._end]]

        return ', '.join(state.label for state in [begin, guard, body, end])

    def apply(self, sdfg):
        begin = sdfg.nodes()[self.subgraph[DoubleBuffering._begin]]
        guard = sdfg.nodes()[self.subgraph[DoubleBuffering._guard]]
        body = sdfg.nodes()[self.subgraph[DoubleBuffering._body]]
        end = sdfg.nodes()[self.subgraph[DoubleBuffering._end]]

        loop_vars = []
        for _, dst, e in sdfg.out_edges(body):
            if dst is guard:
                for var in e.assignments.keys():
                    loop_vars.append(var)

        if len(loop_vars) != 1:
            raise NotImplementedError()

        loop_var = loop_vars[0]
        sym_var = dace.symbolic.pystr_to_symbolic(loop_var)

        # Find source/sink (data) nodes
        input_nodes = nxutil.find_source_nodes(body)
        #output_nodes = nxutil.find_sink_nodes(body)

        copied_nodes = set()
        db_nodes = {}
        for node in input_nodes:
            for _, _, dst, _, mem in body.out_edges(node):
                if (isinstance(dst, dace.graph.nodes.AccessNode)
                        and loop_var in mem.subset.free_symbols):
                    # Create new data and nodes in guard
                    if node not in copied_nodes:
                        guard.add_node(node)
                        copied_nodes.add(node)
                    if dst not in copied_nodes:
                        old_data = dst.desc(sdfg)
                        if isinstance(old_data, dace.data.Array):
                            new_shape = tuple([2] + list(old_data.shape))
                            new_data = sdfg.add_array(
                                old_data.data,
                                old_data.dtype,
                                new_shape,
                                transient=True)
                        elif isinstance(old_data, data.Scalar):
                            new_data = sdfg.add_array(
                                old_data.data,
                                old_data.dtype, (2),
                                transient=True)
                        else:
                            raise NotImplementedError()
                        new_node = dace.graph.nodes.AccessNode(old_data.data)
                        guard.add_node(new_node)
                        copied_nodes.add(dst)
                        db_nodes.update({dst: new_node})
                    # Create memlet in guard
                    new_mem = copy.deepcopy(mem)
                    old_index = new_mem.other_subset
                    if isinstance(old_index, dace.subsets.Range):
                        new_ranges = [(0, 0, 1)] + old_index.ranges
                        new_mem.other_subset = dace.subsets.Range(new_ranges)
                    elif isinstance(old_index, dace.subsets.Indices):
                        new_indices = [0] + old_index.indices
                        new_mem.other_subset = dace.subsets.Indices(
                            new_indices)
                    guard.add_edge(node, None, new_node, None, new_mem)
                    # Create nodes, memlets in body
                    first_node = copy.deepcopy(new_node)
                    second_node = copy.deepcopy(new_node)
                    body.add_nodes_from([first_node, second_node])
                    dace.graph.nxutil.change_edge_dest(body, dst, first_node)
                    dace.graph.nxutil.change_edge_src(body, dst, second_node)
                    for src, _, dest, _, mem in body.edges():
                        if src is node and dest is first_node:
                            old_index = mem.other_subset
                            idx = (sym_var + 1) % 2
                            if isinstance(old_index, dace.subsets.Range):
                                new_ranges = [(idx, idx, 1)] + old_index.ranges
                            elif isinstance(old_index, dace.subsets.Indices):
                                new_ranges = [(idx, idx, 1)]
                                for index in old_index.indices:
                                    new_ranges.append((index, index, 1))
                            mem.other_subset = dace.subsets.Range(new_ranges)
                        elif mem.data == dst.data:
                            old_index = mem.subset
                            idx = sym_var % 2
                            if isinstance(old_index, dace.subsets.Range):
                                new_ranges = [(idx, idx, 1)] + old_index.ranges
                            elif isinstance(old_index, dace.subsets.Indices):
                                new_ranges = [(idx, idx, 1)]
                                for index in old_index.indices:
                                    new_ranges.append((index, index, 1))
                            mem.subset = dace.subsets.Range(new_ranges)
                            mem.data = first_node.data
                    body.remove_node(dst)


pattern_matching.Transformation.register_stateflow_pattern(DoubleBuffering)
