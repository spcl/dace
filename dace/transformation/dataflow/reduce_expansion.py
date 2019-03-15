""" Contains classes that implement the reduce-expansion transformation. """

from copy import deepcopy as dcpy
from dace import sdfg, subsets, types, symbolic
from dace.graph import nodes, nxutil
from dace.graph.graph import OrderedMultiDiGraph
from dace.transformation import pattern_matching as pm


class ReduceExpansion(pm.Transformation):
    """ Implements the reduce-expansion transformation.

        Reduce-expansion replaces a reduce node with nested maps and edges with
        WCR.
    """

    _reduce = nodes.Reduce(wcr='lambda x: x', axes=None)

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(ReduceExpansion._reduce)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        red_node = graph.nodes()[candidate[ReduceExpansion._reduce]]
        return "{}: {} on {}".format(red_node, red_node.wcr, red_node.axes)

    def apply(self, sdfg):
        """ The method creates two nested maps. The inner map ranges over the
            reduction axes, while the outer map ranges over the rest of the 
            input dimensions. The inner map contains a trivial tasklet, while
            the outgoing edges copy the reduction WCR.
        """
        graph = sdfg.nodes()[self.state_id]
        red_node = graph.nodes()[self.subgraph[ReduceExpansion._reduce]]

        inputs = []
        in_memlets = []
        for src, _, _, _, memlet in graph.in_edges(red_node):
            if src not in inputs:
                inputs.append(src)
                in_memlets.append(memlet)
        if len(inputs) > 1:
            raise NotImplementedError

        outputs = []
        out_memlets = []
        for _, _, dst, _, memlet in graph.out_edges(red_node):
            if dst not in outputs:
                outputs.append(dst)
                out_memlets.append(memlet)
        if len(outputs) > 1:
            raise NotImplementedError

        axes = red_node.axes
        if axes is None:
            axes = tuple(i for i in range(in_memlets[0].subset.dims()))

        outer_map_range = {}
        inner_map_range = {}
        for idx, r in enumerate(in_memlets[0].subset):
            if idx in axes:
                inner_map_range.update({
                    "__dim_{}".format(str(idx)):
                    subsets.Range.dim_to_string(r)
                })
            else:
                outer_map_range.update({
                    "__dim_{}".format(str(idx)):
                    subsets.Range.dim_to_string(r)
                })

        if len(outer_map_range) > 0:
            outer_map_entry, outer_map_exit = graph.add_map(
                'reduce_outer', outer_map_range, schedule=red_node.schedule)

        inner_map_entry, inner_map_exit = graph.add_map(
            'reduce_inner',
            inner_map_range,
            schedule=(types.ScheduleType.Default
                      if len(outer_map_range) > 0 else red_node.schedule))

        tasklet = graph.add_tasklet(
            name='red_tasklet',
            inputs={'in_1'},
            outputs={'out_1'},
            code='out_1 = in_1')

        inner_map_entry.in_connectors = {'IN_1'}
        inner_map_entry.out_connectors = {'OUT_1'}

        outer_in_memlet = dcpy(in_memlets[0])

        if len(outer_map_range) > 0:
            outer_map_entry.in_connectors = {'IN_1'}
            outer_map_entry.out_connectors = {'OUT_1'}
            graph.add_edge(inputs[0], None, outer_map_entry, 'IN_1',
                           outer_in_memlet)
        else:
            graph.add_edge(inputs[0], None, inner_map_entry, 'IN_1',
                           outer_in_memlet)

        med_in_memlet = dcpy(in_memlets[0])
        med_in_range = []
        for idx, r in enumerate(med_in_memlet.subset):
            if idx in axes:
                med_in_range.append(r)
            else:
                med_in_range.append(("__dim_{}".format(str(idx)),
                                     "__dim_{}".format(str(idx)), 1))
        med_in_memlet.subset = subsets.Range(med_in_range)
        med_in_memlet.num_accesses = med_in_memlet.subset.num_elements()

        if len(outer_map_range) > 0:
            graph.add_edge(outer_map_entry, 'OUT_1', inner_map_entry, 'IN_1',
                           med_in_memlet)

        inner_in_memlet = dcpy(med_in_memlet)
        inner_in_idx = []
        for idx in range(len(inner_in_memlet.subset)):
            inner_in_idx.append("__dim_{}".format(str(idx)))
        inner_in_memlet.subset = subsets.Indices(inner_in_idx)
        inner_in_memlet.num_accesses = inner_in_memlet.subset.num_elements()
        graph.add_edge(inner_map_entry, 'OUT_1', tasklet, 'in_1',
                       inner_in_memlet)
        inner_map_exit.in_connectors = {'IN_1'}
        inner_map_exit.out_connectors = {'OUT_1'}

        inner_out_memlet = dcpy(out_memlets[0])
        inner_out_idx = []
        for idx, r in enumerate(inner_in_memlet.subset):
            if idx not in axes:
                inner_out_idx.append(r)
        if len(inner_out_idx) == 0:
            inner_out_idx = [0]

        inner_out_memlet.subset = subsets.Indices(inner_out_idx)
        inner_out_memlet.wcr = red_node.wcr
        inner_out_memlet.num_accesses = inner_out_memlet.subset.num_elements()
        graph.add_edge(tasklet, 'out_1', inner_map_exit, 'IN_1',
                       inner_out_memlet)

        outer_out_memlet = dcpy(out_memlets[0])
        outer_out_range = []
        for idx, r in enumerate(outer_out_memlet.subset):
            if idx not in axes:
                outer_out_range.append(r)
        if len(outer_out_range) == 0:
            outer_out_range = [(0, 0, 1)]

        outer_out_memlet.subset = subsets.Range(outer_out_range)
        outer_out_memlet.wcr = red_node.wcr

        if len(outer_map_range) > 0:
            outer_map_exit.in_connectors = {'IN_1'}
            outer_map_exit.out_connectors = {'OUT_1'}
            med_out_memlet = dcpy(inner_out_memlet)
            med_out_memlet.num_accesses = med_out_memlet.subset.num_elements()
            graph.add_edge(inner_map_exit, 'OUT_1', outer_map_exit, 'IN_1',
                           med_out_memlet)

            graph.add_edge(outer_map_exit, 'OUT_1', outputs[0], None,
                           outer_out_memlet)
        else:
            graph.add_edge(inner_map_exit, 'OUT_1', outputs[0], None,
                           outer_out_memlet)

        graph.remove_edge(graph.in_edges(red_node)[0])
        graph.remove_edge(graph.out_edges(red_node)[0])
        graph.remove_node(red_node)

        return


pm.Transformation.register_pattern(ReduceExpansion)
