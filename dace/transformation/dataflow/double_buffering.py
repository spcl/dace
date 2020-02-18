"""Contains classes that implement the double buffering pattern. """

import copy
import itertools

from dace import data, dtypes, sdfg as sd, subsets, symbolic, registry
from dace.memlet import Memlet
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching

from dace.transformation.dataflow.map_for_loop import MapToForLoop


@registry.autoregister_params(singlestate=True)
class DoubleBuffering(pattern_matching.Transformation):
    """ Implements the double buffering pattern, which pipelines reading
        and processing data by creating a second copy of the memory.
        In particular, the transformation takes a 1D map and all internal
        (directly connected) transients, adds an additional dimension of size 2,
        and turns the map into a for loop that processes and reads the data in a
        double-buffered manner. Other memlets will not be transformed.
    """

    _map_entry = nodes.MapEntry(nodes.Map('_', [], []))
    _transient = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(DoubleBuffering._map_entry,
                                   DoubleBuffering._transient)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[DoubleBuffering._map_entry]]
        transient = graph.nodes()[candidate[DoubleBuffering._transient]]

        # Only one dimensional maps are allowed
        if len(map_entry.map.params) != 1:
            return False

        # Verify the map can be transformed to a for-loop
        if not MapToForLoop.can_be_applied(
                graph,
            {MapToForLoop._map_entry: candidate[DoubleBuffering._map_entry]},
                expr_index, sdfg, strict):
            return False

        # Verify that all directly-connected internal access nodes point to
        # transient arrays
        first = True
        for edge in graph.out_edges(map_entry):
            if isinstance(edge.dst, nodes.AccessNode):
                desc = sdfg.arrays[edge.dst.data]
                if not isinstance(desc, data.Array) or not desc.transient:
                    return False
                else:
                    # To avoid duplicate matches, only match the first transient
                    if first and edge.dst != transient:
                        return False
                    first = False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return str(graph.node(candidate[DoubleBuffering._map_entry]))

    def apply(self, sdfg: sd.SDFG):
        graph: sd.SDFGState = sdfg.nodes()[self.state_id]
        map_entry = graph.node(self.subgraph[DoubleBuffering._map_entry])

        map_param = map_entry.map.params[0]  # Assuming one dimensional

        ##############################
        # Change condition of loop to one fewer iteration (so that the
        # final one reads from the last buffer)
        map_rstart, map_rend, map_rstride = map_entry.map.range[0]
        map_rend = symbolic.pystr_to_symbolic(
            '(%s) - (%s)' % (map_rend, map_rstride))
        map_entry.map.range = subsets.Range([(map_rstart, map_rend,
                                              map_rstride)])

        ##############################
        # Gather transients to modify
        transients_to_modify = set(
            edge.dst.data for edge in graph.out_edges(map_entry)
            if isinstance(edge.dst, nodes.AccessNode))

        # Add dimension to transients and modify memlets
        for transient in transients_to_modify:
            desc: data.Array = sdfg.arrays[transient]
            # Using non-python syntax to ensure properties change
            desc.strides = [desc.total_size] + list(desc.strides)
            desc.shape = [2] + list(desc.shape)
            desc.offset = [0] + list(desc.offset)
            desc.total_size = desc.total_size * 2

        ##############################
        # Modify memlets to use map parameter as buffer index
        modified_subsets = []  # Store modified memlets for final state
        for edge in graph.scope_subgraph(map_entry).edges():
            if edge.data.data in transients_to_modify:
                edge.data.subset = self._modify_memlet(sdfg, edge.data.subset,
                                                       edge.data.data)
                modified_subsets.append(edge.data.subset)
            else:  # Could be other_subset
                path = graph.memlet_path(edge)
                src_node = path[0].src
                dst_node = path[-1].dst

                # other_subset could be None. In that case, recreate from array
                dataname = None
                if (isinstance(src_node, nodes.AccessNode)
                        and src_node.data in transients_to_modify):
                    dataname = src_node.data
                elif (isinstance(dst_node, nodes.AccessNode)
                      and dst_node.data in transients_to_modify):
                    dataname = dst_node.data
                if dataname is not None:
                    subset = (edge.data.other_subset or
                              subsets.Range.from_array(sdfg.arrays[dataname]))
                    edge.data.other_subset = self._modify_memlet(
                        sdfg, subset, dataname)
                    modified_subsets.append(edge.data.other_subset)

        ##############################
        # Turn map into for loop
        map_to_for = MapToForLoop(self.sdfg_id, self.state_id, {
            MapToForLoop._map_entry:
            self.subgraph[DoubleBuffering._map_entry]
        }, self.expr_index)
        nsdfg_node, nstate = map_to_for.apply(sdfg)

        ##############################
        # Gather node copies and remove memlets
        edges_to_replace = []
        for node in nstate.source_nodes():
            for edge in nstate.out_edges(node):
                if (isinstance(edge.dst, nodes.AccessNode)
                        and edge.dst.data in transients_to_modify):
                    edges_to_replace.append(edge)
                    nstate.remove_edge(edge)
            if nstate.out_degree(node) == 0:
                nstate.remove_node(node)

        ##############################
        # Add initial reads to initial nested state
        initial_state: sd.SDFGState = nsdfg_node.sdfg.start_state
        initial_state.set_label('%s_init' % map_entry.map.label)
        for edge in edges_to_replace:
            initial_state.add_node(edge.src)
            rnode = edge.src
            wnode = initial_state.add_write(edge.dst.data)
            initial_state.add_edge(rnode, edge.src_conn, wnode, edge.dst_conn,
                                   copy.deepcopy(edge.data))

        # All instances of the map parameter in this state become the loop start
        sd.replace(initial_state, map_param, map_rstart)
        # Initial writes go to the first buffer
        sd.replace(initial_state, '__dace_db_param', '0')

        ##############################
        # Modify main state's memlets

        # Divide by loop stride
        new_expr = symbolic.pystr_to_symbolic(
            '(%s / %s) %% 2' % (map_param, map_rstride))
        sd.replace(nstate, '__dace_db_param', new_expr)

        ##############################
        # Add the main state's contents to the last state, modifying
        # memlets appropriately.
        final_state: sd.SDFGState = nsdfg_node.sdfg.sink_nodes()[0]
        final_state.set_label('%s_final_computation' % map_entry.map.label)
        dup_nstate = copy.deepcopy(nstate)
        final_state.add_nodes_from(dup_nstate.nodes())
        for e in dup_nstate.edges():
            final_state.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

        ##############################
        # Add reads into next buffers to main state
        for edge in edges_to_replace:
            rnode = copy.deepcopy(edge.src)
            nstate.add_node(rnode)
            wnode = nstate.add_write(edge.dst.data)
            new_memlet = copy.deepcopy(edge.data)
            if new_memlet.data in transients_to_modify:
                new_memlet.other_subset = self._replace_in_subset(
                    new_memlet.other_subset, map_param,
                    '(%s + %s)' % (map_param, map_rstride))
            else:
                new_memlet.subset = self._replace_in_subset(
                    new_memlet.subset, map_param,
                    '(%s + %s)' % (map_param, map_rstride))

            nstate.add_edge(rnode, edge.src_conn, wnode, edge.dst_conn,
                            new_memlet)

        nstate.set_label('%s_double_buffered' % map_entry.map.label)
        # Divide by loop stride
        new_expr = symbolic.pystr_to_symbolic(
            '((%s / %s) + 1) %% 2' % (map_param, map_rstride))
        sd.replace(nstate, '__dace_db_param', new_expr)

    @staticmethod
    def _modify_memlet(sdfg, subset, data_name):
        desc = sdfg.arrays[data_name]
        if len(subset) == len(desc.shape):
            # Already in the right shape, modify new dimension
            subset = list(subset)[1:]

        new_subset = subsets.Range([('__dace_db_param', '__dace_db_param',
                                     1)] + list(subset))
        return new_subset

    @staticmethod
    def _replace_in_subset(subset, string_or_symbol, new_string_or_symbol):
        new_subset = copy.deepcopy(subset)

        repldict = {
            symbolic.pystr_to_symbolic(string_or_symbol):
            symbolic.pystr_to_symbolic(new_string_or_symbol)
        }

        for i, dim in enumerate(new_subset):
            try:
                new_subset[i] = tuple(d.subs(repldict) for d in dim)
            except TypeError:
                new_subset[i] = (dim.subs(repldict)
                                 if symbolic.issymbolic(dim) else dim)

        return new_subset
