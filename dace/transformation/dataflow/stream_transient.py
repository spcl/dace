""" Contains classes that implement transformations relating to streams
    and transient nodes. """
import copy
import networkx as nx
from dace import data, types, symbolic, subsets
from dace.properties import Property, make_properties, DataProperty
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


def calc_set_image_index(map_idx, map_set, array_idx):
    image = []
    for a_idx in array_idx.indices:
        new_range = [a_idx, a_idx, 1]
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range[0] = new_range[0].subs(symbol, m_range[0])
            new_range[1] = new_range[1].subs(symbol, m_range[1])
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image_range(map_idx, map_set, array_range):
    image = []
    for a_range in array_range:
        new_range = a_range
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range = [
                new_range[i].subs(symbol, m_range[i]) for i in range(0, 3)
            ]
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image(map_idx, map_set, array_set):
    if isinstance(array_set, subsets.Range):
        return calc_set_image_range(map_idx, map_set, array_set)
    if isinstance(array_set, subsets.Indices):
        return calc_set_image_index(map_idx, map_set, array_set)


@make_properties
class StreamTransient(pattern_matching.Transformation):
    """ Implements the StreamTransient transformation, which adds a transient
        stream node between nested maps that lead to a stream. The transient
        then acts as a local buffer.
    """

    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _outer_map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(StreamTransient._tasklet,
                                   StreamTransient._map_exit,
                                   StreamTransient._outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_exit = graph.nodes()[candidate[StreamTransient._map_exit]]
        outer_map_exit = graph.nodes()[candidate[
            StreamTransient._outer_map_exit]]

        # Check if there is a streaming output
        for _src, _, dest, _, memlet in graph.out_edges(map_exit):
            if isinstance(sdfg.arrays[memlet.data],
                          data.Stream) and dest == outer_map_exit:
                return True

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[StreamTransient._tasklet]
        map_exit = candidate[StreamTransient._map_exit]
        outer_map_exit = candidate[StreamTransient._outer_map_exit]

        return ' -> '.join(
            str(node) for node in [tasklet, map_exit, outer_map_exit])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        tasklet = graph.nodes()[self.subgraph[StreamTransient._tasklet]]
        map_exit = graph.nodes()[self.subgraph[StreamTransient._map_exit]]
        outer_map_exit = graph.nodes()[self.subgraph[
            StreamTransient._outer_map_exit]]
        memlet = None
        edge = None
        for e in graph.out_edges(map_exit):
            memlet = e.data
            # TODO: What if there's more than one?
            if e.dst == outer_map_exit and isinstance(sdfg.arrays[memlet.data],
                                                      data.Stream):
                edge = e
                break
        tasklet_memlet = None
        for e in graph.out_edges(tasklet):
            tasklet_memlet = e.data
            if tasklet_memlet.data == memlet.data:
                break

        bbox = map_exit.map.range.bounding_box_size()
        bbox_approx = [symbolic.overapproximate(dim) for dim in bbox]
        dataname = memlet.data

        # Create the new node: Temporary stream and an access node
        newstream = sdfg.add_stream(
            'tile_' + dataname,
            sdfg.arrays[memlet.data].dtype,
            1,
            bbox_approx[0],
            [1],
            transient=True,
        )
        snode = nodes.AccessNode('tile_' + dataname)

        to_stream_mm = copy.deepcopy(memlet)
        to_stream_mm.data = snode.data
        tasklet_memlet.data = snode.data

        # Reconnect, assuming one edge to the stream
        graph.remove_edge(edge)
        graph.add_edge(map_exit, None, snode, None, to_stream_mm)
        graph.add_edge(snode, None, outer_map_exit, None, memlet)

        return

    def modifies_graph(self):
        return True


pattern_matching.Transformation.register_pattern(StreamTransient)


@make_properties
class AccumulateTransient(pattern_matching.Transformation):
    """ Implements the AccumulateTransient transformation, which adds
        transient stream and data nodes between nested maps that lead to a 
        stream. The transient data nodes then act as a local accumulator.
    """

    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _outer_map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(StreamTransient._tasklet,
                                   StreamTransient._map_exit,
                                   StreamTransient._outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        tasklet = graph.nodes()[candidate[StreamTransient._tasklet]]
        map_exit = graph.nodes()[candidate[StreamTransient._map_exit]]

        # Check if there is a streaming output
        for _src, _, dest, _, memlet in graph.out_edges(tasklet):
            if memlet.wcr is not None and dest == map_exit:
                return True

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[StreamTransient._tasklet]
        map_exit = candidate[StreamTransient._map_exit]
        outer_map_exit = candidate[StreamTransient._outer_map_exit]

        return ' -> '.join(
            str(node) for node in [tasklet, map_exit, outer_map_exit])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        tasklet = graph.nodes()[self.subgraph[StreamTransient._tasklet]]
        map_exit = graph.nodes()[self.subgraph[StreamTransient._map_exit]]
        outer_map_exit = graph.nodes()[self.subgraph[
            StreamTransient._outer_map_exit]]
        memlet = None
        edge = None
        for e in graph.out_edges(tasklet):
            memlet = e.data
            # TODO: What if there's more than one?
            if e.dst == map_exit and e.data.wcr is not None:
                break
        out_memlet = None
        for e in graph.out_edges(map_exit):
            out_memlet = e.data
            if out_memlet.data == memlet.data:
                edge = e
                break
        dataname = memlet.data

        # Create a new node with the same size as the output
        newdata = sdfg.add_array(
            'trans_' + dataname,
            sdfg.arrays[memlet.data].shape,
            sdfg.arrays[memlet.data].dtype,
            transient=True)
        dnode = nodes.AccessNode('trans_' + dataname)

        to_data_mm = copy.deepcopy(memlet)
        to_data_mm.data = dnode.data
        to_data_mm.num_accesses = memlet.num_elements()

        to_exit_mm = copy.deepcopy(out_memlet)
        to_exit_mm.num_accesses = out_memlet.num_elements()
        memlet.data = dnode.data

        # Reconnect, assuming one edge to the stream
        graph.remove_edge(edge)
        graph.add_edge(map_exit, edge.src_conn, dnode, None, to_data_mm)
        graph.add_edge(dnode, None, outer_map_exit, edge.dst_conn, to_exit_mm)

        return

    def modifies_graph(self):
        return True


pattern_matching.Transformation.register_pattern(AccumulateTransient)


@make_properties
class OutLocalStorage(pattern_matching.Transformation):
    """ Implements the OutLocalStorage transformation, which adds a transient
        data node between nested map exits.
    """

    _inner_map_exit = nodes.MapExit(nodes.Map("", [], []))
    _outer_map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(  #OutLocalStorage._tasklet,
                OutLocalStorage._inner_map_exit,
                OutLocalStorage._outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        inner_map_exit = candidate[OutLocalStorage._inner_map_exit]
        outer_map_exit = candidate[OutLocalStorage._outer_map_exit]

        return ' -> '.join(
            str(node) for node in [inner_map_exit, outer_map_exit])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        inner_map_exit = graph.nodes()[self.subgraph[
            OutLocalStorage._inner_map_exit]]
        outer_map_exit = graph.nodes()[self.subgraph[
            OutLocalStorage._outer_map_exit]]

        original_edge = None
        invariant_memlet = None
        array = None
        for edge in graph.in_edges(outer_map_exit):
            src = edge.src
            if src != inner_map_exit:
                continue
            memlet = edge.data
            original_edge = edge
            invariant_memlet = memlet
            array = memlet.data
            break

        new_data = sdfg.add_array(
            graph.label + '_trans_' + invariant_memlet.data, [
                symbolic.overapproximate(r)
                for r in invariant_memlet.bounding_box_size()
            ],
            sdfg.arrays[invariant_memlet.data].dtype,
            transient=True)
        data_node = nodes.AccessNode(graph.label + '_trans_' +
                                     invariant_memlet.data)
        data_node.setzero = True

        from_data_mm = copy.deepcopy(invariant_memlet)
        to_data_mm = copy.deepcopy(invariant_memlet)
        to_data_mm.data = data_node.data
        offset = []
        for ind, r in enumerate(invariant_memlet.subset):
            offset.append(r[0])
            if isinstance(invariant_memlet.subset[ind], tuple):
                begin = invariant_memlet.subset[ind][0] - r[0]
                end = invariant_memlet.subset[ind][1] - r[0]
                step = invariant_memlet.subset[ind][2]
                to_data_mm.subset[ind] = (begin, end, step)
            else:
                to_data_mm.subset[ind] -= r[0]

        # Reconnect, assuming one edge to the stream
        graph.remove_edge(original_edge)
        graph.add_edge(inner_map_exit, original_edge.src_conn, data_node, None,
                       to_data_mm)
        graph.add_edge(data_node, None, outer_map_exit, original_edge.dst_conn,
                       from_data_mm)

        for _parent, _, _child, _, memlet in graph.bfs_edges(
                inner_map_exit, reverse=True):
            if isinstance(_child, nodes.CodeNode):
                break
            if memlet.data != array:
                continue
            for ind, r in enumerate(memlet.subset):
                if isinstance(memlet.subset[ind], tuple):
                    begin = r[0] - offset[ind]
                    end = r[1] - offset[ind]
                    step = r[2]
                    memlet.subset[ind] = (begin, end, step)
                else:
                    memlet.subset[ind] -= offset[ind]
            memlet.data = graph.label + '_trans_' + invariant_memlet.data

        return


pattern_matching.Transformation.register_pattern(OutLocalStorage)


@make_properties
class InLocalStorage(pattern_matching.Transformation):
    """ Implements the InLocalStorage transformation, which adds a transient
        data node between nested map entry nodes.
    """

    _outer_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _inner_map_entry = nodes.MapEntry(nodes.Map("", [], []))

    array = DataProperty(
        desc="Array to create local storage for", default="gpu_V")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(InLocalStorage._outer_map_entry,
                                   InLocalStorage._inner_map_entry)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        outer_map_entry = candidate[InLocalStorage._outer_map_entry]
        inner_map_entry = candidate[InLocalStorage._inner_map_entry]

        return ' -> '.join(
            str(node) for node in [outer_map_entry, inner_map_entry])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        outer_map_entry = graph.nodes()[self.subgraph[
            InLocalStorage._outer_map_entry]]
        inner_map_entry = graph.nodes()[self.subgraph[
            InLocalStorage._inner_map_entry]]

        original_edge = None
        invariant_memlet = None
        for edge in graph.in_edges(inner_map_entry):
            src = edge.src
            if src != outer_map_entry:
                continue
            memlet = edge.data
            if self.array == memlet.data:
                original_edge = edge
                invariant_memlet = memlet
                break
        if invariant_memlet is None:
            for edge in graph.in_edges(inner_map_entry):
                src = edge.src
                if src != outer_map_entry:
                    continue
                original_edge = edge
                invariant_memlet = edge.data
                print('WARNING: Array %s not found! Using array %s instead.' %
                      (self.array, invariant_memlet.data))
                self.array = invariant_memlet.data
                break
        if invariant_memlet is None:
            raise KeyError('Array %s not found!' % self.array)

        new_data = sdfg.add_array(
            'trans_' + invariant_memlet.data, [
                symbolic.overapproximate(r)
                for r in invariant_memlet.bounding_box_size()
            ],
            sdfg.arrays[invariant_memlet.data].dtype,
            transient=True)
        data_node = nodes.AccessNode('trans_' + invariant_memlet.data)

        to_data_mm = copy.deepcopy(invariant_memlet)
        from_data_mm = copy.deepcopy(invariant_memlet)
        from_data_mm.data = data_node.data
        offset = []
        for ind, r in enumerate(invariant_memlet.subset):
            offset.append(r[0])
            if isinstance(invariant_memlet.subset[ind], tuple):
                begin = invariant_memlet.subset[ind][0] - r[0]
                end = invariant_memlet.subset[ind][1] - r[0]
                step = invariant_memlet.subset[ind][2]
                from_data_mm.subset[ind] = (begin, end, step)
            else:
                from_data_mm.subset[ind] -= r[0]
        to_data_mm.other_subset = copy.deepcopy(from_data_mm.subset)

        # Reconnect, assuming one edge to the stream
        graph.remove_edge(original_edge)
        graph.add_edge(outer_map_entry, original_edge.src_conn, data_node,
                       None, to_data_mm)
        graph.add_edge(data_node, None, inner_map_entry,
                       original_edge.dst_conn, from_data_mm)

        for _parent, _, _child, _, memlet in graph.bfs_edges(
                inner_map_entry, reverse=False):
            if memlet.data != self.array:
                continue
            for ind, r in enumerate(memlet.subset):
                if isinstance(memlet.subset[ind], tuple):
                    begin = r[0] - offset[ind]
                    end = r[1] - offset[ind]
                    step = r[2]
                    memlet.subset[ind] = (begin, end, step)
                else:
                    memlet.subset[ind] -= offset[ind]
            memlet.data = 'trans_' + invariant_memlet.data

        return


pattern_matching.Transformation.register_pattern(InLocalStorage)
