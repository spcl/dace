# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement transformations relating to streams
    and transient nodes. """

import copy
import warnings
from dace import data, dtypes, registry, symbolic, subsets
from dace.frontend.operations import detect_reduction_type
from dace.properties import make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


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


@registry.autoregister_params(singlestate=True)
@make_properties
class StreamTransient(transformation.Transformation):
    """ Implements the StreamTransient transformation, which adds a transient
        and stream nodes between nested maps that lead to a stream. The
        transient then acts as a local buffer.
    """

    with_buffer = Property(dtype=bool,
                           default=True,
                           desc="Use an intermediate buffer for accumulation")

    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _outer_map_exit = nodes.MapExit(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(StreamTransient._tasklet,
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

    def apply(self, sdfg: SDFG):
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
        newname, _ = sdfg.add_stream('trans_' + dataname,
                                     sdfg.arrays[memlet.data].dtype,
                                     1,
                                     bbox_approx[0], [1],
                                     transient=True,
                                     find_new_name=True)
        snode = graph.add_access(newname)

        to_stream_mm = copy.deepcopy(memlet)
        to_stream_mm.data = snode.data
        tasklet_memlet.data = snode.data

        if self.with_buffer:
            newname_arr, _ = sdfg.add_transient('strans_' + dataname,
                                                [bbox_approx[0]],
                                                sdfg.arrays[memlet.data].dtype,
                                                find_new_name=True)
            anode = graph.add_access(newname_arr)
            to_array_mm = copy.deepcopy(memlet)
            to_array_mm.data = anode.data
            graph.add_edge(snode, None, anode, None, to_array_mm)
        else:
            anode = snode

        # Reconnect, assuming one edge to the stream
        graph.remove_edge(edge)
        graph.add_edge(map_exit, edge.src_conn, snode, None, to_stream_mm)
        graph.add_edge(anode, None, outer_map_exit, edge.dst_conn, memlet)

        return


@registry.autoregister_params(singlestate=True)
@make_properties
class AccumulateTransient(transformation.Transformation):
    """ Implements the AccumulateTransient transformation, which adds
        transient stream and data nodes between nested maps that lead to a 
        stream. The transient data nodes then act as a local accumulator.
    """

    _tasklet = nodes.Tasklet('_')
    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _outer_map_exit = nodes.MapExit(nodes.Map("", [], []))

    array = Property(
        dtype=str,
        desc="Array to create local storage for (if empty, first available)",
        default=None,
        allow_none=True)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(AccumulateTransient._tasklet,
                                   AccumulateTransient._map_exit,
                                   AccumulateTransient._outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        tasklet = graph.nodes()[candidate[AccumulateTransient._tasklet]]
        map_exit = graph.nodes()[candidate[AccumulateTransient._map_exit]]

        # Check if there is an accumulation output
        for _src, _, dest, _, memlet in graph.out_edges(tasklet):
            if memlet.wcr is not None and dest == map_exit:
                return True

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[AccumulateTransient._tasklet]
        map_exit = candidate[AccumulateTransient._map_exit]
        outer_map_exit = candidate[AccumulateTransient._outer_map_exit]

        return ' -> '.join(
            str(node) for node in [tasklet, map_exit, outer_map_exit])

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)

        # Choose array
        array = self.array
        if array is None or len(array) == 0:
            map_exit = graph.node(self.subgraph[AccumulateTransient._map_exit])
            outer_map_exit = graph.node(
                self.subgraph[AccumulateTransient._outer_map_exit])
            array = next(e.data.data
                         for e in graph.edges_between(map_exit, outer_map_exit)
                         if e.data.wcr is not None)

        # Avoid import loop
        from dace.transformation.dataflow.local_storage import LocalStorage

        local_storage_subgraph = {
            LocalStorage._node_a: self.subgraph[AccumulateTransient._map_exit],
            LocalStorage._node_b:
            self.subgraph[AccumulateTransient._outer_map_exit]
        }
        sdfg_id = sdfg.sdfg_id
        in_local_storage = LocalStorage(sdfg_id, self.state_id,
                                        local_storage_subgraph, self.expr_index)
        in_local_storage.array = array
        in_local_storage.apply(sdfg)

        # Initialize transient to zero in case of summation
        # TODO: Initialize transient in other WCR types
        memlet = graph.in_edges(in_local_storage._data_node)[0].data
        if detect_reduction_type(memlet.wcr) == dtypes.ReductionType.Sum:
            in_local_storage._data_node.setzero = True
        else:
            warnings.warn('AccumulateTransient did not properly initialize'
                          'newly-created transient!')
