# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement transformations relating to streams
    and transient nodes. """

import copy
from dace.symbolic import symstr
import warnings

from numpy.core.numeric import outer
from dace import data, dtypes, registry, symbolic, subsets
from dace.frontend.operations import detect_reduction_type
from dace.properties import SymbolicProperty, make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
import dace
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import SDFGState
from dace.sdfg.nodes import NestedSDFG
from dace.data import Array


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

    tasklet = transformation.PatternNode(nodes.Tasklet)
    map_exit = transformation.PatternNode(nodes.MapExit)
    outer_map_exit = transformation.PatternNode(nodes.MapExit)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(StreamTransient.tasklet,
                                   StreamTransient.map_exit,
                                   StreamTransient.outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_exit = graph.nodes()[candidate[StreamTransient.map_exit]]
        outer_map_exit = graph.nodes()[candidate[
            StreamTransient.outer_map_exit]]

        # Check if there is a streaming output
        for _src, _, dest, _, memlet in graph.out_edges(map_exit):
            if isinstance(sdfg.arrays[memlet.data],
                          data.Stream) and dest == outer_map_exit:
                return True

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[StreamTransient.tasklet]
        map_exit = candidate[StreamTransient.map_exit]
        outer_map_exit = candidate[StreamTransient.outer_map_exit]

        return ' -> '.join(
            str(node) for node in [tasklet, map_exit, outer_map_exit])

    def apply(self, sdfg: SDFG):
        graph = sdfg.nodes()[self.state_id]
        tasklet = graph.nodes()[self.subgraph[StreamTransient.tasklet]]
        map_exit = graph.nodes()[self.subgraph[StreamTransient.map_exit]]
        outer_map_exit = graph.nodes()[self.subgraph[
            StreamTransient.outer_map_exit]]
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
                                     bbox_approx[0],
                                     storage=sdfg.arrays[memlet.data].storage,
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

    map_exit = transformation.PatternNode(nodes.MapExit)
    outer_map_exit = transformation.PatternNode(nodes.MapExit)

    array = Property(
        dtype=str,
        desc="Array to create local storage for (if empty, first available)",
        default=None,
        allow_none=True)

    identity = SymbolicProperty(desc="Identity value to set",
                                default=None,
                                allow_none=True)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(AccumulateTransient.map_exit,
                                   AccumulateTransient.outer_map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_exit = graph.nodes()[candidate[AccumulateTransient.map_exit]]
        outer_map_exit = graph.nodes()[candidate[
            AccumulateTransient.outer_map_exit]]

        # Check if there is an accumulation output
        for e in graph.edges_between(map_exit, outer_map_exit):
            if e.data.wcr is not None:
                return True

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        map_exit = candidate[AccumulateTransient.map_exit]
        outer_map_exit = candidate[AccumulateTransient.outer_map_exit]

        return ' -> '.join(str(node) for node in [map_exit, outer_map_exit])

    def apply(self, sdfg: SDFG):
        graph = sdfg.node(self.state_id)
        map_exit = graph.node(self.subgraph[AccumulateTransient.map_exit])
        outer_map_exit = graph.node(
            self.subgraph[AccumulateTransient.outer_map_exit])

        # Choose array
        array = self.array
        if array is None or len(array) == 0:
            array = next(e.data.data
                         for e in graph.edges_between(map_exit, outer_map_exit)
                         if e.data.wcr is not None)

        # Avoid import loop
        from dace.transformation.dataflow.local_storage import OutLocalStorage

        data_node: nodes.AccessNode = OutLocalStorage.apply_to(
            sdfg,
            dict(array=array),
            verify=False,
            save=False,
            node_a=map_exit,
            node_b=outer_map_exit)

        if self.identity is None:
            warnings.warn('AccumulateTransient did not properly initialize '
                          'newly-created transient!')
            return

        sdfg_state: SDFGState = sdfg.node(self.state_id)

        map_entry = sdfg_state.entry_node(map_exit)

        nested_sdfg: NestedSDFG = nest_state_subgraph(
            sdfg=sdfg,
            state=sdfg_state,
            subgraph=SubgraphView(
                sdfg_state, {map_entry, map_exit}
                | sdfg_state.all_nodes_between(map_entry, map_exit)))

        nested_sdfg_state: SDFGState = nested_sdfg.sdfg.nodes()[0]

        init_state = nested_sdfg.sdfg.add_state_before(nested_sdfg_state)

        temp_array: Array = sdfg.arrays[data_node.data]

        init_state.add_mapped_tasklet(
            name='acctrans_init',
            map_ranges={
                '_o%d' % i: '0:%s' % symstr(d)
                for i, d in enumerate(temp_array.shape)
            },
            inputs={},
            code='out = %s' % self.identity,
            outputs={
                'out':
                dace.Memlet.simple(data=data_node.data,
                                   subset_str=','.join([
                                       '_o%d' % i
                                       for i, _ in enumerate(temp_array.shape)
                                   ]))
            },
            external_edges=True)

        # TODO: use trivial map elimintation here when it will be merged to remove map if it has trivial ranges
