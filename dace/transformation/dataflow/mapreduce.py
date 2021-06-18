# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes and functions that implement the map-reduce-fusion 
    transformation. """

from dace import registry
from dace.sdfg import SDFG, SDFGState
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.properties import Property, make_properties
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.symbolic import symstr
from dace.transformation import transformation as pm

from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_fusion import MapFusion


@registry.autoregister_params(singlestate=True)
@make_properties
class MapReduceFusion(pm.Transformation):
    """ Implements the map-reduce-fusion transformation.
        Fuses a map with an immediately following reduction, where the array
        between the map and the reduction is not used anywhere else.
    """

    no_init = Property(dtype=bool,
                       default=False,
                       desc='If enabled, does not create initialization states '
                       'for reduce nodes with identity')

    _tasklet = nodes.Tasklet('_')
    _tmap_exit = nodes.MapExit(nodes.Map("", [], []))
    _in_array = nodes.AccessNode('_')

    import dace.libraries.standard as stdlib  # Avoid import loop
    _reduce = stdlib.Reduce()

    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(MapReduceFusion._tasklet,
                                   MapReduceFusion._tmap_exit,
                                   MapReduceFusion._in_array,
                                   MapReduceFusion._reduce,
                                   MapReduceFusion._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        tmap_exit = graph.nodes()[candidate[MapReduceFusion._tmap_exit]]
        in_array = graph.nodes()[candidate[MapReduceFusion._in_array]]
        reduce_node = graph.nodes()[candidate[MapReduceFusion._reduce]]
        tasklet = graph.nodes()[candidate[MapReduceFusion._tasklet]]

        # Make sure that the array is only accessed by the map and the reduce
        if any([
                src != tmap_exit
                for src, _, _, _, memlet in graph.in_edges(in_array)
        ]):
            return False
        if any([
                dest != reduce_node
                for _, _, dest, _, memlet in graph.out_edges(in_array)
        ]):
            return False

        tmem = next(e for e in graph.edges_between(tasklet, tmap_exit)
                    if e.data.data == in_array.data).data

        # (strict) Make sure that the transient is not accessed anywhere else
        # in this state or other states
        if strict and (len([
                n for n in graph.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == in_array.data
        ]) > 1 or in_array.data in sdfg.shared_transients()):
            return False

        # If memlet already has WCR and it is different from reduce node,
        # do not match
        if tmem.wcr is not None and tmem.wcr != reduce_node.wcr:
            return False

        # Verify that reduction ranges match tasklet map
        tout_memlet = graph.in_edges(in_array)[0].data
        rin_memlet = graph.out_edges(in_array)[0].data
        if tout_memlet.subset != rin_memlet.subset:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[MapReduceFusion._tasklet]
        map_exit = candidate[MapReduceFusion._tmap_exit]
        reduce = candidate[MapReduceFusion._reduce]

        return ' -> '.join(str(node) for node in [tasklet, map_exit, reduce])

    def apply(self, sdfg: SDFG):
        graph = sdfg.nodes()[self.state_id]
        tmap_exit = graph.nodes()[self.subgraph[MapReduceFusion._tmap_exit]]
        in_array = graph.nodes()[self.subgraph[MapReduceFusion._in_array]]
        reduce_node = graph.nodes()[self.subgraph[MapReduceFusion._reduce]]
        out_array = graph.nodes()[self.subgraph[MapReduceFusion._out_array]]

        # Set nodes to remove according to the expression index
        nodes_to_remove = [in_array]
        nodes_to_remove.append(reduce_node)

        memlet_edge = None
        for edge in graph.in_edges(tmap_exit):
            if edge.data.data == in_array.data:
                memlet_edge = edge
                break
        if memlet_edge is None:
            raise RuntimeError('Reduction memlet cannot be None')

        # Find which indices should be removed from new memlet
        input_edge = graph.in_edges(reduce_node)[0]
        axes = reduce_node.axes or list(range(len(input_edge.data.subset)))
        array_edge = graph.out_edges(reduce_node)[0]

        # Delete relevant edges and nodes
        graph.remove_nodes_from(nodes_to_remove)

        # Delete relevant data descriptors
        for node in set(nodes_to_remove):
            if isinstance(node, nodes.AccessNode):
                # try to delete it
                try:
                    sdfg.remove_data(node.data)
                # will raise ValueError if the datadesc is used somewhere else
                except ValueError:
                    pass

        # Filter out reduced dimensions from subset
        filtered_subset = [
            dim for i, dim in enumerate(memlet_edge.data.subset)
            if i not in axes
        ]
        if len(filtered_subset) == 0:  # Output is a scalar
            filtered_subset = [(0, 0, 1)]

        # Modify edge from tasklet to map exit
        memlet_edge.data.data = out_array.data
        memlet_edge.data.wcr = reduce_node.wcr
        memlet_edge.data.subset = type(memlet_edge.data.subset)(filtered_subset)

        # Add edge from map exit to output array
        graph.add_edge(
            memlet_edge.dst, 'OUT_' + memlet_edge.dst_conn[3:], array_edge.dst,
            array_edge.dst_conn,
            Memlet.simple(array_edge.data.data,
                          array_edge.data.subset,
                          num_accesses=array_edge.data.num_accesses,
                          wcr_str=reduce_node.wcr))

        # Add initialization state as necessary
        if not self.no_init and reduce_node.identity is not None:
            init_state = sdfg.add_state_before(graph)
            init_state.add_mapped_tasklet(
                'freduce_init',
                [('o%d' % i, '%s:%s:%s' % (r[0], r[1] + 1, r[2]))
                 for i, r in enumerate(array_edge.data.subset)], {},
                'out = %s' % reduce_node.identity, {
                    'out':
                    Memlet.simple(
                        array_edge.data.data, ','.join([
                            'o%d' % i
                            for i in range(len(array_edge.data.subset))
                        ]))
                },
                external_edges=True)


@registry.autoregister_params(singlestate=True)
class MapWCRFusion(pm.Transformation):
    """ Implements the map expanded-reduce fusion transformation.
        Fuses a map with an immediately following reduction, where the array
        between the map and the reduction is not used anywhere else, and the
        reduction is divided to two maps with a WCR, denoting partial reduction.
    """

    _tasklet = nodes.Tasklet('_')
    _tmap_exit = nodes.MapExit(nodes.Map("", [], []))
    _in_array = nodes.AccessNode('_')
    _rmap_in_entry = nodes.MapEntry(nodes.Map("", [], []))
    _rmap_in_tasklet = nodes.Tasklet('_')
    _rmap_in_cr = nodes.MapExit(nodes.Map("", [], []))
    _rmap_out_entry = nodes.MapEntry(nodes.Map("", [], []))
    _rmap_out_exit = nodes.MapExit(nodes.Map("", [], []))
    _out_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            # Map, then partial reduction of axes
            sdutil.node_path_graph(
                MapWCRFusion._tasklet, MapWCRFusion._tmap_exit,
                MapWCRFusion._in_array, MapWCRFusion._rmap_out_entry,
                MapWCRFusion._rmap_in_entry, MapWCRFusion._rmap_in_tasklet,
                MapWCRFusion._rmap_in_cr, MapWCRFusion._rmap_out_exit,
                MapWCRFusion._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        tmap_exit = graph.nodes()[candidate[MapWCRFusion._tmap_exit]]
        in_array = graph.nodes()[candidate[MapWCRFusion._in_array]]
        rmap_entry = graph.nodes()[candidate[MapWCRFusion._rmap_out_entry]]

        # Make sure that the array is only accessed by the map and the reduce
        if any([
                src != tmap_exit
                for src, _, _, _, memlet in graph.in_edges(in_array)
        ]):
            return False
        if any([
                dest != rmap_entry
                for _, _, dest, _, memlet in graph.out_edges(in_array)
        ]):
            return False

        # Make sure that there is a reduction in the second map
        rmap_cr = graph.nodes()[candidate[MapWCRFusion._rmap_in_cr]]
        reduce_edge = graph.in_edges(rmap_cr)[0]
        if reduce_edge.data.wcr is None:
            return False

        # (strict) Make sure that the transient is not accessed anywhere else
        # in this state or other states
        if strict and (len([
                n for n in graph.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == in_array.data
        ]) > 1 or in_array.data in sdfg.shared_transients()):
            return False

        # Verify that reduction ranges match tasklet map
        tout_memlet = graph.in_edges(in_array)[0].data
        rin_memlet = graph.out_edges(in_array)[0].data
        if tout_memlet.subset != rin_memlet.subset:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        tasklet = candidate[MapWCRFusion._tasklet]
        map_exit = candidate[MapWCRFusion._tmap_exit]
        reduce = candidate[MapWCRFusion._rmap_in_cr]

        return ' -> '.join(str(node) for node in [tasklet, map_exit, reduce])

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)

        # To apply, collapse the second map and then fuse the two resulting maps
        map_collapse = MapCollapse(
            self.sdfg_id, self.state_id, {
                MapCollapse._outer_map_entry:
                self.subgraph[MapWCRFusion._rmap_out_entry],
                MapCollapse._inner_map_entry:
                self.subgraph[MapWCRFusion._rmap_in_entry]
            }, 0)
        map_entry, _ = map_collapse.apply(sdfg)

        map_fusion = MapFusion(
            self.sdfg_id, self.state_id, {
                MapFusion.first_map_exit:
                self.subgraph[MapWCRFusion._tmap_exit],
                MapFusion.second_map_entry: graph.node_id(map_entry)
            }, 0)
        map_fusion.apply(sdfg)
