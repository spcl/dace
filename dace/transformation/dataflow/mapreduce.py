""" Contains classes and functions that implement the map-reduce-fusion 
    transformation. """

import copy
from dace import data as dt, types, subsets, symbolic
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.sdfg import SDFGState
from dace.transformation import pattern_matching as pm
from dace.properties import ShapeProperty


class MapReduceFusion(pm.Transformation):
    """ Implements the map-reduce-fusion transformation.
        Fuses a map with an immediately following reduction, where the array
        between the map and the reduction is not used anywhere else.
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
    _reduce = nodes.Reduce('lambda: None', None)

    @staticmethod
    def expressions():
        return [
            # Map, then reduce of all axes
            nxutil.node_path_graph(
                MapReduceFusion._tasklet, MapReduceFusion._tmap_exit,
                MapReduceFusion._in_array, MapReduceFusion._rmap_in_entry,
                MapReduceFusion._rmap_in_tasklet, MapReduceFusion._rmap_in_cr,
                MapReduceFusion._out_array),
            # Map, then partial reduction of axes
            nxutil.node_path_graph(
                MapReduceFusion._tasklet, MapReduceFusion._tmap_exit,
                MapReduceFusion._in_array, MapReduceFusion._rmap_out_entry,
                MapReduceFusion._rmap_in_entry,
                MapReduceFusion._rmap_in_tasklet, MapReduceFusion._rmap_in_cr,
                MapReduceFusion._rmap_out_exit, MapReduceFusion._out_array),
            # Map, then reduce node
            nxutil.node_path_graph(
                MapReduceFusion._tasklet, MapReduceFusion._tmap_exit,
                MapReduceFusion._in_array, MapReduceFusion._reduce,
                MapReduceFusion._out_array)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        tmap_exit = graph.nodes()[candidate[MapReduceFusion._tmap_exit]]
        in_array = graph.nodes()[candidate[MapReduceFusion._in_array]]
        if expr_index == 0:  # Reduce without outer map
            rmap_entry = graph.nodes()[candidate[
                MapReduceFusion._rmap_in_entry]]
            # rmap_in_entry = rmap_entry
        elif expr_index == 1:  # Reduce with outer map
            rmap_entry = graph.nodes()[candidate[
                MapReduceFusion._rmap_out_entry]]
            # rmap_in_entry = graph.nodes()[candidate[
            #     MapReduceFusion._rmap_in_entry]]
        else:  # Reduce node
            rmap_entry = graph.nodes()[candidate[MapReduceFusion._reduce]]

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
        if expr_index < 2:
            rmap_cr = graph.nodes()[candidate[MapReduceFusion._rmap_in_cr]]
            reduce_edge = graph.in_edges(rmap_cr)[0]
            if reduce_edge.data.wcr is None:
                return False

        # Make sure that the transient is not accessed by other states
        # if garr.get_unique_name() in cgen_state.sdfg.shared_transients():
        #     return False

        # reduce_inarr = reduce.in_array
        # reduce_outarr = reduce.out_array
        # reduce_inslice = reduce.inslice
        # reduce_outslice = reduce.outslice

        # insize = cgen_state.var_sizes[reduce_inarr]
        # outsize = cgen_state.var_sizes[reduce_outarr]

        # Currently only supports full-range arrays
        # TODO(later): Support fusion of partial reductions and refactor slice/subarray handling
        #if not nxutil.fullrange(reduce_inslice, insize) or \
        #   not nxutil.fullrange(reduce_outslice, outsize):
        #    return False

        # Verify acceses from tasklet through MapExit
        #already_found = False
        #for _src, _, _dest, _, memlet in graph.in_edges(map_exit):
        #    if isinstance(memlet.subset, subsets.Indices):
        #        # Make sure that only one value is reduced at a time
        #        if memlet.data == in_array.desc:
        #            if already_found:
        #                return False
        #            already_found = True

        ## Find axes after reduction
        #indims = len(reduce.inslice)
        #axis_after_reduce = [None] * indims
        #ctr = 0
        #for i in range(indims):
        #    if reduce.axes is not None and i in reduce.axes:
        #        axis_after_reduce[i] = None
        #    else:
        #        axis_after_reduce[i] = ctr
        #        ctr += 1

        ## Match map ranges with reduce ranges
        #curaxis = 0
        #for dim, var in enumerate(memlet.subset):
        #    # Make sure that indices are direct symbols
        #    #if not isinstance(symbolic.pystr_to_symbolic(var), sympy.Symbol):
        #    #    return False
        #    perm = None
        #    for i, mapvar in enumerate(map_exit.map.params):
        #        if symbolic.pystr_to_symbolic(mapvar) == var:
        #            perm = i
        #            break
        #    if perm is None:  # If symbol is not found in map range
        #        return False

        #    # Make sure that map ranges match output slice after reduction
        #    map_range = map_exit.map.range[perm]
        #    if map_range[0] != 0:
        #        return False  # Disallow start from middle
        #    if map_range[2] is not None and map_range[2] != 1:
        #        return False  # Disallow skip
        #    if reduce.axes is not None and dim not in reduce.axes:
        #        if map_range[1] != symbolic.pystr_to_symbolic(
        #                reduce.outslice[axis_after_reduce[dim]][1]):
        #            return False  # Range check (output axis)
        #    else:
        #        if map_range[1] != symbolic.pystr_to_symbolic(reduce.inslice[dim][1]):
        #            return False  # Range check (reduction axis)

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
        if len(candidate) == 5:  # Expression 2
            reduce = candidate[MapReduceFusion._reduce]
        else:
            reduce = candidate[MapReduceFusion._rmap_in_cr]

        return ' -> '.join(str(node) for node in [tasklet, map_exit, reduce])

    @staticmethod
    def find_memlet_map_permutation(memlet: Memlet, map: nodes.Map):
        perm = [None] * len(memlet.subset)
        indices = set()
        for i, dim in enumerate(memlet.subset):
            for j, mapdim in enumerate(map.params):
                if symbolic.pystr_to_symbolic(
                        mapdim) == dim and j not in indices:
                    perm[i] = j
                    indices.add(j)
                    break
        return perm

    @staticmethod
    def find_permutation(tasklet_map: nodes.Map, red_outer_map: nodes.Map,
                         red_inner_map: nodes.Map, tmem: Memlet):
        """ Find permutation between tasklet-exit memlet and tasklet map. """
        result = [], []

        assert len(tasklet_map.range) == len(red_inner_map.range) + len(
            red_outer_map.range)

        # Match map ranges with reduce ranges
        unavailable_ranges_out = set()
        unavailable_ranges_in = set()
        for i, tmap_rng in enumerate(tasklet_map.range):
            found = False
            for j, rng in enumerate(red_outer_map.range):
                if tmap_rng == rng and j not in unavailable_ranges_out:
                    result[0].append(i)
                    unavailable_ranges_out.add(j)
                    found = True
                    break
            if found: continue
            for j, rng in enumerate(red_inner_map.range):
                if tmap_rng == rng and j not in unavailable_ranges_in:
                    result[1].append(i)
                    unavailable_ranges_in.add(j)
                    found = True
                    break
            if not found: break

        # Ensure all map variables matched with reduce variables
        assert len(result[0]) + len(result[1]) == len(tasklet_map.range)

        # Returns ([outer map indices], [inner (CR) map indices])
        return result

    @staticmethod
    def find_permutation_reduce(tasklet_map: nodes.Map,
                                reduce_node: nodes.Reduce, graph: SDFGState,
                                tmem: Memlet):

        in_memlet = graph.in_edges(reduce_node)[0].data
        out_memlet = graph.out_edges(reduce_node)[0].data
        assert len(tasklet_map.range) == in_memlet.subset.dims()

        # Find permutation between tasklet-exit memlet and tasklet map
        tmem_perm = MapReduceFusion.find_memlet_map_permutation(
            tmem, tasklet_map)
        mapred_perm = []

        # Match map ranges with reduce ranges
        unavailable_ranges = set()
        for i, tmap_rng in enumerate(tasklet_map.range):
            found = False

            for j, in_rng in enumerate(in_memlet.subset):
                if tmap_rng == in_rng and j not in unavailable_ranges:
                    mapred_perm.append(i)
                    unavailable_ranges.add(j)
                    found = True
                    break
            if not found: break

        # Ensure all map variables matched with reduce variables
        assert len(tmem_perm) == len(tmem.subset)
        assert len(mapred_perm) == len(in_memlet.subset)

        # Prepare result from the two permutations and the reduction axes
        result = []
        for i in range(len(mapred_perm)):
            if reduce_node.axes is None or i in reduce_node.axes:
                continue
            result.append(mapred_perm[tmem_perm[i]])

        return result

    def apply(self, sdfg):
        def gnode(nname):
            return graph.nodes()[self.subgraph[nname]]

        expr_index = self.expr_index
        graph = sdfg.nodes()[self.state_id]
        tasklet = gnode(MapReduceFusion._tasklet)
        tmap_exit = graph.nodes()[self.subgraph[MapReduceFusion._tmap_exit]]
        in_array = graph.nodes()[self.subgraph[MapReduceFusion._in_array]]
        if expr_index == 0:  # Reduce without outer map
            rmap_entry = graph.nodes()[self.subgraph[
                MapReduceFusion._rmap_in_entry]]
        elif expr_index == 1:  # Reduce with outer map
            rmap_out_entry = graph.nodes()[self.subgraph[
                MapReduceFusion._rmap_out_entry]]
            rmap_out_exit = graph.nodes()[self.subgraph[
                MapReduceFusion._rmap_out_exit]]
            rmap_in_entry = graph.nodes()[self.subgraph[
                MapReduceFusion._rmap_in_entry]]
            rmap_tasklet = graph.nodes()[self.subgraph[
                MapReduceFusion._rmap_in_tasklet]]

        if expr_index == 2:
            rmap_cr = graph.nodes()[self.subgraph[MapReduceFusion._reduce]]
        else:
            rmap_cr = graph.nodes()[self.subgraph[MapReduceFusion._rmap_in_cr]]
        out_array = gnode(MapReduceFusion._out_array)

        # Set nodes to remove according to the expression index
        nodes_to_remove = [in_array]
        if expr_index == 0:
            nodes_to_remove.append(gnode(MapReduceFusion._rmap_in_entry))
        elif expr_index == 1:
            nodes_to_remove.append(gnode(MapReduceFusion._rmap_out_entry))
            nodes_to_remove.append(gnode(MapReduceFusion._rmap_in_entry))
            nodes_to_remove.append(gnode(MapReduceFusion._rmap_out_exit))
        else:
            nodes_to_remove.append(gnode(MapReduceFusion._reduce))

        # If no other edges lead to mapexit, remove it. Otherwise, keep
        # it and remove reduction incoming/outgoing edges
        if expr_index != 2 and len(graph.in_edges(tmap_exit)) == 1:
            nodes_to_remove.append(tmap_exit)

        memlet_edge = None
        for edge in graph.in_edges(tmap_exit):
            if edge.data.data == in_array.data:
                memlet_edge = edge
                break
        if memlet_edge is None:
            raise RuntimeError('Reduction memlet cannot be None')

        if expr_index == 0:  # Reduce without outer map
            # Index order does not matter, merge as-is
            pass
        elif expr_index == 1:  # Reduce with outer map
            tmap = tmap_exit.map
            perm_outer, perm_inner = MapReduceFusion.find_permutation(
                tmap, rmap_out_entry.map, rmap_in_entry.map, memlet_edge.data)

            # Split tasklet map into tmap_out -> tmap_in (according to
            # reduction)
            omap = nodes.Map(
                tmap.label + '_nonreduce',
                [p for i, p in enumerate(tmap.params) if i in perm_outer],
                [r for i, r in enumerate(tmap.range) if i in perm_outer],
                tmap.schedule, tmap.unroll, tmap.is_async)
            tmap.params = [
                p for i, p in enumerate(tmap.params) if i in perm_inner
            ]
            tmap.range = [
                r for i, r in enumerate(tmap.range) if i in perm_inner
            ]
            omap_entry = nodes.MapEntry(omap)
            omap_exit = rmap_out_exit
            rmap_out_exit.map = omap

            # Reconnect graph to new map
            tmap_entry = graph.entry_node(tmap_exit)
            tmap_in_edges = list(graph.in_edges(tmap_entry))
            for e in tmap_in_edges:
                nxutil.change_edge_dest(graph, tmap_entry, omap_entry)
            for e in tmap_in_edges:
                graph.add_edge(omap_entry, e.src_conn, tmap_entry, e.dst_conn,
                               copy.copy(e.data))
        elif expr_index == 2:  # Reduce node
            # Find correspondence between map indices and array outputs
            tmap = tmap_exit.map
            perm = MapReduceFusion.find_permutation_reduce(
                tmap, rmap_cr, graph, memlet_edge.data)

            output_subset = [tmap.params[d] for d in perm]
            if len(output_subset) == 0:  # Output is a scalar
                output_subset = [0]

            array_edge = graph.out_edges(rmap_cr)[0]

            # Delete relevant edges and nodes
            graph.remove_edge(memlet_edge)
            graph.remove_nodes_from(nodes_to_remove)

            # Add new edges and nodes
            #   From tasklet to map exit
            graph.add_edge(
                memlet_edge.src, memlet_edge.src_conn, memlet_edge.dst,
                memlet_edge.dst_conn,
                Memlet(out_array.data, memlet_edge.data.num_accesses,
                       subsets.Indices(output_subset), memlet_edge.data.veclen,
                       rmap_cr.wcr, rmap_cr.identity))

            #   From map exit to output array
            graph.add_edge(
                memlet_edge.dst, 'OUT_' + memlet_edge.dst_conn[3:],
                array_edge.dst, array_edge.dst_conn,
                Memlet(array_edge.data.data, array_edge.data.num_accesses,
                       array_edge.data.subset, array_edge.data.veclen,
                       rmap_cr.wcr, rmap_cr.identity))

            return

        # Remove tmp array node prior to the others, so that a new one
        # can be created in its stead (see below)
        graph.remove_node(nodes_to_remove[0])
        nodes_to_remove = nodes_to_remove[1:]

        # Create tasklet -> tmp -> tasklet connection
        tmp = graph.add_array(
            'tmp',
            memlet_edge.data.subset.bounding_box_size(),
            sdfg.arrays[memlet_edge.data.data].dtype,
            transient=True)
        tasklet_tmp_memlet = copy.deepcopy(memlet_edge.data)
        tasklet_tmp_memlet.data = tmp.data
        tasklet_tmp_memlet.subset = ShapeProperty.to_string(tmp.shape)

        # Modify memlet to point to output array
        memlet_edge.data.data = out_array.data

        # Recover reduction axes from CR reduce subset
        reduce_cr_subset = graph.in_edges(rmap_tasklet)[0].data.subset
        reduce_axes = []
        for ind, crvar in enumerate(reduce_cr_subset.indices):
            if '__i' in str(crvar):
                reduce_axes.append(ind)

        # Modify memlet access index by filtering out reduction axes
        if True:  # expr_index == 0:
            newindices = []
            for ind, ovar in enumerate(memlet_edge.data.subset.indices):
                if ind not in reduce_axes:
                    newindices.append(ovar)
        if len(newindices) == 0:
            newindices = [0]

        memlet_edge.data.subset = subsets.Indices(newindices)

        graph.remove_edge(memlet_edge)

        graph.add_edge(memlet_edge.src, memlet_edge.src_conn, tmp,
                       memlet_edge.dst_conn, tasklet_tmp_memlet)

        red_edges = list(graph.in_edges(rmap_tasklet))
        if len(red_edges) != 1:
            raise RuntimeError('CR edge must be unique')

        tmp_tasklet_memlet = copy.deepcopy(tasklet_tmp_memlet)
        graph.add_edge(tmp, None, rmap_tasklet, red_edges[0].dst_conn,
                       tmp_tasklet_memlet)

        for e in graph.edges_between(rmap_tasklet, rmap_cr):
            e.data.subset = memlet_edge.data.subset

        # Move output edges to point directly to CR node
        if expr_index == 1:
            # Set output memlet between CR node and outer reduction map to
            # contain the same subset as the one pointing to the CR node
            for e in graph.out_edges(rmap_cr):
                e.data.subset = memlet_edge.data.subset

            rmap_out = gnode(MapReduceFusion._rmap_out_exit)
            nxutil.change_edge_src(graph, rmap_out, omap_exit)

        # Remove nodes
        graph.remove_nodes_from(nodes_to_remove)

        # For unrelated outputs, connect original output to rmap_out
        if expr_index == 1 and tmap_exit not in nodes_to_remove:
            other_out_edges = list(graph.out_edges(tmap_exit))
            for e in other_out_edges:
                graph.remove_edge(e)
                graph.add_edge(e.src, e.src_conn, omap_exit, None, e.data)
                graph.add_edge(omap_exit, None, e.dst, e.dst_conn,
                               copy.copy(e.data))

    def modifies_graph(self):
        return True


pm.Transformation.register_pattern(MapReduceFusion)
