# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace import registry, properties, nodes, dtypes, symbolic
from dace import Memlet, SDFG, SDFGState
from dace.frontend.operations import detect_reduction_type
from dace.transformation import transformation as xf, helpers as xfh
from dace.sdfg import utils as sdutil


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class WarpTiling(xf.Transformation):
    """ 
    Implements a GPU specialization tiling that takes a GPU kernel map (with 
    nested maps, but without explicit block sizes) and divides its work across
    a warp. Specifically, it tiles its contents by a configurable warp size 
    (default: 32), and optionally preferring recomputation (map replication) 
    over local storage within the kernel. If write-conflicted reductions happen 
    within the given map, the transformation adds warp reductions to the tiles.
    """

    warp_size = properties.Property(dtype=int,
                                    default=32,
                                    desc='Hardware warp size')
    replicate_maps = properties.Property(
        dtype=bool,
        default=True,
        desc='Replicate tiled maps that lead to multiple other tiled maps')

    mapentry = xf.PatternNode(nodes.MapEntry)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(WarpTiling.mapentry)]

    def can_be_applied(self, graph: SDFGState, candidate, expr_index,
                       sdfg: SDFG, strict) -> bool:
        me: nodes.MapEntry = self.mapentry(sdfg)

        # GPU map that has no predefined thread-block maps
        return (me.schedule == dtypes.ScheduleType.GPU_Device
                and not xfh.gpu_map_has_explicit_threadblocks(graph, me))

    def apply(self, sdfg: SDFG) -> nodes.MapEntry:
        me: nodes.MapEntry = self.mapentry(sdfg)
        graph = sdfg.node(self.state_id)

        # Add new map within map
        mx = graph.exit_node(me)
        new_me, new_mx = graph.add_map('warp_tile',
                                       dict(__tid=f'0:{self.warp_size}'),
                                       dtypes.ScheduleType.GPU_ThreadBlock)
        __tid = symbolic.pystr_to_symbolic('__tid')
        for e in graph.out_edges(me):
            xfh.reconnect_edge_through_map(graph, e, new_me, True)
        for e in graph.in_edges(mx):
            xfh.reconnect_edge_through_map(graph, e, new_mx, False)

        # Stride and offset all internal maps
        maps_to_stride = xfh.get_internal_scopes(graph, new_me, immediate=True)
        for nstate, nmap in maps_to_stride:
            nsdfg = nstate.parent
            nsdfg_node = nsdfg.parent_nsdfg_node

            # Map cannot be partitioned across a warp
            if (nmap.range.size()[-1] < self.warp_size) == True:
                continue

            if nsdfg is not sdfg and nsdfg_node is not None:
                nsdfg_node.symbol_mapping['__tid'] = __tid
                if '__tid' not in nsdfg.symbols:
                    nsdfg.add_symbol('__tid', dtypes.int32)
            nmap.range[-1] = (nmap.range[-1][0], nmap.range[-1][1],
                              nmap.range[-1][2] * self.warp_size)
            subgraph = nstate.scope_subgraph(nmap)
            subgraph.replace(nmap.params[-1], f'{nmap.params[-1]} + __tid')
            inner_map_exit = nstate.exit_node(nmap)
            # If requested, replicate maps with multiple dependent maps
            if self.replicate_maps:
                destinations = [
                    nstate.memlet_path(edge)[-1].dst
                    for edge in nstate.out_edges(inner_map_exit)
                ]

                for dst in destinations:
                    # Transformation will not replicate map with more than one
                    # output
                    if len(destinations) != 1:
                        break
                    if not isinstance(dst, nodes.AccessNode):
                        continue  # Not leading to access node
                    if not xfh.contained_in(nstate, dst, new_me):
                        continue  # Memlet path goes out of map
                    if not nsdfg.arrays[dst.data].transient:
                        continue  # Cannot modify non-transients
                    for edge in nstate.out_edges(dst)[1:]:
                        rep_subgraph = xfh.replicate_scope(
                            nsdfg, nstate, subgraph)
                        rep_edge = nstate.out_edges(
                            rep_subgraph.sink_nodes()[0])[0]
                        # Add copy of data
                        newdesc = copy.deepcopy(sdfg.arrays[dst.data])
                        newname = nsdfg.add_datadesc(dst.data,
                                                     newdesc,
                                                     find_new_name=True)
                        newaccess = nstate.add_access(newname)
                        # Redirect edges
                        xfh.redirect_edge(nstate,
                                          rep_edge,
                                          new_dst=newaccess,
                                          new_data=newname)
                        xfh.redirect_edge(nstate,
                                          edge,
                                          new_src=newaccess,
                                          new_data=newname)

            # If has WCR, add warp-collaborative reduction on outputs
            for out_edge in nstate.out_edges(inner_map_exit):
                if out_edge.data.wcr is not None:
                    ctype = nsdfg.arrays[out_edge.data.data].dtype.ctype
                    redtype = detect_reduction_type(out_edge.data.wcr)
                    if redtype == dtypes.ReductionType.Custom:
                        raise NotImplementedError
                    credtype = ('dace::ReductionType::' +
                                str(redtype)[str(redtype).find('.') + 1:])

                    # Add local access between thread-locan and warp reduction
                    newnode = nstate.add_access(out_edge.data.data)
                    nstate.remove_edge(out_edge)
                    nstate.add_edge(out_edge.src, out_edge.src_conn, newnode,
                                    None, copy.deepcopy(out_edge.data))

                    if out_edge.data.subset.num_elements(
                    ) == 1:  # One element: tasklet
                        wrt = nstate.add_tasklet(
                            'warpreduce', {'__a'}, {'__out'},
                            f'__out = dace::warpReduce<{credtype}, {ctype}>::reduce(__a);',
                            dtypes.Language.CPP)
                        nstate.add_edge(newnode, None, wrt, '__a',
                                        Memlet(out_edge.data.data))
                        out_edge.data.wcr = None
                        nstate.add_edge(wrt, '__out', out_edge.dst, None,
                                        out_edge.data)
                    else:  # More than one element: mapped tasklet
                        raise NotImplementedError
            # End of WCR to warp reduction

        # Make nested SDFG out of new scope
        xfh.nest_state_subgraph(sdfg, graph,
                                graph.scope_subgraph(new_me, False, False))

        return new_me
