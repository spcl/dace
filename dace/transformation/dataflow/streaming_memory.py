# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from typing import Dict, List
from dace.transformation import transformation as xf, helpers as xfh
from dace import data, dtypes, nodes, properties, registry, memlet as mm
from dace.sdfg import SDFG, SDFGState, utils as sdutil, graph as gr


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class StreamingMemory(xf.Transformation):
    """ 
    Converts a read or a write to streaming memory access, where data is
    read/written to/from a stream in a separate connected component than the
    computation.
    """
    access = xf.PatternNode(nodes.AccessNode)
    entry = xf.PatternNode(nodes.EntryNode)
    exit = xf.PatternNode(nodes.ExitNode)

    buffer_size = properties.Property(
        dtype=int,
        default=1,
        desc='Set buffer size for the newly-created stream')

    storage = properties.Property(
        dtype=dtypes.StorageType,
        desc='Set storage type for the newly-created stream',
        choices=dtypes.StorageType,
        default=dtypes.StorageType.FPGA_Local,
        from_string=lambda x: dtypes.StorageType[x])

    @staticmethod
    def expressions() -> List[gr.SubgraphView]:
        return [
            sdutil.node_path_graph(StreamingMemory.access,
                                   StreamingMemory.entry),
            sdutil.node_path_graph(StreamingMemory.exit,
                                   StreamingMemory.access),
        ]

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate: Dict[xf.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:
        access = graph.node(candidate[StreamingMemory.access])
        # Make sure the access node is only accessed once (read or write),
        # and not at the same time
        if graph.in_degree(access) > 1 or graph.out_degree(access) > 1:
            return False
        if graph.out_degree(access) > 0 and graph.in_degree(access) > 0:
            return False

        # If already a stream, skip
        if isinstance(sdfg.arrays[access.data], data.Stream):
            return False

        # Only free nodes are allowed
        if graph.entry_node(access) is not None:
            return False

        # Only one memlet path is allowed
        # TODO: Relax so that repeated application of
        # transformation would yield additional streams
        edge = (graph.out_edges(access)[0]
                if expr_index == 0 else graph.in_edges(access)[0])
        mpath = graph.memlet_path(edge)
        if len(mpath) != len(list(graph.memlet_tree(edge))):
            return False

        # The innermost end of the path must have a clearly defined memory
        # access pattern
        innermost_edge = mpath[-1] if expr_index == 0 else mpath[0]
        if innermost_edge.data.subset.num_elements() != 1:
            return False

        # If already applied on this memlet and this is the I/O component, skip
        if expr_index == 0:
            other_node = graph.node(candidate[StreamingMemory.entry])
        else:
            other_node = graph.node(candidate[StreamingMemory.exit])
            other_node = graph.entry_node(other_node)
        if other_node.label.startswith('__s'):
            return False

        return True

    def apply(self, sdfg: SDFG) -> nodes.AccessNode:
        state = sdfg.node(self.state_id)
        dnode: nodes.AccessNode = self.access(sdfg)
        if self.expr_index == 0:
            edge = state.out_edges(dnode)[0]
        else:
            edge = state.in_edges(dnode)[0]

        # Get memlet path and innermost edge
        mpath = state.memlet_path(edge)
        innermost_edge = copy.deepcopy(mpath[-1] if self.expr_index ==
                                       0 else mpath[0])

        # Create new stream of shape 1
        desc = sdfg.arrays[dnode.data]
        name, _ = sdfg.add_stream(dnode.data,
                                  desc.dtype,
                                  buffer_size=self.buffer_size,
                                  storage=self.storage,
                                  transient=True,
                                  find_new_name=True)

        # Replace memlets in path with stream access
        for e in mpath:
            e.data = mm.Memlet(data=name, subset='0')

        # Replace access node and memlet tree with one access
        if self.expr_index == 0:
            replacement = state.add_read(name)
            state.remove_edge(edge)
            state.add_edge(replacement, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)
        else:
            replacement = state.add_write(name)
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, replacement, edge.dst_conn,
                           edge.data)

        # Make read/write component
        if self.expr_index == 0:
            ionode = state.add_write(name)
            path = [e.dst for e in mpath[:-1]]
            rnode = dnode
            rmemlet = innermost_edge.data
            wnode = ionode
            wmemlet = mm.Memlet(data=name, subset='0')
            opname = 'read'
        else:
            ionode = state.add_read(name)
            path = [state.entry_node(e.src) for e in reversed(mpath[1:])]
            rnode = ionode
            rmemlet = mm.Memlet(data=name, subset='0')
            wnode = dnode
            wmemlet = innermost_edge.data
            opname = 'write'

        # Create map structure for read/write component
        maps = []
        for entry in path:
            map: nodes.Map = entry.map
            maps.append(
                state.add_map(f'__s{opname}_{dnode.data}',
                              [(p, r) for p, r in zip(map.params, map.range)],
                              map.schedule))
        tasklet = state.add_tasklet(f'{opname}_{dnode.data}', {'inp'}, {'out'},
                                    'out = inp')
        state.add_memlet_path(rnode,
                              *(me for me, _ in maps),
                              tasklet,
                              dst_conn='inp',
                              memlet=rmemlet)
        state.add_memlet_path(tasklet,
                              *(mx for _, mx in maps),
                              wnode,
                              src_conn='out',
                              memlet=wmemlet)

        return ionode
