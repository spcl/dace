# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from typing import Dict, List, Tuple
from dace.transformation import transformation as xf
from dace import (data, dtypes, nodes, properties, registry, memlet as mm,
                  subsets, symbolic)
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
        default=dtypes.StorageType.Default,
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


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class StreamingComposition(xf.Transformation):
    """ 
    Converts two connected computations (nodes, map scopes) into two separate
    processing elements, with a stream connecting the results. Only applies
    if the memory access patterns of the two computations match.
    """
    first = xf.PatternNode(nodes.Node)
    access = xf.PatternNode(nodes.AccessNode)
    second = xf.PatternNode(nodes.Node)

    buffer_size = properties.Property(
        dtype=int,
        default=1,
        desc='Set buffer size for the newly-created stream')

    storage = properties.Property(
        dtype=dtypes.StorageType,
        desc='Set storage type for the newly-created stream',
        choices=dtypes.StorageType,
        default=dtypes.StorageType.Default,
        from_string=lambda x: dtypes.StorageType[x])

    @staticmethod
    def expressions() -> List[gr.SubgraphView]:
        return [
            sdutil.node_path_graph(StreamingComposition.first,
                                   StreamingComposition.access,
                                   StreamingComposition.second)
        ]

    @staticmethod
    def can_be_applied(graph: SDFGState,
                       candidate: Dict[xf.PatternNode, int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:
        access = graph.node(candidate[StreamingComposition.access])
        # Make sure the access node is only accessed once (read or write),
        # and not at the same time
        if graph.in_degree(access) > 1 or graph.out_degree(access) > 1:
            return False

        # If already a stream, skip
        if isinstance(sdfg.arrays[access.data], data.Stream):
            return False

        # Only free nodes are allowed
        if graph.entry_node(access) is not None:
            return False

        # Array must not be used anywhere else in the state
        if any(n is not access and n.data == access.data
               for n in graph.data_nodes()):
            return False

        # Only one memlet path on each direction is allowed
        # TODO: Relax so that repeated application of
        # transformation would yield additional streams
        first_edge = graph.in_edges(access)[0]
        second_edge = graph.out_edges(access)[0]
        first_mpath = graph.memlet_path(first_edge)
        second_mpath = graph.memlet_path(second_edge)
        if len(first_mpath) != len(list(graph.memlet_tree(first_edge))):
            return False
        if len(second_mpath) != len(list(graph.memlet_tree(second_edge))):
            return False

        # The innermost ends of the paths must have a clearly defined memory
        # access pattern and no WCR
        first_iedge = first_mpath[0]
        second_iedge = second_mpath[-1]
        if first_iedge.data.subset.num_elements() != 1:
            return False
        if first_iedge.data.volume != 1:
            return False
        if first_iedge.data.wcr is not None:
            return False
        if second_iedge.data.subset.num_elements() != 1:
            return False
        if second_iedge.data.volume != 1:
            return False

        ##################################################################
        # The memory access pattern must be exactly the same

        # Collect all maps and ranges
        ranges_first: List[Tuple[str, subsets.Range]] = []
        ranges_second: List[Tuple[str, subsets.Range]] = []
        for e in reversed(first_mpath):
            if isinstance(e.src, nodes.MapExit):
                entry = graph.entry_node(e.src)
                ranges_first.extend([
                    (p, r) for p, r in zip(entry.params, entry.range)
                ])
        for e in second_mpath:
            if isinstance(e.dst, nodes.MapEntry):
                ranges_second.extend([
                    (p, r) for p, r in zip(e.dst.params, e.dst.range)
                ])

        # Check map ranges
        for (_, frng), (_, srng) in zip(ranges_first, ranges_second):
            if frng != srng:
                return False

        # Check memlets for equivalence
        if len(first_iedge.data.subset) != len(second_iedge.data.subset):
            return False
        for s1, s2 in zip(first_iedge.data.subset, second_iedge.data.subset):
            # Check for matching but disregard parameter names
            s1b = s1[0].subs({
                symbolic.symbol(k1): symbolic.symbol(k2)
                for (k1, _), (k2, _) in zip(ranges_first, ranges_second)
            })
            s2b = s2[0]
            # Since there is one element in both subsets, we can check only
            # the beginning
            if s1b != s2b:
                return False

        return True

    def apply(self, sdfg: SDFG) -> nodes.AccessNode:
        state = sdfg.node(self.state_id)
        access: nodes.AccessNode = self.access(sdfg)

        # Get memlet paths
        first_edge = state.in_edges(access)[0]
        second_edge = state.out_edges(access)[0]
        first_mpath = state.memlet_path(first_edge)
        second_mpath = state.memlet_path(second_edge)

        # Create new stream of shape 1
        desc = sdfg.arrays[access.data]
        name, _ = sdfg.add_stream(access.data,
                                  desc.dtype,
                                  buffer_size=self.buffer_size,
                                  storage=self.storage,
                                  transient=True,
                                  find_new_name=True)

        # Remove transient array if possible
        for ostate in sdfg.nodes():
            if ostate is state:
                continue
            if any(n.data == access.data for n in ostate.data_nodes()):
                break
        else:
            del sdfg.arrays[access.data]

        # Replace memlets in path with stream access
        for e in first_mpath:
            e.data = mm.Memlet(data=name, subset='0')
        for e in second_mpath:
            e.data = mm.Memlet(data=name, subset='0')

        # Replace array access node with two stream access nodes
        wnode = state.add_write(name)
        rnode = state.add_read(name)
        state.remove_edge(first_edge)
        state.add_edge(first_edge.src, first_edge.src_conn, wnode,
                       first_edge.dst_conn, first_edge.data)
        state.remove_edge(second_edge)
        state.add_edge(rnode, second_edge.src_conn, second_edge.dst,
                       second_edge.dst_conn, second_edge.data)

        # Remove original access node
        state.remove_node(access)

        return wnode, rnode
