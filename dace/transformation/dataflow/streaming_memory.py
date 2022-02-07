# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
import copy
from typing import Dict, List, Tuple
import networkx as nx
import warnings

from dace.transformation import transformation as xf
from dace import (data, dtypes, nodes, properties, registry, memlet as mm, subsets, symbolic)
from dace.sdfg import SDFG, SDFGState, utils as sdutil, graph as gr


def _collect_map_ranges(state: SDFGState,
                        memlet_path: List[gr.MultiConnectorEdge[mm.Memlet]]) -> List[Tuple[str, subsets.Range]]:
    """
    Collects a list of parameters and ranges for every map (entry or exit)
    in the given memlet path.
    """
    ranges: List[Tuple[str, subsets.Range]] = []
    # Outgoing (write) memlet path
    if any(isinstance(e.src, nodes.MapExit) for e in memlet_path):
        for e in reversed(memlet_path):
            if isinstance(e.src, nodes.MapExit):
                entry = state.entry_node(e.src)
                ranges.extend([(p, r) for p, r in zip(entry.params, entry.range)])
    else:  # Incoming (read) memlet path
        for e in memlet_path:
            if isinstance(e.dst, nodes.MapEntry):
                ranges.extend([(p, r) for p, r in zip(e.dst.params, e.dst.range)])
    return ranges


def _canonicalize_memlet(memlet: mm.Memlet, mapranges: List[Tuple[str, subsets.Range]]) -> Tuple[symbolic.SymbolicType]:
    """
    Turn a memlet subset expression (of a single element) into an expression
    that does not depend on the map symbol names.
    """
    repldict = {symbolic.symbol(p): symbolic.symbol('__dace%d' % i) for i, (p, _) in enumerate(mapranges)}

    return tuple(rb.subs(repldict) for rb, _, _ in memlet.subset.ndrange())


def _do_memlets_correspond(memlet_a: mm.Memlet, memlet_b: mm.Memlet, mapranges_a: List[Tuple[str, subsets.Range]],
                           mapranges_b: List[Tuple[str, subsets.Range]]) -> bool:
    """
    Returns True if the two memlets correspond to each other, disregarding
    symbols from equivalent maps.
    """
    for s1, s2 in zip(memlet_a.subset, memlet_b.subset):
        # Check for matching but disregard parameter names
        s1b = s1[0].subs(
            {symbolic.symbol(k1): symbolic.symbol(k2)
             for (k1, _), (k2, _) in zip(mapranges_a, mapranges_b)})
        s2b = s2[0]
        # Since there is one element in both subsets, we can check only
        # the beginning
        if s1b != s2b:
            return False
    return True


def _streamify_recursive(node: nodes.NestedSDFG, to_replace: str, desc: data.Stream):
    """ Helper function that changes an array in a nested SDFG to a stream. """
    nsdfg: SDFG = node.sdfg
    newdesc = copy.deepcopy(desc)
    newdesc.transient = False
    nsdfg.arrays[to_replace] = newdesc

    # Replace memlets in path with stream access
    for state in nsdfg.nodes():
        for dnode in state.data_nodes():
            if dnode.data != to_replace:
                continue
            for edge in state.all_edges(dnode):
                mpath = state.memlet_path(edge)
                for e in mpath:
                    e.data = mm.Memlet(data=to_replace, subset='0', other_subset=e.data.other_subset)
                    if isinstance(e.src, nodes.NestedSDFG):
                        e.data.dynamic = True
                        _streamify_recursive(e.src, e.src_conn, newdesc)
                    if isinstance(e.dst, nodes.NestedSDFG):
                        e.data.dynamic = True
                        _streamify_recursive(e.dst, e.dst_conn, newdesc)


@properties.make_properties
class StreamingMemory(xf.SingleStateTransformation):
    """
    Converts a read or a write to streaming memory access, where data is
    read/written to/from a stream in a separate connected component than the
    computation.
    """
    access = xf.PatternNode(nodes.AccessNode)
    entry = xf.PatternNode(nodes.EntryNode)
    exit = xf.PatternNode(nodes.ExitNode)

    buffer_size = properties.Property(dtype=int, default=1, desc='Set buffer size for the newly-created stream')

    storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                      desc='Set storage type for the newly-created stream',
                                      default=dtypes.StorageType.Default)

    @classmethod
    def expressions(cls) -> List[gr.SubgraphView]:
        return [
            sdutil.node_path_graph(cls.access, cls.entry),
            sdutil.node_path_graph(cls.exit, cls.access),
        ]

    def can_be_applied(self, graph: SDFGState,
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        access = self.access
        # Make sure the access node is only accessed once (read or write),
        # and not at the same time
        if graph.out_degree(access) > 0 and graph.in_degree(access) > 0:
            return False

        # If already a stream, skip
        if isinstance(sdfg.arrays[access.data], data.Stream):
            return False
        # If does not exist on off-chip memory, skip
        if sdfg.arrays[access.data].storage not in [
                dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_Pinned, dtypes.StorageType.GPU_Global,
                dtypes.StorageType.FPGA_Global
        ]:
            return False

        # Only free nodes are allowed (search up the SDFG tree)
        curstate = graph
        node = access
        while curstate is not None:
            if curstate.entry_node(node) is not None:
                return False
            if curstate.parent.parent_nsdfg_node is None:
                break
            node = curstate.parent.parent_nsdfg_node
            curstate = curstate.parent.parent

        # Only one memlet path is allowed per outgoing/incoming edge
        edges = (graph.out_edges(access) if expr_index == 0 else graph.in_edges(access))
        for edge in edges:
            mpath = graph.memlet_path(edge)
            if len(mpath) != len(list(graph.memlet_tree(edge))):
                return False

            # The innermost end of the path must have a clearly defined memory
            # access pattern
            innermost_edge = mpath[-1] if expr_index == 0 else mpath[0]
            if (innermost_edge.data.subset.num_elements() != 1 or innermost_edge.data.dynamic
                    or innermost_edge.data.volume != 1):
                return False

            # Check if any of the maps has a dynamic range
            # These cases can potentially work but some nodes (and perhaps
            # tasklets) need to be replicated, which are difficult to track.
            for pe in mpath:
                node = pe.dst if expr_index == 0 else graph.entry_node(pe.src)
                if isinstance(node, nodes.MapEntry) and sdutil.has_dynamic_map_inputs(graph, node):
                    return False

        # If already applied on this memlet and this is the I/O component, skip
        if expr_index == 0:
            other_node = self.entry
        else:
            other_node = self.exit
            other_node = graph.entry_node(other_node)
        if other_node.label.startswith('__s'):
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG) -> nodes.AccessNode:
        dnode: nodes.AccessNode = self.access
        if self.expr_index == 0:
            edges = state.out_edges(dnode)
        else:
            edges = state.in_edges(dnode)

        # To understand how many components we need to create, all map ranges
        # throughout memlet paths must match exactly. We thus create a
        # dictionary of unique ranges
        mapping: Dict[Tuple[subsets.Range], List[gr.MultiConnectorEdge[mm.Memlet]]] = defaultdict(list)
        ranges = {}
        for edge in edges:
            mpath = state.memlet_path(edge)
            ranges[edge] = _collect_map_ranges(state, mpath)
            mapping[tuple(r[1] for r in ranges[edge])].append(edge)

        # Collect all edges with the same memory access pattern
        components_to_create: Dict[Tuple[symbolic.SymbolicType],
                                   List[gr.MultiConnectorEdge[mm.Memlet]]] = defaultdict(list)
        for edges_with_same_range in mapping.values():
            for edge in edges_with_same_range:
                # Get memlet path and innermost edge
                mpath = state.memlet_path(edge)
                innermost_edge = copy.deepcopy(mpath[-1] if self.expr_index == 0 else mpath[0])

                # Store memlets of the same access in the same component
                expr = _canonicalize_memlet(innermost_edge.data, ranges[edge])
                components_to_create[expr].append((innermost_edge, edge))
        components = list(components_to_create.values())

        # Split out components that have dependencies between them to avoid
        # deadlocks
        if self.expr_index == 0:
            ccs_to_add = []
            for i, component in enumerate(components):
                edges_to_remove = set()
                for cedge in component:
                    if any(nx.has_path(state.nx, o[1].dst, cedge[1].dst) for o in component if o is not cedge):
                        ccs_to_add.append([cedge])
                        edges_to_remove.add(cedge)
                if edges_to_remove:
                    components[i] = [c for c in component if c not in edges_to_remove]
            components.extend(ccs_to_add)
        # End of split

        desc = sdfg.arrays[dnode.data]

        # Create new streams of shape 1
        streams = {}
        mpaths = {}
        for edge in edges:
            name, newdesc = sdfg.add_stream(dnode.data,
                                            desc.dtype,
                                            buffer_size=self.buffer_size,
                                            storage=self.storage,
                                            transient=True,
                                            find_new_name=True)
            streams[edge] = name
            mpath = state.memlet_path(edge)
            mpaths[edge] = mpath

            # Replace memlets in path with stream access
            for e in mpath:
                e.data = mm.Memlet(data=name, subset='0', other_subset=e.data.other_subset)
                if isinstance(e.src, nodes.NestedSDFG):
                    e.data.dynamic = True
                    _streamify_recursive(e.src, e.src_conn, newdesc)
                if isinstance(e.dst, nodes.NestedSDFG):
                    e.data.dynamic = True
                    _streamify_recursive(e.dst, e.dst_conn, newdesc)

            # Replace access node and memlet tree with one access
            if self.expr_index == 0:
                replacement = state.add_read(name)
                state.remove_edge(edge)
                state.add_edge(replacement, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
            else:
                replacement = state.add_write(name)
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, replacement, edge.dst_conn, edge.data)

        # Make read/write components
        ionodes = []
        for component in components:

            # Pick the first edge as the edge to make the component from
            innermost_edge, outermost_edge = component[0]
            mpath = mpaths[outermost_edge]
            mapname = streams[outermost_edge]
            innermost_edge.data.other_subset = None

            # Get edge data and streams
            if self.expr_index == 0:
                opname = 'read'
                path = [e.dst for e in mpath[:-1]]
                rmemlets = [(dnode, '__inp', innermost_edge.data)]
                wmemlets = []
                for i, (_, edge) in enumerate(component):
                    name = streams[edge]
                    ionode = state.add_write(name)
                    ionodes.append(ionode)
                    wmemlets.append((ionode, '__out%d' % i, mm.Memlet(data=name, subset='0')))
                code = '\n'.join('__out%d = __inp' % i for i in range(len(component)))
            else:
                # More than one input stream might mean a data race, so we only
                # address the first one in the tasklet code
                if len(component) > 1:
                    warnings.warn(f'More than one input found for the same index for {dnode.data}')
                opname = 'write'
                path = [state.entry_node(e.src) for e in reversed(mpath[1:])]
                wmemlets = [(dnode, '__out', innermost_edge.data)]
                rmemlets = []
                for i, (_, edge) in enumerate(component):
                    name = streams[edge]
                    ionode = state.add_read(name)
                    ionodes.append(ionode)
                    rmemlets.append((ionode, '__inp%d' % i, mm.Memlet(data=name, subset='0')))
                code = '__out = __inp0'

            # Create map structure for read/write component
            maps = []
            for entry in path:
                map: nodes.Map = entry.map
                maps.append(
                    state.add_map(f'__s{opname}_{mapname}',
                                  [(p, r)
                                   for p, r in zip(map.params, map.range)],
                                  map.schedule if ((not map.schedule is ScheduleType.FPGA_Double) and (not map.schedule is ScheduleType.FPGA_Double_out)) else ScheduleType.FPGA_Device)) # TODO the new external interfaces shouldn't be double pumped!
            tasklet = state.add_tasklet(
                f'{opname}_{mapname}',
                {m[1]
                 for m in rmemlets},
                {m[1]
                 for m in wmemlets},
                code,
            )
            for node, cname, memlet in rmemlets:
                state.add_memlet_path(node, *(me for me, _ in maps), tasklet, dst_conn=cname, memlet=memlet)
            for node, cname, memlet in wmemlets:
                state.add_memlet_path(tasklet, *(mx for _, mx in reversed(maps)), node, src_conn=cname, memlet=memlet)

        return ionodes


@properties.make_properties
class StreamingComposition(xf.SingleStateTransformation):
    """
    Converts two connected computations (nodes, map scopes) into two separate
    processing elements, with a stream connecting the results. Only applies
    if the memory access patterns of the two computations match.
    """
    first = xf.PatternNode(nodes.Node)
    access = xf.PatternNode(nodes.AccessNode)
    second = xf.PatternNode(nodes.Node)

    buffer_size = properties.Property(dtype=int, default=1, desc='Set buffer size for the newly-created stream')

    storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                      desc='Set storage type for the newly-created stream',
                                      default=dtypes.StorageType.Default)

    @classmethod
    def expressions(cls) -> List[gr.SubgraphView]:
        return [
            sdutil.node_path_graph(cls.first, cls.access, cls.second)
        ]

    def can_be_applied(self, graph: SDFGState,
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        access = self.access
        # Make sure the access node is only accessed once (read or write),
        # and not at the same time
        if graph.in_degree(access) > 1 or graph.out_degree(access) > 1:
            return False

        # If already a stream, skip
        if isinstance(sdfg.arrays[access.data], data.Stream):
            return False

        # Only free nodes are allowed (search up the SDFG tree)
        curstate = graph
        node = access
        while curstate is not None:
            if curstate.entry_node(node) is not None:
                return False
            if curstate.parent.parent_nsdfg_node is None:
                break
            node = curstate.parent.parent_nsdfg_node
            curstate = curstate.parent.parent

        # Array must not be used anywhere else in the state
        if any(n is not access and n.data == access.data for n in graph.data_nodes()):
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
        ranges_first = _collect_map_ranges(graph, first_mpath)
        ranges_second = _collect_map_ranges(graph, second_mpath)

        # Check map ranges
        for (_, frng), (_, srng) in zip(ranges_first, ranges_second):
            if frng != srng:
                return False

        # Check memlets for equivalence
        if len(first_iedge.data.subset) != len(second_iedge.data.subset):
            return False
        if not _do_memlets_correspond(first_iedge.data, second_iedge.data, ranges_first, ranges_second):
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG) -> nodes.AccessNode:
        access: nodes.AccessNode = self.access

        # Get memlet paths
        first_edge = state.in_edges(access)[0]
        second_edge = state.out_edges(access)[0]
        first_mpath = state.memlet_path(first_edge)
        second_mpath = state.memlet_path(second_edge)

        # Create new stream of shape 1
        desc = sdfg.arrays[access.data]
        name, newdesc = sdfg.add_stream(access.data,
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
            if isinstance(e.src, nodes.NestedSDFG):
                e.data.dynamic = True
                _streamify_recursive(e.src, e.src_conn, newdesc)
            if isinstance(e.dst, nodes.NestedSDFG):
                e.data.dynamic = True
                _streamify_recursive(e.dst, e.dst_conn, newdesc)
        for e in second_mpath:
            e.data = mm.Memlet(data=name, subset='0')
            if isinstance(e.src, nodes.NestedSDFG):
                e.data.dynamic = True
                _streamify_recursive(e.src, e.src_conn, newdesc)
            if isinstance(e.dst, nodes.NestedSDFG):
                e.data.dynamic = True
                _streamify_recursive(e.dst, e.dst_conn, newdesc)

        # Replace array access node with two stream access nodes
        wnode = state.add_write(name)
        rnode = state.add_read(name)
        state.remove_edge(first_edge)
        state.add_edge(first_edge.src, first_edge.src_conn, wnode, first_edge.dst_conn, first_edge.data)
        state.remove_edge(second_edge)
        state.add_edge(rnode, second_edge.src_conn, second_edge.dst, second_edge.dst_conn, second_edge.data)

        # Remove original access node
        state.remove_node(access)

        return wnode, rnode
