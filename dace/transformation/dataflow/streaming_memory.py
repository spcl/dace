# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
import copy
from typing import Dict, List, Tuple
import networkx as nx
import warnings
import sympy

from dace.transformation import transformation as xf
from dace import (data, dtypes, nodes, properties, registry, memlet as mm, subsets, symbolic, symbol, Memlet)
from dace.sdfg import SDFG, SDFGState, utils as sdutil, graph as gr
from dace.libraries.standard import Gearbox


def get_post_state(sdfg: SDFG, state: SDFGState):
    """ 
    Returns the post state (the state that copies the data a back from the FGPA device) if there is one.
    """
    for s in sdfg.all_sdfgs_recursive():
        for post_state in s.states():

            if 'post_' + str(state) == str(post_state):
                return post_state

    return None


def is_int(i):
    return isinstance(i, int) or isinstance(i, sympy.core.numbers.Integer)


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
    If 'use_memory_buffering' is True, the transformation reads/writes data from memory
    using a wider data format (e.g. 512 bits), and then convert it
    on the fly to the right data type used by the computation: 
    """
    access = xf.PatternNode(nodes.AccessNode)
    entry = xf.PatternNode(nodes.EntryNode)
    exit = xf.PatternNode(nodes.ExitNode)

    buffer_size = properties.Property(dtype=int, default=1, desc='Set buffer size for the newly-created stream')

    storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                      desc='Set storage type for the newly-created stream',
                                      default=dtypes.StorageType.Default)

    use_memory_buffering = properties.Property(dtype=bool,
                                               default=False,
                                               desc='Set if memory buffering should be used.')

    memory_buffering_target_bytes = properties.Property(
        dtype=int, default=64, desc='Set bytes read/written from memory if memory buffering is enabled.')

    @classmethod
    def expressions(cls) -> List[gr.SubgraphView]:
        return [
            sdutil.node_path_graph(cls.access, cls.entry),
            sdutil.node_path_graph(cls.exit, cls.access),
        ]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
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

        ## Check Memory Buffering Properties
        if self.use_memory_buffering:

            access = self.access
            desc = sdfg.arrays[access.data]

            # Array has to be global array
            if desc.storage != dtypes.StorageType.FPGA_Global:
                return False

            # Type has to divide target bytes
            if self.memory_buffering_target_bytes % desc.dtype.bytes != 0:
                return False

            # Target bytes has to be >= size of data type
            if self.memory_buffering_target_bytes < desc.dtype.bytes:
                return False

            strides = list(desc.strides)

            # Last stride has to be one
            if strides[-1] != 1:
                return False

            vector_size = int(self.memory_buffering_target_bytes / desc.dtype.bytes)
            strides.pop()  # Remove last element since we already checked it

            # Other strides have to be divisible by vector size
            for stride in strides:

                if is_int(stride) and stride % vector_size != 0:
                    return False

            # Check if map has the right access pattern
            # Stride 1 access by innermost loop, innermost loop counter has to be divisible by vector size
            # Same code as in apply
            state = sdfg.node(self.state_id)
            dnode: nodes.AccessNode = self.access
            if self.expr_index == 0:
                edges = state.out_edges(dnode)
            else:
                edges = state.in_edges(dnode)

            mapping: Dict[Tuple[subsets.Range], List[gr.MultiConnectorEdge[mm.Memlet]]] = defaultdict(list)
            ranges = {}
            for edge in edges:
                mpath = state.memlet_path(edge)
                ranges[edge] = _collect_map_ranges(state, mpath)
                mapping[tuple(r[1] for r in ranges[edge])].append(edge)

            for edges_with_same_range in mapping.values():
                for edge in edges_with_same_range:
                    # Get memlet path and innermost edge
                    mpath = state.memlet_path(edge)
                    innermost_edge = copy.deepcopy(mpath[-1] if self.expr_index == 0 else mpath[0])

                    edge_subset = [a_tuple[0] for a_tuple in list(innermost_edge.data.subset)]

                    if self.expr_index == 0:
                        map_subset = innermost_edge.src.map.params.copy()
                        ranges = list(innermost_edge.src.map.range)
                    else:
                        map_subset = innermost_edge.dst.map.params.copy()
                        ranges = list(innermost_edge.dst.map.range)

                    # Check is correct access pattern
                    # Correct ranges in map
                    if is_int(ranges[-1][1]) and (ranges[-1][1] + 1) % vector_size != 0:
                        return False

                    if ranges[-1][2] != 1:
                        return False

                    # Correct access in array
                    if isinstance(edge_subset[-1], symbol) and str(edge_subset[-1]) == map_subset[-1]:
                        pass

                    elif isinstance(edge_subset[-1], sympy.core.add.Add):

                        counter: int = 0

                        for arg in edge_subset[-1].args:
                            if isinstance(arg, symbol) and str(arg) == map_subset[-1]:
                                counter += 1

                        if counter != 1:
                            return False

                    else:
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

            if self.use_memory_buffering:

                arrname = str(self.access)

                # Add gearbox
                total_size = edge.data.volume
                vector_size = int(self.memory_buffering_target_bytes / desc.dtype.bytes)

                if not is_int(sdfg.arrays[dnode.data].shape[-1]):
                    warnings.warn(
                        "Using the MemoryBuffering transformation is potential unsafe since {sym} is not an integer. There should be no issue if {sym} % {vec} == 0"
                        .format(sym=sdfg.arrays[dnode.data].shape[-1], vec=vector_size))

                for i in sdfg.arrays[dnode.data].strides:
                    if not is_int(i):
                        warnings.warn(
                            "Using the MemoryBuffering transformation is potential unsafe since {sym} is not an integer. There should be no issue if {sym} % {vec} == 0"
                            .format(sym=i, vec=vector_size))

                if self.expr_index == 0:  # Read
                    edges = state.out_edges(dnode)
                    gearbox_input_type = dtypes.vector(desc.dtype, vector_size)
                    gearbox_output_type = desc.dtype
                    gearbox_read_volume = total_size / vector_size
                    gearbox_write_volume = total_size
                else:  # Write
                    edges = state.in_edges(dnode)
                    gearbox_input_type = desc.dtype
                    gearbox_output_type = dtypes.vector(desc.dtype, vector_size)
                    gearbox_read_volume = total_size
                    gearbox_write_volume = total_size / vector_size

                input_gearbox_name, input_gearbox_newdesc = sdfg.add_stream("gearbox_input",
                                                                            gearbox_input_type,
                                                                            buffer_size=self.buffer_size,
                                                                            storage=self.storage,
                                                                            transient=True,
                                                                            find_new_name=True)

                output_gearbox_name, output_gearbox_newdesc = sdfg.add_stream("gearbox_output",
                                                                              gearbox_output_type,
                                                                              buffer_size=self.buffer_size,
                                                                              storage=self.storage,
                                                                              transient=True,
                                                                              find_new_name=True)

                read_to_gearbox = state.add_read(input_gearbox_name)
                write_from_gearbox = state.add_write(output_gearbox_name)

                gearbox = Gearbox(total_size / vector_size)

                state.add_node(gearbox)

                state.add_memlet_path(read_to_gearbox,
                                      gearbox,
                                      dst_conn="from_memory",
                                      memlet=Memlet(input_gearbox_name + "[0]", volume=gearbox_read_volume))
                state.add_memlet_path(gearbox,
                                      write_from_gearbox,
                                      src_conn="to_kernel",
                                      memlet=Memlet(output_gearbox_name + "[0]", volume=gearbox_write_volume))

                if self.expr_index == 0:
                    streams[edge] = input_gearbox_name
                    name = output_gearbox_name
                    newdesc = output_gearbox_newdesc
                else:
                    streams[edge] = output_gearbox_name
                    name = input_gearbox_name
                    newdesc = input_gearbox_newdesc

            else:

                name, newdesc = sdfg.add_stream(dnode.data,
                                                desc.dtype,
                                                buffer_size=self.buffer_size,
                                                storage=self.storage,
                                                transient=True,
                                                find_new_name=True)
                streams[edge] = name

                # Add these such that we can easily use output_gearbox_name and input_gearbox_name without using if statements
                output_gearbox_name = name
                input_gearbox_name = name

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
                replacement = state.add_read(output_gearbox_name)
                state.remove_edge(edge)
                state.add_edge(replacement, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
            else:
                replacement = state.add_write(input_gearbox_name)
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, replacement, edge.dst_conn, edge.data)

        if self.use_memory_buffering:

            arrname = str(self.access)
            vector_size = int(self.memory_buffering_target_bytes / desc.dtype.bytes)

            # Vectorize access to global array.
            dtype = sdfg.arrays[arrname].dtype
            sdfg.arrays[arrname].dtype = dtypes.vector(dtype, vector_size)
            new_shape = list(sdfg.arrays[arrname].shape)
            contigidx = sdfg.arrays[arrname].strides.index(1)
            new_shape[contigidx] /= vector_size
            try:
                new_shape[contigidx] = int(new_shape[contigidx])
            except TypeError:
                pass
            sdfg.arrays[arrname].shape = new_shape

            # Change strides
            new_strides: List = list(sdfg.arrays[arrname].strides)

            for i in range(len(new_strides)):
                if i == len(new_strides) - 1:  # Skip last dimension since it is always 1
                    continue
                new_strides[i] = new_strides[i] / vector_size
            sdfg.arrays[arrname].strides = new_strides

            post_state = get_post_state(sdfg, state)

            if post_state != None:
                # Change subset in the post state such that the correct amount of memory is copied back from the device
                for e in post_state.edges():
                    if e.data.data == self.access.data:
                        new_subset = list(e.data.subset)
                        i, j, k = new_subset[-1]
                        new_subset[-1] = (i, (j + 1) / vector_size - 1, k)
                        e.data = mm.Memlet(data=str(e.src), subset=subsets.Range(new_subset))

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

                ranges = [(p, (r[0], r[1], r[2])) for p, r in zip(map.params, map.range)]

                # Change ranges of map
                if self.use_memory_buffering:
                    # Find edges from/to map

                    edge_subset = [a_tuple[0] for a_tuple in list(innermost_edge.data.subset)]

                    # Change range of map
                    if isinstance(edge_subset[-1], symbol) and str(edge_subset[-1]) == map.params[-1]:

                        if not is_int(ranges[-1][1][1]):

                            warnings.warn(
                                "Using the MemoryBuffering transformation is potential unsafe since {sym} is not an integer. There should be no issue if {sym} % {vec} == 0"
                                .format(sym=ranges[-1][1][1].args[1], vec=vector_size))

                        ranges[-1] = (ranges[-1][0], (ranges[-1][1][0], (ranges[-1][1][1] + 1) / vector_size - 1,
                                                      ranges[-1][1][2]))

                    elif isinstance(edge_subset[-1], sympy.core.add.Add):

                        for arg in edge_subset[-1].args:
                            if isinstance(arg, symbol) and str(arg) == map.params[-1]:

                                if not is_int(ranges[-1][1][1]):
                                    warnings.warn(
                                        "Using the MemoryBuffering transformation is potential unsafe since {sym} is not an integer. There should be no issue if {sym} % {vec} == 0"
                                        .format(sym=ranges[-1][1][1].args[1], vec=vector_size))

                                ranges[-1] = (ranges[-1][0],
                                              (ranges[-1][1][0], (ranges[-1][1][1] + 1) / vector_size - 1,
                                               ranges[-1][1][2]))

                maps.append(state.add_map(f'__s{opname}_{mapname}', ranges, map.schedule))
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
        return [sdutil.node_path_graph(cls.first, cls.access, cls.second)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        access = self.access
        # Make sure the access node is only accessed once (read or write),
        # and not at the same time
        if graph.in_degree(access) > 1 or graph.out_degree(access) > 1:
            return False

        # If already a stream, skip
        desc = sdfg.arrays[access.data]
        if isinstance(desc, data.Stream):
            return False

        # If this check is in the code, almost all applications of StreamingComposition must be permissive
        # if not permissive and desc.transient:
        #     return False

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
            if desc.transient:
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
