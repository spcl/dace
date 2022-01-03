# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes and functions that implement the GPU transformation
    (with local storage). """

import copy

from dace import data, dtypes, registry, sdfg as sd, subsets as sbs, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import Property, make_properties
from dace.config import Config


def in_scope(graph, node, parent):
    """ Returns True if `node` is in the scope of `parent`. """
    scope_dict = graph.scope_dict()
    scope = scope_dict[node]
    while scope is not None:
        if scope == parent:
            return True
        scope = scope_dict[scope]
    return False


def in_path(path, edge, nodetype, forward=True):
    if not forward:
        path.reverse()
    start = path.index(edge)
    for e in path[start:]:
        if isinstance(e.dst, nodetype):
            return True
    return False


@make_properties
class GPUTransformLocalStorage(transformation.SingleStateTransformation):
    """Implements the GPUTransformLocalStorage transformation.

        Similar to GPUTransformMap, but takes multiple maps leading from the
        same data node into account, creating a local storage for each range.

        @see: GPUTransformMap
    """

    _arrays_removed = 0
    _maps_transformed = 0

    fullcopy = Property(desc="Copy whole arrays rather than used subset", dtype=bool, default=False)

    nested_seq = Property(
        desc="Makes nested code semantically-equivalent to single-core code,"
        "transforming nested maps and memory into sequential and "
        "local memory respectively.",
        dtype=bool,
        default=True,
    )

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    import dace.libraries.standard as stdlib  # Avoid import loop
    _reduce = stdlib.Reduce("lambda: None", None)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(GPUTransformLocalStorage._map_entry),
            sdutil.node_path_graph(GPUTransformLocalStorage._reduce),
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        if expr_index == 0:
            map_entry = graph.nodes()[candidate[GPUTransformLocalStorage._map_entry]]
            candidate_map = map_entry.map

            # Disallow GPUTransform on nested maps in permissive mode
            if not permissive:
                if graph.entry_node(map_entry) is not None:
                    return False

            # Map schedules that are disallowed to transform to GPUs
            if (candidate_map.schedule == dtypes.ScheduleType.MPI
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_Device
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
                    or candidate_map.schedule == dtypes.ScheduleType.Sequential):
                return False

            # Dynamic map ranges cannot become kernels
            if sd.has_dynamic_map_inputs(graph, map_entry):
                return False

            # Recursively check parent for GPU schedules
            sdict = graph.scope_dict()
            current_node = map_entry
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

            # Ensure that map does not include internal arrays that are
            # allocated on non-default space
            subgraph = graph.scope_subgraph(map_entry)
            for node in subgraph.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage != dtypes.StorageType.Default
                        and node.desc(sdfg).storage != dtypes.StorageType.Register):
                    return False

            # If one of the outputs is a stream, do not match
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                dst = graph.memlet_path(edge)[-1].dst
                if (isinstance(dst, nodes.AccessNode) and isinstance(sdfg.arrays[dst.data], data.Stream)):
                    return False

            return True
        elif expr_index == 1:
            reduce = graph.nodes()[candidate[GPUTransformLocalStorage._reduce]]

            # Recursively check parent for GPU schedules
            sdict = graph.scope_dict()
            current_node = sdict[reduce]
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

            return True

    @staticmethod
    def match_to_str(graph, candidate):
        if GPUTransformLocalStorage._reduce in candidate:
            return str(graph.nodes()[candidate[GPUTransformLocalStorage._reduce]])
        else:
            map_entry = graph.nodes()[candidate[GPUTransformLocalStorage._map_entry]]
            return str(map_entry)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        if self.expr_index == 0:
            cnode: nodes.MapEntry = graph.nodes()[self.subgraph[GPUTransformLocalStorage._map_entry]]
            # Change schedule
            cnode.schedule = dtypes.ScheduleType.GPU_Device
            exit_node = graph.exit_node(cnode)
        else:
            cnode: nodes.LibraryNode = graph.nodes()[self.subgraph[GPUTransformLocalStorage._reduce]]
            # Change schedule
            cnode.schedule = dtypes.ScheduleType.GPU_Default
            exit_node = cnode

        if Config.get_bool("debugprint"):
            GPUTransformLocalStorage._maps_transformed += 1
        # If nested graph is designated as sequential, transform schedules and
        # storage from Default to Sequential/Register
        if self.nested_seq and self.expr_index == 0:
            for node in graph.scope_subgraph(cnode).nodes():
                if isinstance(node, nodes.AccessNode):
                    arr = node.desc(sdfg)
                    if arr.storage == dtypes.StorageType.Default:
                        arr.storage = dtypes.StorageType.Register
                elif isinstance(node, nodes.MapEntry):
                    if node.map.schedule == dtypes.ScheduleType.Default:
                        node.map.schedule = dtypes.ScheduleType.Sequential

        gpu_storage_types = [
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.GPU_Shared,
        ]

        #######################################################
        # Add GPU copies of CPU arrays (i.e., not already on GPU)

        # First, understand which arrays to clone
        all_out_edges = []
        all_out_edges.extend(list(graph.out_edges(exit_node)))
        in_arrays_to_clone = set()
        out_arrays_to_clone = set()
        for e in graph.in_edges(cnode):
            data_node = sd.find_input_arraynode(graph, e)
            if data_node.desc(sdfg).storage not in gpu_storage_types:
                in_arrays_to_clone.add((data_node, e.data))
        for e in all_out_edges:
            data_node = sd.find_output_arraynode(graph, e)
            if data_node.desc(sdfg).storage not in gpu_storage_types:
                out_arrays_to_clone.add((data_node, e.data))

        if Config.get_bool("debugprint"):
            GPUTransformLocalStorage._arrays_removed += len(in_arrays_to_clone) + len(out_arrays_to_clone)

        # Second, create a GPU clone of each array
        # TODO: Overapproximate union of memlets
        cloned_arrays = {}
        in_cloned_arraynodes = {}
        out_cloned_arraynodes = {}
        for array_node, memlet in in_arrays_to_clone:
            array = array_node.desc(sdfg)
            cloned_name = "gpu_" + array_node.data
            for i, r in enumerate(memlet.bounding_box_size()):
                size = symbolic.overapproximate(r)
                try:
                    if int(size) == 1:
                        suffix = []
                        for c in str(memlet.subset[i][0]):
                            if c.isalpha() or c.isdigit() or c == "_":
                                suffix.append(c)
                            elif c == "+":
                                suffix.append("p")
                            elif c == "-":
                                suffix.append("m")
                            elif c == "*":
                                suffix.append("t")
                            elif c == "/":
                                suffix.append("d")
                        cloned_name += "_" + "".join(suffix)
                except:
                    continue
            if cloned_name in sdfg.arrays.keys():
                cloned_array = sdfg.arrays[cloned_name]
            elif array_node.data in cloned_arrays:
                cloned_array = cloned_arrays[array_node.data]
            else:
                full_shape = []
                for r in memlet.bounding_box_size():
                    size = symbolic.overapproximate(r)
                    try:
                        full_shape.append(int(size))
                    except:
                        full_shape.append(size)
                actual_dims = [idx for idx, r in enumerate(full_shape) if not (isinstance(r, int) and r == 1)]
                if len(actual_dims) == 0:  # abort
                    actual_dims = [len(full_shape) - 1]
                if isinstance(array, data.Scalar):
                    sdfg.add_array(name=cloned_name,
                                   shape=[1],
                                   dtype=array.dtype,
                                   transient=True,
                                   storage=dtypes.StorageType.GPU_Global)
                elif isinstance(array, data.Stream):
                    sdfg.add_stream(name=cloned_name,
                                    dtype=array.dtype,
                                    shape=[full_shape[d] for d in actual_dims],
                                    veclen=array.veclen,
                                    buffer_size=array.buffer_size,
                                    storage=dtypes.StorageType.GPU_Global,
                                    transient=True,
                                    offset=[array.offset[d] for d in actual_dims])
                else:
                    sdfg.add_array(
                        name=cloned_name,
                        shape=[full_shape[d] for d in actual_dims],
                        dtype=array.dtype,
                        transient=True,
                        storage=dtypes.StorageType.GPU_Global,
                        allow_conflicts=array.allow_conflicts,
                        strides=[array.strides[d] for d in actual_dims],
                        offset=[array.offset[d] for d in actual_dims],
                    )
                cloned_arrays[array_node.data] = cloned_name
            cloned_node = type(array_node)(cloned_name)

            in_cloned_arraynodes[array_node.data] = cloned_node
        for array_node, memlet in out_arrays_to_clone:
            array = array_node.desc(sdfg)
            cloned_name = "gpu_" + array_node.data
            for i, r in enumerate(memlet.bounding_box_size()):
                size = symbolic.overapproximate(r)
                try:
                    if int(size) == 1:
                        suffix = []
                        for c in str(memlet.subset[i][0]):
                            if c.isalpha() or c.isdigit() or c == "_":
                                suffix.append(c)
                            elif c == "+":
                                suffix.append("p")
                            elif c == "-":
                                suffix.append("m")
                            elif c == "*":
                                suffix.append("t")
                            elif c == "/":
                                suffix.append("d")
                        cloned_name += "_" + "".join(suffix)
                except:
                    continue
            if cloned_name in sdfg.arrays.keys():
                cloned_array = sdfg.arrays[cloned_name]
            elif array_node.data in cloned_arrays:
                cloned_array = cloned_arrays[array_node.data]
            else:
                full_shape = []
                for r in memlet.bounding_box_size():
                    size = symbolic.overapproximate(r)
                    try:
                        full_shape.append(int(size))
                    except:
                        full_shape.append(size)
                actual_dims = [idx for idx, r in enumerate(full_shape) if not (isinstance(r, int) and r == 1)]
                if len(actual_dims) == 0:  # abort
                    actual_dims = [len(full_shape) - 1]
                if isinstance(array, data.Scalar):
                    sdfg.add_array(name=cloned_name,
                                   shape=[1],
                                   dtype=array.dtype,
                                   transient=True,
                                   storage=dtypes.StorageType.GPU_Global)
                elif isinstance(array, data.Stream):
                    sdfg.add_stream(name=cloned_name,
                                    dtype=array.dtype,
                                    shape=[full_shape[d] for d in actual_dims],
                                    veclen=array.veclen,
                                    buffer_size=array.buffer_size,
                                    storage=dtypes.StorageType.GPU_Global,
                                    transient=True,
                                    offset=[array.offset[d] for d in actual_dims])
                else:
                    sdfg.add_array(
                        name=cloned_name,
                        shape=[full_shape[d] for d in actual_dims],
                        dtype=array.dtype,
                        transient=True,
                        storage=dtypes.StorageType.GPU_Global,
                        allow_conflicts=array.allow_conflicts,
                        strides=[array.strides[d] for d in actual_dims],
                        offset=[array.offset[d] for d in actual_dims],
                    )
                cloned_arrays[array_node.data] = cloned_name
            cloned_node = type(array_node)(cloned_name)
            cloned_node.setzero = True

            out_cloned_arraynodes[array_node.data] = cloned_node

        # Third, connect the cloned arrays to the originals
        for array_name, node in in_cloned_arraynodes.items():
            graph.add_node(node)
            is_scalar = isinstance(sdfg.arrays[array_name], data.Scalar)
            for edge in graph.in_edges(cnode):
                if edge.data.data == array_name:
                    newmemlet = copy.deepcopy(edge.data)
                    newmemlet.data = node.data

                    if is_scalar:
                        newmemlet.subset = sbs.Indices([0])
                    else:
                        offset = []
                        lost_dims = []
                        lost_ranges = []
                        newsubset = [None] * len(edge.data.subset)
                        for ind, r in enumerate(edge.data.subset):
                            offset.append(r[0])
                            if isinstance(edge.data.subset[ind], tuple):
                                begin = edge.data.subset[ind][0] - r[0]
                                end = edge.data.subset[ind][1] - r[0]
                                step = edge.data.subset[ind][2]
                                if begin == end:
                                    lost_dims.append(ind)
                                    lost_ranges.append((begin, end, step))
                                else:
                                    newsubset[ind] = (begin, end, step)
                            else:
                                newsubset[ind] -= r[0]
                        if len(lost_dims) == len(edge.data.subset):
                            lost_dims.pop()
                            newmemlet.subset = type(edge.data.subset)([lost_ranges[-1]])
                        else:
                            newmemlet.subset = type(edge.data.subset)([r for r in newsubset if r is not None])

                    graph.add_edge(node, None, edge.dst, edge.dst_conn, newmemlet)

                    for e in graph.bfs_edges(edge.dst, reverse=False):
                        parent, _, _child, _, memlet = e
                        if parent != edge.dst and not in_scope(graph, parent, edge.dst):
                            break
                        if memlet.data != edge.data.data:
                            continue
                        path = graph.memlet_path(e)
                        if not isinstance(path[-1].dst, nodes.CodeNode):
                            if in_path(path, e, nodes.ExitNode, forward=True):
                                if isinstance(parent, nodes.CodeNode):
                                    # Output edge
                                    break
                                else:
                                    continue
                        if is_scalar:
                            memlet.subset = sbs.Indices([0])
                        else:
                            newsubset = [None] * len(memlet.subset)
                            for ind, r in enumerate(memlet.subset):
                                if ind in lost_dims:
                                    continue
                                if isinstance(memlet.subset[ind], tuple):
                                    begin = r[0] - offset[ind]
                                    end = r[1] - offset[ind]
                                    step = r[2]
                                    newsubset[ind] = (begin, end, step)
                                else:
                                    newsubset[ind] = (
                                        r - offset[ind],
                                        r - offset[ind],
                                        1,
                                    )
                            memlet.subset = type(edge.data.subset)([r for r in newsubset if r is not None])
                        memlet.data = node.data

                    if self.fullcopy:
                        edge.data.subset = sbs.Range.from_array(node.desc(sdfg))
                    edge.data.other_subset = newmemlet.subset
                    graph.add_edge(edge.src, edge.src_conn, node, None, edge.data)
                    graph.remove_edge(edge)

        for array_name, node in out_cloned_arraynodes.items():
            graph.add_node(node)
            is_scalar = isinstance(sdfg.arrays[array_name], data.Scalar)
            for edge in all_out_edges:
                if edge.data.data == array_name:
                    newmemlet = copy.deepcopy(edge.data)
                    newmemlet.data = node.data

                    if is_scalar:
                        newmemlet.subset = sbs.Indices([0])
                    else:
                        offset = []
                        lost_dims = []
                        lost_ranges = []
                        newsubset = [None] * len(edge.data.subset)
                        for ind, r in enumerate(edge.data.subset):
                            offset.append(r[0])
                            if isinstance(edge.data.subset[ind], tuple):
                                begin = edge.data.subset[ind][0] - r[0]
                                end = edge.data.subset[ind][1] - r[0]
                                step = edge.data.subset[ind][2]
                                if begin == end:
                                    lost_dims.append(ind)
                                    lost_ranges.append((begin, end, step))
                                else:
                                    newsubset[ind] = (begin, end, step)
                            else:
                                newsubset[ind] -= r[0]
                        if len(lost_dims) == len(edge.data.subset):
                            lost_dims.pop()
                            newmemlet.subset = type(edge.data.subset)([lost_ranges[-1]])
                        else:
                            newmemlet.subset = type(edge.data.subset)([r for r in newsubset if r is not None])

                    graph.add_edge(edge.src, edge.src_conn, node, None, newmemlet)

                    end_node = graph.entry_node(edge.src)
                    for e in graph.bfs_edges(edge.src, reverse=True):
                        parent, _, _child, _, memlet = e
                        if parent == end_node:
                            break
                        if memlet.data != edge.data.data:
                            continue
                        path = graph.memlet_path(e)
                        if not isinstance(path[0].dst, nodes.CodeNode):
                            if in_path(path, e, nodes.EntryNode, forward=False):
                                if isinstance(parent, nodes.CodeNode):
                                    # Output edge
                                    break
                                else:
                                    continue
                        if is_scalar:
                            memlet.subset = sbs.Indices([0])
                        else:
                            newsubset = [None] * len(memlet.subset)
                            for ind, r in enumerate(memlet.subset):
                                if ind in lost_dims:
                                    continue
                                if isinstance(memlet.subset[ind], tuple):
                                    begin = r[0] - offset[ind]
                                    end = r[1] - offset[ind]
                                    step = r[2]
                                    newsubset[ind] = (begin, end, step)
                                else:
                                    newsubset[ind] = (
                                        r - offset[ind],
                                        r - offset[ind],
                                        1,
                                    )
                            memlet.subset = type(edge.data.subset)([r for r in newsubset if r is not None])
                        memlet.data = node.data

                    edge.data.wcr = None
                    if self.fullcopy:
                        edge.data.subset = sbs.Range.from_array(node.desc(sdfg))
                    edge.data.other_subset = newmemlet.subset
                    graph.add_edge(node, None, edge.dst, edge.dst_conn, edge.data)
                    graph.remove_edge(edge)

        # Fourth, replace memlet arrays as necessary
        if self.expr_index == 0:
            scope_subgraph = graph.scope_subgraph(cnode)
            for edge in scope_subgraph.edges():
                if edge.data.data is not None and edge.data.data in cloned_arrays:
                    edge.data.data = cloned_arrays[edge.data.data]
