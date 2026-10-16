# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Fuses a forward FFT, a computation in the frequency domain, and an inverse FFT into a single kernel in which each GPU
thread block (or CPU map iteration) computes one transform in fast local memory.
"""

import copy
import dataclasses
import math

from dace import SDFG, Memlet, SDFGState, data, dtypes, nodes, properties, subsets, symbolic
from dace.libraries.fft.algorithms import stockham
from dace.libraries.fft.nodes import fft as fft_nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@dataclasses.dataclass
class _Crop:
    """
    A map that reads the result of the inverse FFT (e.g., cropping a window of it) and is fused into the kernel.
    """

    map_entry: nodes.MapEntry
    map_exit: nodes.MapExit
    #: The map parameter that indexes each dimension of the result.
    params: list[str]
    #: The array that the map writes.
    output: nodes.AccessNode


@dataclasses.dataclass
class _Match:
    """
    The nodes of a matched ``source -> FFT -> spectrum -> map -> filtered -> IFFT -> result`` pattern.
    """

    source: nodes.AccessNode
    filtered: nodes.AccessNode
    inverse: fft_nodes.IFFT
    result: nodes.AccessNode
    map_exit: nodes.MapExit
    axes: list[int]
    #: The map parameter that indexes each dimension of the arrays.
    params: list[str]
    #: On the CPU, the number of adjacent transforms that an iteration computes (see ``stockham.local_lanes``), or
    #: None on the GPU.
    lanes: int | None
    #: Whether the source is shorter than the transforms along the axes, i.e., zero-padded (see
    #: ``stockham.padded_dims``).
    padded: bool
    #: The map that reads the result, if it is fused too.
    crop: _Crop | None


@properties.make_properties
@transformation.explicit_cf_compatible
class BlockLocalFFTFusion(transformation.SingleStateTransformation):
    """
    Fuses an FFT, a map, and an inverse FFT (``ifft(f(fft(x)))``) into one GPU kernel with one thread block per
    transform. The block computes the Stockham stages of the FFT, the map, and the stages of the inverse FFT in shared
    memory, synchronizing between them, so that only the input and the output of the inverse FFT are in global memory.
    On the CPU, the kernel is a parallel map in which each iteration computes the same steps for one transform (or a
    few adjacent ones) in buffers on the stack, so that the arrays are read and written once.

    The map must compute each element of its output from the spectrum of the same transform, e.g., a pointwise product
    with a filter or a stencil along the transformed axes. It may read arbitrary other arrays. Both transforms must be
    over the same axes of arrays in GPU global (or host) memory, and a transform must fit into the shared memory of a
    thread block (see ``stockham.fits_block_local``), or into the stack buffers of a CPU map iteration (see
    ``stockham.local_lanes``).

    The input of the FFT may be zero-padded (e.g., ``np.fft.fft2(x, s=...)``), and a map that reads the result of the
    inverse FFT at the index of its transform, shifted by constants along the transformed axes (e.g., the window of a
    convolution, ``y[i, j] = result[i + 3, j + 3].real``), is fused too if the result is only read by it. The kernel
    then reads the unpadded input and writes the output of that map, so that neither the padded input nor the whole
    result is in global memory.
    """

    forward = transformation.PatternNode(fft_nodes.FFT)
    spectrum = transformation.PatternNode(nodes.AccessNode)
    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.forward, cls.spectrum, cls.map_entry)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        return self._match(graph, sdfg) is not None

    def _match(self, state: SDFGState, sdfg: SDFG) -> _Match | None:
        forward, spectrum, map_entry = self.forward, self.spectrum, self.map_entry
        gpu = map_entry.map.schedule == dtypes.ScheduleType.GPU_Device
        cpu_schedules = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)
        if state.entry_node(forward) is not None or not (gpu or map_entry.map.schedule in cpu_schedules):
            return None

        # Linear chain: source -> FFT -> spectrum -> map -> filtered -> IFFT -> result, with full-array memlets
        source_edge = _single_edge(state.in_edges(forward), "_inp")
        if source_edge is None or state.out_degree(forward) != 1:
            return None
        map_exit = state.exit_node(map_entry)
        if state.out_degree(map_exit) != 1:
            return None
        filtered = state.out_edges(map_exit)[0].dst
        if not isinstance(filtered, nodes.AccessNode) or state.out_degree(filtered) != 1:
            return None
        inverse = state.out_edges(filtered)[0].dst
        if not isinstance(inverse, fft_nodes.IFFT) or state.in_degree(inverse) != 1 or state.out_degree(inverse) != 1:
            return None
        result_edge = state.out_edges(inverse)[0]
        source, result = source_edge.src, result_edge.dst
        if not isinstance(source, nodes.AccessNode) or not isinstance(result, nodes.AccessNode):
            return None
        if source.data == result.data:
            return None
        chain_edges = [source_edge, state.out_edges(forward)[0], state.out_edges(filtered)[0], result_edge]
        if any(not _is_full(sdfg, edge.data) for edge in chain_edges):
            return None
        if state.in_degree(spectrum) != 1 or state.out_degree(spectrum) != 1:
            return None
        if not _is_local_transient(sdfg, state, spectrum) or not _is_local_transient(sdfg, state, filtered):
            return None

        # Same shape, complex type, and transformed axes; the input may also be real, and zero-padded
        descs = [sdfg.arrays[node.data] for node in (source, spectrum, filtered, result)]
        if any(not isinstance(desc, data.Array) for desc in descs):
            return None
        storages = (dtypes.StorageType.GPU_Global,) if gpu else stockham.HOST_STORAGES
        if any(desc.storage not in storages for desc in (descs[0], descs[3])):
            return None
        shape = tuple(descs[1].shape)
        if any(tuple(desc.shape) != shape for desc in descs[2:]):
            return None
        dtype = descs[1].dtype
        if dtype not in (dtypes.complex64, dtypes.complex128) or any(desc.dtype != dtype for desc in descs[2:]):
            return None
        axes = fft_nodes.transformed_axes(forward, len(shape))
        if fft_nodes.transformed_axes(inverse, len(shape)) != axes:
            return None
        try:
            padded = bool(stockham.padded_dims(descs[0].shape, shape, axes, sdfg.constants))
        except ValueError:
            return None
        lanes = None
        if gpu and not stockham.fits_block_local(shape, axes, dtype, sdfg.constants):
            return None
        if not gpu:
            lanes = stockham.local_lanes(shape, axes, dtype, sdfg.constants)
            if lanes is None:
                return None

        params = _map_dimensions(sdfg, state, map_entry, filtered.data, shape)
        if params is None or not _reads_one_transform(state, map_entry, spectrum.data, axes, params):
            return None

        # Other inputs of the map are read-only arrays, and its contents can run in device code
        for edge in state.in_edges(map_entry):
            if edge.data.is_empty():
                continue
            if not edge.dst_conn or not edge.dst_conn.startswith("IN_") or not isinstance(edge.src, nodes.AccessNode):
                return None
            if edge.src is not spectrum and edge.data.data in (filtered.data, result.data):
                return None
        for node in state.scope_subgraph(map_entry, include_entry=False, include_exit=False).nodes():
            if isinstance(node, nodes.LibraryNode):
                return None
            if isinstance(node, nodes.AccessNode) and node.data in (spectrum.data, filtered.data, result.data):
                return None
            if isinstance(node, nodes.MapEntry) and node.map.schedule not in (
                dtypes.ScheduleType.Default,
                dtypes.ScheduleType.Sequential,
            ):
                return None

        # Nested SDFGs in the map (e.g., from conditionals) must access the spectrum and the output in a way that
        # can be restated on the buffers of a single transform
        batch_dims = {dim for dim in range(len(shape)) if dim not in axes}
        for edge, node, conn in _nested_connections(
            state.scope_subgraph(map_entry).edges(), (spectrum.data, filtered.data)
        ):
            if (lanes or 1) > 1 or not _restate_nested(node, conn, edge.data.subset, batch_dims, None):
                return None

        crop = self._crop(state, sdfg, result, shape, axes, gpu, source.data)
        return _Match(source, filtered, inverse, result, map_exit, axes, params, lanes, padded, crop)

    def _crop(
        self,
        state: SDFGState,
        sdfg: SDFG,
        result: nodes.AccessNode,
        shape: tuple,
        axes: list[int],
        gpu: bool,
        source: str,
    ) -> _Crop | None:
        """
        Returns the map that reads the result of the inverse FFT if it can be fused too (see the class documentation):
        the result is only read by the map, at the map parameters in the dimensions that are not transformed (whose
        ranges are the whole dimensions), and at a map parameter plus a constant, within the result, along each
        transformed axis. The map writes one array.
        """
        if not _is_local_transient(sdfg, state, result) or state.out_degree(result) != 1:
            return None
        map_entry = state.out_edges(result)[0].dst
        if not isinstance(map_entry, nodes.MapEntry) or state.entry_node(map_entry) is not None:
            return None
        schedules = (
            (dtypes.ScheduleType.GPU_Device,)
            if gpu
            else (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)
        )
        if map_entry.map.schedule not in schedules:
            return None
        map_exit = state.exit_node(map_entry)
        if state.out_degree(map_exit) != 1:
            return None
        output = state.out_edges(map_exit)[0].dst
        if (
            not isinstance(output, nodes.AccessNode)
            or output.data in (result.data, source)
            or state.in_degree(output) != 1
        ):
            return None
        if not isinstance(sdfg.arrays[output.data], data.Array):
            return None
        for edge in state.in_edges(map_entry):
            if edge.data.is_empty():
                continue
            if not edge.dst_conn or not edge.dst_conn.startswith("IN_") or not isinstance(edge.src, nodes.AccessNode):
                return None
            if edge.data.data == output.data or (edge.src is not result and edge.data.data == result.data):
                return None

        scope = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        for node in scope.nodes():
            if isinstance(node, nodes.LibraryNode):
                return None
            if isinstance(node, nodes.AccessNode) and node.data in (result.data, output.data):
                return None
            if isinstance(node, nodes.MapEntry) and node.map.schedule not in (
                dtypes.ScheduleType.Default,
                dtypes.ScheduleType.Sequential,
            ):
                return None
        scope_edges = state.scope_subgraph(map_entry).edges()
        if _nested_connections(scope_edges, (result.data,)):
            return None

        # The map parameter of each dimension of the result, the same in all reads
        map_params = [symbolic.pystr_to_symbolic(param) for param in map_entry.map.params]
        params = [None] * len(shape)
        for edge in scope_edges:
            if edge.data.data != result.data or edge.src is result:
                continue
            if edge.data.subset.num_elements() != 1:
                return None
            for dim, (begin, _, _) in enumerate(edge.data.subset.ndrange()):
                used = [param for param in map_params if param in begin.free_symbols]
                if len(used) != 1 or (params[dim] is not None and params[dim] != str(used[0])):
                    return None
                param = used[0]
                offset = symbolic.simplify(begin - param)
                if offset.free_symbols or (dim not in axes and offset != 0):
                    return None
                first, last, step = map_entry.map.range[map_entry.map.params.index(str(param))]
                if step != 1:
                    return None
                if dim in axes and ((first + offset < 0) == True or (last + offset >= shape[dim]) == True):
                    return None
                if dim not in axes and (first != 0 or symbolic.simplify(last + 1 - shape[dim]) != 0):
                    return None
                params[dim] = str(param)
        if any(param is None for param in params) or sorted(params) != sorted(map_entry.map.params):
            return None
        return _Crop(map_entry, map_exit, params, output)

    def apply(self, state: SDFGState, sdfg: SDFG) -> nodes.NestedSDFG:
        match = self._match(state, sdfg)
        forward, spectrum, map_entry = self.forward, self.spectrum, self.map_entry
        source, result, crop = match.source, match.result, match.crop
        output = crop.output if crop is not None else result
        shape = tuple(sdfg.arrays[spectrum.data].shape)
        dtype = sdfg.arrays[spectrum.data].dtype
        axes = match.axes
        gpu = match.lanes is None
        lanes = match.lanes or 1
        block_shape = stockham.block_local_shape(shape, axes, sdfg.constants, lanes)
        stages = stockham.plan_stages(block_shape, axes, {})
        threads = stockham.block_local_threads(stages, block_shape) if gpu else None
        padding = 0
        if not gpu and len(axes) == 1 and self._pointwise_along(state, match, axes[0]):
            padding = stockham.local_padding(stages)

        # Twiddle factor tables, from the program's cache
        forward_tables = stockham.add_twiddle_tables(state, stages, -1, dtype, gpu, "__fft_block_tw")
        inverse_tables = stockham.add_twiddle_tables(state, stages, 1, dtype, gpu, "__ifft_block_tw")

        # The nested SDFG of one thread block: one state per stage and one for the map
        other_inputs = {}
        map_entries = [map_entry] + ([crop.map_entry] if crop is not None else [])
        for entry in map_entries:
            for edge in state.in_edges(entry):
                if not edge.data.is_empty() and edge.src is not spectrum and edge.src is not result:
                    other_inputs.setdefault(edge.data.data, edge.src)
        table_nodes = list(forward_tables.values()) + list(inverse_tables.values())
        outer_names = {source.data, output.data, *other_inputs, *(node.data for node in table_nodes)}
        block, batch_symbols, batch_index, buffers = stockham.block_local_sdfg(
            "fft_block_local",
            {name: sdfg.arrays[name] for name in outer_names},
            shape,
            axes,
            dtype,
            sdfg.constants,
            storage=dtypes.StorageType.GPU_Shared if gpu else dtypes.StorageType.Register(force=True),
            lanes=lanes,
            padding=padding,
        )

        def stage_steps(sign: int, libnode: nodes.LibraryNode, tables: dict) -> list:
            names = {key: node.data for key, node in tables.items()}
            factor = None if libnode.factor == 1 else libnode.factor
            return [(stage, sign, names, factor if s == len(stages) - 1 else None) for s, stage in enumerate(stages)]

        # Steps: zero-padding the input into a buffer, the stages of the FFT, the map, the stages of the inverse FFT,
        # and the map that reads its result
        steps = stage_steps(-1, forward, forward_tables) + [None] + stage_steps(1, match.inverse, inverse_tables)
        first = 1 if match.padded else 0
        steps = [None] * first + steps + ([None] if crop is not None else [])
        block_states = stockham.add_block_local_steps(
            block, steps, source.data, output.data, buffers, block_shape, batch_index, threads, padding
        )
        if match.padded:
            pad_state, pad_src, pad_dst = block_states[0]
            source_shape = sdfg.arrays[source.data].shape
            stockham.add_padding_step(
                pad_state, pad_src, pad_dst, block_shape, batch_index, source_shape, padding, threads
            )
        comp_state, comp_src, comp_dst = block_states[first + len(stages)]

        dependencies = [edge.src for entry in map_entries for edge in state.in_edges(entry) if edge.data.is_empty()]
        buffer_names = {spectrum.data: comp_src, match.filtered.data: comp_dst}
        self._move_map(
            sdfg,
            state,
            comp_state,
            map_entry,
            match.map_exit,
            match.params,
            buffer_names,
            match.axes,
            block_shape,
            batch_symbols,
            threads,
            padding,
        )
        if crop is not None:
            crop_state, crop_src, _ = block_states[-1]
            self._move_map(
                sdfg,
                state,
                crop_state,
                crop.map_entry,
                crop.map_exit,
                crop.params,
                {result.data: crop_src},
                match.axes,
                block_shape,
                batch_symbols,
                threads,
                padding,
            )

        # Remove the transforms and their intermediate arrays
        removed = [forward, spectrum, match.filtered, match.inverse] + ([result] if crop is not None else [])
        state.remove_nodes_from(removed)
        for node in removed[1:]:
            if isinstance(node, nodes.AccessNode):
                sdfg.remove_data(node.data, validate=False)

        # The kernel: one thread block (or CPU map iteration) per transform
        input_nodes = {source.data: source, **other_inputs, **{node.data: node for node in table_nodes}}
        kernel_entry, nsdfg = stockham.add_block_local_kernel(
            state,
            block,
            input_nodes,
            output,
            shape,
            batch_symbols,
            dtypes.ScheduleType.GPU_Device if gpu else dtypes.ScheduleType.Default,
            lanes,
            sdfg.constants,
        )
        for node in dependencies:
            state.add_nedge(node, kernel_entry, Memlet())
        return nsdfg

    def _pointwise_along(self, state: SDFGState, match: _Match, axis: int) -> bool:
        """
        Returns True if the map accesses the spectrum and its output at single indices along the axis, outside of
        nested SDFGs, so that the accesses can be restated on buffers that are split into padded rows.
        """
        names = (self.spectrum.data, match.filtered.data)
        scope_edges = state.scope_subgraph(self.map_entry).edges()
        if _nested_connections(scope_edges, names):
            return False
        return all(
            edge.data.subset[axis][0] == edge.data.subset[axis][1] for edge in scope_edges if edge.data.data in names
        )

    def _move_map(
        self,
        sdfg: SDFG,
        state: SDFGState,
        comp_state: SDFGState,
        map_entry: nodes.MapEntry,
        map_exit: nodes.MapExit,
        params: list[str],
        buffer_names: dict[str, str],
        axes: list[int],
        block_shape: list[int],
        batch_symbols: dict[int, str],
        threads: int | None,
        padding: int,
    ) -> None:
        """
        Moves a map from the parent state into the state of the nested SDFG that computes it, accessing the given
        arrays of the parent in buffers instead (e.g., reading the spectrum from one buffer and writing to another).
        On the GPU, the threads of the block stride over the iterations of the transformed dimensions, and each
        iteration runs the original map for a single point. On the CPU, the map runs sequentially over the transforms
        of the iteration.

        :param params: The map parameter that indexes each dimension of the arrays.
        :param buffer_names: The buffer of each array of the parent that the map accesses in a buffer.
        :param axes: The transformed axes.
        """
        scope = state.scope_subgraph(map_entry)
        scope_nodes = list(scope.nodes())
        scope_edges = list(scope.edges())
        in_edges = list(state.in_edges(map_entry))
        out_edge = state.out_edges(map_exit)[0]
        out_conn, out_data = out_edge.src_conn, out_edge.data.data
        state.remove_nodes_from(scope_nodes)

        batch_dims = set(batch_symbols)

        def localize(memlet: Memlet) -> Memlet:
            """
            Restates an access to an array that is in a buffer: relative to the first transform of the block in the
            dimensions that are not transformed, and split into padded rows along the axis.
            """
            memlet = copy.deepcopy(memlet)
            if memlet.data not in buffer_names:
                return memlet
            memlet.data = buffer_names[memlet.data]
            ranges = []
            for dim, (begin, end, step) in enumerate(memlet.subset.ndrange()):
                if dim in batch_dims and block_shape[dim] == 1:
                    ranges.append((0, 0, 1))
                elif dim in batch_dims:
                    first = symbolic.pystr_to_symbolic(batch_symbols[dim])
                    ranges.append((begin - first, end - first, step))
                elif padding and dim in axes:
                    row, column = symbolic.int_floor(begin, padding), begin % padding
                    ranges += [(row, row, 1), (column, column, 1)]
                else:
                    ranges.append((begin, end, step))
            memlet.subset = subsets.Range(ranges)
            return memlet

        for node in scope_nodes:
            comp_state.add_node(node)

        # Arrays that are local to the map body move with it
        block = comp_state.sdfg
        for name in {node.data for node in scope_nodes if isinstance(node, nodes.AccessNode)}:
            if name in block.arrays:
                continue
            block.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))
            if not any(node.data == name for other in sdfg.states() for node in other.data_nodes()):
                sdfg.remove_data(name, validate=False)
        for edge, node, conn in _nested_connections(scope_edges, tuple(buffer_names)):
            _restate_nested(node, conn, edge.data.subset, batch_dims, block.arrays[buffer_names[edge.data.data]])
        for edge in scope_edges:
            comp_state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, localize(edge.data))

        # Parameters of the dimensions that are not transformed iterate over the transforms of the block; the others
        # keep their ranges
        param_dim = {param: dim for dim, param in enumerate(params)}
        original = dict(zip(map_entry.map.params, map_entry.map.range))
        ranges = []
        if threads is None:
            for param in map_entry.map.params:
                dim = param_dim[param]
                if dim in batch_symbols:
                    first = symbolic.pystr_to_symbolic(batch_symbols[dim])
                    ranges.append((first, first + block_shape[dim] - 1, 1))
                else:
                    ranges.append(original[param])
            work = 1
        else:
            # Each iteration of the map computes one point: its batch index is the block's, and its index in the
            # transformed dimensions is delinearized from the iteration of the thread-block loop
            loop_params = [param for param in map_entry.map.params if param_dim[param] not in batch_symbols]
            extents = [original[param][1] - original[param][0] + 1 for param in loop_params]
            loop_index = dict(zip(loop_params, stockham.delinearize("__u", extents)))
            for param in map_entry.map.params:
                dim = param_dim[param]
                if dim in batch_symbols:
                    index = symbolic.pystr_to_symbolic(batch_symbols[dim])
                else:
                    index = original[param][0] + symbolic.pystr_to_symbolic(loop_index[param])
                ranges.append((index, index, 1))
            work = math.prod(extents)
        map_entry.map.range = subsets.Range(ranges)
        map_entry.map.schedule = dtypes.ScheduleType.Sequential
        map_entry.map.collapse = 1
        map_entry.map.gpu_block_size = None

        scopes, exits = [map_entry], []
        if threads is not None:
            tb_entry, tb_exit, seq_entry, seq_exit = stockham.block_local_loop(
                comp_state, f"fft_block_map_{comp_state.label}", work, threads
            )
            scopes, exits = [tb_entry, seq_entry, map_entry], [seq_exit, tb_exit]
        reads = {}
        for edge in in_edges:
            if edge.data.is_empty():
                continue
            name = buffer_names.get(edge.data.data, edge.data.data)
            if name not in reads:
                reads[name] = comp_state.add_read(name)
            desc = comp_state.sdfg.arrays[name]
            comp_state.add_memlet_path(
                reads[name],
                *scopes,
                dst_conn=edge.dst_conn,
                memlet=Memlet.from_array(name, desc),
                propagate=False,
            )
        out_name = buffer_names.get(out_data, out_data)
        comp_state.add_memlet_path(
            map_exit,
            *exits,
            comp_state.add_write(out_name),
            src_conn=out_conn,
            memlet=Memlet.from_array(out_name, comp_state.sdfg.arrays[out_name]),
            propagate=False,
        )


def _nested_connections(edges, names: tuple[str, ...]) -> list[tuple]:
    """
    Returns the edges that connect one of the given arrays to a nested SDFG, as ``(edge, nested SDFG node,
    connector)`` tuples.
    """
    result = []
    for edge in edges:
        if edge.data.data not in names:
            continue
        for node, conn in ((edge.dst, edge.dst_conn), (edge.src, edge.src_conn)):
            if isinstance(node, nodes.NestedSDFG) and conn is not None:
                result.append((edge, node, conn))
    return result


def _restate_nested(
    nsdfg: nodes.NestedSDFG,
    name: str,
    outer_subset: subsets.Range,
    batch_dims: set[int],
    desc: data.Data | None,
) -> bool:
    """
    Checks whether the array of a nested SDFG that is connected to the spectrum or the output of a map can be restated
    as a block-local buffer and, if ``desc`` is given, restates it with that descriptor. Every access to the array
    (also in deeper nested SDFGs) must be at the same batch index as the memlet that connects it, given by a symbol
    that is mapped to that index, and becomes an access at index zero. Accesses in conditions are not supported.

    :param nsdfg: The nested SDFG node.
    :param name: The connector, i.e., the name of the array in the nested SDFG.
    :param outer_subset: The subset of the memlet that connects the array, in the symbols of the parent.
    :param batch_dims: The dimensions that are not transformed.
    :param desc: The descriptor of the buffer to restate the array as, or None to only check.
    :return: True if the array can be (and, with ``desc``, was) restated.
    """
    inner = nsdfg.sdfg
    if any(name in edge.data.free_symbols for edge in inner.all_interstate_edges()):
        return False
    for region in inner.all_control_flow_regions():
        if any(memlet.data == name for memlet in region.get_meta_read_memlets()):
            return False

    outer_index = {dim: str(outer_subset[dim][0]) for dim in batch_dims}
    memlets, nested = [], []
    for state in inner.states():
        for edge in state.edges():
            if edge.data.data != name:
                if not edge.data.is_empty() and any(
                    isinstance(node, nodes.AccessNode) and node.data == name for node in (edge.src, edge.dst)
                ):
                    return False  # A copy described from the side of the other array
                continue
            for dim in batch_dims:
                begin, end, _ = edge.data.subset[dim]
                if begin != end or str(nsdfg.symbol_mapping.get(str(begin), begin)) != outer_index[dim]:
                    return False
            memlets.append(edge.data)
            for node, conn in ((edge.dst, edge.dst_conn), (edge.src, edge.src_conn)):
                if isinstance(node, nodes.NestedSDFG) and conn is not None:
                    nested.append((node, conn, copy.deepcopy(edge.data.subset)))

    for node, conn, subset in nested:
        if not _restate_nested(node, conn, subset, batch_dims, desc):
            return False
    if desc is not None:
        for memlet in memlets:
            memlet.subset = subsets.Range(
                [(0, 0, 1) if dim in batch_dims else rng for dim, rng in enumerate(memlet.subset.ndrange())]
            )
        restated = copy.deepcopy(desc)
        restated.transient = False
        restated.lifetime = dtypes.AllocationLifetime.Scope
        inner.remove_data(name, validate=False)
        inner.add_datadesc(name, restated)
        # The batch index may no longer be used
        for symbol in set(nsdfg.symbol_mapping) - set(map(str, inner.used_symbols(False))):
            del nsdfg.symbol_mapping[symbol]
            if symbol in inner.symbols:
                inner.remove_symbol(symbol)
    return True


def _single_edge(edges, connector: str):
    """
    Returns the only edge to the given connector, or None.
    """
    matching = [edge for edge in edges if edge.dst_conn == connector]
    return matching[0] if len(matching) == 1 and len(edges) == 1 else None


def _is_full(sdfg: SDFG, memlet: Memlet) -> bool:
    """
    Returns True if a memlet accesses its whole array without write-conflict resolution.
    """
    if memlet.is_empty() or memlet.wcr is not None or memlet.other_subset is not None:
        return False
    return memlet.subset == subsets.Range.from_array(sdfg.arrays[memlet.data])


def _is_local_transient(sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> bool:
    """
    Returns True if an access node is the only use of a transient in the SDFG.
    """
    if not sdfg.arrays[node.data].transient:
        return False
    for other_state in sdfg.states():
        for other in other_state.data_nodes():
            if other.data == node.data and other is not node:
                return False
    return all(node.data not in edge.data.free_symbols for edge in sdfg.all_interstate_edges())


def _map_dimensions(
    sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, output: str, shape: tuple
) -> list[str] | None:
    """
    Returns the map parameter that indexes each dimension of the output of a map, if the map writes each element of
    the output exactly once, at the values of its parameters (e.g., ``out[i, j] = ...`` over ``i, j`` in the shape).
    """
    map_exit = state.exit_node(map_entry)
    params = None
    for edge in state.in_edges(map_exit):
        if edge.data.is_empty():
            continue
        if edge.data.data != output or edge.data.wcr is not None:
            return None
        if edge.data.subset.num_elements() != 1:
            return None
        indices = [str(begin) for begin, _, _ in edge.data.subset.ndrange()]
        if params is None:
            params = indices
        elif indices != params:
            return None
    if params is None or sorted(params) != sorted(map_entry.map.params):
        return None
    for dim, param in enumerate(params):
        begin, end, step = map_entry.map.range[map_entry.map.params.index(param)]
        if begin != 0 or step != 1 or symbolic.simplify(end + 1 - shape[dim]) != 0:
            return None
    return params


def _reads_one_transform(
    state: SDFGState, map_entry: nodes.MapEntry, spectrum: str, axes: list[int], params: list[str]
) -> bool:
    """
    Returns True if every read of the spectrum in a map is in the transform of the point the map computes, i.e., is
    at the map parameter in every dimension that is not transformed.
    """
    spectrum_connectors = {
        "OUT_" + edge.dst_conn[3:] for edge in state.in_edges(map_entry) if edge.data.data == spectrum
    }
    for edge in state.scope_subgraph(map_entry, include_entry=True, include_exit=False).edges():
        if edge.src is map_entry and edge.src_conn in spectrum_connectors and edge.data.data != spectrum:
            return False  # A copy described from the side of its destination
        if edge.data.data != spectrum:
            continue
        for dim, (begin, end, step) in enumerate(edge.data.subset.ndrange()):
            if dim not in axes and (str(begin) != params[dim] or str(end) != params[dim]):
                return False
    return True
