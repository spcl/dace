# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Real-to-complex (R2C) and complex-to-real (C2R) Stockham FFTs as native SDFGs, with the half spectrum of NumPy's
``rfftn`` and ``irfftn``.

The transform of a real axis of even length ``N = 2 M`` is a complex transform of length ``M``: the R2C transform
reads the real input ``x`` as the complex sequence ``z[m] = x[2 m] + i x[2 m + 1]``, computes its FFT ``Z``, and
separates the transforms of the even and odd elements, ``E[k] = (Z[k] + conj(Z[M - k])) / 2`` and ``O[k] = (Z[k] -
conj(Z[M - k])) / 2i``, into the ``M + 1`` outputs ``X[k] = E[k] + exp(-2 pi i k / N) O[k]``. The C2R transform
reverses these steps. Both read and write the real array directly as pairs of real numbers (see ``add_stage_map``) and
separate (or combine) the halves in the buffers of a CPU map iteration or of a GPU thread block (see
``add_real_axis_kernel``), so that a transform moves about half the data and computes about half the operations of a
complex transform. Other transformed axes are complex transforms of the half spectrum.

Transforms of odd length, with general-radix stages, or that do not fit into the buffers of an iteration are computed
by complex transforms instead.
"""

import math

from dace import SDFG, Memlet, SDFGState, data, dtypes, nodes, symbolic
from dace.libraries.fft.algorithms import stockham


def real_axis_lanes(
    shape: tuple[symbolic.SymbolicType, ...],
    axis: int,
    dtype: dtypes.typeclass,
    constants: dict,
    gpu: bool,
    strides: list[tuple[symbolic.SymbolicType, ...]],
) -> int | None:
    """
    Returns the number of transforms that a CPU map iteration of a real transform computes (see
    ``stockham.local_lanes``), or 1 on a GPU, if ``add_real_axis_kernel`` applies, or None otherwise.

    :param shape: The shape of the real array.
    :param axis: The real axis.
    :param dtype: The complex type of the transform.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :param gpu: Whether the transform is computed by a GPU thread block.
    :param strides: The strides of the real and the complex array. On a GPU, the axis must be contiguous in both.
    """
    length = stockham.fixed_length(shape[axis], constants)
    if length is None or length % 2 != 0 or length < 4:
        return None
    half = list(shape)
    half[axis] = length // 2
    if not stockham.can_expand(tuple(half), [axis], constants):
        return None
    if any(stockham.is_general_radix(radix) for _, radix, _, _ in stockham.plan_stages(tuple(half), [axis], constants)):
        return None
    if gpu:
        if axis != len(shape) - 1 or any(array_strides[axis] != 1 for array_strides in strides):
            return None
        return 1 if stockham.fits_block_local(tuple(half), [axis], dtype, constants) else None
    return stockham.local_lanes(tuple(half), [axis], dtype, constants)


def _real_step_code(inverse: bool, dtype: dtypes.typeclass, k: str, factor: symbolic.SymbolicType | None) -> str:
    """
    Returns the Python tasklet code that separates (R2C) or combines (C2R) the transforms of the even and odd elements
    of a real transform (see the module documentation) for index ``k``. The R2C code reads ``_zk = Z[k mod M]`` and
    ``_zm = Z[(M - k) mod M]`` and writes ``_y = X[k]``. The C2R code reads ``_xk = X[k]`` and ``_xm = X[M - k]`` and
    writes ``_z = Z[k]``, scaled so that the inverse complex transform of length ``M`` yields the real output, and
    ignores the imaginary parts of ``X[0]`` and ``X[M]`` like NumPy. ``_w`` is the table of the ``N`` roots of unity of
    the direction of the transform.
    """
    real = "dace.float32" if dtype == dtypes.complex64 else "dace.float64"
    scale = "" if factor is None else f" * {stockham.factor_code(factor, dtype)}"
    complex_type = f"dace.{dtype.to_string()}"
    if not inverse:
        lines = [
            f"__er = {real}(0.5) * (_zk.real + _zm.real)",
            f"__ei = {real}(0.5) * (_zk.imag - _zm.imag)",
            f"__or = {real}(0.5) * (_zk.imag + _zm.imag)",
            f"__oi = {real}(0.5) * (_zm.real - _zk.real)",
            f"__w = _w[{k}]",
            "__tr = __w.real * __or - __w.imag * __oi",
            "__ti = __w.real * __oi + __w.imag * __or",
            f"_y = {complex_type}((__er + __tr){scale}, (__ei + __ti){scale})",
        ]
        return "\n".join(lines)
    lines = [
        "__xki = _xk.imag",
        "__xmi = _xm.imag",
        f"if {k} == 0:",
        f"    __xki = {real}(0)",
        f"    __xmi = {real}(0)",
        "__ar = _xk.real + _xm.real",
        "__ai = __xki - __xmi",
        "__br = _xk.real - _xm.real",
        "__bi = __xki + __xmi",
        f"__w = _w[{k}]",
        "__cr = __w.real * __br - __w.imag * __bi",
        "__ci = __w.real * __bi + __w.imag * __br",
        f"_z = {complex_type}((__ar - __ci){scale}, (__ai + __cr){scale})",
    ]
    return "\n".join(lines)


def _add_real_step(
    state: SDFGState,
    inverse: bool,
    src: str,
    dst: str,
    shape: list[int],
    axis: int,
    batch_index: dict[str, list[str | None]],
    padding: int,
    roots: str,
    factor: symbolic.SymbolicType | None,
    threads: int | None,
) -> None:
    """
    Adds the step of a real transform (see ``add_real_axis_kernel``) that separates the transforms of the even and odd
    elements from buffer ``src`` into the half spectrum ``dst`` (R2C), or combines them from ``src`` into buffer
    ``dst`` (C2R), as a sequential map on the CPU or a loop of the threads of a GPU thread block.

    :param state: The (empty) state to add the step to.
    :param inverse: Whether the transform is C2R.
    :param src: The array to read from.
    :param dst: The array to write to.
    :param shape: The shape of the complex transforms of length ``M`` of an iteration.
    :param axis: The transformed axis.
    :param batch_index: The batch index of each array (see ``stockham.add_block_local_stage``).
    :param padding: The number of elements of a padded row of the buffers (see ``stockham.block_local_sdfg``), or 0.
    :param roots: The table of the ``N`` roots of unity of the direction of the transform.
    :param factor: A factor to multiply the outputs by, or None.
    :param threads: The number of threads of a GPU thread block, or None on the CPU.
    """
    sdfg = state.sdfg
    half = shape[axis]
    buffer = dst if inverse else src
    count = half if inverse else half + 1
    lane_dims = [dim for dim, extent in enumerate(shape) if extent > 1 and dim != axis]
    extents = [count] + [shape[dim] for dim in lane_dims]
    if threads is None:
        indices = ["__k"] + [f"__l{dim}" for dim in lane_dims]
    else:
        indices = stockham.delinearize("__u", extents)
    k = indices[0]
    lane_index = dict(zip(lane_dims, indices[1:]))

    def element(name: str, axis_index: str) -> Memlet:
        subset = []
        for dim in range(len(shape)):
            base = batch_index[name][dim]
            if dim == axis and padding and name == buffer:
                subset += [f"int_floor({axis_index}, {padding})", f"({axis_index}) % {padding}"]
            elif dim == axis:
                subset.append(axis_index)
            elif dim in lane_index:
                subset.append(lane_index[dim] if base is None else f"{base} + {lane_index[dim]}")
            else:
                subset.append("0" if base is None else base)
        return Memlet(f"{name}[{', '.join(subset)}]")

    if inverse:
        inputs = {"_xk": element(src, k), "_xm": element(src, f"{half} - {k}")}
        outputs = {"_z": element(dst, k)}
    else:
        inputs = {"_zk": element(src, f"({k}) % {half}"), "_zm": element(src, f"({half} - {k}) % {half}")}
        outputs = {"_y": element(dst, k)}
    code = _real_step_code(inverse, stockham.complex_type(sdfg.arrays[dst].dtype), k, factor)
    label = "stockham_real_combine" if inverse else "stockham_real_separate"
    if threads is None:
        state.add_mapped_tasklet(
            label,
            dict(zip(indices, (f"0:{extent}" for extent in extents))),
            {**inputs, "_w": Memlet(f"{roots}[0]")},
            code,
            outputs,
            external_edges=True,
            schedule=dtypes.ScheduleType.Sequential,
        )
        return
    work = 1
    for extent in extents:
        work *= extent
    stockham.add_block_local_tasklet(
        state, label, work, threads, {src: inputs, roots: {"_w": Memlet(f"{roots}[0]")}}, dst, outputs, code
    )


def add_real_axis_kernel(
    state: SDFGState,
    label: str,
    axis: int,
    inverse: bool,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    factor: symbolic.SymbolicType | None,
    lanes: int,
    gpu: bool,
    schedule: dtypes.ScheduleType,
    constants: dict,
) -> nodes.MapEntry:
    """
    Adds the R2C (or C2R) transform of one axis of even length ``N = 2 M`` from ``src_node`` to ``dst_node`` as one map
    over the other dimensions, in which each iteration computes a complex transform of length ``M`` in buffers on the
    stack (CPU, see ``stockham.add_local_axis_kernel``), or as a GPU kernel with one thread block per transform in
    shared memory (see ``stockham.block_local_fft_sdfg``). The stages read (C2R: write) the real array as pairs of real
    numbers, and a separate step separates (C2R: combines) the transforms of the even and odd elements in the buffers.

    :param state: The state to add the kernel to.
    :param label: The label of the kernel.
    :param axis: The real axis.
    :param inverse: Whether the transform is C2R.
    :param src_node: The access node of the real (C2R: complex half spectrum) array.
    :param dst_node: The access node of the complex half spectrum (C2R: real) array.
    :param factor: A factor to multiply the outputs by, or None.
    :param lanes: The number of transforms of an iteration (see ``real_axis_lanes``).
    :param gpu: Whether to compute the transforms in GPU thread blocks.
    :param schedule: The schedule of the map on the CPU.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The entry node of the map or kernel.
    """
    sdfg = state.sdfg
    src, dst = src_node.data, dst_node.data
    real, half_spectrum = (dst, src) if inverse else (src, dst)
    dtype = sdfg.arrays[half_spectrum].dtype
    shape = list(sdfg.arrays[real].shape)
    length = stockham.fixed_length(shape[axis], constants)
    half = list(shape)
    half[axis] = length // 2
    sign = 1 if inverse else -1

    stages = stockham.plan_stages(tuple(half), [axis], constants)
    prefix = f"__{label}_tw"
    table_nodes = stockham.add_twiddle_tables(state, stages, sign, dtype, gpu, prefix)
    device = "true" if gpu else "false"
    roots = stockham.add_table(state, f"{prefix}_roots", dtype, f"Roots<{dtype.ctype}>({length}, {sign}, {device})")
    tables = {key: node.data for key, node in table_nodes.items()}
    arrays = {name: sdfg.arrays[name] for name in (src, dst, *tables.values(), roots.data)}
    padding = 0 if gpu else stockham.local_padding(stages)
    steps = [(stage, sign, tables, None) for stage in stages]
    steps = [None] + steps if inverse else steps + [None]
    storage = dtypes.StorageType.GPU_Shared if gpu else dtypes.StorageType.Register(force=True)
    block, batch_symbols, batch_index, buffers = stockham.block_local_sdfg(
        label,
        arrays,
        tuple(half),
        [axis],
        dtype,
        constants,
        min(len(steps) - 1, 2),
        storage,
        lanes,
        padding,
        rows=not gpu,
    )
    block_shape = stockham.block_local_shape(tuple(half), [axis], constants, lanes)
    threads = stockham.block_local_threads(stages, block_shape) if gpu else None
    block_states = stockham.add_block_local_steps(
        block, steps, src, dst, buffers, block_shape, batch_index, threads, padding, pairs={real}
    )
    step_state, step_src, step_dst = block_states[0] if inverse else block_states[-1]
    _add_real_step(
        step_state, inverse, step_src, step_dst, block_shape, axis, batch_index, padding, roots.data, factor, threads
    )
    if not gpu:
        stockham.add_row_views(block, [src, dst], batch_symbols, block_shape)
    inputs = {src: src_node, roots.data: roots, **{node.data: node for node in table_nodes.values()}}
    kernel_schedule = dtypes.ScheduleType.GPU_Device if gpu else schedule
    entry, _ = stockham.add_block_local_kernel(
        state, block, inputs, dst_node, tuple(half), batch_symbols, kernel_schedule, lanes, constants
    )
    return entry


def _lifetime(shape: tuple[symbolic.SymbolicType, ...], persistent: bool, constants: dict) -> dtypes.AllocationLifetime:
    """
    Returns the lifetime of an intermediate array: allocated once per program if allowed and its size is fixed, as
    per-call allocations of large arrays dominate the runtime (see ``stockham.add_stockham``).
    """
    if persistent and not any(symbolic.issymbolic(size, constants) for size in shape):
        return dtypes.AllocationLifetime.Persistent
    return dtypes.AllocationLifetime.Scope


def _add_real_axis_fallback(
    state: SDFGState,
    axis: int,
    inverse: bool,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    factor: symbolic.SymbolicType,
    constants: dict,
    schedule: dtypes.ScheduleType,
    persistent: bool,
    storage: dtypes.StorageType,
) -> None:
    """
    Adds the R2C (or C2R) transform of one axis as a complex transform of the whole axis: of the real input followed
    by a copy of the half spectrum (R2C), or of the Hermitian extension of the half spectrum followed by a copy of the
    real part (C2R).
    """
    sdfg = state.sdfg
    src, dst = src_node.data, dst_node.data
    real, half_spectrum = (dst, src) if inverse else (src, dst)
    dtype = sdfg.arrays[half_spectrum].dtype
    shape = sdfg.arrays[real].shape
    length = shape[axis]
    half_length = sdfg.arrays[half_spectrum].shape[axis]
    lifetime = _lifetime(shape, persistent, constants)
    full, _ = sdfg.add_transient("__stockham_real_full", shape, dtype, storage, lifetime=lifetime, find_new_name=True)
    map_schedule = schedule
    ranges = {f"__i{dim}": f"0:{symbolic.symstr(extent)}" for dim, extent in enumerate(shape)}
    index = [f"__i{dim}" for dim in range(len(shape))]

    def subset(name: str, axis_index: str) -> str:
        return f"{name}[{', '.join(axis_index if dim == axis else index[dim] for dim in range(len(shape)))}]"

    if not inverse:
        full_node = state.add_access(full)
        stockham.add_stockham(
            state, src_node, full_node, [axis], False, factor, constants, schedule, persistent, "rfft_full"
        )
        state.add_mapped_tasklet(
            "stockham_real_half",
            {**ranges, f"__i{axis}": f"0:{symbolic.symstr(half_length)}"},
            {"_i": Memlet(subset(full, index[axis]))},
            "_o = _i",
            {"_o": Memlet(subset(dst, index[axis]))},
            external_edges=True,
            input_nodes={full: full_node},
            output_nodes={dst: dst_node},
            schedule=map_schedule,
        )
        return

    # The Hermitian extension: X[k] for k <= N / 2, conj(X[N - k]) above
    spectrum, _ = sdfg.add_transient(
        "__stockham_real_spectrum", shape, dtype, storage, lifetime=lifetime, find_new_name=True
    )
    spectrum_node = state.add_access(spectrum)
    state.add_mapped_tasklet(
        "stockham_real_lower",
        {**ranges, f"__i{axis}": f"0:{symbolic.symstr(half_length)}"},
        {"_i": Memlet(subset(src, index[axis]))},
        "_o = _i",
        {"_o": Memlet(subset(spectrum, index[axis]))},
        external_edges=True,
        input_nodes={src: src_node},
        output_nodes={spectrum: spectrum_node},
        schedule=map_schedule,
    )
    state.add_mapped_tasklet(
        "stockham_real_upper",
        {**ranges, f"__i{axis}": f"{symbolic.symstr(half_length)}:{symbolic.symstr(length)}"},
        {"_i": Memlet(subset(src, f"{symbolic.symstr(length)} - {index[axis]}"))},
        f"_o = dace.{dtype.to_string()}(_i.real, -_i.imag)",
        {"_o": Memlet(subset(spectrum, index[axis]))},
        external_edges=True,
        input_nodes={src: src_node},
        output_nodes={spectrum: spectrum_node},
        schedule=map_schedule,
    )
    full_node = state.add_access(full)
    stockham.add_stockham(
        state, spectrum_node, full_node, [axis], True, factor, constants, schedule, persistent, "irfft_full"
    )
    state.add_mapped_tasklet(
        "stockham_real_part",
        ranges,
        {"_i": Memlet(subset(full, index[axis]))},
        "_o = _i.real",
        {"_o": Memlet(subset(dst, index[axis]))},
        external_edges=True,
        input_nodes={full: full_node},
        output_nodes={dst: dst_node},
        schedule=map_schedule,
    )


def real_stockham_sdfg(
    indesc: data.Array,
    outdesc: data.Array,
    axes: list[int],
    inverse: bool,
    factor: symbolic.SymbolicType,
    constants: dict,
    schedule: dtypes.ScheduleType = dtypes.ScheduleType.Default,
    persistent: bool = False,
    name: str = "stockham_real_fft",
) -> SDFG:
    """
    Creates an SDFG that computes the R2C transform of the real ``_inp`` into the half spectrum ``_out`` (like
    ``numpy.fft.rfftn``), or the C2R transform of the half spectrum ``_inp`` into the real ``_out`` (like
    ``numpy.fft.irfftn``), over the given axes. The last axis is the real one: its length in the half spectrum is
    ``N // 2 + 1``, where ``N`` is its length in the real array.

    :param indesc: Descriptor of the input array.
    :param outdesc: Descriptor of the output array.
    :param axes: The (normalized) axes to transform. Their lengths must be fixed.
    :param inverse: Whether to compute the C2R transform.
    :param factor: Factor to multiply the outputs by (e.g., for normalization).
    :param constants: Constants of the parent SDFG, used to evaluate symbolic axis lengths.
    :param schedule: Schedule of the library node (see ``stockham.stockham_sdfg``).
    :param persistent: Whether buffers may be allocated once per program (see ``stockham.stockham_sdfg``).
    :param name: Name of the generated SDFG.
    :return: A single-state SDFG.
    """
    real_desc, half_desc = (outdesc, indesc) if inverse else (indesc, outdesc)
    axis, others = axes[-1], list(axes[:-1])
    if half_desc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"The half spectrum of a real FFT must be complex, got {half_desc.dtype}")
    if real_desc.dtype != {dtypes.complex64: dtypes.float32, dtypes.complex128: dtypes.float64}[half_desc.dtype]:
        raise TypeError(f"Real FFT of {real_desc.dtype} with a {half_desc.dtype} half spectrum")
    length = stockham.fixed_length(real_desc.shape[axis], constants)
    if length is None or stockham.fixed_length(half_desc.shape[axis], constants) != length // 2 + 1:
        raise ValueError(
            f"The half spectrum of a real FFT of length {real_desc.shape[axis]} must have N // 2 + 1 elements"
        )
    if not stockham.can_expand(real_desc.shape, axes, constants):
        raise ValueError(f"Stockham FFT requires fixed axis lengths, got shape {real_desc.shape} and axes {axes}")

    sdfg = SDFG(name)
    sdfg.add_array("_inp", indesc.shape, indesc.dtype, indesc.storage, strides=indesc.strides)
    sdfg.add_array("_out", outdesc.shape, outdesc.dtype, outdesc.storage, strides=outdesc.strides)
    state = sdfg.add_state("stockham_real")
    gpu = schedule == dtypes.ScheduleType.GPU_Device
    storage = dtypes.StorageType.Default
    if gpu:
        storage = dtypes.StorageType.GPU_Global
    elif persistent:
        storage = dtypes.StorageType.CPU_Heap

    # The complex transforms of the other axes, on the half spectrum
    src_node, dst_node = state.add_read("_inp"), state.add_write("_out")
    if others:
        lifetime = _lifetime(half_desc.shape, persistent, constants)
        tmp, _ = sdfg.add_transient(
            "__stockham_real_half", half_desc.shape, half_desc.dtype, storage, lifetime=lifetime
        )
        tmp_node = state.add_access(tmp)
        if inverse:
            stockham.add_stockham(
                state, src_node, tmp_node, others, True, 1, constants, schedule, persistent, "irfft_c2c"
            )
            src_node = tmp_node
        else:
            stockham.add_stockham(
                state, tmp_node, dst_node, others, False, 1, constants, schedule, persistent, "rfft_c2c"
            )
            dst_node = tmp_node

    strides = [real_desc.strides, half_desc.strides]
    lanes = real_axis_lanes(real_desc.shape, axis, half_desc.dtype, constants, gpu, strides)
    if lanes is None:
        _add_real_axis_fallback(
            state, axis, inverse, src_node, dst_node, factor, constants, schedule, persistent, storage
        )
        return sdfg

    map_schedule = dtypes.ScheduleType.Default
    total_size = stockham.fixed_length(math.prod(real_desc.shape), constants)
    if total_size is not None and total_size < stockham.SEQUENTIAL_SIZE_THRESHOLD:
        map_schedule = dtypes.ScheduleType.Sequential
    out_factor = None if factor == 1 else factor
    add_real_axis_kernel(
        state, "stockham_real_axis", axis, inverse, src_node, dst_node, out_factor, lanes, gpu, map_schedule, constants
    )
    return sdfg
