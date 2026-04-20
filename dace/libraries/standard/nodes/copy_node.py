# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Copy library nodes for data movement at various levels of the memory hierarchy.

All copy operations use a single node type, ``CopyLibraryNode``, with
different expansions selected based on storage types:

``pure``
    Same-side mapped tasklet (default fallback).

``CUDA`` / ``CPU``
    API-level memcpy for same-storage contiguous copies.

``CUDAHostToDevice`` / ``CUDADeviceToHost``
    Cross-storage cudaMemcpyAsync calls.

``RegisterCopy``
    In-kernel register-to-register copy (Sequential mapped tasklet).

``SharedMemoryCopy``
    In-kernel copy involving GPU shared memory (GPU_ThreadBlock mapped
    tasklet for cooperative loads).
"""
import copy as _copy

import dace
from dace import library, nodes, properties, dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from functools import reduce
import operator
from dace.codegen.common import sym2cpp

from dace.libraries.standard.helper import add_dynamic_inputs, collapse_shape_and_strides
from dace.sdfg.construction_utils import get_parent_map_and_loop_scopes

# Name of the special in-connector used to pass an explicit GPU stream.
_STREAM_CONN = "stream"

# ===================================================================
# Shared validation helper
# ===================================================================


def _validate_copy_edges(node, sdfg, state):
    """
    Common edge validation for all copy library nodes.

    Checks that there is exactly one ``_in`` data input edge, exactly one
    output edge, collects dynamic scalar inputs, and returns the data
    descriptors and subsets.  Also extracts an optional ``stream``
    in-connector used to pass an explicit GPU stream to the expansion.

    :return: ``(inp_name, inp, in_subset, out_name, out, out_subset,
        dynamic_inputs, stream_input)`` where ``stream_input`` is either
        ``None`` or the data descriptor of the source connected to the
        ``stream`` connector.
    """
    if len(state.out_edges(node)) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one output edge.")

    oe = next(iter(state.out_edges(node)))
    out = sdfg.arrays[oe.data.data]
    out_subset = oe.data.subset
    out_name = oe.src_conn

    # Extract the optional `stream` input connector separately.  Streams are
    # opaque handles (gpuStream_t), not scalars used in map ranges, so they
    # bypass the scalar-only check below.
    stream_ies = [ie for ie in state.in_edges(node) if ie.dst_conn == _STREAM_CONN]
    if len(stream_ies) > 1:
        raise ValueError(f"{type(node).__name__} expects at most one '{_STREAM_CONN}' input edge.")
    stream_input = None
    if stream_ies:
        stream_input = sdfg.arrays[stream_ies[0].data.data]

    # Collect dynamic input connectors (anything not "_in" or "stream")
    dynamic_ies = {ie for ie in state.in_edges(node) if ie.dst_conn not in ("_in", _STREAM_CONN)}
    dynamic_inputs = dict()
    for ie in dynamic_ies:
        dataname = ie.data.data
        datadesc = sdfg.arrays[dataname]
        if not isinstance(datadesc, dace.data.Scalar):
            raise ValueError("Dynamic inputs (not connected to `_in`) must be scalars.")
        dynamic_inputs[ie.dst_conn] = datadesc

    data_ies = {ie for ie in state.in_edges(node) if ie.dst_conn == "_in"}
    if len(data_ies) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one data input edge "
                         "connected to the `_in` connector.")
    ie = data_ies.pop()
    inp = sdfg.arrays[ie.data.data]
    in_subset = ie.data.subset
    inp_name = ie.dst_conn

    if not inp:
        raise ValueError("Missing input data descriptor.")
    if not out:
        raise ValueError("Missing output data descriptor.")
    if inp.dtype != out.dtype:
        raise ValueError("Input and output data types must match "
                         f"(got {inp.dtype} vs {out.dtype}).")

    return inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input


# Storage type sets for cross-boundary detection.
# A mapped tasklet cannot read from one side and write to the other.
_CPU_STORAGES = {
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
}
_GPU_STORAGES = {
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
    dtypes.StorageType.Register,  # registers live on the GPU die when inside a kernel
}


def _is_cross_cpu_gpu(src_storage, dst_storage):
    """
    Returns True if src and dst are on opposite sides of the CPU/GPU
    boundary (one is CPU-side, the other GPU-side).  A mapped tasklet
    cannot handle this; it requires a memcpy API call.
    """
    src_cpu = src_storage in _CPU_STORAGES
    src_gpu = src_storage in _GPU_STORAGES
    dst_cpu = dst_storage in _CPU_STORAGES
    dst_gpu = dst_storage in _GPU_STORAGES
    return (src_cpu and dst_gpu) or (src_gpu and dst_cpu)


# ===================================================================
# Shared expansion helpers
# ===================================================================


def _add_stream_descriptor(sdfg, stream_input):
    """Register an opaque ``stream`` input descriptor on the expansion SDFG.

    Mirrors the parent-side descriptor shape/dtype so connector names on the
    library node map directly to the nested SDFG's top-level arrays.
    """
    if stream_input is None:
        return
    desc = _copy.deepcopy(stream_input)
    desc.transient = False
    sdfg.add_datadesc(_STREAM_CONN, desc)


def _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=False, require_contiguous=False):
    """
    Shared validation and SDFG skeleton used by all CopyLibraryNode expansions.

    :param require_contiguous: If True, checks that both subsets are
        contiguous in their respective arrays.  Expansions that use flat
        memcpy (CPU, CUDA, H2D, D2H) must set this to True.

    Returns a tuple of:
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset,
         map_lengths, in_shape_collapsed, in_strides_collapsed,
         out_shape_collapsed, out_strides_collapsed, stream_input)
    """
    inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input = node.validate(
        parent_sdfg, parent_state, allow_cross_storage=allow_cross_storage)

    if require_contiguous:
        if not in_subset.is_contiguous_subset(inp):
            raise ValueError(f"This expansion requires a contiguous input subset, but "
                             f"'{inp_name}' subset {in_subset} is not contiguous in "
                             f"array shape {inp.shape} with strides {inp.strides}.  "
                             f"Use a strided expansion (pure, CopyND, Assignment) instead, "
                             f"or decompose into contiguous copies in Layer 2.")
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(f"This expansion requires a contiguous output subset, but "
                             f"'{out_name}' subset {out_subset} is not contiguous in "
                             f"array shape {out.shape} with strides {out.strides}.  "
                             f"Use a strided expansion (pure, CopyND, Assignment) instead, "
                             f"or decompose into contiguous copies in Layer 2.")

    map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    _add_stream_descriptor(sdfg, stream_input)

    state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
    map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

    return (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, in_shape_collapsed,
            in_strides_collapsed, out_shape_collapsed, out_strides_collapsed, stream_input)


def _make_mapped_tasklet_expansion(node, parent_state, parent_sdfg, allow_cross_storage=False):
    """
    Creates a pure mapped-tasklet expansion.

    Correct for same-storage copies and for GPU-side cross-storage copies
    (e.g. GPU_Global <-> GPU_Shared).  Raises ValueError for copies that
    cross the CPU/GPU boundary -- those require a memcpy API call.
    """
    (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, _, _, _, _,
     stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=allow_cross_storage)

    # A single map cannot access both CPU and GPU memory.
    if _is_cross_cpu_gpu(inp.storage, out.storage):
        raise ValueError("Pure (mapped tasklet) expansion cannot handle copies across the "
                         f"CPU/GPU boundary (got {inp.storage} -> {out.storage}).  "
                         "Use CUDAHostToDevice or CUDADeviceToHost instead.")

    sdfg.schedule = dace.dtypes.ScheduleType.Default

    map_params = [f"__i{i}" for i in range(len(map_lengths))]
    map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
    in_access_expr = ','.join(map_params)
    out_access_expr = ','.join(map_params)
    inputs = {"_memcpy_inp": dace.memlet.Memlet(f"{inp_name}[{in_access_expr}]")}
    outputs = {"_memcpy_out": dace.memlet.Memlet(f"{out_name}[{out_access_expr}]")}
    code = "_memcpy_out = _memcpy_inp"

    # Pick schedule based on storage
    if (inp.storage == dace.dtypes.StorageType.GPU_Global or out.storage == dace.dtypes.StorageType.GPU_Global
            or inp.storage == dace.dtypes.StorageType.GPU_Shared or out.storage == dace.dtypes.StorageType.GPU_Shared):
        schedule = dace.dtypes.ScheduleType.GPU_Device
    else:
        schedule = dace.dtypes.ScheduleType.Default

    _, map_entry, _ = state.add_mapped_tasklet(f"{node.label}_tasklet",
                                               map_rng,
                                               inputs,
                                               code,
                                               outputs,
                                               schedule=schedule,
                                               external_edges=True)

    # Thread the explicit stream through as a dynamic input on the map when
    # this is a GPU_Device map.  The connector carries the caller-provided
    # ``gpuStream_t`` so downstream passes can bind the kernel launch to it.
    if stream_input is not None and schedule == dace.dtypes.ScheduleType.GPU_Device:
        stream_access = state.add_access(_STREAM_CONN)
        map_entry.add_in_connector(_STREAM_CONN)
        state.add_edge(stream_access, None, map_entry, _STREAM_CONN,
                       dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))

    return sdfg


def _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, direction):
    """
    Creates a CUDA expansion using cudaMemcpyAsync with the given direction
    string (e.g. "DeviceToDevice", "HostToDevice", "DeviceToHost").

    If the library node has a ``stream`` in-connector, the caller-provided
    ``gpuStream_t`` is used as the stream argument to ``cudaMemcpyAsync``.
    Otherwise, the expansion falls back to ``__dace_current_stream``.
    """
    allow_cross = direction != "DeviceToDevice"
    (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, _, _, _, _,
     stream_input) = _make_expansion_sdfg(node,
                                          parent_state,
                                          parent_sdfg,
                                          allow_cross_storage=allow_cross,
                                          require_contiguous=True)

    cp_size = reduce(operator.mul, map_lengths, 1)

    in_access = state.add_access(inp_name)
    out_access = state.add_access(out_name)

    tasklet_inputs = {"_memcpy_in"}
    stream_conn = _STREAM_CONN
    if stream_input is not None:
        tasklet_inputs.add(stream_conn)
        stream_expr = stream_conn
    else:
        stream_expr = "__dace_current_stream"

    tasklet = state.add_tasklet(name="memcpy_tasklet",
                                inputs=tasklet_inputs,
                                outputs={"_memcpy_out"},
                                code=(f"cudaMemcpyAsync(_memcpy_out, _memcpy_in, "
                                      f"{sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}), "
                                      f"cudaMemcpy{direction}, {stream_expr});"),
                                language=dace.Language.CPP)
    tasklet.schedule = dace.dtypes.ScheduleType.GPU_Device

    state.add_edge(in_access, None, tasklet, "_memcpy_in",
                   dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
    state.add_edge(tasklet, "_memcpy_out", out_access, None,
                   dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

    if stream_input is not None:
        stream_access = state.add_access(_STREAM_CONN)
        state.add_edge(stream_access, None, tasklet, stream_conn,
                       dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))

    return sdfg


def _make_thread_level_copy(node, parent_state, parent_sdfg):
    """
    Thread-level copy: Sequential mapped tasklet.
    Used for Register<->Register and Shared<->Register copies.
    """
    inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(parent_sdfg,
                                                                                           parent_state,
                                                                                           allow_cross_storage=True)

    map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

    state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
    map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

    sdfg.schedule = dace.dtypes.ScheduleType.Default

    map_params = [f"__i{i}" for i in range(len(map_lengths))]
    map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
    in_access_expr = ','.join(map_params)
    out_access_expr = ','.join(map_params)
    inputs = {"_cpy_inp": dace.memlet.Memlet(f"{inp_name}[{in_access_expr}]")}
    outputs = {"_cpy_out": dace.memlet.Memlet(f"{out_name}[{out_access_expr}]")}

    state.add_mapped_tasklet(f"{node.label}_tasklet",
                             map_rng,
                             inputs,
                             "_cpy_out = _cpy_inp",
                             outputs,
                             schedule=dace.dtypes.ScheduleType.Sequential,
                             external_edges=True)

    return sdfg


def _build_copynd_call(ctype, copy_shape, src_strides, dst_strides):
    """
    Builds a ``dace::CopyND`` or ``dace::CopyNDDynamic`` call string,
    using the most specific (static) variant possible.

    Selection logic (matches ``cpu.py`` codegen):

    1. If all copy dimensions are concrete integers:
       ``dace::CopyND<T, 1, false, dim0, dim1, ...>``
    2. Otherwise (symbolic dims):
       ``dace::CopyNDDynamic<T, 1, false, ndims>``
    3. If all **dst** strides are concrete: ``::template ConstDst<s0, s1>``
    4. Else if all **src** strides are concrete: ``::template ConstSrc<s0, s1>``
    5. Else: ``::Dynamic``
    6. Remaining (non-template) values passed as runtime args.
    """
    from dace import symbolic

    ndims = len(copy_shape)
    shape_strs = [sym2cpp(s) for s in copy_shape]
    src_stride_strs = [sym2cpp(s) for s in src_strides]
    dst_stride_strs = [sym2cpp(s) for s in dst_strides]

    # --- Determine which parts are compile-time constant ---
    dims_static = not any(symbolic.issymbolic(s) for s in copy_shape)
    src_static = not any(symbolic.issymbolic(s) for s in src_strides)
    dst_static = not any(symbolic.issymbolic(s) for s in dst_strides)

    # --- Build the template ---
    if dims_static:
        copy_tmpl = f"dace::CopyND<{ctype}, 1, false, {', '.join(shape_strs)}>"
    else:
        copy_tmpl = f"dace::CopyNDDynamic<{ctype}, 1, false, {ndims}>"

    # --- Build the stride specialization ---
    if dst_static:
        shape_tmpl = f"template ConstDst<{', '.join(dst_stride_strs)}>"
    elif src_static:
        shape_tmpl = f"template ConstSrc<{', '.join(src_stride_strs)}>"
    else:
        shape_tmpl = "Dynamic"

    # --- Build the runtime arguments ---
    # Per dimension, pass only the values NOT baked into the template.
    # CopyND runtime API order per dim:
    #   CopyND:        [src_stride | dst_stride | src_stride,dst_stride]
    #   CopyNDDynamic: [copydim,] + same as above
    dynshape = not dims_static

    # ConstSrc: src strides are template args, dst strides are runtime
    # ConstDst: dst strides are template args, src strides are runtime
    # Dynamic:  both are runtime
    if dst_static:  # ConstDst chosen
        dynsrc, dyndst = True, False
    elif src_static:  # ConstSrc chosen
        dynsrc, dyndst = False, True
    else:  # Dynamic
        dynsrc, dyndst = True, True

    stride_args = []
    for d in range(ndims):
        if dynshape:
            stride_args.append(shape_strs[d])
        if dynsrc:
            stride_args.append(src_stride_strs[d])
        if dyndst:
            stride_args.append(dst_stride_strs[d])

    all_args = ["_cpy_in", "_cpy_out"] + stride_args
    return f"{copy_tmpl}::{shape_tmpl}::Copy({', '.join(all_args)});"


def _generate_assignment_code(ctype, copy_shape, in_strides, out_strides):
    """
    Generates a C++ unrolled assignment loop for a direct copy.
    For 1D contiguous copies this is a simple ``#pragma unroll`` loop.
    For ND strided copies it delinearizes, respecting strides.
    """
    ndims = len(copy_shape)
    shape_strs = [sym2cpp(s) for s in copy_shape]
    in_stride_strs = [sym2cpp(s) for s in in_strides]
    out_stride_strs = [sym2cpp(s) for s in out_strides]

    total = " * ".join(f"({s})" for s in shape_strs)
    lines = []

    if ndims == 1:
        lines.append(f"#pragma unroll")
        lines.append(f"for (int __i = 0; __i < {shape_strs[0]}; ++__i) {{")
        lines.append(f"  (({ctype}*)_da_out)[__i * {out_stride_strs[0]}] ="
                     f" (({ctype}*)_da_in)[__i * {in_stride_strs[0]}];")
        lines.append("}")
    else:
        lines.append(f"#pragma unroll")
        lines.append(f"for (int __linear = 0; __linear < {total}; ++__linear) {{")
        lines.append("  int __rem = __linear;")

        idx_names = []
        for d in range(ndims):
            idx = f"__idx{d}"
            idx_names.append(idx)
            if d < ndims - 1:
                tail = " * ".join(f"({shape_strs[j]})" for j in range(d + 1, ndims))
                lines.append(f"  const int {idx} = __rem / ({tail});")
                lines.append(f"  __rem = __rem % ({tail});")
            else:
                lines.append(f"  const int {idx} = __rem;")

        src_expr = " + ".join(f"{idx_names[d]} * {in_stride_strs[d]}" for d in range(ndims))
        dst_expr = " + ".join(f"{idx_names[d]} * {out_stride_strs[d]}" for d in range(ndims))
        lines.append(f"  (({ctype}*)_da_out)[{dst_expr}] ="
                     f" (({ctype}*)_da_in)[{src_expr}];")
        lines.append("}")

    return "\n".join(lines)


# ===================================================================
# CopyLibraryNode expansions
# ===================================================================


@library.expansion
class ExpandPure(ExpandTransformation):
    """
    Default expansion: generates a mapped tasklet that copies element by
    element.  Correct for same-storage copies and GPU-side cross-storage
    copies (e.g. GPU_Global <-> GPU_Shared).  Raises for copies that cross
    the CPU/GPU boundary -- those require ``CUDAHostToDevice`` or
    ``CUDADeviceToHost``.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_mapped_tasklet_expansion(node, parent_state, parent_sdfg, allow_cross_storage=True)


@library.expansion
class ExpandCopyND(ExpandTransformation):
    """
    Expansion that lowers to a ``dace::CopyNDDynamic`` call (C++ tasklet).
    Works for any same-side copy (CPU or GPU).  This is the DaCe runtime
    fallback that handles strided ND copies without generating maps.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, in_shape_collapsed,
         in_strides_collapsed, out_shape_collapsed, out_strides_collapsed,
         _stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=True)

        if _is_cross_cpu_gpu(inp.storage, out.storage):
            raise ValueError("CopyND expansion cannot handle copies across the CPU/GPU "
                             f"boundary (got {inp.storage} -> {out.storage}).")

        code = _build_copynd_call(inp.dtype.ctype, map_lengths, in_strides_collapsed, out_strides_collapsed)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        tasklet = state.add_tasklet(name="copynd_tasklet",
                                    inputs={"_cpy_in"},
                                    outputs={"_cpy_out"},
                                    code=code,
                                    language=dace.Language.CPP)

        state.add_edge(
            in_access, None, tasklet, "_cpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_cpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """
    Expansion for GPU_Global -> GPU_Global contiguous copies via
    ``cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)``.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "DeviceToDevice")


@library.expansion
class ExpandCPU(ExpandTransformation):
    """
    Expansion for CPU_Heap -> CPU_Heap contiguous copies via ``std::memcpy``.
    """
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, _, _, _, _,
         _stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, require_contiguous=True)

        cp_size = reduce(operator.mul, map_lengths, 1)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        tasklet = state.add_tasklet(
            name="memcpy_tasklet",
            inputs={"_memcpy_in"},
            outputs={"_memcpy_out"},
            code=f"memcpy(_memcpy_out, _memcpy_in, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}));",
            language=dace.Language.CPP)

        state.add_edge(
            in_access, None, tasklet, "_memcpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_memcpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


@library.expansion
class ExpandCUDAHostToDevice(ExpandTransformation):
    """
    Expansion for CPU_Heap/CPU_Pinned -> GPU_Global contiguous copies via
    ``cudaMemcpyAsync(..., cudaMemcpyHostToDevice)``.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "HostToDevice")


@library.expansion
class ExpandCUDADeviceToHost(ExpandTransformation):
    """
    Expansion for GPU_Global -> CPU_Heap/CPU_Pinned contiguous copies via
    ``cudaMemcpyAsync(..., cudaMemcpyDeviceToHost)``.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "DeviceToHost")


# ===================================================================
# In-kernel expansions (registered on CopyLibraryNode)
# ===================================================================


@library.expansion
class ExpandDirectAssignment(ExpandTransformation):
    """
    Generates a bare ``_out = _in`` assignment tasklet (no map).
    The DaCe codegen resolves the actual data movement from the memlets:
    for scalars it emits a direct assignment, for arrays it emits CopyND.

    Suitable for:

    - GPU_Global -> Register  (thread loads)
    - Register -> Register
    - Same-storage copies **except** GPU_Shared

    This is the lightest expansion -- a single tasklet with no map
    overhead.  Prefer this for thread-private copies inside GPU kernels
    where the copy dimensions are known at compile time.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(parent_sdfg,
                                                                                               parent_state,
                                                                                               allow_cross_storage=True)

        if (inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared):
            raise ValueError("ExpandDirectAssignment cannot handle GPU_Shared storage.  "
                             "Use SharedMemoryCopy instead.")

        if _is_cross_cpu_gpu(inp.storage, out.storage):
            raise ValueError("ExpandDirectAssignment cannot handle copies across the "
                             f"CPU/GPU boundary (got {inp.storage} -> {out.storage}).")

        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state", is_start_block=True)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        tasklet = state.add_tasklet(name="assign",
                                    inputs={"_da_in"},
                                    outputs={"_da_out"},
                                    code=_generate_assignment_code(inp.dtype.ctype, map_lengths, in_strides_collapsed,
                                                                   out_strides_collapsed),
                                    language=dace.Language.CPP)

        in_range = dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])
        out_range = dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])

        state.add_edge(in_access, None, tasklet, "_da_in", dace.memlet.Memlet(data=inp_name, subset=in_range))
        state.add_edge(tasklet, "_da_out", out_access, None, dace.memlet.Memlet(data=out_name, subset=out_range))

        return sdfg


# Keep RegisterCopy as a strict alias: validates Register on both sides,
# then delegates to the same thread-level copy.
@library.expansion
class ExpandRegisterCopy(ExpandTransformation):
    """
    Strict variant of Assignment that validates both sides are Register.
    Equivalent to ExpandAssignment with an additional storage check.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(
            parent_sdfg, parent_state, allow_cross_storage=False)

        if inp.storage != dtypes.StorageType.Register:
            raise ValueError(f"ExpandRegisterCopy expects Register storage for input, "
                             f"got {inp.storage}.")
        if out.storage != dtypes.StorageType.Register:
            raise ValueError(f"ExpandRegisterCopy expects Register storage for output, "
                             f"got {out.storage}.")

        return _make_thread_level_copy(node, parent_state, parent_sdfg)


@library.expansion
class ExpandSharedMemoryCopy(ExpandTransformation):
    """
    Expansion for copies involving GPU shared memory inside a GPU kernel.

    - **Global <-> Shared** or **Shared <-> Shared**: generates a C++
      tasklet that calls ``dace::CopyNDDynamic`` followed by
      ``__syncthreads()`` to ensure all threads see the updated shared
      memory.  This expansion should **not** sit inside a parent
      ``GPU_ThreadBlock`` map -- it *is* the thread-block level operation.

    - **Shared <-> Register**: thread-level copy (Sequential mapped tasklet,
      same as RegisterCopy).  No synchronization needed.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(parent_sdfg,
                                                                                               parent_state,
                                                                                               allow_cross_storage=True)

        valid_storages = {dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global, dtypes.StorageType.Register}

        if inp.storage not in valid_storages:
            raise ValueError(f"ExpandSharedMemoryCopy expects GPU_Shared, GPU_Global, or "
                             f"Register storage for input, got {inp.storage}.")
        if out.storage not in valid_storages:
            raise ValueError(f"ExpandSharedMemoryCopy expects GPU_Shared, GPU_Global, or "
                             f"Register storage for output, got {out.storage}.")
        if (inp.storage != dtypes.StorageType.GPU_Shared and out.storage != dtypes.StorageType.GPU_Shared):
            raise ValueError("ExpandSharedMemoryCopy requires at least one side to be GPU_Shared.")

        # Shared <-> Register: thread-level (no cooperative sync needed)
        involves_register = (inp.storage == dtypes.StorageType.Register or out.storage == dtypes.StorageType.Register)
        if involves_register:
            return _make_thread_level_copy(node, parent_state, parent_sdfg)

        # Global <-> Shared or Shared <-> Shared: block-collective copy.
        # This expansion IS the thread-block-level operation, so it must
        # NOT sit inside a parent GPU_ThreadBlock map.
        root_sdfg = parent_sdfg
        while root_sdfg.parent_nsdfg_node is not None:
            root_sdfg = root_sdfg.parent_sdfg
        parent_scopes = get_parent_map_and_loop_scopes(root_sdfg, node, parent_state)
        for scope in parent_scopes:
            if (isinstance(scope, dace.sdfg.nodes.MapEntry) and scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                raise ValueError("ExpandSharedMemoryCopy (collective) must not be nested "
                                 "inside a GPU_ThreadBlock map.  The cooperative copy IS "
                                 "the thread-block-level operation.")

        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
        map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

        # Use CopyND for the actual data movement, then __syncthreads()
        # to ensure all threads see the updated shared memory.
        # TODO: replace with a proper cooperative copy that distributes
        # work across threads once DaCe's GPU scheduling supports bare
        # thread-block-level tasklets.
        copynd_call = _build_copynd_call(inp.dtype.ctype, map_lengths, in_strides_collapsed, out_strides_collapsed)
        code = copynd_call + "\n__syncthreads();"

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        tasklet = state.add_tasklet(name="shared_copy",
                                    inputs={"_cpy_in"},
                                    outputs={"_cpy_out"},
                                    code=code,
                                    language=dace.Language.CPP)

        state.add_edge(
            in_access, None, tasklet, "_cpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))
        state.add_edge(
            tasklet, "_cpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in map_lengths])))

        return sdfg


# ===================================================================
# CopyLibraryNode
# ===================================================================


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    """
    A library node representing a data copy between two access nodes.

    Carries explicit ``src_storage`` and ``dst_storage`` properties so that
    expansion selection can be done without walking the surrounding graph.

    ============================================ ================================
    Expansion                                    Use case
    ============================================ ================================
    ``pure``                                     Same-side mapped tasklet
    ``DirectAssignment``                         Bare ``_out = _in`` tasklet
                                                 (Reg<->Reg, Global->Reg, same)
    ``CopyND``                                   dace::CopyNDDynamic (any copy)
    ``CUDA``                                     GPU_Global <-> GPU_Global
    ``CPU``                                      CPU_Heap <-> CPU_Heap (memcpy)
    ``CUDAHostToDevice``                         CPU -> GPU (cudaMemcpy H2D)
    ``CUDADeviceToHost``                         GPU -> CPU (cudaMemcpy D2H)
    ``RegisterCopy``                             Register <-> Register (strict)
    ``SharedMemoryCopy``                         Global<->Shared (collective),
                                                 Shared<->Register (thread-level)
    ============================================ ================================
    """

    implementations = {
        "pure": ExpandPure,
        "DirectAssignment": ExpandDirectAssignment,
        "CopyND": ExpandCopyND,
        "CUDA": ExpandCUDA,
        "CPU": ExpandCPU,
        "CUDAHostToDevice": ExpandCUDAHostToDevice,
        "CUDADeviceToHost": ExpandCUDADeviceToHost,
        "RegisterCopy": ExpandRegisterCopy,
        "SharedMemoryCopy": ExpandSharedMemoryCopy,
    }
    default_implementation = 'pure'

    # Storage metadata -- populated by InsertCopyNodes or by the user.
    src_storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                          default=dtypes.StorageType.Default,
                                          desc="Storage type of the source access node.")
    dst_storage = properties.EnumProperty(dtype=dtypes.StorageType,
                                          default=dtypes.StorageType.Default,
                                          desc="Storage type of the destination access node.")

    def __init__(self,
                 name,
                 src_storage=dtypes.StorageType.Default,
                 dst_storage=dtypes.StorageType.Default,
                 *args,
                 **kwargs):
        super().__init__(name, *args, inputs={"_in"}, outputs={"_out"}, **kwargs)
        self.src_storage = src_storage
        self.dst_storage = dst_storage

    def validate(self, sdfg, state, allow_cross_storage=True):
        """
        Validates the copy operation.

        :param allow_cross_storage: If True, allows source and destination to
            reside in different storage types (e.g. CPU -> GPU).  Defaults to
            True so that SDFG-level validation passes; same-storage-only
            expansions call with ``False`` at expansion time.
        :return: A tuple ``(inp_name, inp, in_subset, out_name, out,
            out_subset, dynamic_inputs, stream_input)`` where
            ``stream_input`` is the source data descriptor connected to the
            optional ``stream`` in-connector (``None`` if absent).
        """
        result = _validate_copy_edges(self, sdfg, state)
        inp = result[1]
        out = result[4]

        if not allow_cross_storage and inp.storage != out.storage:
            raise ValueError("Input and output storage types must match for this "
                             f"expansion (got {inp.storage} vs {out.storage}).  "
                             "Use a cross-storage expansion or the pure fallback.")

        return result
