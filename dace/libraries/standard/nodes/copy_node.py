# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CopyLibraryNode`` to represent copies explicitly on the SDFG IR.

The expansion is selected per-instance; all expansions accept an optional
``stream`` in-connector so generated GPU kernels/memcpies bind to a caller-provided
``gpuStream_t`` instead of ``__dace_current_stream``.
"""
import copy
from typing import Optional

import dace
from dace import library, nodes, dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from functools import reduce
import operator
from dace.codegen.common import sym2cpp

from dace.libraries.standard.helper import (STREAM_CONN as _STREAM_CONN, add_dynamic_inputs, collapse_shape_and_strides,
                                            extract_stream_and_dynamic_inputs)
from dace.sdfg.construction_utils import get_parent_map_and_loop_scopes

# Must differ from STREAM_CONN: the expansion adds a nested-SDFG array of
# that name, and DaCe rejects tasklet connectors colliding with array names.
_STREAM_TASKLET_CONN = "_stream_in"

# A mapped tasklet cannot read CPU and write GPU (or vice versa) in one scope.
_CPU_STORAGES = {
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
    dtypes.StorageType.Register,
}
_GPU_STORAGES = {
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
    dtypes.StorageType.Register,
}
_CUDA_MEMCPY_IMPLS = {'CUDA', 'CUDAHostToDevice', 'CUDADeviceToHost'}


def _validate_copy_edges(node, sdfg, state):
    """Validate a copy library node's edges, returns necessary fields for expansion."""
    data_oes = [oe for oe in state.out_edges(node) if oe.src_conn == "_out"]
    if len(data_oes) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one `_out` output edge.")
    oe = data_oes[0]
    out = sdfg.arrays[oe.data.data]
    out_subset = oe.data.subset
    out_name = oe.src_conn

    stream_input, dynamic_inputs = extract_stream_and_dynamic_inputs(node, sdfg, state, reserved_conns=("_in", ))

    data_ies = [ie for ie in state.in_edges(node) if ie.dst_conn == "_in"]
    if len(data_ies) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one data input edge "
                         "connected to the `_in` connector.")
    ie = data_ies[0]
    inp = sdfg.arrays[ie.data.data]
    in_subset = ie.data.subset
    inp_name = ie.dst_conn

    if inp.dtype != out.dtype:
        raise ValueError("Input and output data types must match "
                         f"(got {inp.dtype} vs {out.dtype}).")

    return inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input


def _is_cross_cpu_gpu(src_storage, dst_storage):
    """Return True if src and dst straddle the CPU/GPU boundary. ``Register``
    adopts the side of its enclosing scope and is never reported as cross-boundary."""
    if src_storage == dtypes.StorageType.Register or dst_storage == dtypes.StorageType.Register:
        return False
    src_cpu = src_storage in _CPU_STORAGES
    src_gpu = src_storage in _GPU_STORAGES
    dst_cpu = dst_storage in _CPU_STORAGES
    dst_gpu = dst_storage in _GPU_STORAGES
    return (src_cpu and dst_gpu) or (src_gpu and dst_cpu)


def _auto_select_copy_implementation(src_storage, dst_storage):
    """Return a CUDA-family implementation name for cross-CPU/GPU copies."""
    src_gpu = src_storage == dtypes.StorageType.GPU_Global
    dst_gpu = dst_storage == dtypes.StorageType.GPU_Global
    cpu_side = _CPU_STORAGES | {dtypes.StorageType.Default}
    src_cpu = src_storage in cpu_side and src_storage != dtypes.StorageType.Register
    dst_cpu = dst_storage in cpu_side and dst_storage != dtypes.StorageType.Register

    if src_cpu and dst_gpu:
        return 'CUDAHostToDevice'
    if src_gpu and dst_cpu:
        return 'CUDADeviceToHost'
    if src_gpu and dst_gpu:
        return 'CUDA'
    return None


def _cuda2d_strides_are_supported(copy_shape, src_strides, dst_strides):
    """Return True if the (collapsed) 2D strides match an ``ExpandCUDA2D`` pattern."""
    if len(copy_shape) != 2 or len(src_strides) != 2 or len(dst_strides) != 2:
        return False
    if src_strides[1] == 1 and dst_strides[1] == 1:
        return True
    if src_strides[0] == 1 and dst_strides[0] == 1:
        return True
    try:
        return (src_strides[0] / src_strides[1] == copy_shape[1] and dst_strides[0] / dst_strides[1] == copy_shape[1])
    except (TypeError, ZeroDivisionError):
        return False


def _refine_cuda_impl_for_subsets(node, parent_state, parent_sdfg):
    """Upgrade CUDA-family impls to ``CUDA2D`` for 2D-strided subsets; raise for more complex strided patterns."""
    inp_name, inp, in_subset, out_name, out, out_subset, _dyn, _stream = node.validate(parent_sdfg,
                                                                                       parent_state,
                                                                                       allow_cross_storage=True)

    if in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out):
        return None

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    if (len(in_shape_collapsed) == 2 and len(out_shape_collapsed) == 2
            and _cuda2d_strides_are_supported(in_shape_collapsed, in_strides_collapsed, out_strides_collapsed)):
        return 'CUDA2D'

    # 1D strided ([N] with stride != 1 on both sides) maps onto cudaMemcpy2D as a
    # degenerate (1, N) copy: width = 1 element, height = N, pitch = stride.
    if (len(in_shape_collapsed) == 1 and len(out_shape_collapsed) == 1
            and in_shape_collapsed[0] == out_shape_collapsed[0]):
        return 'CUDA2D'

    # Same-side strided ND (e.g. GPU<->GPU) that cannot be collapsed to 1D/2D.
    # CopyND is the efficient route but assumes both endpoints are C-packed
    # (row-major contiguous, no padding in any dim) — its template stride
    # args are derived per-dim under that assumption. For Fortran-packed or
    # padded arrays fall back to the `pure` mapped tasklet, which addresses
    # each element through the array's own stride-aware indexing and so
    # handles arbitrary same-side stride patterns.
    if not _is_cross_cpu_gpu(inp.storage, out.storage):
        if inp.is_packed_c_strides() and out.is_packed_c_strides():
            return 'CopyND'
        return 'pure'

    # Cross-boundary ND-strided: expand into a Sequential SDFG Map of
    # cudaMemcpyAsync where the chunked dimension is any axis with stride 1
    # on both sides (innermost for C-packed, outermost for Fortran-packed).
    if (len(in_shape_collapsed) == len(out_shape_collapsed) and len(in_shape_collapsed) >= 1
            and any(in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1
                    for d in range(len(in_shape_collapsed)))):
        return 'CUDANDStrided'

    raise ValueError(f"CopyLibraryNode '{node.name}' has a strided cross-CPU/GPU copy pattern that "
                     f"cannot be lowered to a single cudaMemcpy or cudaMemcpy2DAsync and has no "
                     f"common stride-1 axis for chunked memcpy "
                     f"(src_shape={in_shape_collapsed}, src_strides={in_strides_collapsed}, "
                     f"dst_shape={out_shape_collapsed}, dst_strides={out_strides_collapsed}); "
                     f"pick an explicit implementation manually.")


def _add_stream_descriptor(sdfg: dace.SDFG, stream_input: Optional[dace.data.Data]):
    """Mirror the parent-side ``stream`` descriptor onto the expansion SDFG."""
    if stream_input is None:
        return
    desc = copy.deepcopy(stream_input)
    desc.transient = False
    sdfg.add_datadesc(_STREAM_CONN, desc)


def _wire_stream_to(sdfg: dace.SDFG, state: dace.SDFGState, target: nodes.Node, target_conn: str,
                    stream_input: Optional[dace.data.Data]):
    """Connect the SDFG-level ``stream`` access node to ``target`` on ``target_conn``.

    No-op if the node has no ``stream`` input. For map entries the connector is
    added on the target; for tasklets it must already exist.
    """
    if stream_input is None:
        return
    stream_access = state.add_access(_STREAM_CONN)
    if isinstance(target, nodes.MapEntry):
        target.add_in_connector(target_conn)
    state.add_edge(stream_access, None, target, target_conn,
                   dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))


def _require_contiguous_subset(name: str, subset, desc, side: str):
    """Raise if ``subset`` is not contiguous in the array described by ``desc``."""
    if not subset.is_contiguous_subset(desc):
        raise ValueError(f"This expansion requires a contiguous {side} subset, but '{name}' subset "
                         f"{subset} is not contiguous in shape {desc.shape} with strides {desc.strides}; "
                         f"use a strided expansion (pure, CopyND, Assignment) instead.")


def _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=False, require_contiguous=False):
    """Shared validation + SDFG skeleton for copy expansions.

    ``require_contiguous=True`` enforces contiguous subsets on both sides, as needed by
    the flat-memcpy expansions (CPU, CUDA, H2D, D2H).
    """
    inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input = node.validate(
        parent_sdfg, parent_state, allow_cross_storage=allow_cross_storage)

    if require_contiguous:
        _require_contiguous_subset(inp_name, in_subset, inp, "input")
        _require_contiguous_subset(out_name, out_subset, out, "output")

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
    """Mapped-tasklet expansion. Valid for same-storage; raises for copies that cross the CPU/GPU boundary."""
    (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, _, _, _, _,
     stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=allow_cross_storage)

    if _is_cross_cpu_gpu(inp.storage, out.storage):
        raise ValueError("Pure (mapped tasklet) expansion cannot handle copies across the "
                         f"CPU/GPU boundary (got {inp.storage} -> {out.storage}). "
                         "Use CUDAHostToDevice or CUDADeviceToHost expansion instead.")

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

    if schedule == dace.dtypes.ScheduleType.GPU_Device:
        _wire_stream_to(sdfg, state, map_entry, _STREAM_CONN, stream_input)

    return sdfg


def _stream_expr_for_tasklet(tasklet_inputs: set, stream_input) -> str:
    """Add the tasklet stream connector if needed, and return the stream-arg expression."""
    if stream_input is None:
        return "__dace_current_stream"
    tasklet_inputs.add(_STREAM_TASKLET_CONN)
    return _STREAM_TASKLET_CONN


def _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, direction):
    """Return a Tasklet emitting ``cudaMemcpyAsync`` for ``direction``."""
    allow_cross = direction != "DeviceToDevice"
    inp_name, inp, in_subset, out_name, out, out_subset, _dyn, stream_input = node.validate(
        parent_sdfg, parent_state, allow_cross_storage=allow_cross)
    _require_contiguous_subset(inp_name, in_subset, inp, "input")
    _require_contiguous_subset(out_name, out_subset, out, "output")

    cp_size = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in in_subset], 1)

    # GPU side is pointer-typed; single-element CPU side stays value-typed and
    # is addressed via ``&_in`` / ``&_out`` in the tasklet code.
    in_is_cpu = direction.startswith("Host")
    out_is_cpu = direction.endswith("Host")
    one_elem = (cp_size == 1)
    in_value_typed = one_elem and in_is_cpu
    out_value_typed = one_elem and out_is_cpu

    in_conn_type = inp.dtype if in_value_typed else dace.dtypes.pointer(inp.dtype)
    out_conn_type = out.dtype if out_value_typed else dace.dtypes.pointer(out.dtype)
    in_arg = '&_in' if in_value_typed else '_in'
    out_arg = '&_out' if out_value_typed else '_out'

    has_stream = stream_input is not None
    stream_expr = _STREAM_CONN if has_stream else "__dace_current_stream"

    code = (f"cudaMemcpyAsync({out_arg}, {in_arg}, "
            f"{sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}), "
            f"cudaMemcpy{direction}, {stream_expr});")

    in_conns = {"_in": in_conn_type}
    if has_stream:
        in_conns[_STREAM_CONN] = dace.dtypes.gpuStream_t
    tasklet = nodes.Tasklet(node.name,
                            inputs=in_conns,
                            outputs={"_out": out_conn_type},
                            code=code,
                            language=dace.Language.CPP)
    return tasklet


def _make_thread_level_copy(node, parent_state, parent_sdfg):
    """Thread-level copy: Sequential mapped tasklet."""
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


def _build_copynd_call(ctype, copy_shape, src_strides, dst_strides, in_arg='_cpy_in', out_arg='_cpy_out'):
    """
    Builds a ``dace::CopyND`` or ``dace::CopyNDDynamic`` call string, using the most specific (static) variant possible.

    ``in_arg`` / ``out_arg`` are the connector / variable names for the input
    and output pointers in the generated tasklet body — callers pass their
    actual connector names (e.g. ``_in`` / ``_out`` when the expansion
    returns a Tasklet directly into the parent CopyLibraryNode's connectors).

    Selection logic (matches ``cpu.py`` codegen):

    1. If all copy dimensions are concrete integers:
       ``dace::CopyND<T, 1, false, dim0, dim1, ...>``
    2. Otherwise (symbolic dims):
       ``dace::CopyNDDynamic<T, 1, false, ndims>``
    3. If all **dst** strides are constexpr: ``::template ConstDst<s0, s1>``
    4. Else if all **src** strides are constexpr: ``::template ConstSrc<s0, s1>``
    5. Else: ``::Dynamic``
    6. Remaining (non-template) values passed as runtime args.
    """
    from dace import symbolic

    ndims = len(copy_shape)
    shape_strs = [sym2cpp(s) for s in copy_shape]
    src_stride_strs = [sym2cpp(s) for s in src_strides]
    dst_stride_strs = [sym2cpp(s) for s in dst_strides]

    dims_static = not any(symbolic.issymbolic(s) for s in copy_shape)
    src_static = not any(symbolic.issymbolic(s) for s in src_strides)
    dst_static = not any(symbolic.issymbolic(s) for s in dst_strides)

    if dims_static:
        copy_tmpl = f"dace::CopyND<{ctype}, 1, false, {', '.join(shape_strs)}>"
    else:
        copy_tmpl = f"dace::CopyNDDynamic<{ctype}, 1, false, {ndims}>"

    if dst_static:
        shape_tmpl = f"template ConstDst<{', '.join(dst_stride_strs)}>"
    elif src_static:
        shape_tmpl = f"template ConstSrc<{', '.join(src_stride_strs)}>"
    else:
        shape_tmpl = "Dynamic"

    # Per dimension, pass only the values NOT in the template.
    # CopyND runtime API order per dim:
    #   CopyND:        [src_stride | dst_stride | src_stride, dst_stride]
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

    all_args = [in_arg, out_arg] + stride_args
    return f"{copy_tmpl}::{shape_tmpl}::Copy({', '.join(all_args)});"


def _generate_assignment_code(ctype, copy_shape, in_strides, out_strides):
    """Build a C++  assignment loop for a direct copy.

    1D uses a simple unrolled loop; ND delinearizes the flat iteration index into
    per-dimension indices, respecting ``in_strides`` / ``out_strides``.
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


@library.expansion
class ExpandPure(ExpandTransformation):
    """Default expansion: mapped tasklet copying element-by-element.

    Handles same-storage copies and GPU-side cross-storage. Raises across the CPU/GPU boundary.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_mapped_tasklet_expansion(node, parent_state, parent_sdfg, allow_cross_storage=True)


@library.expansion
class ExpandCopyND(ExpandTransformation):
    """Runtime fallback for strided ND copies, via a ``dace::CopyNDDynamic`` C++ tasklet.

    Works for any same-side copy (CPU or GPU) without generating a map.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, _map_lengths, in_shape_collapsed,
         in_strides_collapsed, out_shape_collapsed, out_strides_collapsed,
         _stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=True)

        if _is_cross_cpu_gpu(inp.storage, out.storage):
            raise ValueError("CopyND expansion cannot handle copies across the CPU/GPU "
                             f"boundary (got {inp.storage} -> {out.storage}).")

        # CopyND assumes both endpoints are C-packed (row-major contiguous,
        # no padding). Its `dace::CopyND<>` template stride args are derived
        # per-dim under that assumption. Reject explicitly so a caller that
        # forced ``implementation='CopyND'`` on a Fortran-packed or
        # padded-strides array gets a clean error instead of a silent
        # mis-copy at runtime.
        if not inp.is_packed_c_strides() or not out.is_packed_c_strides():
            raise ValueError(f"CopyND expansion requires C-packed (row-major contiguous) strides on both "
                             f"endpoints; got src strides {tuple(inp.strides)} for shape {tuple(inp.shape)} "
                             f"and dst strides {tuple(out.strides)} for shape {tuple(out.shape)}. "
                             f"Use a different expansion (e.g. 'pure' for same-storage, or convert to a Map "
                             f"via CopyToMap before expansion).")

        # Operate on the *collapsed* rank consistently: the wrapper SDFG's
        # arrays are sized by collapsed shape ([_make_expansion_sdfg] creates
        # them with `in_shape_collapsed` / `out_shape_collapsed`). Mixing in
        # full-rank `map_lengths` here triggers IndexError in
        # `_build_copynd_call` and produces out-of-bounds memlet subsets.
        code = _build_copynd_call(inp.dtype.ctype, in_shape_collapsed, in_strides_collapsed, out_strides_collapsed)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        tasklet = state.add_tasklet(name="copynd_tasklet",
                                    inputs={"_cpy_in"},
                                    outputs={"_cpy_out"},
                                    code=code,
                                    language=dace.Language.CPP)

        state.add_edge(
            in_access, None, tasklet, "_cpy_in",
            dace.memlet.Memlet(data=inp_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in in_shape_collapsed])))
        state.add_edge(
            tasklet, "_cpy_out", out_access, None,
            dace.memlet.Memlet(data=out_name, subset=dace.subsets.Range([(0, e - 1, 1) for e in out_shape_collapsed])))

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """GPU_Global <-> GPU_Global contiguous copy via ``cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)``."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "DeviceToDevice")


@library.expansion
class ExpandCPU(ExpandTransformation):
    """CPU_Heap <-> CPU_Heap contiguous copy via ``std::memcpy``."""
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, map_lengths, _, _, _, _,
         _stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, require_contiguous=True)

        cp_size = reduce(operator.mul, map_lengths, 1)

        in_access = state.add_access(inp_name)
        out_access = state.add_access(out_name)

        # Connector typing for ``memcpy(dst, src, n)``:
        #
        # Both sides are CPU. The codegen passes single-element CPU subsets
        # as ``T&`` (reference) and multi-element subsets as ``T*``. We keep
        # the value-typed connector for the single-element case and prefix
        # ``&`` in the tasklet code, which works regardless of which side
        # the codegen made a reference. See the longer note in
        # ``_make_cuda_memcpy_expansion`` for the codegen-side reasoning.
        one_elem = (cp_size == 1)
        in_conn = inp.dtype if one_elem else dace.dtypes.pointer(inp.dtype)
        out_conn = out.dtype if one_elem else dace.dtypes.pointer(out.dtype)
        in_arg = '&_memcpy_in' if one_elem else '_memcpy_in'
        out_arg = '&_memcpy_out' if one_elem else '_memcpy_out'

        tasklet = state.add_tasklet(
            name="memcpy_tasklet",
            inputs={"_memcpy_in": in_conn},
            outputs={"_memcpy_out": out_conn},
            code=f"memcpy({out_arg}, {in_arg}, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}));",
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
    """CPU_Heap/CPU_Pinned -> GPU_Global contiguous copy via ``cudaMemcpyAsync(..., cudaMemcpyHostToDevice)``."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "HostToDevice")


@library.expansion
class ExpandCUDADeviceToHost(ExpandTransformation):
    """GPU_Global -> CPU_Heap/CPU_Pinned contiguous copy via ``cudaMemcpyAsync(..., cudaMemcpyDeviceToHost)``."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg, "DeviceToHost")


@library.expansion
class ExpandCUDA2D(ExpandTransformation):
    """2D strided copy via ``cudaMemcpy2DAsync`` between any combination of GPU_Global and host storage.

    Handles three stride patterns: row-major contiguous rows, column-major contiguous columns,
    and the degenerate case where the outer stride is a multiple of the inner.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, _dyn, stream_input = node.validate(
            parent_sdfg, parent_state, allow_cross_storage=True)

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        # 1D-collapsed shapes get promoted to (N, 1) so a single cudaMemcpy2D
        # call covers strided 1D patterns.
        if len(in_shape_collapsed) == 1 and len(out_shape_collapsed) == 1:
            in_shape_2d = [in_shape_collapsed[0], 1]
            out_shape_2d = [out_shape_collapsed[0], 1]
            in_strides_2d = [in_strides_collapsed[0], 1]
            out_strides_2d = [out_strides_collapsed[0], 1]
        elif len(in_shape_collapsed) == 2 and len(out_shape_collapsed) == 2:
            in_shape_2d = in_shape_collapsed
            out_shape_2d = out_shape_collapsed
            in_strides_2d = in_strides_collapsed
            out_strides_2d = out_strides_collapsed
        else:
            raise ValueError("ExpandCUDA2D requires 1D or 2D collapsed shapes, got "
                             f"{in_shape_collapsed} (src) / {out_shape_collapsed} (dst).")

        src_loc = "Device" if inp.storage == dtypes.StorageType.GPU_Global else "Host"
        dst_loc = "Device" if out.storage == dtypes.StorageType.GPU_Global else "Host"
        kind = f"cudaMemcpy{src_loc}To{dst_loc}"

        copy_shape = in_shape_2d
        src_strides = in_strides_2d
        dst_strides = out_strides_2d
        ctype = inp.dtype.ctype

        if src_strides[1] == 1 and dst_strides[1] == 1:
            dpitch = f"{sym2cpp(dst_strides[0])} * sizeof({ctype})"
            spitch = f"{sym2cpp(src_strides[0])} * sizeof({ctype})"
            width = f"{sym2cpp(copy_shape[1])} * sizeof({ctype})"
            height = sym2cpp(copy_shape[0])
        elif src_strides[0] == 1 and dst_strides[0] == 1:
            dpitch = f"{sym2cpp(dst_strides[1])} * sizeof({ctype})"
            spitch = f"{sym2cpp(src_strides[1])} * sizeof({ctype})"
            width = f"{sym2cpp(copy_shape[0])} * sizeof({ctype})"
            height = sym2cpp(copy_shape[1])
        elif (src_strides[0] / src_strides[1] == copy_shape[1] and dst_strides[0] / dst_strides[1] == copy_shape[1]):
            dpitch = f"{sym2cpp(dst_strides[1])} * sizeof({ctype})"
            spitch = f"{sym2cpp(src_strides[1])} * sizeof({ctype})"
            width = f"sizeof({ctype})"
            height = sym2cpp(copy_shape[0] * copy_shape[1])
        else:
            raise NotImplementedError(f"Unsupported 2D memory copy: shape={copy_shape}, "
                                      f"src_strides={src_strides}, dst_strides={dst_strides}.")

        has_stream = stream_input is not None
        stream_expr = _STREAM_CONN if has_stream else "__dace_current_stream"
        code = (f"cudaMemcpy2DAsync(_out, {dpitch}, _in, {spitch}, "
                f"{width}, {height}, {kind}, {stream_expr});")

        in_conns = {"_in": dace.dtypes.pointer(inp.dtype)}
        if has_stream:
            in_conns[_STREAM_CONN] = dace.dtypes.gpuStream_t
        tasklet = nodes.Tasklet(node.name,
                                inputs=in_conns,
                                outputs={"_out": dace.dtypes.pointer(out.dtype)},
                                code=code,
                                language=dace.Language.CPP)
        return tasklet


@library.expansion
class ExpandCUDANDStrided(ExpandTransformation):
    """ND-strided cross-boundary copy via a Sequential SDFG Map of ``cudaMemcpyAsync``.

    Fallback for ≥3D-strided patterns that cannot be expressed as a single
    ``cudaMemcpyAsync`` or ``cudaMemcpy2DAsync``. The expansion returns a
    nested SDFG containing a ``Sequential`` Map over all collapsed
    dimensions except the *chunk axis* — the axis with ``stride == 1`` on
    both sides. Each map iteration's tasklet issues one
    ``cudaMemcpyAsync`` for the contiguous run along the chunk axis. For
    C-packed arrays this is the innermost dim; for Fortran-packed arrays
    it is the outermost. The kind of memcpy (``HostToDevice`` /
    ``DeviceToHost`` / ``HostToHost`` / ``DeviceToDevice``) is selected
    from the endpoint storages.

    The inner tasklet's stream connector is named ``_cpy_stream`` (not
    ``stream``) to avoid shadowing the wrapper SDFG's ``stream`` AccessNode/
    array name in the codegen scope, which would otherwise emit
    ``gpuStream_t stream = stream;`` self-init.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        (sdfg, state, inp_name, inp, in_subset, out_name, out, out_subset, _map_lengths, in_shape_collapsed,
         in_strides_collapsed, out_shape_collapsed, out_strides_collapsed,
         stream_input) = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=True)

        if len(in_shape_collapsed) != len(out_shape_collapsed):
            raise NotImplementedError("ExpandCUDANDStrided requires src and dst to share the collapsed rank "
                                      f"(got {in_shape_collapsed} vs {out_shape_collapsed}).")
        ndims = len(in_shape_collapsed)
        if ndims < 1:
            raise NotImplementedError("ExpandCUDANDStrided requires at least one collapsed dimension.")

        # Pick the chunk axis: any dim with stride 1 on both sides. Prefer
        # the innermost (C-packed) when multiple match.
        chunk_dim = None
        for d in reversed(range(ndims)):
            if in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1:
                chunk_dim = d
                break
        if chunk_dim is None:
            raise NotImplementedError("ExpandCUDANDStrided requires at least one common stride-1 axis on both sides "
                                      f"(got src_strides={in_strides_collapsed}, dst_strides={out_strides_collapsed}).")

        ctype = inp.dtype.ctype
        chunk = sym2cpp(in_shape_collapsed[chunk_dim])

        src_loc = "Device" if inp.storage == dtypes.StorageType.GPU_Global else "Host"
        dst_loc = "Device" if out.storage == dtypes.StorageType.GPU_Global else "Host"
        kind = f"cudaMemcpy{src_loc}To{dst_loc}"

        has_stream = stream_input is not None
        # Avoid the connector name `stream` colliding with the wrapper SDFG's
        # `stream` array name in the codegen scope.
        _INNER_STREAM_CONN = "_cpy_stream"
        stream_expr = _INNER_STREAM_CONN if has_stream else "__dace_current_stream"

        # Map over all dims except chunk_dim; the chunk axis is the contiguous
        # run passed to cudaMemcpyAsync. ndims == 1 degenerates to a single
        # contiguous memcpy (no map).
        map_axes = [d for d in range(ndims) if d != chunk_dim]
        map_params = [f"__cpy_i{d}" for d in map_axes]
        map_ranges = {p: f"0:{sym2cpp(in_shape_collapsed[d])}" for d, p in zip(map_axes, map_params)}

        def _row_subset(shape):
            parts = []
            map_pi = 0
            for d in range(ndims):
                if d == chunk_dim:
                    parts.append(f"0:{sym2cpp(shape[d])}")
                else:
                    parts.append(map_params[map_pi])
                    map_pi += 1
            return ", ".join(parts)

        if ndims > 1:
            row_in = _row_subset(in_shape_collapsed)
            row_out = _row_subset(out_shape_collapsed)
        else:
            row_in = f"0:{sym2cpp(in_shape_collapsed[0])}"
            row_out = f"0:{sym2cpp(out_shape_collapsed[0])}"

        in_memlet = dace.memlet.Memlet(data=inp_name, subset=row_in)
        out_memlet = dace.memlet.Memlet(data=out_name, subset=row_out)

        code = f"DACE_GPU_CHECK(cudaMemcpyAsync(_cpy_out, _cpy_in, {chunk} * sizeof({ctype}), {kind}, {stream_expr}));"

        in_conn_type = dace.dtypes.pointer(inp.dtype)
        out_conn_type = dace.dtypes.pointer(out.dtype)

        if ndims == 1:
            in_access = state.add_access(inp_name)
            out_access = state.add_access(out_name)
            tasklet_in_conns = {"_cpy_in": in_conn_type}
            if has_stream:
                tasklet_in_conns[_INNER_STREAM_CONN] = dace.dtypes.gpuStream_t
            tasklet = state.add_tasklet(name=f"{node.label}_memcpy",
                                        inputs=tasklet_in_conns,
                                        outputs={"_cpy_out": out_conn_type},
                                        code=code,
                                        language=dace.Language.CPP)
            state.add_edge(in_access, None, tasklet, "_cpy_in", in_memlet)
            state.add_edge(tasklet, "_cpy_out", out_access, None, out_memlet)
            if has_stream:
                stream_access = state.add_access(_STREAM_CONN)
                state.add_edge(stream_access, None, tasklet, _INNER_STREAM_CONN,
                               dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))
        else:
            inner_tasklet, map_entry, _map_exit = state.add_mapped_tasklet(name=f"{node.label}_tasklet",
                                                                           map_ranges=map_ranges,
                                                                           inputs={"_cpy_in": in_memlet},
                                                                           code=code,
                                                                           outputs={"_cpy_out": out_memlet},
                                                                           schedule=dace.dtypes.ScheduleType.Sequential,
                                                                           language=dace.Language.CPP,
                                                                           external_edges=True)
            # Force pointer connectors on the inner tasklet so the codegen
            # types `_cpy_in`/`_cpy_out` as `T*` (matching cudaMemcpyAsync's
            # signature) instead of dereferencing them as values.
            inner_tasklet.in_connectors["_cpy_in"] = in_conn_type
            inner_tasklet.out_connectors["_cpy_out"] = out_conn_type
            if has_stream:
                # Wire wrapper's `stream` AccessNode -> MapEntry (IN_stream /
                # OUT_stream pass-through) -> inner Tasklet (`_cpy_stream`).
                inner_tasklet.add_in_connector(_INNER_STREAM_CONN, dace.dtypes.gpuStream_t)
                stream_access = state.add_access(_STREAM_CONN)
                in_conn_name = f"IN_{_STREAM_CONN}"
                out_conn_name = f"OUT_{_STREAM_CONN}"
                map_entry.add_in_connector(in_conn_name)
                map_entry.add_out_connector(out_conn_name)
                state.add_edge(stream_access, None, map_entry, in_conn_name,
                               dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))
                state.add_edge(map_entry, out_conn_name, inner_tasklet, _INNER_STREAM_CONN,
                               dace.memlet.Memlet.from_array(_STREAM_CONN, sdfg.arrays[_STREAM_CONN]))

        return sdfg


@library.expansion
class ExpandDirectAssignment(ExpandTransformation):
    """Bare ``_out = _in`` assignment tasklet (no map)."""
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


@library.expansion
class ExpandRegisterCopy(ExpandTransformation):
    """Strict Register<->Register copy; equivalent to ``ExpandDirectAssignment``."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(
            parent_sdfg, parent_state, allow_cross_storage=False)

        if inp.storage != dtypes.StorageType.Register:
            raise ValueError(f"ExpandRegisterCopy expects Register input storage, got {inp.storage}.")
        if out.storage != dtypes.StorageType.Register:
            raise ValueError(f"ExpandRegisterCopy expects Register output storage, got {out.storage}.")

        return _make_thread_level_copy(node, parent_state, parent_sdfg)


@library.expansion
class ExpandSharedMemoryCopy(ExpandTransformation):
    """Copies involving GPU shared memory, inside a GPU kernel.

    - **Global <-> Shared** or **Shared <-> Shared**: C++ tasklet calling
      ``dace::CopyNDDynamic`` followed by ``__syncthreads()``. This expansion IS the thread-block-level operation.
    - **Shared <-> Register**: thread-level copy (Sequential mapped tasklet, like ``RegisterCopy``).
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(parent_sdfg,
                                                                                               parent_state,
                                                                                               allow_cross_storage=True)

        valid_storages = {dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global, dtypes.StorageType.Register}

        if inp.storage not in valid_storages:
            raise ValueError(f"ExpandSharedMemoryCopy: input storage {inp.storage} is not "
                             "GPU_Shared, GPU_Global, or Register.")
        if out.storage not in valid_storages:
            raise ValueError(f"ExpandSharedMemoryCopy: output storage {out.storage} is not "
                             "GPU_Shared, GPU_Global, or Register.")
        if (inp.storage != dtypes.StorageType.GPU_Shared and out.storage != dtypes.StorageType.GPU_Shared):
            raise ValueError("ExpandSharedMemoryCopy requires at least one side to be GPU_Shared.")

        # Shared <-> Register: thread-level (no cooperative sync needed)
        involves_register = (inp.storage == dtypes.StorageType.Register or out.storage == dtypes.StorageType.Register)
        if involves_register:
            return _make_thread_level_copy(node, parent_state, parent_sdfg)

        # Global/Shared <-> Shared: block-collective copy; this expansion must not sit inside
        # a parent GPU_ThreadBlock map because it IS that thread-block-level operation.
        root_sdfg = parent_sdfg
        while root_sdfg.parent_nsdfg_node is not None:
            root_sdfg = root_sdfg.parent_sdfg
        parent_scopes = get_parent_map_and_loop_scopes(root_sdfg, node, parent_state)
        for scope in parent_scopes:
            if (isinstance(scope, dace.sdfg.nodes.MapEntry) and scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                raise ValueError("ExpandSharedMemoryCopy (collective) IS the thread-block-level "
                                 "operation and must not be nested inside a GPU_ThreadBlock map.")

        map_lengths = [(e + 1 - b) // s for (b, e, s) in in_subset]

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
        sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)

        state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
        map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

        # Use CopyND for the data movement; __syncthreads() afterwards so all threads see the update.
        # TODO: replace with a proper cooperative copy distributing work across threads, once DaCe's
        # GPU scheduling supports bare thread-block-level tasklets.
        # Note: pass `in_shape_collapsed` (collapsed rank), not `map_lengths`
        # (full rank) — `_build_copynd_call`'s per-dim loop indexes
        # `src_stride_strs[d]` / `dst_stride_strs[d]` which are sized by
        # the collapsed strides; mismatched ranks would IndexError.
        copynd_call = _build_copynd_call(inp.dtype.ctype, in_shape_collapsed, in_strides_collapsed,
                                         out_strides_collapsed)
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


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    """Library node representing a data copy between two access nodes.

    Carries explicit ``src_storage`` / ``dst_storage`` so expansion selection does not need to
    walk the surrounding graph.

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
    ``CUDA2D``                                   2D strided host<->device /
                                                 device<->device via
                                                 cudaMemcpy2DAsync
    ``RegisterCopy``                             Register <-> Register
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
        "CUDA2D": ExpandCUDA2D,
        "CUDANDStrided": ExpandCUDANDStrided,
        "RegisterCopy": ExpandRegisterCopy,
        "SharedMemoryCopy": ExpandSharedMemoryCopy,
    }
    default_implementation = 'pure'

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_in"}, outputs={"_out"}, **kwargs)

    def src_storage(self, state, sdfg) -> dtypes.StorageType:
        in_edges = [e for e in state.in_edges(self) if e.dst_conn == "_in"]
        if not in_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(in_edges[0])[0].src
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return sdfg.arrays[outer.data].storage

    def dst_storage(self, state, sdfg) -> dtypes.StorageType:
        out_edges = [e for e in state.out_edges(self) if e.src_conn == "_out"]
        if not out_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(out_edges[0])[-1].dst
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return sdfg.arrays[outer.data].storage

    def expand(self, state, sdfg=None, *args, **kwargs):
        actual_sdfg = sdfg if sdfg is not None else state.parent
        src_storage = self.src_storage(state, actual_sdfg)
        dst_storage = self.dst_storage(state, actual_sdfg)

        if self.implementation is None:
            auto = _auto_select_copy_implementation(src_storage, dst_storage)
            if auto is not None:
                self.implementation = auto

        if self.implementation in _CUDA_MEMCPY_IMPLS:
            refinement = _refine_cuda_impl_for_subsets(self, state, actual_sdfg)
            if refinement is not None:
                self.implementation = refinement
        return super().expand(state, sdfg, *args, **kwargs)

    def validate(self, sdfg, state, allow_cross_storage=True):
        result = _validate_copy_edges(self, sdfg, state)
        inp = result[1]
        out = result[4]

        if not allow_cross_storage and inp.storage != out.storage:
            raise ValueError("Input and output storage types must match for this "
                             f"expansion (got {inp.storage} vs {out.storage}).  "
                             "Use a cross-storage expansion or the pure fallback.")

        return result
