# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CopyLibraryNode`` to represent copies explicitly on the SDFG IR.

The expansion is selected per-instance; all expansions accept an optional
``stream`` in-connector so generated GPU kernels/memcpies bind to a caller-provided
``gpuStream_t`` instead of ``__dace_current_stream``.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import dace
from dace import data, library, nodes, dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from functools import reduce
import operator
from dace.codegen.common import sym2cpp

from dace.libraries.standard.helper import (STREAM_CONN as _STREAM_CONN, add_dynamic_inputs, add_stream_descriptor as
                                            _add_stream_descriptor, collapse_shape_and_strides,
                                            extract_stream_and_dynamic_inputs, wire_stream_through_map as
                                            _wire_stream_through_map, wire_stream_to as _wire_stream_to)
from dace.sdfg.construction_utils import get_parent_map_and_loop_scopes

# Must differ from STREAM_CONN: the expansion adds a nested-SDFG array of
# that name, and DaCe rejects tasklet connectors colliding with array names.
_STREAM_TASKLET_CONN = "_stream_in"

# A mapped tasklet cannot read CPU and write GPU (or vice versa) in one
# scope. ``Register`` adopts the side of its enclosing scope and is handled
# explicitly by the predicates below — never a member of either set.
_CPU_STORAGES = {
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
}
_GPU_STORAGES = {
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
}


def _is_cross_cpu_gpu(src_storage, dst_storage):
    """Return True if src and dst straddle the CPU/GPU boundary. ``Register``
    adopts its enclosing scope's side and is never cross-boundary."""
    return ((src_storage in _CPU_STORAGES and dst_storage in _GPU_STORAGES)
            or (src_storage in _GPU_STORAGES and dst_storage in _CPU_STORAGES))


def _coarse_pick_for_storage_pair(src_storage, dst_storage):
    """Return ``'MemcpyCUDA1D'`` for any copy involving GPU_Global on at
    least one side, else ``None``. Direction (H2D / D2H / D2D) is inferred
    inside the expansion from the same storages."""
    host_side = _CPU_STORAGES | {dtypes.StorageType.Default}
    src_gpu = src_storage == dtypes.StorageType.GPU_Global
    dst_gpu = dst_storage == dtypes.StorageType.GPU_Global
    src_cpu = src_storage in host_side
    dst_cpu = dst_storage in host_side

    if (src_cpu and dst_gpu) or (src_gpu and dst_cpu) or (src_gpu and dst_gpu):
        return 'MemcpyCUDA1D'
    return None


def select_copy_implementation(node, parent_state, parent_sdfg) -> str:
    """Single source of truth for resolving ``CopyLibraryNode.implementation``
    when set to ``'Auto'`` (the default). Picks a concrete implementation
    from endpoint storages, subset shapes, and the surrounding scope.

    Returns one of the concrete implementation names registered in
    ``CopyLibraryNode.implementations`` — never ``'Auto'`` itself."""
    from dace.sdfg.scope import is_devicelevel_gpu

    inp_name, inp, in_subset, out_name, out, out_subset, _dyn, _stream = node.validate(parent_sdfg,
                                                                                       parent_state,
                                                                                       allow_cross_storage=True)

    # 1. In-device-scope override: ``cudaMemcpyAsync`` cannot be issued from
    # device code. Inline the copy as device-side element-wise loops.
    if is_devicelevel_gpu(parent_sdfg, parent_state, node):
        return 'MappedTasklet'

    # 2. Coarse pick by storage pair: any copy touching GPU memory goes
    # through the cudaMemcpy family; everything else falls through to
    # MappedTasklet at the end.
    impl = _coarse_pick_for_storage_pair(inp.storage, out.storage)

    # 3. Refine for subset patterns (CUDA2D / CUDANDStrided / fall back to
    # same-side mapped tasklet or CopyNDTemplate).
    if impl == 'MemcpyCUDA1D':
        refined = _refine_cuda_impl_for_subsets(node, parent_state, parent_sdfg)
        if refined is not None:
            impl = refined

    return impl or 'MappedTasklet'


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
    """Upgrade ``MemcpyCUDA1D`` to a more specific impl when the subsets
    aren't simple contiguous: ``MemcpyCUDA2D`` (1-or-2D strided patterns),
    ``MappedTasklet`` (same-side GPU strided — runs inside a kernel),
    ``CopyNDTemplate`` (same-side CPU C-packed), or
    ``MemcpyCUDANDStrided`` (cross-boundary ≥3D)."""
    inp_name, inp, in_subset, out_name, out, out_subset, _dyn, _stream = node.validate(parent_sdfg,
                                                                                       parent_state,
                                                                                       allow_cross_storage=True)

    if in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out):
        return None

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    if (len(in_shape_collapsed) == 2 and len(out_shape_collapsed) == 2
            and _cuda2d_strides_are_supported(in_shape_collapsed, in_strides_collapsed, out_strides_collapsed)):
        return 'MemcpyCUDA2D'

    # 1D strided ([N] with stride != 1 on both sides) maps onto cudaMemcpy2D as a
    # degenerate (1, N) copy: width = 1 element, height = N, pitch = stride.
    if (len(in_shape_collapsed) == 1 and len(out_shape_collapsed) == 1
            and in_shape_collapsed[0] == out_shape_collapsed[0]):
        return 'MemcpyCUDA2D'

    # Same-side strided ND. CopyNDTemplate is the efficient route on the CPU
    # side (host-issued unrolled loop) but would dereference device pointers
    # from the host on a GPU<->GPU copy. For GPU same-side, MappedTasklet
    # lands the loop inside a GPU_Device kernel — the only safe place for
    # device-side pointer arithmetic. Same fallback for Fortran-packed /
    # padded arrays on either side.
    if not _is_cross_cpu_gpu(inp.storage, out.storage):
        gpu_side = (inp.storage == dtypes.StorageType.GPU_Global or out.storage == dtypes.StorageType.GPU_Global
                    or inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared)
        if gpu_side:
            return 'MappedTasklet'
        if inp.is_packed_c_strides() and out.is_packed_c_strides():
            return 'CopyNDTemplate'
        return 'MappedTasklet'

    # Cross-boundary ND-strided: Sequential map of cudaMemcpyAsync along any
    # stride-1 axis on both sides.
    if (len(in_shape_collapsed) == len(out_shape_collapsed) and len(in_shape_collapsed) >= 1
            and any(in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1
                    for d in range(len(in_shape_collapsed)))):
        return 'MemcpyCUDANDStrided'

    raise ValueError(f"CopyLibraryNode '{node.name}' has a strided cross-CPU/GPU copy pattern that "
                     f"cannot be lowered to a single cudaMemcpy or cudaMemcpy2DAsync and has no "
                     f"common stride-1 axis for chunked memcpy "
                     f"(src_shape={in_shape_collapsed}, src_strides={in_strides_collapsed}, "
                     f"dst_shape={out_shape_collapsed}, dst_strides={out_strides_collapsed}); "
                     f"pick an explicit implementation manually.")


def _require_contiguous_subset(name: str, subset, desc, side: str):
    """Raise if ``subset`` is not contiguous in the array described by ``desc``."""
    if not subset.is_contiguous_subset(desc):
        raise ValueError(f"This expansion requires a contiguous {side} subset, but '{name}' subset "
                         f"{subset} is not contiguous in shape {desc.shape} with strides {desc.strides}; "
                         f"use a strided expansion (pure, CopyND, Assignment) instead.")


@dataclass
class CopyExpansion:
    """Inputs + collapsed-shape state shared across CopyLibraryNode expansions
    that build a wrapper SDFG. Returned by :func:`_make_expansion_sdfg`."""
    sdfg: dace.SDFG
    state: dace.SDFGState
    inp_name: str
    inp: data.Data
    in_subset: dace.subsets.Range
    out_name: str
    out: data.Data
    out_subset: dace.subsets.Range
    map_lengths: List
    in_shape_collapsed: List
    in_strides_collapsed: List
    out_shape_collapsed: List
    out_strides_collapsed: List
    stream_input: Optional[data.Data]


def _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=False, require_contiguous=False):
    """Shared validation + SDFG skeleton. ``require_contiguous=True`` enforces
    contiguous subsets on both sides (needed by flat-memcpy expansions)."""
    inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input = node.validate(
        parent_sdfg, parent_state, allow_cross_storage=allow_cross_storage)

    if require_contiguous:
        _require_contiguous_subset(inp_name, in_subset, inp, "input")
        _require_contiguous_subset(out_name, out_subset, out, "output")

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    _add_stream_descriptor(sdfg, stream_input)

    state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
    map_lengths = add_dynamic_inputs(dynamic_inputs, sdfg, in_subset, state)

    return CopyExpansion(sdfg=sdfg,
                         state=state,
                         inp_name=inp_name,
                         inp=inp,
                         in_subset=in_subset,
                         out_name=out_name,
                         out=out,
                         out_subset=out_subset,
                         map_lengths=map_lengths,
                         in_shape_collapsed=in_shape_collapsed,
                         in_strides_collapsed=in_strides_collapsed,
                         out_shape_collapsed=out_shape_collapsed,
                         out_strides_collapsed=out_strides_collapsed,
                         stream_input=stream_input)


def _make_mapped_tasklet_expansion(node, parent_state, parent_sdfg, allow_cross_storage=False):
    """Element-wise mapped tasklet expansion. Schedule comes from the
    storages: ``Sequential`` for Register/Register or Register<->GPU_Shared
    (thread-level), ``GPU_Device`` if any side is GPU storage, else
    ``Default``. Raises across the CPU/GPU boundary."""
    ctx = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=allow_cross_storage)
    inp, out = ctx.inp, ctx.out

    if _is_cross_cpu_gpu(inp.storage, out.storage):
        raise ValueError("MappedTasklet expansion cannot cross the CPU/GPU boundary "
                         f"(got {inp.storage} -> {out.storage}). Use a MemcpyCUDA1D variant.")

    # Schedule from storages.
    is_register = lambda s: s == dtypes.StorageType.Register
    is_thread_local = (is_register(inp.storage) and is_register(out.storage)) or (
        (is_register(inp.storage) and out.storage == dtypes.StorageType.GPU_Shared) or
        (is_register(out.storage) and inp.storage == dtypes.StorageType.GPU_Shared))
    if is_thread_local:
        schedule = dtypes.ScheduleType.Sequential
    elif inp.storage in _GPU_STORAGES or out.storage in _GPU_STORAGES:
        schedule = dtypes.ScheduleType.GPU_Device
    else:
        schedule = dtypes.ScheduleType.Default

    ctx.sdfg.schedule = dtypes.ScheduleType.Default

    map_params = [f"__i{i}" for i in range(len(ctx.map_lengths))]
    map_rng = {i: f"0:{s}" for i, s in zip(map_params, ctx.map_lengths)}
    access_expr = ','.join(map_params)
    inputs = {"_cpy_in": dace.memlet.Memlet(f"{ctx.inp_name}[{access_expr}]")}
    outputs = {"_cpy_out": dace.memlet.Memlet(f"{ctx.out_name}[{access_expr}]")}

    _, map_entry, _ = ctx.state.add_mapped_tasklet(f"{node.label}_tasklet",
                                                   map_rng,
                                                   inputs,
                                                   "_cpy_out = _cpy_in",
                                                   outputs,
                                                   schedule=schedule,
                                                   external_edges=True)

    if schedule == dtypes.ScheduleType.GPU_Device:
        _wire_stream_to(ctx.sdfg, ctx.state, map_entry, _STREAM_CONN, ctx.stream_input)

    return ctx.sdfg


def _stream_expr_for_tasklet(tasklet_inputs: set, stream_input) -> str:
    """Add the tasklet stream connector if needed, and return the stream-arg expression."""
    if stream_input is None:
        return "__dace_current_stream"
    tasklet_inputs.add(_STREAM_TASKLET_CONN)
    return _STREAM_TASKLET_CONN


def _memcpy_kind(inp, out) -> str:
    """``cudaMemcpy<src>To<dst>`` from endpoint storages."""
    src_loc = "Device" if inp.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    dst_loc = "Device" if out.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    return f"cudaMemcpy{src_loc}To{dst_loc}"


def _stream_expr(stream_input, conn_name: str = _STREAM_CONN) -> Tuple[bool, str]:
    """Return ``(has_stream, expr)``: connector name when a stream is wired,
    ``__dace_current_stream`` (the legacy ambient placeholder the codegen
    binds) otherwise."""
    if stream_input is None:
        return False, "__dace_current_stream"
    return True, conn_name


def _memcpy_connector_typing(inp, out, one_elem: bool, in_is_cpu: bool, out_is_cpu: bool, in_conn: str, out_conn: str):
    """Single-elem CPU side stays value-typed and is addressed via ``&``;
    every other side is pointer-typed and passed by name. Used by the
    cudaMemcpy and CPU memcpy expansions for connector + tasklet-arg shape.

    Returns ``(in_conn_type, out_conn_type, in_arg, out_arg)``."""
    in_value_typed = one_elem and in_is_cpu
    out_value_typed = one_elem and out_is_cpu
    in_conn_type = inp.dtype if in_value_typed else dace.dtypes.pointer(inp.dtype)
    out_conn_type = out.dtype if out_value_typed else dace.dtypes.pointer(out.dtype)
    in_arg = f'&{in_conn}' if in_value_typed else in_conn
    out_arg = f'&{out_conn}' if out_value_typed else out_conn
    return in_conn_type, out_conn_type, in_arg, out_arg


def _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg):
    """Return a Tasklet emitting one ``cudaMemcpyAsync``. The transfer
    direction (HostToDevice / DeviceToHost / DeviceToDevice / HostToHost) is
    inferred from endpoint storages; cross-CPU/GPU is allowed."""
    inp_name, inp, in_subset, out_name, out, out_subset, _dyn, stream_input = node.validate(parent_sdfg,
                                                                                            parent_state,
                                                                                            allow_cross_storage=True)
    _require_contiguous_subset(inp_name, in_subset, inp, "input")
    _require_contiguous_subset(out_name, out_subset, out, "output")

    cp_size = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in in_subset], 1)
    in_conn_type, out_conn_type, in_arg, out_arg = _memcpy_connector_typing(
        inp,
        out,
        one_elem=(cp_size == 1),
        in_is_cpu=(inp.storage != dtypes.StorageType.GPU_Global),
        out_is_cpu=(out.storage != dtypes.StorageType.GPU_Global),
        in_conn='_in',
        out_conn='_out')

    has_stream, stream_expr = _stream_expr(stream_input)
    kind = _memcpy_kind(inp, out)

    code = (f"cudaMemcpyAsync({out_arg}, {in_arg}, "
            f"{sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}), {kind}, {stream_expr});")

    in_conns = {"_in": in_conn_type}
    if has_stream:
        in_conns[_STREAM_CONN] = dace.dtypes.gpuStream_t
    return nodes.Tasklet(node.name,
                         inputs=in_conns,
                         outputs={"_out": out_conn_type},
                         code=code,
                         language=dace.Language.CPP)


def _build_copynd_call(ctype, copy_shape, src_strides, dst_strides, in_arg='_cpy_in', out_arg='_cpy_out'):
    """Build a ``dace::CopyND`` / ``dace::CopyNDDynamic`` call string,
    picking the most-specific static template form: ``CopyND<T, 1, false,
    dims...>`` for static shapes (else ``CopyNDDynamic<T, 1, false, ndims>``),
    refined by ``ConstDst`` / ``ConstSrc`` / ``Dynamic`` based on which
    stride set is constexpr. Runtime args are whatever's not in the
    template. ``in_arg``/``out_arg`` override the pointer-variable names."""
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


def _build_copynd_tasklet_in_state(sdfg,
                                   state,
                                   inp_name,
                                   inp,
                                   in_shape_collapsed,
                                   in_strides_collapsed,
                                   out_name,
                                   out,
                                   out_shape_collapsed,
                                   out_strides_collapsed,
                                   *,
                                   name: str,
                                   code_suffix: str = ""):
    """Wire a ``dace::CopyND`` tasklet (``_cpy_in`` → ``_cpy_out``) into
    ``state`` over the collapsed shapes. ``code_suffix`` lets shared-memory
    callers append e.g. ``\\n__syncthreads();``."""
    code = _build_copynd_call(inp.dtype.ctype, in_shape_collapsed, in_strides_collapsed,
                              out_strides_collapsed) + code_suffix
    in_access = state.add_access(inp_name)
    out_access = state.add_access(out_name)
    tasklet = state.add_tasklet(name=name,
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
    return tasklet


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches to the implementation chosen by
    :func:`select_copy_implementation` from endpoint storages, subset
    shapes, and the surrounding scope. Sets ``node.implementation`` to the
    resolved name before delegating so introspection/debug output shows
    what was actually picked."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        impl_name = select_copy_implementation(node, parent_state, parent_sdfg)
        assert impl_name != 'Auto', "select_copy_implementation must not return 'Auto'."
        node.implementation = impl_name
        return CopyLibraryNode.implementations[impl_name].expansion(node, parent_state, parent_sdfg)


@library.expansion
class ExpandMappedTasklet(ExpandTransformation):
    """Mapped element-wise tasklet ``_cpy_out = _cpy_in`` over the collapsed
    copy shape. Schedule is picked from endpoint storages: ``Sequential`` for
    Register / Register<->GPU_Shared (thread-level), ``GPU_Device`` if any
    side is GPU storage, else ``Default``. Raises across the CPU/GPU boundary."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_mapped_tasklet_expansion(node, parent_state, parent_sdfg, allow_cross_storage=True)


@library.expansion
class ExpandCopyNDTemplate(ExpandTransformation):
    """``dace::CopyND<...>`` template tasklet for strided ND copies. Same-side
    only; both endpoints must be C-packed (the template's stride args assume
    row-major contiguous layout)."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ctx = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=True)
        inp, out = ctx.inp, ctx.out

        if _is_cross_cpu_gpu(inp.storage, out.storage):
            raise ValueError("CopyNDTemplate expansion cannot cross the CPU/GPU "
                             f"boundary (got {inp.storage} -> {out.storage}).")

        # CopyND assumes C-packed (row-major contiguous) strides on both
        # endpoints. Reject anything else so callers don't get silent
        # mis-copies from the runtime template's per-dim stride args.
        if not inp.is_packed_c_strides() or not out.is_packed_c_strides():
            raise ValueError(f"CopyNDTemplate expansion requires C-packed strides on both endpoints; "
                             f"got src strides {tuple(inp.strides)} for shape {tuple(inp.shape)} and "
                             f"dst strides {tuple(out.strides)} for shape {tuple(out.shape)}. "
                             f"Use MappedTasklet (same-storage) or convert to a Map via CopyToMap.")

        _build_copynd_tasklet_in_state(ctx.sdfg,
                                       ctx.state,
                                       ctx.inp_name,
                                       inp,
                                       ctx.in_shape_collapsed,
                                       ctx.in_strides_collapsed,
                                       ctx.out_name,
                                       out,
                                       ctx.out_shape_collapsed,
                                       ctx.out_strides_collapsed,
                                       name="copynd_tasklet")

        return ctx.sdfg


@library.expansion
class ExpandMemcpyCUDA1D(ExpandTransformation):
    """One ``cudaMemcpyAsync`` for a contiguous copy. Direction (H2D / D2H /
    D2D / H2H) is inferred from endpoint storages — covers what the legacy
    ``CUDA``, ``CUDAHostToDevice``, and ``CUDADeviceToHost`` impls did separately."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_cuda_memcpy_expansion(node, parent_state, parent_sdfg)


@library.expansion
class ExpandMemcpyCPU(ExpandTransformation):
    """One ``std::memcpy`` for a contiguous CPU<->CPU copy."""
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        ctx = _make_expansion_sdfg(node, parent_state, parent_sdfg, require_contiguous=True)
        inp, out = ctx.inp, ctx.out

        cp_size = reduce(operator.mul, ctx.map_lengths, 1)
        in_access = ctx.state.add_access(ctx.inp_name)
        out_access = ctx.state.add_access(ctx.out_name)

        # CPU<->CPU: both sides are CPU, so single-element gets value-typed.
        in_conn_type, out_conn_type, in_arg, out_arg = _memcpy_connector_typing(inp,
                                                                                out,
                                                                                one_elem=(cp_size == 1),
                                                                                in_is_cpu=True,
                                                                                out_is_cpu=True,
                                                                                in_conn='_cpy_in',
                                                                                out_conn='_cpy_out')

        tasklet = ctx.state.add_tasklet(
            name="memcpy_tasklet",
            inputs={"_cpy_in": in_conn_type},
            outputs={"_cpy_out": out_conn_type},
            code=f"memcpy({out_arg}, {in_arg}, {sym2cpp(cp_size)} * sizeof({inp.dtype.ctype}));",
            language=dace.Language.CPP)

        full_range = dace.subsets.Range([(0, e - 1, 1) for e in ctx.map_lengths])
        ctx.state.add_edge(in_access, None, tasklet, "_cpy_in", dace.memlet.Memlet(data=ctx.inp_name,
                                                                                   subset=full_range))
        ctx.state.add_edge(tasklet, "_cpy_out", out_access, None,
                           dace.memlet.Memlet(data=ctx.out_name, subset=full_range))

        return ctx.sdfg


@library.expansion
class ExpandMemcpyCUDA2D(ExpandTransformation):
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
            raise ValueError("MemcpyCUDA2D requires 1D or 2D collapsed shapes, got "
                             f"{in_shape_collapsed} (src) / {out_shape_collapsed} (dst).")

        kind = _memcpy_kind(inp, out)

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

        has_stream, stream_expr = _stream_expr(stream_input)
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
class ExpandMemcpyCUDANDStrided(ExpandTransformation):
    """ND-strided cross-boundary copy via a Sequential map of ``cudaMemcpyAsync``.

    Fallback for ≥3D-strided patterns that cannot collapse to a single
    ``cudaMemcpyAsync`` / ``cudaMemcpy2DAsync``. Iterates all collapsed
    dimensions except the *chunk axis* (any axis with ``stride == 1`` on
    both sides — innermost for C-packed, outermost for Fortran-packed) and
    emits one ``cudaMemcpyAsync`` per row.

    The inner tasklet's stream connector is named ``_cpy_stream`` (not
    ``stream``) to avoid shadowing the wrapper SDFG's ``stream`` AccessNode/
    array name in the codegen scope, which would otherwise emit
    ``gpuStream_t stream = stream;`` self-init.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ctx = _make_expansion_sdfg(node, parent_state, parent_sdfg, allow_cross_storage=True)
        inp, out = ctx.inp, ctx.out

        if len(ctx.in_shape_collapsed) != len(ctx.out_shape_collapsed):
            raise NotImplementedError("ExpandCUDANDStrided requires src and dst to share the collapsed rank "
                                      f"(got {ctx.in_shape_collapsed} vs {ctx.out_shape_collapsed}).")
        ndims = len(ctx.in_shape_collapsed)
        if ndims < 1:
            raise NotImplementedError("ExpandCUDANDStrided requires at least one collapsed dimension.")

        # Pick the chunk axis: any dim with stride 1 on both sides. Prefer
        # the innermost (C-packed) when multiple match.
        chunk_dim = None
        for d in reversed(range(ndims)):
            if ctx.in_strides_collapsed[d] == 1 and ctx.out_strides_collapsed[d] == 1:
                chunk_dim = d
                break
        if chunk_dim is None:
            raise NotImplementedError(
                "ExpandCUDANDStrided requires at least one common stride-1 axis on both sides "
                f"(got src_strides={ctx.in_strides_collapsed}, dst_strides={ctx.out_strides_collapsed}).")

        ctype = inp.dtype.ctype
        chunk = sym2cpp(ctx.in_shape_collapsed[chunk_dim])

        kind = _memcpy_kind(inp, out)

        # Avoid the connector name `stream` colliding with the wrapper SDFG's
        # `stream` array name in the codegen scope.
        _INNER_STREAM_CONN = "_cpy_stream"
        has_stream, stream_expr = _stream_expr(ctx.stream_input, _INNER_STREAM_CONN)

        # Map over all dims except chunk_dim; the chunk axis is the contiguous
        # run passed to cudaMemcpyAsync. ndims == 1 degenerates to a single
        # contiguous memcpy (no map).
        map_axes = [d for d in range(ndims) if d != chunk_dim]
        map_params = [f"__cpy_i{d}" for d in map_axes]
        map_ranges = {p: f"0:{sym2cpp(ctx.in_shape_collapsed[d])}" for d, p in zip(map_axes, map_params)}

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
            row_in = _row_subset(ctx.in_shape_collapsed)
            row_out = _row_subset(ctx.out_shape_collapsed)
        else:
            row_in = f"0:{sym2cpp(ctx.in_shape_collapsed[0])}"
            row_out = f"0:{sym2cpp(ctx.out_shape_collapsed[0])}"

        in_memlet = dace.memlet.Memlet(data=ctx.inp_name, subset=row_in)
        out_memlet = dace.memlet.Memlet(data=ctx.out_name, subset=row_out)

        code = f"DACE_GPU_CHECK(cudaMemcpyAsync(_cpy_out, _cpy_in, {chunk} * sizeof({ctype}), {kind}, {stream_expr}));"

        in_conn_type = dace.dtypes.pointer(inp.dtype)
        out_conn_type = dace.dtypes.pointer(out.dtype)

        if ndims == 1:
            in_access = ctx.state.add_access(ctx.inp_name)
            out_access = ctx.state.add_access(ctx.out_name)
            tasklet_in_conns = {"_cpy_in": in_conn_type}
            if has_stream:
                tasklet_in_conns[_INNER_STREAM_CONN] = dace.dtypes.gpuStream_t
            tasklet = ctx.state.add_tasklet(name=f"{node.label}_memcpy",
                                            inputs=tasklet_in_conns,
                                            outputs={"_cpy_out": out_conn_type},
                                            code=code,
                                            language=dace.Language.CPP)
            ctx.state.add_edge(in_access, None, tasklet, "_cpy_in", in_memlet)
            ctx.state.add_edge(tasklet, "_cpy_out", out_access, None, out_memlet)
            if has_stream:
                stream_access = ctx.state.add_access(_STREAM_CONN)
                ctx.state.add_edge(stream_access, None, tasklet, _INNER_STREAM_CONN,
                                   dace.memlet.Memlet.from_array(_STREAM_CONN, ctx.sdfg.arrays[_STREAM_CONN]))
        else:
            inner_tasklet, map_entry, _map_exit = ctx.state.add_mapped_tasklet(
                name=f"{node.label}_tasklet",
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
                # Wrapper SDFG ``stream`` access threads through MapEntry's
                # IN_stream / OUT_stream pass-through into the inner Tasklet.
                inner_tasklet.add_in_connector(_INNER_STREAM_CONN, dace.dtypes.gpuStream_t)
                _wire_stream_through_map(ctx.sdfg, ctx.state, map_entry, inner_tasklet, _INNER_STREAM_CONN)

        return ctx.sdfg


@library.expansion
class ExpandTasklet(ExpandTransformation):
    """Single-element same-storage scalar copy: a Python tasklet doing
    ``_out = _in`` directly. No map, no SDFG wrapper, no pointer cast.

    Both subsets must be volume 1 and storages must match (no CPU/GPU
    boundary). For multi-element copies use ``MappedTasklet`` (mapped
    element-wise) or ``CopyNDTemplate`` (strided)."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, _dyn, _stream = node.validate(parent_sdfg,
                                                                                           parent_state,
                                                                                           allow_cross_storage=False)

        in_volume = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in in_subset], 1)
        out_volume = reduce(operator.mul, [(e + 1 - b) // s for (b, e, s) in out_subset], 1)
        if in_volume != 1 or out_volume != 1:
            raise ValueError(f"Tasklet expansion requires single-element subsets "
                             f"(got input volume {in_volume}, output volume {out_volume}). "
                             f"Use MappedTasklet for element-wise multi-element copies, "
                             f"or CopyNDTemplate for strided ones.")

        return nodes.Tasklet(node.name,
                             inputs={"_in": inp.dtype},
                             outputs={"_out": out.dtype},
                             code="_out = _in",
                             language=dace.Language.Python)


@library.expansion
class ExpandSharedMemoryCollective(ExpandTransformation):
    """Block-collective shared-memory copy: ``dace::CopyND`` followed by
    ``__syncthreads()``. Caller is responsible for placing this *outside* any
    enclosing ``GPU_ThreadBlock`` map — this expansion *is* the thread-block-
    level operation."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, _ = node.validate(parent_sdfg,
                                                                                               parent_state,
                                                                                               allow_cross_storage=True)

        valid_storages = {dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global, dtypes.StorageType.Register}

        if inp.storage not in valid_storages:
            raise ValueError(f"SharedMemoryCollective: input storage {inp.storage} is not "
                             "GPU_Shared, GPU_Global, or Register.")
        if out.storage not in valid_storages:
            raise ValueError(f"SharedMemoryCollective: output storage {out.storage} is not "
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

        # CopyND moves the data; ``__syncthreads()`` afterwards so all threads see the update.
        # TODO: replace with a cooperative copy distributing work across threads, once DaCe's
        # GPU scheduling supports bare thread-block-level tasklets.
        _build_copynd_tasklet_in_state(sdfg,
                                       state,
                                       inp_name,
                                       inp,
                                       in_shape_collapsed,
                                       in_strides_collapsed,
                                       out_name,
                                       out,
                                       out_shape_collapsed,
                                       out_strides_collapsed,
                                       name="shared_copy",
                                       code_suffix="\n__syncthreads();")

        return sdfg


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    """Library node representing a data copy between two access nodes.

    Names follow one rule: each describes the C++ shape the expansion emits.

    ============================================ ================================
    Implementation                               C++ shape
    ============================================ ================================
    ``MappedTasklet``                            Mapped element-wise tasklet;
                                                 schedule from storages
                                                 (Sequential / GPU_Device / Default)
    ``Tasklet``                                  Bare assignment tasklet, no map
    ``CopyNDTemplate``                           ``dace::CopyND`` template call
    ``MemcpyCPU``                                ``std::memcpy``
    ``MemcpyCUDA1D``                             one ``cudaMemcpyAsync``
                                                 (direction inferred from storages)
    ``MemcpyCUDA2D``                             one ``cudaMemcpy2DAsync``
    ``MemcpyCUDANDStrided``                      Sequential map of
                                                 ``cudaMemcpyAsync``
    ``SharedMemoryCollective``                   ``dace::CopyND`` + ``__syncthreads()``
                                                 (block-collective)
    ============================================ ================================
    """

    implementations = {
        "Auto": ExpandAuto,
        "MappedTasklet": ExpandMappedTasklet,
        "Tasklet": ExpandTasklet,
        "CopyNDTemplate": ExpandCopyNDTemplate,
        "MemcpyCPU": ExpandMemcpyCPU,
        "MemcpyCUDA1D": ExpandMemcpyCUDA1D,
        "MemcpyCUDA2D": ExpandMemcpyCUDA2D,
        "MemcpyCUDANDStrided": ExpandMemcpyCUDANDStrided,
        "SharedMemoryCollective": ExpandSharedMemoryCollective,
    }
    default_implementation = 'Auto'

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

    def validate(self, sdfg, state, allow_cross_storage=True):
        """Resolve in/out edges, names, subsets, dynamic inputs, and an
        optional stream descriptor. Raises if the libnode is not wired
        with exactly one ``_in`` and one ``_out`` data edge, dtypes mismatch,
        or (when ``allow_cross_storage`` is False) the two storages differ."""
        out_edges = [oe for oe in state.out_edges(self) if oe.src_conn == "_out"]
        if len(out_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one `_out` output edge.")
        oe = out_edges[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        stream_input, dynamic_inputs = extract_stream_and_dynamic_inputs(self, sdfg, state, reserved_conns=("_in", ))

        in_edges = [ie for ie in state.in_edges(self) if ie.dst_conn == "_in"]
        if len(in_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one data input edge "
                             "connected to the `_in` connector.")
        ie = in_edges[0]
        inp = sdfg.arrays[ie.data.data]
        in_subset = ie.data.subset
        inp_name = ie.dst_conn

        if inp.dtype != out.dtype:
            raise ValueError(f"Input and output data types must match (got {inp.dtype} vs {out.dtype}).")

        if not allow_cross_storage and inp.storage != out.storage:
            raise ValueError(f"Input and output storage types must match for this expansion "
                             f"(got {inp.storage} vs {out.storage}). Use a cross-storage "
                             f"expansion or the pure fallback.")

        return inp_name, inp, in_subset, out_name, out, out_subset, dynamic_inputs, stream_input
