# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" ``CopyLibraryNode`` representing copies explicitly. """
from dataclasses import dataclass
from typing import List, Optional

import dace
from dace import data, library, nodes, dtypes, symbolic
from dace.codegen.common import sym2cpp, get_gpu_backend
from dace.libraries.standard.helper import CURRENT_STREAM_NAME, auto_dispatch, collapse_shape_and_strides
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dataclass
class CopyExpansion:
    """Inputs + collapsed-shape state shared across :class:`CopyLibraryNode`
    expansions that build a wrapper SDFG. Returned by :func:`_make_expansion_sdfg`."""
    sdfg: dace.SDFG
    state: dace.SDFGState
    inp_name: str
    inp: data.Data
    in_subset: dace.subsets.Range
    out_name: str
    out: data.Data
    out_subset: dace.subsets.Range
    map_lengths: List[symbolic.SymExpr]
    in_shape_collapsed: List[symbolic.SymExpr]
    out_shape_collapsed: List[symbolic.SymExpr]


def _is_cross_cpu_gpu(src_storage: dtypes.StorageType, dst_storage: dtypes.StorageType, copy_node: "CopyLibraryNode",
                      parent_state: dace.SDFGState) -> bool:
    """Return True if src and dst crosses the CPU/GPU boundary. ``Register``
    depends on the scope, within GPU scope we assume it is in GPU, and in CPU scope we assume it is in CPU."""
    in_gpu = is_devicelevel_gpu(parent_state.sdfg, parent_state, copy_node)

    # A storage is GPU-resident if it's explicitly a GPU storage, or a Register inside a GPU scope
    src_gpu = (src_storage in dtypes.GPU_RESIDENT_STORAGES) or (src_storage == dtypes.StorageType.Register and in_gpu)
    dst_gpu = (dst_storage in dtypes.GPU_RESIDENT_STORAGES) or (dst_storage == dtypes.StorageType.Register and in_gpu)

    # A storage is CPU-resident if it's explicitly a CPU storage, or a Register outside a GPU scope
    src_cpu = (src_storage in dtypes.CPU_RESIDENT_STORAGES) or (src_storage == dtypes.StorageType.Register
                                                                and not in_gpu)
    dst_cpu = (dst_storage in dtypes.CPU_RESIDENT_STORAGES) or (dst_storage == dtypes.StorageType.Register
                                                                and not in_gpu)

    return (src_cpu and dst_gpu) or (src_gpu and dst_cpu)


def _both_packed_same_layout(inp: data.Data, out: data.Data) -> bool:
    """True if both descriptors are packed in the same major order (both C
    or both Fortran)."""
    return ((inp.is_packed_c_strides() and out.is_packed_c_strides())
            or (inp.is_packed_fortran_strides() and out.is_packed_fortran_strides()))


def _delinearized_index(b_i: symbolic.symbol, shape: List[symbolic.SymExpr], layout: str) -> List[symbolic.SymExpr]:
    """Multi-dim index expressions for a 1-D walker into a packed-layout array.
    Only C-style (packed row-major) and Fortran-style (packed column-major) layouts are supported.

    :param b_i: the 1-D map symbol.
    :param shape: per-dim extents in descriptor order.
    :param layout: ``'C'`` (stride-1 is the last dim) or ``'F'`` (stride-1 is the first dim).
    :returns: list of per-dim symbolic index expressions, in descriptor order.
    """
    cum_strides = []
    cum = 1
    iter_shape = reversed(shape) if layout == 'C' else iter(shape)
    for s in iter_shape:
        cum_strides.append(cum)
        cum *= s
    if layout == 'C':
        cum_strides.reverse()
    return [symbolic.int_floor(b_i, cum_strides[d]) % shape[d] for d in range(len(shape))]


def select_copy_implementation(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> str:
    """Resolve ``CopyLibraryNode.implementation`` when set to ``'Auto'`` (the default).

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: a concrete implementation name from
              ``CopyLibraryNode.implementations`` -- never ``'Auto'`` itself.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=True)

    # Invariant: single-element copies never route to ``MappedTasklet``
    # (its 0-D map crashes in memlet propagation). Steps 1 and 2 handle
    # the single-element case explicitly.
    single_elt = (in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1)

    # 1. GPU_Shared involvement. Block-cooperative ``SharedMemoryCollective``
    # (``dace::CopyND<>`` + ``__syncthreads()``) unless the copy is
    # thread-level -- either a Register endpoint or placed inside a
    # ``GPU_ThreadBlock`` map -- in which case it routes per-thread.
    # TODO, FUTURE WORK: replace ``dace::CopyND`` with a vectorized 128-bit
    # collective load.
    if inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared:
        thread_level = (inp.storage == dtypes.StorageType.Register or out.storage == dtypes.StorageType.Register
                        or is_in_scope(parent_state.sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock]))
        if thread_level:
            return 'Tasklet' if single_elt else 'MappedTasklet'
        return 'SharedMemoryCollective'

    # 2. Single-element non-Shared copies. Bare ``Tasklet`` or ``MemcpyCUDA1D``.
    #
    #   endpoints              in kernel  impl          why
    #   ---------------------  ---------  ------------  ------------------------
    #   cross CPU/GPU          any        MemcpyCUDA1D  cudaMemcpyAsync
    #   same side, GPU<->GPU   yes        Tasklet       device-side _out = _in
    #   same side, GPU<->GPU   no         MemcpyCUDA1D  D2D; host cannot deref
    #                                                   device pointers
    #   same side, has host    any        Tasklet       host runs the assignment
    if single_elt:
        if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
            return 'MemcpyCUDA1D'
        inside_kernel = is_devicelevel_gpu(parent_state.sdfg, parent_state, node)
        both_gpu_global = (inp.storage == dtypes.StorageType.GPU_Global
                           and out.storage == dtypes.StorageType.GPU_Global)
        if both_gpu_global and not inside_kernel:
            return 'MemcpyCUDA1D'
        return 'Tasklet'

    # 3. Multi-element in-device-scope: ``cudaMemcpyAsync`` cannot be issued
    # from device code, so emit a map inside the existing kernel scope.
    if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        return 'MappedTasklet'

    # 4. Coarse pick by storage pair: any copy touching GPU memory goes
    # through the cudaMemcpy family; everything else falls through to
    # MappedTasklet at the end.
    gpu = dtypes.StorageType.GPU_Global
    allowed = dtypes.CPU_RESIDENT_STORAGES | {dtypes.StorageType.Default, gpu}
    impl = ('MemcpyCUDA1D' if ((inp.storage == gpu or out.storage == gpu) and inp.storage in allowed
                               and out.storage in allowed) else None)

    # 5. Refine for subset patterns (CUDA2D / CUDANDStrided / fall back to
    # MappedTasklet for unsupported stride mixs).
    if impl == 'MemcpyCUDA1D':
        refined = _refine_cuda_impl_for_subsets(node, parent_state)
        if refined is not None:
            impl = refined

    # Rank-mismatched copies (e.g. ``(2,3,4) -> (8,3)``) fall through to
    # MappedTasklet, whose expansion handles the collapse with a 1-D walker
    # and per-side ``int_floor``/``%`` delinearization -- supported only when
    # both endpoints are packed-same-layout with contiguous subsets; rejected
    # otherwise with a specific error message.
    return impl or 'MappedTasklet'


def _refine_cuda_impl_for_subsets(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> Optional[str]:
    """Upgrade ``MemcpyCUDA1D`` to a more specific impl for non-contiguous subsets.

      condition                                            impl
      ---------------------------------------------------  --------------------
      both subsets are contiguous                          ``None`` (keep CUDA1D)
      collapsed rank == 2 and 2D pitched layout matches    ``MemcpyCUDA2D``
      collapsed rank == 1 (both sides equal length)        ``MemcpyCUDA2D``    (degenerate ``(1, N)`` form)
      same-side (no CPU/GPU boundary)                      ``MappedTasklet``   (per-element loop nest handles arbitrary strides)
      cross CPU/GPU, same rank, common stride-1 axis       ``MemcpyCUDANDStrided`` (Sequential map of ``cudaMemcpyAsync`` over outer dims, one stride-1 chunk per iteration)
      cross CPU/GPU, no common stride-1 axis               raise -- no ``cudaMemcpy*`` lowering exists for this pattern

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: the refined implementation name, or ``None`` when both subsets
              are contiguous (caller keeps ``MemcpyCUDA1D``).
    :raises ValueError: a cross-CPU/GPU strided pattern with no common stride-1
        axis -- the host cannot issue ``cudaMemcpyAsync`` for non-contiguous
        regions and device code cannot issue ``cudaMemcpyAsync`` at all.
    """
    _, inp, in_subset, _, out, out_subset = node.validate(parent_state.sdfg, parent_state, allow_cross_storage=True)

    if in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out):
        return None

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    # ``cudaMemcpy2D``. A 2D pattern is supported when
    # either dim has stride 1 on both sides, or the outer/inner stride ratio equals the inner width.
    src_rank, dst_rank = len(in_shape_collapsed), len(out_shape_collapsed)
    cuda2d_2d = False
    if src_rank == 2 and dst_rank == 2:
        s0, s1 = in_strides_collapsed
        d0, d1 = out_strides_collapsed
        w = in_shape_collapsed[1]
        if (s0 == 1 and d0 == 1) or (s1 == 1 and d1 == 1):
            cuda2d_2d = True
        else:
            try:
                # ``inequal_symbols`` normalizes same-named symbols across both sides
                # (e.g. ``N`` declared once with ``positive=True`` and once without),
                # so the ratio check isn't defeated by sympy-assumption identity drift.
                cuda2d_2d = (not symbolic.inequal_symbols(s0 / s1, w) and not symbolic.inequal_symbols(d0 / d1, w))
            except (TypeError, ZeroDivisionError):
                pass
    cuda2d_1d = (src_rank == 1 and dst_rank == 1
                 and not symbolic.inequal_symbols(in_shape_collapsed[0], out_shape_collapsed[0]))
    if cuda2d_2d or cuda2d_1d:
        return 'MemcpyCUDA2D'

    # Same-side strided ND -- MappedTasklet.
    if not _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
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


def _make_expansion_sdfg(node: "CopyLibraryNode",
                         parent_state: dace.SDFGState,
                         allow_cross_storage: bool = False) -> CopyExpansion:
    """Shared validation + wrapper-SDFG skeleton for expansions.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :param allow_cross_storage: permit differing src/dst storages.
    :returns: a :class:`CopyExpansion` with the skeleton SDFG and collapsed
              shape/stride state.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=allow_cross_storage)

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    # When the experimental GPU codegen has already wired the ambient stream onto this
    # libnode (in-connector ``__dace_current_stream`` typed ``gpuStream_t``), the resulting
    # NestedSDFG inherits that outer connector, so the inner SDFG needs a matching
    # descriptor or NestedSDFG.validate() rejects it. The legacy codegen never adds the
    # connector, so this branch is a no-op there.
    if CURRENT_STREAM_NAME in node.in_connectors:
        sdfg.add_scalar(CURRENT_STREAM_NAME, dtypes.gpuStream_t, transient=False)

    state = sdfg.add_state(f"{node.label}_state", is_start_block=True)
    map_lengths = [s for s in in_subset.size() if s != 1]

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
                         out_shape_collapsed=out_shape_collapsed)


def _make_mapped_tasklet_expansion(node: "CopyLibraryNode",
                                   parent_state: dace.SDFGState,
                                   allow_cross_storage: bool = False) -> dace.SDFG:
    """Element-wise mapped tasklet expansion.

    Schedule comes from the storages:
    ``Sequential`` for Register/Register
    or Register<->GPU_Shared (thread-level) and for any in-kernel copy
    ``GPU_Device`` if any side is GPU storage and
    we're at host level, else ``Default`` (CPU<->CPU -- inferred
    post-expansion).

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :param allow_cross_storage: permit differing src/dst storages.
    :returns: the wrapper SDFG holding the mapped tasklet.
    :raises ValueError: the copy crosses the CPU/GPU boundary.
    """
    ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=allow_cross_storage)
    inp, out = ctx.inp, ctx.out

    if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
        raise ValueError("MappedTasklet expansion cannot cross the CPU/GPU boundary "
                         f"(got {inp.storage} -> {out.storage}). Use a MemcpyCUDA1D variant.")

    # Schedule from storages and surrounding scope.
    is_register = lambda s: s == dtypes.StorageType.Register
    is_thread_local = (is_register(inp.storage) and is_register(out.storage)) or (
        (is_register(inp.storage) and out.storage == dtypes.StorageType.GPU_Shared) or
        (is_register(out.storage) and inp.storage == dtypes.StorageType.GPU_Shared))
    in_kernel = is_devicelevel_gpu(parent_state.sdfg, parent_state, node)
    if is_thread_local or in_kernel:
        schedule = dtypes.ScheduleType.Sequential
    elif inp.storage in dtypes.GPU_RESIDENT_STORAGES or out.storage in dtypes.GPU_RESIDENT_STORAGES:
        schedule = dtypes.ScheduleType.GPU_Device
    else:
        schedule = dtypes.ScheduleType.Default

    ctx.sdfg.schedule = dtypes.ScheduleType.Default

    # Inner-tasklet connectors. Must not collide with the wrapper SDFG's
    # parameter arrays, which are named after the libnode's outer connectors.
    inner_in, inner_out = "_in", "_out"
    in_shape, out_shape = ctx.in_shape_collapsed, ctx.out_shape_collapsed

    if len(in_shape) == len(out_shape):
        # Same-rank: per-dim map params, shared access expression on both sides.
        # Per-dim shapes must match; otherwise the shared index expression walks past
        # the smaller side (transposes / permutations belong to a Transpose libnode;
        # reshapes go through the rank-mismatch branch). ``inequal_symbols`` normalizes
        # same-named SymPy symbols with different assumption sets (e.g. ``Symbol('N',
        # integer=True)`` vs ``Symbol('N', integer=True, positive=True)``) before
        # comparing, so a shape mismatch is real and not a symbol-identity artifact.
        if any(symbolic.inequal_symbols(a, b) for a, b in zip(in_shape, out_shape)):
            raise ValueError(f"MappedTasklet same-rank copy requires matching per-dim shapes; got src "
                             f"{tuple(in_shape)} vs dst {tuple(out_shape)}. Per-dim permutations are not "
                             f"supported -- use a Transpose libnode. Reshapes must change rank.")
        map_params = [f"__i{i}" for i in range(len(ctx.map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, ctx.map_lengths)}
        access_expr = ','.join(map_params)
        inputs = {inner_in: dace.memlet.Memlet(f"{ctx.inp_name}[{access_expr}]")}
        outputs = {inner_out: dace.memlet.Memlet(f"{ctx.out_name}[{access_expr}]")}
    else:
        # Rank-mismatch reshape: 1-D walker + per-side delinearization. Supported
        # only when both endpoints satisfy the collapsing rules:
        #   1. Same packed major order (both C-contiguous or both Fortran).
        #   2. Both subsets contiguous in their parent arrays.
        # The walker iterates the total element count; the per-side delinearization
        # (``_delinearized_index``) maps the walker into the multi-dim index using
        # the shared layout. Mixed C/F is a transpose-reshape; non-packed or
        # non-contiguous endpoints have no unambiguous flat order.
        if not _both_packed_same_layout(inp, out):
            raise ValueError(
                f"MappedTasklet rank-mismatched copy ({tuple(in_shape)} -> {tuple(out_shape)}) requires "
                f"both endpoints to be packed in the same major order (both C-contiguous or both "
                f"Fortran-contiguous). Got src '{ctx.inp_name}' strides {tuple(inp.strides)} on shape "
                f"{tuple(inp.shape)} and dst '{ctx.out_name}' strides {tuple(out.strides)} on shape "
                f"{tuple(out.shape)}. Mixed layouts are transposes -- use a same-rank Tasklet copy instead.")
        in_contig = ctx.in_subset.is_contiguous_subset(inp)
        out_contig = ctx.out_subset.is_contiguous_subset(out)
        if not (in_contig and out_contig):
            raise ValueError(
                f"MappedTasklet rank-mismatched copy ({tuple(in_shape)} -> {tuple(out_shape)}) requires "
                f"contiguous subsets on both endpoints (the 1-D walker treats the data as a flat sequence). "
                f"Got src subset {ctx.in_subset} (contiguous: {in_contig}) on shape {tuple(inp.shape)} and "
                f"dst subset {ctx.out_subset} (contiguous: {out_contig}) on shape {tuple(out.shape)}.")
        layout = 'C' if inp.is_packed_c_strides() else 'F'

        total = ctx.in_subset.num_elements_exact()
        b_i_name = "__b_i"
        b_i = symbolic.symbol(b_i_name)
        map_rng = {b_i_name: f"0:{sym2cpp(total)}"}

        def _side_access(arr_name, shape):
            if len(shape) == 1:
                return f"{arr_name}[{b_i_name}]"
            idx = _delinearized_index(b_i, shape, layout)
            return f"{arr_name}[{','.join(sym2cpp(e) for e in idx)}]"

        inputs = {inner_in: dace.memlet.Memlet(_side_access(ctx.inp_name, in_shape))}
        outputs = {inner_out: dace.memlet.Memlet(_side_access(ctx.out_name, out_shape))}

    _, map_entry, _ = ctx.state.add_mapped_tasklet(f"{node.label}_tasklet",
                                                   map_rng,
                                                   inputs,
                                                   f"{inner_out} = {inner_in}",
                                                   outputs,
                                                   schedule=schedule,
                                                   external_edges=True)

    return ctx.sdfg


def _memcpy_kind(inp: data.Data, out: data.Data) -> str:
    """``cudaMemcpy<src>To<dst>`` from endpoint storages."""
    src_loc = "Device" if inp.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    dst_loc = "Device" if out.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    backend = get_gpu_backend()
    return f"{backend}Memcpy{src_loc}To{dst_loc}"


def _make_memcpy_tasklet(node: "CopyLibraryNode", parent_state: dace.SDFGState, *, cuda: bool) -> nodes.Tasklet:
    """Build a Tasklet emitting one contiguous-block copy.

    Emits ``cudaMemcpyAsync`` when ``cuda`` is set -- cross-CPU/GPU is allowed and
    the direction (HostToDevice / DeviceToHost / DeviceToDevice / HostToHost) is
    inferred from endpoint storages -- otherwise a same-storage ``std::memcpy``.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :param cuda: emit ``cudaMemcpyAsync`` (else ``memcpy``).
    :returns: a :class:`~dace.sdfg.nodes.Tasklet` issuing the copy.
    :raises ValueError: a subset is non-contiguous; the single-call copy form
        would overrun the region. Use ``MappedTasklet`` for strided subsets.
    """
    label = "MemcpyCUDA1D" if cuda else "MemcpyCPU"
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=cuda)
    single_elt = (in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1)
    if single_elt:
        # For a single element we must/can ignore the strides.
        pass
    elif not (in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out)):
        raise ValueError(f"{label} requires contiguous subsets; got src '{inp_name}' subset {in_subset} "
                         f"(shape {inp.shape} strides {inp.strides}) and dst '{out_name}' subset {out_subset} "
                         f"(shape {out.shape} strides {out.strides}). Use MappedTasklet for strided subsets.")

    in_conn = CopyLibraryNode.INPUT_CONNECTOR_NAME
    out_conn = CopyLibraryNode.OUTPUT_CONNECTOR_NAME
    nbytes = f"{sym2cpp(in_subset.num_elements_exact())} * sizeof({inp.dtype.ctype})"
    if cuda:
        backend = get_gpu_backend()
        code = f"{backend}MemcpyAsync({out_conn}, {in_conn}, {nbytes}, {_memcpy_kind(inp, out)}, {CURRENT_STREAM_NAME});"
    else:
        code = f"memcpy({out_conn}, {in_conn}, {nbytes});"

    return nodes.Tasklet(node.name,
                         inputs={in_conn: dace.dtypes.pointer(inp.dtype)},
                         outputs={out_conn: dace.dtypes.pointer(out.dtype)},
                         code=code,
                         language=dace.Language.CPP)


def _build_shmem_collective_copy_code(inp: data.Data, in_subset: dace.subsets.Range, out: data.Data,
                                      out_subset: dace.subsets.Range) -> str:
    """Build the C++ code for ``ExpandSharedMemoryCollective``: a
    ``dace::CopyND<...>::Copy(...)`` call followed by ``__syncthreads()``.

    Picks the most-specific static template form: ``CopyND<T, 1, false,
    dims...>`` for static shapes (else ``CopyNDDynamic<T, 1, false, ndims>``),
    refined by ``ConstDst`` / ``ConstSrc`` / ``Dynamic`` based on which stride
    set is constexpr; runtime args are whatever's not in the template.

    :param inp: source descriptor (provides ``ctype`` and ``strides``).
    :param in_subset: source memlet subset.
    :param out: destination descriptor (provides ``strides``).
    :param out_subset: destination memlet subset.
    :returns: full code: ``...::Copy(...);\\n__syncthreads();``.
    """
    copy_shape, src_strides = collapse_shape_and_strides(in_subset, inp.strides)
    _, dst_strides = collapse_shape_and_strides(out_subset, out.strides)
    ndims = len(copy_shape)
    shape_strs = [sym2cpp(s) for s in copy_shape]
    src_stride_strs = [sym2cpp(s) for s in src_strides]
    dst_stride_strs = [sym2cpp(s) for s in dst_strides]

    dims_static = not any(symbolic.issymbolic(s) for s in copy_shape)
    src_static = not any(symbolic.issymbolic(s) for s in src_strides)
    dst_static = not any(symbolic.issymbolic(s) for s in dst_strides)

    ctype = inp.dtype.ctype
    if dims_static:
        copy_tmpl = f"dace::CopyND<{ctype}, 1, false, {', '.join(shape_strs)}>"
    else:
        copy_tmpl = f"dace::CopyNDDynamic<{ctype}, 1, false, {ndims}>"

    # Prefer ConstDst when dst is static; else ConstSrc; else fully dynamic.
    # The chosen template fixes one stride set; the rest plus the (possibly
    # symbolic) shape are passed as runtime args, in per-dim order.
    if dst_static:
        shape_tmpl = f"template ConstDst<{', '.join(dst_stride_strs)}>"
    elif src_static:
        shape_tmpl = f"template ConstSrc<{', '.join(src_stride_strs)}>"
    else:
        shape_tmpl = "Dynamic"

    stride_args = []
    for d in range(ndims):
        if not dims_static:
            stride_args.append(shape_strs[d])
        if not src_static or dst_static:
            stride_args.append(src_stride_strs[d])
        if not dst_static:
            stride_args.append(dst_stride_strs[d])

    all_args = [CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME] + stride_args
    return f"{copy_tmpl}::{shape_tmpl}::Copy({', '.join(all_args)});\n__syncthreads();"


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches to the implementation chosen by
    :func:`select_copy_implementation` from endpoint storages, subset shapes,
    and the surrounding scope."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return auto_dispatch(node, parent_state, select_copy_implementation, CopyLibraryNode)


@library.expansion
class ExpandMappedTasklet(ExpandTransformation):
    """Mapped element-wise tasklet ``_cpy_out = _cpy_in`` over the collapsed
    copy shape. Schedule is picked from endpoint storages: ``Sequential`` for
    Register / Register<->GPU_Shared (thread-level), ``GPU_Device`` if any
    side is GPU storage, else ``Default``. Raises across the CPU/GPU boundary."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_mapped_tasklet_expansion(node, parent_state, allow_cross_storage=True)


@library.expansion
class ExpandMemcpyCUDA1D(ExpandTransformation):
    """One ``cudaMemcpyAsync`` for a contiguous copy. Direction (H2D / D2H /
    D2D / H2H) is inferred from endpoint storages."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_memcpy_tasklet(node, parent_state, cuda=True)


@library.expansion
class ExpandMemcpyCPU(ExpandTransformation):
    """One ``std::memcpy`` for a contiguous CPU<->CPU copy."""
    environments = [environments.CPU]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _make_memcpy_tasklet(node, parent_state, cuda=False)


@library.expansion
class ExpandMemcpyCUDA2D(ExpandTransformation):
    """2D strided copy via ``cudaMemcpy2DAsync`` between any combination of GPU_Global and host storage.

    Handles three stride patterns: row-major contiguous rows, column-major contiguous columns,
    and the degenerate case where the outer stride is a multiple of the inner.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)

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
        backend = get_gpu_backend()

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
        elif (not symbolic.inequal_symbols(src_strides[0] / src_strides[1], copy_shape[1])
              and not symbolic.inequal_symbols(dst_strides[0] / dst_strides[1], copy_shape[1])):
            dpitch = f"{sym2cpp(dst_strides[1])} * sizeof({ctype})"
            spitch = f"{sym2cpp(src_strides[1])} * sizeof({ctype})"
            width = f"sizeof({ctype})"
            height = sym2cpp(copy_shape[0] * copy_shape[1])
        else:
            raise NotImplementedError(f"Unsupported 2D memory copy: shape={copy_shape}, "
                                      f"src_strides={src_strides}, dst_strides={dst_strides}.")

        code = (
            f"{backend}Memcpy2DAsync({CopyLibraryNode.OUTPUT_CONNECTOR_NAME}, {dpitch}, {CopyLibraryNode.INPUT_CONNECTOR_NAME}, {spitch}, "
            f"{width}, {height}, {kind}, {CURRENT_STREAM_NAME});")

        in_conns = {CopyLibraryNode.INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)}
        tasklet = nodes.Tasklet(node.name,
                                inputs=in_conns,
                                outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                                code=code,
                                language=dace.Language.CPP)
        return tasklet


@library.expansion
class ExpandMemcpyCUDANDStrided(ExpandTransformation):
    """ND-strided cross-boundary copy: a Sequential map of ``cudaMemcpyAsync``.

    Fallback for >=3D-strided patterns that cannot collapse to one
    ``cudaMemcpyAsync`` / ``cudaMemcpy2DAsync``. Emits one
    ``cudaMemcpyAsync`` per row, iterating every collapsed dimension except
    the chunk axis (``stride == 1`` both sides). ``ndims == 1`` degenerates
    to a flat single-tasklet expansion; ``ndims > 1`` wraps the per-row
    ``cudaMemcpyAsync`` in a Sequential-map tasklet inside a wrapper SDFG.
    Both reference ``__dace_current_stream``, bound post-expansion.
    """
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)
        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

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
        kind = _memcpy_kind(inp, out)
        backend = get_gpu_backend()

        if ndims == 1:
            # Degenerate case: a single contiguous run. Emit a flat Tasklet
            # with the libnode's connector naming directly -- no wrapper SDFG.
            code = (
                f"DACE_GPU_CHECK({backend}MemcpyAsync({CopyLibraryNode.OUTPUT_CONNECTOR_NAME}, {CopyLibraryNode.INPUT_CONNECTOR_NAME}, "
                f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));")
            in_conns = {CopyLibraryNode.INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)}
            return nodes.Tasklet(node.name,
                                 inputs=in_conns,
                                 outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                                 code=code,
                                 language=dace.Language.CPP)

        # ndims > 1: Sequential map over all non-chunk dims, one
        # cudaMemcpyAsync per row, inside a wrapper SDFG.
        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)

        # Avoid the connector name ``stream`` colliding with the wrapper SDFG's
        # ``stream`` array name in the codegen scope.
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

        in_memlet = dace.memlet.Memlet(data=ctx.inp_name, subset=_row_subset(ctx.in_shape_collapsed))
        out_memlet = dace.memlet.Memlet(data=ctx.out_name, subset=_row_subset(ctx.out_shape_collapsed))
        # Inner-tasklet connectors. Must not collide with the wrapper SDFG's
        # parameter arrays, which are named after the libnode's outer connectors.
        inner_in, inner_out = "_in", "_out"
        backend = get_gpu_backend()
        code = (f"DACE_GPU_CHECK({backend}MemcpyAsync({inner_out}, {inner_in}, "
                f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));")

        inner_tasklet, map_entry, _map_exit = ctx.state.add_mapped_tasklet(name=f"{node.label}_tasklet",
                                                                           map_ranges=map_ranges,
                                                                           inputs={inner_in: in_memlet},
                                                                           code=code,
                                                                           outputs={inner_out: out_memlet},
                                                                           schedule=dace.dtypes.ScheduleType.Sequential,
                                                                           language=dace.Language.CPP,
                                                                           external_edges=True)
        # Force pointer connectors on the inner tasklet so the codegen types
        # them as ``T*`` (matching cudaMemcpyAsync's signature) instead of
        # dereferencing them as values.
        inner_tasklet.in_connectors[inner_in] = dace.dtypes.pointer(inp.dtype)
        inner_tasklet.out_connectors[inner_out] = dace.dtypes.pointer(out.dtype)

        return ctx.sdfg


@library.expansion
class ExpandTasklet(ExpandTransformation):
    """Single-element same-side scalar copy: ``_cpy_out = _cpy_in`` as a Python tasklet"""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)
        in_volume = in_subset.num_elements_exact()
        out_volume = out_subset.num_elements_exact()
        if in_volume != 1 or out_volume != 1:
            raise ValueError(f"Tasklet expansion requires single-element subsets "
                             f"(got input volume {in_volume}, output volume {out_volume}). "
                             f"Use MappedTasklet for multi-element copies.")
        # Single-element Shared involvement is a valid thread-level
        # assignment; the auto dispatcher routes it here when the copy is
        # inside a thread-block scope.
        if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
            raise ValueError(f"Tasklet expansion: storage types must match (no CPU/GPU boundary); "
                             f"got {inp.storage} -> {out.storage}. Use a MemcpyCUDA1D variant instead.")

        return nodes.Tasklet(node.name,
                             inputs={CopyLibraryNode.INPUT_CONNECTOR_NAME: inp.dtype},
                             outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: out.dtype},
                             code=f"{CopyLibraryNode.OUTPUT_CONNECTOR_NAME} = {CopyLibraryNode.INPUT_CONNECTOR_NAME}",
                             language=dace.Language.Python)


@library.expansion
class ExpandSharedMemoryCollective(ExpandTransformation):
    """Block-collective Shared <-> Shared/Global copy: a single Tasklet
    emitting ``dace::CopyND<...>::Copy + __syncthreads()`` with
    ``_in``/``_out`` connectors matching the libnode's connectors directly
    (no NSDFG wrapper -- the parent kernel's ``__shared__`` array binds
    straight to ``_in``/``_out`` without scope-id name mangling).

    Caller is responsible for placing this outside any enclosing
    ``GPU_ThreadBlock`` map -- this expansion *is* the thread-block-level
    operation. Shared <-> Register goes through ``MappedTasklet`` (auto
    selector routes it there)."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)

        valid_storages = {dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global}
        if inp.storage not in valid_storages or out.storage not in valid_storages:
            raise ValueError(f"SharedMemoryCollective requires GPU_Shared / GPU_Global storages "
                             f"(got {inp.storage} -> {out.storage}). Use MappedTasklet for "
                             "Shared <-> Register thread-level copies.")
        if inp.storage != dtypes.StorageType.GPU_Shared and out.storage != dtypes.StorageType.GPU_Shared:
            raise ValueError("SharedMemoryCollective requires at least one side to be GPU_Shared.")

        # The collective copy IS the thread-block-level operation; it must not
        # sit inside an enclosing GPU_ThreadBlock map (``is_in_scope`` walks the
        # scope dict and up through nested SDFGs).
        if is_in_scope(parent_sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock]):
            raise ValueError("SharedMemoryCollective IS the thread-block-level operation "
                             "and must not be nested inside a GPU_ThreadBlock map.")

        return nodes.Tasklet(node.name,
                             inputs={CopyLibraryNode.INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)},
                             outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=_build_shmem_collective_copy_code(inp, in_subset, out, out_subset),
                             language=dace.Language.CPP)


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    """Library node representing a data copy between two access nodes.

    Each implementation name describes the C++ it emits: ``MappedTasklet``
    (element-wise tasklet, schedule from storages; also handles rank-mismatch
    reshapes via a 1-D walker when both endpoints are packed-same-layout with
    contiguous subsets), ``Tasklet`` (bare assignment, no map), ``MemcpyCPU``
    (``std::memcpy``), ``MemcpyCUDA1D``/``2D`` (one ``cudaMemcpyAsync`` /
    ``cudaMemcpy2DAsync``), ``MemcpyCUDANDStrided`` (Sequential map of
    ``cudaMemcpyAsync``), ``SharedMemoryCollective`` (``dace::CopyND`` +
    ``__syncthreads()``; the only remaining ``dace::CopyND`` user).

    Design rationale: the libnode does NOT accept dynamic (Scalar) input
    connectors -- subset expressions must use symbols already in scope at
    construction time. This keeps the contract simple and lets the auto
    selector reason purely from the static memlet subsets.
    """

    implementations = {
        "Auto": ExpandAuto,
        "MappedTasklet": ExpandMappedTasklet,
        "Tasklet": ExpandTasklet,
        "MemcpyCPU": ExpandMemcpyCPU,
        "MemcpyCUDA1D": ExpandMemcpyCUDA1D,
        "MemcpyCUDA2D": ExpandMemcpyCUDA2D,
        "MemcpyCUDANDStrided": ExpandMemcpyCUDANDStrided,
        "SharedMemoryCollective": ExpandSharedMemoryCollective,
    }
    default_implementation = 'Auto'

    # Connector names exposed for library node builders.
    INPUT_CONNECTOR_NAME = "_cpy_in"
    OUTPUT_CONNECTOR_NAME = "_cpy_out"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={CopyLibraryNode.INPUT_CONNECTOR_NAME},
                         outputs={CopyLibraryNode.OUTPUT_CONNECTOR_NAME},
                         **kwargs)

    def src_storage(self, state) -> dtypes.StorageType:
        """Storage of the array feeding ``_cpy_in``, or ``Default`` if unwired.

        :param state: state containing this libnode (owning SDFG is ``state.sdfg``).
        :returns: the source :class:`~dace.dtypes.StorageType`.
        """
        in_edges = [e for e in state.in_edges(self) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME]
        if not in_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(in_edges[0])[0].src
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return state.sdfg.arrays[outer.data].storage

    def dst_storage(self, state) -> dtypes.StorageType:
        """Storage of the array fed by ``_cpy_out``, or ``Default`` if unwired.

        :param state: state containing this libnode (owning SDFG is ``state.sdfg``).
        :returns: the destination :class:`~dace.dtypes.StorageType`.
        """
        out_edges = [e for e in state.out_edges(self) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME]
        if not out_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(out_edges[0])[-1].dst
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return state.sdfg.arrays[outer.data].storage

    def validate(self, sdfg, state, allow_cross_storage=True):
        """Resolve in/out edges, names, and subsets.

        :param sdfg: SDFG containing ``state``.
        :param state: state containing this libnode.
        :param allow_cross_storage: when False, require matching src/dst storages.
        :returns: ``(inp_name, inp, in_subset, out_name, out, out_subset)``.
        :raises ValueError: the libnode is not wired with exactly one input
            and one output data edge, dtypes mismatch, an extraneous
            non-reserved input connector is wired, or (when
            ``allow_cross_storage`` is False) the two storages differ.
        """
        out_edges = [oe for oe in state.out_edges(self) if oe.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME]
        if len(out_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{CopyLibraryNode.OUTPUT_CONNECTOR_NAME}`` output edge.")
        oe = out_edges[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        # Reject any non-reserved input connector: the libnode does not accept
        # dynamic inputs (see class docstring's design rationale).
        reserved = {CopyLibraryNode.INPUT_CONNECTOR_NAME, CURRENT_STREAM_NAME}
        extra = [ie.dst_conn for ie in state.in_edges(self) if ie.dst_conn not in reserved and not ie.data.is_empty()]
        if extra:
            raise ValueError(f"{type(self).__name__} does not accept dynamic input connectors; got {extra}. "
                             f"Subset expressions must use symbols already in scope.")

        in_edges = [ie for ie in state.in_edges(self) if ie.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME]
        if len(in_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one data input edge "
                             f"connected to the ``{CopyLibraryNode.INPUT_CONNECTOR_NAME}`` connector.")
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

        return inp_name, inp, in_subset, out_name, out, out_subset
