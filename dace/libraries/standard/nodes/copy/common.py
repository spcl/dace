# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared pieces of the ``CopyLibraryNode`` expansions.

Imported by both the node and its expansions, so it must not import either.
"""
import functools
import operator
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import dace
from dace import data, nodes, dtypes, subsets, symbolic
from dace.codegen.common import sym2cpp, get_gpu_backend
from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, CPU_RESIDENT_STORAGES, GPU_RESIDENT_STORAGES,
                                            collapse_shape_and_strides)
from dace.sdfg.scope import devicelevel_block_size, is_devicelevel_gpu

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode

INPUT_CONNECTOR_NAME = "_cpy_in"
OUTPUT_CONNECTOR_NAME = "_cpy_out"


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
    in_shape_collapsed: List[symbolic.SymExpr]
    out_shape_collapsed: List[symbolic.SymExpr]


def _is_cross_cpu_gpu(src_storage: dtypes.StorageType, dst_storage: dtypes.StorageType, copy_node: "CopyLibraryNode",
                      parent_state: dace.SDFGState) -> bool:
    """True if src/dst cross the CPU/GPU boundary. ``Register`` follows scope: GPU scope -> GPU,
    else CPU."""
    in_gpu = is_devicelevel_gpu(parent_state.sdfg, parent_state, copy_node)

    src_gpu = (src_storage in GPU_RESIDENT_STORAGES) or (src_storage == dtypes.StorageType.Register and in_gpu)
    dst_gpu = (dst_storage in GPU_RESIDENT_STORAGES) or (dst_storage == dtypes.StorageType.Register and in_gpu)

    src_cpu = (src_storage in CPU_RESIDENT_STORAGES) or (src_storage == dtypes.StorageType.Register and not in_gpu)
    dst_cpu = (dst_storage in CPU_RESIDENT_STORAGES) or (dst_storage == dtypes.StorageType.Register and not in_gpu)

    return (src_cpu and dst_gpu) or (src_gpu and dst_cpu)


def _copy_waives_volume_check(copy_node: "CopyLibraryNode", parent_state: dace.SDFGState) -> bool:
    """True if a wired memlet carries ``allow_oob``, i.e. the author waived the src/dst volume check.

    ``SDFGState.validate`` (``validation.py``) skips its own equal-volume check on such an edge, and
    plain copy-edge codegen sizes the transfer from the source subset. A lifted ``CopyLibraryNode``
    must honor the same waiver, else an SDFG that is legal as a direct copy edge stops expanding.
    """
    return any(e.data.allow_oob for e in parent_state.all_edges(copy_node) if not e.data.is_empty())


def _both_packed_same_layout(inp: data.Data, out: data.Data) -> bool:
    """True if both descriptors share packed major order (both C or both Fortran)."""
    return ((inp.is_packed_c_strides() and out.is_packed_c_strides())
            or (inp.is_packed_fortran_strides() and out.is_packed_fortran_strides()))


def _delinearized_index(b_i: symbolic.symbol, shape: List[symbolic.SymExpr], layout: str) -> List[symbolic.SymExpr]:
    """Multi-dim index for a 1-D walker into a packed-layout array. Only C (row-major) and
    F (column-major) layouts are supported.

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


def cuda2d_pitch_params(
    copy_shape: List[symbolic.SymExpr], src_strides: List[symbolic.SymExpr], dst_strides: List[symbolic.SymExpr]
) -> Optional[Tuple[symbolic.SymExpr, symbolic.SymExpr, symbolic.SymExpr, symbolic.SymExpr]]:
    """Element-count ``cudaMemcpy2DAsync`` pitch params ``(dpitch, spitch, width, height)`` for a
    2D (or ``(N, 1)``-promoted) copy, or ``None`` if not a single ``cudaMemcpy2DAsync``. Single
    source of truth for the ``MemcpyCUDA2D`` selector gate and the expander, so the two can't
    drift: selector treats a non-``None`` result as "applies"; expander formats the same
    components into the emitted call. Values are in elements; caller multiplies pitch/width by
    ``sizeof(dtype)``.

    :param copy_shape: Two-element collapsed copy shape ``(rows, columns)``.
    :param src_strides: Two-element source strides aligned with ``copy_shape``.
    :param dst_strides: Two-element destination strides aligned with ``copy_shape``.
    :returns: ``(dpitch, spitch, width, height)`` in elements, or ``None`` if not a single
        ``cudaMemcpy2DAsync``.
    """
    if src_strides[1] == 1 and dst_strides[1] == 1:
        return dst_strides[0], src_strides[0], copy_shape[1], copy_shape[0]
    if src_strides[0] == 1 and dst_strides[0] == 1:
        return dst_strides[1], src_strides[1], copy_shape[0], copy_shape[1]
    try:
        if (not symbolic.inequal_symbols(src_strides[0] / src_strides[1], copy_shape[1])
                and not symbolic.inequal_symbols(dst_strides[0] / dst_strides[1], copy_shape[1])):
            return dst_strides[1], src_strides[1], 1, copy_shape[0] * copy_shape[1]
    except (TypeError, ZeroDivisionError):
        return None
    return None


def _make_expansion_sdfg(node: "CopyLibraryNode",
                         parent_state: dace.SDFGState,
                         allow_cross_storage: bool = False) -> CopyExpansion:
    """Shared validation + wrapper-SDFG skeleton for expansions.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :param allow_cross_storage: permit differing src/dst storages.
    :returns: a :class:`CopyExpansion` with the skeleton SDFG and collapsed shape/stride state.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=allow_cross_storage)

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    # The label is built from data names, and a struct member carries a '.' -- the SDFG name reaches
    # C++ as an identifier (``dace/sdfg/sdfg.py`` sanitizes data names the same way).
    label = node.label.replace('.', '_')
    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    # Match the ambient stream connector if the experimental GPU codegen wired one in.
    if CURRENT_STREAM_NAME in node.in_connectors:
        sdfg.add_scalar(CURRENT_STREAM_NAME, dtypes.gpuStream_t, transient=False)

    state = sdfg.add_state(f"{label}_state", is_start_block=True)

    return CopyExpansion(sdfg=sdfg,
                         state=state,
                         inp_name=inp_name,
                         inp=inp,
                         in_subset=in_subset,
                         out_name=out_name,
                         out=out,
                         out_subset=out_subset,
                         in_shape_collapsed=in_shape_collapsed,
                         out_shape_collapsed=out_shape_collapsed)


def _make_mapped_tasklet_expansion(node: "CopyLibraryNode",
                                   parent_state: dace.SDFGState,
                                   allow_cross_storage: bool = False) -> dace.SDFG:
    """Element-wise mapped tasklet expansion. Raises if the copy crosses the CPU/GPU boundary.

    Schedule from storages: ``Sequential`` for thread-level (Register/Register,
    Register<->GPU_Shared) or any in-kernel copy; ``GPU_Device`` if any side is GPU storage
    at host level; else ``Default``.

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

    is_register = lambda s: s == dtypes.StorageType.Register
    is_thread_local = (is_register(inp.storage) and is_register(out.storage)) or (
        (is_register(inp.storage) and out.storage == dtypes.StorageType.GPU_Shared) or
        (is_register(out.storage) and inp.storage == dtypes.StorageType.GPU_Shared))
    in_kernel = is_devicelevel_gpu(parent_state.sdfg, parent_state, node)
    if is_thread_local or in_kernel:
        schedule = dtypes.ScheduleType.Sequential
    elif inp.storage in GPU_RESIDENT_STORAGES or out.storage in GPU_RESIDENT_STORAGES:
        schedule = dtypes.ScheduleType.GPU_Device
    else:
        # Default, not Sequential, inside an enclosing map: SCOPEDEFAULT_SCHEDULE already resolves a
        # map nested in a CPU_Multicore one to single-threaded.
        schedule = dtypes.ScheduleType.Default

    ctx.sdfg.schedule = dtypes.ScheduleType.Default

    # Must not collide with the wrapper SDFG's parameter arrays (named after outer connectors).
    inner_in, inner_out = "_in", "_out"
    in_shape, out_shape = ctx.in_shape_collapsed, ctx.out_shape_collapsed

    # A copy of zero elements moves nothing; the map below has an empty range, so the per-dim shape
    # agreement the transfer would otherwise need is vacuous.
    # ``is_length=False``: the default assumes every operand is positive, which 0 contradicts.
    zero_size = any(symbolic.equal(s, 0, is_length=False) is True for s in in_shape + out_shape)

    if len(in_shape) == len(out_shape):
        # Same-rank: per-dim map params, shared access expr on both sides. Shapes must match
        # (per-dim permutations are transposes, not reshapes), unless the author waived the check.
        # Refuse only on a PROVEN mismatch -- `equal` answers None when it cannot tell, and a
        # symbolic pair such as ``ceiling(N/2)`` vs ``floor(N/2)`` is not grounds to reject.
        if not zero_size and not _copy_waives_volume_check(node, parent_state) and any(
                symbolic.equal(a, b) is False for a, b in zip(in_shape, out_shape)):
            raise ValueError(f"MappedTasklet same-rank copy requires matching per-dim shapes; got src "
                             f"{tuple(in_shape)} vs dst {tuple(out_shape)}. Per-dim permutations are not "
                             f"supported -- use a Transpose libnode. Reshapes must change rank.")
        map_params = [f"__i{i}" for i in range(len(in_shape))]
        # Symbolic bounds, never a rendered string: ``sym2cpp`` spells a symbolic extent as C++
        # (``dace::math::ipow(R, K)``), and the range parser splits on ':', so the qualified name
        # comes back as four bogus tokens.
        map_rng = {i: (0, s - 1, 1) for i, s in zip(map_params, in_shape)}
        access_expr = ','.join(map_params)
        inputs = {inner_in: dace.memlet.Memlet(f"{ctx.inp_name}[{access_expr}]")}
        outputs = {inner_out: dace.memlet.Memlet(f"{ctx.out_name}[{access_expr}]")}
    else:
        # Rank-mismatch reshape: 1-D walker over the total element count. Needs both endpoints
        # packed same major order -- mixed layouts have no shared flat order.
        if not _both_packed_same_layout(inp, out):
            raise ValueError(
                f"MappedTasklet rank-mismatched copy ({tuple(in_shape)} -> {tuple(out_shape)}) requires "
                f"both endpoints to be packed in the same major order (both C-contiguous or both "
                f"Fortran-contiguous). Got src '{ctx.inp_name}' strides {tuple(inp.strides)} on shape "
                f"{tuple(inp.shape)} and dst '{ctx.out_name}' strides {tuple(out.strides)} on shape "
                f"{tuple(out.shape)}. Mixed layouts are transposes -- use a same-rank Tasklet copy instead.")
        layout = 'C' if inp.is_packed_c_strides() else 'F'
        if layout == 'F':
            # Under Fortran order the walker visits the FIRST collapsed dim fastest, but a strided or
            # tiled dim collapses to (step count, tile) with the tile innermost -- only the C walk
            # order then matches subset iteration order, so Fortran still needs flat subsets.
            in_contig = ctx.in_subset.is_contiguous_subset(inp)
            out_contig = ctx.out_subset.is_contiguous_subset(out)
            if not (in_contig and out_contig):
                raise ValueError(
                    f"MappedTasklet rank-mismatched copy ({tuple(in_shape)} -> {tuple(out_shape)}) requires "
                    f"contiguous subsets on both endpoints in Fortran layout (the 1-D walker treats the data "
                    f"as a flat sequence). Got src subset {ctx.in_subset} (contiguous: {in_contig}) on shape "
                    f"{tuple(inp.shape)} and dst subset {ctx.out_subset} (contiguous: {out_contig}) on shape "
                    f"{tuple(out.shape)}.")

        # Product of the collapsed shape, not ``num_elements_exact`` -- the latter is a BOUNDING BOX
        # (``subsets.py``), which overcounts a strided or tiled subset.
        total = functools.reduce(operator.mul, in_shape, 1)
        b_i_name = "__b_i"
        b_i = symbolic.symbol(b_i_name)
        map_rng = {b_i_name: (0, total - 1, 1)}

        def _side_access(arr_name, shape):
            idx = [b_i] if len(shape) == 1 else _delinearized_index(b_i, shape, layout)
            return dace.memlet.Memlet(data=arr_name, subset=subsets.Range([(e, e, 1) for e in idx]))

        inputs = {inner_in: _side_access(ctx.inp_name, in_shape)}
        outputs = {inner_out: _side_access(ctx.out_name, out_shape)}

    _, map_entry, _ = ctx.state.add_mapped_tasklet(f"{node.label}_tasklet",
                                                   map_rng,
                                                   inputs,
                                                   copy_assignment_code(inp, out, inner_in, inner_out),
                                                   outputs,
                                                   schedule=schedule,
                                                   external_edges=True)

    return ctx.sdfg


def copy_assignment_code(inp: data.Data, out: data.Data, in_conn: str, out_conn: str) -> str:
    """The tasklet body for one element of a copy: an assignment, or a CAST when the copy changes
    dtype.

    Written out as ``dace.<dtype>(...)`` rather than left to C++'s implicit conversion so the
    narrowing is visible in the graph and in the emitted code.
    """
    if inp.dtype == out.dtype:
        return f"{out_conn} = {in_conn}"
    name = out.dtype.to_string()
    cast = name if out.dtype in (dtypes.bool, dtypes.bool_) else f"dace.{name}"
    return f"{out_conn} = {cast}({in_conn})"


def _memcpy_kind(inp: data.Data, out: data.Data) -> str:
    """``gpuMemcpy<src>To<dst>`` from endpoint storages."""
    src_loc = "Device" if inp.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    dst_loc = "Device" if out.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    backend = get_gpu_backend()
    return f"{backend}Memcpy{src_loc}To{dst_loc}"


def _make_memcpy_tasklet(node: "CopyLibraryNode", parent_state: dace.SDFGState, *, cuda: bool) -> nodes.Tasklet:
    """Build a Tasklet emitting one contiguous-block copy. Raises ``ValueError`` on a
    non-contiguous subset (the single-call form would overrun the region; use ``MappedTasklet``).

    Emits ``gpuMemcpyAsync`` when ``cuda`` is set -- cross-CPU/GPU allowed, direction
    (HostToDevice/DeviceToHost/DeviceToDevice/HostToHost) inferred from endpoint storages --
    else a same-storage ``std::memcpy``.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :param cuda: emit ``gpuMemcpyAsync`` (else ``memcpy``).
    :returns: a :class:`~dace.sdfg.nodes.Tasklet` issuing the copy.
    :raises ValueError: a subset is non-contiguous.
    """
    label = "MemcpyCUDA1D" if cuda else "MemcpyCPU"
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=cuda)
    single_elt = (in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1)
    if single_elt:
        pass
    elif not (in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out)):
        raise ValueError(f"{label} requires contiguous subsets; got src '{inp_name}' subset {in_subset} "
                         f"(shape {inp.shape} strides {inp.strides}) and dst '{out_name}' subset {out_subset} "
                         f"(shape {out.shape} strides {out.strides}). Use MappedTasklet for strided subsets.")

    in_conn = INPUT_CONNECTOR_NAME
    out_conn = OUTPUT_CONNECTOR_NAME
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


def _build_shmem_collective_copy_code(node: "CopyLibraryNode", parent_state: dace.SDFGState, inp: data.Data,
                                      in_subset: dace.subsets.Range, out: data.Data,
                                      out_subset: dace.subsets.Range) -> str:
    """Build the C++ code for ``ExpandSharedMemoryCollective``.

    A static 1-D transfer inside a kernel uses DaCe's block-collective runtime helpers
    (``dace::GlobalToShared1D`` / ``dace::SharedToGlobal1D``), which split the elements across the
    thread block -- the same call plain copy-edge codegen emits. Everything else falls back to a
    ``dace::CopyND<...>::Copy(...)`` plus ``__syncthreads()``: the most-specific static template
    (``CopyNDDynamic`` for symbolic shapes), refined by ``ConstDst``/``ConstSrc``/``Dynamic`` on
    whichever stride set is constexpr, with the rest passed as runtime args.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node`` (supplies the enclosing thread-block size).
    :param inp: source descriptor (provides ``ctype`` and ``strides``).
    :param in_subset: source memlet subset.
    :param out: destination descriptor (provides ``strides``).
    :param out_subset: destination memlet subset.
    :returns: the tasklet body.
    """
    copy_shape, src_strides = collapse_shape_and_strides(in_subset, inp.strides)
    _, dst_strides = collapse_shape_and_strides(out_subset, out.strides)
    ndims = len(copy_shape)

    in_conn = INPUT_CONNECTOR_NAME
    out_conn = OUTPUT_CONNECTOR_NAME
    block_dims = devicelevel_block_size(parent_state.sdfg, parent_state, node)
    if ndims == 1 and block_dims is not None and not any(
            symbolic.issymbolic(s) for s in (copy_shape[0], src_strides[0], dst_strides[0])):
        bdims = ', '.join(sym2cpp(b) for b in block_dims)
        args = f"{inp.dtype.ctype}, {bdims}, {sym2cpp(copy_shape[0])}"
        if out.storage == dtypes.StorageType.GPU_Shared:
            return (f"dace::GlobalToShared1D<{args}, {sym2cpp(dst_strides[0])}, false>"
                    f"({in_conn}, {sym2cpp(src_strides[0])}, {out_conn});")
        return (f"dace::SharedToGlobal1D<{args}, false>::Copy"
                f"({in_conn}, {sym2cpp(src_strides[0])}, {out_conn}, {sym2cpp(dst_strides[0])});")

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

    # Prefer ConstDst, else ConstSrc, else Dynamic; the rest go as runtime args, per-dim order.
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

    all_args = [in_conn, out_conn] + stride_args
    # Synchronize if moving to/from shared memory collectively, and the sync flag is set (default on)
    sync_barrier = "__syncthreads();\n" if node.sync and (inp.storage == dtypes.StorageType.GPU_Shared
                                                          or out.storage == dtypes.StorageType.GPU_Shared) else ""
    return f"{sync_barrier}{copy_tmpl}::{shape_tmpl}::Copy({', '.join(all_args)});\n{sync_barrier}"
