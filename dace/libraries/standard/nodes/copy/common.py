# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared pieces of the ``CopyLibraryNode`` expansions.

Imported by both the node and its expansions, so it must not import either.
"""

import functools
import operator
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy

import dace
from dace import data, dtypes, nodes, subsets, symbolic
from dace.codegen.common import get_gpu_backend, sym2cpp
from dace.libraries.standard.helper import (
    CPU_RESIDENT_STORAGES,
    CURRENT_STREAM_NAME,
    GPU_RESIDENT_STORAGES,
    collapse_shape_and_strides,
)
from dace.sdfg.scope import devicelevel_block_size, is_devicelevel_gpu, is_in_scope

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode

INPUT_CONNECTOR_NAME = "_cpy_in"
OUTPUT_CONNECTOR_NAME = "_cpy_out"

MEMORY_ENGINE_DESC_INPUT_NAME = "_tma_desc_in"
MEMORY_ENGINE_DESC_OUTPUT_NAME = "_tma_desc_out"
MEMORY_ENGINE_GLOBAL_INPUT_NAME = "_tma_global"
MEMORY_ENGINE_BARRIER_INPUT_NAME = "_tma_bar"
MEMORY_ENGINE_BARRIER_OUTPUT_NAME = "_tma_bar_out"

CUtensorMap = dace.dtypes.opaque("CUtensorMap")

# TODO: keep these values up-to-date if support for new memory engines is added
MEMORY_ENGINE_MAX_RANK = 5
"""Maximum tensor rank the memory engine supports."""
MEMORY_ENGINE_MAX_BOX_DIM = 256
"""Maximum extent of a memory engine box (tile) dimension."""
MEMORY_ENGINE_STRIDE_ALIGNMENT = 16
"""Byte alignment of the global strides and of the contiguous box extent."""
MEMORY_ENGINE_SHARED_ALIGNMENT = 128
"""Byte alignment of the shared memory side of a copy."""

_TMA_DATA_TYPES = {
    dtypes.bool_: "CU_TENSOR_MAP_DATA_TYPE_UINT8",
    dtypes.int8: "CU_TENSOR_MAP_DATA_TYPE_UINT8",
    dtypes.uint8: "CU_TENSOR_MAP_DATA_TYPE_UINT8",
    dtypes.int16: "CU_TENSOR_MAP_DATA_TYPE_UINT16",
    dtypes.uint16: "CU_TENSOR_MAP_DATA_TYPE_UINT16",
    dtypes.int32: "CU_TENSOR_MAP_DATA_TYPE_INT32",
    dtypes.uint32: "CU_TENSOR_MAP_DATA_TYPE_UINT32",
    dtypes.int64: "CU_TENSOR_MAP_DATA_TYPE_INT64",
    dtypes.uint64: "CU_TENSOR_MAP_DATA_TYPE_UINT64",
    dtypes.float16: "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
    dtypes.bfloat16: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    dtypes.float32: "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
    dtypes.float64: "CU_TENSOR_MAP_DATA_TYPE_FLOAT64",
}
"""``CUtensorMapDataType`` per element type.

The memory engine does not care about signedness, only bits."""


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
    in_shape_collapsed: list[symbolic.SymExpr]
    out_shape_collapsed: list[symbolic.SymExpr]


def _is_cross_cpu_gpu(
    src_storage: dtypes.StorageType,
    dst_storage: dtypes.StorageType,
    copy_node: "CopyLibraryNode",
    parent_state: dace.SDFGState,
) -> bool:
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
    return (inp.is_packed_c_strides() and out.is_packed_c_strides()) or (
        inp.is_packed_fortran_strides() and out.is_packed_fortran_strides()
    )


def _delinearized_index(b_i: symbolic.symbol, shape: list[symbolic.SymExpr], layout: str) -> list[symbolic.SymExpr]:
    """Multi-dim index for a 1-D walker into a packed-layout array. Only C (row-major) and
    F (column-major) layouts are supported.

    :param b_i: the 1-D map symbol.
    :param shape: per-dim extents in descriptor order.
    :param layout: ``'C'`` (stride-1 is the last dim) or ``'F'`` (stride-1 is the first dim).
    :returns: list of per-dim symbolic index expressions, in descriptor order.
    """
    cum_strides = []
    cum = 1
    iter_shape = reversed(shape) if layout == "C" else iter(shape)
    for s in iter_shape:
        cum_strides.append(cum)
        cum *= s
    if layout == "C":
        cum_strides.reverse()
    return [symbolic.int_floor(b_i, cum_strides[d]) % shape[d] for d in range(len(shape))]


def cuda2d_pitch_params(
    copy_shape: list[symbolic.SymExpr], src_strides: list[symbolic.SymExpr], dst_strides: list[symbolic.SymExpr]
) -> tuple[symbolic.SymExpr, symbolic.SymExpr, symbolic.SymExpr, symbolic.SymExpr] | None:
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
        if not symbolic.inequal_symbols(
            src_strides[0] / src_strides[1], copy_shape[1]
        ) and not symbolic.inequal_symbols(dst_strides[0] / dst_strides[1], copy_shape[1]):
            return dst_strides[1], src_strides[1], 1, copy_shape[0] * copy_shape[1]
    except (TypeError, ZeroDivisionError):
        return None
    return None


def _make_expansion_sdfg(
    node: "CopyLibraryNode", parent_state: dace.SDFGState, allow_cross_storage: bool = False
) -> CopyExpansion:
    """Shared validation + wrapper-SDFG skeleton for expansions.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :param allow_cross_storage: permit differing src/dst storages.
    :returns: a :class:`CopyExpansion` with the skeleton SDFG and collapsed shape/stride state.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(
        parent_state.sdfg, parent_state, allow_cross_storage=allow_cross_storage
    )

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    # The label is built from data names, and a struct member carries a '.' -- the SDFG name reaches
    # C++ as an identifier (``dace/sdfg/sdfg.py`` sanitizes data names the same way).
    label = node.label.replace(".", "_")
    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array(inp_name, in_shape_collapsed, inp.dtype, inp.storage, strides=in_strides_collapsed)
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    # Match the ambient stream connector if the experimental GPU codegen wired one in.
    if CURRENT_STREAM_NAME in node.in_connectors:
        sdfg.add_scalar(CURRENT_STREAM_NAME, dtypes.gpuStream_t, transient=False)

    state = sdfg.add_state(f"{label}_state", is_start_block=True)

    return CopyExpansion(
        sdfg=sdfg,
        state=state,
        inp_name=inp_name,
        inp=inp,
        in_subset=in_subset,
        out_name=out_name,
        out=out,
        out_subset=out_subset,
        in_shape_collapsed=in_shape_collapsed,
        out_shape_collapsed=out_shape_collapsed,
    )


def _make_mapped_tasklet_expansion(
    node: "CopyLibraryNode", parent_state: dace.SDFGState, allow_cross_storage: bool = False
) -> dace.SDFG:
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
        raise ValueError(
            "MappedTasklet expansion cannot cross the CPU/GPU boundary "
            f"(got {inp.storage} -> {out.storage}). Use a MemcpyCUDA1D variant."
        )

    is_register = lambda s: s == dtypes.StorageType.Register
    is_thread_local = (is_register(inp.storage) and is_register(out.storage)) or (
        (is_register(inp.storage) and out.storage == dtypes.StorageType.GPU_Shared)
        or (is_register(out.storage) and inp.storage == dtypes.StorageType.GPU_Shared)
    )
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
        if (
            not zero_size
            and not _copy_waives_volume_check(node, parent_state)
            and any(symbolic.equal(a, b) is False for a, b in zip(in_shape, out_shape))
        ):
            raise ValueError(
                f"MappedTasklet same-rank copy requires matching per-dim shapes; got src "
                f"{tuple(in_shape)} vs dst {tuple(out_shape)}. Per-dim permutations are not "
                f"supported -- use a Transpose libnode. Reshapes must change rank."
            )
        map_params = [f"__i{i}" for i in range(len(in_shape))]
        # Symbolic bounds, never a rendered string: ``sym2cpp`` spells a symbolic extent as C++
        # (``dace::math::ipow(R, K)``), and the range parser splits on ':', so the qualified name
        # comes back as four bogus tokens.
        map_rng = {i: (0, s - 1, 1) for i, s in zip(map_params, in_shape)}
        access_expr = ",".join(map_params)
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
                f"{tuple(out.shape)}. Mixed layouts are transposes -- use a same-rank Tasklet copy instead."
            )
        layout = "C" if inp.is_packed_c_strides() else "F"
        if layout == "F":
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
                    f"{tuple(out.shape)}."
                )

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

    _, map_entry, _ = ctx.state.add_mapped_tasklet(
        f"{node.label}_tasklet",
        map_rng,
        inputs,
        f"{inner_out} = {inner_in}",
        outputs,
        schedule=schedule,
        external_edges=True,
    )

    return ctx.sdfg


def _memcpy_kind(inp: data.Data, out: data.Data) -> str:
    """``cudaMemcpy<src>To<dst>`` from endpoint storages."""
    src_loc = "Device" if inp.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    dst_loc = "Device" if out.storage == dace.dtypes.StorageType.GPU_Global else "Host"
    backend = get_gpu_backend()
    return f"{backend}Memcpy{src_loc}To{dst_loc}"


def _make_memcpy_tasklet(node: "CopyLibraryNode", parent_state: dace.SDFGState, *, cuda: bool) -> nodes.Tasklet:
    """Build a Tasklet emitting one contiguous-block copy. Raises ``ValueError`` on a
    non-contiguous subset (the single-call form would overrun the region; use ``MappedTasklet``).

    Emits ``cudaMemcpyAsync`` when ``cuda`` is set -- cross-CPU/GPU allowed, direction
    (HostToDevice/DeviceToHost/DeviceToDevice/HostToHost) inferred from endpoint storages --
    else a same-storage ``std::memcpy``.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :param cuda: emit ``cudaMemcpyAsync`` (else ``memcpy``).
    :returns: a :class:`~dace.sdfg.nodes.Tasklet` issuing the copy.
    :raises ValueError: a subset is non-contiguous.
    """
    label = "MemcpyCUDA1D" if cuda else "MemcpyCPU"
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(
        parent_state.sdfg, parent_state, allow_cross_storage=cuda
    )
    single_elt = in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1
    if single_elt:
        pass
    elif not (in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out)):
        raise ValueError(
            f"{label} requires contiguous subsets; got src '{inp_name}' subset {in_subset} "
            f"(shape {inp.shape} strides {inp.strides}) and dst '{out_name}' subset {out_subset} "
            f"(shape {out.shape} strides {out.strides}). Use MappedTasklet for strided subsets."
        )

    in_conn = INPUT_CONNECTOR_NAME
    out_conn = OUTPUT_CONNECTOR_NAME
    nbytes = f"{sym2cpp(in_subset.num_elements_exact())} * sizeof({inp.dtype.ctype})"
    if cuda:
        backend = get_gpu_backend()
        code = (
            f"{backend}MemcpyAsync({out_conn}, {in_conn}, {nbytes}, {_memcpy_kind(inp, out)}, {CURRENT_STREAM_NAME});"
        )
    else:
        code = (
            f"dace::CopyImpl<{inp.dtype.ctype}, 1>({in_conn}, {out_conn}, {sym2cpp(in_subset.num_elements_exact())});"
        )

    return nodes.Tasklet(
        node.name,
        inputs={in_conn: dace.dtypes.pointer(inp.dtype)},
        outputs={out_conn: dace.dtypes.pointer(out.dtype)},
        code=code,
        language=dace.Language.CPP,
    )


def _build_shmem_collective_copy_code(
    node: "CopyLibraryNode",
    parent_state: dace.SDFGState,
    inp: data.Data,
    in_subset: dace.subsets.Range,
    out: data.Data,
    out_subset: dace.subsets.Range,
) -> str:
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
    if (
        ndims == 1
        and block_dims is not None
        and not any(symbolic.issymbolic(s) for s in (copy_shape[0], src_strides[0], dst_strides[0]))
    ):
        bdims = ", ".join(sym2cpp(b) for b in block_dims)
        args = f"{inp.dtype.ctype}, {bdims}, {sym2cpp(copy_shape[0])}"
        if out.storage == dtypes.StorageType.GPU_Shared:
            return (
                f"dace::GlobalToShared1D<{args}, {sym2cpp(dst_strides[0])}, false>"
                f"({in_conn}, {sym2cpp(src_strides[0])}, {out_conn});"
            )
        return (
            f"dace::SharedToGlobal1D<{args}, false>::Copy"
            f"({in_conn}, {sym2cpp(src_strides[0])}, {out_conn}, {sym2cpp(dst_strides[0])});"
        )

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
    sync_barrier = (
        "__syncthreads();\n"
        if node.sync and (inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared)
        else ""
    )
    return f"{sync_barrier}{copy_tmpl}::{shape_tmpl}::Copy({', '.join(all_args)});\n{sync_barrier}"


@dataclass
class MemoryEngineGeometry:
    """Everything a ``MemoryEngineBulk`` copy needs, with all per-dimension lists in TMA order
    (dimension 0 is the contiguous one, i.e. reversed for C-ordered arrays)."""

    is_load: bool
    """If true the copy is from global to shared memory. Reversed otherwise."""
    global_name: str
    """Name of the array in global memory, which is the source if ``is_load`` is true."""
    global_desc: data.Array
    """Array in global memory, which is the source if ``is_load`` is true."""
    global_subset: subsets.Range
    """Subset of the array in global memory, which is the source if ``is_load`` is true."""
    shared_name: str
    """Name of the array in shared memory, which is the source if ``is_load`` is false."""
    shared_desc: data.Array
    """Array in shared memory, which is the source if ``is_load`` is false."""
    shared_subset: subsets.Range
    """Subset of the array in shared memory, which is the source if ``is_load`` is false."""
    dims: List[symbolic.SymExpr]
    """Dimensions (sizes of the axes) of the global array."""
    strides: List[symbolic.SymExpr]
    """Strides of the global array."""
    box: List[symbolic.SymExpr]
    """Number of elements per axis that are being copied per load/store."""
    coords: List[symbolic.SymExpr]
    """Start of copy given in element-coordinates into the global array."""
    data_type: str
    """See ``_TMA_DATA_TYPES``"""

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def bytes(self) -> symbolic.SymExpr:
        """Bytes one copy moves; a load always moves the full box (out-of-bound elements are zero-filled)."""
        return functools.reduce(operator.mul, self.box, 1) * self.global_desc.dtype.bytes


def _memory_engine_edges(node: nodes.Node, state: dace.SDFGState):
    """The ``_cpy_in`` and ``_cpy_out`` edges of a ``CopyLibraryNode`` or of its expanded tasklet."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        return None, None
    return in_edges[0], out_edges[0]


def _sym_equal(a: symbolic.SymExpr, b: symbolic.SymExpr) -> bool:
    """True if ``a`` and ``b`` are provably equal."""
    try:
        return bool(symbolic.simplify(symbolic.pystr_to_symbolic(a) - symbolic.pystr_to_symbolic(b)) == 0)
    except TypeError:
        return False


def _tma_order(strides: List[symbolic.SymExpr]) -> Optional[List[int]]:
    """Dimension order with the contiguous dimension first

    For C-order, the innermost dimension must have stride 1 and the returned order is reversed. For Fortran order
    the outermost dimension must have stride 1. If the given strides are neither C nor Fortran ordered, ``None``
    is returned.
    """
    if _sym_equal(strides[-1], 1):
        return list(reversed(range(len(strides))))
    if _sym_equal(strides[0], 1):
        return list(range(len(strides)))
    return None


def _unclamp(size: symbolic.SymExpr, extent: symbolic.SymExpr) -> symbolic.SymExpr:
    """Remove memlet propagation's clamp to the array bound (``Min(extent - 1, ...)``) from ``size``.

    The memory engine zero-fills out-of-bounds elements. This means, we don't need to clamp the sizes manually.
    This function removes the ``Min(extent - 1, ...)`` clamps.
    """
    bound = symbolic.pystr_to_symbolic(extent) - 1

    def drop_bound(*args):
        # make sure to keep clamps that are not due to array bounds
        rest = [a for a in args if not _sym_equal(a, bound)]
        return sympy.Min(*rest) if rest else bound

    size = symbolic.pystr_to_symbolic(size)
    return symbolic.simplify(size.replace(sympy.Min, drop_bound))


def _static_int(value: symbolic.SymExpr) -> Optional[int]:
    """``value`` as an int if it is a constant, else ``None``."""
    if symbolic.issymbolic(value):
        return None
    return int(value)


def memory_engine_geometry(node: nodes.Node,
                           state: dace.SDFGState) -> Tuple[Optional[MemoryEngineGeometry], Optional[str]]:
    """Derive the tensor map geometry of a ``MemoryEngineBulk`` copy.

    Works on the ``CopyLibraryNode`` and on its expanded tasklet (both use the ``_cpy_in``/``_cpy_out``
    connectors), so that this method can be used in the expansion and its postprocessing.

    :param node: the copy node or expanded tasklet.
    :param state: state containing ``node``.
    :returns: ``(geometry, None)`` if the memory engine can perform the copy, ``(None, reason)`` otherwise.
    """
    sdfg = state.sdfg
    in_edge, out_edge = _memory_engine_edges(node, state)
    if in_edge is None or out_edge is None or in_edge.data.is_empty() or out_edge.data.is_empty():
        return None, "the copy must have exactly one input and one output edge"
    inp, out = sdfg.arrays[in_edge.data.data], sdfg.arrays[out_edge.data.data]

    if (inp.storage, out.storage) == (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
        is_load = True
        (gname, gdesc, gsubset), (sname, sdesc, ssubset) = ((in_edge.data.data, inp, in_edge.data.subset),
                                                            (out_edge.data.data, out, out_edge.data.subset))
    elif (inp.storage, out.storage) == (dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global):
        is_load = False
        (gname, gdesc, gsubset), (sname, sdesc, ssubset) = ((out_edge.data.data, out, out_edge.data.subset),
                                                            (in_edge.data.data, inp, in_edge.data.subset))
    else:
        return None, (
            f"MemoryEngineBulk only supports GPU_Global <-> GPU_Shared copies (got {inp.storage} -> {out.storage})")

    if inp.dtype != out.dtype:
        return None, f"input and output data types must match (got {inp.dtype} vs {out.dtype})"
    if gdesc.dtype not in _TMA_DATA_TYPES:
        return None, f"data type {gdesc.dtype} has no memory engine equivalent"
    for desc in (gdesc, sdesc):
        if not isinstance(desc, data.Array) or isinstance(desc, data.View):
            return None, f"both sides must be (non-view) arrays (got {type(desc).__name__})"
        if not _sym_equal(desc.start_offset, 0):
            return None, "arrays with a start offset are not supported"
    if len(gdesc.shape) > MEMORY_ENGINE_MAX_RANK:
        return None, f"MemoryEngineBulk only supports up to {MEMORY_ENGINE_MAX_RANK} dimensions (got {len(gdesc.shape)})"
    if any(not _sym_equal(step, 1) for subset in (gsubset, ssubset) for _, _, step in subset.ranges):
        return None, "strided (step != 1) copies are not supported"

    gorder = _tma_order(gdesc.strides)
    sorder = _tma_order(sdesc.strides)
    if gorder is None or sorder is None:
        return None, "both sides need a contiguous (stride 1) first or last dimension"

    elem_bytes = gdesc.dtype.bytes
    gsizes = [_unclamp(size, extent) for size, extent in zip(gsubset.size_exact(), gdesc.shape)]
    box = [gsizes[d] for d in gorder]

    # The memory engine writes/reads the box densely: the shared region must be the box, packed in TMA order
    ssizes = ssubset.size_exact()
    sdims = [d for d in sorder if not _sym_equal(ssizes[d], 1)]
    squeezed_box = [b for b in box if not _sym_equal(b, 1)]
    if len(sdims) != len(squeezed_box) or any(not _sym_equal(ssizes[d], b) for d, b in zip(sdims, squeezed_box)):
        return None, (f"the shared region {list(ssizes)} does not match the global box {list(gsizes)} "
                      "(reshapes and transposes are not supported)")
    packed = 1
    for d in sdims:
        if not _sym_equal(sdesc.strides[d], packed):
            return None, f"the shared region of {sname} is not packed"
        packed *= ssizes[d]

    for b in box:
        b_int = _static_int(b)
        if b_int is not None and b_int > MEMORY_ENGINE_MAX_BOX_DIM:
            return None, f"box dimensions must be at most {MEMORY_ENGINE_MAX_BOX_DIM} (got {list(gsizes)})"
    box0 = _static_int(box[0])
    if box0 is not None and (box0 * elem_bytes) % MEMORY_ENGINE_STRIDE_ALIGNMENT != 0:
        return None, (f"the contiguous box extent must be a multiple of {MEMORY_ENGINE_STRIDE_ALIGNMENT} bytes "
                      f"(got {box0} Elements x {elem_bytes}B/Element)")
    for d in gorder[1:]:
        stride = _static_int(gdesc.strides[d])
        if stride is not None and (stride * elem_bytes) % MEMORY_ENGINE_STRIDE_ALIGNMENT != 0:
            return None, (f"global strides must be multiples of {MEMORY_ENGINE_STRIDE_ALIGNMENT} bytes "
                          f"(got {stride} Elements x {elem_bytes}B/Element in {gname})")
    # When offsetting into a shared memory container, the base address must be aligned. For symbolic offsets, this
    # can't be checked statically and the hardware faults at runtime.
    soffset = _static_int(sum(m * s for m, s in zip(ssubset.min_element(), sdesc.strides)))
    if soffset is not None and (soffset * elem_bytes) % MEMORY_ENGINE_SHARED_ALIGNMENT != 0:
        return None, f"the shared region must start {MEMORY_ENGINE_SHARED_ALIGNMENT} byte aligned"

    gmin = gsubset.min_element()
    return MemoryEngineGeometry(is_load=is_load,
                                global_name=gname,
                                global_desc=gdesc,
                                global_subset=gsubset,
                                shared_name=sname,
                                shared_desc=sdesc,
                                shared_subset=ssubset,
                                dims=[gdesc.shape[d] for d in gorder],
                                strides=[gdesc.strides[d] for d in gorder],
                                box=box,
                                coords=[gmin[d] for d in gorder],
                                data_type=_TMA_DATA_TYPES[gdesc.dtype]), None


def _memory_engine_transient_container(node: nodes.Node, state: dace.SDFGState,
                                       geometry: MemoryEngineGeometry) -> Tuple[dace.SDFG, str]:
    """The transient container behind the shared side of a ``MemoryEngineBulk`` copy.

    Inside a NestedSDFG the shared side is an argument; this follows it outward to the transient that the shared
    memory planner places (see ``gpu_shared_memory.PlanSharedMemory``).

    :param node: the copy node or expanded tasklet.
    :param state: state containing ``node``.
    :param geometry: the copy geometry from :func:`memory_engine_geometry`.
    :returns: the SDFG holding the container and its name.
    """
    sdfg = state.sdfg
    in_edge, out_edge = _memory_engine_edges(node, state)
    edge = out_edge if geometry.is_load else in_edge
    while True:
        path = state.memlet_path(edge)
        access_node = path[-1].dst if geometry.is_load else path[0].src
        nsdfg_node = sdfg.parent_nsdfg_node
        if sdfg.arrays[access_node.data].transient or nsdfg_node is None:
            return sdfg, access_node.data
        parent_state = sdfg.parent
        if geometry.is_load:
            edge = next(e for e in parent_state.out_edges(nsdfg_node) if e.src_conn == access_node.data)
        else:
            edge = next(e for e in parent_state.in_edges(nsdfg_node) if e.dst_conn == access_node.data)
        state, sdfg = parent_state, sdfg.parent_sdfg


def _cuda_archs_support_memory_engine() -> bool:
    """False if ``compiler.cuda.cuda_arch`` lists an architecture older than Hopper (empty means native)."""
    archs = dace.Config.get('compiler', 'cuda', 'cuda_arch').replace(";", ",")
    archs = archs.split(",")
    archs = [a.strip() for a in archs]
    for arch in archs:
        # check valid architectures ignoring family or arch specific suffix
        if not all(c.isdigit() for c in arch) and arch[-1] not in ["a", "f"]:
            warnings.warn(f"compiler.cuda.cuda_arch contains unknown architecture: {arch}")

    numbers = [int("".join(c for c in a if c.isdigit()) or 0) for a in archs]
    return all(n >= 90 for n in numbers)


def _shared_container_reason(node: nodes.Node, state: dace.SDFGState, geometry: MemoryEngineGeometry) -> Optional[str]:
    """Why the shared side of a ``MemoryEngineBulk`` copy cannot be placed in static shared memory, which is
    ``MEMORY_ENGINE_SHARED_ALIGNMENT`` aligned (dynamic shared memory is not), or ``None`` if it can."""
    sdfg, name = _memory_engine_transient_container(node, state, geometry)
    desc = sdfg.arrays[name]
    if desc.storage != dtypes.StorageType.GPU_Shared:
        return f"the transient container {name} being {'loaded from' if geometry.is_load else 'stored to'} is not in shared memory"
    if dtypes.is_dynamic_shared(desc.storage):
        # TODO: remove this reason once dynamic shared memory is supported
        return f"the shared container {name} is placed in dynamic shared memory, which is not 128 byte aligned"
    if symbolic.issymbolic(desc.total_size_in_bytes, sdfg.constants):
        # TODO: remove this reason once dynamic shared memory is supported
        return f"the shared container {name} has a symbolic size, so it cannot be placed in static shared memory"
    return None


def memory_engine_bulk_reason(node: nodes.Node, state: dace.SDFGState) -> Optional[str]:
    """Why ``MemoryEngineBulk`` cannot expand ``node``, or ``None`` if it can.

    :param node: the ``CopyLibraryNode`` that is being expanded.
    :param state: state containing ``node``.
    :returns: a human-readable reason, or ``None``.
    """
    geometry, reason = memory_engine_geometry(node, state)
    if reason is not None:
        return reason
    reason = _shared_container_reason(node, state, geometry)
    if reason is not None:
        return reason
    sdfg = state.sdfg
    if not is_in_scope(sdfg, state, node, [dtypes.ScheduleType.GPU_Device]):
        return "MemoryEngineBulk must be placed inside a GPU_Device map (kernel)"
    sub_block_schedules = [
        dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic, dtypes.ScheduleType.GPU_Warp
    ]
    if is_in_scope(sdfg, state, node, sub_block_schedules):
        return ("MemoryEngineBulk IS the thread-block-level operation and must not be nested inside a "
                "GPU_ThreadBlock or GPU_Warp map")
    if get_gpu_backend() != 'cuda':
        # TODO: update once HIP is supported
        return "MemoryEngineBulk requires the CUDA backend"
    if not _cuda_archs_support_memory_engine():
        return "MemoryEngineBulk requires compute capability 9.0 or newer (see compiler.cuda.cuda_arch)"
    return None


def mark_memory_engine_shared_static(node: nodes.Node, state: dace.SDFGState, geometry: MemoryEngineGeometry):
    """Pin the shared side of a ``MemoryEngineBulk`` copy to static shared memory, which is aligned for the memory
    engine. If the static shared memory of the kernel then overflows, the CUDA compiler rejects the kernel instead of
    the copy faulting at run time. Other containers left to the planner move to dynamic shared memory first.

    TODO: Implement dynamic memory alignment
    """
    sdfg, name = _memory_engine_transient_container(node, state, geometry)
    desc = sdfg.arrays[name]
    if desc.storage == dtypes.StorageType.GPU_Shared:
        desc.storage = dtypes.StorageType.GPU_Shared(dynamic=False)


def _build_memory_engine_bulk_copy_code(node: "CopyLibraryNode", geometry: MemoryEngineGeometry) -> str:
    """Build the C++ code for ``ExpandMemoryEngineBulk``.

    The tasklet runs on every thread of the block. A load issues ``cp.async.bulk.tensor`` from the leader
    thread and waits on the mbarrier at ``_tma_bar`` until all bytes have arrived. A store fences the shared
    memory writes, issues the copy and waits for its completion. ``node.sync`` adds a leading
    ``__syncthreads()`` so the shared buffer is no longer read (load) or written (store) by other threads.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param geometry: the copy geometry from :func:`memory_engine_geometry`.
    :returns: the tasklet body.
    """
    coords = ', '.join(f"static_cast<int32_t>({sym2cpp(c)})" for c in geometry.coords)
    code = "__syncthreads();\n" if node.sync else ""
    code += f"int32_t const _tma_coords[{geometry.rank}] = {{{coords}}};\n"
    if geometry.is_load:
        code += (f"dace::cuda::tma_load<{geometry.rank}>({OUTPUT_CONNECTOR_NAME}, &{MEMORY_ENGINE_DESC_INPUT_NAME}, "
                 f"_tma_coords, {MEMORY_ENGINE_BARRIER_INPUT_NAME}, {sym2cpp(geometry.bytes)});\n")
    else:
        code += (f"dace::cuda::tma_store<{geometry.rank}>(&{MEMORY_ENGINE_DESC_INPUT_NAME}, _tma_coords, "
                 f"{INPUT_CONNECTOR_NAME});\n")
    return code


def _build_memory_engine_desc_code(geometry: MemoryEngineGeometry, dims: List[symbolic.SymExpr],
                                   strides: List[symbolic.SymExpr], offset: symbolic.SymExpr) -> str:
    """Build the host code encoding the tensor map of a ``MemoryEngineBulk`` copy.

    :param geometry: the copy geometry from :func:`memory_engine_geometry`.
    :param dims: global extents in TMA order, in host symbols.
    :param strides: global element strides in TMA order, in host symbols.
    :param offset: element offset of the kernel-visible global array from ``_tma_global``, in host symbols.
    :returns: the host tasklet body.
    """
    elem_bytes = geometry.global_desc.dtype.bytes
    dims_str = ', '.join(f"(cuuint64_t)({sym2cpp(d)})" for d in dims)
    strides_str = ', '.join(f"(cuuint64_t)({sym2cpp(s)}) * {elem_bytes}" for s in strides[1:]) or "0"
    box_str = ', '.join(f"(cuuint32_t)({sym2cpp(b)})" for b in geometry.box)
    return (
        f"const cuuint64_t _tma_dims[{geometry.rank}] = {{{dims_str}}};\n"
        f"const cuuint64_t _tma_strides[{max(geometry.rank - 1, 1)}] = {{{strides_str}}};\n"
        f"const cuuint32_t _tma_box[{geometry.rank}] = {{{box_str}}};\n" +
        "{} = dace::cuda::make_tiled_tensor_map(__state, {}, (void *)({} + {}), {}, _tma_dims, _tma_strides, _tma_box);\n"
        .format(
            MEMORY_ENGINE_DESC_OUTPUT_NAME,
            geometry.data_type,
            MEMORY_ENGINE_GLOBAL_INPUT_NAME,
            sym2cpp(offset),
            geometry.rank,
        ))


def _build_memory_engine_barrier_init_code() -> str:
    """Build the device code initializing the mbarrier of a ``MemoryEngineBulk`` load (once per kernel)."""
    return f"dace::cuda::tma_barrier_init({MEMORY_ENGINE_BARRIER_OUTPUT_NAME});\n"
