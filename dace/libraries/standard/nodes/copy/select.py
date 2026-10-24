# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Implementation selection for ``CopyLibraryNode``."""

import itertools
from typing import TYPE_CHECKING

import dace
from dace import dtypes, symbolic
from dace.codegen.common import get_gpu_backend
from dace.libraries.standard.helper import (
    CPU_RESIDENT_STORAGES,
    collapse_shape_and_strides,
    host_storages,
    is_in_parallel_scope,
    is_parallel_cpu_transfer_size,
)
from dace.libraries.standard.nodes.copy.common import (
    _both_packed_same_layout,
    _is_cross_cpu_gpu,
    cuda2d_pitch_params,
    strided_copy_calls,
)
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode


# More gpuMemcpy calls than this for one strided host/device copy, or a count known only at run time, and the copy is
# staged through a packed buffer: one transfer plus one copy on the device or host side.
STAGED_COPY_MIN_CALLS = 5


def select_copy_implementation(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> str:
    """Resolve ``CopyLibraryNode.implementation`` when set to ``'Auto'`` (the default); never
    returns ``'Auto'`` itself.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: a concrete implementation name from ``CopyLibraryNode.implementations``.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(
        parent_state.sdfg, parent_state, allow_cross_storage=True
    )

    # A 0-D map crashes memlet propagation, so single-element copies use Tasklet/MemcpyCUDA1D.
    single_elt = in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1

    # A cast is not a byte move. No memcpy variant -- host, gpuMemcpyAsync or the 2D/ND forms --
    # can carry one, so a converting copy has to reach a tasklet that performs the conversion.
    if inp.dtype != out.dtype:
        # Inside a kernel the host/device boundary does not exist for the operands in hand, the way
        # the single-element case below already reasons.
        if not is_devicelevel_gpu(parent_state.sdfg, parent_state, node) and _is_cross_cpu_gpu(
            inp.storage, out.storage, node, parent_state
        ):
            raise ValueError(
                f"CopyLibraryNode '{node.name}' converts {inp.dtype} to {out.dtype} across the "
                f"CPU/GPU boundary ({inp.storage} -> {out.storage}). Staging the transfer and "
                f"casting on the device is not implemented yet; keep the cast on one side of "
                f"the boundary, or pick an implementation explicitly."
            )
        return "Tasklet" if single_elt else "MappedTasklet"

    # GPU_Shared: SharedMemoryCollective, unless thread-level (Register endpoint or in a map).
    if inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared:
        thread_level = (
            inp.storage == dtypes.StorageType.Register
            or out.storage == dtypes.StorageType.Register
            or is_in_scope(parent_state.sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock])
        )
        if thread_level:
            return "Tasklet" if single_elt else "MappedTasklet"
        return "SharedMemoryCollective"

    # Single-element non-Shared: MemcpyCUDA1D crossing CPU/GPU or GPU<->GPU from host; else Tasklet.
    if single_elt:
        # Device code cannot issue gpuMemcpyAsync at all, so an in-kernel single-element transfer
        # is a plain assignment whichever storages it spans -- a host scalar reaches the kernel as
        # a by-value argument, and CPU_Pinned is directly device-addressable.
        inside_kernel = is_devicelevel_gpu(parent_state.sdfg, parent_state, node)
        if inside_kernel:
            return "Tasklet"
        if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
            return "MemcpyCUDA1D"
        both_gpu_global = inp.storage == dtypes.StorageType.GPU_Global and out.storage == dtypes.StorageType.GPU_Global
        if both_gpu_global:
            return "MemcpyCUDA1D"
        return "Tasklet"

    # gpuMemcpyAsync can't issue from device code, so in-kernel multi-element copies map instead --
    # but only where the kernel can dereference both ends. Crossing the boundary it cannot: there is
    # no implementation of a host-to-device copy issued from inside a kernel, neither a mapped
    # tasklet nor a Memcpy variant, so the answer is that the copy does not belong in the kernel.
    # Saying so here keeps the refusal at the point the choice is made, instead of returning an
    # implementation that then reports the boundary as its own limitation and suggests a Memcpy
    # variant that is just as impossible in device code.
    if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
            raise ValueError(
                f"No copy implementation crosses the CPU/GPU boundary inside a kernel "
                f"(got {inp.storage} -> {out.storage} for '{node.label}' in state "
                f"'{parent_state.label}'). Device code cannot address host memory and "
                f"cannot issue a Memcpy; place the copy outside the kernel."
            )
        return "MappedTasklet"

    # Host CPU-resident: same-shape/contiguous/same-layout below the parallel-transfer threshold
    # is one MemcpyCPU; otherwise falls through to a parallel MappedTasklet. The threshold only
    # applies where the copy runs once: inside a parallel map the mapped form is sequentialized to
    # an element loop, which is strictly worse than the single call at any size.
    same_shape = len(inp.shape) == len(out.shape) and not any(
        itertools.starmap(symbolic.inequal_symbols, zip(in_subset.size(), out_subset.size()))
    )
    if (
        {inp.storage, out.storage} <= host_storages(node, parent_state)
        and same_shape
        and in_subset.is_contiguous_subset(inp)
        and out_subset.is_contiguous_subset(out)
        and _both_packed_same_layout(inp, out)
        and not (
            is_parallel_cpu_transfer_size(in_subset.num_elements(), parent_state.sdfg.facts())
            and not is_in_parallel_scope(node, parent_state)
        )
    ):
        return "MemcpyCPU"

    # Anything that crosses the boundary is a Memcpy: no mapped tasklet can dereference both ends,
    # so a copy that gets here strided or rank-mismatched belongs to the refinement below, which
    # answers with the pitched 2-D form or a loop of ``cudaMemcpyAsync`` per contiguous chunk.
    # ``Register`` is outside ``allowed`` yet reaches here at host level, where it IS host memory.
    gpu = dtypes.StorageType.GPU_Global
    allowed = CPU_RESIDENT_STORAGES | {dtypes.StorageType.Default, gpu}
    crosses_boundary = _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state)
    both_gpu_or_host = inp.storage in allowed and out.storage in allowed
    impl = (
        "MemcpyCUDA1D"
        if (crosses_boundary or ((inp.storage == gpu or out.storage == gpu) and both_gpu_or_host))
        else None
    )

    if impl == "MemcpyCUDA1D":
        refined = _refine_cuda_impl_for_subsets(node, parent_state)
        if refined is not None:
            impl = refined

    # Rank-mismatched copies (e.g. (2,3,4) -> (8,3)) fall through to the MappedTasklet 1-D walker.
    return impl or "MappedTasklet"


def _packed_orders_differ(inp: dace.data.Data, out: dace.data.Data) -> bool:
    """True if one descriptor is packed C only and the other packed Fortran only.

    A 1-D array is both, and a padded one is neither: each keeps the order the other side has.
    """
    c_in, f_in = inp.is_packed_c_strides(), inp.is_packed_fortran_strides()
    c_out, f_out = out.is_packed_c_strides(), out.is_packed_fortran_strides()
    return (c_in and not f_in and f_out and not c_out) or (f_in and not c_in and c_out and not f_out)


def _refine_cuda_impl_for_subsets(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> str | None:
    """Upgrade ``MemcpyCUDA1D`` to a more specific impl for non-contiguous subsets.

      both subsets contiguous, same element order    -> ``None`` (keep CUDA1D)
      collapsed rank 2, 2D pitched layout matches    -> ``MemcpyCUDA2D``
      collapsed rank 1, both sides equal length      -> ``MemcpyCUDA2D`` (degenerate ``(1, N)``)
      same-side (no CPU/GPU boundary)                -> ``MappedTasklet`` (per-element loop nest)
      cross CPU/GPU, no common stride-1 axis, or
        a symbolic number of memcpy calls, or > 5
        calls host to device or per-row (pooled HIP) -> ``MemcpyCUDAStaged`` (one memcpy, a GPU filter map)
      cross CPU/GPU, same rank, common stride-1 axis -> ``MemcpyCUDANDStrided`` (one pitched 2D/3D call)
      cross CPU/GPU, rank or shape mismatch          -> raise

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: the refined impl name, or ``None`` when both subsets are contiguous (keeps
        ``MemcpyCUDA1D``).
    :raises ValueError: cross-CPU/GPU strided pattern whose sides differ in rank or shape.
    """
    _, inp, in_subset, _, out, out_subset = node.validate(parent_state.sdfg, parent_state, allow_cross_storage=True)

    # One flat gpuMemcpyAsync moves the bytes in order. A C-packed and a Fortran-packed array are both
    # contiguous, but they order their elements differently, so a flat copy between them permutes them.
    if (
        in_subset.is_contiguous_subset(inp)
        and out_subset.is_contiguous_subset(out)
        and not _packed_orders_differ(inp, out)
    ):
        return None

    # HIP refuses a pitched copy out of stream-ordered (pooled) memory whose rows end exactly at the pitch,
    # e.g. the last level of a (species, level, column) array, with "invalid argument". A device-side
    # strided copy needs no gpuMemcpy at all: a copy kernel moves it, as CuPy does for strided device copies.
    if get_gpu_backend() == "hip" and not _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
        return "MappedTasklet"

    # ``MemcpyCUDAHull`` is never chosen here: it is correct only where nothing reads the destination between the rows,
    # which canonicalization proves and pins (``pin_span_copies``).
    crosses_boundary = _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state)

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    # A strided copy that would take a loop of more than STAGED_COPY_MIN_CALLS gpuMemcpy calls -- or a number known only
    # at run time -- moves as a packed buffer instead: one gpuMemcpyAsync and one copy on the strided side. HIP refuses
    # pitched copies of pooled memory whose rows end at the pitch, so there a pooled operand counts one call per row.
    # Device to host, the filter runs on the host: a known loop of pitched calls outruns it, so only a symbolic count
    # or one call per row is staged there.
    pooled_hip = get_gpu_backend() == "hip" and (getattr(inp, "pool", False) or getattr(out, "pool", False))
    src_rank, dst_rank = len(in_shape_collapsed), len(out_shape_collapsed)
    same_shape = src_rank == dst_rank and all(
        symbolic.equal(a, b) is True for a, b in zip(in_shape_collapsed, out_shape_collapsed)
    )
    if crosses_boundary and same_shape:
        calls = dace.symbolic.pystr_to_symbolic(
            strided_copy_calls(in_shape_collapsed, in_strides_collapsed, out_strides_collapsed, pitched=not pooled_hip)
        )
        common_row = any(a == 1 and b == 1 for a, b in zip(in_strides_collapsed, out_strides_collapsed))
        many_calls = not calls.is_number or int(calls) > STAGED_COPY_MIN_CALLS
        to_device = out.storage == dace.dtypes.StorageType.GPU_Global
        if not common_row or not calls.is_number or (many_calls and (to_device or pooled_hip)):
            return "MemcpyCUDAStaged"

    if src_rank == 2 and dst_rank == 2:
        # Shared with the expander so selector and expander cannot disagree.
        if cuda2d_pitch_params(in_shape_collapsed, in_strides_collapsed, out_strides_collapsed) is not None:
            return "MemcpyCUDA2D"

    elif src_rank == 1 and dst_rank == 1:
        # Degenerate (1, N) case: neither side needs stride-1, e.g. `a[:, 1] = b[4, :]` (C order).
        if not symbolic.inequal_symbols(in_shape_collapsed[0], out_shape_collapsed[0]):
            return "MemcpyCUDA2D"

    if not crosses_boundary:
        return "MappedTasklet"

    if (
        len(in_shape_collapsed) == len(out_shape_collapsed)
        and len(in_shape_collapsed) >= 1
        and any(in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1 for d in range(len(in_shape_collapsed)))
    ):
        return "MemcpyCUDANDStrided"

    raise ValueError(
        f"CopyLibraryNode '{node.name}' has a strided cross-CPU/GPU copy pattern that "
        f"cannot be lowered to a single gpuMemcpy or cudaMemcpy2DAsync and has no "
        f"common stride-1 axis for chunked memcpy "
        f"(src_shape={in_shape_collapsed}, src_strides={in_strides_collapsed}, "
        f"dst_shape={out_shape_collapsed}, dst_strides={out_strides_collapsed}); "
        f"pick an explicit implementation manually."
    )
