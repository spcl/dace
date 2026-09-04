# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Implementation selection for ``CopyLibraryNode``.
"""
from typing import Optional, TYPE_CHECKING

import dace
from dace import dtypes, symbolic
from dace.libraries.standard.helper import (CPU_RESIDENT_STORAGES, collapse_shape_and_strides, is_in_parallel_scope,
                                            is_parallel_cpu_transfer_size)
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope
from dace.libraries.standard.nodes.copy.common import (cuda2d_pitch_params, _both_packed_same_layout, _is_cross_cpu_gpu)

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode


def select_copy_implementation(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> str:
    """Resolve ``CopyLibraryNode.implementation`` when set to ``'Auto'`` (the default); never
    returns ``'Auto'`` itself.

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: a concrete implementation name from ``CopyLibraryNode.implementations``.
    """
    inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_state.sdfg,
                                                                        parent_state,
                                                                        allow_cross_storage=True)

    # A 0-D map crashes memlet propagation, so single-element copies use Tasklet/MemcpyCUDA1D.
    single_elt = (in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1)

    # A cast is not a byte move. No memcpy variant -- host, gpuMemcpyAsync or the 2D/ND forms --
    # can carry one, so a converting copy has to reach a tasklet that performs the conversion.
    if inp.dtype != out.dtype:
        # Inside a kernel the host/device boundary does not exist for the operands in hand, the way
        # the single-element case below already reasons.
        if not is_devicelevel_gpu(parent_state.sdfg, parent_state, node) and _is_cross_cpu_gpu(
                inp.storage, out.storage, node, parent_state):
            raise ValueError(f"CopyLibraryNode '{node.name}' converts {inp.dtype} to {out.dtype} across the "
                             f"CPU/GPU boundary ({inp.storage} -> {out.storage}). Staging the transfer and "
                             f"casting on the device is not implemented yet; keep the cast on one side of "
                             f"the boundary, or pick an implementation explicitly.")
        return 'Tasklet' if single_elt else 'MappedTasklet'

    # GPU_Shared: SharedMemoryCollective, unless thread-level (Register endpoint or in a map).
    # TODO: replace dace::CopyND with a vectorized 128-bit collective load.
    if inp.storage == dtypes.StorageType.GPU_Shared or out.storage == dtypes.StorageType.GPU_Shared:
        thread_level = (inp.storage == dtypes.StorageType.Register or out.storage == dtypes.StorageType.Register
                        or is_in_scope(parent_state.sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock]))
        if thread_level:
            return 'Tasklet' if single_elt else 'MappedTasklet'
        return 'SharedMemoryCollective'

    # Single-element non-Shared: MemcpyCUDA1D crossing CPU/GPU or GPU<->GPU from host; else Tasklet.
    if single_elt:
        # Device code cannot issue gpuMemcpyAsync at all, so an in-kernel single-element transfer
        # is a plain assignment whichever storages it spans -- a host scalar reaches the kernel as
        # a by-value argument, and CPU_Pinned is directly device-addressable.
        inside_kernel = is_devicelevel_gpu(parent_state.sdfg, parent_state, node)
        if inside_kernel:
            return 'Tasklet'
        if _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
            return 'MemcpyCUDA1D'
        both_gpu_global = (inp.storage == dtypes.StorageType.GPU_Global
                           and out.storage == dtypes.StorageType.GPU_Global)
        if both_gpu_global:
            return 'MemcpyCUDA1D'
        return 'Tasklet'

    # gpuMemcpyAsync can't issue from device code, so in-kernel multi-element copies map instead.
    if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        return 'MappedTasklet'

    # Host CPU-resident: same-shape/contiguous/same-layout below the parallel-transfer threshold
    # is one MemcpyCPU; otherwise falls through to a parallel MappedTasklet. The threshold only
    # applies where the copy runs once: inside a parallel map the mapped form is sequentialized to
    # an element loop, which is strictly worse than the single call at any size.
    host_storages = CPU_RESIDENT_STORAGES | {dtypes.StorageType.Default}
    same_shape = (len(inp.shape) == len(out.shape)
                  and not any(symbolic.inequal_symbols(a, b) for a, b in zip(in_subset.size(), out_subset.size())))
    if ({inp.storage, out.storage} <= host_storages and same_shape and in_subset.is_contiguous_subset(inp)
            and out_subset.is_contiguous_subset(out) and _both_packed_same_layout(inp, out)
            and not (is_parallel_cpu_transfer_size(in_subset.num_elements())
                     and not is_in_parallel_scope(node, parent_state))):
        return 'MemcpyCPU'

    gpu = dtypes.StorageType.GPU_Global
    allowed = CPU_RESIDENT_STORAGES | {dtypes.StorageType.Default, gpu}
    impl = ('MemcpyCUDA1D' if ((inp.storage == gpu or out.storage == gpu) and inp.storage in allowed
                               and out.storage in allowed) else None)

    if impl == 'MemcpyCUDA1D':
        refined = _refine_cuda_impl_for_subsets(node, parent_state)
        if refined is not None:
            impl = refined

    # Rank-mismatched copies (e.g. (2,3,4) -> (8,3)) fall through to the MappedTasklet 1-D walker.
    return impl or 'MappedTasklet'


def _refine_cuda_impl_for_subsets(node: "CopyLibraryNode", parent_state: dace.SDFGState) -> Optional[str]:
    """Upgrade ``MemcpyCUDA1D`` to a more specific impl for non-contiguous subsets.

      both subsets contiguous                       -> ``None`` (keep CUDA1D)
      collapsed rank 2, 2D pitched layout matches    -> ``MemcpyCUDA2D``
      collapsed rank 1, both sides equal length      -> ``MemcpyCUDA2D`` (degenerate ``(1, N)``)
      same-side (no CPU/GPU boundary)                -> ``MappedTasklet`` (per-element loop nest)
      cross CPU/GPU, same rank, common stride-1 axis -> ``MemcpyCUDANDStrided`` (seq gpuMemcpyAsync/chunk)
      cross CPU/GPU, no common stride-1 axis         -> raise (no ``gpuMemcpy*`` lowering exists;
                                                         host can't issue gpuMemcpyAsync for
                                                         non-contiguous regions, device code can't
                                                         issue it at all)

    :param node: the :class:`CopyLibraryNode` being expanded.
    :param parent_state: state containing ``node``.
    :returns: the refined impl name, or ``None`` when both subsets are contiguous (keeps
        ``MemcpyCUDA1D``).
    :raises ValueError: cross-CPU/GPU strided pattern with no common stride-1 axis.
    """
    _, inp, in_subset, _, out, out_subset = node.validate(parent_state.sdfg, parent_state, allow_cross_storage=True)

    if in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out):
        return None

    in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    src_rank, dst_rank = len(in_shape_collapsed), len(out_shape_collapsed)
    if src_rank == 2 and dst_rank == 2:
        # Shared with the expander so selector and expander cannot disagree.
        if cuda2d_pitch_params(in_shape_collapsed, in_strides_collapsed, out_strides_collapsed) is not None:
            return 'MemcpyCUDA2D'

    elif src_rank == 1 and dst_rank == 1:
        # Degenerate (1, N) case: neither side needs stride-1, e.g. `a[:, 1] = b[4, :]` (C order).
        if not symbolic.inequal_symbols(in_shape_collapsed[0], out_shape_collapsed[0]):
            return 'MemcpyCUDA2D'

    if not _is_cross_cpu_gpu(inp.storage, out.storage, node, parent_state):
        return 'MappedTasklet'

    if (len(in_shape_collapsed) == len(out_shape_collapsed) and len(in_shape_collapsed) >= 1
            and any(in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1
                    for d in range(len(in_shape_collapsed)))):
        return 'MemcpyCUDANDStrided'

    raise ValueError(f"CopyLibraryNode '{node.name}' has a strided cross-CPU/GPU copy pattern that "
                     f"cannot be lowered to a single gpuMemcpy or cudaMemcpy2DAsync and has no "
                     f"common stride-1 axis for chunked memcpy "
                     f"(src_shape={in_shape_collapsed}, src_strides={in_strides_collapsed}, "
                     f"dst_shape={out_shape_collapsed}, dst_strides={out_strides_collapsed}); "
                     f"pick an explicit implementation manually.")
