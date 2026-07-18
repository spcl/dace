# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import Callable, List, Tuple

import dace
from dace import dtypes
from dace.sdfg import nodes

# Both the legacy and experimental codegens consume this exact name for stream wiring.
CURRENT_STREAM_NAME = "__dace_current_stream"

# Register is intentionally in neither set: it resolves by scope (GPU register
# in a device scope, host stack slot otherwise).
GPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
})
CPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
})


def host_accessible_info_storage(storage: dtypes.StorageType) -> dtypes.StorageType:
    """
    Return the storage a cuSOLVER/LAPACK status scalar (``devInfo``) should use so that it stays
    host-checkable. When the matrix operand lives in GPU-resident memory, the status is placed in
    pinned host memory (DMA-reachable from the device, so cuSOLVER can write it via the unified
    address space while the host can still read the result). CPU-resident inputs keep their storage.

    Lives here rather than in ``dace.dtypes`` because it keys off ``GPU_RESIDENT_STORAGES``
    (``{GPU_Global, GPU_Shared}``), which is defined above -- note ``dtypes.GPU_STORAGES`` is a
    *different*, narrower set (``{GPU_Shared}``) and is not a substitute.
    """
    if storage in GPU_RESIDENT_STORAGES:
        return dtypes.StorageType.CPU_Pinned
    return storage


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair.

    Surviving strides are scaled by the subset step (``stride * s``) to describe the access as a
    view into the parent array.

    :param subset: The access range, one ``(begin, end, step)`` per dimension.
    :param strides: The parent array strides, aligned with ``subset``.
    :returns: ``(collapsed_shape, collapsed_strides)`` with singletons removed.
    """
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
    return collapsed_shape, collapsed_strides


def is_parallel_cpu_transfer_size(num_elements: dace.symbolic.SymbolicType) -> bool:
    """Whether a contiguous CPU transfer of ``num_elements`` should take the mapped (parallel) path.

    ``True`` ONLY when the count is a compile-time constant ``>=`` the configurable threshold
    ``compiler.cpu.parallel_transfer_min_elements`` (default 1024). A symbolic size (unknown at
    compile time) stays serial: we do not fork an OpenMP region for a size that may be tiny at
    runtime -- an element map schedules parallel at top level regardless, so the guard is what
    keeps a small/unknown transfer a single libc call.

    :param num_elements: total contiguous element count (constant or symbolic).
    :returns: ``True`` to route to the mapped expansion, ``False`` to keep the single libc call.
    """
    try:
        threshold = int(dace.Config.get('compiler', 'cpu', 'parallel_transfer_min_elements'))
        return int(dace.symbolic.simplify(num_elements)) >= threshold
    except (TypeError, ValueError):
        return False


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one picked by ``select_fn``.

    Sets ``node.implementation`` to the resolved name so introspection reflects what was picked.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :param select_fn: callable returning a concrete implementation name (not ``'Auto'``).
    :param library_cls: the library node class with the ``implementations`` dict.
    :returns: whatever the resolved expansion returns.
    """
    impl_name = select_fn(node, parent_state)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_state.sdfg)
