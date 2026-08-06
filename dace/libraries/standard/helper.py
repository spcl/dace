# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions."""
from typing import Callable, List, Tuple

import dace
from dace import dtypes
from dace.sdfg import nodes

# Both legacy and experimental codegens consume this exact name for stream wiring.
CURRENT_STREAM_NAME = "__dace_current_stream"

# Register is intentionally in neither set: resolves by scope (GPU register in a device
# scope, host stack slot otherwise).
GPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
})
CPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
})


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair -> ``(collapsed_shape,
    collapsed_strides)``. Surviving strides are scaled by the subset step (``stride * s``) to
    describe the access as a view into the parent array."""
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

    ``True`` only when the count is a compile-time constant ``>=`` the configurable
    ``compiler.cpu.parallel_transfer_min_elements`` (default 1024). Symbolic (unknown) size stays
    serial -- we don't fork an OpenMP region for a size that may be tiny at runtime; an element
    map schedules parallel regardless, so this guard is what keeps a small/unknown transfer a
    single libc call.
    """
    try:
        threshold = int(dace.Config.get('compiler', 'cpu', 'parallel_transfer_min_elements'))
        return int(dace.symbolic.simplify(num_elements)) >= threshold
    except (TypeError, ValueError):
        return False


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one picked by ``select_fn``.
    Sets ``node.implementation`` to the resolved name so introspection reflects what was picked."""
    impl_name = select_fn(node, parent_state)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_state.sdfg)
