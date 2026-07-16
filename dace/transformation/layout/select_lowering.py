# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Device-aware lowering selection for layout-inserted library nodes.

The layout transformations (``PermuteDimensions``, ``GemmToTensorDot``, ``RewriteCopyForLayout``)
and the ``LayoutChange`` node describe a relayout SEMANTICALLY; they must not bake in a
device-specific library lowering. Choosing the expansion is a separate, device-driven step that
runs just before compilation -- this module.

On CPU the ``pure`` (map) expansion is selected. On GPU the ``cuTENSOR`` expansion is PREFERRED, but
only where it can actually build and run: the cuTENSOR library must be linkable, the operand dtype
must be one cuTENSOR supports, and the operands must be GPU-resident. Where any of those does not
hold the ``pure`` expansion is selected instead -- it still runs on the GPU when the arrays are
``GPU_Global`` (a GPU map), so a GPU sweep degrades gracefully on a box without cuTENSOR rather than
failing every libnode candidate. The fallback is done HERE, in the selection step, because the
library nodes do not all fall back on their own: ``TensorTranspose``'s cuTENSOR expansion drops to
the pure map for unsupported dtypes, but ``TensorDot``'s raises -- so relying on the node would break
the Gemm path. Only nodes whose implementation is still unset are touched, so an explicitly pinned
lowering (e.g. an ``HPTT`` transpose the caller chose) is preserved.
"""
import ctypes.util
from typing import List, Set, Type

import dace
from dace import dtypes
from dace.sdfg import nodes as nd

CPU_IMPL = "pure"
GPU_IMPL = "cuTENSOR"


def layout_node_types() -> Set[Type[nd.LibraryNode]]:
    """The library-node types the layout passes insert (imported lazily to avoid import loops)."""
    from dace.libraries.linalg import TensorTranspose, TensorDot
    from dace.libraries.layout.layout_change import LayoutChange
    return {TensorTranspose, TensorDot, LayoutChange}


def cutensor_is_linkable() -> bool:
    """Whether ``libcutensor`` is findable on the loader path -- a cheap proxy for whether a cuTENSOR
    expansion will link. Conservative: if it cannot be found, the pure GPU map is used instead."""
    return ctypes.util.find_library("cutensor") is not None


def operand_descriptors(node: nd.LibraryNode, state: dace.SDFGState) -> List[dace.data.Data]:
    """The data descriptors on ``node``'s in/out edges (its relayout operands)."""
    sdfg = state.sdfg
    descs: List[dace.data.Data] = []
    for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
        if edge.data is not None and edge.data.data is not None and edge.data.data in sdfg.arrays:
            descs.append(sdfg.arrays[edge.data.data])
    return descs


def gpu_implementation(descs: List[dace.data.Data], cutensor_ok: bool) -> str:
    """``cuTENSOR`` when it can build and run for these operands, else the pure GPU map.

    cuTENSOR requires: the library linkable, every operand a dtype it supports, and the operands
    GPU-resident (it operates on device memory). Any miss falls back to ``pure``, which still emits a
    GPU map for ``GPU_Global`` operands."""
    if not cutensor_ok or not descs:
        return CPU_IMPL
    from dace.libraries.linalg.environments import cuTensor
    on_gpu = all(desc.storage == dtypes.StorageType.GPU_Global for desc in descs)
    dtypes_supported = all(desc.dtype.base_type in cuTensor.TYPE_MAP for desc in descs)
    return GPU_IMPL if (on_gpu and dtypes_supported) else CPU_IMPL


def select_layout_lowering(sdfg: dace.SDFG, device: str = "cpu") -> int:
    """Assign a device-appropriate implementation to every layout library node whose
    implementation is still unset.

    :param sdfg: the SDFG to patch in place (recurses into nested SDFGs).
    :param device: ``"cpu"`` (``pure`` expansion) or ``"gpu"`` (``cuTENSOR`` where it can build and
                   run, else the pure GPU map -- see :func:`gpu_implementation`).
    :returns: the number of nodes whose implementation was set.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    types = tuple(layout_node_types())
    cutensor_ok = cutensor_is_linkable() if device == "gpu" else False
    count = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, types) and node.implementation is None:
            if device == "gpu":
                node.implementation = gpu_implementation(operand_descriptors(node, state), cutensor_ok)
            else:
                node.implementation = CPU_IMPL
            count += 1
    return count
