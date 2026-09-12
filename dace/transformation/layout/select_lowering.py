# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Device-aware lowering selection for layout-inserted library nodes: on CPU a ``TensorDot`` prefers TBLIS (native transpose-free contraction) when the library is linkable and the operand dtype is supported, otherwise ``pure``; a ``TensorTranspose``/``LayoutChange`` always gets ``pure`` (TBLIS only does contraction). GPU prefers ``cuTENSOR``, falling back to ``pure`` when it can't build/run for the operands."""
import ctypes.util
import os
from typing import List, Set, Type

import dace
from dace import dtypes
from dace.sdfg import nodes as nd

CPU_IMPL = "pure"
CPU_TENSOR_IMPL = "TBLIS"
GPU_IMPL = "cuTENSOR"


def layout_node_types() -> Set[Type[nd.LibraryNode]]:
    """The library-node types the layout passes insert (imported lazily to avoid import loops)."""
    from dace.libraries.linalg import TensorTranspose, TensorDot
    from dace.libraries.layout.layout_change import LayoutChange
    return {TensorTranspose, TensorDot, LayoutChange}


def cutensor_is_linkable() -> bool:
    """Whether ``libcutensor`` is on the loader path -- a cheap linkability proxy; unknown treated as no."""
    return ctypes.util.find_library("cutensor") is not None


def tblis_is_linkable() -> bool:
    """Whether TBLIS is available: ``TBLIS_ROOT`` points at an install, or ``libtblis`` is on the loader path."""
    return 'TBLIS_ROOT' in os.environ or ctypes.util.find_library("tblis") is not None


def cpu_implementation(node: nd.LibraryNode, descs: List[dace.data.Data], tblis_ok: bool) -> str:
    """TBLIS for a ``TensorDot`` when TBLIS is linkable and every operand dtype is supported
    (float32/float64); otherwise the pure map. TBLIS only implements contraction, so a
    ``TensorTranspose``/``LayoutChange`` always gets ``pure``."""
    from dace.libraries.linalg import TensorDot
    from dace.libraries.linalg.nodes.tensordot import ExpandTBLIS
    if (tblis_ok and descs and isinstance(node, TensorDot)
            and all(desc.dtype.base_type in ExpandTBLIS.TYPE_MAP for desc in descs)):
        return CPU_TENSOR_IMPL
    return CPU_IMPL


def operand_descriptors(node: nd.LibraryNode, state: dace.SDFGState) -> List[dace.data.Data]:
    """The data descriptors on ``node``'s in/out edges (its relayout operands)."""
    sdfg = state.sdfg
    descs: List[dace.data.Data] = []
    for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
        if edge.data is not None and edge.data.data is not None and edge.data.data in sdfg.arrays:
            descs.append(sdfg.arrays[edge.data.data])
    return descs


def gpu_implementation(descs: List[dace.data.Data], cutensor_ok: bool) -> str:
    """``cuTENSOR`` when it can build and run for these operands (right dtype, GPU-resident), else ``pure``."""
    if not cutensor_ok or not descs:
        return CPU_IMPL
    from dace.libraries.linalg.environments import cuTensor
    on_gpu = all(desc.storage == dtypes.StorageType.GPU_Global for desc in descs)
    dtypes_supported = all(desc.dtype.base_type in cuTensor.TYPE_MAP for desc in descs)
    return GPU_IMPL if (on_gpu and dtypes_supported) else CPU_IMPL


def select_layout_lowering(sdfg: dace.SDFG, device: str = "cpu") -> int:
    """Assign a device-appropriate implementation to layout library nodes with no implementation set yet."""
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    types = tuple(layout_node_types())
    cutensor_ok = cutensor_is_linkable() if device == "gpu" else False
    tblis_ok = tblis_is_linkable() if device == "cpu" else False
    count = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, types) and node.implementation is None:
            if device == "gpu":
                node.implementation = gpu_implementation(operand_descriptors(node, state), cutensor_ok)
            else:
                node.implementation = cpu_implementation(node, operand_descriptors(node, state), tblis_ok)
            count += 1
    return count
