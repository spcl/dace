# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Implementation selection for ``FillLibraryNode``."""
from typing import TYPE_CHECKING

import dace
from dace.libraries.standard.helper import (CPU_RESIDENT_STORAGES, is_in_parallel_scope, is_parallel_cpu_transfer_size)
from dace.libraries.standard.nodes.fill.common import byte_pattern
from dace.sdfg.scope import is_devicelevel_gpu

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


def select_fill_implementation(node: "FillLibraryNode", parent_state: dace.SDFGState) -> str:
    """Resolve an ``'Auto'`` ``FillLibraryNode`` implementation to a concrete one.

    ``'pure'``: device scope (no ``gpuMemsetAsync`` from a kernel), non-contiguous subsets, a GPU
    destination whose constant value is not byte-splat, a dynamic value wider than 32 bits, or a
    contiguous CPU fill that is not provably sub-threshold. ``'CUDA'``: host-issued byte-splat fill
    of GPU memory, or a dynamic value of at most 32 bits. ``'CPU'``: a single ``memset``/
    ``std::fill_n``. ``'tasklet'``: a single element.

    :param node: The fill library node being expanded.
    :param parent_state: The state containing ``node``.
    :returns: One of ``'pure'``, ``'CUDA'``, ``'CPU'``, or ``'tasklet'``.
    """
    _out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
    value_info = node.value_descriptor(parent_state)
    is_dynamic = value_info is not None

    if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        if out_subset.num_elements_exact() == 1:
            return 'tasklet'
        return 'pure'

    if out_subset.num_elements_exact() == 1 and (out.storage in CPU_RESIDENT_STORAGES
                                                 or out.storage == dace.dtypes.StorageType.Register):
        return 'tasklet'

    if not out_subset.is_contiguous_subset(out):
        return 'pure'

    if is_dynamic:
        # A dynamic value wider than 32 bits has no single-call runtime-API memset on either host or
        # GPU; route it to the parallel map-based fill.
        if out.dtype.bytes > 4:
            return 'pure'
        if out.storage == dace.dtypes.StorageType.GPU_Global:
            return 'CUDA'
        # Contiguous CPU/Default/Register destination with a dynamic <=32-bit value.
        allowed = CPU_RESIDENT_STORAGES | {dace.dtypes.StorageType.Default}
        if out.storage in allowed and not (is_parallel_cpu_transfer_size(out_subset.num_elements())
                                           and not is_in_parallel_scope(node, parent_state)):
            return 'CPU'
        return 'pure'

    if out.storage == dace.dtypes.StorageType.GPU_Global:
        # gpuMemsetAsync writes ONE byte over the range; dace links only the CUDA runtime API, and
        # the 32-bit setter (cuMemsetD32Async) is driver-API. A non-byte-splat value fills by kernel.
        return 'CUDA' if byte_pattern(node.value, out.dtype) is not None else 'pure'

    # CPU main-memory fill: the element map ('pure') is the DEFAULT, taken unless the count is
    # PROVABLY below parallel_transfer_min_elements, which keeps the single call ('CPU'). A symbolic
    # count is assumed big. Inside a parallel map the element map is sequentialized anyway, so the
    # single call wins there at any size.
    allowed = CPU_RESIDENT_STORAGES | {dace.dtypes.StorageType.Default}
    if (out.storage in allowed and is_parallel_cpu_transfer_size(out_subset.num_elements())
            and not is_in_parallel_scope(node, parent_state)):
        return 'pure'
    return 'CPU'
