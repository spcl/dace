# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Thin wrapper around ``InsertExplicitCopies`` that restricts insertion to
copies touching GPU_Global or CPU_Pinned storages -- i.e. the copies that
will be serviced by a GPU memcpy expansion (``CUDA``, ``CUDAHostToDevice``,
``CUDADeviceToHost``, ``CUDA2D``).

The heavy lifting -- detecting direct AccessNode->AccessNode edges and map
staging paths, and replacing them with ``CopyLibraryNode`` instances -- is
done by ``InsertExplicitCopies``.  This pass just preconfigures the storage
filters.
"""
from typing import Any, Dict

from dace import SDFG, dtypes, properties
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

_GPU_STORAGES = {
    dtypes.StorageType.GPU_Global,
}

_CPU_STORAGES = {
    dtypes.StorageType.Default,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
}


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitGPUGlobalMemoryCopies(ppl.Pass):
    """
    Insert ``CopyLibraryNode`` instances for CPU<->GPU data movement --
    edges where one endpoint lives in GPU global memory and the other in
    host-side storage (``Default``/``CPU_Heap``/``CPU_Pinned``/
    ``CPU_ThreadLocal``).  GPU<->GPU and CPU<->CPU transfers are left
    untouched; those are not serviced by the GPU stream pipeline.
    """

    def depends_on(self):
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        # CPU -> GPU
        InsertExplicitCopies(src_locations=_CPU_STORAGES,
                             dst_locations=_GPU_STORAGES).apply_pass(sdfg, pipeline_results)
        # GPU -> CPU
        InsertExplicitCopies(src_locations=_GPU_STORAGES,
                             dst_locations=_CPU_STORAGES).apply_pass(sdfg, pipeline_results)
        return {}
