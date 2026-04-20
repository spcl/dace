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

_GPU_COPY_STORAGES = {
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_ThreadLocal,
}


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitGPUGlobalMemoryCopies(ppl.Pass):
    """
    Insert ``CopyLibraryNode`` instances for data movement involving GPU
    global memory or pinned host memory.  Kept for backwards-compatibility;
    new code should call ``InsertExplicitCopies`` directly with explicit
    ``src_locations`` / ``dst_locations`` filters.
    """

    def depends_on(self):
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        inner = InsertExplicitCopies(src_locations=_GPU_COPY_STORAGES, dst_locations=_GPU_COPY_STORAGES)
        inner.apply_pass(sdfg, pipeline_results)
        return {}
