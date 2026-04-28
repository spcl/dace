# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Thin wrapper around ``InsertExplicitCopies``. Lifts every implicit copy edge
to a ``CopyLibraryNode`` with the ``Auto`` implementation;
``select_copy_implementation`` picks the concrete expansion at expand-time
from endpoint storages and surrounding scope.

Bails out before lifting if any ``GPU_Global -> GPU_Global`` direct copy
survives inside a kernel — that means a transient leaked into kernel scope
and ``MoveTransientOutOfKernel`` should have run first.
"""
from typing import Any, Dict, List

from dace import SDFG, dtypes, properties, nodes
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitGPUGlobalMemoryCopies(ppl.Pass):
    """Lift every implicit copy edge to a ``CopyLibraryNode`` (``Auto`` impl).
    Errors if a ``GPU_Global -> GPU_Global`` copy is found inside kernel
    scope — that pattern means ``MoveTransientOutOfKernel`` was skipped."""

    def depends_on(self):
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        self._fail_on_in_kernel_global_global(sdfg)
        InsertExplicitCopies().apply_pass(sdfg, pipeline_results)
        return {}

    def _fail_on_in_kernel_global_global(self, sdfg: SDFG):
        offenders: List[str] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in state.edges():
                    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                        continue
                    if edge.data.is_empty() or edge.data.wcr is not None:
                        continue
                    src_desc = nsdfg.arrays[edge.src.data]
                    dst_desc = nsdfg.arrays[edge.dst.data]
                    if (src_desc.storage == dtypes.StorageType.GPU_Global
                            and dst_desc.storage == dtypes.StorageType.GPU_Global and
                        (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst))):
                        offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state "
                                         f"'{state.label}' (SDFG '{nsdfg.name}')")
        if offenders:
            raise ValueError("GPU_Global -> GPU_Global copies inside a kernel scope are not supported. "
                             "Run ``MoveTransientOutOfKernel`` before this pass to hoist these transients "
                             "out of the kernel. Offenders:\n" + "\n".join(offenders))
