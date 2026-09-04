# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteGPUScalarsToArrays`` -- replace GPU-incompatible ``Scalar``
descriptors with length-1 ``Array`` descriptors. Runs after storage/schedule
inference (depends on ``InferDefaultSchedulesAndStorages``).

Two rules: (1) a ``Scalar`` with ``GPU_Global``/``GPU_Shared`` storage is
widened to length-1 keeping its storage; (2) a ``Scalar`` written by a GPU
map's ``MapExit`` (kernel output) is widened and forced to ``GPU_Global``.
Bare-identifier references to a promoted name are subscripted ``name[0]`` in
interstate/loop/branch code slots and ``symbol_mapping`` values; memlets are
left intact since a ``Scalar`` access already carries subset ``[0]``.
"""
from typing import Any, Dict, Optional

from dace import properties
from dace.sdfg import SDFG, infer_types, nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """Pipeline-shaped wrapper around
    :func:`dace.sdfg.infer_types.set_default_schedule_and_storage_types`.

    Exists so the call can participate in a ``Pipeline`` with a real
    ``depends_on`` edge: ``PromoteGPUScalarsToArrays`` relies on every
    descriptor having a final, non-default storage decision.
    """

    def modifies(self) -> ppl.Modifies:
        # Storage lives on descriptors, schedule on ``Map`` nodes.
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _schedules_and_storages(sdfg: SDFG) -> Dict[Any, Any]:
        """Snapshot every slot ``set_default_schedule_and_storage_types`` can write.

        Those are descriptor storages (keyed by ``(SDFG, name)``) and the schedules of scope
        entry nodes / library nodes. The inference function reports nothing about what it
        resolved, so the pass diffs a before/after snapshot. Exit nodes are skipped: their
        schedule proxies the same :class:`~dace.sdfg.nodes.Map` as the matching entry node.
        """
        snapshot: Dict[Any, Any] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for name, desc in nsdfg.arrays.items():
                snapshot[(nsdfg, name)] = desc.storage
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, (nodes.EntryNode, nodes.LibraryNode)):
                snapshot[node] = node.schedule
        return snapshot

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Resolve every ``Default`` schedule and storage in the SDFG hierarchy.

        :returns: Number of schedule/storage slots whose value changed, or ``None`` if none did.
        """
        before = self._schedules_and_storages(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        after = self._schedules_and_storages(sdfg)

        changed = sum(1 for key, value in after.items() if before.get(key) != value)
        return changed or None


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteGPUScalarsToArrays(PromoteScalarOutputsToArrays):
    """:class:`~dace.transformation.passes.scalar_promotion.PromoteScalarOutputsToArrays` under the
    GPU criteria, plus the storage inference it needs to see a final storage decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = True

    def depends_on(self):
        return {InferDefaultSchedulesAndStorages}
