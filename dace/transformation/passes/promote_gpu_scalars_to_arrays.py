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

from dace import data, dtypes, properties
from dace.sdfg import SDFG, infer_types, nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import written_by_gpu_map_exit
from dace.transformation.passes.scalar_promotion import (invalidate_array_connectors, promote_matching_scalars)


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
class PromoteGPUScalarsToArrays(ppl.Pass):
    """Replace GPU-incompatible ``Scalar`` descriptors with length-1 Arrays."""

    # Register-storage scalars are thread-local; widening would force
    # per-thread ``cudaMalloc`` inside the kernel body.
    _RULE2_EXEMPT_STORAGES = frozenset({dtypes.StorageType.Register})

    non_transient_only = properties.Property(dtype=bool,
                                             default=True,
                                             desc="Rule 2 only promotes non-transient kernel-output scalars. "
                                             "A transient scalar written by a GPU map exit stays a Scalar -- the "
                                             "host never observes the value, so it can live in registers / "
                                             "per-thread stack. Disable to promote every kernel-output scalar.")

    def depends_on(self):
        return {InferDefaultSchedulesAndStorages}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Adding new GPU-storage Scalars (e.g. via library expansion) re-arms
        # the pass; harmless when nothing matches.
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote every GPU-incompatible scalar across the SDFG hierarchy.

        :returns: Number of scalars promoted, or ``None`` if nothing changed.
        """
        promoted = promote_matching_scalars(sdfg, self.needs_promotion, self.storage_for)
        # Unconditional: a connector can be mistyped against an Array inner descriptor that this
        # pass did not create (cuBLAS expansion's ``gpu_streams``), so the reset is still needed
        # when nothing was promoted here.
        invalidate_array_connectors(sdfg)
        return promoted or None

    def needs_promotion(self, sdfg: SDFG, name: str) -> bool:
        """Whether ``name`` is a scalar GPU code generation cannot use as-is."""
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar):
            return False

        # Rule 1: GPU storage is incompatible with Scalar.
        if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            return True

        # Rule 2: kernel output -- written by a GPU map's ``MapExit``.
        if desc.storage in self._RULE2_EXEMPT_STORAGES:
            return False
        if self.non_transient_only and desc.transient:
            return False
        return written_by_gpu_map_exit(sdfg, name)

    def storage_for(self, sdfg: SDFG, name: str) -> dtypes.StorageType:
        """Rule 2 needs real device memory for the kernel write; rule 1 keeps the GPU storage it has."""
        storage = sdfg.arrays[name].storage
        if storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            storage = dtypes.StorageType.GPU_Global
        return storage
