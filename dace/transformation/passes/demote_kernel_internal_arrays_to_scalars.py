# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DemoteKernelInternalArraysToScalars``, the inverse of the GPU scalar promotion.

That promotion widens a scalar crossing a kernel's output boundary into a length-1 ``Array``; this pass
keeps a value living entirely inside the kernel a ``Scalar``, so codegen emits ``double`` rather than a
needless ``double*``. Scalarization is delegated to ``ConvertLengthOneArraysToScalars``, after which the
stale pointer-typed ``NestedSDFG`` connectors are reset and re-inferred.
"""
from typing import Any, Dict, Optional, Set

from dace import data, dtypes, properties
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import SDFG, infer_types
from dace.sdfg.scope import is_in_scope
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.scalar_promotion import written_by_gpu_map_exit


def inside_gpu_kernel(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` is, transitively, the body of a ``GPU_Device`` map."""
    return is_in_scope(sdfg.parent_sdfg, sdfg.parent, sdfg.parent_nsdfg_node, [dtypes.ScheduleType.GPU_Device])


def all_accesses_within_gpu_kernel(sdfg: SDFG, name: str) -> bool:
    """Whether ``name`` is accessed at least once, and only inside ``GPU_Device`` maps."""
    accesses = [(state, node) for state in sdfg.states() for node in state.data_nodes() if node.data == name]
    return bool(accesses) and all(
        is_in_scope(sdfg, state, node, [dtypes.ScheduleType.GPU_Device]) for state, node in accesses)


def kernel_internal_len1_array(sdfg: SDFG, name: str, desc: data.Data, device_function: bool) -> bool:
    """Whether ``name`` is a kernel-internal single value held in a length-1 array."""
    if not (isinstance(desc, data.Array) and tuple(desc.shape) == (1, )):
        return False
    # GPU-resident memory stays addressable, and a kernel output crosses the GPU_Device boundary.
    if desc.storage in GPU_RESIDENT_STORAGES or written_by_gpu_map_exit(sdfg, name):
        return False
    # Inside a device-function SDFG everything is kernel-internal by construction.
    return device_function or all_accesses_within_gpu_kernel(sdfg, name)


def reset_parent_connectors(sub: SDFG, names: Set[str]) -> None:
    """Reset the parent ``NestedSDFG`` connectors of the scalarized descriptors for re-inference."""
    node = sub.parent_nsdfg_node
    if node is None:
        return
    for connectors in (node.in_connectors, node.out_connectors):
        for name in names & connectors.keys():
            connectors[name] = dtypes.typeclass(None)


@properties.make_properties
@transformation.explicit_cf_compatible
class DemoteKernelInternalArraysToScalars(ppl.Pass):
    """Scalarize kernel-internal length-1 ``Array``s (inverse of the GPU scalar promotion)."""

    def modifies(self) -> ppl.Modifies:
        # Array->Scalar (Descriptors), subset collapse (Memlets), NestedSDFG connector reset (Nodes).
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Demote every kernel-internal length-1 array across the SDFG hierarchy.

        :param sdfg: Root SDFG (modified in place).
        :returns: Number of descriptors demoted, or ``None`` if nothing changed.
        """
        demoted = 0
        for sub in list(sdfg.all_sdfgs_recursive()):
            device_function = inside_gpu_kernel(sub)
            names = {
                name
                for name, desc in sub.arrays.items() if kernel_internal_len1_array(sub, name, desc, device_function)
            }
            if not names:
                continue
            # The conversion itself keeps buffers whose neighbors cannot be rewritten to scalar access.
            converted = ConvertLengthOneArraysToScalars(recursive=False, filter=names).apply_pass(sub, {}) or set()
            reset_parent_connectors(sub, set(converted))
            demoted += len(converted)

        if demoted == 0:
            return None
        # Re-derive the reset connectors, now as scalar references.
        for sub in sdfg.all_sdfgs_recursive():
            infer_types.infer_connector_types(sub)
        return demoted
