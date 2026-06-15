# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DemoteKernelInternalArraysToScalars`` -- the inverse of
:class:`~dace.transformation.passes.promote_gpu_scalars_to_arrays.PromoteGPUScalarsToArrays`.

``PromoteGPUScalarsToArrays`` widens the scalars that *cross* a GPU kernel's
output boundary into length-1 ``Array``s (they must live in addressable global
memory to be returned). This pass enforces the symmetric half of the invariant:
*a single value that lives entirely inside the kernel stays a ``Scalar``*. Any
length-1 ``Array`` that is kernel-internal -- a register-scoped temporary or a
``NestedSDFG`` (device-function) connector that merely mirrors an outer scalar
-- is demoted back to a ``Scalar`` so codegen emits ``double`` / ``double&``
instead of a needless ``double*`` indirection.

A length-1 ``Array`` is demoted when **all** hold:

* its storage is kernel-local (not ``GPU_Global`` / ``GPU_Shared`` -- those are
  genuinely addressable / cross-thread and must stay arrays);
* it is inside a ``GPU_Device`` scope -- either declared in a sub-SDFG that *is*
  a device function (:func:`is_inside_gpu_device_kernel`), or every one of its
  access nodes sits within a ``GPU_Device`` map in its own SDFG;
* it is **not** a kernel output -- it is never written across a ``GPU_Device``
  ``MapExit`` (that write is exactly what ``PromoteGPUScalarsToArrays`` keeps as
  an array).

The scalarization itself (descriptor swap, ``[0]`` accessor stripping, memlet
subset collapse) is delegated to
:class:`~dace.transformation.passes.length_one_array_scalar_conversion.ConvertLengthOneArraysToScalars`
via its ``filter`` knob, which scalarizes a named descriptor regardless of its
``transient`` flag (so a non-transient device-function connector like a
reduction's ``_out`` is covered). After the swap the stale pointer-typed
``NestedSDFG`` connectors are reset and re-inferred so they become scalar
references again.
"""
from typing import Any, Dict, Optional, Set

from dace import data, dtypes, properties
from dace.sdfg import SDFG, infer_types, nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_inside_gpu_device_kernel
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

# Length-1 arrays in these storages are addressable global memory or
# cross-thread shared memory -- demoting them to a register scalar would be
# wrong, so they are always kept as arrays (mirrors ``PromoteGPUScalarsToArrays``
# forcing kernel outputs to ``GPU_Global``).
_KEEP_STORAGES = frozenset({dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared})


def written_by_gpu_map_exit(sdfg: SDFG, name: str) -> bool:
    """``True`` iff ``name`` is written across a GPU-scheduled map's ``MapExit``
    in ``sdfg`` -- i.e. it is a kernel output.

    This is the exact predicate ``PromoteGPUScalarsToArrays`` uses to *promote*
    a scalar; the demotion pass uses it (negated) to leave genuine kernel
    outputs alone.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            for in_edge in state.in_edges(node):
                src = in_edge.src
                if not isinstance(src, nodes.ExitNode):
                    continue
                entry = state.entry_node(src)
                if entry is not None and entry.map.schedule in dtypes.GPU_SCHEDULES:
                    return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class DemoteKernelInternalArraysToScalars(ppl.Pass):
    """Scalarize kernel-internal length-1 ``Array``s (inverse of ``PromoteGPUScalarsToArrays``)."""

    def modifies(self) -> ppl.Modifies:
        # Descriptors (Array -> Scalar), Memlets (subset collapse) and NestedSDFG
        # node connectors (reset to re-infer) -- the last live under ``Nodes``.
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
            device_function = is_inside_gpu_device_kernel(sub)
            names = {
                name
                for name, desc in sub.arrays.items()
                if self._is_kernel_internal_len1_array(sub, name, desc, device_function)
            }
            if not names:
                continue
            # Reuse the scalar conversion; ``filter`` scalarizes the named descriptors
            # regardless of their ``transient`` flag (device-function connectors are
            # non-transient). ``recursive=False`` -- we visit the hierarchy ourselves so
            # each level's kernel-internal set is computed independently.
            ConvertLengthOneArraysToScalars(recursive=False, filter=names).apply_pass(sub, {})
            self._reset_parent_connectors(sub, names)
            demoted += len(names)

        if demoted == 0:
            return None

        # Connectors of demoted device-function descriptors were reset to an
        # uninferred typeclass; re-derive them (now scalar references).
        for sub in sdfg.all_sdfgs_recursive():
            infer_types.infer_connector_types(sub)
        return demoted

    def _is_kernel_internal_len1_array(self, sdfg: SDFG, name: str, desc: data.Data, device_function: bool) -> bool:
        """Whether ``name`` is a kernel-internal single value masquerading as a length-1 array."""
        if not (isinstance(desc, data.Array) and tuple(desc.shape) == (1, )):
            return False
        if desc.storage in _KEEP_STORAGES:
            return False
        # Kernel outputs must stay arrays -- they cross the GPU_Device boundary.
        if written_by_gpu_map_exit(sdfg, name):
            return False
        # A descriptor inside a device-function sub-SDFG is kernel-internal by
        # construction. Otherwise (e.g. a transient declared in the kernel-owning
        # SDFG) every access must sit within a GPU_Device map scope.
        if device_function:
            return True
        return self._all_accesses_within_gpu_device(sdfg, name)

    @staticmethod
    def _all_accesses_within_gpu_device(sdfg: SDFG, name: str) -> bool:
        """``True`` iff every access node for ``name`` lies inside a ``GPU_Device`` map
        scope (and at least one access exists)."""
        seen = False
        for state in sdfg.states():
            scope = state.scope_dict()
            for node in state.nodes():
                if not (isinstance(node, nodes.AccessNode) and node.data == name):
                    continue
                seen = True
                parent = scope[node]
                in_gpu = False
                while parent is not None:
                    if isinstance(parent, nodes.MapEntry) and parent.map.schedule == dtypes.ScheduleType.GPU_Device:
                        in_gpu = True
                        break
                    parent = scope[parent]
                if not in_gpu:
                    return False
        return seen

    @staticmethod
    def _reset_parent_connectors(sub: SDFG, names: Set[str]) -> None:
        """Reset the parent ``NestedSDFG`` connectors that carried the now-scalarized
        descriptors so :func:`infer_connector_types` re-derives them as scalar references."""
        node = sub.parent_nsdfg_node
        if node is None:
            return
        uninferred = dtypes.typeclass(None)
        for nm in names:
            if nm in node.in_connectors:
                node.in_connectors[nm] = uninferred
            if nm in node.out_connectors:
                node.out_connectors[nm] = uninferred
