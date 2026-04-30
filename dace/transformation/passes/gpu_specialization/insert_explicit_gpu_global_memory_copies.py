# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lift transient GPU_Global arrays out of kernel scopes (legacy back-compat
fix for SDFGs that allocate GPU_Global inside ``GPU_Device`` maps), then
lift every implicit copy edge to a ``CopyLibraryNode`` with the ``Auto``
implementation; ``select_copy_implementation`` picks the concrete
expansion at expand-time from endpoint storages and surrounding scope.

Bails out if any ``GPU_Global -> GPU_Global`` transient copy still
survives inside a kernel after the hoist — those are the offenders that
need manual restructuring.
"""
import warnings
from typing import Any, Dict, List

from dace import SDFG, dtypes, properties, nodes, data
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import helpers
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel


def _is_true_scalar(desc) -> bool:
    """Return True if every dimension of ``desc.shape`` is the literal
    integer ``1`` (e.g. ``(1,)``, ``(1, 1)``, ``(1, 1, 1)``)."""
    try:
        for dim in desc.shape:
            if isinstance(dim, int):
                if dim != 1:
                    return False
            elif hasattr(dim, 'is_Integer') and dim.is_Integer:
                if int(dim) != 1:
                    return False
            else:
                return False
        return True
    except Exception:
        return False


def _has_wcr_incoming(sdfg, data_name: str) -> bool:
    """Return True if any memlet in the SDFG writes to ``data_name`` with
    a WCR (write-conflict-resolution = atomic accumulator). Such arrays
    must stay shared across threads — demoting them to Register would
    silently break the accumulation."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for e in state.edges():
                if e.data.wcr is None:
                    continue
                if e.data.data == data_name:
                    return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitGPUGlobalMemoryCopies(ppl.Pass):
    """Hoist transient GPU_Global arrays out of kernel scopes, then lift
    every implicit copy edge to a ``CopyLibraryNode`` (``Auto`` impl).

    The hoist runs ``MoveArrayOutOfKernel`` for each transient GPU_Global
    array found inside a ``GPU_Device`` map. After the hoist the array
    lives in the SDFG that owns the kernel as a non-transient connector
    parameter; the kernel body just passes data through. If any
    transient GPU_Global copy still survives inside the kernel after the
    hoist, the post-hoist guard raises with the offender list."""

    def depends_on(self):
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        self._hoist_transient_gpu_global_out_of_kernels(sdfg)
        self._fail_on_in_kernel_global_global(sdfg)
        # Lift every implicit copy edge — including in-kernel ones. The
        # ``MappedTasklet`` expansion forces ``Sequential`` schedule when
        # already inside a kernel, so we don't get a forbidden GPU_Device-in-
        # GPU_Device nesting.
        InsertExplicitCopies().apply_pass(sdfg, pipeline_results)
        return {}

    @staticmethod
    def _hoist_transient_gpu_global_out_of_kernels(sdfg: SDFG) -> None:
        """Run ``MoveArrayOutOfKernel`` for every transient GPU_Global array
        defined inside a ``GPU_Device`` map.

        Mirrors the existing call site in ``GPUTransformSDFG`` (which only
        runs when a user explicitly applies GPU transformations); placing
        it inside the gpu_specialization pipeline ensures the hoist always
        happens before copies are lifted, regardless of how the caller
        produced the SDFG."""
        transients_in_kernels = set()
        transients_outside = set()

        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = node.desc(parent)
            if not isinstance(desc, data.Array) or not desc.transient:
                continue
            if desc.storage != dtypes.StorageType.GPU_Global:
                continue

            kernel_entry = None
            parent_map_info = helpers.get_parent_map(state=parent, node=node)
            while parent_map_info is not None:
                map_entry, map_state = parent_map_info
                if (isinstance(map_entry, nodes.MapEntry) and map_entry.map.schedule == dtypes.ScheduleType.GPU_Device):
                    kernel_entry = map_entry
                    break
                parent_map_info = helpers.get_parent_map(map_state, map_entry)

            if kernel_entry is not None:
                transients_in_kernels.add((node.data, desc, kernel_entry))
            else:
                transients_outside.add((node.data, desc))

        # Only hoist transients that are *only* defined inside the kernel —
        # if the same (name, desc) pair appears outside, leave the inner
        # one alone (``MoveArrayOutOfKernel`` handles naming for us when it
        # runs).
        to_hoist = set()
        for data_name, desc, kernel_entry in transients_in_kernels:
            if (data_name, desc) in transients_outside:
                continue
            to_hoist.add((data_name, desc, kernel_entry))

        for data_name, desc, kernel_entry in to_hoist:
            # If the transient is a true scalar (every dim literal 1)
            # AND has no incoming WCR memlet (which would indicate a
            # cross-thread atomic accumulator that must stay shared),
            # demote it to Register instead of lifting. A per-thread
            # scalar should never have been GPU_Global; lifting it
            # stretches the shape by the kernel's iteration range and
            # leaks block-index symbols into host-side ``cudaMalloc``
            # size expressions.
            if _is_true_scalar(desc) and not _has_wcr_incoming(sdfg, data_name):
                desc.storage = dtypes.StorageType.Register
                continue
            warnings.warn(f"Transient array '{data_name}' with storage type GPU_Global detected inside kernel "
                          f"{kernel_entry}. GPU_Global memory cannot be allocated within GPU kernels; "
                          f"the array will be lifted outside the kernel as a non-transient GPU_Global array.")
            MoveArrayOutOfKernel().apply_pass(sdfg, kernel_entry, data_name)

    def _fail_on_in_kernel_global_global(self, sdfg: SDFG):
        # A transient GPU_Global array inside a kernel scope cannot be
        # allocated by the codegen (no host-side allocator on that path).
        # Non-transient GPU_Global through-flows are fine — they're
        # connector-bound and the kernel just passes data through them.
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
                    if not (src_desc.storage == dtypes.StorageType.GPU_Global
                            and dst_desc.storage == dtypes.StorageType.GPU_Global):
                        continue
                    if not (src_desc.transient or dst_desc.transient):
                        continue
                    if not (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst)):
                        continue
                    offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state "
                                     f"'{state.label}' (SDFG '{nsdfg.name}')")
        if offenders:
            raise ValueError("Transient GPU_Global arrays cannot live inside a kernel scope. "
                             "Run ``MoveArrayOutOfKernel`` before this pass to hoist them. Offenders:\n" +
                             "\n".join(offenders))
