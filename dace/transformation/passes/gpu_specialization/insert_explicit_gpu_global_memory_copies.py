# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift transient ``GPU_Global`` arrays out of kernel scopes (legacy
back-compat for SDFGs allocating ``GPU_Global`` inside ``GPU_Device`` maps),
then lift every implicit copy edge to an ``Auto``-impl ``CopyLibraryNode``.

Raises if any transient ``GPU_Global -> GPU_Global`` copy still survives
inside a kernel after the hoist -- those need manual restructuring.
"""
import warnings
from typing import Any, Dict, List

from dace import SDFG, dtypes, properties, nodes, data
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import helpers
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel


def _is_register_demotable(desc, max_elements: int) -> bool:
    """True if ``desc`` is safe and worth demoting to per-thread ``Register``.

    Requires every shape dim to be a concrete positive integer (a symbol
    would leak into host-side ``cudaMalloc`` and cannot size a per-thread
    array) and ``prod(shape) <= max_elements`` (larger arrays go through
    ``MoveArrayOutOfKernel`` instead of a per-thread slab).
    """
    total = 1
    try:
        for dim in desc.shape:
            if isinstance(dim, int) and dim > 0:
                total *= dim
            elif hasattr(dim, 'is_Integer') and dim.is_Integer and int(dim) > 0:
                total *= int(dim)
            else:
                return False
        return total <= max_elements
    except Exception:
        return False


def _has_wcr_incoming(sdfg, data_name: str) -> bool:
    """True if any memlet writes ``data_name`` with a WCR (atomic accumulator).

    Such arrays must stay shared -- demoting to Register would silently
    break the accumulation.
    """
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
    """Hoist transient ``GPU_Global`` arrays out of kernel scopes, then lift every implicit copy.

    Implicit copy edges become ``Auto``-impl ``CopyLibraryNode``s. The
    hoist runs ``MoveArrayOutOfKernel`` per transient ``GPU_Global``
    array inside a ``GPU_Device`` map; afterwards the array is a
    non-transient connector parameter on the kernel-owning SDFG. A
    post-hoist guard raises with the offender list if any in-kernel
    transient ``GPU_Global`` copy survives.
    """

    register_demotion_max_elements = properties.Property(
        dtype=int,
        default=64,
        desc="Max ``prod(shape)`` for a literal-shape kernel-internal "
        "transient to be demoted from GPU_Global to per-thread Register "
        "storage. Larger transients fall through to MoveArrayOutOfKernel.",
    )

    def __init__(self, register_demotion_max_elements: int = 64):
        super().__init__()
        self.register_demotion_max_elements = register_demotion_max_elements

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        self._hoist_transient_gpu_global_out_of_kernels(sdfg)
        self._fail_on_in_kernel_global_global(sdfg)
        # Lift every implicit copy edge -- including in-kernel ones. The
        # ``MappedTasklet`` expansion forces ``Sequential`` schedule when
        # already inside a kernel, so we don't get a forbidden GPU_Device-in-
        # GPU_Device nesting.
        InsertExplicitCopies().apply_pass(sdfg, pipeline_results)
        return {}

    def _hoist_transient_gpu_global_out_of_kernels(self, sdfg: SDFG):
        """Run ``MoveArrayOutOfKernel`` for every transient ``GPU_Global``
        array defined inside a ``GPU_Device`` map.

        Mirrors the ``GPUTransformSDFG`` call site but runs inside the
        gpu_specialization pipeline so the hoist always precedes copy
        lifting regardless of how the SDFG was produced."""
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

        # Only hoist transients that are *only* defined inside the kernel --
        # if the same (name, desc) pair appears outside, leave the inner
        # one alone (``MoveArrayOutOfKernel`` handles naming for us when it
        # runs).
        to_hoist = set()
        for data_name, desc, kernel_entry in transients_in_kernels:
            if (data_name, desc) in transients_outside:
                continue
            to_hoist.add((data_name, desc, kernel_entry))

        for data_name, desc, kernel_entry in to_hoist:
            # Demote to per-thread Register storage if the transient is
            # safe to make thread-local:
            #   * literal shape with ``prod(shape) <=
            #     register_demotion_max_elements`` (a symbolic dim would
            #     leak into host-side ``cudaMalloc`` size expressions on
            #     the lift path, which is the failure mode this gate
            #     avoids);
            #   * no incoming WCR memlet (a cross-thread atomic
            #     accumulator must stay shared -- per-thread registers
            #     would silently drop the accumulation).
            # Anything else falls through to ``MoveArrayOutOfKernel``.
            if (_is_register_demotable(desc, self.register_demotion_max_elements)
                    and not _has_wcr_incoming(sdfg, data_name)):
                desc.storage = dtypes.StorageType.Register
                continue
            warnings.warn(f"Transient array '{data_name}' with storage type GPU_Global detected inside kernel "
                          f"{kernel_entry}. GPU_Global memory cannot be allocated within GPU kernels; "
                          f"the array will be lifted outside the kernel as a non-transient GPU_Global array.")
            MoveArrayOutOfKernel().apply_pass(sdfg, kernel_entry, data_name)

    def _fail_on_in_kernel_global_global(self, sdfg: SDFG):
        # A transient GPU_Global array inside a kernel scope cannot be
        # allocated by the codegen (no host-side allocator on that path).
        # Non-transient GPU_Global through-flows are fine -- they're
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
