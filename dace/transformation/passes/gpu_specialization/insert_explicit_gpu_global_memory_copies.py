# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist transient ``GPU_Global`` arrays out of kernel scopes, then lift every
implicit copy edge to an ``Auto``-impl ``CopyLibraryNode``.

Raises if any transient ``GPU_Global -> GPU_Global`` copy survives inside a
kernel after the hoist -- those need manual restructuring.
"""
import warnings
from typing import Any, Dict, List

from dace import SDFG, dtypes, properties, nodes, data, symbolic
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import helpers
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel

#: Storage a kernel-internal transient can carry that codegen emits as a plain local declaration
#: inside device code. ``Default`` lands there too: inside a kernel scope it resolves to the same
#: thing. ``GPU_Shared`` is deliberately absent -- shared memory has its own sizing rules.
DEVICE_LOCAL_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def has_literal_shape(desc) -> bool:
    """True if every dimension of ``desc`` is a concrete positive integer.

    A symbolic extent cannot size a per-thread array, and it cannot size a stack array in device
    code either: nvcc rejects a VLA outright ("expression must have a constant value"), where the
    host compiler would have accepted one.
    """
    for dim in desc.shape:
        if symbolic.issymbolic(dim):
            return False
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            # Non-symbolic but not a finite integer (e.g. sympy.oo): cannot size a per-thread array.
            return False
        if dim <= 0:
            return False
    return True


def _is_register_demotable(desc, max_elements: int) -> bool:
    """True if ``desc`` is safe and worth demoting to per-thread ``Register``.

    Requires a literal shape (:func:`has_literal_shape` -- a symbol would leak into host-side
    ``cudaMalloc``) and ``prod(shape) <= max_elements`` (larger arrays go through
    ``MoveArrayOutOfKernel`` instead of a per-thread slab).
    """
    if not has_literal_shape(desc):
        return False
    total = 1
    for dim in desc.shape:
        total *= int(dim)
    return total <= max_elements


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

    A post-hoist guard raises with the offender list if any in-kernel transient
    ``GPU_Global`` copy survives.
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
        """Run ``MoveArrayOutOfKernel`` for every transient ``GPU_Global`` array
        defined inside a ``GPU_Device`` map, so the hoist always precedes copy
        lifting regardless of how the SDFG was produced."""
        transients_in_kernels = set()
        transients_outside = set()

        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = node.desc(parent)
            # ``data.View`` subclasses ``data.Array``, but a view is a pointer into another
            # container -- codegen dispatches it to ``allocate_view`` and never allocates it, so
            # there is nothing to hoist. Passing one to ``MoveArrayOutOfKernel`` reshapes the view
            # descriptor and hangs a second access node off the kernel exit, which destroys the
            # unique view edge ``sdutil.get_view_edge`` needs.
            if not isinstance(desc, data.Array) or isinstance(desc, data.View) or not desc.transient:
                continue
            # A symbolically-sized device-local transient is the second thing that cannot stay in
            # the kernel, and it is the one no guard used to catch: codegen emits it as a stack
            # array, and inside device code that is a VLA, which nvcc refuses outright while the
            # host compiler accepts it (npbench bout_arakawa, ``double jpp[(NZ - 2)];``). It has no
            # per-thread form either, so it takes the same route as an in-kernel ``GPU_Global``
            # array: promote it here and let ``MoveArrayOutOfKernel`` give it one slice per map
            # iteration outside the kernel.
            needs_global = (desc.storage == dtypes.StorageType.GPU_Global
                            or (desc.storage in DEVICE_LOCAL_STORAGE and not has_literal_shape(desc)))
            if not needs_global:
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

        # Only hoist transients defined *solely* inside the kernel -- if the same
        # (name, desc) pair also appears outside, leave the inner one alone.
        to_hoist = set()
        for data_name, desc, kernel_entry in transients_in_kernels:
            if (data_name, desc) in transients_outside:
                continue
            to_hoist.add((data_name, desc, kernel_entry))

        for data_name, desc, kernel_entry in to_hoist:
            # Demote small, WCR-free, literal-shape transients to per-thread
            # Register storage (see the two helpers for why each condition is
            # required); anything else falls through to ``MoveArrayOutOfKernel``.
            # Persistent / external transients must not be demoted to Register;
            # the combination is rejected by validation and cannot be allocated
            # as a per-thread variable across SDFG invocations anyway.
            if (desc.lifetime not in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)
                    and _is_register_demotable(desc, self.register_demotion_max_elements)
                    and not _has_wcr_incoming(sdfg, data_name)):
                desc.storage = dtypes.StorageType.Register
                continue
            reason = ("with storage type GPU_Global"
                      if desc.storage == dtypes.StorageType.GPU_Global else f"of symbolic shape {list(desc.shape)}")
            warnings.warn(f"Transient array '{data_name}' {reason} detected inside kernel {kernel_entry}. "
                          f"Neither GPU_Global memory nor a variable-length local array can be allocated "
                          f"within a GPU kernel; the array will be lifted outside the kernel as a "
                          f"non-transient GPU_Global array, one slice per map iteration.")
            desc.storage = dtypes.StorageType.GPU_Global
            MoveArrayOutOfKernel().apply_pass(sdfg, kernel_entry, data_name)

    def _fail_on_in_kernel_global_global(self, sdfg: SDFG):
        # A transient GPU_Global array inside a kernel scope cannot be allocated
        # by the codegen (no host-side allocator there). Non-transient
        # through-flows are fine -- they're connector-bound pass-through.
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
                    # A view edge aliases, it does not copy: no allocation and no transfer, so it
                    # is not an offender and the hoist this guard demands would break it.
                    if isinstance(src_desc, data.View) or isinstance(dst_desc, data.View):
                        continue
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
