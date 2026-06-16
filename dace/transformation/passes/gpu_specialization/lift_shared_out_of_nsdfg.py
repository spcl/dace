# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift ``GPU_Shared`` transients out of nested SDFGs into the SDFG owning
the enclosing ``GPU_Device`` map.

``__shared__`` is only valid inside a CUDA kernel; a Shared transient buried
in an inner NestedSDFG escapes the ``__global__`` function (the framecode
allocation walker loses the kernel-home signal), leaving an undeclared
identifier. This pass promotes the descriptor to the kernel-owning SDFG,
wires it through the NestedSDFG via connectors, and adds kernel
``MapEntry``/``MapExit`` dependency edges to pin allocation to the kernel.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, SDFGState, dtypes, properties, nodes
from dace.memlet import Memlet
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (dependency_edge, innermost_enclosing_map)
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)


@properties.make_properties
@transformation.explicit_cf_compatible
class LiftSharedOutOfNestedSDFG(ppl.Pass):
    """Promote every ``GPU_Shared`` transient in a nested SDFG inside a
    ``GPU_Device`` map up to the kernel-owning SDFG, wired through the NSDFG
    via connectors with kernel entry/exit dependency edges."""

    def depends_on(self):
        # ``InsertExplicitGPUGlobalMemoryCopies`` must run first: it lifts
        # AccessNode->AccessNode Shared edges into ``CopyLibraryNode``s;
        # without it, Shared transients used only on a copy edge never
        # surface as ``transient=True`` descriptors.
        return {InsertExplicitGPUGlobalMemoryCopies}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        lifted = 0
        worklist: List[Tuple[SDFG, SDFGState, nodes.NestedSDFG, nodes.MapEntry]] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for n in state.nodes():
                    if not isinstance(n, nodes.NestedSDFG):
                        continue
                    kernel_entry = innermost_enclosing_map(state, n, dtypes.ScheduleType.GPU_Device)
                    if kernel_entry is None:
                        continue
                    worklist.append((nsdfg, state, n, kernel_entry))

        for outer_sdfg, outer_state, nsdfg_node, kernel_entry in worklist:
            inner_sdfg: SDFG = nsdfg_node.sdfg
            shared_names = [
                name for name, desc in inner_sdfg.arrays.items()
                if desc.transient and desc.storage == dtypes.StorageType.GPU_Shared
            ]
            for name in shared_names:
                if self._lift_one(name, inner_sdfg, nsdfg_node, outer_sdfg, outer_state, kernel_entry):
                    lifted += 1

        return {'lifted': lifted} if lifted > 0 else None

    def _lift_one(self, name: str, inner_sdfg: SDFG, nsdfg_node: nodes.NestedSDFG, outer_sdfg: SDFG,
                  outer_state: SDFGState, kernel_entry: nodes.MapEntry) -> bool:
        """Promote ``name`` and wire it through ``nsdfg_node``::

            MapEntry --(empty, dep)--> AN_read --(in:name)--> NSDFG
            NSDFG --(out:name)--> AN_write --(empty, dep)--> MapExit

        Separate read/write ``AccessNode``s keep the state acyclic when the
        inner SDFG mutates the array (DaCe rejects a single-AN read+write
        cycle around an NSDFG). ``force=True`` is needed because the name
        appears in both in- and out-connectors (the inout pattern).

        Returns ``False`` (lift skipped) when the inner transient is unused:
        a bare descriptor move with no edges/connectors would corrupt the
        SDFG."""
        is_read, is_written = _classify_inner_usage(inner_sdfg, name)
        if not is_read and not is_written:
            return False  # unused: lifting without edges/connectors corrupts the SDFG

        inner_desc = inner_sdfg.arrays[name]

        # find_new_name=True returns the (possibly renamed) unique name, so the lift never
        # overwrites an existing outer descriptor -- no hand-rolled name search needed.
        outer_name = outer_sdfg.add_datadesc(name, inner_desc, find_new_name=True)
        inner_param_desc = copy.deepcopy(inner_desc)
        inner_param_desc.transient = False
        del inner_sdfg.arrays[name]
        inner_sdfg.add_datadesc(name, inner_param_desc)

        full_subset = Range.from_array(inner_desc)
        kernel_exit = outer_state.exit_node(kernel_entry)
        an_write: Optional[nodes.AccessNode] = None

        if is_read:
            an_read = outer_state.add_access(outer_name)
            outer_state.add_edge(kernel_entry, None, an_read, None, dependency_edge())
            nsdfg_node.add_in_connector(name, force=True)
            outer_state.add_edge(an_read, None, nsdfg_node, name,
                                 Memlet(data=outer_name, subset=copy.deepcopy(full_subset)))

        if is_written:
            an_write = outer_state.add_access(outer_name)
            nsdfg_node.add_out_connector(name, force=True)
            outer_state.add_edge(nsdfg_node, name, an_write, None,
                                 Memlet(data=outer_name, subset=copy.deepcopy(full_subset)))
            outer_state.add_edge(an_write, None, kernel_exit, None, dependency_edge())

        # Write-only: AN_write has no incoming dep from MapEntry, so anchor it.
        if is_written and not is_read:
            outer_state.add_edge(kernel_entry, None, an_write, None, dependency_edge())

        # Topology changed: drop the scope cache so a sibling ``_lift_one``
        # in the same state doesn't read it stale.
        outer_state._clear_scopedict_cache()
        return True


def _classify_inner_usage(inner_sdfg: SDFG, name: str) -> Tuple[bool, bool]:
    """``(is_read, is_written)`` for ``name`` inside ``inner_sdfg``, from
    each state's ``read_and_write_sets``."""
    is_read = False
    is_written = False
    for state in inner_sdfg.states():
        read_set, write_set = state.read_and_write_sets()
        if name in read_set:
            is_read = True
        if name in write_set:
            is_written = True
        if is_read and is_written:
            return True, True
    return is_read, is_written
