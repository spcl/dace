# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Tuple

from ordered_set import OrderedSet

from dace import properties
from dace.sdfg import SDFG
from dace.sdfg.utils import require_structured_control_flow
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes import FullMapFusion
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace.transformation.passes.offloading.host_maps import HostMapSpec, host_maps, maps_pinned_by_host_loops
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.transformation.passes.offloading.phases.copy_analysis import CopyAnalysis
from dace.transformation.passes.offloading.phases.copy_insertion import CopyInsertion
from dace.transformation.passes.offloading.phases.schedules import assign_schedules
from dace.transformation.passes.offloading.phases.single_element_copy_optimization import (
    single_element_copies_into_map)
from dace.transformation.passes.offloading.phases.single_element_values import change_single_element_containers
from dace.transformation.passes.offloading.phases.single_iteration_maps import make_size1_map_wrappers


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Decide what runs on the accelerator, and place the host/device copies that follow.

    Top-level maps and library nodes become kernels; a control-flow IR records where each array is
    wanted, so a copy is placed only where that location changes.
    """

    max_iterations = properties.Property(
        dtype=int,
        default=1000,
        desc="Safety bound on the placement fixpoint. Reaching it is a bug, not a workload property: the loop "
        "converges once no state is hybrid and no container changed.")
    verbose = properties.Property(dtype=bool, default=False, desc="Print the host maps, hybrid states and IR.")

    def __init__(self,
                 host_maps: HostMapSpec = False,
                 max_iterations: Optional[int] = None,
                 verbose: Optional[bool] = None,
                 **kwargs: Any) -> None:
        """
        :param host_maps: maps that keep a host schedule so the maps under them become the kernels; see
            :data:`~dace.transformation.passes.offloading.host_maps.HostMapSpec`. Not a ``Property``: a
            ``MapEntry`` cannot round-trip through JSON.
        :param max_iterations: overrides the safety bound on the placement fixpoint.
        :param verbose: print the host maps, hybrid states and IR.
        :note: a map holding a callback is kept on the host whatever ``host_maps`` says.
        """
        super().__init__(**kwargs)
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if verbose is not None:
            self.verbose = verbose
        # make_properties allows a non-Property attribute only with a leading underscore.
        self._host_maps = host_maps

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> OrderedSet[type[ppl.Pass]]:
        return OrderedSet([ControlFlowRaising])

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[OrderedSet[str]]:
        """
        :return: every container left in a GPU storage, qualified by its SDFG's id, or None if there is none.
        """
        require_structured_control_flow(sdfg, 'OffloadToAccelerator')
        # An early return leaves before the end, so its copy-backs need a state of their own on its path.
        entries = helpers.separate_early_returns(sdfg)

        host_map_entries = host_maps(sdfg, self._host_maps)
        if self.verbose and host_map_entries:
            print(f"host maps: {[entry.map.label for entry in host_map_entries]}")
        assign_schedules(sdfg, host_map_entries, maps_pinned_by_host_loops(sdfg))

        analysis, IR, wrapped = self.place(sdfg)
        insertion = CopyInsertion(sdfg, analysis.scopes)
        insertion.apply(IR)
        helpers.remove_empty_return_entries(entries)

        if wrapped:
            ppl.Pipeline([
                FullMapFusion(strict_dataflow=True,
                              perform_vertical_map_fusion=True,
                              perform_horizontal_map_fusion=True)
            ]).apply_pass(sdfg, {})
        single_element_copies_into_map(sdfg)

        helpers.register_kernel_local_transients(sdfg, insertion.placed_on_gpu)
        helpers.refuse_by_value_scalars_the_device_writes(sdfg)
        # A Pipeline reads the result as "did anything change": nothing on the device is None.
        return helpers.device_resident(sdfg) or None

    def place(self, sdfg: SDFG) -> Tuple[CopyAnalysis, OffloadingIRNode, bool]:
        """Analyze, wrap hybrid states and re-type single elements to a fixpoint; also say if anything was wrapped."""
        converted: OrderedSet[str] = OrderedSet()
        wrapped = False
        for _ in range(self.max_iterations):
            analysis = CopyAnalysis(sdfg, helpers.get_sdfg_scope_dict(sdfg))
            IR = analysis.build_ir()
            changed = change_single_element_containers(sdfg, converted)
            converted |= changed
            if self.verbose and analysis.hybrid_states:
                print(f"hybrid states: {[state.label for state in analysis.hybrid_states]}")
            for state in analysis.hybrid_states:
                make_size1_map_wrappers(sdfg, state)
            if not analysis.hybrid_states and not changed:
                if self.verbose:
                    print(f"offloading IR:\n{IR}")
                return analysis, IR, wrapped
            wrapped = wrapped or bool(analysis.hybrid_states)
        raise RuntimeError(f"OffloadToAccelerator did not reach a fixpoint in {self.max_iterations} iterations. "
                           "The placement loop is expected to converge; treat this as a bug rather than raising "
                           "the bound.")
