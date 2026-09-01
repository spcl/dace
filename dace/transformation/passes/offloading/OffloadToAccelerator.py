
from dace import properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes import FullMapFusion

from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase1_schedules import SchedulePhase
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase2_copy_analysis import CopyAnalysisPhase
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase3_single_element_values import SingleElementValuePhase
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase4_single_iteration_maps import SingleIterationMapPhase
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase6_copy_insertion import CopyInsertionPhase
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offl_phase7b_single_element_copy_optimization import SingleElementCopyOptimization
from dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers import get_sdfg_scope_dict

from typing import Any, Dict, Optional


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    
    CATEGORY: str = 'Offload To Accelerator'
    MAX_ITERATIONS : int = 10
    VERBOSE = False

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        cached_scopes = get_sdfg_scope_dict(sdfg) # cache the result of an expensive operation

        # Phase 1: set sequential / GPU schedules
        SchedulePhase().apply(sdfg, cached_scopes, verbose=OffloadToAccelerator.VERBOSE)

        # Fix Point Iteration of Phases 2 - 4
        changed_containers = set()
        maps_changed = False
        for _ in range(OffloadToAccelerator.MAX_ITERATIONS):

            # Phase 2: build intermediate representation and find hybrid states
            hybrid_states = set()
            IRep = CopyAnalysisPhase().apply(sdfg, hybrid_states, cached_scopes, verbose=OffloadToAccelerator.VERBOSE)

            # Phase 3: decide if single-element values are stored in Scalars or in length-one Arrays
            new_changed_containers = SingleElementValuePhase().apply(sdfg, exceptions=changed_containers, verbose=OffloadToAccelerator.VERBOSE)
            changed_containers |= new_changed_containers

            # Phase 4: resolve hybrid states into pure GPU states by inserting single-iteration maps 
            if hybrid_states:
                SingleIterationMapPhase().apply(sdfg, hybrid_states, verbose=OffloadToAccelerator.VERBOSE)
                maps_changed = True

            # Phase 5: iterate until the SDFG reaches a fixpoint
            if hybrid_states or new_changed_containers: # sdfg has been changed
                cached_scopes = get_sdfg_scope_dict(sdfg)
                continue # repeat
            break

        else:
            raise RuntimeError(f"OffloadToAccelerator pass has reached max. iterations (OffloadToAccelerator.MAX_ITERATIONS = {OffloadToAccelerator.MAX_ITERATIONS}) without conclusive result. Increase limit if this is not a mistake.")

        # Phase 6: insert explicit host-device copies into the SDFG based on the IR
        CopyInsertionPhase().apply(sdfg, IRep, verbose=OffloadToAccelerator.VERBOSE)

        # Phase 7: post-optimization
        # post-optimization 1
        if maps_changed:
            mapfusion_pass = FullMapFusion(
                strict_dataflow=True,
                perform_vertical_map_fusion=True,
                perform_horizontal_map_fusion=True,
            )
            mapfusion_pipeline = ppl.Pipeline([mapfusion_pass])
            mapfusion_pipeline.apply_pass(sdfg, {})

        # post-optimization 2
        SingleElementCopyOptimization().apply(sdfg, verbose=OffloadToAccelerator.VERBOSE)
