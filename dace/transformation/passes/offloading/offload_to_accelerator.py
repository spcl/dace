# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from ordered_set import OrderedSet

from dace import properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes import FullMapFusion

from dace.transformation.passes.offloading.phases.schedules import SchedulePhase
from dace.transformation.passes.offloading.phases.copy_analysis import CopyAnalysisPhase
from dace.transformation.passes.offloading.phases.single_element_values import SingleElementValuePhase
from dace.transformation.passes.offloading.phases.single_iteration_maps import SingleIterationMapPhase
from dace.transformation.passes.offloading.phases.copy_insertion import CopyInsertionPhase
from dace.transformation.passes.offloading.phases.single_element_copy_optimization import SingleElementCopyOptimization
from dace.transformation.passes.offloading.offloading_helpers import (get_sdfg_scope_dict,
                                                                      register_kernel_local_transients)

from typing import Any, Dict, Optional

from dace.transformation.passes.offloading.host_maps import HostMapSpec, host_maps


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Decide what runs on the accelerator, and place the host/device copies that follow.

    Phases 2-4 run to a fixpoint: phase 4 resolves hybrid states by wrapping host code in
    single-iteration maps, and phase 3 rewrites single-element containers. Both consume what they
    resolve -- a state stops being hybrid, a container is recorded in ``changed_containers`` and is
    not revisited -- so the loop terminates in a number of rounds bounded by the graph. The counter
    only stops a runaway from looping forever, so it is set far above what any real SDFG needs.
    """

    max_iterations = properties.Property(
        dtype=int,
        default=1000,
        desc="Safety bound on the phase 2-4 fixpoint iteration. Reaching it is a bug, not a "
        "workload property: the loop converges once no state is hybrid and no container changed.")
    verbose = properties.Property(dtype=bool, default=False, desc="Print what each phase decided.")

    def __init__(self, host_maps: HostMapSpec = None, **kwargs):
        """
        :param host_maps: which maps keep a HOST schedule, so that the maps under them become the
            kernels. ``None`` (default) or ``False`` -- none are named; ``True`` -- detect them
            structurally; a list -- exactly these, each given as a map label or as the ``MapEntry``
            node itself. Not a serialized ``Property``: a ``MapEntry`` cannot round-trip through
            JSON, and a caller handing over node objects is driving the pass in process anyway.
        :note: a map enclosing a device-wide library node is kept on the host whatever this says --
            a call only host code can issue is a requirement, not a preference.
        """
        super().__init__(**kwargs)
        self._host_maps = host_maps

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        cached_scopes = get_sdfg_scope_dict(sdfg)  # cache the result of an expensive operation

        # Which maps stay on the host, so that what they launch becomes the kernels.
        host_map_entries = host_maps(sdfg, self._host_maps)
        if self.verbose and host_map_entries:
            print(f"host maps: {[entry.map.label for entry in host_map_entries]}")

        # Phase 1: set sequential / GPU schedules
        SchedulePhase().apply(sdfg, cached_scopes, verbose=self.verbose, host_map_entries=host_map_entries)

        # Fix Point Iteration of Phases 2 - 4
        changed_containers = OrderedSet()
        maps_changed = False
        for _ in range(self.max_iterations):

            # Phase 2: build intermediate representation and find hybrid states
            hybrid_states = OrderedSet()
            IRep = CopyAnalysisPhase().apply(sdfg, hybrid_states, cached_scopes, verbose=self.verbose)

            # Phase 3: decide if single-element values are stored in Scalars or in length-one Arrays
            new_changed_containers = SingleElementValuePhase().apply(sdfg,
                                                                     exceptions=changed_containers,
                                                                     verbose=self.verbose)
            changed_containers |= new_changed_containers

            # Phase 4: resolve hybrid states into pure GPU states by inserting single-iteration maps
            if hybrid_states:
                SingleIterationMapPhase().apply(sdfg, hybrid_states, verbose=self.verbose)
                maps_changed = True

            # Phase 5: iterate until the SDFG reaches a fixpoint
            if hybrid_states or new_changed_containers:  # sdfg has been changed
                cached_scopes = get_sdfg_scope_dict(sdfg)
                continue  # repeat
            break

        else:
            raise RuntimeError(f"OffloadToAccelerator did not reach a fixpoint in {self.max_iterations} "
                               "iterations. The phase 2-4 loop is expected to converge; treat this as a bug "
                               "rather than raising the bound.")

        # Phase 6: insert explicit host-device copies into the SDFG based on the IR
        CopyInsertionPhase().apply(sdfg, IRep, verbose=self.verbose)

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
        SingleElementCopyOptimization().apply(sdfg, verbose=self.verbose)

        # A transient only device code touches is a register, not a host allocation the dispatcher
        # would have to answer with an illegal copy.
        register_kernel_local_transients(sdfg)
