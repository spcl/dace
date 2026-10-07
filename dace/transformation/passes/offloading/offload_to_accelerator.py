# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any

from ordered_set import OrderedSet

from dace import dtypes, properties
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, UnstructuredControlFlow
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes import FullMapFusion
from dace.transformation.passes.fold_constant_tables import FoldConstantTables
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace.transformation.passes.offloading.host_maps import HostMapSpec, find_host_maps, maps_pinned_by_host_loops
from dace.transformation.passes.offloading.locations import Locations, Wants
from dace.transformation.passes.offloading.nested_bodies import host_level_nested_sdfgs, prepare_body
from dace.transformation.passes.offloading.placement import Placement
from dace.transformation.passes.offloading.single_element import retype_single_elements, single_element_copies_into_map
from dace.transformation.passes.offloading.size1_wrappers import wrap_host_code


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Turn top-level maps and library nodes into kernels and copy a container where it changes side.

    Requires structured control flow: ``ControlFlowRaising`` runs first, and what it cannot raise (an
    ``UnstructuredControlFlow`` region, a block leaving through several or a conditional interstate edge) raises
    ``NotImplementedError``. Loops, conditionals, ``break``, ``continue`` and ``return`` are supported.
    """

    max_iterations = properties.Property(
        dtype=int, default=1000, desc="Safety bound on the placement fixpoint; reaching it is a bug."
    )
    pin_host_loop_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Keep the small maps of a serial host loop on the host when offloading them would copy what they "
        "share with the loop's host code every iteration.",
    )

    def __init__(
        self,
        host_maps: HostMapSpec = False,
        max_iterations: int | None = None,
        pin_host_loop_maps: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        :param host_maps: maps that keep a host schedule so the maps under them become the kernels, see
            :data:`~dace.transformation.passes.offloading.host_maps.HostMapSpec`; a map holding a callback stays on
            the host whatever this says. Not a ``Property``: a ``MapEntry`` cannot round-trip through JSON.
        """
        super().__init__(**kwargs)
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if pin_host_loop_maps is not None:
            self.pin_host_loop_maps = pin_host_loop_maps
        # make_properties admits a plain attribute only with a leading underscore.
        self._host_maps = host_maps

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> OrderedSet[type[ppl.Pass]]:
        return OrderedSet([ControlFlowRaising])

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> OrderedSet[str] | None:
        """
        :return: every container left in a GPU storage, qualified by its SDFG's id, or None if there is none.
        """
        offending = [block.label for block in unstructured_control_flow(sdfg)]
        if offending:
            raise NotImplementedError(
                "OffloadToAccelerator requires structured control flow, which ControlFlowRaising "
                f"could not produce: these blocks branch through interstate edges: {offending}"
            )
        host_map_entries = find_host_maps(sdfg, self._host_maps)
        pinned = maps_pinned_by_host_loops(sdfg) if self.pin_host_loop_maps else OrderedSet()
        assign_schedules(sdfg, host_map_entries, pinned)

        placed_on_gpu = self.offload_level(sdfg, host_map_entries)
        helpers.register_kernel_local_transients(sdfg, placed_on_gpu)
        helpers.refuse_by_value_scalars_the_device_writes(sdfg)
        return helpers.device_resident(sdfg) or None

    def offload_level(self, sdfg: SDFG, host_map_entries: OrderedSet[nodes.MapEntry]) -> OrderedSet[str]:
        """Place the data of one host level, then of every nested SDFG at it; return the transients put on the
        device."""
        FoldConstantTables().apply_pass(sdfg, {})
        # Host code and each kernel declare a constant locally, so it is a register on either side.
        for name in sdfg.constants:
            if name in sdfg.arrays:
                sdfg.arrays[name].storage = dtypes.StorageType.Register
        wants, wrapped = self.settle(sdfg, host_map_entries)
        placed_on_gpu = Placement(sdfg, wants).apply()
        if wrapped:
            ppl.Pipeline(
                [
                    FullMapFusion(
                        strict_dataflow=True, perform_vertical_map_fusion=True, perform_horizontal_map_fusion=True
                    )
                ]
            ).apply_pass(sdfg, {})
        single_element_copies_into_map(sdfg)

        for state in sdfg.states():
            for node in list(host_level_nested_sdfgs(state, host_map_entries)):
                prepare_body(sdfg, state, node)
                placed_on_gpu |= self.offload_level(node.sdfg, host_map_entries)
        return placed_on_gpu

    def settle(self, sdfg: SDFG, host_map_entries: OrderedSet[nodes.MapEntry]) -> tuple[dict[SDFGState, Wants], bool]:
        """Re-type single elements and wrap the host code of hybrid states until no state wants a container on
        both sides; return what the states want and whether anything was wrapped. It converges because a wrapped
        state has no host code left at its top level and a container is re-typed once."""
        retyped: OrderedSet[str] = OrderedSet()
        wrapped = False
        for _ in range(self.max_iterations):
            locations = Locations(sdfg, host_map_entries)
            wants = {state: locations.of_state(state) for state in sdfg.states()}
            changed = retype_single_elements(sdfg, retyped)
            retyped |= changed
            for state in locations.hybrid_states:
                wrap_host_code(sdfg, state, host_map_entries)
            if not locations.hybrid_states and not changed:
                return wants, wrapped
            wrapped = wrapped or bool(locations.hybrid_states)
        raise RuntimeError(
            f"OffloadToAccelerator did not reach a fixpoint in {self.max_iterations} iterations. "
            "The placement loop is expected to converge; treat this as a bug rather than raising "
            "the bound."
        )


def unstructured_control_flow(sdfg: SDFG) -> list[ControlFlowBlock]:
    """The ``UnstructuredControlFlow`` regions in ``sdfg`` and the SDFGs nested in it, and the blocks leaving
    through several interstate edges or a conditional one."""
    found: list[ControlFlowBlock] = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, UnstructuredControlFlow):
            found.append(region)
        elif isinstance(region, ControlFlowRegion):
            for block in region.nodes():
                out_edges = region.out_edges(block)
                if len(out_edges) > 1 or (out_edges and not out_edges[0].data.is_unconditional()):
                    found.append(block)
    return found


def assign_schedules(
    sdfg: SDFG,
    host_map_entries: OrderedSet[nodes.MapEntry],
    pinned: OrderedSet[nodes.MapEntry],
    host_level: bool = True,
) -> None:
    """``GPU_Device`` for the maps and library nodes of a host level, ``Sequential`` below one (``Default`` can be
    lowered to CUDA in the wrong places). A host map keeps its body a host level, and so does a nested SDFG at
    one; a pinned map stays on the host with its subtree."""

    def walk(
        children: dict[nodes.Node | None, list[nodes.Node]], entry: nodes.MapEntry | None, host_level: bool
    ) -> None:
        for node in children[entry]:
            on_host = node in host_map_entries or node in pinned
            if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
                node.schedule = (
                    dtypes.ScheduleType.GPU_Device if host_level and not on_host else dtypes.ScheduleType.Sequential
                )
            if isinstance(node, nodes.MapEntry):
                walk(children, node, host_level and node in host_map_entries and node not in pinned)
            elif isinstance(node, nodes.NestedSDFG):
                assign_schedules(node.sdfg, host_map_entries, pinned, host_level)

    for state in sdfg.states():
        walk(state.scope_children(), None, host_level)
