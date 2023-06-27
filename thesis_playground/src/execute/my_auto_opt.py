from typing import Dict, List, Optional
import sympy

import dace
from dace import dtypes
from dace.sdfg import utils as sdutil
from dace.sdfg import SDFG, nodes, infer_types, SDFGState
from dace.transformation.optimizer import Optimizer

from dace.transformation.auto.auto_optimize import greedy_fuse, tile_wcrs, set_fast_implementations,\
                                                   move_small_arrays_to_stack, make_transients_persistent

# Transformations
from dace.sdfg.nodes import MapEntry
from dace.transformation.dataflow import TrivialMapElimination, MapCollapse, MapInterchange, MapFusion, MapToForLoop, \
                                         MapExpansion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess, MoveAssignmentOutsideIf, SwapLoopOrder
from dace.transformation import helpers as xfh

from utils.general import save_graph


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  program: str = None,
                  validate: bool = True,
                  validate_all: bool = True,
                  symbols: Dict[str, int] = None) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:

        * Simplify
        * Auto-parallelization (loop-to-map)
        * Greedy application of SubgraphFusion
        * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
        * Tiled stream accumulation (MapTiling -> AccumulateTransient)
        * Collapse all maps to parallelize across all dimensions
        * Set all library nodes to expand to ``fast`` expansion, which calls
          the fastest library on the target device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :param symbols: Optional dict that maps symbols (str/symbolic) to int/float
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """

    if symbols:
        specialise_symbols(sdfg, symbols)
    # Simplification and loop parallelization
    transformed = True
    # print(f"Free symbols in graph: {sdfg.free_symbols}")
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_trivial_map_elimination")

    sdfg.simplify(validate=False, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    if program is not None:
        save_graph(sdfg, program, "after_map_interchange")

    # My not working attempt at switching loop order on the for-loop-level (before they get transformed to maps)
    # while transformed:
    #     sdfg.simplify(validate=False, validate_all=validate_all)
    #     if program is not None:
    #         save_graph(sdfg, program, "after_simplify")
    #     for s in sdfg.sdfg_list:
    #         xfh.split_interstate_edges(s)
    #     if program is not None:
    #         save_graph(sdfg, program, "after_splitting_interstate_edges")
    #     transformed = sdfg.apply_transformations_repeated(SwapLoopOrder, validate=validate, validate_all=validate_all) > 0
    #     transformed = False
    #     if program is not None:
    #         save_graph(sdfg, program, "after_swap_loop_order")

    if device == dace.DeviceType.GPU:
        loop_to_map_outside_first(sdfg, validate=validate, validate_all=validate_all, program=program)
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        if program is not None:
            save_graph(sdfg, program, "after_simplify")
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        if program is not None:
            save_graph(sdfg, program, "after_splitting_interstate_edges")
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0
        if program is not None:
            save_graph(sdfg, program, "after_loop_to_map")

    # For some reason this causes a valueError: Found cycles with vert_loop_10
    # sdfg.apply_transformations_repeated(TrivialMapElimination)
    # if program is not None:
    #     save_graph(sdfg, program, "after_trivial_map_elimination")

    make_klev_outermost_map(sdfg)
    if program is not None:
        save_graph(sdfg, program, "after_make_klev_outermost")

    # make_klev_loops_again(sdfg)
    # if program is not None:
    #     save_graph(sdfg, program, "after_make_klev_loop_again")

    # xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapFusion])]
    # print(f"Number of possible MapFusion transformations: {len(xforms)}")

    k_caching_prototype_v1(sdfg, validate, validate_all, program)

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_simplify_and_collapse")

    # fuse subgraphs greedily
    sdfg.simplify()
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)
    if program is not None:
        save_graph(sdfg, program, "after_greedy_fuse")

    # Move Loops inside Maps when possible
    # from dace.transformation.interstate import MoveLoopIntoMap
    # sdfg.apply_transformations_repeated([MoveLoopIntoMap])

    # Apply GPU transformations and set library node implementations
    if device == dtypes.DeviceType.GPU:
        sdfg.apply_gpu_transformations()
        sdfg.simplify()

    if program is not None:
        save_graph(sdfg, program, "after_gpu_transformations")

    # Tiled WCR and streams
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        tile_wcrs(nsdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        # Set OMP collapse property to map length
        if isinstance(node, nodes.MapEntry):
            # FORNOW: Leave out
            # node.map.collapse = len(node.map.range)
            pass

    if program is not None:
        save_graph(sdfg, program, "after_map_colapse")

    sdfg.apply_transformations(MoveAssignmentOutsideIf, validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_move_assignment_outside_if")

    if device == dtypes.DeviceType.Generic:
        # Validate at the end
        if validate or validate_all:
            sdfg.validate()

        return sdfg

    # Set all library nodes to expand to fast library calls
    set_fast_implementations(sdfg, device)

    # NOTE: We need to `infer_types` in case a LibraryNode expands to other LibraryNodes (e.g., np.linalg.solve)
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections on a per-SDFG basis
    for nsdfg in sdfg.all_sdfgs_recursive():
        nsdfg.openmp_sections = False

    # Set all Default storage types that are constant sized to registers
    move_small_arrays_to_stack(sdfg)

    # Make all independent arrays persistent
    make_transients_persistent(sdfg, device)

    if symbols:
        specialise_symbols(sdfg, symbols)
    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg


def specialise_symbols(sdfg: dace.SDFG, symbols: Dict[str, int]):
    """
    Specialise known sybols

    :param sdfg: The SDFG to act upon
    :type sdfg: dace.SDFG
    :param symbols: The dictionary with the known symbols and their value
    :type symbols: Dict[str, int]
    """
    debugprint = True
    # Specialize for all known symbols
    known_symbols = {s: v for (s, v) in symbols.items() if s in sdfg.free_symbols}
    known_symbols = {}
    # print(f"Free symbols in graph: {sdfg.free_symbols}")
    for (s, v) in symbols.items():
        if s in sdfg.free_symbols:
            if isinstance(v, (int, float)):
                known_symbols[s] = v
            if isinstance(v, sympy.core.numbers.Integer):
                try:
                    known_symbols[s] = int(v)
                except TypeError:
                    pass

    if debugprint and len(known_symbols) > 0:
        print("Specializing the SDFG for symbols", known_symbols)
    sdfg.specialize(known_symbols)


def loop_to_map_outside_first(sdfg: SDFG, validate: bool = True, validate_all: bool = False, program: str = None) -> SDFG:
    """
    Performs LoopToMap transformation by applying it to the outer loop first

    :param sdfg: The SDFG to work with
    :type sdfg: SDFG
    :param validate: If True, validates the SDFG after all transformations
                     have been applied, defaults to True
    :type validate: bool, optional
    :param validate_all: If True, validates the SDFG after every step, defaults to False
    :type validate_all: bool, optional
    :param program: The name of the program, used for debug saving graphs, is Optional. If not given will not save
    graphs
    :type program: str
    :return: The optimised SDFG
    :rtype: SDFG
    :note: Works by applying LoopToMap to the outermost loop where the
    transformation can be applied. Has not been thouroughly tested yet.
    """

    sdfg.simplify(validate=False, validate_all=validate_all)
    number_of_transformations_performed = 1
    if program is not None:
        save_graph(sdfg, program, "after_simplify")
    while number_of_transformations_performed > 0:
        outside_loop_transformations = []
        # Get list of all possible transformations
        transformations = [xform for xform in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap])]

        # Find the transformation which is applied to the outermost loop
        for xform in transformations:
            is_outside_loop = True
            # Check if it is the outermoost loop by checking if the loop guard is in any of the loop states of the other
            # found transformations. This could in theory find several outermost loops
            for other_form in transformations:
                if other_form != xform:
                    other_states: List[SDFGState] = list(
                        sdutil.dfs_conditional(sdfg.sdfg_list[other_form.sdfg_id], [other_form.loop_begin],
                                               lambda _, c: c is not other_form.loop_guard))
                    if xform.loop_guard in other_states:
                        is_outside_loop = False
            if is_outside_loop:
                outside_loop_transformations.append(xform)

        # Apply the first of the found transformations
        number_of_transformations_performed = min(len(outside_loop_transformations), 1.0)
        if len(outside_loop_transformations) > 0:
            xform = outside_loop_transformations[0]
            # Apply for the LoopToMap transformations does not use the first argument, thus None is passed here
            xform.apply(None, sdfg.sdfg_list[xform.sdfg_id])

        if program is not None:
            save_graph(sdfg, program, "after_outer_loop_to_map")

    return sdfg


def make_klev_outermost_map(sdfg: SDFG):
    transformed = True
    number_transformed = 0
    while transformed:
        transformations = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapInterchange])]
        transformed = False
        for xform in transformations:
            if xform.inner_map_entry.range.ranges[0][1] == 137 or xform.inner_map_entry.range.ranges[0][1] == 136:
                xform.apply(sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id), sdfg.sdfg_list[xform.sdfg_id])
                transformed = True
                number_transformed += 1
    print(f"Applied {number_transformed} transformation to move KLEV-loop outside")


def make_klev_loops_again(sdfg: SDFG):
    # Throws an error
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapToForLoop])]
    print(f"Number of possible MapToForLoop transformations: {len(xforms)}")
    for xform in xforms:
        if isinstance(xform.map_entry, MapEntry) and xform.map_entry.map.range.ranges[0][1] == 137:
            print(xform.map_entry.map.range)
            xform.apply(sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id), sdfg.sdfg_list[xform.sdfg_id])


def k_caching_prototype_v1(sdfg: SDFG, validate: bool, validate_all: bool, program: Optional[str] = None):
    # if program is not None:
    #     save_graph(sdfg, program, "before_trivial_map_elimination")
    # sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)
    # if program is not None:
    #     save_graph(sdfg, program, "after_trivial_map_elimination")

    if program is not None:
        save_graph(sdfg, program, "before_map_expansion")
    sdfg.apply_transformations_repeated([MapExpansion])
    if program is not None:
        save_graph(sdfg, program, "after_map_expansion")
    # Force KLEV loop with vertical dependency into a map
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap], permissive=True)]
    if len(xforms) > 0:
        for xform in xforms:
            if 'KLEV' in sdfg.sdfg_list[xform.sdfg_id].edges_between(xform.loop_guard, xform.loop_begin)[0].data.free_symbols:
                xform.apply(sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id), sdfg.sdfg_list[xform.sdfg_id])
    else:
        print("WARNING: To transformation found to force KLEV to a map")
    if program is not None:
        save_graph(sdfg, program, "after_force_klev_to_map")

    # TODO: There are 0 possible map fusions, maybe need to remove unneccessary nodes between and maybe our more
    # difficult nature of the transient is a problem
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapFusion], permissive=True)]
    print(f"Number of possible MapFusion transformations: {len(xforms)}")
