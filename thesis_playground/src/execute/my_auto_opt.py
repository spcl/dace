from typing import Dict, List, Optional, Tuple, Union
import sympy
import logging

import dace
from dace import dtypes
from dace.data import Array, Scalar
from dace.sdfg import utils as sdutil
from dace.sdfg import SDFG, nodes, infer_types, SDFGState
from dace.sdfg.graph import SubgraphView
from dace.transformation.optimizer import Optimizer
from dace.memlet import Memlet
from dace.transformation.passes import ScalarFission
from dace.transformation import Pipeline
from dace.transformation.interstate import LoopUnroll, StateFusion
from dace.transformation.interstate.loop_detection import find_for_loop
from dace.subsets import Range

from dace.transformation.auto.auto_optimize import greedy_fuse, tile_wcrs, set_fast_implementations,\
                                                   move_small_arrays_to_stack, make_transients_persistent
from dace.transformation.subgraph import helpers as xfsh

# Transformations
from dace.dtypes import ScheduleType
from dace.transformation.dataflow import TrivialMapElimination, MapCollapse, MapInterchange, MapToForLoop, \
                                         MapExpansion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess, MoveAssignmentOutsideIf
from dace.transformation import helpers as xfh
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.auto.auto_optimize import get_composite_fusion

from utils.general import save_graph
from print_memlets import find_all_strange_memlets

logger = logging.getLogger(__name__)


def apply_subgraph_fusion(
        sdfg: SDFG,
        k_caching_args: Dict[str, int],
        symbols: Dict[str, int],
        verbose_name: Optional[str] = None):
    """
    Apply subgraph fusion to selected maps for K-caching

    :param sdfg: The SDFG to apply fusion on
    :type sdfg: SDFGp
    :param k_caching_args: Arguments regarding k_caching
    :type k_caching_args: Dict[str, int]
    :param symbols: Symbols dictionary
    :type symbols: Dict[str, int]
    :param verbose_name: Name used to save intermediate graphs
    :type verbose_name: Optional[str]
    """
    cloudsc_state = sdfg.find_state('stateCLOUDSC')
    cloudsc_nsdfg = [n for n in cloudsc_state.nodes() if isinstance(n, nodes.NestedSDFG)][0]
    upper_state = cloudsc_nsdfg.sdfg.find_state('_state_l1733_c1733_0')
    lower_state = cloudsc_nsdfg.sdfg.find_state('single_state_body_8__for_it_14_4')
    transformations = [xform for xform in Optimizer(sdfg).get_pattern_matches(patterns=[StateFusion], permissive=True)]
    for xf in transformations:
        if xf.first_state.name =='_state_l1733_c1733_0' and xf.second_state.name == 'single_state_body_8__for_it_14_4':
            xf_sdfg = sdfg.sdfg_list[xf.sdfg_id]
            xf_state = xf_sdfg.find_state(xf.state_id)
            xf.apply(xf_state, xf_sdfg)
    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "after_state_fusion")
    sdfg.validate()

    map_entries = []
    for node in upper_state.nodes():
        if isinstance(node, nodes.MapEntry):
            map_entries.append(node)

    cf = get_composite_fusion(k_caching_args)
    subgraph = xfsh.subgraph_from_maps(cloudsc_nsdfg.sdfg, upper_state, map_entries)
    cf.setup_match(subgraph)
    if cf.can_be_applied(cloudsc_nsdfg.sdfg, subgraph):
        cf.apply(cloudsc_nsdfg.sdfg)

    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "after_subgraph_fusion")
    sdfg.validate()


def auto_optimize_phase_1(
        sdfg: SDFG,
        program: str = None,
        validate: bool = True,
        validate_all: bool = True,
        outside_first: bool = True,
        symbols: Dict[str, int] = None):
    """
    Perform auto optimisation. Only first phase without applying any architecture specific optimisations

    :param sdfg: The SDFG on which to apply the optimisations inplace
    :type sdfg: SDFG
    :param program: The name of the program if graph should be saved between, defaults to None
    :type program: str, optional
    :param validate: If True, validates the SDFG after all transformations have been applied, defaults to True
    :type validate: bool, optional
    :param validate_all: If True, validates the SDFG after every step, defaults to True
    :type validate_all: bool, optional
    """
    logger.debug(f"program: {program}, outside_first: {outside_first}, symbols: {symbols}")

    # Fix for full cloudsc
    symbols_to_remove = {
        'CLOUDSCOUTER2': ['_for_it_49', '_for_it_62', '_for_it_65'],
        'CLOUDSCOUTER3': ['_for_it_49', '_for_it_62', '_for_it_65'],
        'CLOUDSCOUTER4': ['_for_it_59', '_for_it_46', '_for_it_62'],
    }
    sdfg.validate()
    if sdfg.name in symbols_to_remove:
        cloudsc_state = sdfg.find_state('stateCLOUDSC')
        for node in cloudsc_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                logger.debug(f"remove symbols in {node}")
                for symbol in symbols_to_remove[sdfg.name]:
                    if symbol in node.sdfg.symbols:
                        node.sdfg.remove_symbol(symbol)

    if symbols:
        specialise_symbols(sdfg, symbols)
    if validate:
        sdfg.validate()

    if program is not None:
        save_graph(sdfg, program, "after_specialise_symbols")

    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_trivial_map_elimination")

    sdfg.simplify(validate=False, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    pipeline = Pipeline([ScalarFission()])
    for sd in sdfg.all_sdfgs_recursive():
        results = pipeline.apply_pass(sd, {})['ScalarFission']
        logger.debug("Result of ScalarFission: %s", results)

    if program is not None:
        save_graph(sdfg, program, "after_scalar_fission")

    if outside_first:
        loop_to_map_outside_first(sdfg, validate=validate, validate_all=validate_all, program=program)
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        if program is not None:
            save_graph(sdfg, program, "after_simplify")
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        if program is not None:
            save_graph(sdfg, program, "after_splitting_interstate_edges")
        # l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
        #                                            validate=False,
        #                                            validate_all=validate_all)
        l2ms = sdfg.apply_transformations_repeated([LoopToMap], validate=False, validate_all=validate_all)
        if program is not None:
            save_graph(sdfg, program, "after_loop_to_map")
        l2ms += sdfg.apply_transformations_repeated([RefineNestedAccess], validate=False, validate_all=validate_all)
        transformed = l2ms > 0
        if program is not None:
            save_graph(sdfg, program, "after_refine_nested_access")

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify(verbose=True, validate_all=True)
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_simplify_and_collapse")

    # fuse subgraphs greedily
    sdfg.simplify()
    if program is not None:
        save_graph(sdfg, program, "after_simplify")
    if validate:
        sdfg.validate


def auto_optimize_phase_2(sdfg: SDFG,
                          device: dtypes.DeviceType,
                          program: str = None,
                          validate: bool = True,
                          validate_all: bool = True,
                          symbols: Dict[str, int] = None,
                          k_caching: bool = False,
                          move_assignments_outside: bool = True,
                          storage_on_gpu: bool = True,
                          full_cloudsc_fixes: bool = False,
                          skip_fusing: bool = False
                          ) -> SDFG:
    """
    Perform optimisations which are device/architectrie specific. Works inplace on the given SDFG

    :param sdfg: The SDFG to optimise
    :type sdfg: SDFG
    :param device: The device to optimise for
    :type device: dtypes.DeviceType
    :param program: The name of the program. If set will save intermediate graphs using this name, defaults to None
    :type program: str, optional
    :param validate: If True, validates the SDFG after all transformations have been applied, defaults to True
    :type validate: bool, optional
    :param validate_all: If True, validates the SDFG after every step, defaults to True
    :type validate_all: bool, optional
    :param symbols: Dictionary of symbols to specialise and their value, defaults to None
    :type symbols: Dict[str, int], optional
    :param k_caching: If K-caching should be applied, defaults to False
    :type k_caching: bool, optional
    :param move_assignments_outside: If MoveAssignmentsOutsideIf transformation should be applied, defaults to True
    :type move_assignments_outside: bool, optional
    :param storage_on_gpu: If true, assumes that all arrays given as input/output are already on the GPU, defaults to
    True
    :type storage_on_gpu: bool
    :param full_cloudsc_fixes: Set to true if fixes for full cloudsc sdfg should be generated, defaults to False
    :type full_cloudsc_fixes: bool
    """
    logger.debug(f"Program: {program}")
    if symbols:
        specialise_symbols(sdfg, symbols)
    if not skip_fusing:
        if k_caching:
            k_caching_prototype_v1(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)
        else:
            greedy_fuse(sdfg, device=device, validate_all=validate_all)
            # fuse stencils greedily
            greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)
            if program is not None:
                save_graph(sdfg, program, "after_greedy_fuse")

    # Move Loops inside Maps when possible
    # from dace.transformation.interstate import MoveLoopIntoMap, StateFusion
    # sdfg.apply_transformations_repeated([MoveLoopIntoMap, StateFusion, LoopToMap])
    # if program is not None:
    #     save_graph(sdfg, program, "after_move_loop_into_map")

    # Apply GPU transformations and set library node implementations
    if device == dtypes.DeviceType.GPU:
        if storage_on_gpu:
            logger.debug("Transfer arrays to GPU_Global storage")
            for k, v in sdfg.arrays.items():
                if not v.transient and type(v) == dace.data.Array:
                    v.storage = dace.dtypes.StorageType.GPU_Global

        logger.debug("Apply GPU transformations")
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

    if move_assignments_outside:
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


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  program: str = None,
                  validate: bool = True,
                  validate_all: bool = True,
                  symbols: Dict[str, int] = None,
                  k_caching: bool = False,
                  outside_first: bool = True,
                  move_assignments_outside: bool = True
                  ) -> SDFG:
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
    logger.info(f"sdfg: {sdfg.name}, device: {device}, program: {program}, validate: {validate}"
                f", validate_all: {validate_all}, symbols: {symbols}, k_caching: {k_caching}")
    # Fix for full cloudsc
    sdfg.validate()
    if sdfg.name == 'CLOUDSCOUTER':
        cloudsc_state = sdfg.find_state('stateCLOUDSC')
        for node in cloudsc_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                print(f"[my_auto_opt::auto_optimizes] remove symbols in {node}")
                symbols_to_remove = ['_for_it_49', '_for_it_62', '_for_it_65']
                for symbol in symbols_to_remove:
                    if symbol in node.sdfg.symbols:
                        node.sdfg.remove_symbol(symbol)

    auto_optimize_phase_1(sdfg, program, validate, validate_all, outside_first, symbols)
    auto_optimize_phase_2(sdfg, device, program, validate, validate_all, symbols, k_caching, move_assignments_outside)


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
        logger.debug(f"Specializing the SDFG for symbols {known_symbols}")
    sdfg.specialize(known_symbols)


def loop_to_map_outside_first(sdfg: SDFG,
                              validate: bool = True,
                              validate_all: bool = True,
                              program: str = None) -> SDFG:
    """
    Performs LoopToMap transformation by applying it to the outer loop first

    :param sdfg: The SDFG to work with
    :type sdfg: SDFG
    :param validate: If True, validates the SDFG after all transformations
                     have been applied, defaults to True
    :type validate: bool, optional
    :param validate_all: If True, validates the SDFG after every step, defaults to True
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

    for s in sdfg.sdfg_list:
        xfh.split_interstate_edges(s)
    if program:
        save_graph(sdfg, program, "after_split_interstate_edges")

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
            logger.debug(f" apply LoopToMap to guard: {xform.loop_guard}, begin: {xform.loop_begin} on sdfg: "
                         f"{sdfg.sdfg_list[xform.sdfg_id].label}")
            xform.apply(None, sdfg.sdfg_list[xform.sdfg_id])
    if program is not None:
        save_graph(sdfg, program, "after_outer_loop_to_map")
    sdfg.validate()
    sdfg.apply_transformations_repeated([RefineNestedAccess], validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_outer_refine_nested_access")

    return sdfg


def is_map_over_symbol(map_range: Range, symbols: Dict[str, int], symbol_name: str, allowed_difference: int = 0) -> bool:
    """
    Checks if the given maps range ends at the given symbol

    :param map_range: The map range to check
    :type map_range: Range
    :param symbols: The symbol dictionary used to get the value of the given symbol
    :type symbols: Dict[str, int]
    :param symbol_name: The name of the symbol
    :type symbol_name: str
    :param allowed_difference: Max alloed difference between map end and value of symbol, defaults to 0
    :type allowed_difference: int, optional
    :return: True if map range ends at the given symbol
    :rtype: bool
    """
    if dace.symbolic.issymbolic(map_range[0][1]):
        difference = (map_range[0][1] - dace.symbol(symbol_name)).evalf(subs=symbols)
        logger.debug("symbolic, difference: %s", difference)
        return not dace.symbolic.issymbolic(difference) and abs(difference) <= allowed_difference
    else:
        logger.debug("values, difference: %.2f", abs(map_range[0][1] - symbols[symbol_name]))
        return abs(map_range[0][1] - symbols[symbol_name]) <= allowed_difference


def make_outermost_map(sdfg: SDFG, symbols: Dict[str, int], outermost_variable: str, allowed_difference: int = 0):
    """
    Swap maps as often as possible to make sure one is as far outside as possible. Assumes that any maps to be swapped
    are expanded and thus have only one range.

    :param sdfg: The SDFG to act upon
    :type sdfg: SDFG
    :param symbols: Symbols dictionary
    :type symbols: Dict[str, int]
    :param outermost_variable: The name of the variable who is the end of iteration range of the maps to be moved
    outside. Must be in the symbols dict passed.
    :type outermost_variable: str
    :param allowed_difference: max allowed difference between map end and value of symbol, defaults to 0
    :type allowed_difference: int
    """
    transformed = True
    number_transformed = 0
    while transformed:
        transformations = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapInterchange])]
        transformed = False
        for xform in transformations:
            xf_sdfg = sdfg.sdfg_list[xform.sdfg_id]
            xf_state = xf_sdfg.find_state(xform.state_id)
            logger.debug(
                "Consider xform with inner range %s at map: %s in state: %s",
                xform.inner_map_entry.range.ranges, xform.inner_map_entry.map,
                xf_state)
            if is_map_over_symbol(xform.inner_map_entry.range, symbols, outermost_variable, allowed_difference):
                xform.apply(xf_state, xf_sdfg)
                transformed = True
                number_transformed += 1
                logger.debug("Applied it")
    logger.debug(f"Applied {number_transformed} transformation to move KLEV-loop outside")


def apply_transformation_stepwise(sdfg: SDFG,
                                  transformations: List[dace.transformation.transformation.TransformationBase],
                                  program: str, description: str) -> int:
    """
    Applies the given transformation repeteadetly and saves the graph after each applying.

    :param sdfg: The SDFG to apply the transformations on
    :type sdfg: SDFG
    :param transformation: The transformation to apply
    :type transformation: List[dace.transformation.transformation.TransformationBase]
    :param program: The name of the program, used to determine the folder when saving the graph.
    :type program: str
    :param description: Description used when saving the graph. It will be called after_<description>
    :type description: str
    :return: Number of transformations applied
    :rtype: int
    """
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=transformations)]
    count = 0
    while len(xforms) > 1:
        count += 1
        logger.debug(f"apply {xforms[0]}")
        xforms[0].apply(sdfg.sdfg_list[xforms[0].sdfg_id].find_state(xforms[0].state_id),
                        sdfg.sdfg_list[xforms[0].sdfg_id])
        save_graph(sdfg, program, f"after_{description}")
        xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=transformations)]
        sdfg.validate()
    return count


def map_fusion_merge_different_ranges(transformation_class, max_start_difference: int, max_end_difference: int):
    """
    Changes the given MapFusion transformation class to allow fusion maps with slightly different start/ends

    :param transformation_class: The transformation class
    :type transformation_class: class
    :param max_start_difference: Max difference between start of the two maps
    :type max_start_difference: int
    :param max_end_difference: Max difference between end of the two maps
    :type max_end_difference: int
    """
    transformation_class.max_start_difference = 1
    transformation_class.max_end_difference = 1
    return transformation_class


def k_caching_prototype_v1_prepare_fusion(sdfg: SDFG,
                           validate: bool,
                           validate_all: bool,
                           device: dace.DeviceType,
                           symbols: Dict[str, int],
                           program: Optional[str] = None,
                           full_cloudsc_fixes: bool = False):
    """
    K-Caching: Steps before fusion

    :param validate: [TODO:description]
    :type validate: bool
    :param validate_all: [TODO:description]
    :type validate_all: bool
    :param device: [TODO:description]
    :type device: dace.DeviceType
    :param symbols: [TODO:description]
    :type symbols: Dict[str, int]
    :param program: [TODO:description], defaults to None
    :type program: Optional[str], optional
    :param full_cloudsc_fixes: [TODO:description], defaults to False
    :type full_cloudsc_fixes: bool, optional
    """
    sdfg.add_symbol('NCLDTOP', int)
    sdfg.simplify()
    if symbols:
        specialise_symbols(sdfg, symbols)
    logger.debug("Constants: %s, symbols: %s", sdfg.constants, sdfg.symbols)

    xforms = [xform for xform in Optimizer(sdfg).get_pattern_matches(patterns=[LoopUnroll])]
    for xform in xforms:
        xf_sdfg = sdfg.sdfg_list[xform.sdfg_id]
        xf_state = xf_sdfg.find_state(xform.state_id)
        try:
            found = find_for_loop(xf_sdfg, xform.loop_guard, xform.loop_begin)
            itervar = found[0]
            logger.info("Found possible loop to unroll with itervar %s", itervar)
            if itervar == '_for_it_14':
                logger.debug("Apply loop unrolling")
                xform.apply(xf_state, xf_sdfg)
        except Exception as e:
            logger.warning("Encountered error while looking for loop: %s", e)
    if program is not None:
        save_graph(sdfg, program, "after_loop_unroll")

    for s in sdfg.sdfg_list:
        xfh.split_interstate_edges(s)
    if program is not None:
        save_graph(sdfg, program, "after_loop_unroll")

    # Force KLEV loop with vertical dependency into a map
    sdfg.validate()
    to_transform = True
    while to_transform:
        xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap], permissive=True)]
        to_transform = False
        if len(xforms) > 0:
            to_transform = True
            xform = xforms[0]
            xf_sdfg = sdfg.sdfg_list[xform.sdfg_id]
            xf_state = sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id)
            xform.apply(xf_state, xf_sdfg)
            sdfg.validate()
            # logger.debug("Applied LoopToMap to state %s in SDFG %s", xf_state, xf_sdfg.name)
            # if program is not None:
            #     save_graph(sdfg, program, "after_loop_to_map")

    sdfg.validate()
    if program is not None:
        save_graph(sdfg, program, "after_force_klev_to_map")

    amount = sdfg.apply_transformations_repeated([RefineNestedAccess])
    logger.debug("Applied %i RefineNestedAccess transformations", amount)
    if program is not None:
        save_graph(sdfg, program, "after_refine_nested_acces")

    if full_cloudsc_fixes:
        sdfg = find_all_strange_memlets(sdfg)

    SimplifyPass(validate=True, validate_all=True, skip=['RemoveUnusedSymbols']).apply_pass(sdfg, {})

    # Simplify to get rid of any leftover states
    if program is not None:
        save_graph(sdfg, program, "after_simplify")


    # Apply TrivialMapElimination before Fusion to avoid problems with maps over KLON=1
    sdfg.apply_transformations_repeated([TrivialMapElimination])
    if program is not None:
        save_graph(sdfg, program, "after_trivial_map_elimination")

    SimplifyPass(validate=True, validate_all=True, skip=['RemoveUnusedSymbols']).apply_pass(sdfg, {})
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    # Map expansion is required to make sure that klev is outermost afterwards to be able to fuse them properly
    sdfg.apply_transformations_repeated([MapExpansion])
    if program is not None:
        save_graph(sdfg, program, "after_map_expansion")

    make_outermost_map(sdfg, symbols, 'KLEV', allowed_difference=1)
    if program is not None:
        save_graph(sdfg, program, "after_make_klev_outermost")

    sdfg.validate()


def k_caching_prototype_v1_fuse(sdfg: SDFG,
                                validate: bool,
                                validate_all: bool,
                                device: dace.DeviceType,
                                symbols: Dict[str, int],
                                program: Optional[str] = None,
                                full_cloudsc_fixes: bool = False):
    """
    K-Caching: Fusing step

    :param validate: [TODO:description]
    :type validate: bool
    :param validate_all: [TODO:description]
    :type validate_all: bool
    :param device: [TODO:description]
    :type device: dace.DeviceType
    :param symbols: [TODO:description]
    :type symbols: Dict[str, int]
    :param program: [TODO:description], defaults to None
    :type program: Optional[str], optional
    :param full_cloudsc_fixes: [TODO:description], defaults to False
    :type full_cloudsc_fixes: bool, optional
    """
    if full_cloudsc_fixes:
        # Fix some memlets, which are not tight enough, going from the map to the nsdfg
        # The one for ZPFPLS includes a read, which is written just before. The memlet is the
        cloudsc_state = sdfg.find_state('stateCLOUDSC')
        cloudsc_nsdfg = [n for n in cloudsc_state.nodes() if isinstance(n, nodes.NestedSDFG)][0]
        state = cloudsc_nsdfg.sdfg.find_state('single_state_body_8__for_it_14_4')
        maps = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
        for m in maps:
            if m.label == 'single_state_body_map':
                map = m
        for edge in state.out_edges(map):
            #ZPFPLSX
            if edge.data.data == 'ZPFPLSX':
                logger.debug("ZPFPLSX edge: %s", edge.data.subset)
                edge.data.subset[1] = (edge.data.subset[1][0], edge.data.subset[1][0], edge.data.subset[1][2])
                logger.debug("ZPFPLSX edge after fixing: %s", edge.data.subset)
            # ZQX
            # if edge.data.data == "ZQX":
            #     logger.debug("Change ZQX edges")
            #     edge.data.subset[1] = (edge.data.subset[1][1], edge.data.subset[1][1], edge.data.subset[1][2])
            #     nsdfg = edge.dst
            #     nstate = nsdfg.sdfg.find_state('single_state_body_48')
            #     nmap = [n for n in nstate.nodes() if isinstance(n, nodes.MapEntry) and n.label ==
            #             'single_state_body_49_map'][0]
            #     for ie in nstate.in_edges(nmap):
            #         if ie.data.data == "ZQX":
            #             ie.data.subset[1] = (edge.data.subset[1][1], edge.data.subset[1][1], edge.data.subset[1][2])
            #     for oe in nstate.on_edges(nmap):
            #         if oe.data.data == "ZQX":
            #             oe.data.subset[1] = (edge.data.subset[1][1], edge.data.subset[1][1], edge.data.subset[1][2])

        if program is not None:
            save_graph(sdfg, program, "after_memlet_fixing")


        sdfg.validate()
        apply_subgraph_fusion(sdfg, {
            'max_difference_start': symbols['NCLDTOP']+1,
            'max_difference_end': 1,
            'disjoint_subsets': False,
            'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
                                              symbols['KLEV']),
            'fixed_new_shapes': {'ZQXN2D': [symbols['KLON'], 1, symbols['NCLV']]}},
            symbols, program)
    else:
        # Fuse maps to create one big KLEV-map
        greedy_fuse(sdfg, device=device, validate_all=validate_all, k_caching_args={
            'max_difference_start': symbols['NCLDTOP']+1,
            'max_difference_end': 1,
            'disjoint_subsets': False,
            'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
                                              symbols['KLEV'])
        })

        if program is not None:
            save_graph(sdfg, program, "after_greedy_fuse")


    sdfg.validate()
    SimplifyPass(validate=True, validate_all=True, skip=['RemoveUnusedSymbols']).apply_pass(sdfg, {})
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    # continue_search = True
    # while continue_search:
    #     xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapToForLoop], permissive=True)]
    #     logger.debug("Found %i many possible transformations to transform map back to for-loop", len(xforms))
    #     continue_search = False
    #     for xf in xforms:
    #         # expect that maps only have one dimension, as we did the MapExpansion transformation before
    #         xf_sdfg = sdfg.sdfg_list[xf.sdfg_id]
    #         xf_state = xf_sdfg.find_state(xf.state_id)
    #         if is_map_over_symbol(xf.map_entry.map.range, symbols, 'KLEV', 1):
    #             continue_search = True
    #             logger.debug("Found the correct map. Apply it to state %s and sdfg %s", xf_state.name, xf_sdfg.label)
    #             xf.apply(xf_state, xf_sdfg)
    #             if program is not None:
    #                 save_graph(sdfg, program, "after_map_to_for_loop")
    #             break
    sdfg.validate()


def k_caching_prototype_v1(sdfg: SDFG,
                           validate: bool,
                           validate_all: bool,
                           device: dace.DeviceType,
                           symbols: Dict[str, int],
                           program: Optional[str] = None,
                           full_cloudsc_fixes: bool = False):
    """
    Performs K-caching on the given SDFG by mergen all KLEV loops into one and shrink any intermediate arrays.

:param sdfg: The SDFG to act upon
:type sdfg: SDFG
:param validate: If True, validates the SDFG after all transformations
                 have been applied, defaults to True
:type validate: bool, optional
:param validate_all: If True, validates the SDFG after every step, defaults to True
:type validate_all: bool, optional
:param device: The device to optimise fort
:type device: dace.DeviceType
:param symbols: Dictionary of symbols to specialise and their valeus
:type symbols: Dict[str, int]
:param program: Name of the program. If set will save intermediate graphs using this name, defaults to None
:type program: Optional[str], optional
:param full_cloudsc_fixes: Set to true if fixes for full cloudsc sdfg should be generated, defaults to False
:type full_cloudsc_fixes: bool
    """
    k_caching_prototype_v1_prepare_fusion(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)
    k_caching_prototype_v1_fuse(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)

    logger.debug("Done with K-Caching")


def list_access_nodes(
        sdfg: dace.SDFG,
        array_name: str) -> List[Tuple[nodes.AccessNode, Union[dace.SDFGState, dace.SDFG]]]:
    """
    Find all access nodes in the SDFG of the given array name. Does not recourse into nested SDFGs.

    :param sdfg: The SDFG to search through
    :type sdfg: dace.SDFG
    :param array_name: The name of the wanted array
    :type array_name: str
    :return: [TODO:description]
    :rtype: List[Tuple[nodes.AccessNode, Union[dace.SDFGState, dace.SDFG]]]
    """
    found_nodes = []
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == array_name:
                found_nodes.append((node, state))
    return found_nodes


def change_strides(
        sdfg: dace.SDFG,
        stride_one_values: List[str],
        schedule: ScheduleType) -> SDFG:
    """
    Change the strides of the arrays on the given SDFG such that the given dimension has stride 1. Returns a new SDFG.

    :param sdfg: The input SDFG
    :type sdfg: dace.SDFG
    :param stride_one_values: Length of the dimension whose stride should be set to one. Expects that each array has
    only one dimension whose length is in this list. Expects that list contains name of symbols
    :type stride_one_values: List[str]
    :param schedule: Schedule to use to copy the arrays
    :type schedule: ScheduleType
    :return: SDFG with changed strides
    :rtype: SDFG
    """
    # Create new SDFG and copy constants and symbols
    original_name = sdfg.name
    sdfg.name = "changed_strides"
    new_sdfg = SDFG(original_name)
    for dname, value in sdfg.constants.items():
        new_sdfg.add_constant(dname, value)
    for dname, stype in sdfg.symbols.items():
        new_sdfg.add_symbol(dname, stype)

    changed_stride_state = new_sdfg.add_state("with_changed_strides", is_start_state=True)
    inputs, outputs = sdfg.read_and_write_sets()
    # Get all arrays which are persistent == not transient
    persistent_arrays = {name: desc for name, desc in sdfg.arrays.items() if not desc.transient}

    # Get the persistent arrays of all the transient arrays which get copied to GPU
    for dname in persistent_arrays:
        for access, state in list_access_nodes(sdfg, dname):
            if len(state.out_edges(access)) == 1:
                edge = state.out_edges(access)[0]
                if isinstance(edge.dst, nodes.AccessNode):
                    if edge.dst.data in inputs:
                        logger.debug("Replace %s by %s in inputs", edge.dst.data, dname)
                        inputs.remove(edge.dst.data)
                        inputs.add(dname)
            if len(state.in_edges(access)) == 1:
                edge = state.in_edges(access)[0]
                if isinstance(edge.src, nodes.AccessNode):
                    if edge.src.data in inputs:
                        logger.debug("Replace %s by %s in outputs", edge.src.data, dname)
                        outputs.remove(edge.src.data)
                        outputs.add(dname)

    # Only keep inputs and outputs which are persistent
    inputs.intersection_update(persistent_arrays.keys())
    outputs.intersection_update(persistent_arrays.keys())
    logger.debug("Handed SDFG, Inputs: %s, Outputs: %s", inputs, outputs)
    nsdfg = changed_stride_state.add_nested_sdfg(sdfg, new_sdfg, inputs=inputs, outputs=outputs)
    transform_state = new_sdfg.add_state_before(changed_stride_state, label="transform_data", is_start_state=True)
    transform_state_back = new_sdfg.add_state_after(changed_stride_state, "transform_data_back", is_start_state=False)

    # copy arrays
    for dname, desc in sdfg.arrays.items():
        if not desc.transient:
            if isinstance(desc, Array):
                new_sdfg.add_array(dname, desc.shape, desc.dtype, desc.storage,
                                   desc.location, desc.transient, desc.strides,
                                   desc.offset)
            elif isinstance(desc, Scalar):
                new_sdfg.add_scalar(dname, desc.dtype, desc.storage, desc.transient, desc.lifetime, desc.debuginfo)

    new_order = {}
    new_strides_map = {}

    # Map of array names in the nested sdfg:  key: array name in parent sdfg (this sdfg), value: name in the nsdfg
    # Assumes that name changes only appear in the first level of nsdfg nesting
    array_names_map = {}
    for graph in sdfg.sdfg_list:
        if graph.parent_nsdfg_node is not None:
            if graph.parent_sdfg == sdfg:
                for connector in graph.parent_nsdfg_node.in_connectors:
                    for in_edge in graph.parent.in_edges_by_connector(graph.parent_nsdfg_node, connector):
                        array_names_map[str(connector)] = in_edge.data.data

    for containing_sdfg, dname, desc in sdfg.arrays_recursive():
        shape_str = [str(s) for s in desc.shape]
        # Get index of the dimension we want to have stride 1
        stride_one_idx = None
        this_stride_one_value = None
        for dim in stride_one_values:
            if str(dim) in shape_str:
                stride_one_idx = shape_str.index(str(dim))
                this_stride_one_value = dim
                break

        if stride_one_idx is not None:
            new_order[dname] = [stride_one_idx]

            new_strides = list(desc.strides)
            new_strides[stride_one_idx] = sympy.S.One

            previous_size = dace.symbolic.symbol(this_stride_one_value)
            previous_stride = sympy.S.One
            for i in range(len(new_strides)):
                if i != stride_one_idx:
                    new_order[dname].append(i)
                    new_strides[i] = previous_size * previous_stride
                    previous_size = desc.shape[i]
                    previous_stride = new_strides[i]

            new_strides_map[dname] = {}
            # Create a map entry for this data linking old strides to new strides. This assumes that each entry in
            # strides is unique which is given as otherwise there would be two dimension i, j where a[i, j] would point
            # to the same address as a[j, i]
            for new_stride, old_stride in zip(new_strides, desc.strides):
                new_strides_map[dname][old_stride] = new_stride
            desc.strides = tuple(new_strides)
        else:
            parent_name = array_names_map[dname] if dname in array_names_map else dname
            if parent_name in new_strides_map:
                new_strides = []
                for stride in desc.strides:
                    new_strides.append(new_strides_map[parent_name][stride])
                desc.strides = new_strides

    # Add new flipped arrays for every non-transient array
    flipped_names_map = {}
    for dname, desc in sdfg.arrays.items():
        if not desc.transient:
            flipped_name = f"{dname}_flipped"
            flipped_names_map[dname] = flipped_name
            new_sdfg.add_array(flipped_name, desc.shape, desc.dtype,
                               desc.storage, desc.location, True,
                               desc.strides, desc.offset)

    # Deal with the inputs: Create tasklet to flip them and connect via memlets
    # for input in inputs:
    for input in set([*inputs, *outputs]):
        if input in new_order:
            flipped_data = flipped_names_map[input]
            if input in inputs:
                changed_stride_state.add_memlet_path(changed_stride_state.add_access(flipped_data), nsdfg,
                                                     dst_conn=input, memlet=Memlet(data=flipped_data))
            # Simply need to copy the data, the different strides take care of the transposing
            arr = sdfg.arrays[input]
            tasklet, map_entry, map_exit = transform_state.add_mapped_tasklet(
                    name=f"transpose_{input}",
                    map_ranges={f"_i{i}": f"0:{s}" for i, s in enumerate(arr.shape)},
                    inputs={'_in': Memlet(data=input, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                    code='_out = _in',
                    outputs={'_out': Memlet(data=flipped_data,
                                            subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                    external_edges=True,
                    schedule=schedule,
                    )
    # Do the same for the outputs
    for output in outputs:
        if output in new_order:
            flipped_data = flipped_names_map[output]
            changed_stride_state.add_memlet_path(nsdfg, changed_stride_state.add_access(flipped_data),
                                                 src_conn=output, memlet=Memlet(data=flipped_data))
            # Simply need to copy the data, the different strides take care of the transposing
            arr = sdfg.arrays[output]
            tasklet, map_entry, map_exit = transform_state_back.add_mapped_tasklet(
                    name=f"transpose_{output}",
                    map_ranges={f"_i{i}": f"0:{s}" for i, s in enumerate(arr.shape)},
                    inputs={'_in': Memlet(data=flipped_data,
                                          subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                    code='_out = _in',
                    outputs={'_out': Memlet(data=output, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                    external_edges=True,
                    schedule=schedule,
                    )
    # Deal with any arrays which have not been flipped (should only be scalars). Connect them directly
    for dname, desc in sdfg.arrays.items():
        if not desc.transient and dname not in new_order:
            if dname in inputs:
                changed_stride_state.add_memlet_path(changed_stride_state.add_access(dname), nsdfg, dst_conn=dname,
                                                     memlet=Memlet(data=dname))
            if dname in outputs:
                changed_stride_state.add_memlet_path(nsdfg, changed_stride_state.add_access(dname), src_conn=dname,
                                                     memlet=Memlet(data=dname))

    return new_sdfg
