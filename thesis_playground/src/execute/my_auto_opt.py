from typing import Dict, List, Optional, Set, Tuple
import copy
import sympy

import dace
from dace import dtypes
from dace.data import _prod
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
                  symbols: Dict[str, int] = None,
                  k_caching: bool = False) -> SDFG:
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
        # l2ms = apply_transformation_stepwise(sdfg, [LoopToMap, RefineNestedAccess], program, "loop_to_map")
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0
        if program is not None:
            save_graph(sdfg, program, "after_loop_to_map")

    if k_caching:
        make_klev_outermost_map(sdfg)
        if program is not None:
            save_graph(sdfg, program, "after_make_klev_outermost")

        k_caching_prototype_v1(sdfg, validate, validate_all, device, program)

    # change_strides(sdfg, ('NBLOCKS', ), symbols)
    # if program is not None:
    #     save_graph(sdfg, program, "after_change_strides")

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify(verbose=True, validate_all=True)
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
            sdfg.apply_transformations_repeated([RefineNestedAccess], validate=validate, validate_all=validate_all)
            if program is not None:
                save_graph(sdfg, program, "after_outer_refine_nested_access")

    return sdfg


def make_klev_outermost_map(sdfg: SDFG):
    transformed = True
    number_transformed = 0
    while transformed:
        transformations = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapInterchange])]
        transformed = False
        for xform in transformations:
            if xform.inner_map_entry.range.ranges[0][1] == 137:
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
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=transformations, permissive=True)]
    count = 0
    while len(xforms) > 1:
        count += 1
        print(f"[my_auto_opt::apply_transformation_stepwise] apply {xforms[0]}")
        xforms[0].apply(sdfg.sdfg_list[xforms[0].sdfg_id].find_state(xforms[0].state_id),
                        sdfg.sdfg_list[xforms[0].sdfg_id])
        save_graph(sdfg, program, f"after_{description}")
        xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=transformations, permissive=True)]
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


def k_caching_prototype_v1(sdfg: SDFG, validate: bool, validate_all: bool, device: dace.DeviceType, program: Optional[str] = None):
    # Force KLEV loop with vertical dependency into a map
    xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap], permissive=True)]
    if len(xforms) == 1:
        xform = xforms[0]
        xform.apply(sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id), sdfg.sdfg_list[xform.sdfg_id])
    if program is not None:
        save_graph(sdfg, program, "after_force_klev_to_map")

    # Before MapCollapse first to allow MapFusion to work correctly
    sdfg.apply_transformations_repeated([MapCollapse])
    if program is not None:
        save_graph(sdfg, program, "after_map_collapse")

    # Apply TrivialMapElimination before Fusion to avoid problems with maps overt KLON=1
    sdfg.apply_transformations_repeated([TrivialMapElimination])
    if program is not None:
        save_graph(sdfg, program, "after_trivial_map_elimination")

    sdfg.simplify()
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    sdfg.apply_transformations_repeated([MapExpansion])
    if program is not None:
        save_graph(sdfg, program, "after_map_expansion")

    greedy_fuse(sdfg, device=device, validate_all=validate_all, k_caching_args={'max_difference_start': 1,
                                                                                'max_difference_end': 1})
    if program is not None:
        save_graph(sdfg, program, "after_greedy_fuse")

    # sdfg.apply_transformations_repeated([map_fusion_merge_different_ranges(MapFusion, 1, 1)])
    apply_transformation_stepwise(sdfg, [MapFusion], program, "map_fusion")
    if program is not None:
        save_graph(sdfg, program, "after_map_fusion")

    sdfg.simplify()
    if program is not None:
        save_graph(sdfg, program, "after_simplify")


def change_strides(sdfg: dace.SDFG, stride_one_values: List[str], symbols: Dict[str, int]) -> Dict[str, int]:
    # TODO: Add output about how to transform/permute the input
    premutation = dict()

    for _, name, desc in sdfg.arrays_recursive():
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

            # Do I need to check if it is fortran?
            new_strides = list(desc.strides)
            new_strides[stride_one_idx] = sympy.S.One

            previous_size = dace.symbolic.symbol(this_stride_one_value)
            previous_stride = sympy.S.One
            for i in range(len(new_strides)):
                if i != stride_one_idx:
                    new_strides[i] = previous_size * previous_stride
                    previous_size = desc.shape[i]
                    previous_stride = new_strides[i]

            desc.strides = tuple(new_strides)

# Copied and apdated from the microbenchmark my Alex and Lex
def change_strides_old(sdfg: dace.SDFG, klev_vals: Tuple[int], symbols: Dict[str, int],
                   syms_to_add: Set[str] = None) -> Dict[str, int]:

    permutation = dict()
    syms_to_add = syms_to_add or set()

    for name, desc in sdfg.arrays.items():

        # We target arrays that have KLEV or KLEV + 1 in their shape
        shape_str = [str(s) for s in desc.shape]
        klev_idx = None
        divisor = None
        for v in klev_vals:
            if str(v) in shape_str:
                klev_idx = shape_str.index(str(v))
                divisor = v
                break
        if klev_idx is None:
            continue

        permutation[name] = klev_idx

        is_fortran = (desc.strides[0] == 1)

        # Update the strides
        new_strides = list(desc.strides)
        if is_fortran:
            for idx in range(klev_idx + 1, len(desc.shape)):
                new_strides[idx] /= divisor
        else:
            for idx in range(klev_idx):
                new_strides[idx] /= divisor
        new_strides[klev_idx] = _prod(desc.shape) / divisor
        desc.strides = tuple(new_strides)

        # Go to nested SDFGs
        # Assuming only 1 level of nested SDFG
        for sd in sdfg.all_sdfgs_recursive():

            if sd is sdfg:
                continue

            assert sd.parent_sdfg is sdfg

            for s in syms_to_add:
                if s not in sd.parent_nsdfg_node.symbol_mapping:
                        sd.parent_nsdfg_node.symbol_mapping[s] = s
                        sd.add_symbol(s, dace.int32)

            for nname, ndesc in sd.arrays.items():

                if isinstance(ndesc, dace.data.Scalar):
                    continue
                if ndesc.transient:
                    continue

                nsdfg_node = sd.parent_nsdfg_node
                is_input = True
                edges = list(sd.parent.in_edges_by_connector(nsdfg_node, nname))
                if len(edges) == 0:
                    is_input = False
                    edges = list(sd.parent.out_edges_by_connector(nsdfg_node, nname))
                    if len(edges) == 0:
                        raise ValueError
                edge = edges[0]
                if is_input:
                    src = sd.parent.memlet_path(edge)[0].src
                else:
                    src = sd.parent.memlet_path(edge)[-1].dst
                assert isinstance(src, dace.nodes.AccessNode)
                if src.data not in sdfg.arrays:
                    continue

                subset = edge.data.subset
                squeezed = copy.deepcopy(subset)
                rem_idx = squeezed.squeeze()
                assert len(squeezed) == len(ndesc.shape)
                # inv_sqz_idx = [i for i in range(len(desc.shape)) if i not in sqz_idx]
                nnew_strides = [new_strides[i] for i in rem_idx]
                ndesc.strides = tuple(nnew_strides)

    return permutation
