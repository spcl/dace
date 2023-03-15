from typing import Dict, List
import sympy

import dace
from dace import dtypes
from dace.sdfg import utils as sdutil
from dace.sdfg import SDFG, nodes, infer_types, SDFGState
from dace.transformation.optimizer import Optimizer

from dace.transformation.auto.auto_optimize import greedy_fuse, tile_wcrs, set_fast_implementations,\
                                                   move_small_arrays_to_stack, make_transients_persistent

# Transformations
from dace.transformation.dataflow import TrivialMapElimination, MapCollapse
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation import helpers as xfh

from utils import save_graph


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  program: str = None,
                  validate: bool = True,
                  validate_all: bool = False,
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
    debugprint = True

    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_trivial_map_elimination")
    if device == dace.DeviceType.GPU:
        loop_to_map_outside_first(sdfg, program, validate, validate_all)
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        if program is not None:
            save_graph(sdfg, program, "after_simplify")
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        if program is not None:
            sdfg.save(f"{sdfg.hash_sdfg()[:5]}.sdfg")
        print(f"Performed {l2ms} transformations")
        transformed = l2ms > 0

    if program is not None:
        save_graph(sdfg, program, "after_loop_to_map")
    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    if program is not None:
        save_graph(sdfg, program, "after_simplify")

    # fuse subgraphs greedily
    sdfg.simplify()

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
        save_graph(sdfg, program, "after_map_colapse.")
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
            print("Specializing the SDFG for symbols", known_symbols)
        sdfg.specialize(known_symbols)

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg


def loop_to_map_outside_first(
        sdfg: SDFG,
        program: str,
        validate: bool = True,
        validate_all: bool = False) -> SDFG:

    number_of_transformations_performed = 1
    while number_of_transformations_performed > 0:
        outside_loop_transformations = []
        transformations = [xform for xform in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap])]
        for xform in transformations:
            print(f"[gen_graphs::my_optimize] Consider transformation with guard: {xform.loop_guard}")
            is_outside_loop = True
            sdfg.all_sdfgs_recursive()
            for other_form in transformations:
                if other_form != xform:
                    other_states: List[SDFGState] = list(sdutil.dfs_conditional(
                        sdfg.sdfg_list[xform.sdfg_id], [other_form.loop_begin], lambda _, c: c is not other_form.loop_guard))
                    if xform.loop_guard in other_states:
                        is_outside_loop = False
            if is_outside_loop:
                outside_loop_transformations.append(xform)
                print(f"[gen_graphs::my_optimize] add transformation with guard {xform.loop_guard} to list")

        print(f"[gen_graphs::my_optimize] # of outer loops: {len(outside_loop_transformations)}")
        number_of_transformations_performed = len(outside_loop_transformations)
        for xform in outside_loop_transformations:
            # Don't know what to pass as the 1st argument
            print(f"[gen_graphs::my_optimize] Apply with guard {xform.loop_guard}")
            xform.apply(None, sdfg.sdfg_list[xform.sdfg_id])
            if program is not None:
                save_graph(sdfg, program, "after_loop_to_map_in_loop_to_map_outside_first")

    return sdfg
