from argparse import ArgumentParser
import os
from glob import glob
from os import path
from typing import Dict
import sympy

import dace
from dace.sdfg import infer_types
from dace import config, dtypes
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils, SDFG, nodes
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion

from dace.transformation.auto.auto_optimize import greedy_fuse, tile_wcrs, set_fast_implementations, move_small_arrays_to_stack, make_transients_persistent

# Transformations
from dace.transformation.dataflow import TrivialMapElimination, MapCollapse
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation import helpers as xfh

from test import read_source
from utils import get_programs_data


counter = 0
graphs_dir = path.join(path.dirname(__file__), 'sdfg_graphs')


def save_graph(sdfg: SDFG, program: str, name: str):
    global counter
    sdfg.save(path.join(graphs_dir, f"{program}_{counter}_{name}.sdfg"))
    counter = counter + 1


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  program: str,
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
    save_graph(sdfg, program, "after_trivial_map_elimination")
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        save_graph(sdfg, program, "after_simplify")
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        sdfg.save(f"{sdfg.hash_sdfg()[:5]}.sdfg")
        print(f"Performed {l2ms} transformations")
        transformed = l2ms > 0

    save_graph(sdfg, program, "after_loop_to_map")
    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    save_graph(sdfg, program, "after_simplify")

    # fuse subgraphs greedily
    sdfg.simplify()

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)
    save_graph(sdfg, program, "after_greedy_fuse")

    # Move Loops inside Maps when possible
    # from dace.transformation.interstate import MoveLoopIntoMap
    # sdfg.apply_transformations_repeated([MoveLoopIntoMap])

    # Apply GPU transformations and set library node implementations
    if device == dtypes.DeviceType.GPU:
        sdfg.apply_gpu_transformations()
        sdfg.simplify()

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


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)

    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    if normalize_offsets:
        my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    else:
        my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    if normalize_offsets:
        utils.normalize_offsets(sdfg)

    return sdfg


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'program',
            type=str,
            help='Name of the program to generate the SDFGs of')
    parser.add_argument(
            '--normalize-memlets',
            action='store_true',
            default=False)
    parser.add_argument(
            '--only-graph',
            action='store_true',
            help='Does not compile the SDFGs into C++ code, only creates the SDFGs and runs the transformations')

    device = dace.DeviceType.GPU
    args = parser.parse_args()

    for file in glob(os.path.join(graphs_dir, f"{args.program}_*.sdfg")):
        os.remove(file)

    programs = get_programs_data()['programs']
    fsource = read_source(args.program)
    program_name = programs[args.program]
    sdfg = get_sdfg(fsource, program_name, args.normalize_memlets)
    save_graph(sdfg, args.program, "before_auto_opt")
    auto_optimize(sdfg, device, args.program)
    save_graph(sdfg, args.program, "after_auto_opt")
    sdfg.instrument = dace.InstrumentationType.Timer
    if not args.only_graph:
        sdfg.compile()
    return sdfg


if __name__ == '__main__':
    main()
