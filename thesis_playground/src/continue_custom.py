from argparse import ArgumentParser
from typing import List 
import os
import logging
import dace
import copy
from dace import memlet
from dace.sdfg import SDFG, SDFGState, nodes, InterstateEdge
from dace.sdfg.graph import SubgraphView
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlets_state
from dace.dtypes import ScheduleType
from dace.transformation.auto.auto_optimize import greedy_fuse
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation import helpers as xfh
from dace.transformation.subgraph import helpers as xfsh
from dace.sdfg.nodes import NestedSDFG, MapEntry
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import MoveLoopIntoMap, LoopUnroll, StateFusion
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate.loop_detection import find_for_loop
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.auto.auto_optimize import get_composite_fusion

from execute.parameters import ParametersProvider
from execute.my_auto_opt import auto_optimize_phase_2, make_outermost_map, apply_subgraph_fusion, is_map_over_symbol, \
                                k_caching_prototype_v1_fuse, change_strides, fix_all_strange_memlets, \
                                k_caching_prototype_v1_prepare_fusion
from utils.general import save_graph, replace_symbols_by_values
from utils.log import setup_logging
from utils.cli_frontend import add_cloudsc_size_arguments
from utils.general import reset_graph_files
from utils.paths import get_full_cloudsc_log_dir

logger = logging.getLogger(__name__)


def loop_to_nsdfg(sdfg: SDFG, guard: SDFGState, body: SDFGState, itervar: str):
    # Copied from LoopToMap
    itervar, (start, end, step), (_, body_end) = find_for_loop(sdfg, guard, body, itervar=itervar)
    # Find all loop-body states
    states = set()
    to_visit = [body]
    while to_visit:
        state = to_visit.pop(0)
        for _, dst, _ in sdfg.out_edges(state):
            if dst not in states and dst is not guard:
                to_visit.append(dst)
        states.add(state)

    nsdfg = None

    logger.debug("Nest states, there are %i states", len(states))
    # Find read/write sets
    read_set, write_set = set(), set()
    for state in states:
        rset, wset = state.read_and_write_sets()
        read_set |= rset
        write_set |= wset
        logger.debug("state: %s: Add %s to read set and %s to write set", state, rset, wset)
        # Add to write set also scalars between tasklets
        for src_node in state.nodes():
            if not isinstance(src_node, nodes.Tasklet):
                continue
            for dst_node in state.nodes():
                if src_node is dst_node:
                    continue
                if not isinstance(dst_node, nodes.Tasklet):
                    continue
                for e in state.edges_between(src_node, dst_node):
                    if e.data.data and e.data.data in sdfg.arrays:
                        write_set.add(e.data.data)
        # Add data from edges
        for src in states:
            for dst in states:
                for edge in sdfg.edges_between(src, dst):
                    for s in edge.data.free_symbols:
                        if s in sdfg.arrays:
                            # logger.debug("Add %s to read set as it is a free symbol in edge %s -> %s (%s)", s,
                            #              edge.src, edge.dst, edge.data)
                            read_set.add(s)

    # Find NestedSDFG's unique data
    rw_set = read_set | write_set
    unique_set = set()
    for name in rw_set:
        if not sdfg.arrays[name].transient:
            continue
        found = False
        # Look if there is an access outside of the states inside the loop
        for state in sdfg.states():
            if state in states:
                continue
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.data == name):
                    found = True
                    break
        if not found and LoopToMap._is_array_thread_local(name, itervar, sdfg, states):
            unique_set.add(name)

    # Find NestedSDFG's connectors
    read_set = {n for n in read_set if n not in unique_set or not sdfg.arrays[n].transient}
    write_set = {n for n in write_set if n not in unique_set or not sdfg.arrays[n].transient}

    # Create NestedSDFG and add all loop-body states and edges
    # Also, find defined symbols in NestedSDFG
    fsymbols = set(sdfg.free_symbols)
    new_body = sdfg.add_state('single_state_body')
    nsdfg = SDFG(f"loop_body_of_{itervar}", constants=sdfg.constants_prop, parent=new_body)
    nsdfg.add_node(body, is_start_state=True)
    body.parent = nsdfg
    exit_state = nsdfg.add_state('exit')
    nsymbols = dict()
    for state in states:
        if state is body:
            continue
        nsdfg.add_node(state)
        state.parent = nsdfg
    for state in states:
        if state is body:
            continue
        for src, dst, data in sdfg.in_edges(state):
            nsymbols.update({s: sdfg.symbols[s] for s in data.assignments.keys() if s in sdfg.symbols})
            nsdfg.add_edge(src, dst, data)
    nsdfg.add_edge(body_end, exit_state, InterstateEdge())

    # Move guard -> body edge to guard -> new_body
    for src, dst, data, in sdfg.edges_between(guard, body):
        sdfg.add_edge(src, new_body, data)
    # Move body_end -> guard edge to new_body -> guard
    for src, dst, data in sdfg.edges_between(body_end, guard):
        sdfg.add_edge(new_body, dst, data)

    # Delete loop-body states and edges from parent SDFG
    for state in states:
        for e in sdfg.all_edges(state):
            sdfg.remove_edge(e)
        sdfg.remove_node(state)

    # Add NestedSDFG arrays
    for name in read_set | write_set:
        nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])
        nsdfg.arrays[name].transient = False
    for name in unique_set:
        nsdfg.arrays[name] = sdfg.arrays[name]
        del sdfg.arrays[name]

    # Add NestedSDFG node
    cnode = new_body.add_nested_sdfg(nsdfg, None, read_set, write_set)
    logger.debug("Add nsdfg with read: %s and writes: %s and name: %s", read_set, write_set, cnode.label)
    if sdfg.parent:
        for s, m in sdfg.parent_nsdfg_node.symbol_mapping.items():
            if s not in cnode.symbol_mapping:
                cnode.symbol_mapping[s] = m
                nsdfg.add_symbol(s, sdfg.symbols[s])
    for name in read_set:
        r = new_body.add_read(name)
        new_body.add_edge(r, None, cnode, name, memlet.Memlet.from_array(name, sdfg.arrays[name]))
    for name in write_set:
        w = new_body.add_write(name)
        new_body.add_edge(cnode, name, w, None, memlet.Memlet.from_array(name, sdfg.arrays[name]))

    # Fix SDFG symbols
    for sym in sdfg.free_symbols - fsymbols:
        if sym in sdfg.symbols:
            del sdfg.symbols[sym]
    for sym, dtype in nsymbols.items():
        nsdfg.symbols[sym] = dtype

    # Reset all nested SDFG parent pointers
    if nsdfg is not None:
        if isinstance(nsdfg, nodes.NestedSDFG):
            nsdfg = nsdfg.sdfg

        for nstate in nsdfg.nodes():
            for nnode in nstate.nodes():
                if isinstance(nnode, nodes.NestedSDFG):
                    nnode.sdfg.parent_nsdfg_node = nnode
                    nnode.sdfg.parent = nstate
                    nnode.sdfg.parent_sdfg = nsdfg

def loop_to_nsdfg_2(sdfg: SDFG, guard: SDFGState, body: SDFGState, itervar: str):
    # Copied from LoopToMap
    itervar, (start, end, step), (_, body_end) = find_for_loop(sdfg, guard, body, itervar=itervar)
    # Find all loop-body states
    states = set()
    # edges = set()
    to_visit = [body]
    new_body = sdfg.add_state('single_state_body')
    while to_visit:
        state = to_visit.pop(0)
        for edge in sdfg.out_edges(state):
            if edge.dst not in states and edge.dst is not guard:
                to_visit.append(edge.dst)
        states.add(state)
        # subgraph.add_node(state)
        # subgraph.add_edge(edge.src, edge.dst, edge.data)
        # edges.append(edge)

    subgraph = SubgraphView(sdfg, states)
    nsdfg = nest_state_subgraph(sdfg, new_body, subgraph, full_data=False)



def continue_full_cloudsc_fuse(sdfg: SDFG, device: dace.DeviceType, verbose_name: str):
    validate = True
    validate_all = True
    params = ParametersProvider('cloudscexp4', update={'NBLOCKS': 1})
    symbols = params.get_dict()
    full_cloudsc_fixes = True
    program = verbose_name
    # k_caching_prototype_v1_prepare_fusion(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)
    k_caching_prototype_v1_fuse(sdfg, validate, validate_all, device, symbols, program, full_cloudsc_fixes)
    # save_graph(sdfg, verbose_name, "after_fuse")
    # auto_optimize_phase_2(sdfg, device, program, validate, validate_all, symbols, k_caching=False,
    #                       move_assignments_outside=True, storage_on_gpu=False, full_cloudsc_fixes=True, skip_fusing=True)
    # auto_optimize_phase_2(sdfg, device, program, validate, validate_all, symbols, k_caching=False, storage_on_gpu=False,
    #                       move_assignments_outside=False, full_cloudsc_fixes=True, skip_fusing=True)

    if False:
        schedule = ScheduleType.GPU_Device if device == dace.DeviceType.GPU else ScheduleType.Default
        schedule = ScheduleType.Default
        logger.info("Change strides using schedule %s", schedule)
        sdfg = change_strides(sdfg, ('NBLOCKS', ), schedule)
        logger.info("Set gpu block size to (32, 1, 1)")
        for state in sdfg.states():
            for node, state in state.all_nodes_recursive():
                if isinstance(node, MapEntry):
                    logger.debug(f"Set block size for {node}")
                    node.map.gpu_block_size = (32, 1, 1)
    sdfg_path = os.path.join(get_full_cloudsc_log_dir(), "cloudscexp4_all_opt_custom.sdfg")     
    sdfg.save(sdfg_path)
    logger.info(f"Save into {sdfg_path}")


def force_klev_to_maps(sdfg: SDFG, verbose_name: str, blacklist_states: List = None, whitelist_states: List = None):
    to_transform = True
    while to_transform:
        xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[LoopToMap], permissive=True)]
        to_transform = False
        if len(xforms) > 0:
            for xform in xforms:
                xf_sdfg = sdfg.sdfg_list[xform.sdfg_id]
                xf_state = sdfg.sdfg_list[xform.sdfg_id].find_state(xform.state_id)
                if (
                        (blacklist_states is not None and xf_state.name not in blacklist_states and not to_transform) or
                        (whitelist_states is not None and xf_state.name in whitelist_states and not to_transform) or
                        (whitelist_states is None and blacklist_states is None)
                    ):
                    to_transform = True
                    loop_guard = xform.loop_guard.name
                    logger.debug(f"Apply LoopToMap to state {xf_state} in {xf_sdfg.label} with guard {loop_guard}")
                    xform.additional_rw.add('ZQXNM1')
                    xform.apply(xf_state, xf_sdfg)
                    save_graph(sdfg, verbose_name, f"after_loop_to_map_{xf_state}_{loop_guard}")
                    sdfg.validate()
                    break
    if verbose_name is not None:
        save_graph(sdfg, verbose_name, "after_force_klev_to_map")


def main():
    parser = ArgumentParser()
    parser.add_argument('sdfg_file', type=str, help='Path to the sdfg file to load')
    parser.add_argument('--verbose-name',
                        type=str,
                        default=None,
                        help='Foldername under which intermediate SDFGs should be stored, uses the program name by '
                        'default')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--log-level', default='DEBUG', help='Log level for console, defaults to DEBUG')
    parser.add_argument('--log-file', default=None)
    add_cloudsc_size_arguments(parser)
    args = parser.parse_args()

    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    add_args = {}
    if args.log_file is not None:
        add_args['full_logfile'] = args.log_file
    setup_logging(level=args.log_level.upper(), **add_args)

    verbose_name = None
    if args.verbose_name is not None:
        verbose_name = args.verbose_name
        reset_graph_files(verbose_name)

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_file)
    sdfg.validate()
    params = ParametersProvider('cloudscexp4', update={'NBLOCKS': 16384})
    symbols = params.get_dict()
    validate_all = True
    continue_full_cloudsc_fuse(sdfg, device, verbose_name)

    # sdfg = fix_all_strange_memlets(sdfg)
    # sdfg.validate()
    # save_graph(sdfg, verbose_name, "after_memlet_fixing")


    # force_klev_to_maps(sdfg, verbose_name, blacklist_states=['state_9'])
    # force_klev_to_maps(sdfg, verbose_name)
    # force_klev_to_maps(sdfg, verbose_name, whitelist_states=['state_9'])

    # continue_search = True
    # while continue_search:
    #     xforms = [xf for xf in Optimizer(sdfg).get_pattern_matches(patterns=[MapToForLoop], permissive=True)]
    #     logger.debug("Found %i many possible transformations to transform map back to for-loop", len(xforms))
    #     continue_search = False
    #     for xf in xforms:
    #         # expect that maps only have one dimension, as we did the MapExpansion transformation before
    #         xf_sdfg = sdfg.sdfg_list[xf.sdfg_id]
    #         xf_state = xf_sdfg.find_state(xf.state_id)
    #         if is_map_over_symbol(xf.map_entry.map.range, symbols, 'KLEV', symbols['NCLDTOP']+1):
    #             continue_search = True
    #             logger.debug("Found the correct map. Apply it to state %s and sdfg %s", xf_state.name, xf_sdfg.label)
    #             xf.apply(xf_state, xf_sdfg)
    #             if verbose_name is not None:
    #                 save_graph(sdfg, verbose_name, "after_map_to_for_loop")
    #             break
    # sdfg.validate()


    # cloudsc_state = sdfg.find_state('stateCLOUDSC')
    # cloudsc_nsdfg = [n for n in cloudsc_state.nodes() if isinstance(n, NestedSDFG)][0]
    # loop_to_nsdfg(cloudsc_nsdfg.sdfg, cloudsc_nsdfg.sdfg.find_state('GuardFOR_l_1921_c_1921'),
    #               cloudsc_nsdfg.sdfg.find_state('single_state_body_11'), '_for_it_21')
    # force_klev_to_maps(sdfg, verbose_name, whitelist_states=['state_9'])
    # save_graph(sdfg, verbose_name, "after_loop_to_nsdfg")
    # sdfg_path = os.path.join(get_full_cloudsc_log_dir(), "cloudscexp4_all_opt_custom.sdfg")     
    # sdfg.save(sdfg_path)
    # logger.info(f"Save into {sdfg_path}")
    # sdfg.simplify()
    # sdfg.validate()
    # continue_full_cloudsc_fuse(sdfg, device, verbose_name)
    # sdfg.apply_transformations_repeated([StateFusion])
    # sdfg.apply_gpu_transformations()
    # save_graph(sdfg, verbose_name, "after_gpu_transformations")
    # apply_subgraph_fusion(sdfg, {
    #     'max_difference_start': symbols['NCLDTOP']+1,
    #     'max_difference_end': 1,
    #     'disjoint_subsets': False,
    #     'is_map_sequential': lambda map: (str(map.range.ranges[0][1]) == 'KLEV' or map.range.ranges[0][1] ==
    #                                       symbols['KLEV']),
    #     'fixed_new_shapes': {'ZQXN2D': [symbols['KLON'], 1, symbols['NCLV']]},
    #     # 'forced_subgraph_contains_data': set(['ZTP1', 'ZLI', 'ZLNEG', 'ZPFPLSX'])
    #     'forced_subgraph_contains_data': set(['ZTP1', 'ZLI' ])
    #     },
    #         symbols, verbose_name)


if __name__ == '__main__':
    main()
