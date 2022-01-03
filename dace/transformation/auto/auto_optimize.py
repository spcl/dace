# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

import dace
import sympy
from dace.sdfg.state import SDFGState
from dace.sdfg.graph import SubgraphView
from dace.sdfg.propagation import propagate_states
from dace.sdfg.scope import is_devicelevel_gpu
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import SDFG, nodes, graph as gr
from typing import Set, Tuple, Union, List, Iterable, Dict
import warnings

# Transformations
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, MapFusion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import ReduceExpansion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

# Enumerator
from dace.transformation.estimator.enumeration import GreedyEnumerator

# FPGA AutoOpt
from dace.transformation.auto import fpga as fpga_auto_opt

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def greedy_fuse(graph_or_subgraph: GraphViewType,
                validate_all: bool,
                device: dace.dtypes.DeviceType = dace.dtypes.DeviceType.CPU,
                recursive: bool = True,
                stencil: bool = False,
                stencil_tile=None,
                permutations_only: bool = True,
                expand_reductions: bool = False) -> None:
    '''
    Greedily fuses maps of an SDFG or graph, operating in-place.
    :param graph_or_subgraph: SDFG, SDFGState or Subgraph
    :param validate_all: Validate SDFG or graph at each fusion step 
    :param device: Device type to specialize for 
    :param recursive: Fuse recursively within (fused and unfused) scopes
    :param stencil: Perform stencil fusion instead of regular fusion 
    :param stencil_tile: StencilTiling Tile size, default if None
    :param permutations_only: Disallow splitting of maps during MultiExpansion stage
    :param expand_reductions: Expand all reduce nodes before fusion
    '''
    debugprint = config.Config.get_bool('debugprint')
    if isinstance(graph_or_subgraph, SDFG):
        # If we have an SDFG, recurse into graphs
        graph_or_subgraph.coarsen_dataflow(validate_all=validate_all)
        # MapFusion for trivial cases
        graph_or_subgraph.apply_transformations_repeated(MapFusion, validate_all=validate_all)
        # recurse into graphs
        for graph in graph_or_subgraph.nodes():

            greedy_fuse(graph,
                        validate_all=validate_all,
                        device=device,
                        recursive=recursive,
                        stencil=stencil,
                        stencil_tile=stencil_tile,
                        permutations_only=permutations_only,
                        expand_reductions=expand_reductions)
    else:
        # we are in graph or subgraph
        sdfg, graph, subgraph = None, None, None
        if isinstance(graph_or_subgraph, SDFGState):
            sdfg = graph_or_subgraph.parent
            sdfg.apply_transformations_repeated(MapFusion, validate_all=validate_all)
            graph = graph_or_subgraph
            subgraph = SubgraphView(graph, graph.nodes())
        else:
            sdfg = graph_or_subgraph.graph.parent
            graph = graph_or_subgraph.graph
            subgraph = graph_or_subgraph

        # create condition function object
        fusion_condition = CompositeFusion(SubgraphView(graph, graph.nodes()))

        # within SDFGState: greedily enumerate fusible components
        # and apply transformation
        applied_transformations = 0
        reverse = True if stencil else False

        if stencil:
            # adjust tiling settings
            fusion_condition.allow_tiling = True
            fusion_condition.schedule_innermaps = dtypes.ScheduleType.Sequential
            if device == dtypes.DeviceType.GPU:
                fusion_condition.stencil_unroll_loops = True
            # tile size
            if stencil_tile:
                fusion_condition.stencil_strides = stencil_tile
            # always only permutate for now with stencil tiles
            fusion_condition.expansion_split = False

        else:
            fusion_condition.allow_tiling = False
            # expand reductions
            if expand_reductions:
                for graph in sdfg.nodes():
                    for node in graph.nodes():
                        if isinstance(node, dace.libraries.standard.nodes.Reduce):
                            try:
                                ReduceExpansion.apply_to(sdfg, _reduce=node)
                            except ValueError as e:
                                pass
            # permutation settings
            fusion_condition.expansion_split = not permutations_only

        condition_function = lambda sdfg, subgraph: fusion_condition.can_be_applied(sdfg, subgraph)
        enumerator = GreedyEnumerator(sdfg, graph, subgraph, condition_function=condition_function)
        for map_entries in enumerator:
            if len(map_entries) > 1:
                current_subgraph = xfsh.subgraph_from_maps(sdfg, graph, map_entries)
                cf = CompositeFusion(current_subgraph)
                # transfer settings
                cf.allow_tiling = fusion_condition.allow_tiling
                cf.schedule_innermaps = fusion_condition.schedule_innermaps
                cf.expansion_split = fusion_condition.expansion_split
                cf.stencil_strides = fusion_condition.stencil_strides

                cf.apply(sdfg)
                applied_transformations += 1

            if recursive:
                global_entry = cf._global_map_entry if len(map_entries) > 1 else map_entries[0]

                greedy_fuse(graph.scope_subgraph(global_entry, include_entry=False, include_exit=False),
                            validate_all=validate_all,
                            device=device,
                            recursive=recursive,
                            stencil=stencil,
                            stencil_tile=stencil_tile,
                            permutations_only=permutations_only,
                            expand_reductions=expand_reductions)

        for node in graph_or_subgraph.nodes():
            if isinstance(node, nodes.NestedSDFG):
                greedy_fuse(node.sdfg,
                            validate_all=validate_all,
                            device=device,
                            stencil=stencil,
                            stencil_tile=stencil_tile,
                            recursive=recursive,
                            permutations_only=permutations_only,
                            expand_reductions=expand_reductions)

        if applied_transformations > 0:
            if debugprint:
                if stencil:
                    print(f"Applied {applied_transformations} TileFusion")
                else:
                    print(f"Applied {applied_transformations} SubgraphFusion")

        if validate_all:
            graph.validate()


def tile_wcrs(graph_or_subgraph: GraphViewType, validate_all: bool, prefer_partial_parallelism: bool = None) -> None:
    """
    Tiles parallel write-conflict resolution maps in an SDFG, state,
    or subgraphs thereof. Reduces the number of atomic operations by tiling
    and introducing transient arrays to accumulate atomics on.
    :param graph_or_subgraph: The SDFG/state/subgraph to optimize within.
    :param validate_all: If True, runs SDFG validation after every tiling.
    :param prefer_partial_parallelism: If set, prefers extracting non-conflicted
                                       map dimensions over tiling WCR map (may
                                       not perform well if parallel dimensions
                                       are small).
    :note: This function operates in-place.
    """
    # Avoid import loops
    from dace.codegen.targets import cpp
    from dace.frontend import operations
    from dace.transformation import dataflow, helpers as xfh

    # Determine on which nodes to run the operation
    graph = graph_or_subgraph
    if isinstance(graph_or_subgraph, gr.SubgraphView):
        graph = graph_or_subgraph.graph
    if isinstance(graph, SDFG):
        for state in graph_or_subgraph.nodes():
            tile_wcrs(state, validate_all)
        return
    if not isinstance(graph, SDFGState):
        raise TypeError('Graph must be a state, an SDFG, or a subgraph of either')
    sdfg = graph.parent

    edges_to_consider: Set[Tuple[gr.MultiConnectorEdge[Memlet], nodes.MapEntry]] = set()
    for edge in graph_or_subgraph.edges():
        if edge.data.wcr is not None:
            if (isinstance(edge.src, (nodes.MapExit, nodes.NestedSDFG)) or isinstance(edge.dst, nodes.MapEntry)):
                # Do not consider intermediate edges
                continue
            reason = cpp.is_write_conflicted_with_reason(graph, edge)
            if reason is None or not isinstance(reason, nodes.MapEntry):
                # Do not consider edges that will not generate atomics or
                # atomics we cannot transform
                continue
            if reason not in graph_or_subgraph.nodes():
                # Skip if conflict exists outside of nested SDFG
                continue

            # Check if identity value can be inferred
            redtype = operations.detect_reduction_type(edge.data.wcr)
            dtype = sdfg.arrays[edge.data.data].dtype
            identity = dtypes.reduction_identity(dtype, redtype)
            if identity is None:  # Cannot infer identity value
                continue

            edges_to_consider.add((edge, reason))

    tile_size = config.Config.get('optimizer', 'autotile_size')
    debugprint = config.Config.get_bool('debugprint')
    if prefer_partial_parallelism is None:
        prefer_partial_parallelism = config.Config.get_bool('optimizer', 'autotile_partial_parallelism')

    maps_to_consider: Set[nodes.MapEntry] = set(me for _, me in edges_to_consider)

    transformed: Set[nodes.MapEntry] = set()

    # Heuristic: If the map is only partially conflicted, extract
    # parallel dimensions instead of tiling
    if prefer_partial_parallelism:
        for mapentry in maps_to_consider:
            # Check the write-conflicts of all WCR edges in map
            conflicts: Set[str] = set()
            for edge, me in edges_to_consider:
                if me is not mapentry:
                    continue
                conflicts |= set(cpp.write_conflicted_map_params(mapentry, edge))

            nonconflicted_dims = set(mapentry.params) - conflicts
            if nonconflicted_dims:
                dims = [i for i, p in enumerate(mapentry.params) if p in nonconflicted_dims]
                if ((dt._prod(s for i, s in enumerate(mapentry.range.size()) if i in dims) < tile_size) == True):
                    # Map has a small range, extracting parallelism may not be
                    # beneficial
                    continue
                xfh.extract_map_dims(sdfg, mapentry, dims)
                transformed.add(mapentry)

    # Tile and accumulate other not-transformed maps
    for edge, mapentry in edges_to_consider:
        if mapentry in transformed:
            continue
        transformed.add(mapentry)

        # NOTE: The test "(x < y) == True" below is crafted for SymPy
        # to be "definitely True"
        if all((s < tile_size) == True for s in mapentry.map.range.size()):
            # If smaller than tile size, don't transform and instead
            # make map sequential
            if debugprint:
                print(f'Making map "{mapentry}" sequential due to being ' 'smaller than tile size')
            mapentry.map.schedule = dtypes.ScheduleType.Sequential
            continue

        # MapTiling -> AccumulateTransient / AccumulateStream
        outer_mapentry = dataflow.MapTiling.apply_to(sdfg, dict(tile_sizes=(tile_size, )), map_entry=mapentry)

        # Transform all outgoing WCR and stream edges
        mapexit = graph.exit_node(mapentry)
        outer_mapexit = graph.exit_node(outer_mapentry)

        # Tuple of (transformation type, options, pattern)
        to_apply: Tuple[Union[dataflow.StreamTransient, dataflow.AccumulateTransient], Dict[str, Any],
                        Dict[str, nodes.Node]] = None
        for e in graph.out_edges(mapexit):
            if isinstance(sdfg.arrays[e.data.data], dt.Stream):
                mpath = graph.memlet_path(e)
                tasklet = mpath[0].src
                if not isinstance(tasklet, nodes.Tasklet) or len(mpath) != 3:
                    # TODO(later): Implement StreamTransient independently of tasklet
                    continue

                # Make transient only if there is one WCR/stream
                if to_apply is not None:
                    to_apply = None
                    break

                to_apply = (dataflow.StreamTransient, {},
                            dict(tasklet=tasklet, map_exit=mapexit, outer_map_exit=outer_mapexit))
            else:
                if (e.data.is_empty() or e.data.wcr is None or e.data.wcr_nonatomic or
                    (e.data.dst_subset is not None and e.data.dst_subset.num_elements() != 0 and e.data.dynamic)):
                    continue

                dtype = sdfg.arrays[e.data.data].dtype
                redtype = operations.detect_reduction_type(e.data.wcr)
                identity = dtypes.reduction_identity(dtype, redtype)
                if identity is None:  # Cannot infer identity value
                    continue
                # Make transient only if there is one WCR/stream
                if to_apply is not None:
                    to_apply = None
                    break

                to_apply = (dataflow.AccumulateTransient, dict(identity=identity, array=e.data.data),
                            dict(map_exit=mapexit, outer_map_exit=outer_mapexit))
        if to_apply is not None:
            xform, opts, pattern = to_apply
            xform.apply_to(sdfg, options=opts, **pattern)

    if debugprint and len(transformed) > 0:
        print(f'Optimized {len(transformed)} write-conflicted maps')


def find_fast_library(device: dtypes.DeviceType) -> List[str]:
    # Returns the optimized library node implementations for the given target
    # device
    if device is dtypes.DeviceType.GPU:
        return ['cuBLAS', 'CUB', 'pure']
    elif device is dtypes.DeviceType.FPGA:
        return ['FPGA_PartialSums', 'FPGAPartialReduction', 'FPGA_Accumulate', 'FPGA1DSystolic', 'pure']
    elif device is dtypes.DeviceType.CPU:
        result = []

        # BLAS calls
        if mkl.IntelMKL.is_installed():
            result.append('MKL')
        if openblas.OpenBLAS.is_installed():
            result.append('OpenBLAS')

        return result + ['OpenMP', 'pure']

    return ['pure']


def move_small_arrays_to_stack(sdfg: SDFG) -> None:
    """
    Set all Default storage types that are constant sized and less than 
    the auto-tile size to the stack (as StorageType.Register).
    :param sdfg: The SDFG to operate on.
    :note: Operates in-place on the SDFG.
    """
    converted = 0
    tile_size = config.Config.get('optimizer', 'autotile_size')
    for sd, aname, array in sdfg.arrays_recursive():
        if isinstance(array, dt.Stream):
            continue
        if (array.transient and array.storage == dtypes.StorageType.Default
                and array.lifetime == dtypes.AllocationLifetime.Scope):
            if not symbolic.issymbolic(array.total_size, sd.constants):
                eval_size = symbolic.evaluate(array.total_size, sd.constants)
                if (eval_size <= tile_size) == True:
                    array.storage = dtypes.StorageType.Register
                    converted += 1

    if config.Config.get_bool('debugprint') and converted > 0:
        print(f'Statically allocating {converted} transient arrays')


def set_fast_implementations(sdfg: SDFG, device: dtypes.DeviceType, blocklist: List[str] = None):
    """
    Set fast library node implementations for the given device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param blocklist: list of disallowed implementations.
    :note: Operates in-place on the given SDFG.
    """
    if blocklist is None:
        implementation_prio = find_fast_library(device)
    else:
        implementation_prio = [i for i in find_fast_library(device) if i not in blocklist]

    # specialized nodes: pre-expand
    for current_sdfg in sdfg.all_sdfgs_recursive():
        for state in current_sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.LibraryNode):
                    if (node.default_implementation == 'specialize'
                            and (len(set(node.implementations)
                                     & set(implementation_prio))) == 0):
                        node.expand(current_sdfg, state)

    # general nodes
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.LibraryNode):
            for impl in implementation_prio:
                if impl in node.implementations:
                    if isinstance(
                            node,
                            dace.libraries.standard.nodes.reduce.Reduce) and node.implementation == 'CUDA (block)':
                        continue
                    node.implementation = impl
                    break

    # reduce nodes
    if device == dtypes.DeviceType.GPU:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.LibraryNode):
                # Use CUB for device-level reductions
                if ('CUDA (device)' in node.implementations and not is_devicelevel_gpu(state.parent, state, node)
                        and state.scope_dict()[node] is None):
                    node.implementation = 'CUDA (device)'


def make_transients_persistent(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    ''' 
    Helper function to change several storage and scheduling properties
    - Makes non-view array lifetimes persistent, with some 
      restrictions depending on the device 
    - Reset nonatomic WCR edges on GPU 
    :param sdfg: SDFG
    :param device: Device type
    '''
    for nsdfg in sdfg.all_sdfgs_recursive():
        for aname, arr in nsdfg.arrays.items():
            if arr.transient and not isinstance(arr, dt.View) and not symbolic.issymbolic(arr.total_size):
                if arr.storage != dtypes.StorageType.Register:
                    arr.lifetime = dtypes.AllocationLifetime.Persistent

    if device == dtypes.DeviceType.GPU:
        for aname, arr in sdfg.arrays.items():
            if arr.transient and not isinstance(arr, dt.View):  #and size only depends on SDFG params
                if arr.storage == dtypes.StorageType.GPU_Global:
                    arr.lifetime = dtypes.AllocationLifetime.Persistent

        # Reset nonatomic WCR edges
        for n, _ in sdfg.all_nodes_recursive():
            if isinstance(n, SDFGState):
                for edge in n.edges():
                    edge.data.wcr_nonatomic = False


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  validate: bool = True,
                  validate_all: bool = False,
                  symbols: Dict[str, int] = None) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:
        * Dataflow coarsening
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
    debugprint = config.Config.get_bool('debugprint')

    # Dataflow coarsening and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)
    while transformed:
        sdfg.coarsen_dataflow(validate=False, validate_all=validate_all)
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0

    # Collapse maps and eliminate trivial dimensions
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)

    # Apply GPU transformations and set library node implementations

    if device == dtypes.DeviceType.GPU:
        sdfg.apply_gpu_transformations()
        sdfg.coarsen_dataflow()

    # fuse subgraphs greedily
    sdfg.coarsen_dataflow()

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)

    if device == dtypes.DeviceType.FPGA:
        # apply FPGA Transformations
        sdfg.apply_fpga_transformations()
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # Set all library nodes to expand to fast library calls
        set_fast_implementations(sdfg, device)
        return sdfg

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

    # Set all library nodes to expand to fast library calls
    set_fast_implementations(sdfg, device)

    sdfg.expand_library_nodes()

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections on a per-SDFG basis
    for nsdfg in sdfg.all_sdfgs_recursive():
        nsdfg.openmp_sections = False

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

    # Set all Default storage types that are constant sized to registers
    move_small_arrays_to_stack(sdfg)
    '''
    # Fix storage and allocation properties, e.g., for benchmarking purposes
    # FORNOW: Leave out
    make_transients_persistent(sdfg, device)
    '''

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
