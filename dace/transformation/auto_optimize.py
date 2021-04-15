# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

from dace.sdfg import SDFG, SDFGState
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import SDFG, nodes, graph as gr
from typing import Set, Tuple, Union, List
import warnings

# Transformations
from dace.transformation.dataflow import MapCollapse, MapFusion
from dace.transformation.interstate import LoopToMap

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def greedy_fuse(graph_or_subgraph: GraphViewType, validate_all: bool) -> None:
    """
    Greedily fuses maps in an SDFG or subgraphs thereof. This fuses
    maps if they are connected via a shared access node (parallel or
    in sequence).
    :param graph_or_subgraph: The SDFG or subgraph thereof to fuse within.
    :param validate_all: If True, runs SDFG validation after every fusion.
    :note: This function operates in-place.
    """
    # If two maps share connected nodes (horizontal/vertical), fuse
    # TODO: Run MultiExpansion first, use SubgraphFusion
    sdfg = graph_or_subgraph
    if isinstance(graph_or_subgraph, gr.SubgraphView):
        sdfg = graph_or_subgraph.graph
    if not isinstance(sdfg, SDFG):
        raise TypeError('Graph must be an SDFG or a subgraph thereof')
    states = graph_or_subgraph.nodes()

    # Vertical Fusion
    sdfg.apply_transformations_repeated(MapFusion,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all,
                                        states=states)


def tile_wcrs(graph_or_subgraph: GraphViewType,
              validate_all: bool,
              prefer_partial_parallelism: bool = None) -> None:
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
        raise TypeError(
            'Graph must be a state, an SDFG, or a subgraph of either')
    sdfg = graph.parent

    edges_to_consider: Set[Tuple[gr.MultiConnectorEdge[Memlet],
                                 nodes.MapEntry]] = set()
    for edge in graph_or_subgraph.edges():
        if edge.data.wcr is not None:
            if (isinstance(edge.src, (nodes.MapExit, nodes.NestedSDFG))
                    or isinstance(edge.dst, nodes.MapEntry)):
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
        prefer_partial_parallelism = config.Config.get_bool(
            'optimizer', 'autotile_partial_parallelism')

    maps_to_consider: Set[nodes.MapEntry] = set(me
                                                for _, me in edges_to_consider)

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
                conflicts |= set(cpp.write_conflicted_map_params(
                    mapentry, edge))

            nonconflicted_dims = set(mapentry.params) - conflicts
            if nonconflicted_dims:
                dims = [
                    i for i, p in enumerate(mapentry.params)
                    if p in nonconflicted_dims
                ]
                if ((dt._prod(s for i, s in enumerate(mapentry.range.size())
                              if i in dims) < tile_size) == True):
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
                print(f'Making map "{mapentry}" sequential due to being '
                      'smaller than tile size')
            mapentry.map.schedule = dtypes.ScheduleType.Sequential
            continue

        # MapTiling -> AccumulateTransient / AccumulateStream
        outer_mapentry = dataflow.MapTiling.apply_to(
            sdfg, dict(tile_sizes=(tile_size, )), map_entry=mapentry)

        # Transform all outgoing WCR and stream edges
        mapexit = graph.exit_node(mapentry)
        outer_mapexit = graph.exit_node(outer_mapentry)
        for e in graph.out_edges(mapexit):
            if isinstance(sdfg.arrays[e.data.data], dt.Stream):
                mpath = graph.memlet_path(e)
                tasklet = mpath[0].src
                if not isinstance(tasklet, nodes.Tasklet) or len(mpath) != 3:
                    # TODO(later): Implement StreamTransient independently of tasklet
                    continue
                dataflow.StreamTransient.apply_to(sdfg,
                                                  tasklet=tasklet,
                                                  map_exit=mapexit,
                                                  outer_map_exit=outer_mapexit)
            else:
                if (e.data.is_empty() or e.data.wcr is None
                        or e.data.wcr_nonatomic
                        or (e.data.dst_subset is not None
                            and e.data.dst_subset.num_elements() > 0
                            and e.data.dynamic)):
                    continue

                dtype = sdfg.arrays[e.data.data].dtype
                redtype = operations.detect_reduction_type(e.data.wcr)
                identity = dtypes.reduction_identity(dtype, redtype)
                if identity is None:  # Cannot infer identity value
                    continue
                dataflow.AccumulateTransient.apply_to(
                    sdfg,
                    options=dict(identity=identity, array=e.data.data),
                    map_exit=mapexit,
                    outer_map_exit=outer_mapexit)

    if debugprint and len(transformed) > 0:
        print(f'Optimized {len(transformed)} write-conflicted maps')


def find_fast_library(device: dtypes.DeviceType) -> str:
    # Returns the optimized library node implementations for the given target
    # device
    if device is dtypes.DeviceType.GPU:
        return ['cuBLAS', 'CUB', 'pure']
    elif device is dtypes.DeviceType.CPU:
        result = []

        # BLAS calls
        if mkl.IntelMKL.is_installed():
            result.append('MKL')
        if openblas.OpenBLAS.is_installed():
            result.append('OpenBLAS')

        return result + ['pure']

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
                if eval_size <= tile_size:
                    array.storage = dtypes.StorageType.Register
                    converted += 1

    if config.Config.get_bool('debugprint') and converted > 0:
        print(f'Statically allocating {converted} transient arrays')


def set_fast_implementations(sdfg: SDFG,
                             device: dtypes.DeviceType,
                             blocklist: List[str] = None):
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
        implementation_prio = [
            i for i in find_fast_library(device) if i not in blocklist
        ]

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.LibraryNode):
            for impl in implementation_prio:
                if impl in node.implementations:
                    node.implementation = impl
                    break
            else:
                warnings.warn('No fast library implementation found for "%s", '
                              'falling back to default.' % node.name)


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  validate: bool = True,
                  validate_all: bool = False) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:
        * Strict transformations
        * Strict auto-parallelization (loop-to-map)
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
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """
    # Strict transformations
    sdfg.apply_strict_transformations(validate=False, validate_all=validate_all)

    # Try to parallelize loops
    sdfg.apply_transformations_repeated(LoopToMap,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all)

    # Map fusion
    greedy_fuse(sdfg, validate_all)

    # Tiled WCR and streams
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        tile_wcrs(nsdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.collapse = len(node.map.range)

    # Set all library nodes to expand to fast library calls
    set_fast_implementations(sdfg, device)

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections
    # TODO(later): Set on a per-SDFG basis
    config.Config.set('compiler', 'cpu', 'openmp_sections', value=False)

    # Set all Default storage types that are constant sized to registers
    move_small_arrays_to_stack(sdfg)

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
