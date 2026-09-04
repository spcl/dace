# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

import os

import dace
import sympy
from dace.sdfg import infer_types
from dace.sdfg.state import SDFGState, ControlFlowRegion
from dace.sdfg.graph import SubgraphView
from dace.sdfg.scope import is_devicelevel_gpu_kernel
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import SDFG, nodes, graph as gr
from dace.ordered import OrderedSet
from typing import Any, Callable, Dict, List, Set, Tuple, Union

# Transformations
from dace.transformation.passes import FullMapFusion
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, ReduceExpansion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh, pass_pipeline as ppl

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

# Enumerator
from dace.transformation.estimator.enumeration import GreedyEnumerator

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView, ControlFlowRegion]


def greedy_fuse(graph_or_subgraph: GraphViewType,
                validate_all: bool,
                device: dace.dtypes.DeviceType = dace.dtypes.DeviceType.CPU,
                recursive: bool = True,
                stencil: bool = False,
                stencil_tile=None,
                permutations_only: bool = True,
                expand_reductions: bool = False) -> None:
    """
    Greedily fuses maps of an SDFG or graph, operating in-place.

    :param graph_or_subgraph: SDFG, SDFGState or Subgraph
    :param validate_all: Validate SDFG or graph at each fusion step
    :param device: Device type to specialize for
    :param recursive: Fuse recursively within (fused and unfused) scopes
    :param stencil: Perform stencil fusion instead of regular fusion
    :param stencil_tile: StencilTiling Tile size, default if None
    :param permutations_only: Disallow splitting of maps during MultiExpansion stage
    :param expand_reductions: Expand all reduce nodes before fusion
    """
    debugprint = config.Config.get_bool('debugprint')
    if isinstance(graph_or_subgraph, ControlFlowRegion):
        if isinstance(graph_or_subgraph, SDFG):
            # If we have an SDFG, recurse into graphs
            graph_or_subgraph.simplify(validate_all=validate_all)
            # Apply MapFusionVertical for the more trivial cases
            full_map_fusion_pass = FullMapFusion(
                strict_dataflow=True,
                validate_all=validate_all,
            )
            full_map_fusion_pileline = ppl.Pipeline([full_map_fusion_pass])
            full_map_fusion_pileline.apply_pass(graph_or_subgraph, {})

        # recurse into graphs
        for graph in graph_or_subgraph.nodes():
            if isinstance(graph, (SDFGState, ControlFlowRegion)):
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
            # Apply MapFusionVertical for the more trivial cases.
            #  For backwards compatibility we only perform vertical map fusion.
            full_map_fusion_pass = FullMapFusion(
                strict_dataflow=True,
                validate_all=validate_all,
                perform_horizontal_map_fusion=False,
                perform_vertical_map_fusion=True,
            )
            full_map_fusion_pileline = ppl.Pipeline([full_map_fusion_pass])
            full_map_fusion_pileline.apply_pass(sdfg, {})
            graph = graph_or_subgraph
            subgraph = SubgraphView(graph, graph.nodes())
        else:
            sdfg = graph_or_subgraph.graph.parent
            graph = graph_or_subgraph.graph
            subgraph = graph_or_subgraph

        # create condition function object
        fusion_condition = CompositeFusion()
        fusion_condition.setup_match(SubgraphView(graph, graph.nodes()))

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
                for graph in sdfg.states():
                    for node in graph.nodes():
                        if isinstance(node, dace.libraries.standard.nodes.Reduce):
                            try:
                                ReduceExpansion.apply_to(sdfg, reduce=node)
                            except ValueError as e:
                                pass
            # permutation settings
            fusion_condition.expansion_split = not permutations_only

        condition_function = lambda sdfg, subgraph: fusion_condition.can_be_applied(sdfg, subgraph)
        enumerator = GreedyEnumerator(sdfg, graph, subgraph, condition_function=condition_function)
        for map_entries in enumerator:
            if len(map_entries) > 1:
                current_subgraph = xfsh.subgraph_from_maps(sdfg, graph, map_entries)
                cf = CompositeFusion()
                cf.setup_match(current_subgraph)
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


def _map_touches_gpu_global(state, mapentry: nodes.MapEntry, sdfg: SDFG) -> bool:
    """True iff the scope rooted at ``mapentry`` reads or writes a
    ``GPU_Global`` array through any of its boundary memlet paths.
    Used by ``tile_wcrs`` to decide whether a small map is safe to
    demote to ``Sequential`` (host) scheduling."""
    mapexit = state.exit_node(mapentry)
    for boundary_edge in list(state.in_edges(mapentry)) + list(state.out_edges(mapexit)):
        for path_edge in state.memlet_path(boundary_edge):
            for endpoint in (path_edge.src, path_edge.dst):
                if isinstance(endpoint, nodes.AccessNode):
                    if sdfg.arrays[endpoint.data].storage == dtypes.StorageType.GPU_Global:
                        return True
    return False


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
    if isinstance(graph, ControlFlowRegion):
        for block in graph_or_subgraph.nodes():
            if isinstance(block, SDFGState):
                tile_wcrs(block, validate_all)
        return

    if not isinstance(graph, SDFGState):
        raise TypeError('Graph must be a state, an SDFG, a control flow region, or a subgraph of either')
    sdfg = graph.parent

    # Ordered, not plain sets: every one of these is ITERATED to decide which map is tiled and in
    # what order, and a plain set of unhashable-by-value graph objects iterates by id(), which
    # tracks allocation history. Two runs of the same program then transform in different orders --
    # and for a partially-conflicted 2-D reduction that is the difference between a right and a
    # wrong answer, not just different code.
    edges_to_consider: OrderedSet[Tuple[gr.MultiConnectorEdge[Memlet], nodes.MapEntry]] = OrderedSet()
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

    maps_to_consider: OrderedSet[nodes.MapEntry] = OrderedSet(me for _, me in edges_to_consider)

    transformed: OrderedSet[nodes.MapEntry] = OrderedSet()

    # Heuristic: If the map is only partially conflicted, extract
    # parallel dimensions instead of tiling
    if prefer_partial_parallelism:
        for mapentry in maps_to_consider:
            # Check the write-conflicts of all WCR edges in map
            conflicts: OrderedSet[str] = OrderedSet()
            for edge, me in edges_to_consider:
                if me is not mapentry:
                    continue
                conflicts |= OrderedSet(cpp.write_conflicted_map_params(mapentry, edge))

            nonconflicted_dims = OrderedSet(mapentry.params) - conflicts
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
            # make map sequential -- but only when the data the map
            # touches is host-accessible. A Sequential schedule emits a
            # host loop; if any neighbouring AccessNode is GPU_Global
            # the loop would read/write device memory, which the
            # validator rightly rejects.
            if _map_touches_gpu_global(graph, mapentry, sdfg):
                if debugprint:
                    print(f'Keeping map "{mapentry}" device-scheduled '
                          f'(smaller than tile size but touches GPU_Global data)')
                continue
            if debugprint:
                print(f'Making map "{mapentry}" sequential due to being smaller than tile size')
            mapentry.map.schedule = dtypes.ScheduleType.Sequential
            continue

        # MapTiling -> AccumulateTransient / AccumulateStream
        outer_mapentry = dataflow.MapTiling.apply_to(sdfg, dict(tile_sizes=(tile_size, )), map_entry=mapentry)

        # The tile body accumulates into the per-tile transient one element after another -- that
        # accumulation is what removes the atomics, and it is sequential by construction. MapTiling
        # copies the original schedule onto both halves, so on GPU the body stayed GPU_Device: a
        # kernel inside a kernel, which the codegen refuses and NestedGPUDeviceMapLowering then has
        # to flatten by hoisting a range naming the outer map's own parameter. Sequential here is a
        # device-side loop, not the host loop the small-map branch above has to guard against,
        # because this map now sits inside the outer kernel's scope.
        mapentry.map.schedule = dtypes.ScheduleType.Sequential

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
    from dace.codegen.common import get_gpu_backend

    # Returns the optimized library node implementations for the given target
    # device
    if device == dtypes.DeviceType.GPU:
        try:
            backend = get_gpu_backend()
        except RuntimeError:
            backend = 'none'

        if backend == 'cuda':
            # ``CUDA`` for the same reason the CPU branch below carries everything past the vendor
            # BLAS: it is the key the CUB-backed nodes register under (``Scan``'s ``gpucub::DeviceScan``,
            # ``IntegerSort``'s ``DeviceRadixSort``, ``ArgReduce``'s ``DeviceReduce::ArgMax``,
            # ``FindFirst``, ``ScatterConflictCheck``, ``Symmetrize``'s parallel bounding box). Without
            # it every one of them fell through to the serial ``pure`` loop here while canonicalize
            # took the device form, so the GPU column compared library selection, not pipelines.
            return ['cuBLAS', 'cuSolverDn', 'GPUAuto', 'cuTENSOR', 'CUB', 'CUDA', 'pure']
        elif backend == 'hip':
            # Mirrors the CUDA row entry for entry, and must keep doing so. The two backends are
            # compared column against column, so a node that takes a tuned expansion under one and
            # the serial ``pure`` loop under the other measures the priority LIST rather than the
            # hardware. ``CUB`` and ``CUDA`` earn their place here for the same reason they do
            # above: they are the keys the device-primitive nodes register under (``Scan``,
            # ``IntegerSort``, ``ArgReduce``, ``FindFirst``, ``ScatterConflictCheck``,
            # ``Symmetrize``), and their emitted code names the backend-neutral ``gpucub`` /
            # ``gpu*`` aliases, so one expansion serves both. Each node's own environment still
            # gates whether the library is actually present.
            return ['rocBLAS', 'rocSOLVER', 'GPUAuto', 'hipTENSOR', 'CUB', 'CUDA', 'pure']
        else:
            return ['GPUAuto', 'pure']
    elif device == dtypes.DeviceType.CPU:
        result = []

        # BLAS calls
        if mkl.IntelMKL.is_installed():
            result.append('MKL')
        if openblas.OpenBLAS.is_installed():
            result.append('OpenBLAS')

        # Same order as canonicalize's ``canonicalize_fast_library_priority``, deliberately: the two
        # pipelines are compared column against column, so a node that lowers to a tuned expansion
        # under one and to the serial ``pure`` loop under the other measures the priority list rather
        # than the pipeline. Everything past the vendor BLAS was previously missing here, which left
        # a tensor transpose/contraction or a threaded reduction falling through to ``pure`` under
        # auto_optimize while canonicalize took the fast form.
        #
        # HPTT needs its own install (gated on HPTT_ROOT); TTGT is transpose+GEMM with no external
        # dependency; ``OpenMP`` covers Reduce/ArgReduce and ``CPU`` the OpenMP-5 Scan / radix sort /
        # ScatterConflictCheck. ``apply_cpu_library_parallelism`` below still has the last word on the
        # scope-dependent types, so a node nested in a parallel map keeps its sequential expansion.
        if 'HPTT_ROOT' in os.environ:
            result.append('HPTT')
        result.append('TTGT')

        return result + ['OpenMP', 'CPU', 'pure']

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


def libnode_work_is_below_break_even(node: nodes.LibraryNode, state: SDFGState) -> bool:
    """Whether ``node`` moves PROVABLY too few elements to pay for its own OpenMP region.

    The compile-time home of a decision the runtime used to re-take for itself: ``dace/scan.hpp``
    carried a ``PARALLEL_MIN_ELEMENTS_CONTIGUOUS`` gate that re-tested the element count on every
    call. Canonical form is parallel and the specialization band decides what goes back to
    sequential, so the gate belonged here, once, and not in the emitted kernel forever.

    Only a PROVABLY small count is sequential. A symbolic one is assumed big and stays parallel --
    reading "unknown" as "small" would single-thread every dynamically sized reduction and scan in
    the program, which is the opposite of the canonical form. The threshold is
    ``compiler.cpu.parallel_min_work_per_region``, calibrated to the host by
    :class:`~dace.transformation.passes.cpu_specialization.calibrate_thresholds.CalibrateCpuThresholds`
    before this runs.

    :param node: the library node to classify.
    :param state: the state containing it.
    :returns: ``True`` only when the element count is provably below the break-even.
    """
    threshold = int(config.Config.get('compiler', 'cpu', 'parallel_min_work_per_region'))
    if threshold <= 0:  # the size rule is disabled
        return False
    counts = [e.data.subset.num_elements() for e in state.in_edges(node) if e.data.subset is not None]
    if not counts:
        return False
    biggest = counts[0]
    for count in counts[1:]:
        biggest = sympy.Max(biggest, count)
    return symbolic.ask('negative', symbolic.simplify(biggest - threshold)) is True


def libnode_runs_multicore(node: nodes.LibraryNode) -> bool:
    """Whether ``node``'s OWN schedule is one that runs as an OpenMP team on the CPU.

    The other half of the top-level rule. Being top-level only says nobody re-enters the node; it
    does not say the node would run multicore. Opening a parallel region is right only when both
    hold, so this answers the second question and :func:`libnode_is_sequential` the first.

    ``Default`` counts as multicore: the enum documents it as the scope-default PARALLEL schedule and
    ``dtypes.SCOPEDEFAULT_SCHEDULE[None]`` resolves a top-level scope to ``CPU_Multicore``, so a node
    a pass introduces before :func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types`
    has run is not misread as sequential and silently single-threaded. Every other schedule
    (``Sequential``, ``MPI``, ``SVE_Map``, the GPU and Snitch ones) names an execution context that
    is not an OpenMP team, and takes the single-core expansion.

    :param node: the library node to classify.
    :returns: True if the node's schedule would run as an OpenMP team.
    """
    return node.schedule in (dtypes.ScheduleType.Default, *dtypes.CPU_SCHEDULES)


def libnode_is_sequential(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """Whether ``node`` is re-entered inside an outer parallel/repeated scope and so must NOT open
    its own (nested) parallel region -- it lowers to its efficient single-core expansion instead.

    The storage-derived ``node.schedule`` is NOT a reliable signal here:
    :func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types` sets a library node's
    schedule from the *storage* of its neighbouring memlets (``CPU_Heap`` ->
    ``ScheduleType.CPU_Multicore``), NOT from the parallelism of the enclosing scope, so a ``Reduce``
    nested in a parallel map can carry ``CPU_Multicore`` and would then wrongly open a nested
    ``#pragma omp parallel`` per outer iteration -- the "constant parallel reductions" catastrophe.
    Determine sequentiality from SCOPE instead: a libnode is sequential if it has a parallel parent
    map or an enclosing loop (both re-enter it).

    A ``NestedSDFG`` node carries no ``schedule`` of its own, so "lives inside a sequential nested
    SDFG" is not a distinct case to probe for:
    :func:`~dace.transformation.helpers.get_parent_map_and_loop_scopes` walks OUT across every
    nested-SDFG boundary up to the root and yields every enclosing ``MapEntry`` / ``LoopRegion``
    regardless of how many nsdfg levels separate ``node`` from them. A genuinely top-level node
    returns ``False`` and is free to open its own OpenMP / device-parallel region.

    :param node: The library node to classify.
    :param state: The state containing ``node``.
    :param sdfg: The root SDFG.
    :returns: True if ``node`` is re-entered by an enclosing parallel map or loop.
    """
    from dace.sdfg.state import LoopRegion  # Avoid an import cycle at module load.
    if node.schedule == dtypes.ScheduleType.Sequential:
        return True
    for scope in xfh.get_parent_map_and_loop_scopes(sdfg, node, state):
        if isinstance(scope, nodes.MapEntry):
            if scope.map.schedule != dtypes.ScheduleType.Sequential:
                return True
        elif isinstance(scope, LoopRegion):
            return True
    return False


def apply_cpu_library_parallelism(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """Pick the CPU lowering of the library nodes whose parallel form depends on their SCOPE.

    The single home for the CPU parallel-lowering rule, shared by :func:`set_fast_implementations`
    and canonicalize's
    :func:`~dace.transformation.passes.canonicalize.finalize.canonicalize_set_fast_implementations`,
    so the ``auto_optimize`` and ``canonicalize`` paths cannot drift onto different implementations
    of the same node. :func:`find_fast_library` lists only the vendor BLAS names, so without this the
    nodes below all fell through to the terminal ``pure`` fallback and a top-level reduction/scan lost
    its parallelism.

    A node opens its own parallel region only when it is top-level (nothing re-enters it --
    :func:`libnode_is_sequential`) AND its own schedule would run as an OpenMP team
    (:func:`libnode_runs_multicore`). Fail either half -- re-entered inside a parallel map or a loop,
    or scheduled onto something that is not an OpenMP team -- and it takes its efficient single-core
    expansion instead.

    * ``Reduce`` / ``ArgReduce``: ``OpenMP`` (privatized ``reduction(op:var)``; for ArgReduce a
      ``declare reduction`` over the (value, index) pair) vs the plain ``pure`` accumulate loop --
      never a contended ``omp atomic`` or a re-forked team per outer iteration.
    * ``Scan`` / ``ScatterConflictCheck``: ``CPU`` (OpenMP 5.0 ``reduction(inscan,..)`` +
      ``#pragma omp scan`` for the scan; an OpenMP tagged-write + verify for the conflict check) vs
      the serial ``pure`` form.
    * ``Copy`` / ``Memset``: ``Auto`` at top level, whose own size gate routes a large contiguous
      transfer to the element map that DaCe parallelizes across threads. A nested node asks its OWN
      selector for a concrete expansion, because the contiguous-only single-call forms would raise.

    :param node: The library node to select an implementation for.
    :param state: The state containing ``node``.
    :param sdfg: The root SDFG.
    :returns: True if ``node`` is one of the governed types and its implementation was set.
    """
    from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce
    from dace.libraries.standard.nodes.copy import CopyLibraryNode, select_copy_implementation
    from dace.libraries.standard.nodes.fill import FillLibraryNode, select_fill_implementation
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.libraries.standard.nodes.scan import Scan

    if not isinstance(node, (Reduce, ArgReduce, Scan, ScatterConflictCheck, CopyLibraryNode, FillLibraryNode)):
        return False
    impls = type(node).implementations
    # The rule, both halves. A node opens its own parallel region only when nothing re-enters it
    # (:func:`libnode_is_sequential`) AND its own schedule would run as an OpenMP team
    # (:func:`libnode_runs_multicore`); a node re-entered by an outer parallel map or loop, or one
    # scheduled onto anything that is not an OpenMP team, takes the single-core expansion. Then the
    # size rule: provably too little work to pay for a region of its own. That one covers the
    # copy/fill nodes through their own selectors, so it is applied to the rest.
    sequential = libnode_is_sequential(node, state, sdfg) or not libnode_runs_multicore(node)
    if not sequential and not isinstance(node, (CopyLibraryNode, FillLibraryNode)):
        sequential = libnode_work_is_below_break_even(node, state)
    if isinstance(node, (Reduce, ArgReduce)):
        # ``pure-seq`` needs an ``identity`` a lifted node may not carry, so ``pure`` is the robust
        # single-core choice (it lowers to a plain accumulate loop when Sequential).
        node.implementation = 'pure' if sequential else ('OpenMP' if 'OpenMP' in impls else node.implementation)
    elif isinstance(node, (Scan, ScatterConflictCheck)):
        node.implementation = 'pure' if sequential else ('CPU' if 'CPU' in impls else node.implementation)
    elif isinstance(node, CopyLibraryNode):
        node.implementation = select_copy_implementation(node, state) if sequential else 'Auto'
    else:
        node.implementation = select_fill_implementation(node, state) if sequential else 'Auto'
    return True


def set_fast_implementations(sdfg: SDFG,
                             device: dtypes.DeviceType,
                             blocklist: List[str] = None,
                             find_fast_library_fn: Callable[[dtypes.DeviceType], List[str]] = None) -> None:
    """
    Set fast library node implementations for the given device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param blocklist: list of disallowed implementations.
    :param find_fast_library_fn: function that returns the prioritized list of
                                  implementations for the given device, which will take priority over
                                  the built-in ``find_fast_library`` function.
    :note: Operates in-place on the given SDFG.
    """
    implementation_prio = []
    if find_fast_library_fn is not None:
        implementation_prio.extend(find_fast_library_fn(device))
    implementation_prio.extend(find_fast_library(device))

    if blocklist is not None:
        implementation_prio = [i for i in implementation_prio if i not in blocklist]

    # specialized nodes: pre-expand
    for current_sdfg in sdfg.all_sdfgs_recursive():
        for state in current_sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.LibraryNode):
                    if (node.default_implementation == 'specialize'
                            and (len(set(node.implementations)
                                     & set(implementation_prio))) == 0):
                        node.expand(state)

    # general nodes
    for node, _ in sdfg.all_nodes_recursive():
        # ``auto_select_implementation`` opts a node out entirely: its lowering was chosen by a
        # transformation (the tile ops, from the vectorizer's ``target_isa``), and every branch below
        # would silently reset it to the generic fallback -- no error, just the ISA path gone.
        if isinstance(node, nodes.LibraryNode) and node.auto_select_implementation:
            # NOTE: LibraryNodes with sequential schedule on GPU must be expanded to CUDA kernel-compatible code.
            # NOTE: Pure implementations are a safe choice for now but this should be revisited in the future.
            if device == dtypes.DeviceType.GPU and node.schedule == dtypes.ScheduleType.Sequential:
                node.implementation = "pure"
                continue
            for impl in implementation_prio:
                if impl in node.implementations:
                    if isinstance(
                            node,
                            dace.libraries.standard.nodes.reduce.Reduce) and node.implementation == 'CUDA (block)':
                        continue
                    node.implementation = impl
                    break

    # CPU: the nodes whose parallel lowering depends on scope. ``implementation_prio`` names only the
    # vendor BLAS libraries, so a Reduce / ArgReduce / Scan / Copy / Fill fell through to the
    # terminal ``pure`` fallback above and a TOP-LEVEL one silently lost its parallelism. Runs after
    # the priority loop so it has the last word on exactly those types.
    if device == dtypes.DeviceType.CPU:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.LibraryNode) and node.auto_select_implementation:
                apply_cpu_library_parallelism(node, state, sdfg)

    # reduce nodes
    if device == dtypes.DeviceType.GPU:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.LibraryNode) and node.auto_select_implementation:
                if device == dtypes.DeviceType.GPU and node.schedule == dtypes.ScheduleType.Sequential:
                    node.implementation = "pure"
                    continue
                # use GPUAuto expansion if applicable
                if ('GPUAuto' in node.implementations and not is_devicelevel_gpu_kernel(state.parent, state, node)
                        and state.scope_dict()[node] is None):
                    node.implementation = 'GPUAuto'
                    continue
                # Use CUB for device-level reductions
                if ('CUDA (device)' in node.implementations
                        and not is_devicelevel_gpu_kernel(state.parent, state, node)
                        and state.scope_dict()[node] is None):
                    node.implementation = 'CUDA (device)'
                    continue
                # The whole-array algorithms -- Scan, FindFirst, ScatterConflictCheck -- name their
                # device lowering ``CUDA``, which is in none of the ``find_fast_library`` priority
                # lists (those name vendor BLAS). Without this they fall through to ``pure`` on the
                # GPU: a scan loses its Blelloch/CUB sweep, a search and a conflict check lose theirs,
                # and each becomes a serial walk inside a kernel launch. Host-side only -- a
                # device-level instance keeps the pure expansion, which is what runs in-kernel.
                if ('CUDA' in node.implementations and not is_devicelevel_gpu_kernel(state.parent, state, node)
                        and state.scope_dict()[node] is None):
                    node.implementation = 'CUDA'


def make_transients_persistent(sdfg: SDFG,
                               device: dtypes.DeviceType,
                               toplevel_only: bool = True) -> Dict[int, Set[str]]:
    """
    Helper function to change several storage and scheduling properties

        * Makes non-view array lifetimes persistent, with some restrictions depending on the device
        * Reset nonatomic WCR edges on GPU

    The only arrays that are made persistent by default are ones that do not exist inside a scope (and thus may be
    allocated multiple times), and whose symbols are always given as parameters to the SDFG (so that they can be
    allocated in a persistent manner).

    :param sdfg: SDFG
    :param device: Device type
    :param toplevel_only: If True, only converts access nodes that do not appear in any scope.
    :return: A dictionary mapping SDFG IDs to a set of transient arrays that were made persistent.
    """
    result: Dict[int, Set[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        fsyms: Set[str] = nsdfg.free_symbols
        persistent: Set[str] = set()
        not_persistent: Set[str] = set()

        for state in nsdfg.states():
            for dnode in state.data_nodes():
                if dnode.data in not_persistent:
                    continue
                # Only convert arrays and scalars that are not compile-time constants
                if dnode.data in nsdfg.constants_prop:
                    not_persistent.add(dnode.data)
                    continue
                desc = dnode.desc(nsdfg)
                # Only convert what is not a member of a non-persistent struct.
                if (dnode.root_data != dnode.data
                        and nsdfg.arrays[dnode.root_data].lifetime != dtypes.AllocationLifetime.Persistent):
                    continue
                # Only convert transient arrays -- never a scalar or a provably single-element array.
                # Persistent (state-struct) allocation buys nothing for a single element, and a
                # persistent scalar has no state-struct-value form in the readable code generator; a
                # symbolically-sized array (possibly >1) is still eligible.
                if not desc.transient or type(desc) is not dt.Array:
                    not_persistent.add(dnode.data)
                    continue
                if all(symbolic.equal(s, 1) is True for s in desc.shape):
                    not_persistent.add(dnode.data)
                    continue
                if desc.storage == dtypes.StorageType.Register:
                    not_persistent.add(dnode.data)
                    continue
                # Only convert arrays where the size depends on SDFG parameters
                try:
                    if set(map(str, desc.total_size.free_symbols)) - fsyms:
                        not_persistent.add(dnode.data)
                        continue
                except AttributeError:  # total_size is an integer / has no free symbols
                    pass

                # Only convert arrays with top-level access nodes
                if xfh.get_parent_map(state, dnode) is not None:
                    if toplevel_only:
                        not_persistent.add(dnode.data)
                        continue
                    elif desc.lifetime == dtypes.AllocationLifetime.Scope:
                        not_persistent.add(dnode.data)
                        continue

                if desc.lifetime == dtypes.AllocationLifetime.External:
                    not_persistent.add(dnode.data)
                    continue

                persistent.add(dnode.data)

        for aname in (persistent - not_persistent):
            nsdfg.arrays[aname].lifetime = dtypes.AllocationLifetime.Persistent

        result[nsdfg.cfg_id] = (persistent - not_persistent)

    if device == dtypes.DeviceType.GPU:
        # Reset nonatomic WCR edges
        for state in sdfg.states():
            for edge in state.edges():
                edge.data.wcr_nonatomic = False

    return result


def interstate_read_names(sdfg: SDFG) -> OrderedSet:
    """The containers this SDFG's interstate edges read, which is host code reading them.

    An interstate edge's reads live in its condition and its assignments rather than on any
    AccessNode, so no dataflow analysis of the states reports them and a pass that walks nodes
    alone concludes the container is only ever touched by the maps it can see.
    """
    names: OrderedSet = OrderedSet()
    for edge in sdfg.all_interstate_edges():
        for memlet in edge.data.get_read_memlets(sdfg.arrays, include_scalars=True):
            names.add(memlet.data)
    return names


def apply_gpu_storage(sdfg: SDFG) -> None:
    """ Changes the storage of the SDFG's input and output arrays to GPU global memory.

    Scalars stay on the host: host code reads them (loop bounds, branch conditions, a tasklet
    outside any map), and a device-resident scalar makes every such read invalid. A device map
    that writes one gets a GPU transient and a copy back from the offload pass.

    An array an interstate edge indexes stays for the same reason, and it is the same host read --
    ``A[0] < N`` on a loop condition is not a scalar, so the check above does not cover it, and the
    graph it produces is refused only later, by validation, as a host read of device memory.
    """
    host_read = interstate_read_names(sdfg)
    for name, desc in sdfg.arrays.items():
        if isinstance(desc, dt.Scalar) or name in host_read:
            continue
        if not desc.transient and desc.storage == dtypes.StorageType.Default:
            desc.storage = dtypes.StorageType.GPU_Global


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  validate: bool = True,
                  validate_all: bool = False,
                  symbols: Dict[str, int] = None,
                  use_gpu_storage: bool = False,
                  find_fast_library_fn: Callable[[dtypes.DeviceType], List[str]] = None,
                  expand: bool = True) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:

        * Simplify
        * Auto-parallelization (loop-to-map)
        * Greedy application of SubgraphFusion
        * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
        * Tiled stream accumulation (MapTiling -> AccumulateTransient)
        * Collapse all maps to parallelize across all dimensions
        * Set all library nodes to their ``fast`` implementation, which calls
          the fastest library on the target device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :param symbols: Optional dict that maps symbols (str/symbolic) to int/float
    :param use_gpu_storage: If True, changes the storage of non-transient data to GPU global memory.
    :param find_fast_library_fn: Optional function that returns the prioritized list of
                                 implementations for the given device, which will take priority over
                                 the existing set of fast libraries found using auto-optimize.
    :param expand: If True (default), select fast library implementations for the device. Library
                   nodes are left un-expanded either way -- codegen expands whatever remains -- so
                   this only decides whether ``implementation`` is set.
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """
    debugprint = config.Config.get_bool('debugprint')

    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)

    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)

    # Apply GPU transformations and set library node implementations

    if device == dtypes.DeviceType.GPU:
        if use_gpu_storage:
            apply_gpu_storage(sdfg)
        sdfg.apply_gpu_transformations()
        sdfg.simplify()

    # fuse subgraphs greedily
    sdfg.simplify()
    sdfg.reset_cfg_list()
    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)

    # Move Loops inside Maps when possible
    from dace.transformation.interstate import MoveLoopIntoMap
    sdfg.apply_transformations_repeated([MoveLoopIntoMap])

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

    # Pick each library node's fast implementation, but leave it UNEXPANDED. Expansion is codegen's
    # job (``generate_code`` expands what is left), and doing it here throws away the one node the
    # later passes can still reason about -- a Gemm is a Gemm until it becomes three nested maps.
    # ``infer_types`` still runs, because choosing an implementation can change connector types.
    if expand:
        set_fast_implementations(sdfg, device, find_fast_library_fn=find_fast_library_fn)
        infer_types.infer_connector_types(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)

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
        known_symbols = {}
        for (s, v) in symbols.items():
            if s in sdfg.free_symbols:
                if isinstance(v, (int, float)):
                    known_symbols[s] = v
                if isinstance(v, sympy.Integer):
                    try:
                        known_symbols[s] = int(v)
                    except TypeError:
                        pass

        if debugprint and len(known_symbols) > 0:
            print("Specializing the SDFG for symbols", known_symbols)
        sdfg.specialize(known_symbols)

    sdfg.reset_cfg_list()

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
