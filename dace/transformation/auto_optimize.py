# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

from dace.sdfg.state import SDFGState
from dace.sdfg.graph import SubgraphView
from dace import config, dtypes
from dace.sdfg import SDFG, nodes, graph as gr
from typing import Union
import warnings

# Transformations
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import LoopToMap
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

# Enumerator 
from dace.transformation.estimator.enumeration import GreedyEnumerator

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def greedy_fuse(graph_or_subgraph: GraphViewType, 
                validate_all: bool,
                apply_multi_expansion: bool = False,
                apply_stencil_tiling: bool = False,
                recursive: bool = False) -> None:

    if isinstance(graph_or_subgraph, SDFG):
        # If we have an SDFG, recurse into graphs 
        for graph in graph_or_subgraph.nodes():
            greedy_fuse(graph, validate_all)
    else:
        # we are in graph or subgraph
        sdfg, graph, subgraph = None, None, None 
        if isinstance(graph_or_subgraph, SDFGState):
            sdfg = graph_or_subgraph.parent
            graph = graph_or_subgraph
            subgraph = SubgraphView(graph, graph.nodes()) 
        else:
            sdfg = graph_or_subgraph.graph.parent
            graph = graph_or_subgraph.graph
            subgraph = graph_or_subgraph
        
        # greedily enumerate fusible components 
        # and apply transformation
        applied_transformations = 0
        enumerator = GreedyEnumerator(sdfg, graph, subgraph)
        for map_entries in enumerator:
            print(f"Processing map subgraph {map_entries}")
            if len(map_entries) > 1:
                current_subgraph = helpers.subgraph_from_maps(sdfg, graph, map_entries)
                cf = CompositeFusion(current_subgraph)
                cf.allow_expansion = apply_multi_expansion
                cf.allow_tiling = apply_stencil_tiling
                cf.apply(sdfg)
                applied_transformations += 1
                if recursive:
                    # advanced: for each scope subgraph, 
                    # see whether any parts inside could be fused together
                    global_entry = cf._global_map_entry
                    greedy_fuse(graph.scope_subgraph(global_entry))
        
        if applied_transformations > 0:
            print(f"Applied {applied_transformations} SubgraphFusion")
           

        # TODO [OK]: If two maps share connected nodes (horizontal/vertical), fuse -> fuse directly after enumerator pass
        # TODO [OK]: run multiexpansion first -> this is actually an option you can trigger 


        if validate_all:
            graph.validate()



def tile_wcrs(graph_or_subgraph: GraphViewType, validate_all: bool) -> None:
    # MapTiling (unless constant sized and less than tile size) -> AccumulateTransient and AccumulateStream
    # if smaller than tile size, don't transform and make sequential
    config.Config.get('optimizer', 'autotile_size')
    pass


def find_fast_library(device: dtypes.DeviceType) -> str:
    # Returns the optimized library node implementations for the given target
    # device
    if device is dtypes.DeviceType.GPU:
        return ['cuBLAS', 'CUB', 'pure']
    elif device is dtypes.DeviceType.CPU:
        # TODO: add "is_installed" checks to environments
        result = []

        # BLAS calls
        if mkl.IntelMKL.is_installed():
            result.append('MKL')
        elif openblas.OpenBLAS.is_installed():
            result.append('OpenBLAS')

        return result + ['pure']

    return ['pure']


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
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
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
    tile_wcrs(sdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.collapse = len(node.map.range)

    # Set all library nodes to expand to fast library calls
    implementation_prio = find_fast_library(device)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.LibraryNode):
            for impl in implementation_prio:
                if impl in node.implementations:
                    node.implementation = impl
                    break
            else:
                warnings.warn('No fast library implementation found for "%s", '
                              'falling back to default.' % node.name)

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections
    # TODO(later): Set on a per-SDFG basis
    config.Config.set('compiler', 'cpu', 'openmp_sections', value=False)

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
