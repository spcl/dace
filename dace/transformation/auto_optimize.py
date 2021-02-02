# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

from dace.sdfg.state import SDFGState
from dace import config, dtypes
from dace.sdfg import SDFG, nodes, graph as gr
from typing import Union
import warnings

# Transformations
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import LoopToMap

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def greedy_fuse(graph_or_subgraph: GraphViewType, validate_all: bool) -> None:
    # TODO: If two maps share connected nodes (horizontal/vertical), fuse
    # TODO: run multiexpansion first
    pass


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
