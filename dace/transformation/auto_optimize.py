# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

from dace import dtypes
from dace.sdfg import SDFG, nodes

# Transformations
from dace.transformation.dataflow import MapCollapse


def greedy_fuse(sdfg: SDFG, validate_all: bool) -> None:
    # If two maps share connected nodes (horizontal/vertical), fuse
    pass


def tile_wcrs(sdfg: SDFG, validate_all: bool) -> None:
    pass


def auto_optimize(sdfg: SDFG,
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

    # TODO: LoopToMap

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
    for node, _ in sdfg.all_nodes_recursive():
        if (isinstance(node, nodes.LibraryNode)
                and 'fast' in node.implementations):
            node.implementation = 'fast'

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
