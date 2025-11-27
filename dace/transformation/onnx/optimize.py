# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import logging
from typing import Optional, Callable

import dace
from dace import nodes as nd
from dace.libraries import blas
from dace.sdfg.utils import expand_nodes
from dace.transformation import dataflow
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.dataflow import CopyToMap

log = logging.getLogger(__name__)


def expand_onnx_nodes(sdfg: dace.SDFG, predicate: Optional[Callable[[nd.Node], bool]] = None):
    """ Recursively expand all onnx library nodes in the SDFG, resulting in an SDFG that can be optimized by
        dace transformations. Will also specialize dace matmuls.

        :param sdfg: the sdfg to expand nodes on.
        :param predicate: a predicate that will be called to check if a node should be expanded.
    """

    try:
        from dace.libraries.onnx.nodes.onnx_op import ONNXOp  # avoid import loop
    except ImportError:
        raise ImportError("expand_onnx_nodes requires ONNX. Install with: pip install dace[ml]")

    if predicate is None:
        new_predicate = lambda n: isinstance(n, (ONNXOp, blas.MatMul))
    else:
        new_predicate = lambda n: predicate(n) and isinstance(n, (ONNXOp, blas.MatMul))

    expand_nodes(sdfg, new_predicate)


def auto_optimize_onnx(sdfg: dace.SDFG, cuda, simplify=False, fold_constants=True):
    """ Automatically optimize ``sdfg``.

        :param sdfg: the sdfg to optimize (inplace).
        :param cuda: whether to optimize for cuda.
        :param simplify: whether to apply simplification transformations to the sdfg after optimization.
        :param fold_constants: whether to apply constant folding.
    """

    try:
        from dace.transformation.onnx import ConstantFolding  # avoid import loop
    except ImportError:
        raise ImportError("auto_optimize_onnx requires ONNX. Install with: pip install dace[ml]")

    log.debug("Applying automatic optimizations")
    if fold_constants:
        log.debug("Applying constant folding")
        sdfg.apply_transformations_repeated([ConstantFolding, dataflow.RedundantSecondArray], validate_all=True)
    log.debug("Expanding ONNX nodes")
    expand_onnx_nodes(sdfg)
    log.debug("Setting fast implementations")
    set_fast_implementations(sdfg, dace.DeviceType.GPU if cuda else dace.DeviceType.CPU)
    if simplify:
        log.debug("Applying simplification transforms")
        sdfg.simplify()
        if cuda:
            sdfg.apply_transformations_once_everywhere(CopyToMap)
