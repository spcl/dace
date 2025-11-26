# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import functools
import logging
from typing import Optional, Callable

import dace
from dace import nodes as nd, data as dt
from dace.libraries import blas
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import dataflow
from dace import SDFG, SDFGState, dtypes
import dace.data as dt
from dace import dtypes
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.dataflow import CopyToMap

log = logging.getLogger(__name__)


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


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


def expand_nodes(sdfg: dace.SDFG, predicate: Callable[[nd.Node], bool]):
    """ Recursively expand library nodes in the SDFG using a given predicate.

        :param sdfg: the sdfg to expand nodes on.
        :param predicate: a predicate that will be called to check if a node should be expanded.
    """
    if sdfg is None:
        return
    states = list(sdfg.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):  # Make sure we have a copy
            if isinstance(node, nd.NestedSDFG):
                expand_nodes(node.sdfg, predicate=predicate)
            elif isinstance(node, nd.LibraryNode):
                if predicate(node):
                    impl_name = node.expand(sdfg, state)
                    if dace.Config.get_bool('debugprint'):
                        print("Automatically expanded library node \"{}\" with implementation \"{}\".".format(
                            str(node), impl_name))
                    # We made a copy of the original list of nodes, so we keep
                    # iterating even though this list has now changed
                    expanded_something = True

        if expanded_something:
            states.append(state)  # Nodes have changed. Check state again


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
        # there is a nondeterministic bug in redundant array that appears if
        sdfg.simplify()
        if cuda:
            sdfg.apply_transformations_once_everywhere(CopyToMap)


def iterables_equal(a, b) -> bool:
    """ Return whether the two iterables ``a`` and ``b`` are equal. """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))


def prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


def is_cuda(storage: dtypes.StorageType) -> bool:
    """ Check if a descriptor storage type is a GPU array """
    if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, storage):
        return False
    elif dtypes.can_access(dtypes.ScheduleType.FPGA_Device, storage):
        return False
    elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, storage):
        return True
    else:
        raise ValueError(f"Unsupported storage {storage}")


def platform_library_name(libname: str) -> str:
    """ Get the filename of a library.

        :param libname: the name of the library.
        :return: the filename of the library.
    """
    prefix = dace.Config.get('compiler', 'library_prefix')
    suffix = dace.Config.get('compiler', 'library_extension')
    return f"{prefix}{libname}.{suffix}"


def all_equal(a, b) -> bool:
    """
    Check whether two iterables are equal
    """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))
