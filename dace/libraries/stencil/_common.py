import collections
import copy
import functools
import operator
from typing import Tuple

import dace


def dim_to_abs_val(input, dimensions):
    """Compute scalar number from independent dimension unit."""
    vec = [
        functools.reduce(operator.mul, dimensions[i + 1:], 1)
        for i in range(len(dimensions))
    ]
    return functools.reduce(operator.add, map(operator.mul, input, vec), 0)


def make_iterators(dimensions, halo_sizes=None, parameters=None):
    def add_halo(i):
        if i == len(dimensions) - 1 and halo_sizes is not None:
            return " + " + str(-halo_sizes[0] + halo_sizes[1])
        else:
            return ""

    if parameters is None:
        return collections.OrderedDict([("i" + str(i),
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])
    else:
        return collections.OrderedDict([(parameters[i],
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])


def _check_stencil_shape(shape: Tuple, other: Tuple):
    """
    Compares the existing shape with a proposed shape, setting it to the new
    shape if the new shape has higher dimensionality. If the dimensionality is
    the same, they must be identical.
    """
    if len(other) > len(shape):
        shape = copy.copy(other)
    elif len(other) == len(shape):
        if shape != other:
            raise ValueError(f"Inconsistent input sizes: {shape} "
                             f"vs. {other}")
    else:
        # Allow lower-dimensional accesses
        pass
    return shape


def _parse_connectors(node, state, sdfg):
    """
    Collects the inputs and outputs, infers the shape of the iteration space,
    and creates a dictionary mapping each data name to its descriptor in the
    parent SDFG.
    """
    inputs: List[str] = []
    outputs: List[str] = []
    shape: Tuple[int] = []
    field_to_desc: Dict[str, dace.Data] = {}
    for e in state.in_edges(node):
        field = e.dst_conn
        inputs.append(field)
        desc = sdfg.data(dace.sdfg.find_input_arraynode(state, e).data)
        field_to_desc[field] = desc
        shape = _check_stencil_shape(shape, desc.shape)
    for e in state.out_edges(node):
        field = e.src_conn
        outputs.append(field)
        desc = sdfg.data(dace.sdfg.find_output_arraynode(state, e).data)
        field_to_desc[field] = desc
        shape = _check_stencil_shape(shape, desc.shape)
    return inputs, outputs, shape, field_to_desc
