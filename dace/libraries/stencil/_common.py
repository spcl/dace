# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
import functools
import numpy as np
import operator
from typing import Dict, List, Tuple

import dace
from dace.frontend.python import astutils
from dace.codegen.targets.cpp import sym2cpp
from .subscript_converter import SubscriptConverter


def dim_to_abs_val(input, _dimensions, sdfg):
    """Compute scalar number from independent dimension unit."""
    input = [dace.symbolic.resolve_symbol_to_constant(x, sdfg) for x in input]
    dimensions = [dace.symbolic.resolve_symbol_to_constant(x, sdfg) for x in _dimensions]
    for i, dim in enumerate(dimensions[1:]):
        if dim is None:
            raise ValueError(f"Shape size \"{_dimensions[i + 1]}\" must " f"evaluate to a constant.")
    vec = [functools.reduce(operator.mul, dimensions[i + 1:], 1) for i in range(len(dimensions))]
    return functools.reduce(operator.add, map(operator.mul, input, vec), 0)


def make_iterators(dimensions, halo_sizes=None, parameters=None, vector_length=1):
    def add_halo(i):
        if i == len(dimensions) - 1 and halo_sizes is not None:
            return " + " + str(-halo_sizes[0] + halo_sizes[1])
        else:
            return ""

    if parameters is None:
        iterators = collections.OrderedDict([("i" + str(i), "0:" + str(d) + add_halo(i))
                                             for i, d in enumerate(dimensions)])
    else:
        iterators = collections.OrderedDict([(parameters[i], "0:" + str(d) + add_halo(i))
                                             for i, d in enumerate(dimensions)])
    if vector_length > 1:
        iterators[parameters[-1]] += "/{}".format(vector_length)
    return iterators


def check_stencil_shape(shape: Tuple, other: Tuple):
    """
    Compares the existing shape with a proposed shape, setting it to the new
    shape if the new shape has higher dimensionality. If the dimensionality is
    the same, they must be identical.
    """
    if len(other) > len(shape):
        shape = copy.copy(other)
    elif len(other) == len(shape):
        if shape != other:
            raise ValueError(f"Inconsistent input sizes: {shape} " f"vs. {other}")
    else:
        # Allow lower-dimensional accesses
        pass
    return shape


def parse_connectors(node, state, sdfg):
    """
    Collects the inputs and outputs, infers the shape of the iteration space,
    creates dictionaries mapping each connector to the data name in the
    parent SDFG and to the incoming/outgoing edge in the parent state, and
    collects vectorization widths.
    """
    inputs: List[str] = []
    outputs: List[str] = []
    shape: Tuple[int] = []
    field_to_data: Dict[str, str] = {}
    field_to_desc: Dict[str, dace.Data] = {}
    field_to_edge = {}
    memlets: Dict[str, dace.Memlet] = {}
    vector_lengths = {}
    for e in state.in_edges(node):
        field = e.dst_conn
        inputs.append(field)
        data_name = dace.sdfg.find_input_arraynode(state, e).data
        desc = sdfg.data(data_name)
        field_to_data[field] = data_name
        field_to_desc[field] = desc
        field_to_edge[field] = e
        vector_lengths[field] = desc.veclen
        if not isinstance(desc, dace.data.Scalar):
            shape = check_stencil_shape(shape, desc.shape)
    for e in state.out_edges(node):
        field = e.src_conn
        outputs.append(field)
        data_name = dace.sdfg.find_output_arraynode(state, e).data
        desc = sdfg.data(data_name)
        field_to_data[field] = data_name
        field_to_desc[field] = desc
        field_to_edge[field] = e
        vector_lengths[field] = desc.veclen
        shape = check_stencil_shape(shape, desc.shape)
    # Adjust shape for vector length
    vector_length = max(vector_lengths.values())
    shape = tuple(s * vector_length if i == len(shape) - 1 else s for i, s in enumerate(shape))
    return (inputs, outputs, shape, field_to_data, field_to_desc, field_to_edge, vector_lengths)


# def parse_accesses(code, outputs: List[str]):
def parse_accesses(sdfg, state, node, outputs: List[str]):
    """
    Runs the subscript converter to extract accesses of the format a[0, -1] into
    their accesses tuple (0, -1) and a generated memlet name a_0_m1.
    If an offset is found on the output, all indices are adjusted accordingly,
    such that the output is written at the central index.
    """

    # Run subscript converter
    converter = SubscriptConverter()
    code = node.code.as_string
    new_ast = converter.visit(ast.parse(code))
    field_accesses: Dict[str, List[Tuple[int]]] = converter.mapping

    # Check that there's only one write to the output
    offset = None
    for output in outputs:
        if len(field_accesses[output]) > 1:
            raise ValueError(f"Stencil {node.label} can only write {output} once.")
        _offset = next(iter(field_accesses[output].keys()))
        if offset is None:
            offset = _offset
        else:
            if _offset != offset:
                raise ValueError(f"Inconsistent output offset for " f"{node.label}: {offset} and {_offset}")

    # If the offset is non-zero, rerun the converter to adjust
    if offset is not None and any(o != 0 for o in offset):
        converter = SubscriptConverter(offset=tuple(-o for o in offset))
        new_ast = converter.visit(ast.parse(code))
        field_accesses = converter.mapping
    new_code = astutils.unparse(new_ast)

    # Add scalar data accesses (NOTE: supporting only scalar input)
    scalar_data = set()
    for c in node.in_connectors:
        if c not in field_accesses:
            e = list(state.in_edges_by_connector(node, c))[0]
            name = dace.sdfg.find_input_arraynode(state, e).data
            desc = sdfg.data(name)
            if isinstance(desc, dace.data.Scalar):
                field_accesses[c] = {tuple(): c}
                scalar_data.add(c)

    return new_code, field_accesses, scalar_data


def make_iterator_mapping(node, field_accesses, shape) -> Dict[str, Tuple[int]]:
    """
    Builds a complete iterator mapping dictionary for all data, including both
    explicitly specified mappings and implicit ones (that access all
    dimensions).
    """
    iterator_mapping: Dict[str, Tuple[int]] = {}
    for field_name, accesses in field_accesses.items():
        if field_name in node.iterator_mapping:
            iterators = node.iterator_mapping[field_name]
        else:
            # Scalar data access
            if len(accesses) == 1 and list(accesses.keys())[0] == tuple():
                iterators = tuple(False for _ in range(len(shape)))
            else:
                iterators = tuple(True for _ in range(len(shape)))
        iterator_mapping[field_name] = iterators
        num_dims = sum(iterators, 0)
        if num_dims == 0:
            continue  # Scalar input
        if len(iterators) != len(shape):
            raise ValueError(f"Invalid iterator mapping for {field_name}: {iterators}")
    return iterator_mapping


def generate_boundary_conditions(node, shape, field_accesses, field_to_desc, iterator_mapping):
    boundary_code = ""
    # Conditions where the output should not be written
    oob_cond = set()
    # Loop over each input
    for field_name in node.in_connectors:
        accesses = field_accesses[field_name]
        dtype = field_to_desc[field_name].dtype
        veclen = dtype.veclen
        iterators = iterator_mapping[field_name]
        num_dims = sum(iterators, 0)
        # Loop over each access to this data
        for indices, memlet_name in accesses.items():
            if len(indices) != num_dims:
                raise ValueError(f"Access {field_name}[{indices}] inconsistent " f"with iterator mapping {iterators}.")
            cond = set()
            cond_global = set()
            # Loop over each index of this access
            for i, offset in enumerate(indices):
                if i == len(indices) - 1 and dtype.veclen > 1:
                    unroll_boundary = f"*{dtype.veclen} + i_unroll"
                    unroll_write = f"*{dtype.veclen} - i_unroll"
                else:
                    unroll_boundary = ""
                    unroll_write = ""
                if offset < 0:
                    offset_str = sym2cpp(-offset)
                    term = f"_i{i}{unroll_boundary} < {offset_str}"
                    if i != len(indices) - 1:
                        offset_str = sym2cpp(-offset)
                        cond_global.add(f"_i{i} < {offset_str}")
                    elif offset <= -veclen:
                        offset_str = sym2cpp(-offset // veclen)
                        cond_global.add(f"_i{i} < {offset_str}")
                elif offset > 0:
                    offset_str = sym2cpp(shape[i] - offset)
                    term = f"_i{i}{unroll_boundary} >= {offset_str}"
                    if i != len(indices) - 1:
                        cond_global.add(f"_i{i} >= {offset_str}")
                    elif offset >= veclen:
                        offset_str = sym2cpp((shape[i] - offset) // veclen)
                        cond_global.add(f"_i{i} >= {offset_str}")
                else:
                    continue
                cond.add(term)
            if len(cond) == 0:
                boundary_code += "{} = _{}\n".format(memlet_name, memlet_name)
            else:
                if field_name in node.boundary_conditions:
                    bc = node.boundary_conditions[field_name]
                else:
                    bc = {"btype": "shrink"}
                btype = bc["btype"]
                if btype == "copy":
                    center_memlet = accesses[center]
                    boundary_val = "_{}".format(center_memlet)
                elif btype == "constant":
                    boundary_val = bc["value"]
                elif btype == "shrink":
                    # We don't need to do anything here, it's up to the
                    # user to not use the junk output
                    if np.issubdtype(dtype.type, np.floating):
                        boundary_val = np.finfo(dtype.type).min
                    else:
                        # If not a float, assume it's some kind of integer
                        boundary_val = np.iinfo(dtype.type).min
                    # Add this to the output condition
                    oob_cond |= cond_global
                else:
                    raise ValueError(f"Unsupported boundary condition type: {btype}")
                boundary_code += ("{} = {} if {} else _{}\n".format(memlet_name, boundary_val,
                                                                    " or ".join(sorted(cond)), memlet_name))
    return boundary_code, oob_cond


def validate_vector_lengths(vector_lengths, iterator_mapping):
    """
    Assert that vector lengths are valid and consistent.
    """
    # All fields must be vectorized if they access the innermost dimension,
    # and cannot be vectorized if they don't
    expected = max(vector_lengths.values())
    for field_name, vector_length in vector_lengths.items():
        dim_mask = iterator_mapping[field_name]
        if dim_mask[-1] == True:
            if vector_length != expected:
                raise ValueError(f"Field {field_name} has vectorization width "
                                 f"{vector_length}, expected {expected}.")
        else:
            if vector_length != 1:
                raise ValueError(f"Field {field_name} cannot be vectorized, "
                                 "because it doesn't read the innermost dimension.")
    return expected
