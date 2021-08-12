# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
import collections
import copy
from typing import Dict, List, Tuple

import dace
import numpy as np
from .subscript_converter import SubscriptConverter


def _check_stencil_shape(shape, other):
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


@dace.library.expansion
class ExpandStencilCPU(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        # Find outer data descriptor
        field_desc = {}
        shape = []
        inputs = []
        outputs = []
        for e in parent_state.in_edges(node):
            field = e.dst_conn
            inputs.append(field)
            desc = parent_sdfg.data(
                dace.sdfg.find_input_arraynode(parent_state, e).data)
            field_desc[field] = desc
            shape = _check_stencil_shape(shape, desc.shape)
        for e in parent_state.out_edges(node):
            field = e.src_conn
            outputs.append(field)
            desc = parent_sdfg.data(
                dace.sdfg.find_output_arraynode(parent_state, e).data)
            field_desc[field] = desc
            shape = _check_stencil_shape(shape, desc.shape)

        parameters = [f"_i{i}" for i in range(len(shape))]

        #######################################################################
        # Tasklet code generation
        #######################################################################

        code = node.code.as_string

        # Replace relative indices with memlet names
        converter = SubscriptConverter()
        new_ast = converter.visit(ast.parse(code))
        code = astunparse.unparse(new_ast)
        field_accesses: Dict[str, List[Tuple[int]]] = converter.names

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        iterator_mapping: Dict[str, Tuple[int]] = {}
        oob_cond = set()
        # Loop over each input
        for field_name in inputs:
            accesses = field_accesses[field_name]
            if field_name in node.iterator_mapping:
                iterators = node.iterator_mapping[field_name]
            else:
                iterators = tuple(True for _ in range(len(shape)))
            iterator_mapping[field_name] = iterators
            num_dims = sum(iterators, 0)
            if num_dims == 0:
                continue  # Scalar input
            if len(iterators) != len(shape):
                raise ValueError(
                    f"Invalid iterator mapping for {field_name}: {iterators}")
            dtype = field_desc[field_name].dtype.type
            # Loop over each access to this data
            for indices, memlet_name in accesses.items():
                if len(indices) != num_dims:
                    raise ValueError(f"Access {indices} inconsistent with "
                                     f"iterator mapping {iterators}.")
                cond = set()
                # Loop over each index of this access
                for i, offset in enumerate(indices):
                    if offset < 0:
                        term = parameters[i] + " < " + str(-offset)
                    elif offset > 0:
                        term = parameters[i] + " >= " + str(shape[i] - offset)
                    cond.add(term)
                if len(cond) == 0:
                    boundary_code += "{} = {}_in\n".format(
                        memlet_name, memlet_name)
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
                        if np.issubdtype(dtype, np.floating):
                            boundary_val = np.nan
                        else:
                            # If not a float, assume it's some kind of integer
                            boundary_val = np.iinfo(dtype).min
                        # Add this to the output condition
                        oob_cond |= cond
                    else:
                        raise ValueError(
                            f"Unsupported boundary condition type: {btype}")
                    boundary_code += ("{} = {} if {} else {}_in\n".format(
                        memlet_name, boundary_val, " or ".join(sorted(cond)),
                        memlet_name))

        #######################################################################
        # Write all output memlets
        #######################################################################

        write_code = ""
        if len(oob_cond) > 1:
            write_code += "if not (" + " or ".join(sorted(oob_cond)) + "):\n"
        write_code += "\n".join("{}{}_out = {}".format(
            "\t" if len(oob_cond) > 0 else "", field_accesses[output][tuple(
                0 for _ in range(len(shape)))], field_accesses[output][tuple(
                    0 for _ in range(len(shape)))], output)
                                for output in outputs)

        code = boundary_code + "\n" + code + "\n" + write_code

        input_connectors = sum(
            [
                [f"{c}_in" for c in field_accesses[k].values()] for k in inputs
                # Don't include scalar variables
                if sum(iterator_mapping[k], 0) > 0
            ],
            [])
        output_connectors = sum([[f"{c}_out" for c in field_accesses[k].values()]
                              for k in outputs], [])

        #######################################################################
        # Create tasklet
        #######################################################################

        tasklet = state.add_tasklet(node.label + "_compute",
                                    input_connectors,
                                    output_connectors,
                                    code,
                                    language=dace.dtypes.Language.Python)

        #######################################################################
        # Build dataflow state
        #######################################################################

        entry, exit = state.add_map(
            node.name + "_map",
            collections.OrderedDict((parameters[i], "0:" + str(shape[i]))
                                    for i in range(len(shape))))

        for field in inputs:

            dtype = field_desc[field].dtype

            read_node = state.add_read(field)
            input_dims = iterator_mapping[field]
            input_shape = tuple(s for s, v in zip(shape, input_dims) if v)
            data = sdfg.add_array(field, input_shape, dtype)
            field_parameters = tuple(p for p, v in zip(parameters, input_dims)
                                     if v)
            for indices, connector in field_accesses[field].items():
                access_str = ", ".join(
                    f"{p} + ({i})" for p, i in zip(field_parameters, indices))
                memlet = dace.Memlet.simple(field, access_str, num_accesses=-1)
                memlet.allow_oob = True
                state.add_memlet_path(read_node,
                                      entry,
                                      tasklet,
                                      dst_conn=connector + "_in",
                                      memlet=memlet)

        index_tuple = ", ".join(parameters)
        for field in outputs:

            dtype = field_desc[field].dtype

            data = sdfg.add_array(field, shape, dtype)
            write_node = state.add_write(field)
            for indices, connector in field_accesses[field].items():
                state.add_memlet_path(tasklet,
                                      exit,
                                      write_node,
                                      src_conn=connector + "_out",
                                      memlet=dace.Memlet(
                                          f"{field}[{index_tuple}]",
                                          dynamic=len(oob_cond) > 0))

        # Add scalars as symbols
        for field_name, mapping in iterator_mapping.items():
            if not any(mapping):
                sdfg.add_symbol(field_name, parent_sdfg.symbols[field_name])

        #######################################################################

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg
