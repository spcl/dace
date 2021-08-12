# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
import collections
import copy

import dace
import numpy as np
from .subscript_converter import SubscriptConverter

JUNK_VAL = -100000


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
        shape = []
        field_desc = {}
        outputs = []
        for e in parent_state.in_edges(node):
            field = e.dst_conn
            if field in node.accesses:
                desc = parent_sdfg.data(
                    dace.sdfg.find_input_arraynode(parent_state, e).data)
                field_desc[field] = desc
                shape = _check_stencil_shape(shape, desc.shape)
            else:
                raise ValueError(
                    f"Input {field} not found in list of stencil accesses.")
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
        code_memlet_names = converter.names

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        # Loop over each input
        for field_name, (iterators, accesses) in node.accesses.items():
            if sum(iterators, 0) == 0:
                continue  # Scalar input
            # Loop over each access to this data
            for indices in accesses:
                try:
                    memlet_name = code_memlet_names[field_name][indices]
                except KeyError:
                    import pdb
                    pdb.set_trace()
                    raise KeyError("Missing access in code: {}[{}]".format(
                        field_name, ", ".join(map(str, indices))))
                cond = []
                # Loop over each index of this access
                for i, offset in enumerate(indices):
                    if offset < 0:
                        cond.append(parameters[i] + " < " + str(-offset))
                    elif offset > 0:
                        cond.append(parameters[i] + " >= " +
                                    str(shape[i] - offset))
                ctype = field_desc[field_name].dtype
                if len(cond) == 0:
                    boundary_code += "{} = {}_in\n".format(
                        memlet_name, memlet_name)
                else:
                    bc = node.boundary_conditions[field_name]
                    btype = bc["btype"]
                    if btype == "copy":
                        center_memlet = code_memlet_names[field_name][center]
                        boundary_val = "_{}".format(center_memlet)
                    elif btype == "constant":
                        boundary_val = bc["value"]
                    elif btype == "shrink":
                        # We don't need to do anything here, it's up to the
                        # user to not use the junk output
                        boundary_val = JUNK_VAL
                        pass
                    else:
                        raise ValueError(
                            "Unsupported boundary condition type: {}".format(
                                node.boundary_conditions[field_name]["btype"]))
                    boundary_code += ("{} = {} if {} else {}_in\n".format(
                        memlet_name, boundary_val, " or ".join(cond),
                        memlet_name))

        #######################################################################
        # Write all output memlets
        #######################################################################

        write_code = "\n".join("{}_out = {}".format(
            code_memlet_names[output][tuple(
                0 for _ in range(len(shape)))], code_memlet_names[output][tuple(
                    0 for _ in range(len(shape)))], output)
                               for output in outputs)

        code = boundary_code + "\n" + code + "\n" + write_code

        input_memlets = sum(
            [
                ["{}_in".format(c) for c in v.values()]
                for k, v in code_memlet_names.items()
                # Don't include scalar variables
                if k in node.accesses and sum(node.accesses[k][0], 0) > 0
            ],
            [])
        output_memlets = sum(
            [["{}_out".format(c) for c in v.values()]
             for k, v in code_memlet_names.items() if k in outputs], [])

        #######################################################################
        # Create tasklet
        #######################################################################

        tasklet = state.add_tasklet(node.label + "_compute",
                                    input_memlets,
                                    output_memlets,
                                    code,
                                    language=dace.dtypes.Language.Python)

        #######################################################################
        # Build dataflow state
        #######################################################################

        entry, exit = state.add_map(
            node.name + "_map",
            collections.OrderedDict((parameters[i], "0:" + str(shape[i]))
                                    for i in range(len(shape))))

        for field in node.accesses:

            dtype = field_desc[field].dtype

            read_node = state.add_read(field)
            input_dims = node.accesses[field][0]
            input_shape = tuple(s for s, v in zip(shape, input_dims) if v)
            data = sdfg.add_array(field, input_shape, dtype)
            field_parameters = tuple(p for p, v in zip(parameters, input_dims)
                                     if v)
            for indices, connector in code_memlet_names[field].items():
                access_str = ", ".join(
                    "{} + ({})".format(p, i)
                    for p, i in zip(field_parameters, indices))
                memlet = dace.Memlet.simple(field, access_str, num_accesses=-1)
                memlet.allow_oob = True
                state.add_memlet_path(read_node,
                                      entry,
                                      tasklet,
                                      dst_conn=connector + "_in",
                                      memlet=memlet)

        for field in outputs:

            dtype = field_desc[field].dtype

            data = sdfg.add_array(field, shape, dtype)
            write_node = state.add_write(field)
            for indices, connector in code_memlet_names[field].items():
                state.add_memlet_path(tasklet,
                                      exit,
                                      write_node,
                                      src_conn=connector + "_out",
                                      memlet=dace.Memlet.simple(
                                          field, ", ".join(parameters)))

        # Add scalars as symbols
        for field_name, (indices, accesses) in node.accesses.items():
            if not any(indices):
                sdfg.add_symbol(field_name, parent_sdfg.symbols[field_name])

        #######################################################################

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg
