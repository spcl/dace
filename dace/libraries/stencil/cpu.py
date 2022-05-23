# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
import numpy as np

import dace

from ._common import *


@dace.library.expansion
class ExpandStencilCPU(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        (inputs, outputs, shape, field_to_data, field_to_desc, _,
         vector_lengths) = parse_connectors(node, parent_state, parent_sdfg)

        #######################################################################
        # Tasklet code generation
        #######################################################################

        code = node.code.as_string

        # Replace relative indices with memlet names
        code, field_accesses = parse_accesses(code, outputs)
        iterator_mapping = make_iterator_mapping(node, field_accesses, shape)
        validate_vector_lengths(vector_lengths, iterator_mapping)

        #######################################################################
        # Boundary condition generation
        #######################################################################

        boundary_code, oob_cond = generate_boundary_conditions(node, shape, field_accesses, field_to_desc,
                                                               iterator_mapping)

        #######################################################################
        # Write all output memlets
        #######################################################################

        write_code = ""
        if len(oob_cond) > 1:
            write_code += "if not (" + " or ".join(sorted(oob_cond)) + "):\n"
        write_code += "\n".join("{}_{} = {}".format("\t" if len(oob_cond) > 0 else "", field_accesses[output][tuple(
            0 for _ in range(len(shape)))], field_accesses[output][tuple(0 for _ in range(len(shape)))], output)
                                for output in outputs)

        code = boundary_code + "\n" + code + "\n" + write_code

        input_connectors = sum(
            [
                [f"_{c}" for c in field_accesses[k].values()] for k in inputs
                # Don't include scalar variables
                if sum(iterator_mapping[k], 0) > 0
            ],
            [])
        output_connectors = sum([[f"_{c}" for c in field_accesses[k].values()] for k in outputs], [])

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

        parameters = [f"_i{i}" for i in range(len(shape))]

        entry, exit = state.add_map(
            node.name + "_map", collections.OrderedDict(
                (parameters[i], "0:" + str(shape[i])) for i in range(len(shape))))

        for field in inputs:

            dtype = field_to_desc[field].dtype

            read_node = state.add_read(field)
            input_dims = iterator_mapping[field]
            input_shape = tuple(s for s, v in zip(shape, input_dims) if v)
            data = sdfg.add_array(field, input_shape, dtype)
            field_parameters = tuple(p for p, v in zip(parameters, input_dims) if v)
            for indices, connector in field_accesses[field].items():
                access_str = ", ".join(f"{p} + ({i})" for p, i in zip(field_parameters, indices))
                memlet = dace.Memlet(f"{field}[{access_str}]", dynamic=True)
                memlet.allow_oob = True
                state.add_memlet_path(read_node, entry, tasklet, dst_conn=f"_{connector}", memlet=memlet)

        index_tuple = ", ".join(parameters)
        for field in outputs:

            dtype = field_to_desc[field].dtype

            data = sdfg.add_array(field, shape, dtype)
            write_node = state.add_write(field)
            for indices, connector in field_accesses[field].items():
                state.add_memlet_path(tasklet,
                                      exit,
                                      write_node,
                                      src_conn=f"_{connector}",
                                      memlet=dace.Memlet(f"{field}[{index_tuple}]", dynamic=len(oob_cond) > 0))

        # Add scalars as symbols
        for field_name, mapping in iterator_mapping.items():
            if not any(mapping):
                sdfg.add_symbol(field_name, parent_sdfg.symbols[field_name])

        #######################################################################

        return sdfg
