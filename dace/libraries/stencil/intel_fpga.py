# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
import collections
import functools
import itertools
import operator
import re

import dace
from dace import data as dt, subsets as sbs
import numpy as np
from .subscript_converter import SubscriptConverter
from ._common import *


@dace.library.expansion
class ExpandStencilIntelFPGA(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        (inputs, outputs, shape, field_to_data, field_to_desc, field_to_edge,
         vector_lengths) = parse_connectors(node, parent_state, parent_sdfg)

        #######################################################################
        # Parse the tasklet code
        #######################################################################

        # Replace relative indices with memlet names
        converter = SubscriptConverter()

        # Add copy boundary conditions
        for field in node.boundary_conditions:
            if node.boundary_conditions[field]["btype"] == "copy":
                center_index = tuple(0 for _ in range(
                    len(parent_sdfg.arrays[field_to_data[field]].shape)))
                # This will register the renaming
                converter.convert(field, center_index)

        # Replace accesses in the code
        code, field_accesses = parse_accesses(node.code.as_string, outputs)

        iterator_mapping = make_iterator_mapping(node, field_accesses, shape)
        vector_length = validate_vector_lengths(vector_lengths,
                                                iterator_mapping)

        # Extract which fields to read from streams and what to buffer
        buffer_sizes = collections.OrderedDict()
        buffer_accesses = collections.OrderedDict()
        scalars = {}  # {name: type}
        for field_name in inputs:
            relative = field_accesses[field_name]
            dim_mask = iterator_mapping[field_name]
            if not any(dim_mask):
                # This is a scalar, no buffer needed. Instead, the SDFG must
                # take this as a symbol
                scalars[field_name] = parent_sdfg.symbols[field_name]
                sdfg.add_symbol(field_name, parent_sdfg.symbols[field_name])
                continue
            abs_indices = ([
                dim_to_abs_val(i, tuple(s for s, m in zip(shape, dim_mask)
                                        if m), parent_sdfg) for i in relative
            ] + ([0] if field_name in node.boundary_conditions
                 and node.boundary_conditions[field_name]["btype"] == "copy"
                 else []))
            max_access = max(abs_indices)
            min_access = min(abs_indices)
            buffer_size = max_access - min_access + vector_lengths[field_name]
            buffer_sizes[field_name] = buffer_size
            # (indices relative to center, buffer indices, center index)
            buffer_accesses[field_name] = ([tuple(r) for r in relative], [
                i - min_access for i in abs_indices
            ], -min_access)

        # Create a initialization phase corresponding to the highest distance
        # to the center
        init_sizes = [
            (buffer_sizes[key] - vector_lengths[key] - val[2]) // vector_length
            for key, val in buffer_accesses.items()
        ]
        init_size_max = int(np.max(init_sizes))

        parameters = [f"_i{i}" for i in range(len(shape))]

        # Dimensions we need to iterate over
        iterator_mask = np.array([s != 0 and s != 1 for s in shape], dtype=bool)
        iterators = make_iterators(
            tuple(s for s, m in zip(shape, iterator_mask) if m),
            parameters=tuple(s for s, m in zip(parameters, iterator_mask) if m),
            vector_length=vector_length)

        # Manually add pipeline entry and exit nodes
        pipeline_range = dace.properties.SubsetProperty.from_string(', '.join(
            iterators.values()))
        pipeline = dace.sdfg.nodes.Pipeline(
            "compute_" + node.label,
            list(iterators.keys()),
            pipeline_range,
            dace.dtypes.ScheduleType.FPGA_Device,
            False,
            init_size=init_size_max,
            init_overlap=False,
            drain_size=init_size_max,
            drain_overlap=True)
        entry = dace.sdfg.nodes.PipelineEntry(pipeline)
        exit = dace.sdfg.nodes.PipelineExit(pipeline)
        state.add_nodes_from([entry, exit])

        # Add nested SDFG to do 1) shift buffers 2) read from input 3) compute
        nested_sdfg = dace.SDFG(node.label + "_inner", parent=state)
        nested_sdfg_tasklet = state.add_nested_sdfg(
            nested_sdfg,
            sdfg,
            # Input connectors
            [k + "_in" for k in inputs if any(iterator_mapping[k])] +
            [name + "_buffer_in" for name, _ in buffer_sizes.items()],
            # Output connectors
            [k + "_out" for k in outputs] +
            [name + "_buffer_out" for name, _ in buffer_sizes.items()],
            schedule=dace.ScheduleType.FPGA_Device)
        # Propagate symbols
        for sym_name, sym_type in parent_sdfg.symbols.items():
            nested_sdfg.add_symbol(sym_name, sym_type)
            nested_sdfg_tasklet.symbol_mapping[sym_name] = sym_name
        # Map iterators
        for p in parameters:
            nested_sdfg.add_symbol(p, dace.int64)
            nested_sdfg_tasklet.symbol_mapping[p] = p

        # Shift state, which shifts all buffers by one
        shift_state = nested_sdfg.add_state(node.label + "_shift")

        # Update state, which reads new values from memory
        update_state = nested_sdfg.add_state(node.label + "_update")

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code, oob_cond = generate_boundary_conditions(
            node, shape, field_accesses, field_to_desc, iterator_mapping)

        #######################################################################
        # Only write if we're in bounds
        #######################################################################

        write_code = ("\n".join([
            "{}_inner_out = {}\n".format(
                output,
                field_accesses[output][tuple(0 for _ in range(len(shape)))])
            for output in outputs
        ]))
        if init_size_max > 0 or len(oob_cond) > 0:
            write_cond = []
            if init_size_max > 0:
                init_cond = pipeline.init_condition()
                write_cond.append("not " + init_cond)
                nested_sdfg_tasklet.symbol_mapping[init_cond] = init_cond
                nested_sdfg.add_symbol(init_cond, dace.bool)
            if len(oob_cond) > 0:
                oob_cond = " or ".join(sorted(oob_cond))
                oob_cond = f"not ({oob_cond})"
                write_cond.append(oob_cond)
            write_cond = " and ".join(write_cond)
            write_cond = f"if {write_cond}:\n\t"
        else:
            write_cond = ""

        code = boundary_code + "\n" + code + "\n" + write_code

        #######################################################################
        # Create DaCe compute state
        #######################################################################

        # Compute state, which reads from input channels, performs the compute,
        # and writes to the output channel(s)
        compute_state = nested_sdfg.add_state(node.label + "_compute")
        compute_inputs = list(
            itertools.chain.from_iterable(
                [["_" + v for v in field_accesses[f].values()] for f in inputs
                 if any(iterator_mapping[f])]))
        compute_tasklet = compute_state.add_tasklet(
            node.label + "_compute",
            compute_inputs, {name + "_inner_out"
                             for name in outputs},
            code,
            language=dace.dtypes.Language.Python)
        if vector_length > 1:
            compute_unroll_entry, compute_unroll_exit = compute_state.add_map(
                compute_state.label + "_unroll",
                {"i_unroll": f"0:{vector_length}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

        # Connect the three nested states
        nested_sdfg.add_edge(shift_state, update_state,
                             dace.sdfg.InterstateEdge())
        nested_sdfg.add_edge(update_state, compute_state,
                             dace.sdfg.InterstateEdge())

        # First, grab scalar variables
        for scalar, scalar_type in scalars.items():
            nested_sdfg.add_symbol(scalar, scalar_type)

        # Code to increment custom iterators
        iterator_code = ""

        for (field_name, size), init_size in zip(buffer_sizes.items(),
                                                 init_sizes):

            data_name = field_to_data[field_name]
            connector = field_to_edge[field_name].dst_conn
            data_name_outer = connector
            data_name_inner = field_name + "_in"
            desc_outer = parent_sdfg.arrays[data_name].clone()
            desc_outer.transient = False
            sdfg.add_datadesc(data_name_outer, desc_outer)

            mapping = iterator_mapping[field_name]
            is_array = not isinstance(desc_outer, dt.Stream)

            # If this array is part of the initialization phase, it needs its
            # own iterator, which we need to instantiate and increment in the
            # outer SDFG
            if is_array:
                if init_size == 0:
                    field_index = [s for s, p in zip(parameters, mapping) if p]
                else:
                    # Create custom iterators for this array
                    num_dims = sum(mapping, 0)
                    field_iterators = [(f"_{field_name}_i{i}", shape[i])
                                       for i in range(num_dims) if mapping[i]]
                    start_index = init_size_max - init_size
                    tab = ""
                    if start_index > 0:
                        iterator_code += (
                            f"if {pipeline.iterator_str()} >= {start_index}:\n")
                        tab += "  "
                    for i, (it, s) in enumerate(reversed(field_iterators)):
                        iterator_code += f"""\
{tab}if {it} < {s} - 1:
{tab}  {it} = {it} + 1
{tab}else:
{tab}  {it} = 0\n"""
                        tab += "  "
                    field_index = [fi[0] for fi in field_iterators]
                    for fi in field_index:
                        pipeline.additional_iterators[fi] = "0"
                        nested_sdfg.add_symbol(fi, dace.int64)
                        nested_sdfg_tasklet.symbol_mapping[fi] = fi
                field_index = ", ".join(field_index)
            else:
                field_index = "0"

            # Outer memory read
            read_node_outer = state.add_read(data_name_outer)
            if isinstance(desc_outer, dt.Stream):
                subset = "0"
            else:
                subset = str(sbs.Range.from_array(desc_outer))
            state.add_memlet_path(
                read_node_outer,
                entry,
                nested_sdfg_tasklet,
                dst_conn=data_name_inner,
                memlet=dace.Memlet(f"{data_name_outer}[{subset}]"))

            # Create inner memory pipe
            desc_inner = desc_outer.clone()
            nested_sdfg.add_datadesc(data_name_inner, desc_inner)

            buffer_name_outer = f"{node.label}_{field_name}_buffer"
            buffer_name_inner_read = f"{field_name}_buffer_in"
            buffer_name_inner_write = f"{field_name}_buffer_out"

            # Create buffer transient in outer SDFG
            field_dtype = parent_sdfg.data(data_name).dtype
            _, desc_outer = sdfg.add_array(
                buffer_name_outer, (size, ),
                field_dtype.base_type,
                storage=dace.dtypes.StorageType.FPGA_Local,
                transient=True)

            # Create read and write nodes
            read_node_outer = state.add_read(buffer_name_outer)
            write_node_outer = state.add_write(buffer_name_outer)

            # Outer buffer read
            state.add_memlet_path(read_node_outer,
                                  entry,
                                  nested_sdfg_tasklet,
                                  dst_conn=buffer_name_inner_read,
                                  memlet=dace.Memlet(
                                      f"{buffer_name_outer}[0:{size}]",
                                      dynamic=True))

            # Outer buffer write
            state.add_memlet_path(nested_sdfg_tasklet,
                                  exit,
                                  write_node_outer,
                                  src_conn=buffer_name_inner_write,
                                  memlet=dace.Memlet(
                                      f"{write_node_outer.data}[0:{size}]",
                                      dynamic=True))

            # Inner copy
            desc_inner_read = desc_outer.clone()
            desc_inner_read.transient = False
            desc_inner_read.name = buffer_name_inner_read
            desc_inner_write = desc_inner_read.clone()
            desc_inner_write.name = buffer_name_inner_write
            nested_sdfg.add_datadesc(buffer_name_inner_read, desc_inner_read)
            nested_sdfg.add_datadesc(buffer_name_inner_write, desc_inner_write)

            # Make shift state if necessary
            if size > 1:
                shift_read = shift_state.add_read(buffer_name_inner_read)
                shift_write = shift_state.add_write(buffer_name_inner_write)
                shift_entry, shift_exit = shift_state.add_map(
                    f"shift_{field_name}",
                    {"i_shift": f"0:{size} - {vector_lengths[field_name]}"},
                    schedule=dace.dtypes.ScheduleType.FPGA_Device,
                    unroll=True)
                shift_tasklet = shift_state.add_tasklet(
                    f"shift_{field_name}", {f"{field_name}_shift_in"},
                    {f"{field_name}_shift_out"},
                    f"{field_name}_shift_out = {field_name}_shift_in")
                shift_state.add_memlet_path(
                    shift_read,
                    shift_entry,
                    shift_tasklet,
                    dst_conn=field_name + "_shift_in",
                    memlet=dace.Memlet(
                        f"{shift_read.data}"
                        f"[i_shift + {vector_lengths[field_name]}]"))
                shift_state.add_memlet_path(
                    shift_tasklet,
                    shift_exit,
                    shift_write,
                    src_conn=field_name + "_shift_out",
                    memlet=dace.Memlet(f"{shift_write.data}[i_shift]"))

            # Begin reading according to this field's own buffer size, which is
            # translated to an index by subtracting it from the maximum buffer
            # size
            begin_reading = (init_size_max - init_size)
            end_reading = (
                functools.reduce(operator.mul, shape, 1) / vector_length +
                init_size_max - init_size)

            update_read = update_state.add_read(data_name_inner)
            update_write = update_state.add_write(buffer_name_inner_write)
            update_tasklet = update_state.add_tasklet(
                "read_wavefront", {"wavefront_in"}, {"buffer_out"},
                "if {it} >= {begin} and {it} < {end}:\n"
                "\tbuffer_out = wavefront_in\n".format(
                    it=pipeline.iterator_str(),
                    begin=begin_reading,
                    end=end_reading),
                language=dace.dtypes.Language.Python)
            nested_sdfg_tasklet.symbol_mapping[pipeline.iterator_str()] = (
                pipeline.iterator_str())
            iterator_str = pipeline.iterator_str()
            if iterator_str not in nested_sdfg.symbols:
                nested_sdfg.add_symbol(iterator_str, dace.int64)
            update_state.add_memlet_path(
                update_read,
                update_tasklet,
                memlet=dace.Memlet(f"{update_read.data}[{field_index}]",
                                   dynamic=True),
                dst_conn="wavefront_in")
            subset = f"{size} - {vector_length}:{size}" if size > 1 else "0"
            update_state.add_memlet_path(update_tasklet,
                                         update_write,
                                         memlet=dace.Memlet(
                                             f"{update_write.data}[{subset}]",
                                             dynamic=True),
                                         src_conn="buffer_out")

            # Make compute state
            compute_read = compute_state.add_read(buffer_name_inner_read)
            for relative, offset in zip(buffer_accesses[field_name][0],
                                        buffer_accesses[field_name][1]):
                memlet_name = field_accesses[field_name][tuple(relative)]
                if vector_length > 1:
                    if vector_lengths[field_name] > 1:
                        offset = f"{offset} + i_unroll"
                    else:
                        offset = str(offset)
                    path = [compute_read, compute_unroll_entry, compute_tasklet]
                else:
                    offset = str(offset)
                    path = [compute_read, compute_tasklet]
                compute_state.add_memlet_path(
                    *path,
                    dst_conn="_" + memlet_name,
                    memlet=dace.Memlet(f"{compute_read.data}[{offset}]"))

        # Tasklet to update iterators
        update_iterator_tasklet = state.add_tasklet(
            f"{node.label}_update_iterators", {}, {}, iterator_code)
        state.add_memlet_path(nested_sdfg_tasklet,
                              update_iterator_tasklet,
                              memlet=dace.Memlet())
        state.add_memlet_path(update_iterator_tasklet,
                              exit,
                              memlet=dace.Memlet())

        for field_name in outputs:

            for offset in field_accesses[field_name]:
                if offset is not None and list(offset) != [0] * len(offset):
                    raise NotImplementedError("Output offsets not implemented")

            data_name = field_to_data[field_name]

            # Outer write
            data_name_outer = field_name
            data_name_inner = field_name + "_out"
            desc_outer = parent_sdfg.arrays[data_name].clone()
            desc_outer.transient = False
            array_index = ", ".join(map(str, parameters))
            try:
                sdfg.add_datadesc(data_name_outer, desc_outer)
            except NameError:  # Already an input
                parent_sdfg.arrays[data_name].access = (
                    dace.AccessType.ReadWrite)
            write_node_outer = state.add_write(data_name_outer)
            if isinstance(desc_outer, dt.Stream):
                subset = "0"
            else:
                subset = str(sbs.Range.from_array(desc_outer))
            state.add_memlet_path(
                nested_sdfg_tasklet,
                exit,
                write_node_outer,
                src_conn=data_name_inner,
                memlet=dace.Memlet(f"{data_name_outer}[{subset}]"))

            # Create inner stream
            desc_inner = desc_outer.clone()
            nested_sdfg.add_datadesc(data_name_inner, desc_inner)

            # Inner write
            write_node_inner = compute_state.add_write(data_name_inner)

            # Intermediate buffer, mostly relevant for vectorization
            output_buffer_name = field_name + "_output_buffer"
            nested_sdfg.add_array(output_buffer_name, (vector_length, ),
                                  desc_inner.dtype.base_type,
                                  storage=dace.StorageType.FPGA_Registers,
                                  transient=True)
            output_buffer = compute_state.add_access(output_buffer_name)

            # Condition write tasklet
            output_tasklet = compute_state.add_tasklet(
                field_name + "_conditional_write", {f"_{output_buffer_name}"},
                {f"_{data_name_inner}"},
                (write_cond + f"_{data_name_inner} = _{output_buffer_name}"))

            # If vectorized, we need to pass through the unrolled scope
            if vector_length > 1:
                compute_state.add_memlet_path(
                    compute_tasklet,
                    compute_unroll_exit,
                    output_buffer,
                    src_conn=field_name + "_inner_out",
                    memlet=dace.Memlet(f"{output_buffer_name}[i_unroll]"))
            else:
                compute_state.add_memlet_path(
                    compute_tasklet,
                    output_buffer,
                    src_conn=field_name + "_inner_out",
                    memlet=dace.Memlet(f"{output_buffer_name}[0]")),

            # Final memlet to the output
            compute_state.add_memlet_path(
                output_buffer,
                output_tasklet,
                dst_conn=f"_{output_buffer_name}",
                memlet=dace.Memlet(f"{output_buffer.data}[0:{vector_length}]")),
            if isinstance(desc_inner, dt.Stream):
                subset = "0"
            else:
                subset = array_index
            compute_state.add_memlet_path(
                output_tasklet,
                write_node_inner,
                src_conn=f"_{data_name_inner}",
                memlet=dace.Memlet(f"{write_node_inner.data}[{subset}]",
                                   dynamic=True)),

        return sdfg
