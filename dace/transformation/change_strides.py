# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module provides a function to change the stride in a given SDFG """
from typing import List, Union, Tuple
import sympy

import dace
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG, nodes, SDFGState
from dace.data import Array, Scalar
from dace.memlet import Memlet


def list_access_nodes(sdfg: dace.SDFG, array_name: str) -> List[Tuple[nodes.AccessNode, Union[SDFGState, dace.SDFG]]]:
    """
    Find all access nodes in the SDFG of the given array name. Does not recourse into nested SDFGs.

    :param sdfg: The SDFG to search through
    :type sdfg: dace.SDFG
    :param array_name: The name of the wanted array
    :type array_name: str
    :return: List of the found access nodes together with their state
    :rtype: List[Tuple[nodes.AccessNode, Union[dace.SDFGState, dace.SDFG]]]
    """
    found_nodes = []
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == array_name:
                found_nodes.append((node, state))
    return found_nodes


def change_strides(sdfg: dace.SDFG, stride_one_values: List[str], schedule: ScheduleType) -> SDFG:
    """
    Change the strides of the arrays on the given SDFG such that the given dimension has stride 1. Returns a new SDFG.

    :param sdfg: The input SDFG
    :type sdfg: dace.SDFG
    :param stride_one_values: Length of the dimension whose stride should be set to one. Expects that each array has
    only one dimension whose length is in this list. Expects that list contains name of symbols
    :type stride_one_values: List[str]
    :param schedule: Schedule to use to copy the arrays
    :type schedule: ScheduleType
    :return: SDFG with changed strides
    :rtype: SDFG
    """
    # Create new SDFG and copy constants and symbols
    original_name = sdfg.name
    sdfg.name = "changed_strides"
    new_sdfg = SDFG(original_name)
    for dname, value in sdfg.constants.items():
        new_sdfg.add_constant(dname, value)
    for dname, stype in sdfg.symbols.items():
        new_sdfg.add_symbol(dname, stype)

    changed_stride_state = new_sdfg.add_state("with_changed_strides", is_start_block=True)
    inputs, outputs = sdfg.read_and_write_sets()
    # Get all arrays which are persistent == not transient
    persistent_arrays = {name: desc for name, desc in sdfg.arrays.items() if not desc.transient}

    # Get the persistent arrays of all the transient arrays which get copied to GPU
    for dname in persistent_arrays:
        for access, state in list_access_nodes(sdfg, dname):
            if len(state.out_edges(access)) == 1:
                edge = state.out_edges(access)[0]
                if isinstance(edge.dst, nodes.AccessNode):
                    if edge.dst.data in inputs:
                        inputs.remove(edge.dst.data)
                        inputs.add(dname)
            if len(state.in_edges(access)) == 1:
                edge = state.in_edges(access)[0]
                if isinstance(edge.src, nodes.AccessNode):
                    if edge.src.data in inputs:
                        outputs.remove(edge.src.data)
                        outputs.add(dname)

    # Only keep inputs and outputs which are persistent
    inputs.intersection_update(persistent_arrays.keys())
    outputs.intersection_update(persistent_arrays.keys())
    nsdfg = changed_stride_state.add_nested_sdfg(sdfg, inputs=inputs, outputs=outputs)
    transform_state = new_sdfg.add_state_before(changed_stride_state, label="transform_data", is_start_block=True)
    transform_state_back = new_sdfg.add_state_after(changed_stride_state, "transform_data_back", is_start_block=False)

    # copy arrays
    for dname, desc in sdfg.arrays.items():
        if not desc.transient:
            if isinstance(desc, Array):
                new_sdfg.add_array(dname, desc.shape, desc.dtype, desc.storage, desc.location, desc.transient,
                                   desc.strides, desc.offset)
            elif isinstance(desc, Scalar):
                new_sdfg.add_scalar(dname, desc.dtype, desc.storage, desc.transient, desc.lifetime, desc.debuginfo)

    new_order = {}
    new_strides_map = {}

    # Map of array names in the nested sdfg:  key: array name in parent sdfg (this sdfg), value: name in the nsdfg
    # Assumes that name changes only appear in the first level of nsdfg nesting
    array_names_map = {}
    for graph in sdfg.cfg_list:
        if graph.parent_nsdfg_node is not None:
            if graph.parent_sdfg == sdfg:
                for connector in graph.parent_nsdfg_node.in_connectors:
                    for in_edge in graph.parent.in_edges_by_connector(graph.parent_nsdfg_node, connector):
                        array_names_map[str(connector)] = in_edge.data.data

    for containing_sdfg, dname, desc in sdfg.arrays_recursive():
        shape_str = [str(s) for s in desc.shape]
        # Get index of the dimension we want to have stride 1
        stride_one_idx = None
        this_stride_one_value = None
        for dim in stride_one_values:
            if str(dim) in shape_str:
                stride_one_idx = shape_str.index(str(dim))
                this_stride_one_value = dim
                break

        if stride_one_idx is not None:
            new_order[dname] = [stride_one_idx]

            new_strides = list(desc.strides)
            new_strides[stride_one_idx] = sympy.S.One

            previous_size = dace.symbolic.symbol(this_stride_one_value)
            previous_stride = sympy.S.One
            for i in range(len(new_strides)):
                if i != stride_one_idx:
                    new_order[dname].append(i)
                    new_strides[i] = previous_size * previous_stride
                    previous_size = desc.shape[i]
                    previous_stride = new_strides[i]

            new_strides_map[dname] = {}
            # Create a map entry for this data linking old strides to new strides. This assumes that each entry in
            # strides is unique which is given as otherwise there would be two dimension i, j where a[i, j] would point
            # to the same address as a[j, i]
            for new_stride, old_stride in zip(new_strides, desc.strides):
                new_strides_map[dname][old_stride] = new_stride
            desc.strides = tuple(new_strides)
        else:
            parent_name = array_names_map[dname] if dname in array_names_map else dname
            if parent_name in new_strides_map:
                new_strides = []
                for stride in desc.strides:
                    new_strides.append(new_strides_map[parent_name][stride])
                desc.strides = new_strides

    # Add new flipped arrays for every non-transient array
    flipped_names_map = {}
    for dname, desc in sdfg.arrays.items():
        if not desc.transient:
            flipped_name = f"{dname}_flipped"
            flipped_names_map[dname] = flipped_name
            new_sdfg.add_array(flipped_name, desc.shape, desc.dtype, desc.storage, desc.location, True, desc.strides,
                               desc.offset)

    # Deal with the inputs: Create tasklet to flip them and connect via memlets
    # for input in inputs:
    for input in set([*inputs, *outputs]):
        if input in new_order:
            flipped_data = flipped_names_map[input]
            if input in inputs:
                changed_stride_state.add_memlet_path(changed_stride_state.add_access(flipped_data),
                                                     nsdfg,
                                                     dst_conn=input,
                                                     memlet=Memlet(data=flipped_data))
            # Simply need to copy the data, the different strides take care of the transposing
            arr = sdfg.arrays[input]
            tasklet, map_entry, map_exit = transform_state.add_mapped_tasklet(
                name=f"transpose_{input}",
                map_ranges={
                    f"_i{i}": f"0:{s}"
                    for i, s in enumerate(arr.shape)
                },
                inputs={'_in': Memlet(data=input, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                code='_out = _in',
                outputs={
                    '_out': Memlet(data=flipped_data, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))
                },
                external_edges=True,
                schedule=schedule,
            )
    # Do the same for the outputs
    for output in outputs:
        if output in new_order:
            flipped_data = flipped_names_map[output]
            changed_stride_state.add_memlet_path(nsdfg,
                                                 changed_stride_state.add_access(flipped_data),
                                                 src_conn=output,
                                                 memlet=Memlet(data=flipped_data))
            # Simply need to copy the data, the different strides take care of the transposing
            arr = sdfg.arrays[output]
            tasklet, map_entry, map_exit = transform_state_back.add_mapped_tasklet(
                name=f"transpose_{output}",
                map_ranges={
                    f"_i{i}": f"0:{s}"
                    for i, s in enumerate(arr.shape)
                },
                inputs={'_in': Memlet(data=flipped_data, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                code='_out = _in',
                outputs={'_out': Memlet(data=output, subset=", ".join(f"_i{i}" for i, _ in enumerate(arr.shape)))},
                external_edges=True,
                schedule=schedule,
            )
    # Deal with any arrays which have not been flipped (should only be scalars). Connect them directly
    for dname, desc in sdfg.arrays.items():
        if not desc.transient and dname not in new_order:
            if dname in inputs:
                changed_stride_state.add_memlet_path(changed_stride_state.add_access(dname),
                                                     nsdfg,
                                                     dst_conn=dname,
                                                     memlet=Memlet(data=dname))
            if dname in outputs:
                changed_stride_state.add_memlet_path(nsdfg,
                                                     changed_stride_state.add_access(dname),
                                                     src_conn=dname,
                                                     memlet=Memlet(data=dname))

    return new_sdfg
