# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for filtering functions. This module includes functions from both
NumPy's Indexing Routines and Sorting, Searching, and Counting Functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, broadcast_together
from dace import data, dtypes, subsets, Memlet, SDFG, SDFGState, nodes

from typing import List, Optional, Set


@oprepo.replaces('numpy.where')
def _array_array_where(visitor: ProgramVisitor,
                       sdfg: SDFG,
                       state: SDFGState,
                       cond_operand: str,
                       left_operand: str = None,
                       right_operand: str = None,
                       generated_nodes: Optional[Set[nodes.Node]] = None,
                       left_operand_node: Optional[nodes.AccessNode] = None,
                       right_operand_node: Optional[nodes.AccessNode] = None):
    from dace.frontend.python.replacements.operators import result_type

    if left_operand is None or right_operand is None:
        raise ValueError('numpy.where is only supported for the case where x and y are given')

    cond_arr = sdfg.arrays[cond_operand]
    try:
        left_arr = sdfg.arrays[left_operand]
    except KeyError:
        left_arr = None
    try:
        right_arr = sdfg.arrays[right_operand]
    except KeyError:
        right_arr = None

    left_type = left_arr.dtype if left_arr else dtypes.dtype_to_typeclass(type(left_operand))
    right_type = right_arr.dtype if right_arr else dtypes.dtype_to_typeclass(type(right_operand))

    # Implicit Python coversion implemented as casting
    arguments = [cond_arr, left_arr or left_type, right_arr or right_type]
    tasklet_args = ['__incond', '__in1' if left_arr else left_operand, '__in2' if right_arr else right_operand]
    result_type, casting = result_type(arguments[1:])
    left_cast = casting[0]
    right_cast = casting[1]

    if left_cast is not None:
        tasklet_args[1] = f"{str(left_cast).replace('::', '.')}({tasklet_args[1]})"
    if right_cast is not None:
        tasklet_args[2] = f"{str(right_cast).replace('::', '.')}({tasklet_args[2]})"

    left_shape = left_arr.shape if left_arr else [1]
    right_shape = right_arr.shape if right_arr else [1]
    cond_shape = cond_arr.shape if cond_arr else [1]

    # First broadcast left and right
    (temp_shape, _, _, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    # Then broadcast with condition to get the final output shape
    # numpy.where broadcasts all three arrays together
    (out_shape, all_idx_dict, out_idx, cond_idx, temp_idx) = broadcast_together(cond_shape, temp_shape)

    # Update left_idx and right_idx for the final output shape if temp_shape != out_shape
    if list(temp_shape) != list(out_shape):
        _, _, _, left_idx, _ = broadcast_together(left_shape, out_shape)
        _, _, _, right_idx, _ = broadcast_together(right_shape, out_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])
    if isinstance(cond_arr, data.Scalar):
        cond_idx = subsets.Range([(0, 0, 1)])

    if left_arr is None and right_arr is None:
        raise ValueError('Both x and y cannot be scalars in numpy.where')
    storage = left_arr.storage if left_arr else right_arr.storage

    out_operand, out_arr = sdfg.add_transient(visitor.get_target_name(),
                                              out_shape,
                                              result_type,
                                              storage,
                                              find_new_name=True)

    if list(out_shape) == [1]:
        tasklet = state.add_tasklet('_where_', {'__incond', '__in1', '__in2'}, {'__out'},
                                    '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1], i2=tasklet_args[2]))
        n0 = state.add_read(cond_operand)
        n3 = state.add_write(out_operand)
        if generated_nodes is not None:
            generated_nodes.add(tasklet)
            generated_nodes.add(n0)
            generated_nodes.add(n3)
        state.add_edge(n0, None, tasklet, '__incond', Memlet.from_array(cond_operand, cond_arr))
        if left_arr:
            if left_operand_node:
                n1 = left_operand_node
            else:
                n1 = state.add_read(left_operand)
                if generated_nodes is not None:
                    generated_nodes.add(n1)
            state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_arr))
        if right_arr:
            if right_operand_node:
                n2 = right_operand_node
            else:
                n2 = state.add_read(right_operand)
                if generated_nodes is not None:
                    generated_nodes.add(n2)
            state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_arr))
    else:
        inputs = {}
        inputs['__incond'] = Memlet.simple(cond_operand, cond_idx)
        if left_arr:
            inputs['__in1'] = Memlet.simple(left_operand, left_idx)
        if right_arr:
            inputs['__in2'] = Memlet.simple(right_operand, right_idx)

        input_nodes = {}
        if left_operand_node:
            input_nodes[left_operand] = left_operand_node
        if right_operand_node:
            input_nodes[right_operand] = right_operand_node
        tasklet, me, mx = state.add_mapped_tasklet("_where_",
                                                   all_idx_dict,
                                                   inputs,
                                                   '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1],
                                                                                               i2=tasklet_args[2]),
                                                   {'__out': Memlet.simple(out_operand, out_idx)},
                                                   external_edges=True,
                                                   input_nodes=input_nodes)
        if generated_nodes is not None:
            generated_nodes.add(tasklet)
            generated_nodes.add(me)
            for ie in state.in_edges(me):
                if ie.src is not left_operand_node and ie.src is not right_operand_node:
                    generated_nodes.add(ie.src)
            generated_nodes.add(mx)
            for oe in state.out_edges(mx):
                generated_nodes.add(oe.dst)

    return out_operand


@oprepo.replaces('numpy.select')
def _array_array_select(visitor: ProgramVisitor,
                        sdfg: SDFG,
                        state: SDFGState,
                        cond_list: List[str],
                        choice_list: List[str],
                        default=None):
    if len(cond_list) != len(choice_list):
        raise ValueError('numpy.select is only valid with same-length condition and choice lists')

    default_operand = default if default is not None else 0

    i = len(cond_list) - 1
    cond_operand = cond_list[i]
    left_operand = choice_list[i]
    right_operand = default_operand
    right_operand_node = None
    out_operand = None
    while i >= 0:
        generated_nodes = set()
        out_operand = _array_array_where(visitor,
                                         sdfg,
                                         state,
                                         cond_operand,
                                         left_operand,
                                         right_operand,
                                         generated_nodes=generated_nodes,
                                         right_operand_node=right_operand_node)
        i -= 1
        cond_operand = cond_list[i]
        left_operand = choice_list[i]
        right_operand = out_operand
        right_operand_node = None
        for nd in generated_nodes:
            if isinstance(nd, nodes.AccessNode) and nd.data == out_operand:
                right_operand_node = nd

    return out_operand
