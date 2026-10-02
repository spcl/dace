# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for filtering functions. This module includes functions from both
NumPy's Indexing Routines and Sorting, Searching, and Counting Functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, broadcast_together
from dace import dtypes, Memlet, SDFG, SDFGState, nodes

from typing import List, Optional, Set, Tuple


def constant_scalar(sdfg: SDFG, state: SDFGState, value, dtype: dtypes.typeclass,
                    storage: dtypes.StorageType) -> Tuple[str, nodes.AccessNode, nodes.Tasklet]:
    """Writes the constant ``value`` to a new transient scalar, which a library node can then read."""
    name, desc = sdfg.add_scalar('__where_constant', dtype, transient=True, storage=storage, find_new_name=True)
    tasklet = state.add_tasklet('_where_constant_', {}, {'__out'}, f'__out = {value}')
    node = state.add_write(name)
    state.add_edge(tasklet, '__out', node, None, Memlet.from_array(name, desc))
    return name, node, tasklet


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
    from dace.libraries.standard.nodes import MergeLibraryNode  # Avoid import loop

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
    if left_arr is None and right_arr is None:
        raise ValueError('Both x and y cannot be scalars in numpy.where')

    left_type = left_arr.dtype if left_arr else dtypes.dtype_to_typeclass(type(left_operand))
    right_type = right_arr.dtype if right_arr else dtypes.dtype_to_typeclass(type(right_operand))
    out_type, _ = result_type([left_arr or left_type, right_arr or right_type])
    storage = left_arr.storage if left_arr else right_arr.storage

    left_shape = left_arr.shape if left_arr else [1]
    right_shape = right_arr.shape if right_arr else [1]
    out_shape = broadcast_together(broadcast_together(left_shape, right_shape)[0], cond_arr.shape)[0]
    out_operand, out_arr = sdfg.add_transient(visitor.get_target_name(),
                                              out_shape,
                                              out_type,
                                              storage,
                                              find_new_name=True)

    node = MergeLibraryNode('_where_')
    out_node = state.add_write(out_operand)
    new_nodes = [node, out_node]
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, out_node, None, Memlet.from_array(out_operand, out_arr))
    for connector, operand, desc, given, operand_type in (
        (node.TRUE_CONNECTOR_NAME, left_operand, left_arr, left_operand_node, left_type),
        (node.FALSE_CONNECTOR_NAME, right_operand, right_arr, right_operand_node, right_type),
        (node.MASK_CONNECTOR_NAME, cond_operand, cond_arr, None, None),
    ):
        if desc is None:
            operand, given, tasklet = constant_scalar(sdfg, state, operand, operand_type, storage)
            desc = sdfg.arrays[operand]
            new_nodes += [tasklet, given]
        elif given is None:
            given = state.add_read(operand)
            new_nodes.append(given)
        state.add_edge(given, None, node, connector, Memlet.from_array(operand, desc))
    if generated_nodes is not None:
        generated_nodes.update(new_nodes)

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
