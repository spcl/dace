# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for filtering functions. This module includes functions from both
NumPy's Indexing Routines and Sorting, Searching, and Counting Functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, broadcast_together
from dace import data, dtypes, subsets, symbolic, Memlet, SDFG, SDFGState, nodes

from typing import List, Optional, Set


def branch_type(operand, arr: Optional[data.Data]) -> dtypes.typeclass:
    """dtype of one ``numpy.where`` branch: the array's own, else the scalar's.

    A symbolic scalar (``2 * N``) reaches here as a sympy expression, whose Python ``type()`` --
    ``sympy.Mul`` and friends -- is in no dtype map.
    """
    if arr is not None:
        return arr.dtype
    if symbolic.issymbolic(operand):
        return symbolic.symtype(operand)
    return dtypes.dtype_to_typeclass(type(operand))


def branch_code(operand) -> str:
    """The branch spelled for the tasklet: sympy prints ``**`` and ``/``, C++ needs neither."""
    return symbolic.symstr(operand) if symbolic.issymbolic(operand) else operand


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

    left_type = branch_type(left_operand, left_arr)
    right_type = branch_type(right_operand, right_arr)

    # Implicit Python coversion implemented as casting
    arguments = [cond_arr, left_arr or left_type, right_arr or right_type]
    tasklet_args = [
        '__incond', '__in1' if left_arr else branch_code(left_operand),
        '__in2' if right_arr else branch_code(right_operand)
    ]
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

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    # Broadcast condition with broadcasted left+right
    cond_out_shape, cond_all_idx_dict, cond_out_idx, cond_idx, _ = broadcast_together(cond_shape, out_shape)

    # Fix for Scalars
    if isinstance(left_arr, data.Scalar):
        left_idx = subsets.Range([(0, 0, 1)])
    if isinstance(right_arr, data.Scalar):
        right_idx = subsets.Range([(0, 0, 1)])
    if isinstance(cond_arr, data.Scalar):
        cond_idx = subsets.Range([(0, 0, 1)])

    if left_arr is None and right_arr is None:
        # Both x and y are constants: NumPy broadcasts them against the condition, so the result -- and the
        # iteration space -- is shaped like `cond`.
        if cond_arr is None or isinstance(cond_arr, data.Scalar):
            raise ValueError('numpy.where with scalar x, y and a scalar condition returns a 0-dimensional array, '
                             'which DaCe cannot represent')
        out_shape, all_idx_dict, out_idx = cond_out_shape, cond_all_idx_dict, cond_out_idx
        storage = cond_arr.storage
    else:
        storage = left_arr.storage if left_arr else right_arr.storage

    out_operand, out_arr = sdfg.add_transient(visitor.get_target_name(),
                                              out_shape,
                                              result_type,
                                              storage,
                                              find_new_name=True)

    if list(out_shape) == [1]:
        # Constant operands are inlined in the tasklet code, so they get no connector
        in_connectors = {'__incond': None}
        if left_arr:
            in_connectors['__in1'] = None
        if right_arr:
            in_connectors['__in2'] = None
        tasklet = state.add_tasklet('_where_', in_connectors, {'__out': None},
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
    elif (left_arr is not None and right_arr is not None and left_cast is None and right_cast is None
          and list(cond_out_shape) == list(out_shape)):
        # A per-element select IS Fortran ``MERGE``, so hand the whole thing to the library node
        # and let its expansion do the broadcasting -- that keeps one implementation of the select
        # for both frontends and leaves the choice of lowering (vectorised, GPU, a vendor call) to
        # whoever picks the node's implementation later.
        #
        # Restricted to the case the node can express exactly: three real arrays, no cast to insert
        # (the node's tasklet assigns straight across and has nowhere to put one), and a condition
        # that broadcasts into the result rather than widening it.
        from dace.libraries.standard.nodes import MergeLibraryNode  # Avoid import loop

        node = MergeLibraryNode('_where_')
        state.add_node(node)
        n_cond = state.add_read(cond_operand)
        n_left = left_operand_node if left_operand_node else state.add_read(left_operand)
        n_right = right_operand_node if right_operand_node else state.add_read(right_operand)
        n_out = state.add_write(out_operand)
        state.add_edge(n_left, None, node, MergeLibraryNode.TRUE_CONNECTOR_NAME,
                       Memlet.from_array(left_operand, left_arr))
        state.add_edge(n_right, None, node, MergeLibraryNode.FALSE_CONNECTOR_NAME,
                       Memlet.from_array(right_operand, right_arr))
        state.add_edge(n_cond, None, node, MergeLibraryNode.MASK_CONNECTOR_NAME,
                       Memlet.from_array(cond_operand, cond_arr))
        state.add_edge(node, MergeLibraryNode.OUTPUT_CONNECTOR_NAME, n_out, None,
                       Memlet.from_array(out_operand, out_arr))
        if generated_nodes is not None:
            generated_nodes.add(node)
            generated_nodes.add(n_cond)
            generated_nodes.add(n_out)
            if not left_operand_node:
                generated_nodes.add(n_left)
            if not right_operand_node:
                generated_nodes.add(n_right)
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
