# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for filtering functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, broadcast_together
from dace import data, dtypes, subsets, Memlet, SDFG, SDFGState


@oprepo.replaces('numpy.where')
def _array_array_where(visitor: ProgramVisitor,
                       sdfg: SDFG,
                       state: SDFGState,
                       cond_operand: str,
                       left_operand: str = None,
                       right_operand: str = None):
    from dace.frontend.python.replacements.operators import result_type

    if left_operand is None or right_operand is None:
        raise ValueError('numpy.where is only supported for the case where x and y are given')

    cond_arr = sdfg.arrays[cond_operand]
    left_arr = sdfg.arrays.get(left_operand, None)
    right_arr = sdfg.arrays.get(right_operand, None)

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

    (out_shape, all_idx_dict, out_idx, left_idx, right_idx) = broadcast_together(left_shape, right_shape)

    # Broadcast condition with broadcasted left+right
    _, _, _, cond_idx, _ = broadcast_together(cond_shape, out_shape)

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

    out_operand, out_arr = sdfg.add_temp_transient(out_shape, result_type, storage)

    if list(out_shape) == [1]:
        tasklet = state.add_tasklet('_where_', {'__incond', '__in1', '__in2'}, {'__out'},
                                    '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1], i2=tasklet_args[2]))
        n0 = state.add_read(cond_operand)
        n3 = state.add_write(out_operand)
        state.add_edge(n0, None, tasklet, '__incond', Memlet.from_array(cond_operand, cond_arr))
        if left_arr:
            n1 = state.add_read(left_operand)
            state.add_edge(n1, None, tasklet, '__in1', Memlet.from_array(left_operand, left_arr))
        if right_arr:
            n2 = state.add_read(right_operand)
            state.add_edge(n2, None, tasklet, '__in2', Memlet.from_array(right_operand, right_arr))
        state.add_edge(tasklet, '__out', n3, None, Memlet.from_array(out_operand, out_arr))
    else:
        inputs = {}
        inputs['__incond'] = Memlet.simple(cond_operand, cond_idx)
        if left_arr:
            inputs['__in1'] = Memlet.simple(left_operand, left_idx)
        if right_arr:
            inputs['__in2'] = Memlet.simple(right_operand, right_idx)
        state.add_mapped_tasklet("_where_",
                                 all_idx_dict,
                                 inputs,
                                 '__out = {i1} if __incond else {i2}'.format(i1=tasklet_args[1], i2=tasklet_args[2]),
                                 {'__out': Memlet.simple(out_operand, out_idx)},
                                 external_edges=True)

    return out_operand
