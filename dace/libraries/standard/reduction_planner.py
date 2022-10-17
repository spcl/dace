# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Helper function to compute GPU schedule for reduction node "GPUAuto" expansion. """

from dace.data import Array
from typing import List
import warnings
import dataclasses
from dace.frontend.python.replacements import Size


def combine(shape, strides, dims):
    # combines the contiguous dimensions in dims
    new_shape_element = 1
    for d in dims:
        new_shape_element *= shape[d]

    combined_shape = shape[0:dims[0]]
    combined_shape.append(new_shape_element)
    combined_shape.extend(shape[dims[-1] + 1:])

    new_stride_element = strides[dims[0]]
    for d in dims[0:]:
        new_stride_element = min(new_stride_element, strides[d])

    combined_strides = strides[0:dims[0]]
    combined_strides.append(new_stride_element)
    combined_strides.extend(strides[dims[-1] + 1:])

    return combined_shape, combined_strides


def get_reduction_schedule(in_array: Array,
                           axes: List[int],
                           use_vectorization=True,
                           use_mini_warps=True,
                           warp_size=32,
                           wide_load_bytes=16):
    """
    Computes a data movement minimizing GPU reduction schedule depending on the input data shape and 
    the axes to reduce.

    :param in_array: DaCe array describing the input data
    :param axes: List of all the axes to reduce
    :param use_vectorization: If True the schedule uses vectorization if applicable. If False, vectorization will never be considered
    :param use_mini_warps: If True the schedule uses mini_warps if applicable. If False, mini_warps will never be considered
    :param warp_size: The number of threads in a warp (32 for Nvidia GPUs)
    :param wide_load_bytes: The number of bytes loaded in a wide load (depends on target GPU)

    :return: ReductionSchedule object that descibes the GPU schedule used to perform the reduction
    """

    @dataclasses.dataclass
    class ReductionSchedule:
        grid: List[Size]  #: dimension of the grid (grid = [10, 20] gives us 200 blocks of threads)
        block: List[Size]  #: dimension of the thread blocks (block = [32, 32] gives us 1024 threads)
        sequential: List[Size]  #: number of sequentially summed elements
        # if len(sequential) == 3, then this list is of the form [start, end, stride]
        contiguous_dim: bool  #: whether reducing contiguous elements in memory or not
        in_shape: List[Size]  #: input tensor shape
        in_strides: List[Size]  #: input tensor strides
        out_shape: List[Size]  #: output tensor shape
        out_strides: List[Size]  #: output tensor strides
        axes: List[int]  #: axes to reduce
        vectorize: bool  #: whether vectorization will be used or not
        mini_warps: bool  #: whether mini_warps optimization will be used or not
        num_mini_warps: int  #: number of mini_warps
        vec_len: int  #: size of vectors
        error: bool  #: True, if an error occured

    # initialize empty schedule
    schedule = ReductionSchedule([], [], [], False, [], [], [], [], [], False, False, 1, 1, False)

    shape = []
    strides = []

    dtype = in_array.dtype
    bytes = dtype.bytes

    num_loaded_elements = 1
    if use_vectorization:
        num_loaded_elements = wide_load_bytes // bytes
        schedule.vec_len = num_loaded_elements

    # Remove degenerate (size 1) dimensions
    degenerate_dims_indices = [i for i, s in enumerate(in_array.shape) if s == 1]
    shape = [s for i, s in enumerate(in_array.shape) if i not in degenerate_dims_indices]
    strides = [s for i, s in enumerate(in_array.strides) if i not in degenerate_dims_indices]

    for i in degenerate_dims_indices:
        for j in range(len(axes)):
            if axes[j] > i:
                axes[j] -= 1

    # Combine contiguous axes:
    dimensions_to_combine = []

    prev = axes[0]
    prev_combined = False
    curr_dimensions = []
    for curr in axes[1:]:
        if prev == curr - 1:
            # found contiguous axes
            if not prev_combined:
                curr_dimensions.append(prev)
            curr_dimensions.append(curr)
            prev_combined = True
        else:
            prev_combined = False
            if curr_dimensions != []:
                dimensions_to_combine.append(curr_dimensions)
                curr_dimensions = []
        prev = curr
    if curr_dimensions != []:
        dimensions_to_combine.append(curr_dimensions)

    for dims in dimensions_to_combine:
        shape, strides = combine(shape, strides, dims)
        axes = [a for a in axes if a <= dims[0]]
        axes.extend([a - len(dims) for a in axes if a > dims[-1]])

    schedule.axes = axes

    out_shape = [d for i, d in enumerate(shape) if i not in axes]

    if out_shape == []:
        out_shape = [1]

    schedule.out_shape = out_shape
    schedule.in_shape = shape
    schedule.in_strides = strides

    out_strides = [os for i, os in enumerate(strides) if i not in axes]
    removed = [(i,os) for i, os in enumerate(strides) if i in axes]
    for rem in removed:
        r = rem[1]
        s = shape[rem[0]]
        for i in range(len(out_strides)):
            if out_strides[i] > r:
                out_strides[i] //= s

    schedule.out_strides = out_strides

    for i, s in enumerate(strides):
        if s == 1:
            contiguous_dimension = i

    # non-neighbouring multi-axes reduction not supported yet (e.g. reduce axes [0,2])
    # --> TODO
    if len(axes) > 1:
        warnings.warn('Multi-axes reduction not supported yet. Falling back to pure expansion.')
        schedule.error = True
        return schedule

    if contiguous_dimension in axes:
        # we are reducing the contigious dimension
        schedule.contiguous_dim = True

        # TODO: Fix vectorization for non-exact-fitting sizes
        if (shape[contiguous_dimension] > 32) == True and (shape[contiguous_dimension] % num_loaded_elements
                                                           == 0) == True and use_vectorization:
            schedule.vectorize = True

        for i, s in enumerate(shape):
            if i != contiguous_dimension:
                schedule.grid.append(s)

        if schedule.grid == []:
            # TODO: solve this issue
            warnings.warn('Falling back to pure expansion due to invalid schedule.')
            schedule.error = True
            return schedule

        threads_per_block = warp_size if (
            shape[contiguous_dimension] > warp_size) == True else shape[contiguous_dimension]
        schedule.block = [threads_per_block]

        stride = warp_size * num_loaded_elements if schedule.vectorize else warp_size
        schedule.sequential.append([0, shape[contiguous_dimension], stride])

    else:
        # we are reducing a non-contiguous dimension

        schedule.sequential = [shape[axes[0]]]  # one thread sums up the whole axis

        schedule.block = [shape[contiguous_dimension]]

        # if too many threads per block, throw error for now --> TODO
        if (schedule.block[0] > 1024) == True:
            warnings.warn('Falling back to pure expansion due to invalid schedule.')
            schedule.error = True
            return schedule

        # grid is all the dims expect the one to reduce and the last
        schedule.grid = []
        for i, s in enumerate(shape):
            if i != contiguous_dimension and i != axes[0]:
                schedule.grid.append(s)

        if use_mini_warps:
            # Check if we can use mini_warps
            # i.e. check if warp is not filled and power of 2 (for now only works for powers of 2)
            if (schedule.block[0] < 32) == True and (schedule.block[0] & (schedule.block[0] - 1) == 0) == True:
                schedule.mini_warps = True
                schedule.num_mini_warps = warp_size // schedule.block[0]
        if use_vectorization and not schedule.mini_warps:
            # check if we can use vectorization
            if (shape[contiguous_dimension] % num_loaded_elements == 0) == True:
                schedule.vectorize = True

        if schedule.grid == []:
            # TODO: solve this issue
            warnings.warn('Falling back to pure expansion due to invalid schedule.')
            schedule.error = True
            return schedule

    return schedule
