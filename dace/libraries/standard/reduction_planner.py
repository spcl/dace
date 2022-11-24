# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Helper function to compute GPU schedule for reduction node "GPUAuto" expansion. """

from dace.data import Array
from typing import List
import dataclasses
from dace.frontend.python.replacements import Size


def combine(shape, strides, dims):
    # combines the contiguous dimensions in dims and returns new shape and strides
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

def simplify_input(shape, strides, axes):
    # simplifies the input tensor by combining neighboring reduced axes and neighboring non-reduced axes
    # returns new shape, new strides, new axes and also output shape and output strides
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

    num_axes_combined = 0
    for dims in dimensions_to_combine:
        # update current axes to combine to prevent out of bounds
        dims = [d - num_axes_combined for d in dims]
        num_axes_combined += len(dims) - 1

        # do the combine and update axes
        shape, strides = combine(shape, strides, dims)
        new_axes = [a for a in axes if a <= dims[0]]
        new_axes.extend([a - len(dims) + 1 for a in axes if a > dims[-1]])
        axes = new_axes

    # now combine the non-reduced axes
    dimensions_to_combine = []
    prev_axis = axes[0]
    if axes[0] >= 2:
        dimensions_to_combine.append(list(range(axes[0])))
    for ax in axes[1:]:
        if ax - prev_axis > 2:
            dimensions_to_combine.append(list(range(prev_axis + 1, ax)))
        prev_axis = ax
    
    if len(shape) - axes[-1] > 2:
        dimensions_to_combine.append(list(range(axes[-1] + 1, len(shape))))

    num_axes_combined = 0
    for dims in dimensions_to_combine:
        # update current axes to combine to prevent out of bounds
        dims = [d - num_axes_combined for d in dims]
        num_axes_combined += len(dims) - 1

        # do the combine and update axes
        shape, strides = combine(shape, strides, dims)
        new_axes = [a for a in axes if a <= dims[0]]
        new_axes.extend([a - len(dims) + 1 for a in axes if a > dims[-1]])
        axes = new_axes

    out_shape = [d for i, d in enumerate(shape) if i not in axes]

    if out_shape == []:
        out_shape = [1]

    out_strides = [os for i, os in enumerate(strides) if i not in axes]
    removed = [(i,os) for i, os in enumerate(strides) if i in axes]
    for rem in removed:
        r = rem[1]
        s = shape[rem[0]]
        for i in range(len(out_strides)):
            if out_strides[i] > r:
                out_strides[i] //= s
            
    return shape, strides, axes, out_shape, out_strides


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
        grid: List[Size]  #: dimension of the grid
        block: List[Size]  #: dimension of the thread blocks
        sequential: List[Size]  #: number of sequentially summed elements. If len(sequential) == 3, then this list is of the form [start, end, stride]
        shared_mem_size: int #: number of shared memory expansion has to allocate

        in_shape: List[Size]  #: input tensor shape
        in_strides: List[Size]  #: input tensor strides
        out_shape: List[Size]  #: output tensor shape
        out_strides: List[Size]  #: output tensor strides
        axes: List[int]  #: axes to reduce
        contiguous_dim: bool  #: whether reducing contiguous elements in memory or not

        vectorize: bool  #: whether vectorization will be used or not
        vec_len: int  #: size of vectors
        mini_warps: bool  #: whether mini_warps optimization will be used or not
        num_mini_warps: int  #: number of mini_warps
        one_d_reduction: bool #: True, if we have a 1D reduction (i.e. sum up all input elements to one output element)

        multi_axes: bool #: True, if the reduction reduces multiple axes
        additional_grid: List[Size] #: For multi-axes reduction, we have additional grid dimensions
        changed_in_shape: List[Size] #: The input shape of the single axis reduction inside a multi-axes reduction
        changed_in_strides: List[Size] #: The input strides of the single axis reduction inside a multi-axes reduction
        changed_axes: List[int] #: The axis to reduce of the single axis reduction inside a multi-axes reduction

        error: str  #: if not "", error contains the error reason as warning

    # initialize empty schedule
    schedule = ReductionSchedule([], [], [], 0, [], [], [], [], [], False, False, 1, False, 1, False, False, [], [], [], [], '')

    initial_shape = in_array.shape
    initial_strides = in_array.strides
    dtype = in_array.dtype
    bytes = dtype.bytes

    shape = []
    strides = []

    num_loaded_elements = 1
    if use_vectorization:
        num_loaded_elements = wide_load_bytes // bytes
        schedule.vec_len = num_loaded_elements

    # Remove degenerate (size 1) dimensions
    degenerate_dims_indices = [i for i, s in enumerate(initial_shape) if s == 1]
    shape = [s for i, s in enumerate(initial_shape) if i not in degenerate_dims_indices]
    strides = [s for i, s in enumerate(initial_strides) if i not in degenerate_dims_indices]

    iteration = 0
    for i in degenerate_dims_indices:
        # update the current indices i if previous axes were removed
        i -= iteration
        iteration += 1
        for j in range(len(axes)):
            if axes[j] > i:
                axes[j] -= 1

    
    # simplify the input
    shape, strides, axes, out_shape, out_strides = simplify_input(shape, strides, axes)

    schedule.in_shape = shape
    schedule.in_strides = strides
    schedule.axes = axes
    schedule.out_shape = out_shape
    schedule.out_strides = out_strides

    
    for i, s in enumerate(strides):
        if s == 1:
            contiguous_dimension = i

    if len(axes) > 1:
        # we need to compute a multi-axes reduction
        schedule.multi_axes = True
        schedule.additional_grid = [shape[i] for i in axes[:-1]]
        schedule.changed_in_shape = [s for i, s in enumerate(schedule.in_shape) if i not in axes[:-1]]
        schedule.changed_axes = [axes[-1] - len(axes) + 1]


        removed_shapes = [s for i, s in enumerate(schedule.in_shape) if i in axes[:-1]]
        removed_strides = [s for i, s in enumerate(schedule.in_strides) if i in axes[:-1]]
        schedule.changed_in_strides = [s for i, s in enumerate(schedule.in_strides) if i not in axes[:-1]]
        for i in range(len(schedule.changed_in_strides)):
            for r in range(len(removed_strides)):
                schedule.changed_in_strides[i] = schedule.changed_in_strides[i] if schedule.changed_in_strides[i] < removed_strides[r] else schedule.changed_in_strides[i] // removed_shapes[r]
        
        axes = schedule.changed_axes
        shape = schedule.changed_in_shape
        for i, s in enumerate(schedule.changed_in_strides):
            if s == 1:
                contiguous_dimension = i
        

    # now compute the schedule depending on contiguous or strided reduction
    if contiguous_dimension in axes:
        # we are reducing the contiguous dimension
        schedule.contiguous_dim = True

        # TODO: Fix vectorization for non-exact-fitting sizes
        if (shape[contiguous_dimension] > 32) == True and (shape[contiguous_dimension] % num_loaded_elements
                                                        == 0) == True and use_vectorization:
            schedule.vectorize = True

        # all non-reduced dimensions in grid
        for i, s in enumerate(shape):
            if i != contiguous_dimension:
                schedule.grid.append(s)

        # if grid is empty, set it to 1
        if schedule.grid == []:
            schedule.grid = [1]

        # 32 threads per block unless contiguous dimension too small
        threads_per_block = warp_size if (
            shape[contiguous_dimension] > warp_size) == True else shape[contiguous_dimension]
        schedule.block = [threads_per_block]

        stride = warp_size * num_loaded_elements if schedule.vectorize else warp_size
        # 1 thread block sums up the shape[contiguous_dimension] elements with a stride 
        schedule.sequential.append([0, shape[contiguous_dimension], stride])

        # check if we have 1D reduction
        if len(schedule.in_shape) == 1 and schedule.out_shape == [1]:
            schedule.one_d_reduction = True
            # increase schedule.grid to appropriate value --> each block sums up 1024 values
            schedule.grid = [(schedule.in_shape[0] + 1024 -1) // 1024]

    else:
        # we are reducing a non-contiguous dimension

        schedule.grid = shape[:axes[0]]   # add all leading dimensions into the grid
        schedule.grid.append(shape[contiguous_dimension] / 32) # each block computes 32 output values

        schedule.block = [16, 32]   # we use 16 threads per output value (could be any value in {1, ... , 32})

        schedule.shared_mem_size = 32 # each block uses 32 shared memory locations
        schedule.sequential = [shape[axes[0]]] # the 16 threads sum up the whole axis

        if use_mini_warps and shape[contiguous_dimension] <= 16:
            # we turn on mini_warps
            schedule.mini_warps = True
            schedule.num_mini_warps = warp_size // shape[contiguous_dimension]
            # we now use 16 * schedule.num_mini_warps threads to compute 1 output element
            schedule.block = [16, shape[contiguous_dimension]]
            schedule.shared_mem_size = shape[contiguous_dimension]


    # basic validity checks for computed schedule
    # maybe these checks could be done in the CUDA code generator for all generated CUDA code?
    num_threads = 1
    for t in schedule.block:
        num_threads *= t
    if num_threads > 1024:
        # too many threads per block
        schedule.error = 'Schedule is invalid (more than 1024 threads per block). Falling back to pure expansion.'
    whole_grid = schedule.additional_grid + schedule.grid
    last_grid_dim = 1
    for b in whole_grid[:-2]:
        last_grid_dim *= b
    if whole_grid[-1] > 2147483647 or (len(whole_grid) > 1 and whole_grid[-2] > 65535) or last_grid_dim > 65535:
        # grid dimension must not exceed 2147483647, resp . 65535
        schedule.error = 'Schedule is invalid (some grid dimension too large). Falling back to pure expansion.'

    return schedule

