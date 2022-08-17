# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helper function to compute GPU schedule for reduction node "auto" expansion
"""

from dace import dtypes
from dace.data import Array
from typing import List


def get_reduction_schedule(in_array: Array, axes: List[int]):
    
    # apply vectorization or not
    use_vectorization = True
    # apply mini-warps optimitation or not
    use_mini_warps = True

    # define the warp size
    warp_size = 32
    wide_load_size = 16     #bytes

    schedule = {'grid': [], 'block': [], 'sequential': [], 'shared_mem': 0, 'contiguous_dim': False,
                'in_shape' : [], 'out_shape': [], 'in_strides' : [], 'out_strides' : [], 'axes' : [],
                'vectorize' : False, 'mini_warps' : False, 'vec_len' : 1, 'error' : False, 'num_mini_warps' : 0}

    shape = []
    strides = []


    dtype = in_array.dtype
    bytes = dtype.bytes


    num_loaded_elements = 1
    if use_vectorization:
        num_loaded_elements = wide_load_size // bytes
        schedule['vec_len'] = num_loaded_elements



    # remove "fake" dimensions
    for i in range(len(in_array.shape)):
        if in_array.shape[i] != 1:
            shape.append(in_array.shape[i])
            strides.append(in_array.strides[i])
        else:
            # decrease axes if necessary
            for j in range(len(axes)):
                if axes[j] > i:
                    axes[j] -= 1

    

    # combine axes:
    combined_shape = []
    combined_axes = []
    combined_strides = []
    
    curr_axis_index = 0
    i = 0
    num_combined = 0
    prev_reduced = False

    while i < len(shape):
        if curr_axis_index < len(axes):
            curr_axis = axes[curr_axis_index]
        else:
            curr_axis = -1
        if not prev_reduced and i != 0 and not i == curr_axis:
            combined_shape[-1] *= shape[i]
            combined_strides[-1] = strides[i]
            num_combined += 1
        else:
            combined_shape.append(shape[i])
            combined_strides.append(strides[i])
        i += 1
        prev_reduced = False
        if i-1 == curr_axis:
            prev_reduced = True
            # add curr dim and axis
            combined_axes.append(curr_axis - num_combined)
            curr_axis_index += 1
            while curr_axis_index < len(axes) and axes[curr_axis_index-1] + 1 == axes[curr_axis_index]:
                # actually combine
                combined_shape[-1] *= shape[i]
                combined_strides[-1] = strides[i]
                i += 1
                curr_axis_index += 1
                num_combined += 1


    shape = combined_shape
    axes = combined_axes
    strides = combined_strides

    schedule['axes'] = axes

    out_shape = []

    for i, d in enumerate(shape):
        if i not in axes:
            out_shape.append(d)

    if out_shape == []:
        out_shape = [1]
    
    schedule['out_shape'] = out_shape
    schedule['in_shape'] = shape
    schedule['in_strides'] = strides

    out_strides = [1] * len(out_shape)
    for i, d in enumerate(out_shape):
        for j in range(i):
            out_strides[j] *= d

    schedule['out_strides'] = out_strides



    # get contiguity:
    contiguity = []
    enum_strides = list(enumerate(strides))

    for i, s in enum_strides:
        if s == 1:
            contiguous_dimension = i

    try:
        # may fail if strides arent numbers (i.e. symbols)
        enum_strides.sort(key=lambda a: a[1])
        for i, s in enum_strides:
            contiguity.append(i)
    except Exception:
        pass



    # non-neighbouring multi-axes reduction not supported yet (e.g. reduce axes [0,2])
    if len(axes) > 1:
        schedule['error'] = True
        return schedule
                    

    if contiguous_dimension in axes:
        # we are reducing the contigious dimension
        schedule['contiguous_dim'] = True

        # TODO: Fix vectorization for non-exact-fitting sizes
        if (shape[contiguous_dimension] > 32) == True and (shape[contiguous_dimension] % num_loaded_elements == 0) == True and use_vectorization:
            schedule['vectorize'] = True

        for i, s in enumerate(shape):
            if i != contiguous_dimension:
                schedule['grid'].append(s)
        
        if schedule['grid'] == []:
            # TODO: solve this issue
            schedule['error'] = True
            return schedule


        threads_per_block = warp_size if (shape[contiguous_dimension] > warp_size) == True else shape[contiguous_dimension]
        schedule['block'] = [threads_per_block]


        stride = warp_size * num_loaded_elements if schedule['vectorize'] else warp_size
        schedule['sequential'].append([0, shape[contiguous_dimension], stride])




    else:
        # we are reducing a non-contiguous dimension


        schedule['sequential'] = [shape[axes[0]]]      # one thread sums up the whole axis

        schedule['block'] = [shape[contiguous_dimension]]

        # if too many threads per block, throw error for now --> TODO
        if (schedule['block'][0] > 1024) == True:
            schedule['error'] = True
            return schedule

        # grid is all the dims expect the one to reduce and the last
        schedule['grid'] = []
        for i, s in enumerate(shape):
            if i != contiguous_dimension and i != axes[0]:
                schedule['grid'].append(s)

        if use_mini_warps:
            #check if we can use mini_warps
            # i.e. check if warp is not filled and power of 2 (for now only works for powers of 2)
            if (schedule['block'][0] < 32) == True and (schedule['block'][0] & (schedule['block'][0]-1) == 0) == True:
                schedule['mini_warps'] = True
                schedule['num_mini_warps'] = warp_size // schedule['block'][0]
        if use_vectorization and not schedule['num_mini_warps']:
            #check if we can use vectorization
            if (shape[contiguous_dimension] % num_loaded_elements == 0) == True:
                schedule['vectorize'] = True

        if schedule['grid'] == []:
            # TODO: solve this issue
            schedule['error'] = True
            return schedule
    
    return schedule



