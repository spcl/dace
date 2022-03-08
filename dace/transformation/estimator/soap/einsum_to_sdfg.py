import dace
import numpy as np
import opt_einsum as oe
import re

from dace.dtypes import DTYPE_TO_TYPECLASS as dt
from dace import subsets
from typing import List


def sdfg_gen(subscripts: str, arrays: List[np.ndarray] = None, inp_dim: int = 30) -> dace.SDFG:
    """ 
    Generates an SDFG for the given einsum discription. The SDFG comprises
        separates map for each contraction, based on the analysis done by the
        opt_einsum module.

    Args:
        subscripts (str): The einsum's subscript-description.
        arrays [Optional] (List[np.ndarray]): The einsum's inputs. If not provided,
                the arrays are auto-generated using the same size in each tensor mode (inp_dim)
        inp_dim [Optinal] (int): If arrays are not provided, inp_dim is used to auto-generate them


    Returns:
        dace.SDFG: The SDFG implementing the einsum.
    """

    # if input arrays are not provided, create them
    if not arrays:
        arrays = []
        inputs = subscripts.replace(' ', '').split('->')[0].split(',')         
        for input in inputs:
            order = len(input)
            A = np.random.rand(inp_dim**order).reshape([inp_dim] * order)           
            arrays.append(A)
            

    # Extract symbols
    path_info = oe.contract_path(subscripts, *arrays)

    counter = 0
    array_names = []
    unique_arrays = []
    dace_arrays = {}
    symbols = {}
    symidx = 0
    array_shapes = {}
    tokens = re.split(',|->', subscripts)
    for arr, t in zip(arrays, tokens[:-1]):
        try:
            idx = unique_arrays.index(arr)
            arr_name = f'inp{idx}'
        except ValueError:
            unique_arrays.append(arr)
            arr_name = f'inp{counter}'
            counter += 1
            dace_arrays[arr_name] = arr
            shape = []
            for char in t.strip():
                if char not in symbols:
                    newsym = dace.symbol(f'S{symidx}')
                    symidx += 1
                    symbols[char] = newsym
                shape.append(symbols[char])
            array_shapes[arr_name] = shape
        array_names.append(arr_name)
    
    sdfg = dace.SDFG('tensor_contractions')
    for name, arr in dace_arrays.items():
        sdfg.add_array(name, array_shapes[name], dt[arr.dtype.type])
    
    counter = 0
    state = None
    for contraction in path_info[1].contraction_list:

        print(contraction)

        tokens = re.split(',|->', contraction[2])
        assert(len(tokens) == 3)

        first_idx, second_idx = contraction[0]
        if first_idx > second_idx:
            first_inp = array_names.pop(first_idx)
            second_inp = array_names.pop(second_idx)
        else:
            second_inp = array_names.pop(second_idx)
            first_inp = array_names.pop(first_idx)

        first_arr = dace_arrays[first_inp]
        second_arr = dace_arrays[second_inp]
        output = np.einsum(contraction[2], first_arr, second_arr)
        out_name = f'out{counter}'
        counter += 1
        array_names.append(out_name)
        dace_arrays[out_name] = output
        output_shape = [symbols[char] for char in tokens[-1].strip()]
        sdfg.add_transient(out_name, output_shape, dt[output.dtype.type])

        prv_state = state
        state = sdfg.add_state(f'contraction_{first_inp}_{second_inp}')
        if prv_state:
            sdfg.add_edge(prv_state, state, dace.InterstateEdge())

        map_ranges = {}
        needs_wcr = set(tokens[0]).union(set(tokens[1])) != set(tokens[2])
        wcr = None
        if needs_wcr:
            wcr = 'lambda a, b: a + b'
        for t, arr in zip(tokens, [first_arr, second_arr, output]):
            for i, idx in enumerate(t.strip()):
                if idx in map_ranges.keys():
                    continue
                # map_ranges[idx] = (0, arr.shape[i]-1, 1)
                map_ranges[idx] = (0, symbols[idx]-1, 1)
        in_memlets = {}
        in_memlets['__in0'] = dace.Memlet(
                data=first_inp,
                subset=','.join(c for c in tokens[0].strip()))
        in_memlets['__in1'] = dace.Memlet(
                data=second_inp,
                subset=','.join(c for c in tokens[1].strip()))
        out_memlets = {}
        out_memlets['__out'] = dace.Memlet(
            data = out_name,
            subset=','.join(c for c in tokens[-1].strip()),
            wcr=wcr)
        state.add_mapped_tasklet(state.label,
                                 map_ranges,
                                 in_memlets, 
                                 '__out = __in0 * __in1',
                                 out_memlets,
                                 external_edges=True)
    
    counter -= 1
    sdfg.arrays[f'out{counter}'].transient = False

    return sdfg


if __name__ == '__main__':

    dim = 30
    I = np.random.rand(dim, dim, dim, dim)
    A = np.random.rand(dim, dim)
    B = np.random.rand(dim, dim)
    C = np.random.rand(dim, dim)
    D = np.random.rand(dim, dim)
    E = np.random.rand(dim, dim, dim)

    # einsum_string = 'ik, kjl->ijl'
    einsum_string = 'ijk,jl,kl->il'
    # # einsum_string = 'ik,kj->ijk'
    sdfg = sdfg_gen(einsum_string, [E, A, B])
    # sdfg = sdfg_gen(einsum_string, [A, B])

    # einsum_string = 'pi,qj,ijkl,rk,sl->pqrs'
    # sdfg = sdfg_gen(einsum_string, [A, B, I, C, D])
    import pathlib
    sdfg.save(f'{pathlib.Path(__file__).parent.resolve()}/test.sdfg')




def explicit_correlation(image, kernel):
    hi, wi= image.shape
    hk, wk = kernel.shape
    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))    
    image_padded[hk//2:-hk//2, wk//2:-wk//2] = image
    out = np.zeros(shape=image.shape)
    for row in range(hi):
        for col in range(wi):
            for i in range(hk):
                for j in range(wk):
                    out[row, col] += image_padded[row + i, col + j]*kernel[i, j]
    return out
