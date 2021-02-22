# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np
from typing import List

# For optimizations
from dace.transformation.dataflow import (DoubleBuffering, MapCollapse,
                                          MapExpansion, MapReduceFusion,
                                          StripMining, InLocalStorage,
                                          AccumulateTransient, Vectorization)

# For library node implementations
import dace.libraries.blas

# Define symbolic sizes for arbitrary inputs
M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

#####################################################################
# Data-centric functions


# Map-Reduce version of matrix multiplication
@dace.program
def matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


# Library node version of matrix multiplication, using the numpy interface
@dace.program
def matmul_lib(A: dtype[M, K], B: dtype[K, N]):
    return A @ B


#####################################################################
# Data-centric optimization helpers


def tile(sdfg: dace.SDFG, map_entry: dace.nodes.MapEntry, divides_evenly: bool,
         skew: bool, **tile_sizes: dace.symbolic.SymbolicType):
    """ Helper function that tiles a Map scope by the given sizes. """
    for k, v in tile_sizes.items():
        StripMining.apply_to(sdfg,
                             dict(dim_idx=map_entry.params.index(k),
                                  tile_size=str(v),
                                  divides_evenly=divides_evenly,
                                  skew=skew),
                             _map_entry=map_entry)


def permute_map(map_entry: dace.nodes.MapEntry, perm: List[int]):
    """ Permutes indices of map according to list of integers. """
    map_entry.map.params = [map_entry.map.params[p] for p in perm]
    map_entry.map.range = [map_entry.map.range[p] for p in perm]


def extract_map_dim(sdfg: dace.SDFG, map_entry: dace.nodes.MapEntry, dim: int):
    """ Helper function that extracts a map dimension into an outer map. """
    # Make extracted dimension first
    permute_map(map_entry, [dim] +
                [i for i in range(len(map_entry.map.params)) if i != dim])
    # Expand map
    entries = MapExpansion.apply_to(sdfg, map_entry=map_entry)
    # Collapse remaining maps
    map_to_collapse = entries[1]
    for idx in range(len(entries) - 2):
        map_to_collapse, _ = MapCollapse.apply_to(
            sdfg,
            _outer_map_entry=map_to_collapse,
            _inner_map_entry=entries[idx + 2],
        )

    return entries[0], map_to_collapse


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_mapexit_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapExit:
    """ Finds the first map exit node by the given parameter name. """
    state, entry = next(
        (p, n) for n, p in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and pname in n.params)
    return state.exit_node(entry)


#####################################################################
# Matrix multiplication data-centric optimization schemes


def optimize_for_cpu(sdfg: dace.SDFG, m: int, n: int, k: int):
    """ Optimize the matrix multiplication example for multi-core CPUs. """
    # Ensure integers are 32-bit by default
    dace.Config.set('compiler', 'default_data_types', value='C')

    # Fuse the map and reduce nodes
    sdfg.apply_transformations(MapReduceFusion)

    # Find multiplication map
    entry = find_map_by_param(sdfg, 'k')

    # Create a tiling strategy
    divides_evenly = (m % 32 == 0) and (n % 32 == 0) and (k % 256 == 0)
    tile(sdfg, entry, divides_evenly, False, k=256, i=32, j=32)
    tile(sdfg, entry, divides_evenly, divides_evenly, j=16, i=4)

    # Reorder internal map to "k,i,j"
    permute_map(entry, [2, 0, 1])

    # Add local storage for B in j tile: we apply InLocalStorage with a
    # parameter "array" named B, between the two maps of j and i
    regtile_j = find_map_by_param(sdfg, 'tile1_j')
    regtile_i = find_map_by_param(sdfg, 'tile1_i')
    InLocalStorage.apply_to(sdfg,
                            dict(array='B'),
                            node_a=regtile_j,
                            node_b=regtile_i)

    if divides_evenly:
        # Add local storage for C
        exit_inner = find_mapexit_by_param(sdfg, 'k')
        exit_rti = find_mapexit_by_param(sdfg, 'tile1_i')
        AccumulateTransient.apply_to(sdfg,
                                     dict(array='C'),
                                     _map_exit=exit_inner,
                                     _outer_map_exit=exit_rti)

        # Vectorize microkernel map
        postamble = n % 4 != 0
        sdfg.apply_transformations(
            Vectorization,
            dict(vector_len=4, preamble=False, postamble=postamble))

    # Mark outer tile map as sequential to remove atomics
    find_map_by_param(sdfg,
                      'tile_k').map.schedule = dace.ScheduleType.Sequential

    # Collapse maps for more parallelism
    find_map_by_param(sdfg, 'o0').map.collapse = 2
    tile_i = find_map_by_param(sdfg, 'tile_i')
    tile_j = find_map_by_param(sdfg, 'tile_j')
    MapCollapse.apply_to(sdfg, _outer_map_entry=tile_i, _inner_map_entry=tile_j)
    tile_ij = find_map_by_param(sdfg, 'tile_i')  # Find newly created map
    tile_ij.map.schedule = dace.ScheduleType.CPU_Multicore
    tile_ij.map.collapse = 2


def optimize_for_gpu(sdfg: dace.SDFG, m: int, n: int, k: int):
    """ Optimize the matrix multiplication example for GPUs. """
    # Ensure integers are 32-bit by default
    dace.Config.set('compiler', 'default_data_types', value='C')

    # Fuse the map and reduce nodes
    sdfg.apply_transformations(MapReduceFusion)

    # Apply GPU transformation
    sdfg.apply_gpu_transformations()

    # Find multiplication map
    entry = find_map_by_param(sdfg, 'k')

    # Create a tiling strategy
    divides_evenly = (m % 64 == 0) and (n % 64 == 0) and (k % 8 == 0)
    tile(sdfg, entry, divides_evenly, True, i=64, j=64, k=8)
    tile(sdfg, entry, divides_evenly, True, i=8, j=4)

    # Create kernel schedule by collapsing and reordering maps
    gtile_i = find_map_by_param(sdfg, 'tile_i')
    gtile_j = find_map_by_param(sdfg, 'tile_j')
    btile_i = find_map_by_param(sdfg, 'tile1_i')
    btile_j = find_map_by_param(sdfg, 'tile1_j')
    MapCollapse.apply_to(sdfg,
                         _outer_map_entry=gtile_i,
                         _inner_map_entry=gtile_j)
    MapCollapse.apply_to(sdfg,
                         _outer_map_entry=btile_i,
                         _inner_map_entry=btile_j)
    btile = find_map_by_param(sdfg, 'tile1_i')
    btile.map.schedule = dace.ScheduleType.GPU_ThreadBlock

    # Add local storage (shared memory) for A and B on GPU
    ktile = find_map_by_param(sdfg, 'tile_k')
    smem_a = InLocalStorage.apply_to(sdfg,
                                     dict(array='A'),
                                     node_a=ktile,
                                     node_b=btile)
    smem_b = InLocalStorage.apply_to(sdfg,
                                     dict(array='B'),
                                     node_a=ktile,
                                     node_b=btile)
    sdfg.arrays[smem_a.data].storage = dace.StorageType.GPU_Shared
    sdfg.arrays[smem_b.data].storage = dace.StorageType.GPU_Shared

    # Add local storage (registers) for A and B
    ttile = find_map_by_param(sdfg, 'k')
    warptile, ttile = extract_map_dim(sdfg, ttile, 2)
    InLocalStorage.apply_to(sdfg,
                            dict(array='trans_gpu_A'),
                            node_a=warptile,
                            node_b=ttile)
    InLocalStorage.apply_to(sdfg,
                            dict(array='trans_gpu_B'),
                            node_a=warptile,
                            node_b=ttile)

    # Add local storage (registers) for C
    state = next(s for s in sdfg.nodes() if warptile in s.nodes())
    warptile_exit = state.exit_node(warptile)
    btile_exit = state.exit_node(btile)
    AccumulateTransient.apply_to(sdfg,
                                 _map_exit=warptile_exit,
                                 _outer_map_exit=btile_exit)
    # Set C tile to zero on allocation
    c_access = next(n for n in state.data_nodes() if n.data == 'trans_gpu_C')
    c_access.setzero = True

    # Unroll microkernel maps
    ttile.map.unroll = True

    # Apply double-buffering on shared memory
    DoubleBuffering.apply_to(sdfg, _map_entry=ktile, _transient=smem_a)


#####################################################################
# Main function

if __name__ == "__main__":
    # Arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, nargs="?", default=64)
    parser.add_argument("-K", type=int, nargs="?", default=64)
    parser.add_argument("-N", type=int, nargs="?", default=64)
    parser.add_argument('--version',
                        choices=[
                            'unoptimized', 'optimize_cpu', 'optimize_gpu',
                            'mkl', 'cublas'
                        ],
                        default='unoptimized',
                        help='''Different available versions: 
unoptimized: Run `matmul` without optimizations;  
optimize_cpu: Transform `matmul` to a reasonably-optimized version for
                multicore CPU;  
optimize_gpu: Transform `matmul` to a reasonably-optimized version for GPU;  
mkl: Use `matmul_lib` with the MKL library node implementation;  
cublas: Use `matmul_lib` with the CUBLAS library node implementation.''')
    parser.add_argument('--noverify',
                        dest='verify',
                        action='store_false',
                        help="If set, skips numpy verification.",
                        default=True)

    args = vars(parser.parse_args())
    version = args["version"]

    # Prepare data with numpy
    m = args["M"]
    k = args["K"]
    n = args["N"]
    A = np.random.rand(m, k).astype(np_dtype)
    B = np.random.rand(k, n).astype(np_dtype)
    C = np.zeros((m, n), dtype=np_dtype)

    print(f'Matrix multiplication {m}x{k}x{n} (version: {version})')

    if version == 'unoptimized':
        # Simply call the program to run it
        matmul(A, B, C)
    elif version.startswith('optimize_'):
        # Get the SDFG from the program
        sdfg: dace.SDFG = matmul.to_sdfg()
        # Call transformations to optimize
        if version == 'optimize_cpu':
            optimize_for_cpu(sdfg, m, n, k)
        elif version == 'optimize_gpu':
            optimize_for_gpu(sdfg, m, n, k)
        # Invoke the SDFG to run the optimized program (notice that now we must
        # also directly feed in the symbols)
        sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    elif version == 'mkl':
        # Set default implementation to MKL
        dace.libraries.blas.default_implementation = 'MKL'
        # Call program
        C = matmul_lib(A, B)
    elif version == 'cublas':
        # Set default implementation to CUBLAS
        dace.libraries.blas.default_implementation = 'cuBLAS'
        # Call program
        C = matmul_lib(A, B)
    else:
        raise ValueError('Invalid version %s' % version)

    if args["verify"]:
        expected = A @ B
        diff = np.linalg.norm(C - expected) / (m * n)
        print('Difference:', diff)
        exit(0 if diff <= 1e-6 else 1)
