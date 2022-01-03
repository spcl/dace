# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import click
import dace
import numpy as np
from typing import List, Tuple

# For optimizations
from dace.transformation.dataflow import (DoubleBuffering, MapCollapse,
                                          MapExpansion, MapReduceFusion,
                                          StripMining, InLocalStorage,
                                          AccumulateTransient, Vectorization)
from dace.transformation.interstate import FPGATransformSDFG
from dace.transformation import helpers as xfutil

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


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_map_and_state_by_param(
        sdfg: dace.SDFG,
        pname: str) -> Tuple[dace.nodes.MapEntry, dace.SDFGState]:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, p) for n, p in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_mapexit_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapExit:
    """ Finds the first map exit node by the given parameter name. """
    entry, state = find_map_and_state_by_param(sdfg, pname)
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
    xfutil.tile(sdfg, entry, divides_evenly, False, k=256, i=32, j=32)
    xfutil.tile(sdfg, entry, divides_evenly, divides_evenly, j=16, i=4)

    # Reorder internal map to "k,i,j"
    xfutil.permute_map(entry, [2, 0, 1])

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
                                     dict(array='C', identity=0),
                                     map_exit=exit_inner,
                                     outer_map_exit=exit_rti)

        # Vectorize microkernel map
        postamble = n % 4 != 0
        entry_inner, inner_state = find_map_and_state_by_param(sdfg, 'k')
        Vectorization.apply_to(inner_state.parent,
                               dict(vector_len=4,
                                    preamble=False,
                                    postamble=postamble),
                               _map_entry=entry_inner)

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
    xfutil.tile(sdfg, entry, divides_evenly, True, i=64, j=64, k=8)
    xfutil.tile(sdfg, entry, divides_evenly, True, i=8, j=4)

    # Create kernel schedule by collapsing and reordering maps
    gtile_i = find_map_by_param(sdfg, 'tile_i')
    gtile_j = find_map_by_param(sdfg, 'tile_j')
    btile_i = find_map_by_param(sdfg, 'tile1_i')
    btile_j = find_map_by_param(sdfg, 'tile1_j')
    MapCollapse.apply_to(sdfg,
                         _outer_map_entry=gtile_i,
                         _inner_map_entry=gtile_j,
                         permissive=True)
    MapCollapse.apply_to(sdfg,
                         _outer_map_entry=btile_i,
                         _inner_map_entry=btile_j,
                         permissive=True)
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
    warptile, ttile = xfutil.extract_map_dims(sdfg, ttile, [2])
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
                                 map_exit=warptile_exit,
                                 outer_map_exit=btile_exit)
    # Set C tile to zero on allocation
    c_access = next(n for n in state.data_nodes() if n.data == 'trans_gpu_C')
    c_access.setzero = True

    # Unroll microkernel maps
    ttile.map.unroll = True

    # Apply double-buffering on shared memory
    DoubleBuffering.apply_to(sdfg, _map_entry=ktile, _transient=smem_a)


#####################################################################
# Main function


@click.command()
@click.option('-M', type=int, default=64)
@click.option('-K', type=int, default=64)
@click.option('-N', type=int, default=64)
@click.option('--version',
              type=click.Choice(
                  ('unoptimized', 'optimize_cpu', 'optimize_gpu', 'mkl',
                   'cublas', 'fpga_naive', 'fpga_library')),
              default='unoptimized')
@click.option('--verify/--no-verify', default=True)
def cli(m, k, n, version, verify):
    """
    Different available versions:
    unoptimized: Run `matmul` without optimizations;
    optimize_cpu: Transform `matmul` to a reasonably-optimized version for
                    multicore CPU;
    optimize_gpu: Transform `matmul` to a reasonably-optimized version for GPU;
    mkl: Use `matmul_lib` with the MKL library node implementation;
    cublas: Use `matmul_lib` with the CUBLAS library node implementation.
    """

    # Prepare data with numpy
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
    elif version == 'fpga_naive':
        matmul_sdfg = matmul.to_sdfg()
        matmul_sdfg.apply_transformations(FPGATransformSDFG)
        matmul_sdfg(A=A, B=B, C=C, N=n, K=k, M=m)
    elif version == 'fpga_systolic':
        dace.libraries.blas.default_implementation = 'FPGA1DSystolic'
        C = matmul_lib(A, B)
    else:
        raise ValueError('Invalid version %s' % version)

    if verify:
        expected = A @ B
        diff = np.linalg.norm(C - expected) / (m * n)
        print('Difference:', diff)
        return 0 if diff <= 1e-6 else 1


if __name__ == "__main__":
    cli()
