# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly multi GPU distributed Jacobi-2D sample."""
import dace
import numpy as np
import os
import timeit
from typing import List
import argparse

from dace.transformation.dataflow import MapFusion, RedundantSecondArray, RedundantArray, GPUMultiTransformMap
from dace.transformation.interstate import InlineSDFG, StateFusion, GPUTransformSDFG
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.libraries.nccl import utils as nutil

d_int = dace.int64
d_float = dace.float64
np_float = np.float64

TSTEPS = dace.symbol('TSTEPS', dtype=d_int, integer=True, positive=True)
N = dace.symbol('N', dtype=d_int, integer=True, positive=True)
lNy = dace.symbol('lNy', dtype=d_int, integer=True, positive=True)
Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
pi = dace.symbol('pi', dtype=dace.int32, integer=True, nonnegative=True)
size = dace.symbol('size', dtype=dace.int32, integer=True, positive=True)
Ny = Py * lNy


@dace.program
def jacobi_2d_shared(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):

    for t in range(TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def jacobi_2d_sgpu(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):
    for t in range(TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


def init_data(N, datatype):

    A = np.fromfunction(lambda i, j: i * (j + 2) / N,
                        shape=(N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N,
                        shape=(N, N),
                        dtype=datatype)

    return A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_data_desc(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return next(d for s, n, d in sdfg.arrays_recursive() if n == name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ts", type=int, nargs="?", default=10)
    parser.add_argument("n", type=int, nargs="?", default=128)
    parser.add_argument("g", type=int, nargs="?", default=4)
    args = vars(parser.parse_args())

    ts = args['ts']
    n = args['n']
    number_of_gpus = args['g']
    sdfg = jacobi_2d_sgpu.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray])

    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUTransformSDFG, options={'gpu_id': 0})
    sdfg.apply_strict_transformations()

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/jacobi/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

    print('GPU: start')
    A, B = init_data(n, np_float)
    sdfg(A, B)
    print('GPU: done')

    print('CPU: start')
    refA, refB = init_data(n, np_float)
    shared_sdfg = jacobi_2d_shared.to_sdfg()
    shared_sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    shared_sdfg(A=refA, B=refB)
    print('CPU: done')

    print("=======Validation=======")
    assert (np.allclose(A, refA)), f'A:\n{repr(A)}\nrefA:\n{repr(refA)}'
    assert (np.allclose(B, refB)), f'A:\n{repr(B)}\nrefA:\n{repr(refB)}'
    print("OK")
