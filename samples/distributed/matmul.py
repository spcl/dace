#!/usr/bin/env python

import argparse
import dace
from sympy import Mod, floor, ceiling
import math
import numpy as np
import scipy as sp
from mpi4py import MPI

from dace.sdfg import SDFG
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL

from functools import reduce
from itertools import product
from typing import List, Tuple

S0 = dace.symbol('S0')
S1 = dace.symbol('S1')
S2 = dace.symbol('S2')

P0A = dace.symbol('P0A')
P1A = dace.symbol('P1A')
P0B = dace.symbol('P0B')
P1B = dace.symbol('P1B')
P0C = dace.symbol('P0C')
P1C = dace.symbol('P1C')
P0I = dace.symbol('P0I')
P1I = dace.symbol('P1I')
P2I = dace.symbol('P2I')

B0A = dace.symbol('B0A')
B1A = dace.symbol('B1A')
B0B = dace.symbol('B0B')
B1B = dace.symbol('B1B')
B0C = dace.symbol('B0C')
B1C = dace.symbol('B1C')
B0I = dace.symbol('B0I')
B1I = dace.symbol('B1I')
B2I = dace.symbol('B2I')


@dace.program
def matmul(A: dace.float64[S0, S2], B: dace.float64[S2, S1],
           C: dace.float64[S0, S1]):
    @dace.map(_[0:S0, 0:S1, 0:S2])
    def multiplication(i0, i1, i2):
        in_A << A[i0, i2]
        in_B << B[i2, i1]
        out >> C(1, lambda x, y: x + y)[i0, i1]

        out = in_A * in_B


if __name__ == "__main__":

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Arguments/symbols
    parser = argparse.ArgumentParser()
    parser.add_argument("S0", type=int, nargs="?", default=128)
    parser.add_argument("S1", type=int, nargs="?", default=128)
    parser.add_argument("S2", type=int, nargs="?", default=128)
    parser.add_argument("P0A", type=int, nargs="?", default=2)
    parser.add_argument("P1A", type=int, nargs="?", default=2)
    parser.add_argument("P0B", type=int, nargs="?", default=2)
    parser.add_argument("P1B", type=int, nargs="?", default=2)
    parser.add_argument("P0C", type=int, nargs="?", default=2)
    parser.add_argument("P1C", type=int, nargs="?", default=2)
    parser.add_argument("P0I", type=int, nargs="?", default=2)
    parser.add_argument("P1I", type=int, nargs="?", default=2)
    parser.add_argument("P2I", type=int, nargs="?", default=1)
    parser.add_argument("B0A", type=int, nargs="?", default=16)
    parser.add_argument("B1A", type=int, nargs="?", default=16)
    parser.add_argument("B0B", type=int, nargs="?", default=16)
    parser.add_argument("B1B", type=int, nargs="?", default=16)
    parser.add_argument("B0C", type=int, nargs="?", default=16)
    parser.add_argument("B1C", type=int, nargs="?", default=16)
    parser.add_argument("B0I", type=int, nargs="?", default=16)
    parser.add_argument("B1I", type=int, nargs="?", default=16)
    parser.add_argument("B2I", type=int, nargs="?", default=16)
    args = vars(parser.parse_args())

    # Set symbols
    S0.set(args["S0"])
    S1.set(args["S1"])
    S2.set(args["S2"])
    P0A.set(args["P0A"])
    P1A.set(args["P1A"])
    P0B.set(args["P0B"])
    P1B.set(args["P1B"])
    P0C.set(args["P0C"])
    P1C.set(args["P1C"])
    P0I.set(args["P0I"])
    P1I.set(args["P1I"])
    P2I.set(args["P2I"])
    B0A.set(args["B0A"])
    B1A.set(args["B1A"])
    B0B.set(args["B0B"])
    B1B.set(args["B1B"])
    B0C.set(args["B0C"])
    B1C.set(args["B1C"])
    B0I.set(args["B0I"])
    B1I.set(args["B1I"])
    B2I.set(args["B2I"])

    # Initialize arrays: Randomize A and B, set C to zero
    A = np.random.rand(S0.get(), S2.get()).astype(np.float64)
    B = np.random.rand(S2.get(), S1.get()).astype(np.float64)
    C = np.zeros((S0.get(), S1.get()), dtype=np.float64)

    from distr_helper import distr_exec
    args = {'A': A, 'B': B, 'C': C, 'S0': S0, 'S1': S1, 'S2': S2}
    data_distr = {'A': ([P0A, P1A], [B0A, B1A]),
                  'B': ([P0B, P1B], [B0B, B1B]),
                  'C': ([P0C, P1C], [B0C, B1C])}
    itsp_distr = ([P0I, P1I, P2I], [B0I, B1I, B2I])
    distr_exec(matmul, args,
               output=['C'], ref_func=lambda A, B: [A @ B],
               data_distr=data_distr, itsp_distr=itsp_distr)
    
    # TODO: Fix validation
    # print('Verifying reshaping of array A ... ', end='')
    # array_A = sdfg.arrays['A']
    # assert(array_A.dist_shape == (P0A, P1A))
    # # assert(array_A.shape == (ceiling(S0/(B0A*P0A)), ceiling(S2/(B1A*P1A)), B0A, B1A))
    # print('OK!')
    # print('Verifying reshaping of array B ... ', end='')
    # array_B = sdfg.arrays['B']
    # assert(array_B.dist_shape == (P0B, P1B))
    # # assert(array_B.shape == (ceiling(S2/(B0B*P0B)), ceiling(S1/(B1B*P1B)), B0B, B1B))
    # print('OK!')
    # print('Verifying reshaping of array C ... ', end='')
    # array_C = sdfg.arrays['C']
    # assert(array_C.dist_shape == (P0C, P1C))
    # # assert(array_C.shape == (ceiling(S0/(B0C*P0C)), ceiling(S1/(B1C*P1C)), B0C, B1C))
    # print('OK!')

    # state = sdfg.states()[0]

    # p0 = dace.symbol('p_i0')
    # p1 = dace.symbol('p_i1')
    # p2 = dace.symbol('p_i2')

    # l0 = dace.symbol('l_i0')
    # l1 = dace.symbol('l_i1')
    # l2 = dace.symbol('l_i2')

    # i0 = dace.symbol('i0')
    # i1 = dace.symbol('i1')
    # i2 = dace.symbol('i2')

    # # for src, src_conn, dst, dst_conn, memlet in state.edges():
    # #     if isinstance(dst, dace.nodes.Tasklet):
    # #         if dst_conn == 'in_A':
    # #             print('Verifying memlet for array A ... ', end='')
    # #             rng_A = dace.subsets.Range([
    # #                 (Mod(floor(i0/B0A), P0A), Mod(floor(i0/B0A), P0A), 1),
    # #                 (Mod(floor(i2/B1A), P1A), Mod(floor(i2/B1A), P1A), 1)
    # #             ])
    # #             assert(memlet.dist_subset == rng_A)
    # #             rng_A = dace.subsets.Range([
    # #                 (floor(i0/(B0A*P0A)), floor(i0/(B0A*P0A)), 1),
    # #                 (floor(i2/(B1A*P1A)), floor(i2/(B1A*P1A)), 1),
    # #                 (Mod(i0, B0A), Mod(i0, B0A), 1),
    # #                 (Mod(i2, B1A), Mod(i2, B1A), 1)
    # #             ])
    # #             assert(memlet.subset == rng_A)
    # #             print('OK!')
    # #         if dst_conn == 'in_B':
    # #             print('Verifying memlet for array B ... ', end='')
    # #             rng_B = dace.subsets.Range([
    # #                 (Mod(floor(i2/B0B), P0B), Mod(floor(i2/B0B), P0B), 1),
    # #                 (Mod(floor(i1/B1B), P1B), Mod(floor(i1/B1B), P1B), 1)
    # #             ])
    # #             assert(memlet.dist_subset == rng_B)
    # #             rng_B = dace.subsets.Range([
    # #                 (floor(i2/(B0B*P0B)), floor(i2/(B0B*P0B)), 1),
    # #                 (floor(i1/(B1B*P1B)), floor(i1/(B1B*P1B)), 1),
    # #                 (Mod(i2, B0B), Mod(i2, B0B), 1),
    # #                 (Mod(i1, B1B), Mod(i1, B1B), 1)
    # #             ])
    # #             assert(memlet.subset == rng_B)
    # #             print('OK!')
    # #     elif isinstance(src, dace.nodes.Tasklet):
    # #         print('Verifying memlet for array C ... ', end='')
    # #         rng_C = dace.subsets.Range([
    # #             (Mod(floor(i0/B0C), P0C), Mod(floor(i0/B0C), P0C), 1),
    # #             (Mod(floor(i1/B1C), P1C), Mod(floor(i1/B1C), P1C), 1)
    # #         ])
    # #         assert(memlet.dist_subset == rng_C)
    # #         rng_C = dace.subsets.Range([
    # #             (floor(i0/(B0C*P0C)), floor(i0/(B0C*P0C)), 1),
    # #             (floor(i1/(B1C*P1C)), floor(i1/(B1C*P1C)), 1),
    # #             (Mod(i0, B0C), Mod(i0, B0C), 1),
    # #             (Mod(i1, B1C), Mod(i1, B1C), 1)
    # #         ])
    # #         assert(memlet.subset == rng_C)
    # #         print('OK!')

    # for node in state.nodes():
    #     if isinstance(node, dace.nodes.MapEntry):
    #         params = node.map.params
    #         if params[0] == 'p_i0':
    #             print('Verifying range of distributed Map ... ', end='')
    #             rng = dace.subsets.Range([
    #                 (0, P0I - 1, 1),
    #                 (0, P1I - 1, 1),
    #                 (0, P2I - 1, 1)
    #             ])
    #             assert(node.map.range == rng)
    #             print('OK!')
    #         elif params[0] == 'l_i0':
    #             print('Verifying range of process-local coordinates Map ... ', end='')
    #             rng = dace.subsets.Range([
    #                 (0, ceiling(S0/(B0I*P0I)) - 1, 1),
    #                 (0, ceiling(S1/(B1I*P1I)) - 1, 1),
    #                 (0, ceiling(S2/(B2I*P2I)) - 1, 1)
    #             ])
    #             # assert(node.map.range == rng)
    #             print('OK!')
    #         elif params[0] == 'i0':
    #             print('Verifying range of offset coordinates Map ... ', end='')
    #             rng = dace.subsets.Range([
    #                 ((l0*P0I+p0)*B0I, (l0*P0I+p0+1)*B0I - 1, 1),
    #                 ((l1*P1I+p1)*B1I, (l1*P1I+p1+1)*B1I - 1, 1),
    #                 ((l2*P2I+p2)*B2I, (l2*P2I+p2+1)*B2I - 1, 1)
    #             ])
    #             # assert(node.map.range == rng)
    #             print('OK!')
