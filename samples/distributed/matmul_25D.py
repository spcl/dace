#!/usr/bin/env python

import argparse
import dace
from sympy import Mod, floor, ceiling, sqrt, simplify
import numpy as np
import scipy as sp

S = dace.symbol('S')
P = dace.symbol('P')
Cap = dace.symbol('Cap')

P0 = sqrt(P/Cap)
P1 = sqrt(P/Cap)
P2 = Cap

B0 = S * sqrt(Cap/P)
B1 = S * sqrt(Cap/P)
B2 = S / Cap


@dace.program
def matmul_25D(A: dace.float64[S, S], B: dace.float64[S, S],
               C: dace.float64[S, S]):
    @dace.map(_[0:S, 0:S, 0:S])
    def multiplication(i0, i1, i2):
        in_A << A[i0, i2]
        in_B << B[i2, i1]
        out >> C(1, lambda x, y: x + y)[i0, i1]

        out = in_A * in_B


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("S", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    S.set(args["S"])

    print('Matrix-Matrix Multiplication %dx%dx%d' % (S.get(), S.get(), S.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(S.get(), S.get()).astype(np.float64)
    B = np.random.rand(S.get(), S.get()).astype(np.float64)
    C = np.zeros((S.get(), S.get()), dtype=np.float64)
    C_regression = np.zeros((S.get(), S.get()), dtype=np.float64)

    from dace.transformation.dataflow import (BlockCyclicData, BlockCyclicMap)
    sdfg = matmul_25D.to_sdfg()
    sdfg.add_process_grid("A", (P0, P1))
    sdfg.add_process_grid("B", (P0, P1))
    sdfg.add_process_grid("C", (P0, P1))
    sdfg.add_process_grid("I", (P0, P1, P2))
    sdfg.apply_transformations([BlockCyclicData, BlockCyclicData,
                                BlockCyclicData, BlockCyclicMap],
                                options=[
                                    {'dataname': 'A',
                                     'gridname': 'A',
                                     'block': (B0, B1)},
                                     {'dataname': 'B',
                                     'gridname': 'B',
                                     'block': (B0, B1)},
                                     {'dataname': 'C',
                                     'gridname': 'C',
                                     'block': (B0, B1)},
                                    {'gridname': 'I',
                                     'block': (B0, B1, B2)}],
                                validate=False)
    
    print('Verifying reshaping of array A ... ', end='')
    array_A = sdfg.arrays['A']
    assert(array_A.dist_shape == (P0, P1))
    assert(array_A.shape == (ceiling(S/(B0*P0)), ceiling(S/(B1*P1)), B0, B1))
    print('OK!')
    print('Verifying reshaping of array B ... ', end='')
    array_B = sdfg.arrays['B']
    assert(array_B.dist_shape == (P0, P1))
    assert(array_B.shape == (ceiling(S/(B0*P0)), ceiling(S/(B1*P1)), B0, B1))
    print('OK!')
    print('Verifying reshaping of array C ... ', end='')
    array_C = sdfg.arrays['C']
    assert(array_C.dist_shape == (P0, P1))
    assert(array_C.shape == (ceiling(S/(B0*P0)), ceiling(S/(B1*P1)), B0, B1))
    print('OK!')

    state = sdfg.states()[0]

    p0 = dace.symbol('p_i0')
    p1 = dace.symbol('p_i1')
    p2 = dace.symbol('p_i2')

    l0 = dace.symbol('l_i0')
    l1 = dace.symbol('l_i1')
    l2 = dace.symbol('l_i2')

    i0 = dace.symbol('i0')
    i1 = dace.symbol('i1')
    i2 = dace.symbol('i2')

    for src, src_conn, dst, dst_conn, memlet in state.edges():
        if isinstance(dst, dace.nodes.Tasklet):
            if dst_conn == 'in_A':
                print('Verifying memlet for array A ... ', end='')
                rng_A = dace.subsets.Range([
                    (Mod(floor(i0/B0), P0), Mod(floor(i0/B0), P0), 1),
                    (Mod(floor(i2/B1), P1), Mod(floor(i2/B1), P1), 1)
                ])
                assert(memlet.dist_subset == rng_A)
                rng_A = dace.subsets.Range([
                    (floor(i0/(B0*P0)), floor(i0/(B0*P0)), 1),
                    (floor(i2/(B1*P1)), floor(i2/(B1*P1)), 1),
                    (Mod(i0, B0), Mod(i0, B0), 1),
                    (Mod(i2, B1), Mod(i2, B1), 1)
                ])
                assert(memlet.subset == rng_A)
                print('OK!')
            if dst_conn == 'in_B':
                print('Verifying memlet for array B ... ', end='')
                rng_B = dace.subsets.Range([
                    (Mod(floor(i2/B0), P0), Mod(floor(i2/B0), P0), 1),
                    (Mod(floor(i1/B1), P1), Mod(floor(i1/B1), P1), 1)
                ])
                assert(memlet.dist_subset == rng_B)
                rng_B = dace.subsets.Range([
                    (floor(i2/(B0*P0)), floor(i2/(B0*P0)), 1),
                    (floor(i1/(B1*P1)), floor(i1/(B1*P1)), 1),
                    (Mod(i2, B0), Mod(i2, B0), 1),
                    (Mod(i1, B1), Mod(i1, B1), 1)
                ])
                assert(memlet.subset == rng_B)
                print('OK!')
        elif isinstance(src, dace.nodes.Tasklet):
            print('Verifying memlet for array C ... ', end='')
            rng_C = dace.subsets.Range([
                (Mod(floor(i0/B0), P0), Mod(floor(i0/B0), P0), 1),
                (Mod(floor(i1/B1), P1), Mod(floor(i1/B1), P1), 1)
            ])
            assert(memlet.dist_subset == rng_C)
            rng_C = dace.subsets.Range([
                (floor(i0/(B0*P0)), floor(i0/(B0*P0)), 1),
                (floor(i1/(B1*P1)), floor(i1/(B1*P1)), 1),
                (Mod(i0, B0), Mod(i0, B0), 1),
                (Mod(i1, B1), Mod(i1, B1), 1)
            ])
            assert(memlet.subset == rng_C)
            print('OK!')

    for node in state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            params = node.map.params
            if params[0] == 'p_i0':
                print('Verifying range of distributed Map ... ', end='')
                rng = dace.subsets.Range([
                    (0, P0 - 1, 1),
                    (0, P1 - 1, 1),
                    (0, P2 - 1, 1)
                ])
                assert(node.map.range == rng)
                print('OK!')
            elif params[0] == 'l_i0':
                print('Verifying range of process-local coordinates Map ... ', end='')
                rng = dace.subsets.Range([
                    (0, ceiling(S/(B0*P0)) - 1, 1),
                    (0, ceiling(S/(B1*P1)) - 1, 1),
                    (0, ceiling(S/(B2*P2)) - 1, 1)
                ])
                assert(node.map.range == rng)
                print('OK!')
            elif params[0] == 'i0':
                print('Verifying range of offset coordinates Map ... ', end='')
                rng = dace.subsets.Range([
                    ((l0*P0+p0)*B0, (l0*P0+p0+1)*B0 - 1, 1),
                    ((l1*P1+p1)*B1, (l1*P1+p1+1)*B1 - 1, 1),
                    ((l2*P2+p2)*B2, (l2*P2+p2+1)*B2 - 1, 1)
                ])
                assert(node.map.range == rng)
                print('OK!')

    # sdfg(A=A, B=B, C=C, S0=S0, S1=S1, S2=S2,
    #      P0A=P0A, P1A=P1A, P0B=P0B, P1B=P1B, P0C=P0C, P1C=P1C,
    #      P0I=P0I, P1I=P1I, P2I=P2I,
    #      B0A=B0A, B1A=B1A, B0B=B0B, B1B=B1B, B0C=B0C, B1C=B1C,
    #      B0I=B0I, B1I=B1I, B2I=B2I)

    # C_regression = A @ B

    # diff = np.linalg.norm(C_regression - C) / np.linalg_norm(C_regression)
    # print("Difference:", diff)
    # print("==== Program end ====")
    # exit(0 if diff <= 1e-15 else 1)
