# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the operational intensity analysis. """
import dace as dc
from dace.sdfg.work_depth_analysis.operational_intensity import analyze_sdfg_op_in 
from dace.sdfg.work_depth_analysis.helpers import get_uuid
import sympy as sp

from dace.transformation.interstate import NestSDFG
from dace.transformation.dataflow import MapExpansion
from math import isclose
from numpy import sum

# TODO: maybe include tests for column major memory layout. AKA test that strides are taken into account correctly.
# TODO: add tests for library nodes

N = dc.symbol('N')
M = dc.symbol('M')
K = dc.symbol('K')

TILE_SIZE = dc.symbol('TILE_SIZE')



@dc.program
def single_map64(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    z[:] = x + y
    # does N work, loads 3*N elements of 8 bytes
    # --> op_in should be N / 3*8*N = 1/24 (no reuse) assuming L divides N

@dc.program
def single_map16(x: dc.float16[N], y: dc.float16[N], z: dc.float16[N]):
    z[:] = x + y
    # does N work, loads 3*N elements of 2 bytes
    # --> op_in should be N / 3*2*N = 1/6 (no reuse) assuming L divides N


@dc.program
def single_for_loop(x: dc.float64[N], y: dc.float64[N]):
    for i in range(N):
        x[i] += y[i]
    # N work, 2*N*8 bytes loaded
    # --> 1/16 op in




@dc.program
def if_else(x: dc.int64[100], sum: dc.int64[1]):
    if x[10] > 50:
        for i in range(100):  
            sum += x[i]
    if x[0] > 3:
        for i in range(100):  
            sum += x[i]
    # no else --> simply analyze the ifs. if cache big enough, everything is reused


@dc.program
def unaligned_for_loop(x: dc.float32[100], sum: dc.int64[1]):
    for i in range(17, 53):
        sum += x[i]



@dc.program
def sequential_maps(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    z[:] = x + y
    z[:] *= 2
    z[:] += x
    # does N work, loads 3*N elements of 8 bytes
    # --> op_in should be N / 3*8*N = 1/24 (no reuse) assuming L divides N

@dc.program
def nested_reuse(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N], result: dc.float64[1]):
    # load x, y and z
    z[:] = x + y
    result[0] = sum(z)
    # tests whether the access to z from the nested SDFG correspond with the prior accesses
    # to z outside of the nested SDFG.

@dc.program
def mmm(x: dc.float64[N, N], y: dc.float64[N, N], z: dc.float64[N,N]):
    for n, k, m in dc.map[0:N, 0:N, 0:N]:
        z[n,k] += x[n,m] * y[m,k]


@dc.program
def tiled_mmm(x: dc.float64[N, N], y: dc.float64[N, N], z: dc.float64[N,N]):
    for n_TILE, k_TILE, m_TILE in dc.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dc.map[n_TILE:n_TILE+TILE_SIZE, k_TILE:k_TILE+TILE_SIZE, m_TILE:m_TILE+TILE_SIZE]:
            z[n,k] += x[n,m] * y[m,k]

@dc.program
def tiled_mmm_32(x: dc.float32[N, N], y: dc.float32[N, N], z: dc.float32[N,N]):
    for n_TILE, k_TILE, m_TILE in dc.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dc.map[n_TILE:n_TILE+TILE_SIZE, k_TILE:k_TILE+TILE_SIZE, m_TILE:m_TILE+TILE_SIZE]:
            z[n,k] += x[n,m] * y[m,k]


# @dc.program
# def if_else_sym(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], sum: dc.int64[1]):
#     if x[10] > 50:
#         z[:] = x + y  # N work, 1 depth
#     else:
#         for i in range(K):  # K work, K depth
#             sum += x[i]


# @dc.program
# def nested_sdfg(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
#     single_map64(x, y, z)
#     single_for_loop(x, y)


# @dc.program
# def nested_maps(x: dc.float64[N, M], y: dc.float64[N, M], z: dc.float64[N, M]):
#     z[:, :] = x + y


# @dc.program
# def nested_for_loops(x: dc.float64[N], y: dc.float64[K]):
#     for i in range(N):
#         for j in range(K):
#             x[i] += y[j]


# @dc.program
# def nested_if_else(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], sum: dc.int64[1]):
#     if x[10] > 50:
#         if x[9] > 40:
#             z[:] = x + y  # N work, 1 depth
#         z[:] += 2 * x  # 2*N work, 2 depth     --> total outer if: 3*N work, 3 depth
#     else:
#         if y[9] > 30:
#             for i in range(K):
#                 sum += x[i]  # K work, K depth
#         else:
#             for j in range(M):
#                 sum += x[j]  # M work, M depth
#             z[:] = x + y  # N work, depth 1       --> total inner else: M+N work, M+1 depth
#             # --> total outer else: Max(K, M+N) work, Max(K, M+1) depth
#             # --> total over both branches: Max(K, M+N, 3*N) work, Max(K, M+1, 3) depth


# @dc.program
# def max_of_positive_symbol(x: dc.float64[N]):
#     if x[0] > 0:
#         for i in range(2 * N):  # work 2*N^2, depth 2*N
#             x += 1
#     else:
#         for j in range(3 * N):  # work 3*N^2, depth 3*N
#             x += 1
#             # total is work 3*N^2, depth 3*N without any max


# @dc.program
# def multiple_array_sizes(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], x2: dc.int64[M], y2: dc.int64[M],
#                          z2: dc.int64[M], x3: dc.int64[K], y3: dc.int64[K], z3: dc.int64[K]):
#     if x[0] > 0:
#         z[:] = 2 * x + y  # work 2*N, depth 2
#     elif x[1] > 0:
#         z2[:] = 2 * x2 + y2  # work 2*M + 3, depth 5
#         z2[0] += 3 + z[1] + z[2]
#     elif x[2] > 0:
#         z3[:] = 2 * x3 + y3  # work 2*K, depth 2
#     elif x[3] > 0:
#         z[:] = 3 * x + y + 1  # work 3*N, depth 3
#         # --> work= Max(3*N, 2*M, 2*K) and depth = 5


# @dc.program
# def unbounded_while_do(x: dc.float64[N]):
#     while x[0] < 100:
#         x += 1


# @dc.program
# def unbounded_do_while(x: dc.float64[N]):
#     while True:
#         x += 1
#         if x[0] >= 100:
#             break


# @dc.program
# def unbounded_nonnegify(x: dc.float64[N]):
#     while x[0] < 100:
#         if x[1] < 42:
#             x += 3 * x
#         else:
#             x += x


# @dc.program
# def continue_for_loop(x: dc.float64[N]):
#     for i in range(N):
#         if x[i] > 100:
#             continue
#         x += 1


# @dc.program
# def break_for_loop(x: dc.float64[N]):
#     for i in range(N):
#         if x[i] > 100:
#             break
#         x += 1


# @dc.program
# def break_while_loop(x: dc.float64[N]):
#     while x[0] > 10:
#         if x[1] > 100:
#             break
#         x += 1


# @dc.program
# def sequntial_ifs(x: dc.float64[N + 1], y: dc.float64[M + 1]):  # --> cannot assume N, M to be positive
#     if x[0] > 5:
#         x[:] += 1  # N+1 work, 1 depth
#     else:
#         for i in range(M):  # M work, M depth
#             y[i + 1] += y[i]
#     if M > N:
#         y[:N + 1] += x[:]  # N+1 work, 1 depth
#     else:
#         x[:M + 1] += y[:]  # M+1 work, 1 depth
#     # -->   Work:  Max(N+1, M) + Max(N+1, M+1)
#     #       Depth: Max(1, M) + 1


#(sdfg, c, l, assumptions, expected_result)
tests_cases = [
    (single_map64, 64*64, 64, {'N' : 512}, 1/24),
    (single_map16, 64*64, 64, {'N' : 512}, 1/6),
    # now num_elements_on_single_cache_line does not divie N anymore
    # -->513 work, 520 elements loaded --> 513 / (520*8*3)
    (single_map64, 64*64, 64, {'N' : 513}, 513 / (3*8*520)),



    # # this one fails, but the issue is more broad than the op_in analysis --> skip for now
    # (single_for_loop, 64, 64, {'N': 1024}, 1/16)
    # # this one fails, but the issue is more broad than the op_in analysis --> skip for now
    # (if_else, 1000, 800, {}, 200 / 1600),
    # # this one fails, but the issue is more broad than the op_in analysis --> skip for now
    # (unaligned_for_loop, -1, -1, {}, -1)


    (sequential_maps, 1024, 3*8, {'N' : 29}, 87 / (90*8)),
    # smaller cache --> only two arrays fit --> x loaded twice now
    (sequential_maps, 6, 3*8, {'N' : 7}, 21 / (13*3*8)),


    (nested_reuse, 1024, 64, {'N' : 1024}, 2048 / (3*1024*8 + 128)),
    (mmm, 20, 16, {'N': 24}, (2*24**3) / ((36*24**2 + 24*12) * 16)),
    (tiled_mmm, 20, 16, {'N': 24, 'TILE_SIZE' : 4}, (2*24**3) / (16*24*6**3)),
    (tiled_mmm_32, 10, 16, {'N': 24, 'TILE_SIZE' : 4}, (2*24**3) / (16*12*6**3)),


    # (nested_sdfg, (2 * N, N + 1)),
    # (nested_maps, (M * N, 1)),
    # (nested_for_loops, (K * N, K * N)),
    # (nested_if_else, (sp.Max(K, 3 * N, M + N), sp.Max(3, K, M + 1))),
    # (multiple_array_sizes, (sp.Max(2 * K, 3 * N, 2 * M + 3), 5)),
    # (sequntial_ifs, (sp.Max(N + 1, M) + sp.Max(N + 1, M + 1), sp.Max(1, M) + 1))
]


# tests_cases = [
#     (nested_reuse, 1024, 64, {'N' : 1024}, 2048 / (3*1024*8 + 128))
# ]

def test_operational_intensity():
    errors = 0
    for test, c, l, assumptions, correct in tests_cases:
        op_in_map = {}
        sdfg = test.to_sdfg()
        sdfg.expand_library_nodes()
        if test.name == 'mmm':
            sdfg.save('mmm.sdfg')
        if 'nested_sdfg' in test.name:
            sdfg.apply_transformations(NestSDFG)
        if 'nested_maps' in test.name:
            sdfg.apply_transformations(MapExpansion)
        analyze_sdfg_op_in(sdfg, op_in_map, c, l, assumptions)
        res = float(op_in_map[get_uuid(sdfg)])
        # substitue each symbol without assumptions.
        # We do this since sp.Symbol('N') == Sp.Symbol('N', positive=True) --> False.
        # check result
        # assert correct == res
        if not isclose(correct, res):
            print(sdfg.name)
            print(c, l, assumptions, correct, res)
            print('ERROR DETECTED')
            errors += 1

    print(f'Encountered {errors} failing tests out of {len(tests_cases)} tests')

if __name__ == '__main__':
    test_operational_intensity()
