# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the operational intensity analysis. """
import dace as dc
import sympy as sp
import numpy as np
from dace.sdfg.performance_evaluation.operational_intensity import analyze_sdfg_op_in
from dace.sdfg.performance_evaluation.helpers import get_uuid

from math import isclose

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
    result[0] = np.sum(z)
    # tests whether the access to z from the nested SDFG correspond with the prior accesses
    # to z outside of the nested SDFG.


@dc.program
def mmm(x: dc.float64[N, N], y: dc.float64[N, N], z: dc.float64[N, N]):
    for n, k, m in dc.map[0:N, 0:N, 0:N]:
        z[n, k] += x[n, m] * y[m, k]


@dc.program
def tiled_mmm(x: dc.float64[N, N], y: dc.float64[N, N], z: dc.float64[N, N]):
    for n_TILE, k_TILE, m_TILE in dc.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dc.map[n_TILE:n_TILE + TILE_SIZE, k_TILE:k_TILE + TILE_SIZE, m_TILE:m_TILE + TILE_SIZE]:
            z[n, k] += x[n, m] * y[m, k]


@dc.program
def tiled_mmm_32(x: dc.float32[N, N], y: dc.float32[N, N], z: dc.float32[N, N]):
    for n_TILE, k_TILE, m_TILE in dc.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dc.map[n_TILE:n_TILE + TILE_SIZE, k_TILE:k_TILE + TILE_SIZE, m_TILE:m_TILE + TILE_SIZE]:
            z[n, k] += x[n, m] * y[m, k]


@dc.program
def reduction_library_node(x: dc.float64[N]):
    return np.sum(x)


#(sdfg, c, l, assumptions, expected_result)
tests_cases = [
    (single_map64, 64 * 64, 64, {
        'N': 512
    }, 1 / 24),
    (single_map16, 64 * 64, 64, {
        'N': 512
    }, 1 / 6),
    # now num_elements_on_single_cache_line does not divie N anymore
    # -->513 work, 520 elements loaded --> 513 / (520*8*3)
    (single_map64, 64 * 64, 64, {
        'N': 513
    }, 513 / (3 * 8 * 520)),
    (sequential_maps, 1024, 3 * 8, {
        'N': 29
    }, 87 / (90 * 8)),
    # smaller cache --> only two arrays fit --> x loaded twice now
    (sequential_maps, 6, 3 * 8, {
        'N': 7
    }, 21 / (13 * 3 * 8)),
    (nested_reuse, 1024, 64, {
        'N': 1024
    }, 2048 / (3 * 1024 * 8 + 128)),
    (mmm, 20, 16, {
        'N': 24
    }, (2 * 24**3) / ((36 * 24**2 + 24 * 12) * 16)),
    (tiled_mmm, 20, 16, {
        'N': 24,
        'TILE_SIZE': 4
    }, (2 * 24**3) / (16 * 24 * 6**3)),
    (tiled_mmm_32, 10, 16, {
        'N': 24,
        'TILE_SIZE': 4
    }, (2 * 24**3) / (16 * 12 * 6**3)),
    (reduction_library_node, 1024, 64, {
        'N': 128
    }, 128.0 / (dc.symbol('Reduce_misses') * 64.0 + 64.0)),
]


def test_operational_intensity():
    for test, c, l, assumptions, correct in tests_cases:
        op_in_map = {}
        sdfg = test.to_sdfg()
        if test.name == 'nested_reuse':
            sdfg.expand_library_nodes()
        analyze_sdfg_op_in(sdfg, op_in_map, c * l, l, assumptions)
        res = (op_in_map[get_uuid(sdfg)])
        if test.name == 'reduction_library_node':
            # substitue each symbol without assumptions.
            # We do this since sp.Symbol('N') == Sp.Symbol('N', positive=True) --> False.
            reps = {s: sp.Symbol(s.name) for s in res.free_symbols}
            res = res.subs(reps)
            reps = {s: sp.Symbol(s.name) for s in sp.sympify(correct).free_symbols}
            correct = sp.sympify(correct).subs(reps)
            assert correct == res
        else:
            assert isclose(correct, res)


if __name__ == '__main__':
    test_operational_intensity()
