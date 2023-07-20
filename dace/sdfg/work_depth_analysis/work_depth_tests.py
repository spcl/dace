# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the work depth analysis. """
import dace as dc
from dace.sdfg.work_depth_analysis.work_depth import analyze_sdfg, get_tasklet_work_depth
from dace.sdfg.work_depth_analysis.helpers import get_uuid
import sympy as sp

from dace.transformation.interstate import NestSDFG
from dace.transformation.dataflow import MapExpansion

# TODO: add tests for library nodes (e.g. reduce, matMul)

N = dc.symbol('N')
M = dc.symbol('M')
K = dc.symbol('K')


@dc.program
def single_map(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    z[:] = x + y


@dc.program
def single_for_loop(x: dc.float64[N], y: dc.float64[N]):
    for i in range(N):
        x[i] += y[i]


@dc.program
def if_else(x: dc.int64[1000], y: dc.int64[1000], z: dc.int64[1000], sum: dc.int64[1]):
    if x[10] > 50:
        z[:] = x + y  # 1000 work, 1 depth
    else:
        for i in range(100):  # 100 work, 100 depth
            sum += x[i]


@dc.program
def if_else_sym(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], sum: dc.int64[1]):
    if x[10] > 50:
        z[:] = x + y  # N work, 1 depth
    else:
        for i in range(K):  # K work, K depth
            sum += x[i]


@dc.program
def nested_sdfg(x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    single_map(x, y, z)
    single_for_loop(x, y)


@dc.program
def nested_maps(x: dc.float64[N, M], y: dc.float64[N, M], z: dc.float64[N, M]):
    z[:, :] = x + y


@dc.program
def nested_for_loops(x: dc.float64[N], y: dc.float64[K]):
    for i in range(N):
        for j in range(K):
            x[i] += y[j]


@dc.program
def nested_if_else(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], sum: dc.int64[1]):
    if x[10] > 50:
        if x[9] > 50:
            z[:] = x + y  # N work, 1 depth
        z[:] += 2 * x  # 2*N work, 2 depth     --> total outer if: 3*N work, 3 depth
    else:
        if y[9] > 50:
            for i in range(K):
                sum += x[i]  # K work, K depth
        else:
            for j in range(M):
                sum += x[j]  # M work, M depth
            z[:] = x + y  # N work, depth 1       --> total inner else: M+N work, M+1 depth
            # --> total outer else: Max(K, M+N) work, Max(K, M+1) depth
            # --> total over both branches: Max(K, M+N, 3*N) work, Max(K, M+1, 3) depth


@dc.program
def max_of_positive_symbol(x: dc.float64[N]):
    if x[0] > 0:
        for i in range(2 * N):  # work 2*N^2, depth 2*N
            x += 1
    else:
        for j in range(3 * N):  # work 3*N^2, depth 3*N
            x += 1
            # total is work 3*N^2, depth 3*N without any max


@dc.program
def multiple_array_sizes(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], x2: dc.int64[M], y2: dc.int64[M],
                         z2: dc.int64[M], x3: dc.int64[K], y3: dc.int64[K], z3: dc.int64[K]):
    if x[0] > 0:
        z[:] = 2 * x + y  # work 2*N, depth 2
    elif x[1] > 0:
        z2[:] = 2 * x2 + y2  # work 2*M + 3, depth 5
        z2[0] += 3 + z[1] + z[2]
    elif x[2] > 0:
        z3[:] = 2 * x3 + y3  # work 2*K, depth 2
    elif x[3] > 0:
        z[:] = 3 * x + y + 1  # work 3*N, depth 3
        # --> work= Max(3*N, 2*M, 2*K) and depth = 5


@dc.program
def unbounded_while_do(x: dc.float64[N]):
    while x[0] < 100:
        x += 1


@dc.program
def unbounded_do_while(x: dc.float64[N]):
    while True:
        x += 1
        if x[0] >= 100:
            break


@dc.program
def unbounded_nonnegify(x: dc.float64[N]):
    while x[0] < 100:
        if x[1] < 42:
            x += 3 * x
        else:
            x += x


@dc.program
def continue_for_loop(x: dc.float64[N]):
    for i in range(N):
        if x[i] > 100:
            continue
        x += 1


@dc.program
def break_for_loop(x: dc.float64[N]):
    for i in range(N):
        if x[i] > 100:
            break
        x += 1


@dc.program
def break_while_loop(x: dc.float64[N]):
    while x[0] > 10:
        if x[1] > 100:
            break
        x += 1


tests_cases = [
    (single_map, (N, 1)),
    (single_for_loop, (N, N)),
    (if_else, (1000, 100)),
    (if_else_sym, (sp.Max(K, N), sp.Max(1, K))),
    (nested_sdfg, (2 * N, N + 1)),
    (nested_maps, (M * N, 1)),
    (nested_for_loops, (K * N, K * N)),
    (nested_if_else, (sp.Max(K, 3 * N, M + N), sp.Max(3, K, M + 1))),
    (max_of_positive_symbol, (3 * N**2, 3 * N)),
    (multiple_array_sizes, (sp.Max(2 * K, 3 * N, 2 * M + 3), 5)),
    (unbounded_while_do, (sp.Symbol('num_execs_0_2', nonnegative=True) * N, sp.Symbol('num_execs_0_2',
                                                                                      nonnegative=True))),
    # TODO: why we get this ugly max(1, num_execs) here??
    (unbounded_do_while, (sp.Max(1, sp.Symbol('num_execs_0_1', nonnegative=True)) * N,
                          sp.Max(1, sp.Symbol('num_execs_0_1', nonnegative=True)))),
    (unbounded_nonnegify, (2 * sp.Symbol('num_execs_0_7', nonnegative=True) * N,
                           2 * sp.Symbol('num_execs_0_7', nonnegative=True))),
    (continue_for_loop, (sp.Symbol('num_execs_0_6', nonnegative=True) * N, sp.Symbol('num_execs_0_6',
                                                                                     nonnegative=True))),
    (break_for_loop, (N**2, N)),
    (break_while_loop, (sp.Symbol('num_execs_0_5', nonnegative=True) * N, sp.Symbol('num_execs_0_5', nonnegative=True)))
]


def test_work_depth():
    good = 0
    failed = 0
    exception = 0
    failed_tests = []
    for test, correct in tests_cases:
        w_d_map = {}
        sdfg = test.to_sdfg()
        if 'nested_sdfg' in test.name:
            sdfg.apply_transformations(NestSDFG)
        if 'nested_maps' in test.name:
            sdfg.apply_transformations(MapExpansion)
        try:
            analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth)
            res = w_d_map[get_uuid(sdfg)]

            # check result
            if correct == res:
                good += 1
            else:
                failed += 1
                failed_tests.append(test.name)
                print(f'Test {test.name} failed:')
                print('correct', correct)
                print('result', res)
                print()
        except Exception as e:
            print(e)
            failed += 1
            exception += 1

    print(100 * '-')
    print(100 * '-')
    print(f'Ran {len(tests_cases)} tests. {good} succeeded and {failed} failed '
          f'({exception} of those triggered an exception)')
    print(100 * '-')
    print('failed tests:', failed_tests)
    print(100 * '-')


if __name__ == '__main__':
    test_work_depth()
