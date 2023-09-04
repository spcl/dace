# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the work depth analysis. """
import dace as dc
from dace.sdfg.work_depth_analysis.work_depth import analyze_sdfg, get_tasklet_work_depth, parse_assumptions
from dace.sdfg.work_depth_analysis.helpers import get_uuid
from dace.sdfg.work_depth_analysis.assumptions import ContradictingAssumptions
import sympy as sp

from dace.transformation.interstate import NestSDFG
from dace.transformation.dataflow import MapExpansion

from pytest import raises

# TODO: add tests for library nodes (e.g. reduce, matMul)
# TODO: add tests for average parallelism

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
        if x[9] > 40:
            z[:] = x + y  # N work, 1 depth
        z[:] += 2 * x  # 2*N work, 2 depth     --> total outer if: 3*N work, 3 depth
    else:
        if y[9] > 30:
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

@dc.program
def sequntial_ifs(x: dc.float64[N + 1], y: dc.float64[M + 1]): # --> cannot assume N, M to be positive
    if x[0] > 5:
        x[:] += 1              # N+1 work, 1 depth
    else:
        for i in range(M):  # M work, M depth
            y[i+1] += y[i]
    if M > N:
        y[:N+1] += x[:]     # N+1 work, 1 depth
    else:
        x[:M+1] += y[:]     # M+1 work, 1 depth
    # -->   Work:  Max(N+1, M) + Max(N+1, M+1)
    #       Depth: Max(1, M) + 1


#(sdfg, (expected_work, expected_depth))
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
    (unbounded_while_do, (sp.Symbol('num_execs_0_2') * N, sp.Symbol('num_execs_0_2'))),
    # We get this Max(1, num_execs), since it is a do-while loop, but the num_execs symbol does not capture this.
    (unbounded_do_while, (sp.Max(1, sp.Symbol('num_execs_0_1')) * N,
                          sp.Max(1, sp.Symbol('num_execs_0_1')))),
    (unbounded_nonnegify, (2 * sp.Symbol('num_execs_0_7') * N,
                           2 * sp.Symbol('num_execs_0_7'))),
    (continue_for_loop, (sp.Symbol('num_execs_0_6') * N, sp.Symbol('num_execs_0_6'))),
    (break_for_loop, (N**2, N)),
    (break_while_loop, (sp.Symbol('num_execs_0_5') * N, sp.Symbol('num_execs_0_5'))),
    (sequntial_ifs, (sp.Max(N+1, M) + sp.Max(N+1, M+1), sp.Max(1, M) + 1))
]


def test_work_depth():
    for test, correct in tests_cases:
        w_d_map = {}
        sdfg = test.to_sdfg()
        if 'nested_sdfg' in test.name:
            sdfg.apply_transformations(NestSDFG)
        if 'nested_maps' in test.name:
            sdfg.apply_transformations(MapExpansion)
        analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth, [], False)
        res = w_d_map[get_uuid(sdfg)]
        # substitue each symbol without assumptions.
        # We do this since sp.Symbol('N') == Sp.Symbol('N', positive=True) --> False.
        reps = {s: sp.Symbol(s.name) for s in (res[0].free_symbols | res[1].free_symbols)}
        res = (res[0].subs(reps), res[1].subs(reps))
        reps = {s: sp.Symbol(s.name) for s in (sp.sympify(correct[0]).free_symbols | sp.sympify(correct[1]).free_symbols)}
        correct = (sp.sympify(correct[0]).subs(reps), sp.sympify(correct[1]).subs(reps))
        # check result
        assert correct == res


x, y, z, a = sp.symbols('x y z a')

# (expr, assumptions, result)
assumptions_tests=[
    (sp.Max(x, y), ['x>y'], x),
    (sp.Max(x, y, z), ['x>y'], sp.Max(x, z)),
    (sp.Max(x, y), ['x==y'], y),
    (sp.Max(x, 11) + sp.Max(x, 3), ['x<11'], 11 + sp.Max(x,3)),
    (sp.Max(x, 11) + sp.Max(x, 3), ['x<11', 'x>3'], 11 + x),
    (sp.Max(x, 11), ['x>5', 'x>3', 'x>11'], x),
    (sp.Max(x, 11), ['x==y', 'x>11'], y),
    (sp.Max(x, 11) + sp.Max(a, 5), ['a==b', 'b==c', 'c==x', 'a<11', 'c>7'], x + 11),
    (sp.Max(x, 11) + sp.Max(a, 5), ['a==b', 'b==c', 'c==x', 'b==7'], 18),
    (sp.Max(x, y), ['y>x', 'y==1000'], 1000),
    (sp.Max(x, y), ['y<x', 'y==1000'], x)
    # This test is not working yet and is here as an example of what can still be improved in the assumption system.
    # Further details in the TODO in the parse_assumptions method.
    # (sp.Max(M, N), ['N>0', 'N<5', 'M>5'], M)

]

# These assumptions should trigger the ContradictingAssumptions exception.
tests_for_exception = [
    ['x>10', 'x<9'],
    ['x==y', 'x>10', 'y<9'],
    ['a==b', 'b==c', 'c==d', 'd==e', 'e==f', 'x==y', 'y==z', 'z>b', 'x==5', 'd==100'],
    ['x==5', 'x<4']
]

def test_assumption_system():
    for expr, assums, res in assumptions_tests:
        equality_subs, all_subs = parse_assumptions(assums, set())
        initial_expr = expr
        expr = expr.subs(equality_subs[0])
        expr = expr.subs(equality_subs[1])
        for subs1, subs2 in all_subs:
            expr = expr.subs(subs1)
            expr = expr.subs(subs2)
        assert expr == res

    for assums in tests_for_exception:
        # check that the Exception gets raised.
        with raises(ContradictingAssumptions):
            parse_assumptions(assums, set())


if __name__ == '__main__':
    test_work_depth()
    test_assumption_system()
