# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the work depth analysis. """
from typing import Dict, List, Tuple

import pytest
import dace
from dace import symbolic
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.performance_evaluation.work_depth import (analyze_sdfg, get_tasklet_work_depth, get_tasklet_avg_par,
                                                         parse_assumptions, count_arithmetic_ops_code, count_depth_code)
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.sdfg.performance_evaluation.assumptions import ContradictingAssumptions
import sympy as sp
import numpy as np

from dace.sdfg.utils import inline_control_flow_regions
from dace.transformation.interstate import NestSDFG
from dace.transformation.dataflow import MapExpansion

from pytest import raises

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')


@dace.program
def single_map(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    z[:] = x + y


@dace.program
def single_for_loop(x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        x[i] += y[i]


@dace.program
def if_else(x: dace.int64[1000], y: dace.int64[1000], z: dace.int64[1000], sum: dace.int64[1]):
    if x[10] > 50:
        z[:] = x + y  # 1000 work, 1 depth
    else:
        for i in range(100):  # 100 work, 100 depth
            sum += x[i]


@dace.program
def if_else_sym(x: dace.int64[N], y: dace.int64[N], z: dace.int64[N], sum: dace.int64[1]):
    if x[10] > 50:
        z[:] = x + y  # N work, 1 depth
    else:
        for i in range(K):  # K work, K depth
            sum += x[i]


@dace.program
def nested_sdfg(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    single_map(x, y, z)
    single_for_loop(x, y)


@dace.program
def nested_maps(x: dace.float64[N, M], y: dace.float64[N, M], z: dace.float64[N, M]):
    z[:, :] = x + y


@dace.program
def nested_for_loops(x: dace.float64[N], y: dace.float64[K]):
    for i in range(N):
        for j in range(K):
            x[i] += y[j]


@dace.program
def nested_if_else(x: dace.int64[N], y: dace.int64[N], z: dace.int64[N], sum: dace.int64[1]):
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


@dace.program
def max_of_positive_symbol(x: dace.float64[N]):
    if x[0] > 0:
        for i in range(2 * N):  # work 2*N^2, depth 2*N
            x += 1
    else:
        for j in range(3 * N):  # work 3*N^2, depth 3*N
            x += 1
            # total is work 3*N^2, depth 3*N without any max


@dace.program
def multiple_array_sizes(x: dace.int64[N], y: dace.int64[N], z: dace.int64[N], x2: dace.int64[M], y2: dace.int64[M],
                         z2: dace.int64[M], x3: dace.int64[K], y3: dace.int64[K], z3: dace.int64[K]):
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


@dace.program
def unbounded_while_do(x: dace.float64[N]):
    while x[0] < 100:
        x += 1


@dace.program
def unbounded_nonnegify(x: dace.float64[N]):
    while x[0] < 100:
        if x[1] < 42:
            x += 3 * x
        else:
            x += x


@dace.program
def break_for_loop(x: dace.float64[N]):
    for i in range(N):
        if x[i] > 100:
            break
        x += 1


@dace.program
def break_while_loop(x: dace.float64[N]):
    while x[0] > 10:
        if x[1] > 100:
            break
        x += 1


@dace.program
def early_return(x: dace.float64[N]):
    if x[0] > 0:
        return
    x += 1


@dace.program
def sequntial_ifs(x: dace.float64[N + 1], y: dace.float64[M + 1]):  # --> cannot assume N, M to be positive
    if x[0] > 5:
        x[:] += 1  # N+1 work, 1 depth
    else:
        for i in range(M):  # M work, M depth
            y[i + 1] += y[i]
    if M > N:
        y[:N + 1] += x[:]  # N+1 work, 1 depth
    else:
        x[:M + 1] += y[:]  # M+1 work, 1 depth
    # -->   Work:  Max(N+1, M) + Max(N+1, M+1)
    #       Depth: Max(1, M) + 1


@dace.program
def reduction_library_node(x: dace.float64[456]):
    return np.sum(x)


@dace.program
def reduction_library_node_symbolic(x: dace.float64[N]):
    return np.sum(x)


@dace.program
def gemm_library_node(x: dace.float64[456, 200], y: dace.float64[200, 111], z: dace.float64[456, 111]):
    z[:] = x @ y


@dace.program
def gemm_library_node_symbolic(x: dace.float64[M, K], y: dace.float64[K, N], z: dace.float64[M, N]):
    z[:] = x @ y


@dace.program
def loop_var_dependent_work(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    for i in range(1, N + 1):
        z[i - 1] = np.dot(x[:i], y[:i])


#(sdfg, (expected_work, expected_depth))
work_depth_test_cases: Dict[str, Tuple[DaceProgram, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]] = {
    'single_map': (single_map, (N, 1)),
    'single_for_loop': (single_for_loop, (N, N)),
    'if_else': (if_else, (1000, 100)),
    'if_else_sym': (if_else_sym, (sp.Max(K, N), sp.Max(1, K))),
    'nested_sdfg': (nested_sdfg, (2 * N, N + 1)),
    'nested_maps': (nested_maps, (M * N, 1)),
    'nested_for_loops': (nested_for_loops, (K * N, K * N)),
    'nested_if_else': (nested_if_else, (sp.Max(K, 3 * N, M + N), sp.Max(3, K, M + 1))),
    'max_of_positive_symbols': (max_of_positive_symbol, (3 * N**2, 3 * N)),
    'multiple_array_sizes': (multiple_array_sizes, (sp.Max(2 * K, 3 * N, 2 * M + 3), 5)),
    'unbounded_while_do': (unbounded_while_do, (dace.symbol('num_execs_0_0', nonnegative=True) * N,
                                                dace.symbol('num_execs_0_0', nonnegative=True))),
    # We get this Max(1, num_execs), since it is a do-while loop, but the num_execs symbol does not capture this.
    'unbounded_nonnegify': (unbounded_nonnegify, (2 * dace.symbol('num_execs_0_0', nonnegative=True) * N,
                                                  2 * dace.symbol('num_execs_0_0', nonnegative=True))),
    'sequential_ifs': (sequntial_ifs, (sp.Max(N + 1, M) + sp.Max(N + 1, M + 1), sp.Max(1, M) + 1)),
    'reduction_library_node': (reduction_library_node, (456, sp.log(456) / sp.log(2))),
    'reduction_library_node_symbolic': (reduction_library_node_symbolic, (N, sp.log(sp.Max(1, N)) / sp.log(2))),
    'gemm_library_node': (gemm_library_node, (2 * 456 * 200 * 111, sp.log(200) / sp.log(2))),
    'gemm_library_node_symbolic':
    (gemm_library_node_symbolic, (2 * M * K * N, sp.Max(1,
                                                        sp.log(sp.Max(1, K)) / sp.log(2)))),
    'loop_var_dependent_work':
    (loop_var_dependent_work, (N**2, N + sp.Sum(sp.log(dace.symbol("_p_i", nonnegative=True) + 1),
                                                (dace.symbol("_p_i", nonnegative=True), 0, N - 1)) / sp.log(2)))
}


@pytest.mark.parametrize('test_name', list(work_depth_test_cases.keys()))
def test_work_depth(test_name):
    if (dace.Config.get_bool('optimizer', 'automatic_simplification') == False
            and test_name in ['unbounded_while_do', 'unbounded_nonnegify']):
        pytest.skip('Malformed loop when not simplifying')
    test, correct = work_depth_test_cases[test_name]
    w_d_map: Dict[str, sp.Expr] = {}
    sdfg = test.to_sdfg()
    if 'nested_sdfg' in test.name:
        sdfg.apply_transformations(NestSDFG)
    if 'nested_maps' in test.name:
        sdfg.apply_transformations(MapExpansion)

    analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth, [], False)
    res = w_d_map[get_uuid(sdfg)]
    correct = (sp.sympify(correct[0]), sp.sympify(correct[1]))
    assert res[0].expand() == correct[0].expand()
    assert res[1].expand() == correct[1].expand()


#(sdfg, expected_avg_par)
tests_cases_avg_par = {
    'single_map': (single_map, N),
    'single_for_loop': (single_for_loop, 1),
    'if_else': (if_else, 1),
    'nested_sdfg': (nested_sdfg, 2 * N / (N + 1)),
    'nested_maps': (nested_maps, N * M),
    'nested_for_loops': (nested_for_loops, 1),
    'max_of_positive_symbol': (max_of_positive_symbol, N),
    'unbounded_while_do': (unbounded_while_do, N),
    'unbounded_nonnegify': (unbounded_nonnegify, N),
    'reduction_library_node': (reduction_library_node, 456 / (sp.log(456) / sp.log(2))),
    'reduction_library_node_symbolic': (reduction_library_node_symbolic, N * sp.log(2) / sp.log(sp.Max(1, N))),
    'gemm_library_node': (gemm_library_node, 2 * 456 * 200 * 111 / (sp.log(200) / sp.log(2))),
    'gemm_library_node_symbolic':
    (gemm_library_node_symbolic, 2 * K * M * N / sp.Max(1,
                                                        sp.log(sp.Max(1, K)) / sp.log(2))),
}


@pytest.mark.parametrize('test_name', list(tests_cases_avg_par.keys()))
def test_avg_par(test_name: str):
    if (dace.Config.get_bool('optimizer', 'automatic_simplification') == False
            and test_name in ['unbounded_while_do', 'unbounded_nonnegify']):
        pytest.skip('Malformed loop when not simplifying')

    test, correct = tests_cases_avg_par[test_name]
    w_d_map: Dict[str, Tuple[sp.Expr, sp.Expr]] = {}
    sdfg = test.to_sdfg()
    if 'nested_sdfg' in test_name:
        sdfg.apply_transformations(NestSDFG)
    if 'nested_maps' in test_name:
        sdfg.apply_transformations(MapExpansion)

    analyze_sdfg(sdfg, w_d_map, get_tasklet_avg_par, [], False)
    res = w_d_map[get_uuid(sdfg)]
    correct = sp.sympify(correct)
    assert res.expand() == correct.expand()


@pytest.mark.parametrize('prog', [break_for_loop, break_while_loop, early_return])
def test_work_depth_bails_on_nonlocal_exit(prog: DaceProgram):
    """ ``break`` / ``continue`` / ``return`` are not supported (non-local exits are not modeled);
    the analysis must warn and produce a zero (work, depth) result rather than a wrong one. """
    sdfg = prog.to_sdfg()
    w_d_map: Dict[str, sp.Expr] = {}
    with pytest.warns(UserWarning, match='structured control flow'):
        analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth, [], False)
    assert w_d_map[get_uuid(sdfg)] == (sp.sympify(0), sp.sympify(0))


def test_work_depth_bails_on_unstructured_control_flow():
    """ Inlined control flow (LoopRegions / ConditionalBlocks flattened to a legacy state machine)
    is not supported; the analysis must warn and produce a zero (work, depth) result. """
    sdfg = single_for_loop.to_sdfg()
    inline_control_flow_regions(sdfg)
    for sd in sdfg.all_sdfgs_recursive():
        sd.using_explicit_control_flow = False
    w_d_map: Dict[str, sp.Expr] = {}
    with pytest.warns(UserWarning, match='structured control flow'):
        analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth, [], False)
    assert w_d_map[get_uuid(sdfg)] == (sp.sympify(0), sp.sympify(0))


x, y, z, a = dace.symbol('x'), dace.symbol('y'), dace.symbol('z'), dace.symbol('a')

# (expr, assumptions, result)
assumptions_tests = [
    (sp.Max(x, y), ['x>y'], x), (sp.Max(x, y, z), ['x>y'], sp.Max(x, z)), (sp.Max(x, y), ['x==y'], y),
    (sp.Max(x, 11) + sp.Max(x, 3), ['x<11'], 11 + sp.Max(x, 3)), (sp.Max(x, 11) + sp.Max(x, 3), ['x<11',
                                                                                                 'x>3'], 11 + x),
    (sp.Max(x, 11), ['x>5', 'x>3', 'x>11'], x), (sp.Max(x, 11), ['x==y', 'x>11'], y),
    (sp.Max(x, 11) + sp.Max(a, 5), ['a==b', 'b==c', 'c==x', 'a<11', 'c>7'], x + 11),
    (sp.Max(x, 11) + sp.Max(a, 5), ['a==b', 'b==c', 'c==x', 'b==7'], 18), (sp.Max(x, y), ['y>x', 'y==1000'], 1000),
    (sp.Max(x, y), ['y<x', 'y==1000'], x)
    # This test is not working yet and is here as an example of what can still be improved in the assumption system.
    # Further details in the TODO in the parse_assumptions method.
    # (sp.Max(M, N), ['N>0', 'N<5', 'M>5'], M)
]

# These assumptions should trigger the ContradictingAssumptions exception.
tests_for_exception = [['x>10', 'x<9'], ['x==y', 'x>10', 'y<9'],
                       ['a==b', 'b==c', 'c==d', 'd==e', 'e==f', 'x==y', 'y==z', 'z>b', 'x==5', 'd==100'],
                       ['x==5', 'x<4']]


@pytest.mark.parametrize('expr,assums,res', assumptions_tests)
def test_assumption_system(expr: sp.Expr, assums: List[str], res: sp.Expr):
    equality_subs, all_subs = parse_assumptions(assums, set())
    expr = expr.subs(equality_subs[0])
    expr = expr.subs(equality_subs[1])
    for subs1, subs2 in all_subs:
        expr = expr.subs(subs1)
        expr = expr.subs(subs2)
    assert expr == res


@pytest.mark.parametrize('assumptions', tests_for_exception)
def test_assumption_system_contradictions(assumptions):
    # check that the Exception gets raised.
    with raises(ContradictingAssumptions):
        parse_assumptions(assumptions, set())


def test_depth_counter_vs_work_counter():
    """
    Test that the DepthCounter correctly computes depth (longest chain of dependent operations)
    which can differ from work (total number of operations).
    Depth measures the critical path through the expression tree, while work measures
    the total number of operations.
    """
    # Test case 1: (a + b) * (c + d)
    # Work = 3 (two additions + one multiplication)
    # Depth = 2 (additions can be parallel, then multiplication)
    code1 = "(a + b) * (c + d)"
    assert count_arithmetic_ops_code(code1) == 3, "Work should be 3"
    assert count_depth_code(code1) == 2, "Depth should be 2"

    # Test case 2: a + b + c + d (left-associative: ((a + b) + c) + d)
    # Work = 3 (three additions)
    # Depth = 3 (sequential chain of additions)
    code2 = "a + b + c + d"
    assert count_arithmetic_ops_code(code2) == 3, "Work should be 3"
    assert count_depth_code(code2) == 3, "Depth should be 3"

    # Test case 3: (a + b) * (c + d) + (e + f) * (g + h)
    # Work = 7 (4 additions + 2 multiplications + 1 addition)
    # Depth = 3 (parallel adds, then parallel mults, then final add)
    code3 = "(a + b) * (c + d) + (e + f) * (g + h)"
    assert count_arithmetic_ops_code(code3) == 7, "Work should be 7"
    assert count_depth_code(code3) == 3, "Depth should be 3"

    # Test case 4: Simple single operation
    # Work = 1, Depth = 1
    code4 = "a + b"
    assert count_arithmetic_ops_code(code4) == 1, "Work should be 1"
    assert count_depth_code(code4) == 1, "Depth should be 1"

    # Test case 5: Unary operation with binary operation
    # -a + b: Work = 2, Depth = 2 (unary then add, but unary on a, so depth is 1+1=2)
    code5 = "-a + b"
    assert count_arithmetic_ops_code(code5) == 2, "Work should be 2"
    assert count_depth_code(code5) == 2, "Depth should be 2"

    # Test case 6: Function call with independent arguments
    # max(a + b, c + d): Work = 2 (two adds, max is 0), Depth = 1 (parallel adds, max is 0)
    code6 = "max(a + b, c + d)"
    assert count_arithmetic_ops_code(code6) == 2, "Work should be 2"
    assert count_depth_code(code6) == 1, "Depth should be 1"

    # Test case 7: Nested function calls
    # sqrt(a + b): Work = 2 (add + sqrt), Depth = 2 (add then sqrt)
    code7 = "sqrt(a + b)"
    assert count_arithmetic_ops_code(code7) == 2, "Work should be 2"
    assert count_depth_code(code7) == 2, "Depth should be 2"

    # Test case 8: AugAssign with parallel sub-expressions
    # x += a * b + c * d: Work = 4 (2 mults + 1 add + 1 augassign), Depth = 3
    code8 = "x += a * b + c * d"
    assert count_arithmetic_ops_code(code8) == 4, "Work should be 4"
    assert count_depth_code(code8) == 3, "Depth should be 3"

    # Test case 9: Multiple independent statements (no data dependency)
    # a = x + y; b = z + w  --> Work = 2, Depth = 1 (parallel, no dependency)
    code9 = """
a = x + y
b = z + w
"""
    assert count_arithmetic_ops_code(code9) == 2, "Work should be 2"
    assert count_depth_code(code9) == 1, "Depth should be 1 (independent statements)"

    # Test case 10: Multiple statements WITH data dependency
    # a = x + y; b = a + z  --> Work = 2, Depth = 2 (b depends on a)
    code10 = """
a = x + y
b = a + z
"""
    assert count_arithmetic_ops_code(code10) == 2, "Work should be 2"
    assert count_depth_code(code10) == 2, "Depth should be 2 (b depends on a)"

    # Test case 11: Chain of 3 dependent statements
    # a = x + y; b = a * 2; c = b + z  --> Work = 3, Depth = 3
    code11 = """
a = x + y
b = a * 2
c = b + z
"""
    assert count_arithmetic_ops_code(code11) == 3, "Work should be 3"
    assert count_depth_code(code11) == 3, "Depth should be 3 (chain: a -> b -> c)"

    # Test case 12: Diamond dependency pattern
    # a = x + y; b = a + 1; c = a + 2; d = b + c  --> Work = 4, Depth = 3
    # a has depth 1, b and c both have depth 2 (depend on a), d has depth 3
    code12 = """
a = x + y
b = a + 1
c = a + 2
d = b + c
"""
    assert count_arithmetic_ops_code(code12) == 4, "Work should be 4"
    assert count_depth_code(code12) == 3, "Depth should be 3 (diamond: a -> b,c -> d)"

    # Test case 13: AugAssign chain
    # x += 1; x += 2; x += 3  --> Work = 3, Depth = 3 (each depends on previous x)
    code13 = """
x += 1
x += 2
x += 3
"""
    assert count_arithmetic_ops_code(code13) == 3, "Work should be 3"
    assert count_depth_code(code13) == 3, "Depth should be 3 (augassign chain)"

    # Test case 14: Single complex statement with tree structure
    # result = (a+b)*(c+d) + (e+f)*(g+h) + (i+j)*(k+l)
    # The AST is left-associative: ((prod1 + prod2) + prod3)
    # prod1 has depth 2, prod2 has depth 2, prod3 has depth 2
    # (prod1 + prod2) = max(2,2) + 1 = 3
    # ((prod1 + prod2) + prod3) = max(3, 2) + 1 = 4
    code14 = "(a+b)*(c+d) + (e+f)*(g+h) + (i+j)*(k+l)"
    assert count_arithmetic_ops_code(code14) == 11, "Work should be 11"
    assert count_depth_code(code14) == 4, "Depth should be 4"


if __name__ == '__main__':
    for test_name in work_depth_test_cases.keys():
        test_work_depth(test_name)

    for test_name in tests_cases_avg_par.keys():
        test_avg_par(test_name)

    for expr, assums, res in assumptions_tests:
        test_assumption_system(expr, assums, res)

    for assumptions in tests_for_exception:
        test_assumption_system_contradictions(assumptions)

    test_depth_counter_vs_work_counter()
