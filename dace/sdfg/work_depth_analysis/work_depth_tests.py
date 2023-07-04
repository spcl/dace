import dace as dc
import numpy as np
from dace.sdfg.work_depth_analysis.work_depth_analysis import analyze_sdfg, get_tasklet_work_depth
from dace.sdfg.work_depth_analysis.helpers import get_uuid
import sympy as sp

from dace.transformation.interstate import NestSDFG
from dace.transformation.dataflow import MapExpansion





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
        z[:] = x + y            # 1000 work, 1 depth
    else:
        for i in range(100):    # 100 work, 100 depth
            sum += x[i]

@dc.program
def if_else_sym(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N], sum: dc.int64[1]):
    if x[10] > 50:
        z[:] = x + y            # N work, 1 depth
    else:
        for i in range(K):    # K work, K depth
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
            z[:] = x + y            # N work, 1 depth
        z[:] += 2 * x               # 2*N work, 2 depth         --> total outer if: 3*N work, 3 depth
    else:
        if y[9] > 50:
            for i in range(K):      
                sum += x[i]         # K work, K depth
        else:
            for j in range(M):      
                sum += x[j]         # M work, M depth
            z[:] = x + y            # N work, depth 1   --> total inner else: M+N work, M+1 depth
                                # --> total outer else: Max(K, M+N) work, Max(K, M+1) depth
                    # --> total over both branches: Max(K, M+N, 3*N) work, Max(K, M+1, 3) depth

@dc.program
def max_of_positive_symbol(x: dc.float64[N]):
    if x[0] > 0:
        for i in range(2*N):        # work 2*N^2, depth 2*N
            x += 1
    else:
        for j in range(3*N):        # work 3*N^2, depth 3*N
            x += 1
                                    # total is work 3*N^2, depth 3*N without any max



@dc.program
def multiple_array_sizes(x: dc.int64[N], y: dc.int64[N], z: dc.int64[N],
                         x2: dc.int64[M], y2: dc.int64[M], z2: dc.int64[M],
                         x3: dc.int64[K], y3: dc.int64[K], z3: dc.int64[K]):
    if x[0] > 0:
        z[:] = 2 * x + y            # work 2*N, depth 2
    elif x[1] > 0:
        z2[:] = 2 * x2 + y2         # work 2*M + 3, depth 5
        z2[0] += 3 + z[1] + z[2]
    elif x[2] > 0:
        z3[:] = 2 * x3 + y3         # work 2*K, depth 2
    elif x[3] > 0:
        z[:] = 3 * x + y + 1        # work 3*N, depth 3
                        # --> work= Max(3*N, 2*M, 2*K)      and depth = 5


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
            x += 3*x
        else:
            x += x

@dc.program
def continue_for_loop(x:dc.float64[N]):
    for i in range(N):
        if x[i] > 100:
            continue
        x += 1

@dc.program
def break_for_loop(x:dc.float64[N]):
    for i in range(N):
        if x[i] > 100:
            break
        x += 1   

@dc.program
def break_while_loop(x:dc.float64[N]):
    while x[0] > 10:
        if x[1] > 100:
            break
        x += 1   

# @dc.program
# def continue_for_loop2(x:dc.float64[N]):
#     i = 0
#     while True:
#         i += 1
#         if i % 2 == 0:
#             continue
#         x += 1
#         if x[0] > 10:
#             break


tests = [single_map,
         single_for_loop,
         if_else,
         if_else_sym,
         nested_sdfg,
         nested_maps,
         nested_for_loops,
         nested_if_else,
         max_of_positive_symbol,
         multiple_array_sizes,
         unbounded_while_do,
         unbounded_do_while,
         unbounded_nonnegify,
         continue_for_loop,
         break_for_loop,
         break_while_loop]
# tests = [single_map]
results = [(N, 1),
           (N, N),
           (1000, 100),
           (sp.Max(N, K), sp.Max(K,1)),
           (2*N, N + 1),
           (N*M, 1),
           (N*K, N*K),
           (sp.Max(K, M+N, 3*N), sp.Max(K, M+1, 3)),
           (3*N**2, 3*N),
           (sp.Max(3*N, 2*M + 3, 2*K), 5),
           (N*sp.Symbol('num_execs_0_2'), sp.Symbol('num_execs_0_2')),
           (N*sp.Symbol('num_execs_0_1'), sp.Symbol('num_execs_0_1')),
           (sp.Max(N*sp.Symbol('num_execs_0_5'), 2*N*sp.Symbol('num_execs_0_3')), sp.Max(sp.Symbol('num_execs_0_5'), 2*sp.Symbol('num_execs_0_3'))),
           (sp.Symbol('num_execs_0_2')*N, sp.Symbol('num_execs_0_2')),
           (N**2, N),
           (sp.Symbol('num_execs_0_3')*N, sp.Symbol('num_execs_0_3'))]





def test_work_depth():
    good = 0
    failed = 0
    exception = 0
    failed_tests = []
    for test, correct in zip(tests, results):
        w_d_map = {}
        sdfg = test.to_sdfg()#simplify=False)
        if 'nested_sdfg' in test.name:
            sdfg.apply_transformations(NestSDFG)
        if 'nested_maps' in test.name:
            sdfg.apply_transformations(MapExpansion)
        # sdfg.view()
        # try:
        analyze_sdfg(sdfg, w_d_map, get_tasklet_work_depth)
        res = w_d_map[get_uuid(sdfg)]

        # check result
        if correct == res:
            good += 1
        else:
            # sdfg.view()
            failed += 1
            failed_tests.append(test.name)
            print(f'Test {test.name} failed:')
            print('correct', correct)
            print('result',  res)
            print()
        # except Exception as e:
        # print(e)
        # failed += 1
        # exception += 1
        
    print(100*'-')
    print(100*'-')
    print(f'Ran {len(tests)} tests. {good} succeeded and {failed} failed '
          f'({exception} of those triggered an exception)')
    print(100*'-')
    print('failed tests:', failed_tests)
    print(100*'-')










if __name__ == '__main__':
    test_work_depth()