# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import math
import numpy as np
from dace import propagate_memlets_sdfg
from poly_loop_to_map import PolyLoopToMap
from polytopes_to_ranges import polytopes_to_ranges
from ranges_to_polytopes import ranges_to_polytopes


def example_0():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i in range(1, N, 2):
            for j in range(i):
                if i + j != N:
                    with dace.tasklet:
                        a_in << A[i, j]
                        b_out >> B[i, j]
                        b_out = a_in + 1
                    with dace.tasklet:
                        a_in << A[i, j]
                        b_out >> B[i, j]
                        b_out = a_in + 1

    N = np.int32(1000)
    sdfg = fun.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap,
                               options={
                                   "use_scheduler": False,
                                   "parallelize_loops": False
                               },
                               validate=True)
    sdfg.apply_strict_transformations()
    propagate_memlets_sdfg(sdfg)
    # sdfg.view()

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)

    A_exp = np.copy(A)
    B_exp = np.copy(B)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, N=N)

    for i in range(1, N, 2):
        for j in range(i):
            if i + j != N:
                B_exp[i, j] = A_exp[i, j] + 1

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_1():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for j in range(N):
            for i in range(1, N, 2):
                with dace.tasklet:
                    c_in << C[i - 1]
                    c_out >> C[i]
                    c_out = c_in
                with dace.tasklet:
                    a_in << A[j]
                    c_in << C[i]
                    b_out >> B[i]
                    b_out = a_in + c_in

    N = np.int32(10)
    sdfg = polynomial_product.to_sdfg(strict=True)
    sdfg.apply_transformations(PolyLoopToMap,
                               options={
                                   "use_scheduler": True,
                                   "parallelize_loops": True
                               },
                               validate=True)
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    B_exp = np.copy(B)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, N=N)

    for j in range(N):
        for i in range(1, N, 2):
            C_exp[i] = C_exp[i - 1]
            B_exp[i] = A[j] + C_exp[i]

    print('Difference B:', np.linalg.norm(B_exp - B))
    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C) and np.allclose(B_exp, B))
    return np.allclose(C_exp, C) and np.allclose(B_exp, B)


def example_2():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[2 * N], B: dace.float64[2 * N],
                           C: dace.float64[2 * N]):
        for i in range(0, N, 2):
            with dace.tasklet:
                a_in << A[2 * i + 1]
                b_in << B[2 * i + 1]
                c_in << C[2 * i]
                c_out >> C[2 * i + 1]
                c_out = c_in + a_in * b_in
            with dace.tasklet:
                c_in << C[2 * i + 1]
                c_out >> C[2 * i + 1]
                c_out = c_in + 1

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, 2 * N, dtype=np.float64)
    B = np.random.rand(2 * N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)

    sdfg(A=A, B=B, C=C, N=N)

    for i in range(0, N, 2):
        C_exp[2 * i + 1] = C_exp[2 * i] + A[2 * i + 1] * B[2 * i + 1]
        C_exp[2 * i + 1] = C_exp[2 * i + 1] + 1

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_3():
    N = dace.symbol('N')
    """
    parameter: N
    
        for k in range(2 * N - 1):
    S(k):   C[k] = 0                                      
        
        for m in range(1, N-1, 2):
    R(m):   C[m-1] = 2 * C[m]
            for i in range(N):
                for j in range(N):
    T(m,i,j):       C[i + j] = C[i + j] + A[i] * B[j]
    """
    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for k in range(2 * N - 1):
            with dace.tasklet:
                c_out >> C[k]
                c_out = 2
        for m in range(1, N - 1, 2):
            with dace.tasklet:
                c_ik << C[m]
                c_ok >> C[m - 1]
                c_ok = c_ik * 2
            for i in range(N):
                for j in range(N):
                    with dace.tasklet:
                        a_in << A[i]
                        b_in << B[j]
                        c_in << C[i + j]
                        c_out >> C[i + j]
                        c_out = c_in + a_in * b_in

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)

    sdfg(A=A, B=B, C=C, N=N)
    # original
    for k in range(2 * N - 1):
        C_exp[k] = 2
    for m in range(1, N - 1, 2):
        C_exp[m - 1] = C_exp[m] * 2
        for i in range(N):
            for j in range(N):
                C_exp[i + j] = C_exp[i + j] + A[i] * B[j]

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_4():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in range(0, N / 2, 1):
            with dace.tasklet:
                a_in << A[2 * i + 1]
                b_in << B[2 * i + 1]
                c_in << C[2 * i]
                c_out >> C[2 * i + 1]
                c_out = c_in + a_in * b_in
        for j in range(1, N / 2, 1):
            with dace.tasklet:
                c_in << C[2 * j + 1]
                c_out >> C[2 * j + 1]
                c_out = c_in + 1

    N = np.int32(100)
    sdfg = fun.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.add_constant('N', N)
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=N, dtype=np.float64)
    C_exp = np.copy(C)
    sdfg(A=A, B=B, C=C)

    for i in range(0, int(N / 2), 1):
        C_exp[2 * i + 1] = C_exp[2 * i] + A[2 * i + 1] * B[2 * i + 1]
    for i in range(1, int(N / 2), 1):
        C_exp[2 * i + 1] = C_exp[2 * i + 1] + 1

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_5():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for i in range(1, N - 1, 2):
            for j in range(N):
                with dace.tasklet:
                    a_in << A[i]
                    b_in << B[j]
                    c_in << C[i + j]
                    c_out >> C[i + j]
                    c_out = c_in + a_in * b_in

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    sdfg(A=A, B=B, C=C, N=N)

    for i in range(1, N - 1, 2):
        for j in range(N):
            C_exp[i + j] = C_exp[i + j] + A[i] * B[j]

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_6():
    N = dace.symbol('N')

    @dace.program
    def example(A: dace.float64[N, N]):
        for k in range(N):
            for i in range(0, N, 2):
                with dace.tasklet:
                    A_in << A[i, k]
                    A_out >> A[i, k]
                    A_out = 1 + 2 * A_in
            for j in range(1, N, 2):
                with dace.tasklet:
                    A_in << A[j, k]
                    A_out >> A[j, k]
                    A_out = 2 * A_in

    sdfg = example.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = 10
    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)
    sdfg(A=A, N=N)

    for k in range(N):
        for i in range(0, N, 2):
            A_exp[i, k] = 1 + 2 * A_exp[i, k]
        for j in range(1, N, 2):
            A_exp[j, k] = 2 * A_exp[j, k]

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_loop_skewing():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N, N]):
        """
        for(int i = 1; i < n; i++)
            for(int j = 1; j < n-1; j++)
        T:      A(i, j) = A(i-1, j+1) + 1
        """
        for i in range(1, N, 1):
            for j in range(1, N - 1, 1):
                with dace.tasklet:
                    a_in << A[i - 1, j + 1]
                    a_out >> A[i, j]
                    a_out = a_in + 1

    N = np.int32(1000)
    sdfg = fun.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap,
                               validate=True)
    sdfg.apply_strict_transformations()
    propagate_memlets_sdfg(sdfg)
    # sdfg.view()

    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)

    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

    for i in range(1, N, 1):
        for j in range(1, N - 1, 1):
            A_exp[i, j] = A_exp[i - 1, j + 1] + 1

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_if():
    N = dace.symbol('N')

    @dace.program
    def if_example(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[2 * N]):
        for i in range(N):
            for j in range(N):
                if (j + i > N):
                    with dace.tasklet:
                        a_in << A[i]
                        b_in << B[j]
                        c_in << C[i + j]
                        c_out >> C[i + j]
                        c_out = c_in + a_in * b_in
                elif (j + i == N):
                    with dace.tasklet:
                        a_in << A[i]
                        c_in << C[i + j]
                        c_out >> C[i + j]
                        c_out = c_in + a_in
                else:
                    with dace.tasklet:
                        b_in << B[j]
                        c_in << C[i + j]
                        c_out >> C[i + j]
                        c_out = c_in * b_in

    N = np.int32(100)
    sdfg = if_example.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, N=N)

    for i in range(N):
        for j in range(N):
            if j + i > N:
                C_exp[i + j] = C_exp[i + j] + A[i] * B[j]
            elif j + i == N:
                C_exp[i + j] = C_exp[i + j] + A[i]
            else:
                C_exp[i + j] = C_exp[i + j] * B[j]

    print('Difference:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def reverse_loop():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):
        for j in range(N - 1, 4, -1):
            for i in range(1, j, 2):
                if j % 2 == 0 and j > 10:
                    with dace.tasklet:
                        c_in << C[i - 1]
                        c_out >> C[i]
                        c_out = c_in
                with dace.tasklet:
                    a_in << A[j]
                    c_in << C[i]
                    b_out >> B[i]
                    b_out = a_in + c_in

    N = np.int32(100)
    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    B_exp = np.copy(B)

    sdfg(A=A, B=B, C=C, N=N)

    start = N - 1
    end = 5
    step = -1
    for j in range(start, end - 1, step):
        for i in range(1, j, 2):
            if j % 2 == 0 and j > 10:
                C_exp[i] = C_exp[i - 1]
            B_exp[i] = A[j] + C_exp[i]

    print('Difference B:', np.linalg.norm(B_exp - B))
    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C) and np.allclose(B_exp, B))
    return np.allclose(C_exp, C) and np.allclose(B_exp, B)


def reverse_loop2():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):
        for j in range(N - 1, -1, -1):
            with dace.tasklet:
                c_in << C[j]
                c_out >> C[j]
                c_out = c_in + 5
            for i in range(1, j, 2):
                with dace.tasklet:
                    a_in << A[j]
                    c_in << C[i]
                    b_out >> B[i]
                    b_out = a_in + c_in

    N = 10
    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    B_exp = np.copy(B)

    sdfg(A=A, B=B, C=C, N=N)

    start = N - 1
    end = 0
    step = -1
    for j in range(start, end - 1, step):
        C_exp[j] = C_exp[j] + 5
        for i in range(1, j, 2):
            B_exp[i] = A[j] + C_exp[i]

    print('Difference B:', np.linalg.norm(B_exp - B))
    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C) and np.allclose(B_exp, B))
    return np.allclose(C_exp, C) and np.allclose(B_exp, B)


def map_in_loop():
    N = dace.symbol('N')

    @dace.program(dace.float64[N, N])
    def example(A):
        for k in range(N):
            for i in dace.map[0:N]:
                with dace.tasklet:
                    A_in << A[i, k]
                    A_out >> A[i, k]
                    A_out = 1 + 2 * A_in

    sdfg = example.to_sdfg(strict=True)
    # sdfg.view()

    N = dace.int32(10)
    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)
    sdfg(A=A, N=N)

    for k in range(N):
        for i in range(N):
            A_exp[i, k] = 1 + 2 * A_exp[i, k]

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_polybench_trisolv():
    N = dace.symbol('N')
    datatype = dace.float64

    def init_array(L, x, b):
        x[:] = datatype(-999)
        for i in range(0, N, 1):
            b[i] = datatype(i)
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                L[i, j] = 2 * datatype(i + N - j + 1) / N
            for j in range(i + 1, N, 1):
                L[i, j] = datatype(0)

    @dace.program(datatype[N, N], datatype[N], datatype[N])
    def trisolv(L, x, b):
        """
        for i in range(N):
            x[i] = b[i]  # trisolv2_475(i, j)
            for j in range(i-1):
                x[i] = x[i] - L[i][j] * x[j]  # trisolv2_480(i, j)
            if i:
                x[i] = x[i] - L[i][i-1] * x[i-1]  # trisolv2_487(i)
            x[i] = x[i] / L[i][i]  # trisolv2_494(i)
        """
        for i in range(N):
            with dace.tasklet:
                x_out >> x[i]
                b_in << b[i]
                x_out = b_in
            # x[i] = b[i]
            for j in range(i):
                with dace.tasklet:
                    L_in << L[i][j]
                    xi_in << x[i]
                    xj_in << x[j]
                    xi_out >> x[i]
                    xi_out = xi_in - L_in * xj_in
                # x[i] = x[i] - L[i][j] * x[j]
            with dace.tasklet:
                L_in << L[i][i]
                x_in << x[i]
                x_out >> x[i]
                x_out = x_in / L_in
            # x[i] = x[i] / L[i][i]

    N = dace.int32(100)
    L = dace.ndarray([N, N])
    x = dace.ndarray([N])
    b = dace.ndarray([N])
    init_array(L, x, b)

    sdfg = trisolv.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    sdfg(L=L, x=x, b=b, N=N)
    csdfg = sdfg.compile()

    from scipy.linalg import solve_triangular
    x_exp = solve_triangular(L, b, lower=True)

    print('Difference:', np.linalg.norm(x_exp - x))
    print('Correct: ', np.allclose(x_exp, x))
    return np.allclose(x_exp, x)


def example_polybench_2mm():
    NI = dace.symbol('NI')
    NJ = dace.symbol('NJ')
    NK = dace.symbol('NK')
    NL = dace.symbol('NL')
    datatype = dace.float64

    def init_array(A, B, C, D, alpha, beta):
        ni = NI
        nj = NJ
        nk = NK
        nl = NL

        alpha[0] = datatype(1.5)
        beta[0] = datatype(1.2)

        for i in range(ni):
            for j in range(nk):
                A[i, j] = datatype((i * j + 1) % ni) / ni
        for i in range(nk):
            for j in range(nj):
                B[i, j] = datatype(i * (j + 1) % nj) / nj
        for i in range(nj):
            for j in range(nl):
                C[i, j] = datatype((i * (j + 3) + 1) % nl) / nl
        for i in range(ni):
            for j in range(nl):
                D[i, j] = datatype(i * (j + 2) % nk) / nk

    @dace.program
    def k2mm(A: datatype[NI, NK], B: datatype[NK, NJ], C: datatype[NJ, NL],
             D: datatype[NI, NL], alpha: datatype[1], beta: datatype[1]):
        """
        # D := alpha*A*B*C + beta*D
        for (i = 0; i < ni; i++) {
            for (j = 0; j < nj; j++) {
                tmp[i][j] = 0;
                for (k = 0; k < nk; ++k) {
                    tmp[i][j] += alpha * A[i][k] * B[k][j];
                }
            }
        }
        for (i = 0; i < ni; i++) {
            for (j = 0; j < nl; j++) {
                D[i][j] *= beta;
                for (k = 0; k < nj; ++k) {
                    D[i][j] += tmp[i][k] * C[k][j];
                }
            }
        }
        """
        tmp = dace.define_local([NI, NJ], dtype=datatype)

        for i in range(NI):
            for j in range(NJ):
                with dace.tasklet:
                    temp_out >> tmp[i, j]
                    temp_out = 0.0
                for k in range(NK):
                    with dace.tasklet:
                        temp_in << tmp[i, j]
                        temp_out >> tmp[i, j]
                        alpha_in << alpha
                        A_in << A[i, k]
                        B_in << B[k, j]
                        temp_out = temp_in + alpha_in * A_in * B_in

        for i in range(NI):
            for j in range(NL):
                with dace.tasklet:
                    D_out >> D[i, j]
                    D_in << D[i, j]
                    beta_in << beta
                    D_out = D_in * beta_in
                for k in range(NJ):
                    with dace.tasklet:
                        D_out >> D[i, j]
                        D_in << D[i, j]
                        temp_in << tmp[i, k]
                        C_in << C[k, j]
                        D_out = D_in + temp_in * C_in

    @dace.program
    def k2mm_simple(A: datatype[NI, NK], B: datatype[NK, NJ],
                    C: datatype[NJ, NL], D: datatype[NI, NL],
                    alpha: datatype[1], beta: datatype[1]):
        tmp = dace.define_local([NI, NJ], dtype=datatype)
        for i in range(NI):
            for j in range(NJ):
                tmp[i, j] = 0.0
                for k in range(NK):
                    tmp[i, j] = tmp[i, j] + alpha[0] * A[i, k] * B[k, j]
        for i in range(NI):
            for j in range(NL):
                D[i, j] = D[i, j] * beta[0]
                for k in range(NJ):
                    D[i, j] = D[i, j] + tmp[i, k] * C[k, j]

    @dace.program
    def k2mm_par(A: datatype[NI, NK], B: datatype[NK, NJ], C: datatype[NJ, NL],
                 D: datatype[NI, NL], alpha: datatype[1], beta: datatype[1]):
        tmp = dace.define_local([NI, NJ], dtype=datatype)

        for i in dace.map[0:NI]:
            for j in dace.map[0:NJ]:
                with dace.tasklet:
                    temp_out >> tmp[i, j]
                    temp_out = 0.0
                for k in range(NK):
                    with dace.tasklet:
                        temp_in << tmp[i, j]
                        temp_out >> tmp[i, j]
                        alpha_in << alpha
                        A_in << A[i, k]
                        B_in << B[k, j]
                        temp_out = temp_in + alpha_in * A_in * B_in

        for i in dace.map[0:NI]:
            for j in dace.map[0:NL]:
                with dace.tasklet:
                    D_out >> D[i, j]
                    D_in << D[i, j]
                    beta_in << beta
                    D_out = D_in * beta_in
                for k in range(NJ):
                    with dace.tasklet:
                        D_out >> D[i, j]
                        D_in << D[i, j]
                        temp_in << tmp[i, k]
                        C_in << C[k, j]
                        D_out = D_in + temp_in * C_in

    NI = dace.uint32(18)
    NJ = dace.uint32(19)
    NK = dace.uint32(21)
    NL = dace.uint32(22)

    A = dace.ndarray([NI, NK], datatype)
    B = dace.ndarray([NK, NJ], datatype)
    C = dace.ndarray([NJ, NL], datatype)
    D = dace.ndarray([NI, NL], datatype)
    alpha = dace.ndarray([1], datatype)
    beta = dace.ndarray([1], datatype)
    init_array(A, B, C, D, alpha, beta)
    D2 = np.copy(D)
    D_exp = np.copy(D)
    D_exp2 = np.copy(D)

    # sdfg_par = k2mm_par.to_sdfg(strict=True)
    # sdfg_par.view()

    sdfg = k2mm.to_sdfg(strict=True)
    # # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    csdfg = sdfg.compile()
    csdfg(A=A,
          B=B,
          C=C,
          D=D,
          alpha=alpha,
          beta=beta,
          NI=NI,
          NJ=NJ,
          NK=NK,
          NL=NL)

    # sdfg_simple = k2mm_simple.to_sdfg(strict=True)
    # sdfg_simple.view()
    # sdfg_simple.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg_simple.view()
    # sdfg_simple.apply_strict_transformations()
    # sdfg_simple.view()
    # csdfg_simple = sdfg_simple.compile()
    # csdfg_simple(A=A, B=B, C=C, D=D2, alpha=alpha, beta=beta, NI=NI, NJ=NJ,
    #              NK=NK, NL=NL)

    # 2 Matrix Multiplications (D=A.B; E=C.D)
    tmp2 = dace.ndarray([NI, NJ], datatype)
    for i in range(NI):
        for i in range(NI):
            for j in range(NJ):
                tmp2[i, j] = 0.0
                for k in range(NK):
                    tmp2[i, j] = tmp2[i, j] + alpha[0] * A[i, k] * B[k, j]
    for i in range(NI):
        for j in range(NL):
            D_exp[i, j] = D_exp[i, j] * beta[0]
            for k in range(NJ):
                D_exp[i, j] = D_exp[i, j] + tmp2[i, k] * C[k, j]

    D_exp2 = alpha * A.dot(B).dot(C) + beta * D_exp2
    assert np.allclose(D_exp, D_exp2)

    print('Difference:', np.linalg.norm(D_exp - D))
    print('Correct: ', np.allclose(D_exp, D))
    return np.allclose(D_exp, D)


def bug_example():
    N = dace.symbol('N')

    def init_array(L, x, b):
        x[:] = dace.float64(-999)
        for i in range(0, N, 1):
            b[i] = dace.float64(i)
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                L[i, j] = 2 * dace.float64(i + N - j + 1) / N
            for j in range(i + 1, N, 1):
                L[i, j] = dace.float64(0)

    @dace.program(dace.float64[N, N], dace.float64[N], dace.float64[N])
    def example(L, x, b):
        """
        for (j = 0; j < N; j += 1) {
            for (i = j; i < N; i += 1) {
                if (j == 0) {
                    x[i] = b[i];
                }
                if (i == j) {
                    x[j] = x[j] / L[(N * j) + j];
                } else {
                    x[i] = x[i] - (L[(N * i) + j] * x[j]);
                }
            }
        }
        """
        for j in range(N):
            for i in range(j, N, 1):
                if j == 0:
                    with dace.tasklet:
                        b_in << b[i]
                        x_out >> x[i]
                        x_out = b_in
                if i == j:
                    with dace.tasklet:
                        x_in << x[j]
                        L_in << L[j, j]
                        x_out >> x[j]
                        x_out = x_in / L_in
                else:
                    with dace.tasklet:
                        xi_in << x[i]
                        xj_in << x[j]
                        L_in << L[i, j]
                        x_out >> x[i]
                        x_out = xi_in - L_in * xj_in

    sdfg = example.to_sdfg(strict=False)
    sdfg.apply_strict_transformations()

    N = dace.int32(100)
    L = dace.ndarray([N, N])
    x = dace.ndarray([N])
    b = dace.ndarray([N])
    init_array(L, x, b)
    x_exp = np.copy(x)
    sdfg(L=L, x=x, b=b, N=N)

    for j in range(N):
        for i in range(j, N, 1):
            if j == 0:
                x_exp[i] = b[i]
            if i == j:
                x_exp[j] = x_exp[j] / L[j, j]
            else:
                x_exp[i] = x_exp[i] - L[i, j] * x_exp[j]

    print('Difference:', np.linalg.norm(x_exp - x))
    print('Correct: ', np.allclose(x_exp, x))
    return np.allclose(x_exp, x)


def example_polybench_lu():
    N = dace.symbol('N')
    datatype = dace.float64

    def init_array(A):
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                # Python does modulo, while C does remainder ...
                A[i, j] = datatype(-(j % N)) / N + 1
            for j in range(i + 1, N, 1):
                A[i, j] = datatype(0)
            A[i, i] = datatype(1)
        A[:] = np.dot(A, np.transpose(A))

    @dace.program
    def lu(A: datatype[N, N]):
        """
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] -= A[i, k] * A[k, j]
                A[i, j] /= A[j, j]
            for j in range(i, N, 1):
                for k in range(i):
                    A[i, j] -= A[i, k] * A[k, j]
        """
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    with dace.tasklet:
                        Aij << A[i, j]
                        Aik << A[i, k]
                        Akj << A[k, j]
                        A_out >> A[i, j]
                        A_out = Aij - Aik * Akj
                    # A[i, j] = A[i, j] - A[i, k] * A[k, j]
                with dace.tasklet:
                    Aij << A[i, j]
                    Ajj << A[j, j]
                    A_out >> A[i, j]
                    A_out = Aij / Ajj
                # A[i, j] = A[i, j] / A[j, j]
            for j in range(i, N, 1):
                for k in range(i):
                    with dace.tasklet:
                        Aij << A[i, j]
                        Aik << A[i, k]
                        Akj << A[k, j]
                        A_out >> A[i, j]
                        A_out = Aij - Aik * Akj
                    # A[i, j] = A[i, j] - A[i, k] * A[k, j]

    N = dace.int32(10)
    A = dace.ndarray([N, N], datatype)
    init_array(A)
    A_exp = np.copy(A)
    A_exp2 = np.copy(A)

    sdfg = lu.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

    for i in range(N):
        for j in range(i):
            for k in range(j):
                A_exp[i, j] -= A_exp[i, k] * A_exp[k, j]
            A_exp[i, j] /= A_exp[j, j]
        for j in range(i, N, 1):
            for k in range(i):
                A_exp[i, j] -= A_exp[i, k] * A_exp[k, j]

    from scipy.linalg import decomp_lu
    p, l, u = decomp_lu.lu(A_exp2)
    A_exp2 = l + u - p
    assert np.allclose(A_exp, A_exp2)

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_polybench_nussinov():
    N = dace.symbol('N')
    datatype = dace.float64

    def init_array(seq, table):
        for i in range(0, N):
            seq[i] = datatype((i + 1) % 4)
        table[:] = datatype(0)

    @dace.program
    def nussinov(seq: datatype[N], table: datatype[N, N]):
        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N, 1):
                if j - 1 >= 0:
                    with dace.tasklet:
                        center << table[i, j]
                        west << table[i, j - 1]
                        out >> table[i, j]
                        out = max(center, west)
                if i + 1 < N:
                    with dace.tasklet:
                        center << table[i, j]
                        south << table[i + 1, j]
                        out >> table[i, j]
                        out = max(center, south)
                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        with dace.tasklet:
                            center << table[i, j]
                            swest << table[i + 1, j - 1]
                            seq_i << seq[i]
                            seq_j << seq[j]
                            out >> table[i, j]
                            out = max(center, swest + int(seq_i + seq_j == 3))
                    else:
                        with dace.tasklet:
                            center << table[i, j]
                            swest << table[i + 1, j - 1]
                            out >> table[i, j]
                            out = max(center, swest)
                for k in range(i + 1, j, 1):
                    with dace.tasklet:
                        center << table[i, j]
                        k_center << table[i, k]
                        k_south << table[k + 1, j]
                        out >> table[i, j]
                        out = max(center, k_center + k_south)

    N = dace.int32(10)
    seq = dace.ndarray([N], datatype)
    table = dace.ndarray([N, N], datatype)
    init_array(seq, table)
    table2 = np.copy(table)
    table_sdfg = np.copy(table)

    sdfg = nussinov.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    csdfg = sdfg.compile()
    csdfg(seq=seq, table=table_sdfg, N=N)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N, 1):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(
                        table[i, j],
                        table[i + 1, j - 1] + int(seq[i] + seq[j] == 3))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j, 1):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

    # Converted by hand (removed negative stride)
    for i in range(0, N, 1):
        for j in range(N - i, N, 1):
            if j > 0:
                table2[N - 1 - i, j] = max(table2[N - 1 - i, j],
                                           table2[N - 1 - i, j - 1])
            if i > 0:
                table2[N - 1 - i, j] = max(table2[N - 1 - i, j], table2[N - i,
                                                                        j])
            if j > 0 and N - i < N:
                if N - i < j:
                    table2[N - 1 - i, j] = max(
                        table2[N - 1 - i, j], table2[N - i, j - 1] +
                        int(seq[N - 1 - i] + seq[j] == 3))
                else:
                    table2[N - 1 - i, j] = max(table2[N - 1 - i, j],
                                               table2[N - i, j - 1])

            for k in range(N - i, j, 1):
                table2[N - 1 - i,
                       j] = max(table2[N - 1 - i, j],
                                table2[N - 1 - i, k] + table2[k + 1, j])

    print('Difference1:', np.linalg.norm(table2 - table))
    print('Difference2:', np.linalg.norm(table_sdfg - table))
    print('Correct: ', np.allclose(table_sdfg, table))
    return np.allclose(table_sdfg, table)


def example_polybench_adi():
    N = dace.symbol('N')
    tsteps = dace.symbol('tsteps')
    datatype = dace.float64

    @dace.program
    def adi(u: datatype[N, N]):
        v = dace.define_local([N, N], datatype)
        p = dace.define_local([N, N], datatype)
        q = dace.define_local([N, N], datatype)

        A = dace.define_local([1], datatype)
        B = dace.define_local([1], datatype)
        C = dace.define_local([1], datatype)
        D = dace.define_local([1], datatype)
        E = dace.define_local([1], datatype)
        F = dace.define_local([1], datatype)

        with dace.tasklet:
            out_a >> A
            out_b >> B
            out_c >> C
            out_d >> D
            out_e >> E
            out_f >> F
            out_a = -(datatype(2) * (datatype(1) / tsteps) /
                      (datatype(1) / (N * N))) / datatype(2)
            out_b = datatype(1) + (datatype(2) * (datatype(1) / tsteps) /
                                   (datatype(1) / (N * N)))
            out_c = -(datatype(2) * (datatype(1) / tsteps) /
                      (datatype(1) / (N * N))) / datatype(2)
            out_d = -(datatype(1) * (datatype(1) / tsteps) /
                      (datatype(1) / (N * N))) / datatype(2)
            out_e = datatype(1) + (datatype(1) * (datatype(1) / tsteps) /
                                   (datatype(1) / (N * N)))
            out_f = -(datatype(1) * (datatype(1) / tsteps) /
                      (datatype(1) / (N * N))) / datatype(2)

        for t in range(tsteps):
            # Column Sweep
            # for i in dace.map[1:N - 1]:
            for i in range(1, N - 1):
                with dace.tasklet:
                    v0i >> v[0, i]
                    pi0 >> p[i, 0]
                    qi0 >> q[i, 0]
                    v0i = 1.0
                    pi0 = 0.0
                    qi0 = 1.0

                for j in range(1, N - 1):
                    with dace.tasklet:
                        a << A
                        b << B
                        c << C
                        d << D
                        f << F
                        pjm1 << p[i, j - 1]
                        qjm1 << q[i, j - 1]
                        uim1 << u[j, i - 1]
                        uji << u[j, i]
                        uip1 << u[j, i + 1]
                        pij >> p[i, j]
                        qij >> q[i, j]

                        pij = -c / (a * pjm1 + b)
                        qij = (-d * uim1 + (1.0 + 2.0 * d) * uji - f * uip1 -
                               a * qjm1) / (a * pjm1 + b)
                with dace.tasklet:
                    out >> v[N - 1, i]
                    out = 1.0
                for j in range(N - 2, 0, -1):
                    with dace.tasklet:
                        pij << p[i, j]
                        vjp1 << v[j + 1, i]
                        qij << q[i, j]
                        vji >> v[j, i]
                        vji = pij * vjp1 + qij

            for i in range(1, N - 1):
                with dace.tasklet:
                    ui0 >> u[i, 0]
                    pi0 >> p[i, 0]
                    qi0 >> q[i, 0]
                    ui0 = 1.0
                    pi0 = 0.0
                    qi0 = 1.0

                for j in range(1, N - 1):
                    with dace.tasklet:
                        a << A
                        c << C
                        d << D
                        e << E
                        f << F
                        pjm1 << p[i, j - 1]
                        qjm1 << q[i, j - 1]
                        vim1 << v[i - 1, j]
                        vij << v[i, j]
                        vip1 << v[i + 1, j]
                        pij >> p[i, j]
                        qij >> q[i, j]

                        pij = -f / (d * pjm1 + e)
                        qij = (-a * vim1 + (1.0 + 2.0 * a) * vij - c * vip1 -
                               d * qjm1) / (d * pjm1 + e)
                with dace.tasklet:
                    out >> u[i, N - 1]
                    out = 1.0
                for j in range(N - 2, 0, -1):
                    with dace.tasklet:
                        pij << p[i, j]
                        ujp1 << u[i, j + 1]
                        qij << q[i, j]
                        uij >> u[i, j]
                        uij = pij * ujp1 + qij

    N = 20
    tsteps = 10

    u = np.random.rand(N, N).astype(np.float64)
    u2 = np.copy(u)

    sdfg_orig = adi.to_sdfg(strict=True)
    sdfg_orig(u=u2, N=N, tsteps=tsteps)

    sdfg = adi.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    csdfg = sdfg.compile()
    csdfg(u=u, N=N, tsteps=tsteps)

    print('Difference:', np.linalg.norm(u - u2))
    print('Correct: ', np.allclose(u, u2))
    return np.allclose(u, u2)


def example_polybench_cholesky():
    N = dace.symbol('N')
    datatype = dace.float64

    def init_array(A):
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                A[i, j] = datatype(-(j % N)) / N + 1
            for j in range(i + 1, N, 1):
                A[i, j] = datatype(0)
            A[i, i] = datatype(1)
        A[:] = np.dot(A, np.transpose(A))

    @dace.program
    def cholesky(A: datatype[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    with dace.tasklet:
                        Aij_out >> A[i][j]
                        Aij_in << A[i][j]
                        Aik_in << A[i][k]
                        Ajk_in << A[j][k]
                        Aij_out = Aij_in - Aik_in * Ajk_in
                with dace.tasklet:
                    Aij_out >> A[i][j]
                    Aij_in << A[i][j]
                    Ajj_in << A[j][j]
                    Aij_out = Aij_in / Ajj_in
            for k in range(i):
                with dace.tasklet:
                    Aii_out >> A[i][i]
                    Aii_in << A[i][i]
                    Aik_in << A[i][k]
                    Aii_out = Aii_in / Aik_in * Aik_in
            with dace.tasklet:
                Aii_out >> A[i][i]
                Aii_in << A[i][i]
                Aii_out = math.sqrt(Aii_in)

    N = 20
    A = dace.ndarray([N, N])
    init_array(A)
    A_exp = np.copy(A)

    sdfg = cholesky.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # # sdfg.view()
    # ranges_to_polytopes(sdfg)
    # sdfg.view()
    propagate_memlets_sdfg(sdfg)
    polytopes_to_ranges(sdfg)

    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()
    propagate_memlets_sdfg(sdfg)
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

    for i in range(N):
        for j in range(i):
            for k in range(j):
                A_exp[i][j] = A_exp[i][j] - A_exp[i][k] * A_exp[j][k]
            A_exp[i][j] = A_exp[i][j] / A_exp[j][j]
        for k in range(i):
            A_exp[i][i] = A_exp[i][i] / A_exp[i][k] * A_exp[i][k]
        A_exp[i][i] = math.sqrt(A_exp[i][i])

    print('Difference:', np.linalg.norm(A - A_exp))
    print('Correct: ', np.allclose(A, A_exp))
    return np.allclose(A, A_exp)


def example_polybench_seidel2d():
    N = dace.symbol('N')
    tsteps = dace.symbol('tsteps')

    datatype = dace.float64

    def init_array(A):
        for i in range(0, N, 1):
            for j in range(0, i + 1, 1):
                A[i, j] = datatype(-(j % N)) / N + 1
            for j in range(i + 1, N, 1):
                A[i, j] = datatype(0)
            A[i, i] = datatype(1)
        A[:] = np.dot(A, np.transpose(A))

    @dace.program(datatype[N, N])
    def seidel2d_orig(A):
        for t in range(tsteps):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    with dace.tasklet:
                        a1 << A[i - 1, j - 1]
                        a2 << A[i - 1, j]
                        a3 << A[i - 1, j + 1]
                        a4 << A[i, j - 1]
                        a5 << A[i, j]
                        a6 << A[i, j + 1]
                        a7 << A[i + 1, j - 1]
                        a8 << A[i + 1, j]
                        a9 << A[i + 1, j + 1]
                        out >> A[i, j]
                        out = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 +
                               a9) / datatype(9.0)

    @dace.program(datatype[N, N])
    def seidel2d(A):
        for t in range(tsteps):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    with dace.tasklet:
                        a1 << A[i - 1, j - 1]
                        a2 << A[i - 1, j]
                        a3 << A[i - 1, j + 1]
                        a4 << A[i, j - 1]
                        a5 << A[i, j]
                        a6 << A[i, j + 1]
                        a7 << A[i + 1, j - 1]
                        a8 << A[i + 1, j]
                        a9 << A[i + 1, j + 1]
                        out >> A[i, j]
                        out = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 +
                               a9) / datatype(9.0)

    N = 20
    tsteps = 10
    A = dace.ndarray([N, N])
    init_array(A)
    A_exp = np.copy(A)

    sdfg_orig = seidel2d_orig.to_sdfg(strict=False)
    sdfg_orig(A=A_exp, N=N, tsteps=tsteps)

    sdfg = seidel2d.to_sdfg(strict=False)
    # propagate_memlets_sdfg(sdfg)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap,
                               validate=True)
    # propagate_memlets_sdfg(sdfg)
    # sdfg.apply_strict_transformations()

    polytopes_to_ranges(sdfg)
    sdfg.apply_strict_transformations()
    # sdfg.view()
    # return
    #
    # from ranges_to_polytopes import RangeMapsToPolytopeMaps
    # sdfg.apply_strict_transformations()
    # sdfg.apply_transformations(RangeMapsToPolytopeMaps, validate=True)
    # # sdfg.view()
    # sdfg.apply_transformations(MapExpansion, validate=True)
    #
    # # sdfg.view()
    # return

    # from ranges_to_polytopes import RangeMapsToPolytopeMaps
    # sdfg.apply_transformations(RangeMapsToPolytopeMaps, validate=True)
    # sdfg.apply_strict_transformations()
    # propagate_memlets_sdfg(sdfg)
    # # sdfg.view()
    # sdfg.apply_transformations(MapExpansion, validate=True)
    # sdfg.apply_strict_transformations()
    # propagate_memlets_sdfg(sdfg)
    # # sdfg.view()

    csdfg = sdfg.compile()

    csdfg(A=A, N=N, tsteps=tsteps)

    print('Difference:', np.linalg.norm(A - A_exp))
    print('Correct: ', np.allclose(A, A_exp))
    return np.allclose(A, A_exp)


def example_loop_triangle():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N, N]):
        for i in range(0, N, 2):
            with dace.tasklet:
                a_in << A[i, 0]
                a_out >> A[i, 0]
                a_out = 2 * a_in + 1
            for j in range(0, i + 1, 1):
                if i + j != N:
                    with dace.tasklet:
                        a_in << A[i, j]
                        a_out >> A[i, j]
                        a_out = 2 * a_in + 1

    N = np.int32(6)
    sdfg = fun.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

    for i in range(0, N, 2):
        A_exp[i, 0] = 2 * A_exp[i, 0] + 1
        for j in range(i + 1):
            if i + j != N:
                A_exp[i, j] = 2 * A_exp[i, j] + 1

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def simple_loop_triangle():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in range(N):
                with dace.tasklet:
                    a_in << A[i, j]
                    a_out >> A[i, j]
                    a_out = 2 * a_in + 1

    N = np.int32(10)
    sdfg_orig = fun.to_sdfg(strict=True)
    csdfg_orig = sdfg_orig.compile()

    sdfg = fun.to_sdfg(strict=False)

    ranges_to_polytopes(sdfg)
    polytopes_to_ranges(sdfg)

    sdfg.apply_strict_transformations()
    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)
    csdfg_orig(A=A_exp, N=N)

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_convert_0():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i in range(1, N, 2):
            for j in range(i):
                if i + j != N:
                    with dace.tasklet:
                        a_in << A[i, j]
                        b_out >> B[i, j]
                        b_out = a_in + 1
                    with dace.tasklet:
                        a_in << A[i, j]
                        b_out >> B[i, j]
                        b_out = a_in + 1

    N = np.int32(1000)
    sdfg = fun.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap,
                               options={
                                   "use_scheduler": True,
                                   "parallelize_loops": True,
                                   "use_polytopes": False
                               },
                               validate=True)
    sdfg.apply_strict_transformations()
    propagate_memlets_sdfg(sdfg)

    sdfg.view()
    ranges_to_polytopes(sdfg)
    sdfg.view()
    polytopes_to_ranges(sdfg)
    sdfg.view()

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)

    A_exp = np.copy(A)
    B_exp = np.copy(B)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, N=N)

    for i in range(1, N, 2):
        for j in range(i):
            if i + j != N:
                B_exp[i, j] = A_exp[i, j] + 1

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def example_convert_1():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for j in range(N):
            for i in range(1, N, 2):
                with dace.tasklet:
                    c_in << C[i - 1]
                    c_out >> C[i]
                    c_out = c_in
                with dace.tasklet:
                    a_in << A[j]
                    c_in << C[i]
                    b_out >> B[i]
                    b_out = a_in + c_in

    N = np.int32(10)
    sdfg = polynomial_product.to_sdfg(strict=True)
    sdfg.apply_transformations(PolyLoopToMap,
                               options={
                                   "use_scheduler": True,
                                   "parallelize_loops": True
                               },
                               validate=True)
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    B_exp = np.copy(B)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, N=N)

    for j in range(N):
        for i in range(1, N, 2):
            C_exp[i] = C_exp[i - 1]
            B_exp[i] = A[j] + C_exp[i]

    print('Difference B:', np.linalg.norm(B_exp - B))
    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C) and np.allclose(B_exp, B))
    return np.allclose(C_exp, C) and np.allclose(B_exp, B)


def example_convert_2():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[2 * N], B: dace.float64[2 * N],
                           C: dace.float64[2 * N]):
        for i in range(0, N, 2):
            with dace.tasklet:
                a_in << A[2 * i + 1]
                b_in << B[2 * i + 1]
                c_in << C[2 * i]
                c_out >> C[2 * i + 1]
                c_out = c_in + a_in * b_in
            with dace.tasklet:
                c_in << C[2 * i + 1]
                c_out >> C[2 * i + 1]
                c_out = c_in + 1

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, 2 * N, dtype=np.float64)
    B = np.random.rand(2 * N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)

    sdfg(A=A, B=B, C=C, N=N)

    for i in range(0, N, 2):
        C_exp[2 * i + 1] = C_exp[2 * i] + A[2 * i + 1] * B[2 * i + 1]
        C_exp[2 * i + 1] = C_exp[2 * i + 1] + 1

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_convert_3():
    N = dace.symbol('N')
    """
    parameter: N

        for k in range(2 * N - 1):
    S(k):   C[k] = 0                                      

        for m in range(1, N-1, 2):
    R(m):   C[m-1] = 2 * C[m]
            for i in range(N):
                for j in range(N):
    T(m,i,j):       C[i + j] = C[i + j] + A[i] * B[j]
    """

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for k in range(2 * N - 1):
            with dace.tasklet:
                c_out >> C[k]
                c_out = 2
        for m in range(1, N - 1, 2):
            with dace.tasklet:
                c_ik << C[m]
                c_ok >> C[m - 1]
                c_ok = c_ik * 2
            for i in range(N):
                for j in range(N):
                    with dace.tasklet:
                        a_in << A[i]
                        b_in << B[j]
                        c_in << C[i + j]
                        c_out >> C[i + j]
                        c_out = c_in + a_in * b_in

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)

    sdfg(A=A, B=B, C=C, N=N)
    # original
    for k in range(2 * N - 1):
        C_exp[k] = 2
    for m in range(1, N - 1, 2):
        C_exp[m - 1] = C_exp[m] * 2
        for i in range(N):
            for j in range(N):
                C_exp[i + j] = C_exp[i + j] + A[i] * B[j]

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_convert_4():
    N = dace.symbol('N')

    @dace.program
    def fun(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in range(0, N / 2, 1):
            with dace.tasklet:
                a_in << A[2 * i + 1]
                b_in << B[2 * i + 1]
                c_in << C[2 * i]
                c_out >> C[2 * i + 1]
                c_out = c_in + a_in * b_in
        for j in range(1, N / 2, 1):
            with dace.tasklet:
                c_in << C[2 * j + 1]
                c_out >> C[2 * j + 1]
                c_out = c_in + 1

    N = np.int32(100)
    sdfg = fun.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.add_constant('N', N)
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=N, dtype=np.float64)
    C_exp = np.copy(C)
    sdfg(A=A, B=B, C=C)

    for i in range(0, int(N / 2), 1):
        C_exp[2 * i + 1] = C_exp[2 * i] + A[2 * i + 1] * B[2 * i + 1]
    for i in range(1, int(N / 2), 1):
        C_exp[2 * i + 1] = C_exp[2 * i + 1] + 1

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_convert_5():
    N = dace.symbol('N')

    @dace.program
    def polynomial_product(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[2 * N]):

        for i in range(1, N - 1, 2):
            for j in range(N):
                with dace.tasklet:
                    a_in << A[i]
                    b_in << B[j]
                    c_in << C[i + j]
                    c_out >> C[i + j]
                    c_out = c_in + a_in * b_in

    sdfg = polynomial_product.to_sdfg(strict=True)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = np.int32(20)
    A = np.arange(0, N, dtype=np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(shape=2 * N, dtype=np.float64)
    C_exp = np.copy(C)
    sdfg(A=A, B=B, C=C, N=N)

    for i in range(1, N - 1, 2):
        for j in range(N):
            C_exp[i + j] = C_exp[i + j] + A[i] * B[j]

    print('Difference C:', np.linalg.norm(C_exp - C))
    print('Correct: ', np.allclose(C_exp, C))
    return np.allclose(C_exp, C)


def example_convert_6():
    N = dace.symbol('N')

    @dace.program
    def example(A: dace.float64[N, N]):
        for k in range(N):
            for i in range(0, N, 2):
                with dace.tasklet:
                    A_in << A[i, k]
                    A_out >> A[i, k]
                    A_out = 1 + 2 * A_in
            for j in range(1, N, 2):
                with dace.tasklet:
                    A_in << A[j, k]
                    A_out >> A[j, k]
                    A_out = 2 * A_in

    sdfg = example.to_sdfg(strict=False)
    # sdfg.view()
    sdfg.apply_transformations(PolyLoopToMap, validate=True)
    # sdfg.view()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    N = 10
    A = np.random.rand(N, N).astype(np.float64)
    A_exp = np.copy(A)
    sdfg(A=A, N=N)

    for k in range(N):
        for i in range(0, N, 2):
            A_exp[i, k] = 1 + 2 * A_exp[i, k]
        for j in range(1, N, 2):
            A_exp[j, k] = 2 * A_exp[j, k]

    print('Difference:', np.linalg.norm(A_exp - A))
    print('Correct: ', np.allclose(A_exp, A))
    return np.allclose(A_exp, A)


def run_test():
    assert example_0()
    assert example_1()
    assert example_2()
    assert example_3()
    assert example_4()
    assert example_5()
    assert example_6()
    assert map_in_loop()
    assert reverse_loop()
    assert reverse_loop2()
    assert example_if()
    assert example_loop_skewing()
    assert example_polybench_trisolv()
    assert example_polybench_nussinov()
    assert example_polybench_2mm()
    assert example_polybench_lu()
    assert example_polybench_adi()
    assert example_polybench_cholesky()
    assert example_polybench_seidel2d()


if __name__ == '__main__':
    example_convert_0()
    # example_convert_1()
    # example_convert_2()
    # example_convert_3()
    # example_convert_4()
    # example_convert_5()
    # example_convert_6()

    # example_0()
    # run_test()
