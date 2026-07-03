# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import itertools
import sys
import dace
import random
import numpy as np
from dace.libraries.blas import Gemm

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')
L = dace.symbol('L')
O = dace.symbol('O')


@pytest.mark.parametrize(
    ('implementation', ),
    [('pure', ), pytest.param('MKL', marks=pytest.mark.mkl),
     pytest.param('cuBLAS', marks=pytest.mark.gpu)])
def test_gemm_no_c(implementation):

    Gemm.default_implementation = implementation

    @dace.program
    def simple_gemm(A: dace.float64[10, 15], B: dace.float64[15, 3]):
        return A @ B

    A = np.random.rand(10, 15)
    B = np.random.rand(15, 3)

    result = simple_gemm(A, B)
    assert np.allclose(result, A @ B)

    Gemm.default_implementation = None


def create_gemm_sdfg(dtype, A_shape, B_shape, C_shape, Y_shape, transA, transB, alpha, beta, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", B_shape, dtype)
    C, C_arr = sdfg.add_array("C", Y_shape, dtype)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    libnode = Gemm('_Gemm_', transA=transA, transB=transB, alpha=alpha, beta=beta)
    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b', dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None, dace.Memlet.from_array(C, C_arr))
    if beta != 0.0:
        rC = state.add_read('C')
        state.add_edge(rC, None, libnode, '_c', dace.Memlet.from_array(C, C_arr))

    return sdfg


_impls = ['pure', pytest.param('MKL', marks=pytest.mark.mkl), pytest.param('cuBLAS', marks=pytest.mark.gpu)]
_param_grid_trans = dict(
    transA=[True, False],
    transB=[True, False],
)
_param_grid_scalars = dict(
    alpha=[1.0, 0.0, random.random()],
    beta=[1.0, 0.0, random.random()],
)
_param_grid_complex = dict(
    complex=[True],
    alpha=[random.random(), complex(random.random(), random.random())],
    beta=[random.random(), complex(random.random(), random.random())],
)

_param_grid_broadcast_C = dict(
    alpha=[random.random()],
    beta=[random.random()],
    C_shape=[None, ["M", "N"], ["M", 1], ["N"], [1, "N"]],
)


def params_generator(grid):
    keys, values = zip(*grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        yield params


_test_params = []
for param_grid in [_param_grid_trans, _param_grid_scalars, _param_grid_complex, _param_grid_broadcast_C]:
    for params in params_generator(param_grid):
        print("Testing params:", params)
        _test_params.append(params)


def _do_test_gemm(implementation, params):
    M = params.get('M', 25)
    N = params.get('N', 24)
    K = params.get('K', 23)
    complex = params.get('complex', False)
    transA = params.get('transA', False)
    transB = params.get('transB', False)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    C_shape = params.get('C_shape', ["M", "N"])

    if C_shape is not None:
        replace_map = dict(M=M, N=N)
        C_shape = [s if isinstance(s, int) else replace_map[s] for s in C_shape]

    # unique name for sdfg
    C_str = "None" if C_shape is None else (str(C_shape[0]) if len(C_shape) == 1 else f"{C_shape[0]}_{C_shape[1]}")
    sdfg_name = f"{implementation}_{M}_{N}_{K}_{complex}_{transA}_{transB}_{alpha}_{beta}_{C_str}".replace(
        ".", "_dot_").replace("+", "_plus_").replace("-", "_minus_").replace("(", "").replace(")", "")

    # shape of the transposed arrays
    A_shape = trans_A_shape = [M, K]
    B_shape = trans_B_shape = [K, N]
    Y_shape = [M, N]

    # A_shape is the actual shape of A
    if transA:
        A_shape = list(reversed(trans_A_shape))

    if transB:
        B_shape = list(reversed(trans_B_shape))

    print(f'Matrix multiplication {M}x{K}x{N} (alpha={alpha}, beta={beta})')

    np_dtype = np.complex64 if complex else np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    B = np.random.rand(*B_shape).astype(np_dtype)
    if C_shape is not None:
        C = np.random.rand(*C_shape).astype(np_dtype)
    else:
        C = None
    Y = np.zeros(Y_shape, dtype=np_dtype)

    def numpy_gemm(A, B, C, transA, transB, alpha, beta):
        A_t = np.transpose(A) if transA else A
        B_t = np.transpose(B) if transB else B
        if C is not None:
            return alpha * (A_t @ B_t) + (beta * C)
        else:
            return alpha * (A_t @ B_t)

    Y_regression = numpy_gemm(A, B, C, transA, transB, alpha, beta)

    sdfg = create_gemm_sdfg(dace.complex64 if complex else dace.float32, A_shape, B_shape, C_shape, Y_shape, transA,
                            transB, alpha, beta, implementation, sdfg_name)

    if C_shape is not None:
        Y[:] = C
        sdfg(A=A, B=B, C=Y)
    else:
        sdfg(A=A, B=B, C=Y)

    diff = np.linalg.norm(Y_regression - Y) / (M * N)
    print("Difference:", diff)
    assert diff <= 1e-5


@pytest.mark.parametrize('params', _test_params)
def do_test_pure(params):
    impl = 'pure'
    _do_test_gemm(impl, params)


@pytest.mark.gpu
@pytest.mark.parametrize('params', _test_params)
def do_test_cuBLAS(params):
    impl = 'cuBLAS'
    _do_test_gemm(impl, params)


@pytest.mark.mkl
@pytest.mark.parametrize('params', _test_params)
def do_test_mkl(params):
    impl = 'MKL'
    _do_test_gemm(impl, params)


def test_gemm_symbolic():
    sdfg = dace.SDFG("gemm")
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", [M, K], dace.float64)
    B, B_arr = sdfg.add_array("B", [L, N], dace.float64)
    C, C_arr = sdfg.add_array("C", [O, N], dace.float64)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    with pytest.warns(match="may not match"):
        libnode = Gemm('_Gemm_', transA=False, transB=False, alpha=1.0, beta=0.0)
        state.add_node(libnode)

        state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
        state.add_edge(rB, None, libnode, '_b', dace.Memlet.from_array(B, B_arr))
        state.add_edge(libnode, '_c', wC, None, dace.Memlet.from_array(C, C_arr))

        sdfg.validate()


def test_gemm_symbolic_1():
    sdfg = dace.SDFG("gemm")
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", [M, K], dace.float64)
    B, B_arr = sdfg.add_array("B", [K + 2, N], dace.float64)
    C, C_arr = sdfg.add_array("C", [M, N], dace.float64)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    libnode = Gemm('_Gemm_', transA=False, transB=False, alpha=1.0, beta=0.0)
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b', dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None, dace.Memlet.from_array(C, C_arr))

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


# ---------------------------------------------------------------------------
# Row-wise (ikj) pure expansion: ``for i (parallel): for k (seq): for j (vector)``.
# Selected by canonicalization for known-small GEMMs (see
# dace/transformation/passes/canonicalize/finalize.py).
# ---------------------------------------------------------------------------
_rowwise_params = [
    dict(M=8, N=6, K=5),  # plain, non-square (catches index/transpose swaps); beta=1 -> C accumulate
    dict(M=8, N=6, K=5, transA=True),
    dict(M=8, N=6, K=5, transB=True),
    dict(M=8, N=6, K=5, transA=True, transB=True),
    dict(M=8, N=6, K=5, alpha=0.5, beta=0.0),  # alpha scaling, no C
    dict(M=8, N=6, K=5, alpha=0.5, beta=0.25),  # alpha/beta scaling + C accumulate
    dict(M=8, N=6, K=5, complex=True, alpha=complex(0.5, 0.5), beta=0.25),
]


@pytest.mark.parametrize('params', _rowwise_params)
def test_gemm_rowwise_preserves_value(params):
    """The 'rowwise' (ikj) GEMM expansion matches numpy across trans / alpha / beta / complex."""
    _do_test_gemm('rowwise', params)


def test_gemm_rowwise_loop_structure():
    """'rowwise' expands to a 3-deep i/k/j map nest with exactly the two inner (k, j) maps
    Sequential -- the outer i-map stays parallel. This is what makes j (unit-stride) vectorize
    and keeps K a sequential accumulation rather than a per-element WCR."""
    sdfg = create_gemm_sdfg(dace.float64, [8, 5], [5, 6], [8, 6], [8, 6],
                            transA=False,
                            transB=False,
                            alpha=1.0,
                            beta=0.0,
                            implementation='rowwise',
                            sdfg_name='rowwise_struct')
    sdfg.expand_library_nodes()
    unidim = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.sdfg.nodes.MapEntry) and len(n.map.params) == 1
    ]
    # The contraction nest is exactly i / k / j (the beta=0 init map is 2D, excluded).
    assert len(unidim) == 3
    seq = [me for me in unidim if me.map.schedule == dace.ScheduleType.Sequential]
    # k and j are the sequential inner maps; the outer i-map stays parallel.
    assert len(seq) == 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        for params in _test_params:
            _do_test_gemm('cuBLAS', params)
    # test_library_gemm('pure')
    # test_library_gemm('MKL')
    test_gemm_symbolic()
    test_gemm_symbolic_1()
    for params in _rowwise_params:
        test_gemm_rowwise_preserves_value(params)
    test_gemm_rowwise_loop_structure()
