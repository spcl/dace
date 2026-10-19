# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Corpus of small DaCe programs exercising the SDFG features modeled in ``formal/``.

Each entry of ``PROGRAMS`` is ``(name, dace_program, argument factory)``; the
argument factory takes a global ``seed`` (see :func:`_rng`) and returns concrete
NumPy inputs (and symbol values) for differential testing.  Keep programs small;
their corpus is generated (not committed) under ``formal/tests/corpus``.
"""

import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")
H = dace.symbol("H")
W = dace.symbol("W")
nnz = dace.symbol("nnz")


@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda x, y: x + y)[i, j]
            c = a * b


@dace.program
def stencil2d(A: dace.float64[H, W], B: dace.float64[H, W]):
    for i, j in dace.map[1 : H - 1, 1 : W - 1]:
        with dace.tasklet:
            c << A[i, j]
            n << A[i - 1, j]
            s << A[i + 1, j]
            w << A[i, j - 1]
            e << A[i, j + 1]
            out >> B[i, j]
            out = 0.2 * (c + n + s + w + e)


@dace.program
def strided_sum(A: dace.float64[N], out: dace.float64[1]):
    for i in dace.map[0:N:2]:
        with dace.tasklet:
            a << A[i]
            o >> out(1, lambda x, y: x + y)[0]
            o = a


@dace.program
def reduce_rows(A: dace.float64[M, N], out: dace.float64[M]):
    out[:] = np.sum(A, axis=1)


@dace.program
def spmv(
    A_row: dace.uint32[H + 1],
    A_col: dace.uint32[nnz],
    A_val: dace.float32[nnz],
    x: dace.float32[W],
    b: dace.float32[H],
):
    for i in dace.map[0:H]:
        for j in dace.map[A_row[i] : A_row[i + 1]]:
            with dace.tasklet:
                aval << A_val[j]
                xval << x[A_col[j]]
                out >> b(1, lambda a, b: a + b)[i]
                out = aval * xval


@dace.program
def loop_with_branch(A: dace.float64[N], out: dace.float64[N]):
    s = 0.0
    for i in range(N):
        if A[i] > 0.0:
            s = s + A[i]
        else:
            s = s - 1.0
        out[i] = s


@dace.program
def while_loop(A: dace.float64[N], out: dace.int32[1]):
    i = 0
    count = 0
    while i < N and A[i] >= 0.0:
        count = count + 1
        i = i + 1
    out[0] = count


@dace.program
def inner_scale(x: dace.float64[N], factor: dace.float64):
    x[:] = x * factor


@dace.program
def nested_call(A: dace.float64[M, N]):
    for i in range(M):
        inner_scale(A[i, :], 2.0)


@dace.program
def slice_view(A: dace.float64[N], B: dace.float64[N]):
    B[1 : N - 1] = A[0 : N - 2] + A[2:N]


@dace.program
def augassign_wcr(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] += A[i] * 2.0


@dace.program
def two_states(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    tmp = np.ndarray([N], dtype=np.float64)
    tmp[:] = A + B
    C[:] = tmp * 3.0


@dace.program
def nested_maps_transient(A: dace.float64[M, N], B: dace.float64[M]):
    for i in dace.map[0:M]:
        acc = np.zeros([1], dtype=np.float64)
        for j in dace.map[0:N]:
            with dace.tasklet:
                a << A[i, j]
                o >> acc(1, lambda x, y: x + y)[0]
                o = a
        B[i] = acc[0]


def _rng(program_seed: int, base_seed: int = 0) -> np.random.Generator:
    """Random generator for one program's inputs.

    ``program_seed`` is fixed per program; ``base_seed`` is the global seed of a
    corpus recording (``difftest.py --record --seed``).  Base seed 0 reproduces
    the historical per-program streams exactly; any other base seed mixes both
    into a ``SeedSequence``.

    :param program_seed: The per-program seed.
    :param base_seed: The global recording seed.
    :return: A NumPy random generator.
    """
    if base_seed == 0:
        return np.random.default_rng(program_seed)
    return np.random.default_rng([base_seed, program_seed])


def args_axpy(seed: int = 0) -> dict:
    r = _rng(1, seed)
    return dict(a=2.0, x=r.random(16), y=r.random(16), N=16)


def args_matmul(seed: int = 0) -> dict:
    r = _rng(2, seed)
    return dict(A=r.random((4, 6)), B=r.random((6, 5)), C=np.zeros((4, 5)), M=4, K=6, N=5)


def args_stencil(seed: int = 0) -> dict:
    r = _rng(3, seed)
    return dict(A=r.random((6, 7)), B=np.zeros((6, 7)), H=6, W=7)


def args_strided(seed: int = 0) -> dict:
    r = _rng(4, seed)
    return dict(A=r.random(11), out=np.zeros(1), N=11)


def args_reduce(seed: int = 0) -> dict:
    r = _rng(5, seed)
    return dict(A=r.random((3, 5)), out=np.zeros(3), M=3, N=5)


def args_spmv(seed: int = 0) -> dict:
    r = _rng(6, seed)
    h, w = 4, 5
    dense = (r.random((h, w)) > 0.5).astype(np.float32) * r.random((h, w)).astype(np.float32)
    rows = [0]
    cols, vals = [], []
    for i in range(h):
        for j in range(w):
            if dense[i, j] != 0:
                cols.append(j)
                vals.append(dense[i, j])
        rows.append(len(cols))
    return dict(
        A_row=np.array(rows, dtype=np.uint32),
        A_col=np.array(cols, dtype=np.uint32),
        A_val=np.array(vals, dtype=np.float32),
        x=r.random(w).astype(np.float32),
        b=np.zeros(h, dtype=np.float32),
        H=h,
        W=w,
        nnz=len(cols),
    )


def args_loop_branch(seed: int = 0) -> dict:
    r = _rng(7, seed)
    return dict(A=r.random(8) - 0.5, out=np.zeros(8), N=8)


def args_while(seed: int = 0) -> dict:
    r = _rng(8, seed)
    a = r.random(8)
    a[5] = -1.0
    return dict(A=a, out=np.zeros(1, dtype=np.int32), N=8)


def args_nested_call(seed: int = 0) -> dict:
    r = _rng(9, seed)
    return dict(A=r.random((3, 4)), M=3, N=4)


def args_slice_view(seed: int = 0) -> dict:
    r = _rng(10, seed)
    return dict(A=r.random(9), B=np.zeros(9), N=9)


def args_augassign(seed: int = 0) -> dict:
    r = _rng(11, seed)
    return dict(A=r.random(7), B=r.random(7), N=7)


def args_two_states(seed: int = 0) -> dict:
    r = _rng(12, seed)
    return dict(A=r.random(5), B=r.random(5), C=np.zeros(5), N=5)


def args_nested_maps_transient(seed: int = 0) -> dict:
    r = _rng(13, seed)
    return dict(A=r.random((3, 4)), B=np.zeros(3), M=3, N=4)


PROGRAMS = [
    ("axpy", axpy, args_axpy),
    ("matmul", matmul, args_matmul),
    ("stencil2d", stencil2d, args_stencil),
    ("strided_sum", strided_sum, args_strided),
    ("reduce_rows", reduce_rows, args_reduce),
    ("spmv", spmv, args_spmv),
    ("loop_with_branch", loop_with_branch, args_loop_branch),
    ("while_loop", while_loop, args_while),
    ("nested_call", nested_call, args_nested_call),
    ("slice_view", slice_view, args_slice_view),
    ("augassign_wcr", augassign_wcr, args_augassign),
    ("two_states", two_states, args_two_states),
    ("nested_maps_transient", nested_maps_transient, args_nested_maps_transient),
]
