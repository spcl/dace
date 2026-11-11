# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A parallel loop around one NumPy library call per iteration, through canonicalize + offload + finalize, is ONE GPU
kernel: the call takes its thread-block lowering (or expands in the kernel) and no host vendor call remains."""

from collections.abc import Callable

import numpy as np
import pytest
from host_library_calls_test import gpu_final, outer_maps

import dace
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.block_reduce import GPU_BLOCK_IMPLEMENTATIONS
from dace.sdfg import nodes

N = dace.symbol("N")
B = 4
L = 16
BATCH = 5
HOST_CALLS = {"cuSolverDn", "hipSolverDn", "cuFFT", "hipFFT", "OpenBLAS", "MKL", "cuBLAS", "hipBLAS", "FFTW", "LAPACKE"}


@dace.program
def solve_rows(A: dace.float64[N, B, B], b: dace.float64[N, B], x: dace.float64[N, B]):
    for i in dace.map[0:N]:
        x[i] = np.linalg.solve(A[i], b[i])


@dace.program
def inv_rows(A: dace.float64[N, B, B], C: dace.float64[N, B, B]):
    for i in dace.map[0:N]:
        C[i] = np.linalg.inv(A[i])


@dace.program
def cholesky_rows(A: dace.float64[N, B, B], C: dace.float64[N, B, B]):
    for i in dace.map[0:N]:
        C[i] = np.linalg.cholesky(A[i])


@dace.program
def eigh_rows(A: dace.float64[N, B, B], W: dace.float64[N, B], V: dace.float64[N, B, B]):
    for i in dace.map[0:N]:
        W[i], V[i] = np.linalg.eigh(A[i])


@dace.program
def fft_rows(X: dace.complex128[N, L], Y: dace.complex128[N, L]):
    for i in dace.map[0:N]:
        Y[i] = np.fft.fft(X[i])


@dace.program
def argmax_rows(X: dace.float64[N, L], idx: dace.int64[N]):
    for i in dace.map[0:N]:
        idx[i] = np.argmax(X[i])


@dace.program
def count_nonzero_rows(X: dace.float64[N, L], c: dace.int64[N]):
    for i in dace.map[0:N]:
        c[i] = np.count_nonzero(X[i])


@dace.program
def any_rows(X: dace.bool_[N, L], c: dace.bool_[N]):
    for i in dace.map[0:N]:
        c[i] = np.any(X[i])


@dace.program
def all_rows(X: dace.bool_[N, L], c: dace.bool_[N]):
    for i in dace.map[0:N]:
        c[i] = np.all(X[i])


@dace.program
def roll_rows(X: dace.float64[N, L], Y: dace.float64[N, L]):
    for i in dace.map[0:N]:
        Y[i] = np.roll(X[i], 3)


def spd(rng: np.random.Generator) -> np.ndarray:
    a = rng.random((BATCH, B, B))
    return a @ a.transpose(0, 2, 1) + B * np.eye(B)


def bools(rng: np.random.Generator) -> np.ndarray:
    x = rng.random((BATCH, L)) < 0.5
    x[0], x[1] = True, False  # An all-true and an all-false row
    return x


def sparse_rows(rng: np.random.Generator) -> np.ndarray:
    x = rng.random((BATCH, L))
    x[x < 0.4] = 0.0
    x[0] = 0.0
    return x


#: (program, inputs from an rng, output names and dtypes, outputs NumPy computes from the inputs)
Case = tuple[DaceProgram, Callable, dict[str, tuple[tuple[int, ...], type]], Callable]

CASES: dict[str, Case] = {
    "solve": (
        solve_rows,
        lambda r: {"A": spd(r), "b": r.random((BATCH, B))},
        {"x": ((BATCH, B), np.float64)},
        lambda A, b: {"x": np.linalg.solve(A, b[..., None])[..., 0]},
    ),
    "inv": (
        inv_rows,
        lambda r: {"A": r.random((BATCH, B, B)) + B * np.eye(B)},
        {"C": ((BATCH, B, B), np.float64)},
        lambda A: {"C": np.linalg.inv(A)},
    ),
    "cholesky": (
        cholesky_rows,
        lambda r: {"A": spd(r)},
        {"C": ((BATCH, B, B), np.float64)},
        lambda A: {"C": np.linalg.cholesky(A)},
    ),
    "eigh": (
        eigh_rows,
        lambda r: {"A": spd(r)},
        {"W": ((BATCH, B), np.float64), "V": ((BATCH, B, B), np.float64)},
        lambda A: {"W": np.linalg.eigh(A)[0]},
    ),
    "fft": (
        fft_rows,
        lambda r: {"X": r.random((BATCH, L)) + 1j * r.random((BATCH, L))},
        {"Y": ((BATCH, L), np.complex128)},
        lambda X: {"Y": np.fft.fft(X, axis=1)},
    ),
    "argmax": (
        argmax_rows,
        lambda r: {"X": r.random((BATCH, L))},
        {"idx": ((BATCH,), np.int64)},
        lambda X: {"idx": np.argmax(X, axis=1)},
    ),
    "count_nonzero": (
        count_nonzero_rows,
        lambda r: {"X": sparse_rows(r)},
        {"c": ((BATCH,), np.int64)},
        lambda X: {"c": np.count_nonzero(X, axis=1)},
    ),
    "any": (any_rows, lambda r: {"X": bools(r)}, {"c": ((BATCH,), np.bool_)}, lambda X: {"c": np.any(X, axis=1)}),
    "all": (all_rows, lambda r: {"X": bools(r)}, {"c": ((BATCH,), np.bool_)}, lambda X: {"c": np.all(X, axis=1)}),
    "roll": (
        roll_rows,
        lambda r: {"X": r.random((BATCH, L))},
        {"Y": ((BATCH, L), np.float64)},
        lambda X: {"Y": np.roll(X, 3, axis=1)},
    ),
}

#: The calls that stay a library node into the kernel and must pick their block lowering there
BLOCK_LOWERED = {"solve", "inv", "cholesky", "eigh", "fft", "any", "all"}


@pytest.mark.parametrize("name", list(CASES))
def test_a_loop_around_a_library_call_is_one_kernel(name: str):
    sdfg = gpu_final(f"library_e2e_{name}_structure", CASES[name][0])
    implementations = [n.implementation for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]

    assert [m.map.schedule for m in outer_maps(sdfg)] == [dace.ScheduleType.GPU_Device]
    assert not set(implementations) & HOST_CALLS
    if name in BLOCK_LOWERED:
        assert implementations and set(implementations) <= set(GPU_BLOCK_IMPLEMENTATIONS)


@pytest.mark.gpu
@pytest.mark.parametrize("name", list(CASES))
def test_a_loop_around_a_library_call_computes_what_numpy_computes(name: str):
    import cupy

    program, make_inputs, outputs, reference = CASES[name]
    inputs = make_inputs(np.random.default_rng(0))
    results = {k: cupy.zeros(shape, dtype) for k, (shape, dtype) in outputs.items()}
    device_inputs = {k: cupy.asarray(v) for k, v in inputs.items()}
    gpu_final(f"library_e2e_{name}_numerics", program)(**device_inputs, **results, N=BATCH)

    for k, expected in reference(**inputs).items():
        np.testing.assert_allclose(cupy.asnumpy(results[k]), expected, rtol=1e-10, atol=1e-12)
    if name == "eigh":
        # Eigenvectors are unique up to sign only: check they diagonalize A instead
        V, W = cupy.asnumpy(results["V"]), cupy.asnumpy(results["W"])
        np.testing.assert_allclose(V @ (W[..., None] * V.transpose(0, 2, 1)), inputs["A"], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    for case in CASES:
        test_a_loop_around_a_library_call_is_one_kernel(case)
        test_a_loop_around_a_library_call_computes_what_numpy_computes(case)
