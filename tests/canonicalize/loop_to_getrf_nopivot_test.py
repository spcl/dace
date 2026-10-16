# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToGetrfNoPivot`` lifts a hand-written LU elimination without pivoting to a ``GetrfNoPivot`` node and is a
strict no-op on every other nest.

The matcher reads the normalized nest ``lift_semantics`` receives and compares what one outer iteration stores
against elimination step ``k``: the column below the pivot divided by it, the trailing submatrix less the outer
product. The npbench slice spelling and the scalar loop spelling must both lift; nests with the same access shapes
but other arithmetic, a pivot search, the row-scaled (Crout) variant, or a partial range must not. On CPU the
lifted node is expanded at once to the plain nest the parallelism phase maps; on GPU it stays for the vendor
solver.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

import dace
from dace.libraries.lapack.nodes.getrf_nopivot import GetrfNoPivot
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.loop_to_getrf_nopivot import LoopToGetrfNoPivot

N = dace.symbol("N")
SIZE = 23


@dace.program
def lu_slices(A: dace.float64[N, N]):
    """The spelling HPCAgent-Bench emits for npbench ``lu``."""
    for k in range(N):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]


@dace.program
def lu_loops(A: dace.float64[N, N]):
    for k in range(N):
        for i in range(k + 1, N):
            A[i, k] = A[i, k] / A[k, k]
        for i in range(k + 1, N):
            for j in range(k + 1, N):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]


@dace.program
def lu_loops_to_n_minus_1(A: dace.float64[N, N]):
    """The polybench bound: the last step has nothing below the pivot."""
    for k in range(N - 1):
        for i in range(k + 1, N):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, N):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]


@dace.program
def ludcmp_like(A: dace.float64[N, N], b: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
    for k in range(N):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]


@dace.program
def crout_rows(A: dace.float64[N, N]):
    """The row below the pivot is scaled instead of the column: ``U`` unit diagonal, not ``GetrfNoPivot``."""
    for k in range(N):
        A[k, k + 1 :] = A[k, k + 1 :] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]


@dace.program
def plus_update(A: dace.float64[N, N]):
    for k in range(N):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] + A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]


@dace.program
def unscaled_update(A: dace.float64[N, N]):
    """The trailing update reads the column before it is scaled (the scale comes after)."""
    for k in range(N):
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]


@dace.program
def partial_update(A: dace.float64[N, N]):
    for k in range(N):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 2 :] = A[k + 1 :, k + 2 :] - A[k + 1 :, k][:, None] * A[k, k + 2 :][None, :]


@dace.program
def second_output(A: dace.float64[N, N], d: dace.float64[N]):
    for k in range(N):
        d[k] = A[k, k]
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]


@dace.program
def left_looking(A: dace.float64[N, N]):
    """polybench ``lu``: the same factors, another nest; stays on the wavefront path."""
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]


def reference_lu(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    for k in range(a.shape[0]):
        a[k + 1 :, k] /= a[k, k]
        a[k + 1 :, k + 1 :] -= np.outer(a[k + 1 :, k], a[k, k + 1 :])
    return a


def matrix(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.random((size, size)) + size * np.eye(size)


def normalized(program) -> dace.SDFG:
    """``program`` as the phased recipe hands it to ``lift_semantics``: the phases before it, no lift run."""
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, target="gpu", stages=["cleanup", "normalize_statements"])
    return sdfg


def getrf_nodes(sdfg: dace.SDFG) -> list[GetrfNoPivot]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, GetrfNoPivot)]


@pytest.mark.parametrize("program", [lu_slices, lu_loops, lu_loops_to_n_minus_1])
def test_an_elimination_nest_lifts_to_one_node_that_factors(program):
    sdfg = normalized(program)
    assert LoopToGetrfNoPivot(target="gpu").apply_pass(sdfg, {}) == 1
    (node,) = getrf_nodes(sdfg)
    assert not any(isinstance(n, LoopRegion) for n in sdfg.nodes()), "the nest must be gone"
    a = matrix()
    want = reference_lu(a)
    sdfg(A=a, N=SIZE)
    assert np.allclose(a, want, rtol=1e-12, atol=1e-12), np.abs(a - want).max()


def test_ludcmp_lifts_the_factorization_and_keeps_the_solves():
    sdfg = normalized(ludcmp_like)
    assert LoopToGetrfNoPivot(target="gpu").apply_pass(sdfg, {}) == 1
    assert len(getrf_nodes(sdfg)) == 1
    a, rng = matrix(), np.random.default_rng(5)
    b = rng.random(SIZE)
    x, y = np.zeros(SIZE), np.zeros(SIZE)
    sdfg(A=a, b=b.copy(), x=x, y=y, N=SIZE)
    original = matrix()
    assert np.allclose(original @ x, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "program", [crout_rows, plus_update, unscaled_update, partial_update, second_output, left_looking]
)
def test_a_nest_that_is_not_the_elimination_is_left_alone(program):
    sdfg = normalized(program)
    assert not LoopToGetrfNoPivot(target="gpu").apply_pass(sdfg, {})
    assert getrf_nodes(sdfg) == []


def test_cpu_expands_the_lifted_node_into_the_mapped_elimination():
    """CPU: the node is expanded in the pipeline, so its nest reaches the parallelism phase, which maps the
    trailing update: no node is left, a parallel map is, and the factors are numpy's."""
    sdfg = lu_slices.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, target="cpu")
    assert getrf_nodes(sdfg) == []
    labels = [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert any(label.startswith("lu_column") for label in labels), labels
    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())
    a = matrix()
    want = reference_lu(a)
    sdfg(A=a, N=SIZE)
    assert np.allclose(a, want, rtol=1e-12, atol=1e-12)


def test_gpu_keeps_the_node_for_the_vendor_solver():
    sdfg = lu_slices.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, target="gpu")
    assert len(getrf_nodes(sdfg)) == 1


if __name__ == "__main__":
    for p in (lu_slices, lu_loops, lu_loops_to_n_minus_1):
        test_an_elimination_nest_lifts_to_one_node_that_factors(p)
    test_ludcmp_lifts_the_factorization_and_keeps_the_solves()
    for p in (crout_rows, plus_update, unscaled_update, partial_update, second_output, left_looking):
        test_a_nest_that_is_not_the_elimination_is_left_alone(p)
    test_cpu_expands_the_lifted_node_into_the_mapped_elimination()
    test_gpu_keeps_the_node_for_the_vendor_solver()
