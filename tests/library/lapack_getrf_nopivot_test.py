# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GetrfNoPivot``: in-place LU without pivoting, ``A = L @ U`` with ``L`` unit lower triangular, on every
expansion. The vendor expansions are checked against the same row-major factors as ``pure``: the GPU solvers are
column-major, and a call on the operand as it lies would factor its transpose."""

import numpy as np
import pytest

import dace
from dace.libraries.lapack import GetrfNoPivot
from dace.memlet import Memlet

n = dace.symbol("n")


def doolittle(a: np.ndarray) -> np.ndarray:
    """The reference: right-looking elimination without pivoting, factors packed in place."""
    a = a.copy()
    for k in range(a.shape[0]):
        a[k + 1 :, k] /= a[k, k]
        a[k + 1 :, k + 1 :] -= np.outer(a[k + 1 :, k], a[k, k + 1 :])
    return a


def make_sdfg(implementation: str, dtype, storage=dace.StorageType.Default, offset: int = 0) -> dace.SDFG:
    """Factor ``x[offset:offset+n, offset:offset+n]`` of an ``(n + offset) x (n + offset)`` host array; a device
    storage stages it through a device copy."""
    sdfg = dace.SDFG(f"getrf_nopivot_{implementation}_{dtype.to_string()}_{offset}")
    state = sdfg.add_state()
    sdfg.add_array("x", [n + offset, n + offset], dtype)
    region = f"{offset}:{offset} + n, {offset}:{offset} + n"
    node = GetrfNoPivot("lu")
    node.implementation = implementation
    if storage == dace.StorageType.Default:
        state.add_edge(state.add_read("x"), None, node, "_x", Memlet(f"x[{region}]"))
        state.add_edge(node, "_x", state.add_write("x"), None, Memlet(f"x[{region}]"))
        return sdfg
    sdfg.add_array("x_dev", [n + offset, n + offset], dtype, storage=storage, transient=True)
    full = f"0:n + {offset}, 0:n + {offset}"
    dev_in, dev_out = state.add_access("x_dev"), state.add_access("x_dev")
    state.add_nedge(state.add_read("x"), dev_in, Memlet(f"x[{full}]"))
    state.add_edge(dev_in, None, node, "_x", Memlet(f"x_dev[{region}]"))
    state.add_edge(node, "_x", dev_out, None, Memlet(f"x_dev[{region}]"))
    state.add_nedge(dev_out, state.add_write("x"), Memlet(f"x_dev[{full}]"))
    return sdfg


def matrix(size: int, dtype) -> np.ndarray:
    """Diagonally dominant, so elimination without pivoting is stable."""
    rng = np.random.default_rng(42)
    a = rng.random((size, size)) + size * np.eye(size)
    if dtype in (dace.complex64, dace.complex128):
        a = a + 1j * rng.random((size, size))
    return a.astype(dtype.type)


def run(implementation: str, dtype, storage=dace.StorageType.Default, offset: int = 0, size: int = 9) -> None:
    sdfg = make_sdfg(implementation, dtype, storage, offset)
    x = matrix(size + offset, dtype)
    want = x.copy()
    want[offset:, offset:] = doolittle(x[offset:, offset:])
    sdfg(x=x, n=size)
    tol = 1e-4 if dtype in (dace.float32, dace.complex64) else 1e-12
    assert np.allclose(x, want, rtol=tol, atol=tol), np.abs(x - want).max()
    # The factors are the factorization: L @ U recovers the operand.
    lu = x[offset:, offset:]
    lower, upper = np.tril(lu, -1) + np.eye(size), np.triu(lu)
    original = matrix(size + offset, dtype)[offset:, offset:]
    assert np.allclose(lower @ upper, original, rtol=tol * 10, atol=tol * 10)


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64, dace.complex128])
def test_pure_factors_in_place(dtype):
    run("pure", dtype)


def test_pure_factors_a_strided_submatrix():
    """A block of a larger matrix: the row stride is the parent's, the corner the offset."""
    run("pure", dace.float64, offset=3)


def test_pure_expansion_is_one_container():
    """The nest reads what it wrote, so the expansion names the operand once: two inner names would let an
    analysis of the nested SDFG take the reads for reads of a read-only input."""
    sdfg = make_sdfg("pure", dace.float64)
    sdfg.expand_library_nodes()
    (nsdfg,) = [nd for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, dace.nodes.NestedSDFG)]
    assert set(nsdfg.sdfg.arrays) == {"_x"}
    assert set(nsdfg.in_connectors) == set(nsdfg.out_connectors) == {"_x"}


def test_validate_refuses_out_of_place():
    sdfg = dace.SDFG("getrf_nopivot_out_of_place")
    state = sdfg.add_state()
    sdfg.add_array("x", [n, n], dace.float64)
    sdfg.add_array("y", [n, n], dace.float64)
    node = GetrfNoPivot("lu")
    state.add_edge(state.add_read("x"), None, node, "_x", Memlet("x[0:n, 0:n]"))
    state.add_edge(node, "_x", state.add_write("y"), None, Memlet("y[0:n, 0:n]"))
    with pytest.raises(ValueError, match="in place"):
        node.validate(sdfg, state)


def test_validate_refuses_non_square():
    sdfg = dace.SDFG("getrf_nopivot_rectangular")
    state = sdfg.add_state()
    sdfg.add_array("x", [n, n + 1], dace.float64)
    node = GetrfNoPivot("lu")
    state.add_edge(state.add_read("x"), None, node, "_x", Memlet("x[0:n, 0:n + 1]"))
    state.add_edge(node, "_x", state.add_write("x"), None, Memlet("x[0:n, 0:n + 1]"))
    with pytest.raises(ValueError, match="square"):
        node.validate(sdfg, state)


@pytest.mark.mkl
@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
def test_mkl(dtype):
    run("MKL", dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [dace.float32, dace.float64, dace.complex128])
@pytest.mark.parametrize("offset", [0, 2])
def test_cusolverdn(dtype, offset):
    run("cuSolverDn", dtype, dace.StorageType.GPU_Global, offset)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [dace.float32, dace.float64, dace.complex128])
@pytest.mark.parametrize("offset", [0, 2])
def test_hipsolver(dtype, offset):
    run("hipSolverDn", dtype, dace.StorageType.GPU_Global, offset)


if __name__ == "__main__":
    for t in (dace.float32, dace.float64, dace.complex128):
        test_pure_factors_in_place(t)
    test_pure_factors_a_strided_submatrix()
    test_pure_expansion_is_one_container()
    test_validate_refuses_out_of_place()
    test_validate_refuses_non_square()
