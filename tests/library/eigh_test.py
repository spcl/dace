# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.linalg.eigh`` / ``eigvalsh`` lower to the ``Eigh`` library node and match numpy.

Eigenvalues are compared directly (both sides sort ascending). Eigenvectors are fixed only up to a
sign or phase per column, so they are checked by what defines them: ``A v = v diag(w)`` on the
full Hermitian matrix numpy reads from the requested triangle, and ``v^H v = I``.
"""
import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.libraries.linalg import Eigh
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = 12
DTYPES = [dace.float32, dace.float64, dace.complex64, dace.complex128]


def hermitian(n: int, dtype: dace.typeclass, seed: int = 7, batch: tuple[int, ...] = ()) -> np.ndarray:
    """A random Hermitian matrix (stack) whose unused triangle is garbage, so a solver reading the
    wrong triangle is caught."""
    rng = np.random.default_rng(seed)
    np_dtype = dtype.type
    a = rng.standard_normal((*batch, n, n))
    if np.issubdtype(np_dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal((*batch, n, n))
    return (a + np.conj(np.swapaxes(a, -1, -2))).astype(np_dtype)


def tolerance(dtype: dace.typeclass) -> float:
    return 2e-4 if dtype in (dace.float32, dace.complex64) else 1e-10


def full_matrix(a: np.ndarray, uplo: str) -> np.ndarray:
    """The Hermitian matrix numpy decomposes: the ``uplo`` triangle of ``a``, mirrored."""
    tri = np.tril(a) if uplo == 'L' else np.triu(a)
    diag = np.real(np.diagonal(tri, axis1=-2, axis2=-1))
    full = tri + np.conj(np.swapaxes(tri, -1, -2))
    idx = np.arange(a.shape[-1])
    full[..., idx, idx] = diag
    return full


def check_decomposition(a: np.ndarray, w: np.ndarray, v: np.ndarray, uplo: str, dtype: dace.typeclass) -> None:
    tol = tolerance(dtype)
    scale = np.abs(a).max()
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a, UPLO=uplo), rtol=tol, atol=tol * scale)
    full = full_matrix(a, uplo)
    np.testing.assert_allclose(full @ v, v * w[..., None, :], rtol=tol, atol=tol * scale)
    eye = np.broadcast_to(np.eye(a.shape[-1]), v.shape)
    np.testing.assert_allclose(np.conj(np.swapaxes(v, -1, -2)) @ v, eye, rtol=tol, atol=tol)


def eigh_sdfg(dtype: dace.typeclass, uplo: str = 'L', batch: tuple[int, ...] = (), tag: str = '') -> dace.SDFG:
    """The kernel's SDFG under a name unique to its configuration, so no case loads another's stale ``.so``."""
    real = {dace.complex64: dace.float32, dace.complex128: dace.float64}.get(dtype, dtype)

    @dace.program
    def eigh_kernel(a: dtype[(*batch, N, N)], w: real[(*batch, N)], v: dtype[(*batch, N, N)]):
        ww, vv = np.linalg.eigh(a, UPLO=uplo)
        w[:] = ww
        v[:] = vv

    sdfg = eigh_kernel.to_sdfg()
    sdfg.name = '_'.join(['eigh', dtype.to_string(), uplo, *map(str, batch), tag])
    return sdfg


def eigh_nodes(sdfg: dace.SDFG) -> list[Eigh]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Eigh)]


def run_cpu(sdfg: dace.SDFG, a: np.ndarray, real: type) -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros(a.shape[:-1], dtype=real)
    v = np.zeros_like(a)
    sdfg(a=a.copy(), w=w, v=v)
    return w, v


@pytest.mark.lapack
@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.to_string())
@pytest.mark.parametrize('implementation', ['OpenBLAS', 'pure'])
def test_eigh_matches_numpy(implementation: str, dtype: dace.typeclass) -> None:
    """Every implementation decomposes every dtype, reading the lower triangle only."""
    sdfg = eigh_sdfg(dtype, tag=implementation)
    for node in eigh_nodes(sdfg):
        node.implementation = implementation
    a = hermitian(N, dtype)
    a[np.triu_indices(N, 1)] = 1e3  # garbage in the triangle UPLO='L' does not read
    w, v = run_cpu(sdfg, a, np.real(a).dtype.type)
    check_decomposition(a, w, v, 'L', dtype)


@pytest.mark.lapack
@pytest.mark.parametrize('implementation', ['OpenBLAS', 'pure'])
def test_eigh_reads_the_upper_triangle_for_uplo_u(implementation: str) -> None:
    """``UPLO='U'`` reads the upper triangle, as numpy does; the lower one is never looked at."""
    dtype = dace.complex128
    sdfg = eigh_sdfg(dtype, uplo='U', tag=implementation)
    for node in eigh_nodes(sdfg):
        node.implementation = implementation
    a = hermitian(N, dtype, seed=3)
    a[np.tril_indices(N, -1)] = 1e3
    w, v = run_cpu(sdfg, a, np.float64)
    check_decomposition(a, w, v, 'U', dtype)


@pytest.mark.lapack
def test_eigh_solves_a_stack_of_matrices() -> None:
    """A ``(..., n, n)`` operand is decomposed one matrix at a time, like numpy's."""
    dtype = dace.float64
    batch = (2, 3)
    sdfg = eigh_sdfg(dtype, batch=batch)
    a = hermitian(N, dtype, seed=11, batch=batch)
    w, v = run_cpu(sdfg, a, np.float64)
    check_decomposition(a, w, v, 'L', dtype)


@pytest.mark.lapack
@pytest.mark.parametrize('implementation', ['OpenBLAS', 'pure'])
def test_eigvalsh_needs_no_eigenvector_output(implementation: str) -> None:
    """``eigvalsh`` leaves the eigenvector output unread, and every expansion still has to produce it."""

    @dace.program
    def eigvalsh_kernel(a: dace.float64[N, N], w: dace.float64[N]):
        w[:] = np.linalg.eigvalsh(a)

    sdfg = eigvalsh_kernel.to_sdfg()
    sdfg.name = f'eigvalsh_{implementation}'
    for node in eigh_nodes(sdfg):
        node.implementation = implementation
    a = hermitian(N, dace.float64, seed=5)
    w = np.zeros(N)
    sdfg(a=a.copy(), w=w)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-10, atol=1e-10)


def test_eigh_rejects_an_unknown_triangle() -> None:
    with pytest.raises(Exception, match="UPLO argument must be 'L' or 'U'"):
        eigh_sdfg(dace.float64, uplo='X')


def test_vendor_expansion_raises_on_a_nonzero_info_code() -> None:
    """A failed decomposition must not hand back eigenvectors that were never computed."""
    sdfg = eigh_sdfg(dace.float64, tag='info')
    for node in eigh_nodes(sdfg):
        node.implementation = 'OpenBLAS'
    sdfg.expand_library_nodes()
    code = '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert 'LAPACKE_dsyevd(LAPACK_ROW_MAJOR' in code, code
    assert 'if (_res != 0) throw std::runtime_error' in code, code


@pytest.mark.lapack
@pytest.mark.parametrize('dtype', [dace.float64, dace.complex128], ids=lambda d: d.to_string())
def test_canonicalize_cpu_lowers_eigh_to_lapack(dtype: dace.typeclass) -> None:
    """The canonicalize CPU tail selects LAPACK, never the Jacobi loop, and the result is numpy's."""
    sdfg = eigh_sdfg(dtype, tag='canon_cpu')
    canonicalize(sdfg)
    finalize_for_target(sdfg, 'cpu')
    assert [n.implementation for n in eigh_nodes(sdfg)] == ['OpenBLAS']
    a = hermitian(N, dtype, seed=13)
    w, v = run_cpu(sdfg, a, np.real(a).dtype.type)
    check_decomposition(a, w, v, 'L', dtype)


def device_solver() -> str:
    return 'rocSOLVER' if common.get_gpu_backend() == 'hip' else 'cuSolverDn'


def run_gpu(sdfg: dace.SDFG, a: np.ndarray, real: type) -> tuple[np.ndarray, np.ndarray]:
    import cupy
    w = cupy.zeros(a.shape[:-1], dtype=real)
    v = cupy.zeros(a.shape, dtype=a.dtype)
    sdfg(a=cupy.asarray(a), w=w, v=v)
    return cupy.asnumpy(w), cupy.asnumpy(v)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES, ids=lambda d: d.to_string())
def test_eigh_device_solver_matches_numpy(dtype: dace.typeclass) -> None:
    """The vendor GPU solver decomposes every dtype; the column-major staging is transparent."""
    sdfg = eigh_sdfg(dtype, tag='device')
    sdfg.apply_gpu_transformations()
    for node in eigh_nodes(sdfg):
        node.implementation = device_solver()
    a = hermitian(N, dtype, seed=17)
    a[np.triu_indices(N, 1)] = 1e3
    w, v = run_cpu(sdfg, a, np.real(a).dtype.type)  # the transformed graph copies to the device itself
    check_decomposition(a, w, v, 'L', dtype)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [dace.float64, dace.complex128], ids=lambda d: d.to_string())
def test_canonicalize_gpu_lowers_eigh_to_the_device_solver(dtype: dace.typeclass) -> None:
    """The canonicalize GPU pipeline selects the backend's vendor solver, and the result is numpy's."""
    sdfg = eigh_sdfg(dtype, tag='canon_gpu')
    canonicalize(sdfg, target='gpu')
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, 'gpu')
    assert [n.implementation for n in eigh_nodes(sdfg)] == [device_solver()]
    a = hermitian(N, dtype, seed=19)
    w, v = run_gpu(sdfg, a, np.real(a).dtype.type)
    check_decomposition(a, w, v, 'L', dtype)


if __name__ == '__main__':
    for dtype in DTYPES:
        for implementation in ('OpenBLAS', 'pure'):
            test_eigh_matches_numpy(implementation, dtype)
    test_eigh_reads_the_upper_triangle_for_uplo_u('OpenBLAS')
    test_eigh_reads_the_upper_triangle_for_uplo_u('pure')
    test_eigh_solves_a_stack_of_matrices()
    test_eigvalsh_needs_no_eigenvector_output('OpenBLAS')
    test_eigvalsh_needs_no_eigenvector_output('pure')
    test_eigh_rejects_an_unknown_triangle()
    test_vendor_expansion_raises_on_a_nonzero_info_code()
    test_canonicalize_cpu_lowers_eigh_to_lapack(dace.float64)
    test_canonicalize_cpu_lowers_eigh_to_lapack(dace.complex128)
    test_eigh_device_solver_matches_numpy(dace.float32)
    test_eigh_device_solver_matches_numpy(dace.float64)
    test_eigh_device_solver_matches_numpy(dace.complex64)
    test_eigh_device_solver_matches_numpy(dace.complex128)
    test_canonicalize_gpu_lowers_eigh_to_the_device_solver(dace.float64)
    test_canonicalize_gpu_lowers_eigh_to_the_device_solver(dace.complex128)
