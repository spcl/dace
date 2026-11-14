# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map around library calls no kernel can run stays on the host whole, as in the CPU form (quatrex_rgf's energy map);
one around a call a kernel can run is that kernel (polybench doitgen's matrix-vector product per ``(r, q)``)."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")
BS = 8
NR, NQ, NP = (dace.symbol(s) for s in ("NR", "NQ", "NP"))


@dace.program
def inverse_products(A: dace.float64[N, BS, BS], B: dace.float64[N, BS, BS], C: dace.float64[N, BS, BS]):
    for i in dace.map[0:N]:
        C[i] = np.linalg.inv(A[i]) @ B[i]


@dace.program
def doitgen_rows(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
    for r, q in dace.map[0:NR, 0:NQ]:
        A[r, q, :] = A[r, q, :] @ C4


def gpu_final(name: str, program=inverse_products) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    return sdfg


def outer_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry) and s.entry_node(n) is None]


def test_a_map_around_an_inverse_is_a_kernel():
    sdfg = gpu_final("host_library_calls_structure")
    libraries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]

    # The inverse has an in-kernel thread-block lowering, so the map is a kernel and no host library call remains
    assert [m.map.schedule for m in outer_maps(sdfg)] == [dace.ScheduleType.GPU_Device]
    assert not {n.implementation for n in libraries} & {"OpenBLAS", "MKL", "cuSolverDn", "hipSolverDn"}


@pytest.mark.gpu
def test_a_map_around_an_inverse_computes_what_numpy_computes():
    import cupy

    rng = np.random.default_rng(0)
    n = 5
    A = rng.random((n, BS, BS)) + BS * np.eye(BS)
    B = rng.random((n, BS, BS))
    C = cupy.zeros((n, BS, BS))
    gpu_final("host_library_calls_numerics")(A=cupy.asarray(A), B=cupy.asarray(B), C=C, N=n)
    assert np.allclose(cupy.asnumpy(C), np.linalg.inv(A) @ B)


def test_a_map_around_a_matrix_vector_product_is_a_kernel():
    sdfg = gpu_final("doitgen_rows_structure", doitgen_rows)
    libraries = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]

    assert [m.map.schedule for m in outer_maps(sdfg)] == [dace.ScheduleType.GPU_Device]
    assert libraries and not {n.implementation for n in libraries} & {"cuBLAS", "hipBLAS", "rocBLAS", "OpenBLAS"}


@pytest.mark.gpu
def test_a_kernel_around_a_matrix_vector_product_computes_what_numpy_computes():
    import cupy

    rng = np.random.default_rng(0)
    A, C4 = rng.random((3, 4, 5)), rng.random((5, 5))
    out = cupy.asarray(A)
    gpu_final("doitgen_rows_numerics", doitgen_rows)(A=out, C4=cupy.asarray(C4), NR=3, NQ=4, NP=5)
    np.testing.assert_allclose(cupy.asnumpy(out), A @ C4, rtol=1e-13)


if __name__ == "__main__":
    test_a_map_around_an_inverse_is_a_kernel()
    test_a_map_around_an_inverse_computes_what_numpy_computes()
    test_a_map_around_a_matrix_vector_product_is_a_kernel()
    test_a_kernel_around_a_matrix_vector_product_computes_what_numpy_computes()
