# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map around library calls no kernel can run stays on the host whole, as in the CPU form (quatrex_rgf's energy map)."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")
BS = 8


@dace.program
def inverse_products(A: dace.float64[N, BS, BS], B: dace.float64[N, BS, BS], C: dace.float64[N, BS, BS]):
    for i in dace.map[0:N]:
        C[i] = np.linalg.inv(A[i]) @ B[i]


def gpu_final(name: str) -> dace.SDFG:
    sdfg = inverse_products.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    return sdfg


def outer_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry) and s.entry_node(n) is None]


def test_a_map_around_an_inverse_is_threaded_on_the_host_with_the_cpu_library():
    sdfg = gpu_final("host_library_calls_structure")
    libraries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]

    assert [m.map.schedule for m in outer_maps(sdfg)] == [dace.ScheduleType.CPU_Multicore]
    assert libraries and not any(n.schedule in dace.dtypes.GPU_SCHEDULES for n in libraries)
    assert not {n.implementation for n in libraries} & {"cuBLAS", "hipBLAS", "cuSolverDn", "hipSolverDn"}


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


if __name__ == "__main__":
    test_a_map_around_an_inverse_is_threaded_on_the_host_with_the_cpu_library()
    test_a_map_around_an_inverse_computes_what_numpy_computes()
