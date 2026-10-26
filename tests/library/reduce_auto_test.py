# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Reduce``'s default ``auto`` implementation dispatches on the node's schedule, and the expansions it picks
compile and compute: OpenMP min/max, a sequential reduction inside a parallel map, a mixed-type sum and complex
operands."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.reduce import Reduce

N = 64


def reduce_sdfg(name: str, wcr: str, in_dtype, out_dtype, identity) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), in_dtype)
    sdfg.add_array("B", (1,), out_dtype)
    state = sdfg.add_state()
    red = Reduce("reduce", wcr, axes=None, identity=identity)
    state.add_node(red)
    state.add_edge(state.add_read("A"), None, red, None, dace.Memlet(f"A[0:{N}]"))
    state.add_edge(red, None, state.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg


@pytest.mark.parametrize("wcr, ref", [("lambda a, b: min(a, b)", np.min), ("lambda a, b: max(a, b)", np.max)])
def test_openmp_min_max_compile_and_compute(wcr, ref):
    sdfg = reduce_sdfg(f"reduce_omp_{ref.__name__}", wcr, dace.float64, dace.float64, None)
    sdfg.expand_library_nodes()
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1 and "#pragma omp" in tasklets[0].code.as_string

    # Without an identity the reduction starts from what B holds
    a, b = np.random.rand(N), np.full(1, np.inf if ref is np.min else -np.inf)
    sdfg(A=a, B=b)
    assert b[0] == ref(a)


def test_mixed_type_sum_keeps_the_input_type_until_the_store():
    """A Fortran ``SUM`` of an integer array into a real: the accumulator stays integer and a tasklet casts."""
    sdfg = reduce_sdfg("reduce_mixed_seq", "lambda a, b: a + b", dace.int32, dace.float64, 0)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.implementation = "pure-seq"
    a, b = np.arange(N, dtype=np.int32), np.zeros(1)
    sdfg(A=a, B=b)
    assert b[0] == float(a.sum())


def test_reduction_inside_a_parallel_map_is_sequential():
    @dace.program
    def rowsum(A: dace.float64[8, N], B: dace.float64[8]):
        for i in dace.map[0:8]:
            B[i] = np.sum(A[i])

    sdfg = rowsum.to_sdfg(simplify=True)
    a, b = np.random.rand(8, N), np.zeros(8)
    sdfg(A=a, B=b)
    assert np.allclose(b, a.sum(axis=1))


def test_complex_sum_with_openmp():
    sdfg = reduce_sdfg("reduce_omp_complex", "lambda a, b: a + b", dace.complex128, dace.complex128, 0)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.implementation = "OpenMP"
    a = (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
    b = np.zeros(1, dtype=np.complex128)
    sdfg(A=a, B=b)
    assert np.isclose(b[0], a.sum())


@pytest.mark.gpu
def test_gpu_reduction_into_a_host_scalar():
    import cupy as cp

    sdfg = reduce_sdfg("reduce_gpu_host_out", "lambda a, b: a + b", dace.float64, dace.float64, 0)
    sdfg.arrays["A"].storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.schedule = dace.ScheduleType.GPU_Device
    a, b = cp.random.rand(N), np.zeros(1)
    sdfg(A=a, B=b)
    assert np.isclose(b[0], float(a.sum()))
