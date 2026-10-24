# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A small constant-shape GEMM that a map or loop runs once per iteration is that iteration's work.

On the CPU it lowers to the row-wise loop nest, single-threaded; on the GPU it lowers to the in-thread
``pure`` loop before the offload, so the map around it becomes the kernel instead of a host loop that
launches one vendor GEMM per iteration (scattering_self_energies: ``Norb x Norb`` products, Norb = 10).
"""

import numpy as np

import dace
from dace import dtypes
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations
from dace.transformation.passes.offloading.taskloop import is_device_wide_libnode
from dace.transformation.passes.specialize_small_gemms import SpecializeSmallGemms

N = dace.symbol("N")
K = dace.symbol("K")


@dace.program
def products_per_row(a: dace.float64[N, 10, 10], b: dace.float64[N, 10, 10], c: dace.float64[N, 10, 10]):
    for i in dace.map[0:N]:
        c[i] = a[i] @ b[i]


@dace.program
def products_of_runtime_size(a: dace.float64[N, K, K], b: dace.float64[N, K, K], c: dace.float64[N, K, K]):
    for i in dace.map[0:N]:
        c[i] = a[i] @ b[i]


def products(sdfg: dace.SDFG) -> list[nodes.LibraryNode]:
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, (MatMul, Gemm))]


def test_a_small_product_per_map_iteration_is_one_rowwise_sequential_loop_on_the_cpu():
    sdfg = products_per_row.to_sdfg(simplify=True)
    assert SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {}) == 1
    # The implementation selection after the specialization stages keeps the pinned choice.
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    found = products(sdfg)
    assert found, "the program lost its matrix products"
    for node in found:
        assert node.implementation == "rowwise", node.implementation
        assert node.schedule == dtypes.ScheduleType.Sequential, node.schedule

    rng = np.random.default_rng(20261015)
    a, b = rng.random((4, 10, 10)), rng.random((4, 10, 10))
    c = np.zeros((4, 10, 10))
    sdfg(a=a, b=b, c=c, N=4)
    assert np.allclose(c, a @ b)


def test_a_small_product_under_a_parallel_map_runs_inside_the_kernel_on_the_gpu():
    """Selected before the offload: a ``MatMul`` becomes its ``Gemm`` and takes the in-thread loop, which the
    offload reads as kernel code, so the map stays the kernel."""
    sdfg = products_per_row.to_sdfg(simplify=True)
    assert any(is_device_wide_libnode(node) for node in products(sdfg)), "the test needs a host-issued product"
    assert SpecializeSmallGemms(dtypes.DeviceType.GPU).apply_pass(sdfg, {}) == 1
    found = products(sdfg)
    assert [type(node) for node in found] == [Gemm], found
    assert found[0].implementation == "pure"
    assert not is_device_wide_libnode(found[0])


def test_specialized_extents_count_as_compile_time_constants():
    sdfg = products_of_runtime_size.to_sdfg(simplify=True)
    assert SpecializeSmallGemms(dtypes.DeviceType.GPU).apply_pass(sdfg, {}) is None, (
        "a runtime-sized product is not known to be small"
    )
    sdfg = products_of_runtime_size.to_sdfg(simplify=True)
    sdfg.specialize(dict(K=10))
    assert SpecializeSmallGemms(dtypes.DeviceType.GPU).apply_pass(sdfg, {}) == 1


if __name__ == "__main__":
    test_a_small_product_per_map_iteration_is_one_rowwise_sequential_loop_on_the_cpu()
    test_a_small_product_under_a_parallel_map_runs_inside_the_kernel_on_the_gpu()
    test_specialized_extents_count_as_compile_time_constants()
