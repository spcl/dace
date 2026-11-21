# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A small constant-shape GEMM that a map or loop runs once per iteration is that iteration's work.

On the CPU it lowers to the row-wise loop nest, single-threaded; on the GPU it lowers to the in-thread
``pure`` loop before the offload, so the map around it becomes the kernel instead of a host loop that
launches one vendor GEMM per iteration (scattering_self_energies: ``Norb x Norb`` products, Norb = 10).
"""

import numpy as np

import dace
from dace import dtypes, symbolic
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
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


def test_an_extent_a_fact_bounds_counts_as_small():
    sdfg = products_of_runtime_size.to_sdfg(simplify=True)
    sdfg.add_symbol_relation(symbolic.Relation(symbolic.RelationKind.LE, K, 16))
    assert SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {}) == 1
    rng = np.random.default_rng(20261112)
    a, b = rng.random((4, 5, 5)), rng.random((4, 5, 5))
    c = np.zeros((4, 5, 5))
    sdfg(a=a, b=b, c=c, N=4, K=5)
    assert np.allclose(c, a @ b)


def test_a_small_scale_alone_bounds_no_extent():
    sdfg = products_of_runtime_size.to_sdfg(simplify=True)
    sdfg.set_symbol_scale("K", symbolic.Scale.SMALL)
    assert SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {}) is None


def library_node_per_row(name: str, node: nodes.LibraryNode, operands: dict, output: tuple) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("rows", {"r": "0:N"}, schedule=dtypes.ScheduleType.CPU_Multicore)
    for conn, (data, shape, subset) in operands.items():
        sdfg.add_array(data, shape, dace.float64)
        state.add_memlet_path(state.add_read(data), entry, node, dst_conn=conn, memlet=dace.Memlet(f"{data}[{subset}]"))
    conn, (data, shape, subset) = output
    sdfg.add_array(data, shape, dace.float64)
    state.add_memlet_path(node, exit_, state.add_write(data), src_conn=conn, memlet=dace.Memlet(f"{data}[{subset}]"))
    return sdfg


def test_a_tiny_einsum_under_a_parallel_map_lowers_to_the_rowwise_gemm():
    einsum = Einsum("einsum")
    einsum.einsum_str = "ij,jk->ik"
    einsum.add_in_connector("x")
    einsum.add_in_connector("y")
    einsum.add_out_connector("z")
    sdfg = library_node_per_row(
        "tiny_einsum_per_row",
        einsum,
        {"x": ("a", (N, 3, 4), "r, 0:3, 0:4"), "y": ("b", (N, 4, 5), "r, 0:4, 0:5")},
        ("z", ("c", (N, 3, 5), "r, 0:3, 0:5")),
    )
    assert SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {}) == 1
    assert not any(isinstance(n, Einsum) for n, _ in sdfg.all_nodes_recursive())
    assert [n.implementation for n in products(sdfg)] == ["rowwise"]
    rng = np.random.default_rng(20261113)
    a, b = rng.random((4, 3, 4)), rng.random((4, 4, 5))
    c = np.zeros((4, 3, 5))
    sdfg(a=a, b=b, c=c, N=4)
    assert np.allclose(c, a @ b)


def test_a_tiny_tensor_transpose_under_a_parallel_map_is_the_inlined_map():
    transpose = TensorTranspose("transpose", axes=[0, 2, 1, 3])
    sdfg = library_node_per_row(
        "tiny_transpose_per_row",
        transpose,
        {"_inp_tensor": ("a", (N, 2, 3, 4), "r, 0:2, 0:3, 0:4")},
        ("_out_tensor", ("b", (N, 3, 2, 4), "r, 0:3, 0:2, 0:4")),
    )
    assert SpecializeSmallGemms(dtypes.DeviceType.CPU).apply_pass(sdfg, {}) == 1
    assert transpose.implementation == "pure" and transpose.schedule == dtypes.ScheduleType.Sequential
    a = np.random.default_rng(20261114).random((4, 2, 3, 4))
    b = np.zeros((4, 3, 2, 4))
    sdfg(a=a, b=b, N=4)
    assert np.allclose(b, a.transpose(0, 2, 1, 3))


if __name__ == "__main__":
    test_a_small_product_per_map_iteration_is_one_rowwise_sequential_loop_on_the_cpu()
    test_a_small_product_under_a_parallel_map_runs_inside_the_kernel_on_the_gpu()
    test_specialized_extents_count_as_compile_time_constants()
    test_an_extent_a_fact_bounds_counts_as_small()
    test_a_small_scale_alone_bounds_no_extent()
    test_a_tiny_einsum_under_a_parallel_map_lowers_to_the_rowwise_gemm()
    test_a_tiny_tensor_transpose_under_a_parallel_map_is_the_inlined_map()
