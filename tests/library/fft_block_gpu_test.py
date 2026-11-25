# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An FFT library node inside a GPU kernel lowers to one thread block per invocation (``dace/cuda/block_fft.cuh``)."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.fft.nodes.fft import FFT, IFFT
from dace.libraries.standard.block_reduce import gpu_block_implementation
from dace.sdfg import nodes
from dace.transformation.passes.offloading.taskloop import is_kernel_refused_libnode

B = dace.symbol("B", dtype=dace.int64)
N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)

BLOCK = "CUDA (block strided)"


def batched_kernel(node_type: type, dtype: dace.typeclass = dace.complex128, **kwargs):
    """A ``GPU_Device`` map over ``b`` transforming row ``X[b]`` into ``Y[b]``."""
    sdfg = dace.SDFG(f"block_{node_type.__name__.lower()}_{dtype.to_string()}")
    sdfg.add_array("X", [B, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("Y", [B, N], dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("lines", {"b": "0:B"}, schedule=dtypes.ScheduleType.GPU_Device)
    node = node_type("transform", **kwargs)
    state.add_memlet_path(state.add_read("X"), entry, node, dst_conn="_inp", memlet=dace.Memlet("X[b, 0:N]"))
    state.add_memlet_path(node, exit_node, state.add_write("Y"), src_conn="_out", memlet=dace.Memlet("Y[b, 0:N]"))
    return sdfg, state, node


def plane_kernel(**kwargs):
    """A 2-D transform per kernel iteration: more than one axis, which the block refuses."""
    sdfg = dace.SDFG("block_fft_plane")
    sdfg.add_array("X", [B, N, M], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("Y", [B, N, M], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("planes", {"b": "0:B"}, schedule=dtypes.ScheduleType.GPU_Device)
    node = FFT("transform", **kwargs)
    state.add_memlet_path(state.add_read("X"), entry, node, dst_conn="_inp", memlet=dace.Memlet("X[b, 0:N, 0:M]"))
    state.add_memlet_path(node, exit_node, state.add_write("Y"), src_conn="_out", memlet=dace.Memlet("Y[b, 0:N, 0:M]"))
    return state, node


def in_place_row_kernel():
    """A ``GPU_Device`` map over ``b`` transforming row ``X[b]`` into itself through two views, as QE's vexx inverts
    a band's grid in place: the views hide from the block lowering that the transform is in place."""
    inner = dace.SDFG("in_place_row")
    inner.add_array("x", [B, N], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    inner.add_symbol("b", dace.int64)
    inner.add_view("xin", [N], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    inner.add_view("xout", [N], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    state = inner.add_state()
    node = IFFT("transform")
    node.factor = 1 / N
    node.implementation = BLOCK
    xin, xout = state.add_access("xin"), state.add_access("xout")
    state.add_edge(state.add_read("x"), None, xin, "views", dace.Memlet("x[b, 0:N]"))
    state.add_edge(xin, None, node, "_inp", dace.Memlet("xin[0:N]"))
    state.add_edge(node, "_out", xout, None, dace.Memlet("xout[0:N]"))
    state.add_edge(xout, "views", state.add_write("x"), None, dace.Memlet("x[b, 0:N]"))

    sdfg = dace.SDFG("block_ifft_in_place_rows")
    sdfg.add_array("X", [B, N], dace.complex128, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"b": "0:B"}, schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {"x"}, {"x"}, {"N": "N", "B": "B", "b": "b"})
    state.add_memlet_path(state.add_read("X"), entry, nsdfg, dst_conn="x", memlet=dace.Memlet("X[0:B, 0:N]"))
    state.add_memlet_path(nsdfg, exit_node, state.add_write("X"), src_conn="x", memlet=dace.Memlet("X[0:B, 0:N]"))
    return sdfg


def block_tasklet(sdfg: dace.SDFG) -> nodes.Tasklet:
    """Expand the node and return its one tasklet, checking it sits in a 256-lane map inside the device map."""
    sdfg.expand_library_nodes()
    schedules = [n.map.schedule for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert schedules.count(dtypes.ScheduleType.GPU_Device) == 1
    lanes = [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.map.range.size()[0] for lane in lanes] == [256]
    (tasklet,) = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    return tasklet


def test_a_one_axis_fft_in_a_kernel_offers_the_block_lowering():
    sdfg, state, node = batched_kernel(FFT)
    assert gpu_block_implementation(node, state, sdfg) == BLOCK
    assert not is_kernel_refused_libnode(node, state)


def test_the_block_fft_calls_the_device_transform_once_per_block():
    sdfg, state, node = batched_kernel(IFFT, dace.complex64)
    node.factor = 1 / N
    node.implementation = BLOCK
    code = block_tasklet(sdfg).code.as_string
    assert "dace::block_fft::transform<dace::complex64>(__in, __out, N, 1," in code
    assert ", true, (float)(" in code and ", 256);" in code


def test_a_multi_axis_or_half_spectrum_fft_stays_device_wide():
    state, node = plane_kernel()
    assert gpu_block_implementation(node, state, state.sdfg) is None
    assert is_kernel_refused_libnode(node, state)
    sdfg, state, node = batched_kernel(FFT, half_spectrum=True)
    assert gpu_block_implementation(node, state, sdfg) is None


def test_an_explicit_block_fft_of_a_refused_shape_raises():
    state, node = plane_kernel()
    node.implementation = BLOCK
    with pytest.raises(ValueError, match="more than one axis"):
        state.sdfg.expand_library_nodes()


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 1, 12, 37])
@pytest.mark.parametrize("dtype", [dace.complex64, dace.complex128])
@pytest.mark.parametrize("inverse", [False, True])
def test_batched_block_fft_matches_numpy(n: int, dtype: dace.typeclass, inverse: bool):
    import cupy

    sdfg, state, node = batched_kernel(IFFT if inverse else FFT, dtype)
    if inverse:
        node.factor = 1 / N
    node.implementation = BLOCK
    rows = 7
    rng = np.random.default_rng(n)
    x = (rng.uniform(-1, 1, (rows, n)) + 1j * rng.uniform(-1, 1, (rows, n))).astype(dtype.type)
    y = cupy.zeros((rows, n), dtype=dtype.type)
    sdfg(X=cupy.asarray(x), Y=y, B=rows, N=n)
    expected = np.fft.ifft(x.astype(np.complex128)) if inverse else np.fft.fft(x.astype(np.complex128))
    tolerance = 1e-4 if dtype == dace.complex64 else 1e-12
    np.testing.assert_allclose(y.get(), expected, rtol=tolerance, atol=tolerance * n)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 2048])
def test_an_in_place_block_ifft_longer_than_the_block_matches_numpy(n: int):
    import cupy

    rows = 5
    rng = np.random.default_rng(n)
    x = rng.uniform(-1, 1, (rows, n)) + 1j * rng.uniform(-1, 1, (rows, n))
    out = cupy.asarray(x)
    in_place_row_kernel()(X=out, B=rows, N=n)
    np.testing.assert_allclose(out.get(), np.fft.ifft(x), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_a_one_axis_fft_in_a_kernel_offers_the_block_lowering()
    test_the_block_fft_calls_the_device_transform_once_per_block()
    test_a_multi_axis_or_half_spectrum_fft_stays_device_wide()
    test_an_explicit_block_fft_of_a_refused_shape_raises()
    for size in (64, 1, 12, 37):
        for element in (dace.complex64, dace.complex128):
            for backward in (False, True):
                test_batched_block_fft_matches_numpy(size, element, backward)
    for size in (64, 2048):
        test_an_in_place_block_ifft_longer_than_the_block_matches_numpy(size)
