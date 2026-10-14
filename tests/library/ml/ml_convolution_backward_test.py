# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pure expansions of the convolution backward nodes (dace.libraries.ml) vs torch.autograd.grad."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import dace  # noqa: E402
import dace.libraries.ml as ml  # noqa: E402

_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def _out_size(h, k, pad, dil, stride):
    return (h + 2 * pad - dil * (k - 1) - 1) // stride + 1


def _references(xshape, wshape, s, p, d, groups, dtype=np.float64, seed=0):
    nd = len(wshape) - 2
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal(xshape).astype(dtype)).requires_grad_(True)
    w = torch.from_numpy(rng.standard_normal(wshape).astype(dtype)).requires_grad_(True)
    y = _CONV[nd](x, w, None, [s] * nd, [p] * nd, [d] * nd, groups)
    g = torch.from_numpy(rng.standard_normal(tuple(y.shape)).astype(dtype))
    dx, dw = torch.autograd.grad(y, [x, w], g)
    return x, w, g, dx, dw, list(y.shape)


def _gpu_implementation() -> str:
    from dace.codegen.common import get_gpu_backend

    return {"cuda": "cuDNN", "hip": "MIOpen"}[get_gpu_backend()]


def _prepare(sdfg, node_type, gpu: bool):
    """For GPU runs: moves the data to the GPU and selects the vendor implementation of the node."""
    if gpu:
        sdfg.apply_gpu_transformations()
        node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, node_type))
        node.implementation = _gpu_implementation()


def _run_data(xshape, wshape, s, p, d, groups, nsym=None, gpu=False, **sym):
    nd = len(wshape) - 2
    dtype, atol = (dace.float32, 1e-3) if gpu else (dace.float64, 1e-8)
    x, w, g, dx_t, _, oshape = _references(xshape, wshape, s, p, d, groups, dtype.as_numpy_dtype())
    n = nsym if nsym is not None else xshape[0]
    sdfg = dace.SDFG("cbd")
    sdfg.add_array("dy", [n] + oshape[1:], dtype)
    sdfg.add_array("w", list(wshape), dtype)
    sdfg.add_array("dx", [n] + list(xshape[1:]), dtype)
    st = sdfg.add_state()
    node = ml.ConvolutionBackwardData("cbd", strides=[s] * nd, pads=[p] * nd, dilations=[d] * nd, groups=groups)
    st.add_node(node)
    st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
    st.add_edge(st.add_read("w"), None, node, "_w", dace.Memlet("w"))
    st.add_edge(node, "_dx", st.add_write("dx"), None, dace.Memlet("dx"))
    _prepare(sdfg, ml.ConvolutionBackwardData, gpu)
    dx = np.zeros(xshape, dtype.as_numpy_dtype())
    sdfg(dy=g.numpy().copy(), w=w.detach().numpy().copy(), dx=dx, **sym)
    assert np.allclose(dx, dx_t.numpy(), atol=atol), np.max(np.abs(dx - dx_t.numpy()))


def _run_filter(xshape, wshape, s, p, d, groups, nsym=None, gpu=False, **sym):
    nd = len(wshape) - 2
    dtype, atol = (dace.float32, 1e-3) if gpu else (dace.float64, 1e-8)
    x, w, g, _, dw_t, oshape = _references(xshape, wshape, s, p, d, groups, dtype.as_numpy_dtype())
    n = nsym if nsym is not None else xshape[0]
    sdfg = dace.SDFG("cbf")
    sdfg.add_array("x", [n] + list(xshape[1:]), dtype)
    sdfg.add_array("dy", [n] + oshape[1:], dtype)
    sdfg.add_array("dw", list(wshape), dtype)
    st = sdfg.add_state()
    node = ml.ConvolutionBackwardFilter("cbf", strides=[s] * nd, pads=[p] * nd, dilations=[d] * nd, groups=groups)
    st.add_node(node)
    st.add_edge(st.add_read("x"), None, node, "_x", dace.Memlet("x"))
    st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
    st.add_edge(node, "_dw", st.add_write("dw"), None, dace.Memlet("dw"))
    _prepare(sdfg, ml.ConvolutionBackwardFilter, gpu)
    dw = np.zeros(wshape, dtype.as_numpy_dtype())
    sdfg(x=x.detach().numpy().copy(), dy=g.numpy().copy(), dw=dw, **sym)
    assert np.allclose(dw, dw_t.numpy(), atol=atol), np.max(np.abs(dw - dw_t.numpy()))


_CASES = [
    ((2, 3, 11, 13), (5, 3, 3, 3), 2, 1, 1, 1),
    ((2, 3, 10, 8), (5, 3, 3, 3), 2, 2, 2, 1),
    ((2, 8, 9, 7), (8, 2, 3, 3), 1, 1, 1, 4),
    ((3, 4, 20), (6, 4, 3), 2, 1, 1, 1),
    ((2, 4, 10, 12), (6, 2, 3, 3), 1, 0, 1, 2),
]


@pytest.mark.parametrize("xshape,wshape,s,p,d,groups", _CASES)
def test_backward_data(xshape, wshape, s, p, d, groups):
    _run_data(xshape, wshape, s, p, d, groups)


@pytest.mark.parametrize("xshape,wshape,s,p,d,groups", _CASES)
def test_backward_filter(xshape, wshape, s, p, d, groups):
    _run_filter(xshape, wshape, s, p, d, groups)


def test_backward_data_symbolic_batch():
    N = dace.symbol("N")
    _run_data((4, 3, 10, 12), (5, 3, 3, 3), 2, 1, 1, 1, nsym=N, N=4)


def test_backward_filter_symbolic_batch():
    N = dace.symbol("N")
    _run_filter((4, 3, 10, 12), (5, 3, 3, 3), 2, 1, 1, 1, nsym=N, N=4)


def test_transposed_not_implemented():
    sdfg = dace.SDFG("tr")
    sdfg.add_array("dy", [2, 2, 4, 4], dace.float32)
    sdfg.add_array("w", [2, 2, 3, 3], dace.float32)
    sdfg.add_array("dx", [2, 2, 6, 6], dace.float32)
    st = sdfg.add_state()
    node = ml.ConvolutionBackwardData("cbd", strides=[1, 1], pads=[0, 0], dilations=[1, 1], groups=1, transposed=True)
    st.add_node(node)
    st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
    st.add_edge(st.add_read("w"), None, node, "_w", dace.Memlet("w"))
    st.add_edge(node, "_dx", st.add_write("dx"), None, dace.Memlet("dx"))
    with pytest.raises((NotImplementedError, Exception)):
        sdfg.expand_library_nodes()


_GPU_CALLS = {
    ("cuDNN", "data"): "cudnnConvolutionBackwardData",
    ("cuDNN", "filter"): "cudnnConvolutionBackwardFilter",
    ("MIOpen", "data"): "miopenConvolutionBackwardData",
    ("MIOpen", "filter"): "miopenConvolutionBackwardWeights",
}


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
@pytest.mark.parametrize("kind", ["data", "filter"])
def test_gpu_codegen(impl, kind):
    """The GPU expansions of the backward nodes generate code (compiling it requires a CUDA/HIP toolchain)."""
    N = dace.symbol("N")
    gpu = dace.StorageType.GPU_Global
    sdfg = dace.SDFG(f"cb_{kind}")
    st = sdfg.add_state()
    sdfg.add_array("dy", [N, 4, 8, 8], dace.float32, storage=gpu)
    if kind == "data":
        sdfg.add_array("w", [4, 3, 3, 3], dace.float32, storage=gpu)
        sdfg.add_array("dx", [N, 3, 8, 8], dace.float32, storage=gpu)
        node = ml.ConvolutionBackwardData("cbd", strides=[1, 1], pads=[1, 1], dilations=[1, 1])
        st.add_node(node)
        st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
        st.add_edge(st.add_read("w"), None, node, "_w", dace.Memlet("w"))
        st.add_edge(node, "_dx", st.add_write("dx"), None, dace.Memlet("dx"))
    else:
        sdfg.add_array("x", [N, 3, 8, 8], dace.float32, storage=gpu)
        sdfg.add_array("dw", [4, 3, 3, 3], dace.float32, storage=gpu)
        node = ml.ConvolutionBackwardFilter("cbf", strides=[1, 1], pads=[1, 1], dilations=[1, 1])
        st.add_node(node)
        st.add_edge(st.add_read("x"), None, node, "_x", dace.Memlet("x"))
        st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
        st.add_edge(node, "_dw", st.add_write("dw"), None, dace.Memlet("dw"))
    node.implementation = impl
    try:
        sdfg.expand_library_nodes()
        code = sdfg.generate_code()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{impl} codegen needs a GPU toolchain that is absent: {e}")
    joined = "\n".join(c.clean_code for c in code)
    assert _GPU_CALLS[impl, kind] in joined
    assert "Configure(" in joined and "(int)(N)" in joined  # the plan key holds the symbolic batch


@pytest.mark.gpu
@pytest.mark.parametrize("xshape,wshape,s,p,d,groups", _CASES[:3])
def test_gpu_backward_data(xshape, wshape, s, p, d, groups):
    _run_data(xshape, wshape, s, p, d, groups, gpu=True)


@pytest.mark.gpu
@pytest.mark.parametrize("xshape,wshape,s,p,d,groups", _CASES[:3])
def test_gpu_backward_filter(xshape, wshape, s, p, d, groups):
    _run_filter(xshape, wshape, s, p, d, groups, gpu=True)


@pytest.mark.gpu
def test_gpu_backward_symbolic_batch():
    N = dace.symbol("N")
    _run_data((4, 3, 10, 12), (5, 3, 3, 3), 2, 1, 1, 1, nsym=N, gpu=True, N=4)
    _run_filter((4, 3, 10, 12), (5, 3, 3, 3), 2, 1, 1, 1, nsym=N, gpu=True, N=4)


if __name__ == "__main__":
    for case in _CASES:
        test_backward_data(*case)
        test_backward_filter(*case)
    test_backward_data_symbolic_batch()
    test_backward_filter_symbolic_batch()
    print("ok")
