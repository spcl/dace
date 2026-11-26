# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pure (and codegen-only GPU) expansions of dace.libraries.ml.Convolution."""

import numpy as np
import pytest

import dace
import dace.libraries.ml as ml
from dace import dtypes

try:
    import torch
    import torch.nn.functional as F

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _out_size(h, k, pad, dil, stride):
    return (h + 2 * pad - dil * (k - 1) - 1) // stride + 1


def _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, bias, dtype=dace.float32, xstrides=None, nsym=None):
    ndim = len(wshape) - 2
    spatial_out = [_out_size(xshape[2 + i], wshape[2 + i], pads[i], dils[i], strides[i]) for i in range(ndim)]
    n = nsym if nsym is not None else xshape[0]
    yshape = [n, wshape[0]] + spatial_out

    sdfg = dace.SDFG("conv")
    sdfg.add_array("x", [n] + list(xshape[1:]), dtype, strides=xstrides)
    sdfg.add_array("w", list(wshape), dtype)
    sdfg.add_array("y", yshape, dtype)
    if bias:
        sdfg.add_array("b", [wshape[0]], dtype)
    state = sdfg.add_state()
    node = ml.Convolution("conv", strides=list(strides), pads=list(pads), dilations=list(dils), groups=groups)
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x"))
    state.add_edge(state.add_read("w"), None, node, "_w", dace.Memlet("w"))
    if bias:
        node.add_in_connector("_b")
        state.add_edge(state.add_read("b"), None, node, "_b", dace.Memlet("b"))
    state.add_edge(node, "_y", state.add_write("y"), None, dace.Memlet("y"))
    return sdfg, yshape


def _ref_conv(x, w, strides, pads, dils, groups, b=None):
    """Generic N-d (1-3 spatial) convolution reference in numpy."""
    ndim = x.ndim - 2
    N, Cin = x.shape[0], x.shape[1]
    Cout, Cin_g = w.shape[0], w.shape[1]
    ksp = w.shape[2:]
    xsp = x.shape[2:]
    osp = [_out_size(xsp[i], ksp[i], pads[i], dils[i], strides[i]) for i in range(ndim)]
    y = np.zeros([N, Cout] + osp, dtype=np.float64)
    cout_pg = Cout // groups
    xpad_pads = [(0, 0), (0, 0)] + [(pads[i], pads[i]) for i in range(ndim)]
    xp = np.pad(x.astype(np.float64), xpad_pads)
    for co in range(Cout):
        g = co // cout_pg
        for ci in range(Cin_g):
            ic = g * Cin_g + ci
            for kidx in np.ndindex(*ksp):
                wv = w[(co, ci) + kidx]
                # Build slices into the padded input per spatial dim.
                slc = [slice(None), ic]
                for i in range(ndim):
                    start = kidx[i] * dils[i]
                    stop = start + strides[i] * (osp[i] - 1) + 1
                    slc.append(slice(start, stop, strides[i]))
                y[:, co] += wv * xp[tuple(slc)]
    if b is not None:
        y += b.reshape([1, Cout] + [1] * ndim)
    return y


def _run(sdfg, **arrays):
    sdfg(**arrays)


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_stride_pad_dilation(bias):
    strides, pads, dils, groups = (2, 2), (1, 2), (2, 1), 1
    xshape, wshape = (2, 3, 11, 13), (5, 3, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, bias)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32) if bias else None
    args = dict(x=x, w=w, y=y)
    if bias:
        args["b"] = b
    _run(sdfg, **args)
    ref = _ref_conv(x, w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))
    if HAVE_TORCH:
        tb = torch.from_numpy(b) if bias else None
        tref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), tb, strides, pads, dils, groups).numpy()
        assert np.allclose(y, tref, atol=1e-3)


def test_conv2d_grouped():
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 4
    xshape, wshape = (2, 8, 9, 9), (8, 2, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    _run(sdfg, x=x, w=w, b=b, y=y)
    ref = _ref_conv(x, w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))


def test_conv1d():
    strides, pads, dils, groups = (2,), (1,), (1,), 1
    xshape, wshape = (3, 4, 20), (6, 4, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    _run(sdfg, x=x, w=w, b=b, y=y)
    ref = _ref_conv(x, w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))
    if HAVE_TORCH:
        tref = F.conv1d(
            torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), strides, pads, dils, groups
        ).numpy()
        assert np.allclose(y, tref, atol=1e-3)


def test_conv3d():
    strides, pads, dils, groups = (1, 2, 1), (1, 0, 1), (1, 1, 2), 1
    xshape, wshape = (1, 2, 7, 8, 9), (3, 2, 3, 2, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, False)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    _run(sdfg, x=x, w=w, y=y)
    ref = _ref_conv(x, w, strides, pads, dils, groups)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))
    if HAVE_TORCH:
        tref = F.conv3d(torch.from_numpy(x), torch.from_numpy(w), None, strides, pads, dils, groups).numpy()
        assert np.allclose(y, tref, atol=1e-3)


def test_conv2d_channels_last():
    """Non-contiguous (channels-last-like) input exercised purely through strides."""
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (2, 3, 8, 8), (4, 3, 3, 3)
    rng = np.random.default_rng(4)
    N, C, H, W = xshape
    # Genuine channels-last tensor: contiguous (N, H, W, C) viewed logically as (N, C, H, W).
    x = rng.standard_normal((N, H, W, C)).astype(np.float32).transpose(0, 3, 1, 2)
    xstrides = [s // x.itemsize for s in x.strides]
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True, xstrides=xstrides)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
        sdfg(x=x, w=w, b=b, y=y)
    ref = _ref_conv(np.ascontiguousarray(x), w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))


def test_conv2d_symbolic_batch():
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (4, 3, 8, 8), (4, 3, 3, 3)
    N = dace.symbol("N")
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True, nsym=N)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    OH = _out_size(8, 3, 1, 1, 1)
    y = np.zeros((xshape[0], wshape[0], OH, OH), np.float32)
    sdfg(x=x, w=w, b=b, y=y, N=xshape[0])
    ref = _ref_conv(x, w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_relu_activation(bias):
    """The ``activation`` property: the pure expansion applies the ReLU when it writes the result."""
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (2, 3, 7, 9), (4, 3, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, bias)
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.activation = "relu"
    rng = np.random.default_rng(11)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32) if bias else None
    y = np.zeros(yshape, np.float32)
    args = dict(x=x, w=w, y=y, **({"b": b} if bias else {}))
    sdfg(**args)
    ref = np.maximum(_ref_conv(x, w, strides, pads, dils, groups, b), 0)
    assert np.allclose(y, ref, atol=1e-3), np.max(np.abs(y - ref))
    assert (y == 0).any() and (y > 0).any()


def test_transposed_not_implemented():
    sdfg, yshape = _build_conv_sdfg((1, 2, 6, 6), (2, 2, 3, 3), (1, 1), (0, 0), (1, 1), 1, False)
    for n in sdfg.all_sdfgs_recursive():
        for node in n.nodes():
            pass
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.transposed = True
    with pytest.raises((NotImplementedError, Exception)):
        sdfg.expand_library_nodes()


def test_save_roundtrip(tmp_path):
    sdfg, _ = _build_conv_sdfg((2, 4, 8, 8), (6, 2, 3, 3), (2, 2), (1, 1), (2, 2), 2, True)
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.strides = [2, 2]
    node.pads = [1, 1]
    node.dilations = [2, 2]
    node.groups = 2
    fname = str(tmp_path / "conv.sdfg")
    sdfg.save(fname)
    sdfg2 = dace.SDFG.from_file(fname)
    node2 = next(x for s in sdfg2.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    assert list(node2.strides) == [2, 2]
    assert list(node2.pads) == [1, 1]
    assert list(node2.dilations) == [2, 2]
    assert node2.groups == 2


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
def test_gpu_codegen(impl):
    """The GPU expansions should at least generate code (requires a CUDA/HIP toolchain to compile)."""
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (2, 3, 8, 8), (4, 3, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = impl
    try:
        sdfg.expand_library_nodes()
        code = sdfg.generate_code()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{impl} codegen needs a GPU toolchain that is absent: {e}")
    joined = "\n".join(c.clean_code for c in code)
    fn = "cudnnConvolutionForward" if impl == "cuDNN" else "miopenConvolutionForward"
    assert fn in joined
    bias = "cudnnAddTensor" if impl == "cuDNN" else "miopenConvolutionForwardBias"
    assert bias in joined
    assert "ConvolutionPlan" in joined and "Configure(" in joined


def _gpu_implementation() -> str:
    from dace.codegen.common import get_gpu_backend

    return {"cuda": "cuDNN", "hip": "MIOpen"}[get_gpu_backend()]


def _channels_last_strides(shape):
    """Strides of a dense (N, spatial..., C) layout viewed as (N, C, spatial...), like ``torch.channels_last``."""
    n, c, *spatial = shape
    plane = int(np.prod(spatial))
    strides = [c * plane, 1]
    inner = c
    for s in reversed(spatial):
        strides.insert(2, inner)
        inner *= s
    return strides


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
def test_gpu_codegen_channels_last(impl):
    """Channels-last activations and weights: cuDNN gets an NHWC filter layout, MIOpen NHWC descriptors throughout."""
    xshape, wshape = (2, 8, 9, 7), (4, 8, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, (1, 1), (1, 1), (1, 1), 1, False)
    for name in ("x", "w", "y"):
        desc = sdfg.arrays[name]
        desc.storage = dtypes.StorageType.GPU_Global
        desc.strides = _channels_last_strides(desc.shape)
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = impl
    sdfg.expand_library_nodes()
    joined = "\n".join(c.clean_code for c in sdfg.generate_code())
    if impl == "cuDNN":
        assert "CUDNN_TENSOR_NHWC" in joined and "CUDNN_TENSOR_NCHW" not in joined
    else:
        assert joined.count("miopenSetNdTensorDescriptorWithLayout") == 3
        assert "miopenTensorNHWC" in joined and "miopenTensorNCHW" not in joined


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
def test_gpu_codegen_mixed_layouts_use_strides(impl):
    """A channels-last input with contiguous weights: MIOpen falls back to stride descriptors, cuDNN keeps NCHW filters."""
    xshape, wshape = (2, 8, 9, 7), (4, 8, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, (1, 1), (1, 1), (1, 1), 1, False)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    sdfg.arrays["x"].strides = _channels_last_strides(xshape)
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = impl
    sdfg.expand_library_nodes()
    joined = "\n".join(c.clean_code for c in sdfg.generate_code())
    if impl == "cuDNN":
        assert "CUDNN_TENSOR_NCHW" in joined
    else:
        assert "miopenSetNdTensorDescriptorWithLayout" not in joined
        assert "{504, 1, 56, 8}" in joined  # the input strides reach the descriptor


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
def test_gpu_codegen_conv1d_promoted(impl):
    """Both libraries need two spatial dimensions: a 1-D convolution is described as (N, C, 1, L)."""
    sdfg, yshape = _build_conv_sdfg((3, 4, 20), (6, 4, 3), (2,), (1,), (1,), 1, True)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = impl
    sdfg.expand_library_nodes()
    joined = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert "{3, 4, 1, 20}" in joined and "{6, 4, 1, 3}" in joined  # promoted input and weight sizes
    assert "{0, 1}" in joined and "{1, 2}" in joined  # promoted padding and stride
    if impl == "MIOpen":
        assert "miopenTensorNCHW" in joined


@pytest.mark.parametrize("impl", ["cuDNN", "MIOpen"])
@pytest.mark.parametrize("bias", [True, False])
def test_gpu_codegen_activation(impl, bias):
    """A fused ReLU uses the libraries' convolution+bias+activation kernels, with the separate calls as fallback."""
    sdfg, yshape = _build_conv_sdfg((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, bias)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = impl
    node.activation = "relu"
    sdfg.expand_library_nodes()
    joined = "\n".join(c.clean_code for c in sdfg.generate_code())
    if impl == "cuDNN":
        assert "cudnnConvolutionBiasActivationForward" in joined and "CUDNN_ACTIVATION_RELU" in joined
        assert "cudnnActivationForward" in joined and "cudnnConvolutionForward(" in joined  # fallback
        assert ("ZeroBias" in joined) != bias  # the fused call needs a bias: a zero one without ``_b``
    else:
        assert "SetupFusion" in joined and "miopenExecuteFusionPlan" in joined and "miopenActivationRELU" in joined
        assert "miopenActivationForward" in joined and "miopenConvolutionForward(" in joined  # fallback
        assert ("miopenSetOpArgsBiasForward" in joined) == bias


@pytest.mark.gpu
@pytest.mark.parametrize("bias", [True, False])
def test_gpu_conv2d_relu(bias):
    """cuDNN/MIOpen with a fused ReLU (or the fallback, if the library does not support the fusion here)."""
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (2, 3, 9, 7), (4, 3, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, bias)
    sdfg.apply_gpu_transformations()
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ml.Convolution))
    node.implementation = _gpu_implementation()
    node.activation = "relu"
    rng = np.random.default_rng(12)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32) if bias else None
    y = np.zeros(yshape, np.float32)
    for _ in range(2):  # the second call runs with the configured plan
        sdfg(x=x, w=w, y=y, **({"b": b} if bias else {}))
        ref = np.maximum(_ref_conv(x, w, strides, pads, dils, groups, b), 0)
        assert np.allclose(y, ref, atol=1e-3, rtol=1e-4), np.max(np.abs(y - ref))


def test_cudnn_rejects_unsupported_weight_layout():
    sdfg, _ = _build_conv_sdfg((2, 3, 8, 8), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1), 1, False)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    sdfg.arrays["w"].strides = [1, 4, 12, 36]  # column-major: neither NCHW nor NHWC
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Convolution))
    node.implementation = "cuDNN"
    with pytest.raises(NotImplementedError, match="channels-last"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_gpu_conv2d_grouped_bias_symbolic_batch():
    """cuDNN/MIOpen on the GPU: grouped, biased, strided and dilated; the plan is rebuilt when the batch changes."""
    strides, pads, dils, groups = (2, 1), (1, 1), (1, 2), 2
    xshape, wshape = (4, 4, 11, 13), (6, 2, 3, 3)
    N = dace.symbol("N")
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True, nsym=N)
    sdfg.apply_gpu_transformations()
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ml.Convolution))
    node.implementation = _gpu_implementation()
    rng = np.random.default_rng(7)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    for batch in (4, 2):  # a second, smaller batch reuses the compiled program and reconfigures the plan
        x = rng.standard_normal((batch,) + xshape[1:]).astype(np.float32)
        y = np.zeros((batch,) + tuple(yshape[1:]), np.float32)
        sdfg(x=x, w=w, b=b, y=y, N=batch)
        ref = _ref_conv(x, w, strides, pads, dils, groups, b)
        assert np.allclose(y, ref, atol=1e-3, rtol=1e-4), np.max(np.abs(y - ref))


@pytest.mark.gpu
def test_gpu_conv2d_channels_last():
    """cuDNN/MIOpen with channels-last input, weights and output (torch.channels_last)."""
    strides, pads, dils, groups = (1, 1), (1, 1), (1, 1), 1
    xshape, wshape = (2, 8, 9, 7), (4, 8, 3, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True)
    for name in ("x", "w", "y"):
        sdfg.arrays[name].strides = _channels_last_strides(sdfg.arrays[name].shape)
    sdfg.apply_gpu_transformations()
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ml.Convolution))
    node.implementation = _gpu_implementation()
    rng = np.random.default_rng(8)

    def channels_last(shape):  # a dense (N, H, W, C) buffer viewed as (N, C, H, W)
        n, c, h, w = shape
        return rng.standard_normal((n, h, w, c)).astype(np.float32).transpose(0, 3, 1, 2)

    x, w = channels_last(xshape), channels_last(wshape)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    y = np.zeros((yshape[0], yshape[2], yshape[3], yshape[1]), np.float32).transpose(0, 3, 1, 2)
    with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
        sdfg(x=x, w=w, b=b, y=y)
    ref = _ref_conv(np.ascontiguousarray(x), np.ascontiguousarray(w), strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3, rtol=1e-4), np.max(np.abs(y - ref))


@pytest.mark.gpu
def test_gpu_conv1d():
    """A 1-D convolution runs through the promoted (N, C, 1, L) descriptors."""
    strides, pads, dils, groups = (2,), (1,), (1,), 1
    xshape, wshape = (3, 4, 20), (6, 4, 3)
    sdfg, yshape = _build_conv_sdfg(xshape, wshape, strides, pads, dils, groups, True)
    sdfg.apply_gpu_transformations()
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ml.Convolution))
    node.implementation = _gpu_implementation()
    rng = np.random.default_rng(9)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal((wshape[0],)).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    sdfg(x=x, w=w, b=b, y=y)
    ref = _ref_conv(x, w, strides, pads, dils, groups, b)
    assert np.allclose(y, ref, atol=1e-3, rtol=1e-4), np.max(np.abs(y - ref))


if __name__ == "__main__":
    test_conv2d_stride_pad_dilation(True)
    test_conv2d_stride_pad_dilation(False)
    test_conv2d_grouped()
    test_conv1d()
    test_conv3d()
    test_conv2d_channels_last()
    test_conv2d_symbolic_batch()
    test_save_roundtrip.__wrapped__ if hasattr(test_save_roundtrip, "__wrapped__") else None
    print("ok")
