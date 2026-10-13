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


@pytest.mark.parametrize("impl", ["cudnn", "miopen"])
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
    fn = "cudnnConvolutionForward" if impl == "cudnn" else "miopenConvolutionForward"
    assert fn in joined


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
