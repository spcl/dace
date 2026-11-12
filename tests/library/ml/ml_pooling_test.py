# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pure expansions of dace.libraries.ml pooling nodes (max/avg pool and the max-pool backward)."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import dace  # noqa: E402
import dace.libraries.ml as ml  # noqa: E402

_MAXPOOL = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
_AVGPOOL = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}


def _out_size(h, k, pad, dil, stride, ceil_mode):
    base = h + 2 * pad - dil * (k - 1) - 1
    if not ceil_mode:
        return base // stride + 1
    out = -(-base // stride) + 1
    if (out - 1) * stride >= h + pad:
        out -= 1
    return out


def _build_maxpool(xshape, k, s, p, d, ceil_mode, want_idx, nsym=None, dtype=dace.float32):
    nd = len(xshape) - 2
    out_sp = [_out_size(xshape[2 + i], k, p, d, s, ceil_mode) for i in range(nd)]
    n = nsym if nsym is not None else xshape[0]
    yshape = [n, xshape[1]] + out_sp
    sdfg = dace.SDFG("maxpool")
    sdfg.add_array("x", [n] + list(xshape[1:]), dtype)
    sdfg.add_array("y", yshape, dtype)
    if want_idx:
        sdfg.add_array("idx", yshape, dace.int64)
    st = sdfg.add_state()
    node = ml.MaxPool(
        "mp", kernel_shape=[k] * nd, strides=[s] * nd, pads=[p] * nd, dilations=[d] * nd, ceil_mode=ceil_mode
    )
    st.add_node(node)
    st.add_edge(st.add_read("x"), None, node, "_x", dace.Memlet("x"))
    st.add_edge(node, "_y", st.add_write("y"), None, dace.Memlet("y"))
    if want_idx:
        node.add_out_connector("_indices")
        st.add_edge(node, "_indices", st.add_write("idx"), None, dace.Memlet("idx"))
    return sdfg, [xshape[0], xshape[1]] + out_sp


def _run_maxpool(xshape, k, s, p, d, ceil_mode, want_idx=True, nsym=None, **sym):
    sdfg, yshape = _build_maxpool(xshape, k, s, p, d, ceil_mode, want_idx, nsym=nsym)
    nd = len(xshape) - 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    args = dict(x=x, y=y, **sym)
    if want_idx:
        idx = np.zeros(yshape, np.int64)
        args["idx"] = idx
    sdfg(**args)
    yt, it = _MAXPOOL[nd](torch.from_numpy(x), k, s, p, d, ceil_mode=ceil_mode, return_indices=True)
    assert np.allclose(y, yt.numpy(), atol=1e-5), np.max(np.abs(y - yt.numpy()))
    if want_idx:
        assert np.array_equal(idx, it.numpy())


@pytest.mark.parametrize("want_idx", [True, False])
def test_max_pool_2d(want_idx):
    _run_maxpool((2, 3, 8, 10), 2, 2, 0, 1, False, want_idx)


def test_max_pool_padding_dilation():
    _run_maxpool((2, 3, 10, 8), 3, 2, 1, 2, False)


def test_max_pool_ceil_mode():
    _run_maxpool((2, 3, 9, 7), 3, 2, 1, 1, True)


def test_max_pool_1d():
    _run_maxpool((2, 4, 15), 3, 2, 1, 1, False)


def test_max_pool_3d():
    _run_maxpool((2, 2, 7, 8, 9), 2, 2, 1, 1, False)


def test_max_pool_symbolic_batch():
    N = dace.symbol("N")
    _run_maxpool((4, 3, 9, 7), 3, 2, 1, 1, False, nsym=N, N=4)


def _run_avgpool(xshape, k, s, p, ceil_mode, cip, div):
    nd = len(xshape) - 2
    out_sp = [_out_size(xshape[2 + i], k, p, 1, s, ceil_mode) for i in range(nd)]
    yshape = [xshape[0], xshape[1]] + out_sp
    sdfg = dace.SDFG("avgpool")
    sdfg.add_array("x", list(xshape), dace.float32)
    sdfg.add_array("y", yshape, dace.float32)
    st = sdfg.add_state()
    node = ml.AvgPool(
        "ap",
        kernel_shape=[k] * nd,
        strides=[s] * nd,
        pads=[p] * nd,
        ceil_mode=ceil_mode,
        count_include_pad=cip,
        divisor_override=(div or 0),
    )
    st.add_node(node)
    st.add_edge(st.add_read("x"), None, node, "_x", dace.Memlet("x"))
    st.add_edge(node, "_y", st.add_write("y"), None, dace.Memlet("y"))
    rng = np.random.default_rng(1)
    x = rng.standard_normal(xshape).astype(np.float32)
    y = np.zeros(yshape, np.float32)
    sdfg(x=x, y=y)
    kw = dict(ceil_mode=ceil_mode, count_include_pad=cip)
    if nd > 1:
        kw["divisor_override"] = div
    yt = _AVGPOOL[nd](torch.from_numpy(x), k, s, p, **kw)
    assert np.allclose(y, yt.numpy(), atol=1e-5), np.max(np.abs(y - yt.numpy()))


@pytest.mark.parametrize("cip", [True, False])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_avg_pool_count_and_ceil(cip, ceil_mode):
    _run_avgpool((2, 3, 9, 7), 3, 2, 1, ceil_mode, cip, None)


def test_avg_pool_divisor_override():
    _run_avgpool((2, 3, 9, 7), 2, 2, 1, False, True, 5)


def test_avg_pool_1d():
    _run_avgpool((2, 4, 15), 3, 2, 1, False, True, None)


def test_avg_pool_3d():
    _run_avgpool((2, 2, 7, 8, 9), 2, 2, 1, False, True, None)


@pytest.mark.parametrize("shape,k,s,p,d", [((2, 3, 9, 7), 3, 2, 1, 1), ((2, 2, 7, 8, 9), 2, 2, 1, 1)])
def test_max_pool_backward(shape, k, s, p, d):
    nd = len(shape) - 2
    rng = np.random.default_rng(2)
    xt = torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).requires_grad_(True)
    yt, it = _MAXPOOL[nd](xt, k, s, p, d, return_indices=True)
    g = torch.from_numpy(rng.standard_normal(tuple(yt.shape)).astype(np.float32))
    dx_t = torch.autograd.grad(yt, xt, g)[0]

    sdfg = dace.SDFG("maxpool_bw")
    sdfg.add_array("dy", list(yt.shape), dace.float32)
    sdfg.add_array("idx", list(it.shape), dace.int64)
    sdfg.add_array("dx", list(shape), dace.float32)
    st = sdfg.add_state()
    node = ml.MaxPoolBackward("mpb")
    st.add_node(node)
    st.add_edge(st.add_read("dy"), None, node, "_dy", dace.Memlet("dy"))
    st.add_edge(st.add_read("idx"), None, node, "_indices", dace.Memlet("idx"))
    st.add_edge(node, "_dx", st.add_write("dx"), None, dace.Memlet("dx"))
    dx = np.zeros(shape, np.float32)
    sdfg(dy=g.numpy().copy(), idx=it.numpy().copy(), dx=dx)
    assert np.allclose(dx, dx_t.numpy(), atol=1e-5), np.max(np.abs(dx - dx_t.numpy()))


def test_save_roundtrip(tmp_path):
    sdfg, _ = _build_maxpool((2, 3, 9, 7), 3, 2, 1, 2, True, True)
    fname = str(tmp_path / "mp.sdfg")
    sdfg.save(fname)
    sdfg2 = dace.SDFG.from_file(fname)
    node = next(n for s in sdfg2.states() for n in s.nodes() if isinstance(n, ml.MaxPool))
    assert list(node.kernel_shape) == [3, 3]
    assert list(node.strides) == [2, 2]
    assert list(node.dilations) == [2, 2]
    assert node.ceil_mode is True


def test_validate_rejects_wrong_output_shape():
    sdfg = dace.SDFG("bad")
    sdfg.add_array("x", [2, 3, 8, 8], dace.float32)
    sdfg.add_array("y", [2, 3, 5, 5], dace.float32)  # wrong: stride-2 kernel-2 gives 4x4
    st = sdfg.add_state()
    node = ml.MaxPool("mp", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0], dilations=[1, 1])
    st.add_node(node)
    st.add_edge(st.add_read("x"), None, node, "_x", dace.Memlet("x"))
    st.add_edge(node, "_y", st.add_write("y"), None, dace.Memlet("y"))
    with pytest.raises((ValueError, Exception)):
        sdfg.expand_library_nodes()


if __name__ == "__main__":
    test_max_pool_2d(True)
    test_max_pool_2d(False)
    test_max_pool_padding_dilation()
    test_max_pool_ceil_mode()
    test_max_pool_1d()
    test_max_pool_3d()
    test_max_pool_symbolic_batch()
    for _cip in (True, False):
        for _cm in (False, True):
            test_avg_pool_count_and_ceil(_cip, _cm)
    test_avg_pool_divisor_override()
    test_avg_pool_1d()
    test_avg_pool_3d()
    test_max_pool_backward((2, 3, 9, 7), 3, 2, 1, 1)
    test_max_pool_backward((2, 2, 7, 8, 9), 2, 2, 1, 1)
    print("ok")
