# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for dace.libraries.ml.FuseConvolutionActivation (folding a ReLU map into the convolution feeding it)."""

import numpy as np
import pytest

import dace
import dace.libraries.ml as ml
from dace.sdfg import nodes as nd

N = dace.symbol("N")


def _conv_relu_sdfg(code: str = "__out = max(__in, 0)", out_channels: int = 4, offset: int = 0, extra_reader=False):
    """``conv -> t -> map{code} -> y[:, offset:offset+4]``; optionally a second reader of the pre-activation ``t``."""
    sdfg = dace.SDFG("conv_relu")
    sdfg.add_array("x", [N, 3, 8, 8], dace.float32)
    sdfg.add_array("w", [4, 3, 3, 3], dace.float32)
    sdfg.add_array("b", [4], dace.float32)
    sdfg.add_transient("t", [N, 4, 8, 8], dace.float32)
    sdfg.add_array("y", [N, out_channels, 8, 8], dace.float32)
    state = sdfg.add_state()
    conv = ml.Convolution("conv", strides=[1, 1], pads=[1, 1], dilations=[1, 1])
    conv.add_in_connector("_b")
    state.add_node(conv)
    state.add_edge(state.add_read("x"), None, conv, "_x", dace.Memlet("x"))
    state.add_edge(state.add_read("w"), None, conv, "_w", dace.Memlet("w"))
    state.add_edge(state.add_read("b"), None, conv, "_b", dace.Memlet("b"))
    t = state.add_access("t")
    state.add_edge(conv, "_y", t, None, dace.Memlet("t"))
    params = {f"i{k}": f"0:{s}" for k, s in enumerate(sdfg.arrays["t"].shape)}
    state.add_mapped_tasklet(
        "relu",
        params,
        {"__in": dace.Memlet("t[i0, i1, i2, i3]")},
        code,
        {"__out": dace.Memlet(f"y[i0, {offset} + i1, i2, i3]")},
        input_nodes={"t": t},
        external_edges=True,
    )
    if extra_reader:
        sdfg.add_array("z", [N, 4, 8, 8], dace.float32)
        state.add_nedge(t, state.add_write("z"), dace.Memlet("t"))
    return sdfg, conv


def _reference(x, w, b):
    import tests.library.ml.ml_convolution_test as conv_test

    return np.maximum(conv_test._ref_conv(x, w, (1, 1), (1, 1), (1, 1), 1, b), 0)


def _run(sdfg, out_channels=4, offset=0):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    y = np.zeros((2, out_channels, 8, 8), np.float32)
    sdfg(x=x, w=w, b=b, y=y, N=2)
    assert np.allclose(y[:, offset : offset + 4], _reference(x, w, b), atol=1e-3)
    return y


def _maps(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]


@pytest.mark.parametrize(
    "code", ["__out = max(__in, 0)", "__out = dace.float32((__in if ((__in != __in) or (__in > 0)) else 0))"]
)
def test_fuses_relu(code):
    sdfg, conv = _conv_relu_sdfg(code)
    assert ml.fuse_activations(sdfg) == 1
    assert conv.activation == "relu" and not _maps(sdfg) and "t" not in sdfg.arrays
    _run(sdfg)


def test_fuses_into_slice():
    """The map writes a channel slice of a larger container (a concatenation): the convolution then writes there."""
    sdfg, conv = _conv_relu_sdfg(out_channels=8, offset=4)
    assert ml.fuse_activations(sdfg) == 1
    state = next(iter(sdfg.states()))
    (edge,) = state.out_edges(conv)
    assert edge.data.data == "y" and str(edge.data.subset) == "0:N, 4:8, 0:8, 0:8"
    y = _run(sdfg, out_channels=8, offset=4)
    assert not y[:, :4].any()


def test_keeps_other_activations():
    sdfg, conv = _conv_relu_sdfg("__out = __in * 2")
    assert ml.fuse_activations(sdfg) == 0
    assert conv.activation == "none" and len(_maps(sdfg)) == 1


def test_keeps_shared_pre_activation():
    """When something else reads the convolution's result the ReLU must stay a separate map."""
    sdfg, conv = _conv_relu_sdfg(extra_reader=True)
    assert ml.fuse_activations(sdfg) == 0
    assert conv.activation == "none" and len(_maps(sdfg)) == 1


def test_save_roundtrip(tmp_path):
    sdfg, conv = _conv_relu_sdfg()
    ml.fuse_activations(sdfg)
    path = str(tmp_path / "fused.sdfg")
    sdfg.save(path)
    loaded = dace.SDFG.from_file(path)
    node = next(n for n, _ in loaded.all_nodes_recursive() if isinstance(n, ml.Convolution))
    assert node.activation == "relu"


if __name__ == "__main__":
    test_fuses_relu("__out = max(__in, 0)")
    test_fuses_into_slice()
    test_keeps_other_activations()
    test_keeps_shared_pre_activation()
