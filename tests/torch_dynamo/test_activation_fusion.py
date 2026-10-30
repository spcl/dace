# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The opt-in ``fuse_activations=True`` option of ``dace.ml.to_sdfg``/``dace.ml.compile`` folds a ReLU that follows a
convolution into the convolution node (its ``activation`` property), removing the separate element-wise map. It is off
by default, applies to both the ahead-of-time and the ``torch.compile`` routes, and never changes results.
"""

import copy

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.libraries.torch  # noqa: E402
import dace.ml  # noqa: E402
from dace.sdfg import nodes as nd  # noqa: E402

N = dace.symbol("N")


def _plain():
    """A convolution directly followed by a ReLU: the ReLU folds into the convolution."""
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())


class _Fire(nn.Module):
    """A SqueezeNet-style fire module: a squeeze convolution, two expand convolutions (one bias-free), concatenated."""

    def __init__(self):
        super().__init__()
        self.squeeze = nn.Conv2d(8, 4, 1)
        self.e1 = nn.Conv2d(4, 6, 1)
        self.e3 = nn.Conv2d(4, 6, 3, padding=1, bias=False)

    def forward(self, x):
        s = torch.relu(self.squeeze(x))
        return torch.cat([torch.relu(self.e1(s)), torch.relu(self.e3(s))], 1)


class _Shared(nn.Module):
    """The pre-activation result is also an output, so the ReLU cannot be folded into the convolution."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return torch.relu(y), y


def _convs(sdfg):
    """The ATen convolution wrapper nodes of ``sdfg`` (``dace.libraries.torch.Convolution``)."""
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.libraries.torch.Convolution)]


def _maps(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]


@pytest.mark.torch
def test_default_keeps_relu_map():
    """Without the option the ReLU stays a separate map and the convolution's activation is 'none'."""
    torch.manual_seed(0)
    model = _plain().eval()
    sdfg = dace.ml.to_sdfg(model, dace.float32[N, 3, 7, 9])
    convs = _convs(sdfg)
    assert len(convs) == 1 and convs[0].activation == "none"
    assert _maps(sdfg)


@pytest.mark.torch
def test_fuse_plain_and_numerics():
    """With the option the plain model is one convolution node (activation 'relu', no maps) and matches torch."""
    torch.manual_seed(1)
    model = _plain().eval()
    sdfg = dace.ml.to_sdfg(model, dace.float32[N, 3, 7, 9], fuse_activations=True)
    convs = _convs(sdfg)
    assert len(convs) == 1 and convs[0].activation == "relu"
    assert not _maps(sdfg)

    compiled = dace.ml.compile(model, dace.float32[N, 3, 7, 9], fuse_activations=True)  # ahead-of-time, symbolic batch
    for n in (2, 3):
        x = torch.randn(n, 3, 7, 9)
        with torch.no_grad():
            ref = model(x)
        torch.testing.assert_close(compiled(x), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.torch
def test_fuse_fire_module_with_concat():
    """All three convolutions of a fire module fuse (including the bias-free one) and write into the concat buffer."""
    torch.manual_seed(2)
    model = _Fire().eval()
    desc = dace.float32[N, 8, 6, 5]
    sdfg = dace.ml.to_sdfg(model, desc, fuse_activations=True)
    convs = _convs(sdfg)
    assert len(convs) == 3 and all(c.activation == "relu" for c in convs)
    assert not _maps(sdfg)

    compiled = dace.ml.compile(model, desc, fuse_activations=True)
    for n in (2, 3):
        x = torch.randn(n, 8, 6, 5)
        with torch.no_grad():
            ref = model(x)
        torch.testing.assert_close(compiled(x), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.torch
def test_shared_preactivation_not_fused():
    """A model returning both the ReLU and the pre-activation keeps the map and stays correct for both outputs."""
    torch.manual_seed(3)
    model = _Shared().eval()
    desc = dace.float32[N, 3, 7, 9]
    sdfg = dace.ml.to_sdfg(model, desc, fuse_activations=True)
    convs = _convs(sdfg)
    assert len(convs) == 1 and convs[0].activation == "none"
    assert _maps(sdfg)

    compiled = dace.ml.compile(model, desc, fuse_activations=True)
    for n in (2, 3):
        x = torch.randn(n, 3, 7, 9)
        with torch.no_grad():
            ref = model(x)
        got = compiled(x)
        for g, e in zip(got, ref):
            torch.testing.assert_close(g, e, rtol=1e-4, atol=1e-4)


@pytest.mark.torch
def test_lazy_route_fuses_and_numerics():
    """The torch.compile route folds the ReLU too: the captured SDFG has the fused node and results match torch."""
    torch.manual_seed(4)
    model = _plain().eval()
    compiled = dace.ml.compile(model, fuse_activations=True)  # lazy route, no descriptors
    with torch.no_grad():  # trace (and run) the inference graph, so the ReLU is a plain forward map that folds
        for n in (2, 5):
            x = torch.randn(n, 3, 7, 9)
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)

    sdfg = compiled._dace_backend.last_sdfg
    convs = _convs(sdfg)
    assert len(convs) == 1 and convs[0].activation == "relu"
    assert not _maps(sdfg)


@pytest.mark.torch
def test_training_unaffected_by_option():
    """Training with the option on is still correct: input and weight gradients match eager torch."""
    torch.manual_seed(5)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    model.train()
    reference = copy.deepcopy(model)  # shares the initial parameters; run eagerly for the reference
    compiled = dace.ml.compile(model, fuse_activations=True)

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = compiled(x)
    ref = reference(x_ref)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    out.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(model[0].weight.grad, reference[0].weight.grad, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_default_keeps_relu_map()
    test_fuse_plain_and_numerics()
    test_fuse_fire_module_with_concat()
    test_shared_preactivation_not_fused()
    test_lazy_route_fuses_and_numerics()
    test_training_unaffected_by_option()
    print("ok")
