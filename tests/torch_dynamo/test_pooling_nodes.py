# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pooling and convolution-backward as library nodes through the TorchDynamo frontend: no padded copy or padding map in
the SDFG, forward numerics through ``torch.compile``, and a CNN training step whose gradients match PyTorch.
"""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend, capture  # noqa: E402
from dace.libraries.torch import (  # noqa: E402
    AvgPool,
    ConvolutionBackward,
    MaxPoolWithIndices,
    MaxPoolWithIndicesBackward,
)
from dace.sdfg import nodes  # noqa: E402


def _nodes(sdfg, kind):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, kind)]


def _no_padding(sdfg):
    assert not any("pad" in str(getattr(n, "label", "")).lower() for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.torch
def test_pooling_nodes_no_padding():
    """Max and average pooling are library nodes; the imported SDFG has no padding node or map."""

    def f(x):
        return F.max_pool2d(x, 3, stride=2, padding=1), F.avg_pool2d(x, 3, stride=2, padding=1)

    sdfg = capture(f, torch.randn(2, 3, 9, 11), dynamic_shapes={"x": {0: "N"}}).to_sdfg("pool_nodes")
    assert _nodes(sdfg, MaxPoolWithIndices)
    assert _nodes(sdfg, AvgPool)
    assert not any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())
    _no_padding(sdfg)


def _check_forward(fn, inputs_per_shape, expected_compiles=1, rtol=1e-4, atol=1e-4):
    backend = DaceBackend()
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            torch.testing.assert_close(compiled(*inputs), fn(*inputs), rtol=rtol, atol=atol)
    if expected_compiles is not None:
        assert backend.compile_count == expected_compiles
    _no_padding(backend.last_sdfg)
    return backend


@pytest.mark.torch
def test_max_pool_forward():
    def f(x):
        return F.max_pool2d(x, 2), F.max_pool2d(x, 3, stride=2, padding=1)

    _check_forward(f, [(torch.randn(2, 3, 8, 10),), (torch.randn(3, 3, 10, 6),)])


@pytest.mark.torch
def test_max_pool_dilation_ceil_mode():
    def f(x):
        return F.max_pool2d(x, 3, stride=2, padding=1, dilation=2, ceil_mode=True)

    _check_forward(f, [(torch.randn(2, 3, 11, 13),), (torch.randn(3, 3, 10, 14),)])


@pytest.mark.torch
def test_avg_pool_variants():
    def f(x):
        a = F.avg_pool2d(x, 3, stride=2, padding=1)
        b = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
        c = F.avg_pool2d(x, 2, stride=2, divisor_override=5)
        return a, b, c

    _check_forward(f, [(torch.randn(2, 3, 9, 11),), (torch.randn(3, 3, 12, 8),)])


@pytest.mark.torch
def test_pool_1d_3d_forward():
    def f1(x):
        return F.max_pool1d(x, 3, stride=2, padding=1) + F.avg_pool1d(x, 3, stride=2, padding=1)

    def f3(x):
        return F.max_pool3d(x, 2, stride=2, padding=1)

    # max-pool-with-indices for 1-d/3-d triggers an extra Dynamo shape specialization at capture time (unrelated to the
    # library nodes), so the compile count is not asserted here; only the numerics are.
    _check_forward(f1, [(torch.randn(2, 4, 15),), (torch.randn(3, 4, 11),)], expected_compiles=None)
    _check_forward(f3, [(torch.randn(2, 2, 7, 8, 9),), (torch.randn(3, 2, 9, 7, 8),)], expected_compiles=None)


class _CNN(nn.Module):
    """A LeNet-style network exercising conv, max pool and their gradients."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@pytest.mark.torch
def test_cnn_training_step():
    """A training step through the CNN: output and all gradients match PyTorch, using the backward library nodes."""
    torch.manual_seed(0)
    model = _CNN()
    reference = _CNN()
    reference.load_state_dict(model.state_dict())
    backend = DaceBackend()
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for n in (2, 3):
        x = torch.randn(n, 1, 28, 28, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out, ref = compiled(x), reference(x_ref)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
        out.square().sum().backward()
        ref.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-3, atol=1e-4)
        for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-3, atol=1e-4, msg=name)
        model.zero_grad()
        reference.zero_grad()
    assert backend.compile_count == 1
    assert _nodes(backend.last_sdfg, ConvolutionBackward)
    assert _nodes(backend.last_sdfg, MaxPoolWithIndicesBackward)
    _no_padding(backend.last_sdfg)


if __name__ == "__main__":
    test_pooling_nodes_no_padding()
    test_max_pool_forward()
    test_max_pool_dilation_ceil_mode()
    test_avg_pool_variants()
    test_pool_1d_3d_forward()
    test_cnn_training_step()
    print("ok")
