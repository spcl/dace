# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Training through torch.compile: AOTAutograd's forward and backward graphs are both compiled with DaCe."""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


def _check_gradients(backend, make_model, make_input, sizes, expected_compiles=2):
    """Compares outputs and parameter/input gradients with eager PyTorch for several input sizes."""
    torch.manual_seed(0)
    model = make_model()
    reference = make_model()
    reference.load_state_dict(model.state_dict())
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for size in sizes:
        x = make_input(size).requires_grad_(True)
        x_ref = x.detach().clone().requires_grad_(True)
        out, ref = compiled(x), reference(x_ref)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
        out.square().sum().backward()
        ref.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
        for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)
        model.zero_grad()
        reference.zero_grad()
    assert backend.compile_count == expected_compiles, f'expected {expected_compiles} compilations'


@pytest.mark.torch
def test_mlp_training(backend):
    _check_gradients(backend, lambda: nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
                     lambda n: torch.randn(n, 8), (5, 9))


@pytest.mark.torch
def test_tanh_sigmoid_training(backend):
    _check_gradients(backend, lambda: nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 3), nn.Sigmoid()),
                     lambda n: torch.randn(n, 6), (4, 7))


@pytest.mark.torch
def test_layernorm_softmax_training(backend):
    _check_gradients(backend, lambda: nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8), nn.Softmax(-1)),
                     lambda n: torch.randn(n, 8), (3, 6))


class _TinyTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.attention = nn.MultiheadAttention(16, 2, batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, x):
        h = self.norm(x)
        x = x + self.attention(h, h, h, need_weights=False)[0]
        return x + self.feed_forward(x)


class _Embedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, tokens):
        return self.head(self.embedding(tokens)).sum(dim=1)


@pytest.mark.torch
def test_cnn_training(backend):
    _check_gradients(
        backend, lambda: nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                       nn.Conv2d(4, 2, 3, stride=2, groups=2), nn.Flatten(), nn.Linear(2, 3)),
        lambda n: torch.randn(n, 3, 8, 8), (2, 3))


@pytest.mark.torch
def test_transformer_training(backend):
    _check_gradients(backend, _TinyTransformer, lambda n: torch.randn(2, n, 16), (5, 7))


@pytest.mark.torch
def test_embedding_training(backend):
    """Embedding gradients accumulate over repeated tokens (index_put with accumulate=True)."""
    torch.manual_seed(0)
    model, reference = _Embedding(), _Embedding()
    reference.load_state_dict(model.state_dict())
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for length in (4, 6):
        tokens = torch.randint(0, 3, (2, length))  # Few distinct tokens: many repeats
        compiled(tokens).square().sum().backward()
        reference(tokens).square().sum().backward()
        for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)
        model.zero_grad()
        reference.zero_grad()
    assert backend.compile_count == 2


if __name__ == '__main__':
    for test in (test_mlp_training, test_tanh_sigmoid_training, test_layernorm_softmax_training, test_cnn_training,
                 test_transformer_training, test_embedding_training):
        torch._dynamo.reset()
        test(DaceBackend())
