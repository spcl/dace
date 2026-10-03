# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Convolution, pooling, indexing, and a LeNet-style CNN."""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


def _check(backend, fn, inputs_per_shape, expected_compiles=1, rtol=1e-4, atol=1e-4, **compile_kwargs):
    compile_kwargs.setdefault('dynamic', True)
    compiled = torch.compile(fn, backend=backend, **compile_kwargs)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    assert backend.compile_count == expected_compiles, f'expected {expected_compiles} compilation(s), got {backend.compile_count}'
    return compiled


@pytest.mark.torch
def test_conv2d(backend):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, 3, stride=2, padding=1, bias=True).eval()
    _check(backend, conv, [(torch.randn(2, 3, 9, 11), ), (torch.randn(3, 3, 12, 8), )])


@pytest.mark.torch
def test_conv2d_groups_dilation(backend):
    torch.manual_seed(0)
    conv = nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2, bias=False).eval()
    # Distinct spatial sizes: equal sizes would be duck-shaped into one symbol and recompile on the second call
    _check(backend, conv, [(torch.randn(2, 4, 10, 12), ), (torch.randn(2, 4, 7, 13), )])


@pytest.mark.torch
def test_conv1d(backend):
    torch.manual_seed(0)
    conv = nn.Conv1d(3, 4, 5, padding=2).eval()
    _check(backend, conv, [(torch.randn(2, 3, 16), ), (torch.randn(4, 3, 9), )])


@pytest.mark.torch
def test_max_avg_pool(backend):

    def f(x):
        return F.max_pool2d(x, 2) + F.avg_pool2d(x, 2), F.max_pool2d(x, 3, stride=2, padding=1)

    _check(backend, f, [(torch.randn(2, 3, 8, 10), ), (torch.randn(2, 3, 10, 6), )])


@pytest.mark.torch
def test_embedding_gather_argmax(backend):
    torch.manual_seed(0)
    emb = nn.Embedding(50, 8).eval()

    def f(tokens, x, idx):
        return emb(tokens).sum(dim=1), x.gather(1, idx), x.argmax(dim=1), x.argmax()

    _check(backend, f, [(torch.randint(0, 50, (4, 6)), torch.randn(4, 6), torch.randint(0, 6, (4, 2))),
                        (torch.randint(0, 50, (3, 9)), torch.randn(3, 9), torch.randint(0, 9, (3, 2)))])


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


@pytest.mark.torch
def test_lenet(backend):
    torch.manual_seed(0)
    model = LeNet().eval()
    _check(backend, model, [(torch.randn(2, 1, 28, 28), ), (torch.randn(5, 1, 28, 28), )], rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    for test in (test_conv2d, test_conv2d_groups_dilation, test_conv1d, test_max_avg_pool, test_embedding_gather_argmax,
                 test_lenet):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, 'ok')
