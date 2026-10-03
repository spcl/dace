# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pointwise + matmul models compile once across batch sizes and match eager."""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


def _check(backend, fn, inputs_per_shape, expected_compiles=1, **compile_kwargs):
    compile_kwargs.setdefault('dynamic', True)
    compiled = torch.compile(fn, backend=backend, **compile_kwargs)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    assert backend.compile_count == expected_compiles, f'expected {expected_compiles} compilation(s), got {backend.compile_count}'
    return compiled


@pytest.mark.torch
def test_identity(backend):
    _check(backend, lambda x: x * 1.0, [(torch.randn(4, 6), ), (torch.randn(5, 7), )])


@pytest.mark.torch
def test_pointwise_broadcast(backend):

    def f(x, b):
        return torch.relu(x * 2 + b) - torch.sin(x) / (1 + torch.exp(-x))

    _check(backend, f, [(torch.randn(4, 6), torch.randn(6)), (torch.randn(8, 3), torch.randn(3))])


@pytest.mark.torch
def test_mlp_compiles_once(backend):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8), nn.GELU()).eval()
    _check(backend, model, [(torch.randn(3, 16), ), (torch.randn(7, 16), ), (torch.randn(64, 16), )])


@pytest.mark.torch
def test_noncontiguous_input_no_recompile(backend):
    # The transposed input has symbolic strides; the SDFG must not be recompiled when only strides change value.
    f = lambda x: x * 2 + 1
    _check(backend, f, [(torch.randn(6, 4).t(), ), (torch.randn(9, 5).t(), )])


@pytest.mark.torch
def test_sum_mean_amax(backend):

    def f(x):
        return x.sum(dim=1) + x.mean(dim=0, keepdim=True).amax() + x.sum()

    _check(backend, f, [(torch.randn(4, 6), ), (torch.randn(5, 7), )])


@pytest.mark.torch
def test_scalar_and_symbolic_outputs(backend):

    def f(x):
        return x.sum(), x.shape[0] * 2

    compiled = torch.compile(f, backend=backend, dynamic=True)
    with torch.no_grad():
        for x in (torch.randn(4, 6), torch.randn(5, 7)):
            out = compiled(x)
            ref = f(x)
            torch.testing.assert_close(out[0], ref[0], rtol=1e-4, atol=1e-5)
            assert out[1] == ref[1]
    assert backend.compile_count == 1


if __name__ == '__main__':
    for test in (test_identity, test_pointwise_broadcast, test_mlp_compiles_once, test_noncontiguous_input_no_recompile,
                 test_sum_mean_amax, test_scalar_and_symbolic_outputs):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, 'ok')
