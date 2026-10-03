# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Views, reductions, normalization/softmax decompositions, attention, and a transformer encoder block."""
import math

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
def test_views_and_slices(backend):

    def f(x):
        a = x.view(x.shape[0], -1)
        b = a[:, 1:].t().contiguous()
        c = x[1].unsqueeze(0).expand(3, -1, -1)
        d = torch.cat([x, x * 2], dim=1)
        return b.sum() + c.sum() + d.sum() + x.permute(2, 0, 1)[0].sum() + x[:, ::2].sum()

    _check(backend, f, [(torch.randn(4, 6, 5), ), (torch.randn(3, 8, 5), )])


@pytest.mark.torch
def test_softmax_layernorm(backend):
    ln = nn.LayerNorm(16).eval()

    def f(x):
        return F.softmax(ln(x), dim=-1) + F.log_softmax(x, dim=1)

    _check(backend, f, [(torch.randn(4, 16), ), (torch.randn(9, 16), )])


@pytest.mark.torch
def test_bmm_and_broadcast_matmul(backend):

    def f(q, k):
        return torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])

    _check(backend, f, [(torch.randn(2, 4, 8), torch.randn(2, 6, 8)), (torch.randn(3, 5, 8), torch.randn(3, 7, 8))])


@pytest.mark.torch
def test_scaled_dot_product_attention(backend):

    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    # Note: a batch size of 1 would trigger Dynamo's 0/1 specialization (a legitimate recompile), so avoid it here.
    shapes = [(2, 3, 5, 8), (4, 3, 7, 8)]
    _check(backend, f, [(torch.randn(*s), torch.randn(*s), torch.randn(*s)) for s in shapes])


class EncoderBlock(nn.Module):

    def __init__(self, d_model=32, heads=4, d_ff=64):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, S, self.heads, D // self.heads).transpose(1, 2)
        k = k.view(B, S, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, S, self.heads, D // self.heads).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(att)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


@pytest.mark.torch
def test_transformer_block(backend):
    torch.manual_seed(0)
    model = EncoderBlock().eval()
    _check(backend,
           model, [(torch.randn(2, 5, 32), ), (torch.randn(3, 9, 32), ), (torch.randn(4, 4, 32), )],
           rtol=1e-3,
           atol=1e-4)


if __name__ == '__main__':
    for test in (test_views_and_slices, test_softmax_layernorm, test_bmm_and_broadcast_matmul,
                 test_scaled_dot_product_attention, test_transformer_block):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, 'ok')
