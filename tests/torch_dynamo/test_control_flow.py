# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Higher-order control flow (cond / while_loop / scan / map) lowered to native DaCe control flow."""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.sdfg.state import ConditionalBlock, LoopRegion  # noqa: E402


def _check(backend, fn, inputs_per_shape, expected_compiles=1, **compile_kwargs):
    compile_kwargs.setdefault('dynamic', True)
    compile_kwargs.setdefault('fullgraph', True)
    compiled = torch.compile(fn, backend=backend, **compile_kwargs)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    assert backend.compile_count == expected_compiles, f'expected {expected_compiles} compilation(s), got {backend.compile_count}'
    return compiled


def _regions(backend, kind):
    return [r for r in backend.last_sdfg.all_control_flow_regions(recursive=True) if isinstance(r, kind)]


# ---------------------------------------------------------------------- cond
def _cond_fn(x):
    y = x * 2

    def t(y):
        return y.sin()

    def e(y):
        return y.cos()

    z = torch.cond(y.sum() > 0, t, e, (y, ))
    return z + 1


@pytest.mark.torch
def test_cond_both_branches(backend):
    pos = torch.rand(4, 6) + 0.5
    neg = -(torch.rand(5, 7) + 0.5)
    _check(backend, _cond_fn, [(pos, ), (neg, ), (torch.randn(3, 2), )])
    assert len(_regions(backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_cond_symbolic_strides(backend):
    # Transposed operands have symbolic strides inside the branches
    _check(backend, _cond_fn, [(torch.randn(6, 4).t(), ), (torch.randn(7, 5).t(), )])


@pytest.mark.torch
def test_cond_multiple_outputs(backend):

    def f(x, p):

        def t(x):
            return x + 1, x * 2

        def e(x):
            return x - 1, x / 2

        a, b = torch.cond(p, t, e, (x, ))
        return a @ b.t()

    _check(backend, f, [(torch.randn(4, 6), torch.tensor(True)), (torch.randn(5, 7), torch.tensor(False))])


# ---------------------------------------------------------------------- while_loop
@pytest.mark.torch
def test_while_loop_symbolic_trip_count(backend):

    def f(x):
        n = x.shape[0]

        def cond_fn(i, acc):
            return i < n

        def body_fn(i, acc):
            return i + 1, acc + x

        _, out = torch.while_loop(cond_fn, body_fn, (torch.zeros((), dtype=torch.int64), torch.zeros_like(x)))
        return out

    _check(backend, f, [(torch.randn(4, 6), ), (torch.randn(5, 7), )])
    assert len(_regions(backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_loop_symbolic_strides(backend):

    def f(x):

        def cond_fn(i, acc):
            return i < 3

        def body_fn(i, acc):
            return i + 1, acc * 2 + x

        _, out = torch.while_loop(cond_fn, body_fn, (torch.zeros((), dtype=torch.int64), torch.zeros_like(x)))
        return out

    _check(backend, f, [(torch.randn(6, 4).t(), ), (torch.randn(7, 5).t(), )])


@pytest.mark.torch
def test_cond_inside_while(backend):

    def f(x):
        n = x.shape[0]

        def cond_fn(i, acc):
            return i < n

        def body_fn(i, acc):
            acc = torch.cond(acc.sum() > 10, lambda a: a * 0.5, lambda a: a + x, (acc, ))
            return i + 1, acc

        _, out = torch.while_loop(cond_fn, body_fn, (torch.zeros((), dtype=torch.int64), torch.ones_like(x)))
        return out

    _check(backend, f, [(torch.rand(4, 6), ), (torch.rand(9, 3), )])
    assert len(_regions(backend, LoopRegion)) == 1 and len(_regions(backend, ConditionalBlock)) == 1


# ---------------------------------------------------------------------- scan / map
@pytest.mark.torch
def test_scan_symbolic_length(backend):
    from torch._higher_order_ops.scan import scan

    def f(x):

        def combine(c, xi):
            nc = c + xi
            return nc, nc * 2

        c, ys = scan(combine, torch.zeros(x.shape[1]), x)
        return c, ys

    _check(backend, f, [(torch.randn(4, 6), ), (torch.randn(7, 3), )])
    assert len(_regions(backend, LoopRegion)) == 1


@pytest.mark.torch
def test_map_symbolic_length(backend):
    from torch._higher_order_ops.map import map as hmap

    def f(x, w):
        return hmap(lambda xi, w: (xi * 2) @ w, x, w)

    _check(backend, f, [(torch.randn(4, 6, 5), torch.randn(5, 3)), (torch.randn(2, 7, 5), torch.randn(5, 3))])


if __name__ == '__main__':
    for test in (test_cond_both_branches, test_cond_symbolic_strides, test_cond_multiple_outputs,
                 test_while_loop_symbolic_trip_count, test_while_loop_symbolic_strides, test_cond_inside_while,
                 test_scan_symbolic_length, test_map_symbolic_length):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, 'ok')
