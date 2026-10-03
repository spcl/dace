# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Bytecode-level capture of plain Python ``if``/``while`` on tensor data (``cfg/goto_capture``).

The experimental ``CfgDaceBackend`` patches Dynamo's conditional-jump handlers so that data-dependent branches become
``torch.cond`` / ``torch.while_loop`` HOPs instead of graph breaks; the existing importer lowers those to
``ConditionalBlock`` / ``LoopRegion``. Tests assert compile-once behaviour (``compile_count == 1``) across shapes.
"""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo.cfg import CfgDaceBackend  # noqa: E402
from dace.sdfg.state import ConditionalBlock, LoopRegion  # noqa: E402


@pytest.fixture
def cfg_backend():
    return CfgDaceBackend()


def _check(backend, fn, inputs_per_shape, expected_compiles=1, **compile_kwargs):
    compile_kwargs.setdefault('dynamic', True)
    compile_kwargs.setdefault('fullgraph', True)
    compiled = torch.compile(fn, backend=backend, **compile_kwargs)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    assert backend.compile_count == expected_compiles, \
        f'expected {expected_compiles} compilation(s), got {backend.compile_count}'
    return compiled


def _regions(backend, kind):
    return [r for r in backend.last_sdfg.all_control_flow_regions(recursive=True) if isinstance(r, kind)]


def _events(backend):
    return backend.capture.kinds()


_POS = [(torch.rand(3, 4) + 0.5, ), (torch.rand(4, 5) + 0.5, ), (torch.rand(5, 2) * 0.1 + 0.01, )]
_MIXED = [(torch.rand(4, 6) + 0.5, ), (-(torch.rand(5, 7) + 0.5), ), (torch.randn(3, 2), )]


# ---------------------------------------------------------------------- if / else
def _if_else(x):
    y = x * 2
    if y.sum() > 0:
        y = y.sin()
    else:
        y = y.cos()
    return y + 1


@pytest.mark.torch
def test_if_else_both_continuations(cfg_backend):
    _check(cfg_backend, _if_else, _MIXED)
    assert _events(cfg_backend) == ['if']
    assert len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_if_without_else(cfg_backend):

    def f(x):
        y = x * 2
        if y.sum() > 0:
            y = y.sin()
        return y + 1

    _check(cfg_backend, f, _MIXED)
    assert len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_if_with_int_and_symint_locals(cfg_backend):
    # Live locals that are Python ints / SymInts travel through the cond operands

    def f(x, k):
        y = x * k
        n = x.shape[0]
        if y.sum() > 0:
            y = y.sin() + n
        else:
            y = y.cos() * k
        return y + 1

    _check(cfg_backend, f, [(torch.rand(4, 6) + 0.5, 3), (-(torch.rand(5, 7) + 0.5), 3)])


@pytest.mark.torch
def test_if_elif_else_nested(cfg_backend):

    def f(x):
        y = x * 2
        if y.sum() > 100:
            y = y.sin()
        elif y.sum() > 0:
            if y.mean() > 1:
                y = y.cos()
            else:
                y = y.tan()
        else:
            y = -y
        return y + 1

    _check(cfg_backend, f, [(torch.rand(4, 6) * 20, ), (torch.rand(5, 7) * 0.1, ), (-(torch.rand(3, 2) + 0.5), )])
    assert _events(cfg_backend).count('if') >= 3  # tail duplication traces nested ifs per continuation


@pytest.mark.torch
def test_ternary_and_multiple_returns(cfg_backend):

    def f(x):
        y = x * 2
        z = y.sin() if y.sum() > 0 else y.cos()
        if z.mean() > 10:
            return z
        return z + 1

    _check(cfg_backend, f, _MIXED)


# ---------------------------------------------------------------------- while
def _while(x):
    acc = x
    while acc.sum() < 100:
        acc = acc * 2
    return acc + 1


@pytest.mark.torch
def test_while_tensor_predicate(cfg_backend):
    _check(cfg_backend, _while, _POS)
    assert _events(cfg_backend) == ['while', 'backedge']
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_with_if_in_body(cfg_backend):

    def f(x):
        acc = x
        while acc.sum() < 100:
            if acc.mean() > 1:
                acc = acc * 1.5
            else:
                acc = acc * 2
        return acc + 1

    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 1 and len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_while_break_and_continue(cfg_backend):

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 1e9:
                continue
            if acc.max() > 20:
                break
            acc = acc + 1
        return acc + 1

    _check(cfg_backend, f, _POS)
    assert 'exit' in _events(cfg_backend)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_true_break(cfg_backend):

    def f(x):
        acc = x
        while True:
            acc = acc * 2
            if acc.sum() > 100:
                break
        return acc + 1

    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_nested_while(cfg_backend):

    def f(x):
        acc = x
        while acc.sum() < 1000:
            inner = acc
            while inner.sum() < 100:
                inner = inner * 2
            acc = inner + acc
        return acc

    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 2


@pytest.mark.torch
def test_if_then_while_tail_duplicates(cfg_backend):
    # The loop after the if is traced into both cond branches (tail duplication); still one SDFG

    def f(x):
        y = x * 2
        if y.sum() > 0:
            y = y.sin()
        else:
            y = y.cos()
        y = y.abs() + 0.1
        while y.sum() < 100:
            y = y * 2
        return y

    _check(cfg_backend, f, _MIXED)
    assert len(_regions(cfg_backend, LoopRegion)) == 2 and len(_regions(cfg_backend, ConditionalBlock)) == 1


# ---------------------------------------------------------------------- fallbacks (stock Dynamo behaviour)
@pytest.mark.torch
def test_fallback_return_inside_loop(cfg_backend):
    # Early return inside a loop cannot be expressed with cond/while_loop: the capture fails, the jump is blacklisted
    # and Dynamo restarts with its normal graph break (frame runs correctly, more than one graph).

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 20:
                return acc
        return acc + 1

    compiled = torch.compile(f, backend=cfg_backend, dynamic=True, fullgraph=False)
    with torch.no_grad():
        for (x, ) in _POS:
            torch.testing.assert_close(compiled(x), f(x))
    assert 'error' in _events(cfg_backend)


@pytest.mark.torch
def test_fallback_non_empty_stack(cfg_backend):
    # A tensor-dependent while inside a Python-unrolled for: the range iterator sits on the value stack

    def f(x):
        acc = x
        for i in range(2):
            while acc.sum() < 100:
                acc = acc * 2
            acc = acc * 0.25
        return acc

    compiled = torch.compile(f, backend=cfg_backend, dynamic=True, fullgraph=False)
    with torch.no_grad():
        for (x, ) in _POS:
            torch.testing.assert_close(compiled(x), f(x))
    assert 'fallback' in _events(cfg_backend)


@pytest.mark.torch
@pytest.mark.xfail(strict=False, reason='for over range(SymInt) is not captured yet (FOR_ITER specialises the range)')
def test_for_range_symint_compiles_once(cfg_backend):

    def f(x):
        acc = torch.zeros_like(x[0])
        for i in range(x.shape[0]):
            acc = acc + x[i]
        return acc

    _check(cfg_backend, f, [(torch.randn(3, 4), ), (torch.randn(5, 4), )])


@pytest.mark.torch
def test_production_backend_unaffected():
    # The patch is gated on the backend object: the stock DaceBackend still graph-breaks (fullgraph=True errors)
    from dace.frontend.ml.torch.dynamo import DaceBackend
    CfgDaceBackend()  # ensure the patch is installed
    compiled = torch.compile(_if_else, backend=DaceBackend(), dynamic=True, fullgraph=True)
    with pytest.raises(Exception, match='Data-dependent branching'):
        compiled(torch.rand(3, 4))


if __name__ == '__main__':
    for test in (test_if_else_both_continuations, test_if_without_else, test_if_with_int_and_symint_locals,
                 test_if_elif_else_nested, test_ternary_and_multiple_returns, test_while_tensor_predicate,
                 test_while_with_if_in_body, test_while_break_and_continue, test_while_true_break, test_nested_while,
                 test_if_then_while_tail_duplicates, test_fallback_return_inside_loop, test_fallback_non_empty_stack):
        torch._dynamo.reset()
        test(CfgDaceBackend())
        print(test.__name__, 'ok')
