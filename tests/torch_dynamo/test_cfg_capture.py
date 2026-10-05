# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Capture of data-dependent Python control flow as a control-flow graph of traced blocks (``dace::cfg``).

The tests are semantic (outputs match eager PyTorch for inputs that take different paths), so they hold for every
bytecode layout CPython 3.10-3.14 produces for the same Python code.
"""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dace.frontend.ml.torch.dynamo.cfg.blocks import ControlFlowBackend  # noqa: E402

OFFSET = torch.tensor([1.5])


def _check(fn, inputs, expect_capture=True, expected_compiles=1):
    """Compares ``fn`` compiled with control-flow capture against eager PyTorch on every input."""
    backend = ControlFlowBackend()
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for args in inputs:
            args = args if isinstance(args, tuple) else (args, )
            torch.testing.assert_close(compiled(*args), fn(*args), rtol=1e-5, atol=1e-5)
    if expect_capture:
        assert 'cfg' in backend.kinds(), backend.events
        assert backend.compile_count == expected_compiles, backend.events
    return backend


@pytest.mark.torch
def test_if_else():

    def f(x):
        if x.sum() > 0:
            y = x * 2
        else:
            y = x - OFFSET
        return y + 1

    _check(f, [torch.rand(4), -torch.rand(5)])


@pytest.mark.torch
def test_if_without_else_and_elif():

    def f(x):
        y = x + 1
        if y.mean() > 1.5:
            y = y * 3
        elif y.mean() > 1.2:
            y = y * 2
        if y.max() > 4:
            y = y - 4
        return y

    _check(f, [torch.full((3, ), 0.9), torch.full((4, ), 0.3), torch.full((2, ), 0.0), torch.rand(5)])


@pytest.mark.torch
def test_while_with_break_and_continue():

    def f(x):
        y = x
        while y.sum() < 100:
            y = y * 2 + 1
            if y.max() > 60:
                break
            if y.min() > 5:
                continue
            y = y + 1
        return y

    _check(f, [torch.rand(4), torch.full((3, ), 20.0), torch.full((2, ), 0.1), torch.full((5, ), 1000.0)])


@pytest.mark.torch
def test_while_else():

    def f(x):
        y = x
        while y.sum() < 50:
            y = y * 2 + 1
            if y.max() > 30:
                y = y - 100
                break
        else:
            y = y + 0.5
        return y

    _check(f, [torch.rand(3), torch.full((2, ), 14.0), torch.full((4, ), 100.0)])


@pytest.mark.torch
def test_nested_loops_and_early_return():

    def f(x):
        y = x
        while y.sum() < 200:
            z = y
            while z.max() < 10:
                z = z * 3
                if z.min() > 8:
                    return z * -1
            y = y + z
        return y

    _check(f, [torch.rand(3) + 0.5, torch.full((2, ), 9.5), torch.full((4, ), 300.0)])


@pytest.mark.torch
def test_sequential_ifs_are_linear():
    """Every block is traced once: n sequential conditionals need O(n) traces, not 2**n."""

    def f(x):
        y = x
        if y[0] > 0:
            y = y + 1
        if y[1] > 0:
            y = y * 2
        if y[2] > 0:
            y = y - 3
        if y[3] > 0:
            y = y * -1
        if y[0] > 1:
            y = y + 5
        if y[1] > 1:
            y = y / 2
        return y

    backend = _check(f, [torch.randn(4) for _ in range(6)])
    blocks = int(next(detail for kind, _, detail in backend.events if kind == 'cfg').split()[0])
    assert blocks <= 4 * 6, f'{blocks} blocks for 6 sequential conditionals'


class _Gated(nn.Module):
    """A module whose forward branches on data and loops, using parameters and attributes on every path."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.scale = 2.5  # A float attribute is a 0-d tensor read with .item() in the blocks

    def forward(self, x):
        h = self.fc(x)
        if h.sum() > 0:
            h = torch.relu(h) * self.scale
        else:
            h = self.fc(h)
        steps = 0
        while h.abs().sum() < 10:
            h = h * 2 + self.fc.bias
            steps = steps  # A Python value live across blocks (unchanged)
        return h


@pytest.mark.torch
def test_module_forward():
    torch.manual_seed(0)
    model = _Gated().eval()
    _check(model, [torch.randn(3, 4), torch.randn(5, 4), -torch.rand(2, 4) * 3])


@pytest.mark.torch
def test_fallback_keeps_semantics():
    """A data-dependent branch inside a for loop is not captured yet; Dynamo's graph break keeps the result."""

    def f(x):
        y = x
        for _ in range(3):
            if y.sum() > 0:
                y = y - 1
            else:
                y = y + 2
        return y

    backend = _check(f, [torch.rand(4), -torch.rand(4)], expect_capture=False)
    assert 'cfg' not in backend.kinds()


@pytest.mark.torch
def test_python_and_symbolic_locals():
    """Python ints and SymInts live across blocks are passed along (and specialize the blocks)."""

    def f(x, k):
        y = x * k
        n = x.shape[0]
        if y.sum() > 0:
            y = y.sin() + n
        else:
            y = y.cos() * k
        return y + 1

    _check(f, [(torch.rand(4, 6) + 0.5, 3), (-(torch.rand(5, 7) + 0.5), 3)])


@pytest.mark.torch
def test_conditional_expression_and_multiple_returns():
    """A conditional expression leaves the value stack non-empty at its join: Dynamo's graph break handles it."""

    def f(x):
        y = x * 2
        z = y.sin() if y.sum() > 0 else y.cos()
        if z.mean() > 10:
            return z
        return z + 1

    _check(f, [torch.rand(4), -torch.rand(5)], expect_capture=False)


@pytest.mark.torch
def test_while_true_break():

    def f(x):
        acc = x
        while True:
            acc = acc * 2
            if acc.sum() > 100:
                break
        return acc + 1

    _check(f, [torch.rand(3), torch.full((2, ), 60.0)])


@pytest.mark.torch
def test_stock_backend_unaffected():
    """The handlers are only active for ControlFlowBackend: the stock DaceBackend still graph-breaks."""
    from dace.frontend.ml.torch.dynamo import DaceBackend
    ControlFlowBackend()  # Installs the handlers

    def f(x):
        if x.sum() > 0:
            return x * 2
        return x - 1

    compiled = torch.compile(f, backend=DaceBackend(), dynamic=True, fullgraph=True)
    with pytest.raises(Exception, match='Data-dependent branching'):
        compiled(torch.rand(3, 4))


@pytest.mark.torch
def test_match_on_python_value():
    """Patterns on Python values are resolved by Dynamo; the data-dependent branch after the match is captured."""

    def f(x, mode: int):
        match mode:
            case 0:
                y = x * 2
            case 1:
                y = x + 1
            case _:
                y = x
        if y.sum() > 0:
            return y
        return -y

    # Dynamo specializes on each value of ``mode`` (one compilation per value)
    _check(f, [(torch.rand(3), 0), (-torch.rand(4), 1), (torch.rand(2), 5)], expected_compiles=3)


@pytest.mark.torch
def test_match_with_tensor_guard():

    def f(x):
        y = x * 1
        match 0:
            case 0 if y.sum() > 0:
                y = y * 3
            case _:
                y = y - 3
        return y

    _check(f, [torch.rand(3), -torch.rand(4)])


@pytest.mark.torch
def test_for_else_falls_back():
    """
    Data-dependent control flow inside a for loop is not captured yet (the iterator is on the value stack); with
    FOR_ITER capture (phase 2e) the else clause becomes the exhaustion edge. Until then the graph break keeps the
    semantics.
    """

    def f(x):
        y = x
        for _ in range(3):
            y = y * 2
            if y.sum() > 50:
                break
        else:
            y = y - 100
        return y

    backend = _check(f, [torch.rand(3), torch.full((2, ), 20.0)], expect_capture=False)
    assert 'cfg' not in backend.kinds()


@pytest.mark.torch
def test_match_on_bool_of_tensor_falls_back():
    """
    ``bool(tensor)`` is a call, not a conditional jump, so Dynamo graph-breaks at it. Capturing it (as the predicate
    of the jumps that test its result) is planned as a special case (phase 2e).
    """

    def f(x):
        match bool(x.sum() > 0):
            case True:
                return x * 2
            case False:
                return x - 1

    backend = _check(f, [torch.rand(3), -torch.rand(3)], expect_capture=False)
    assert 'cfg' not in backend.kinds()


if __name__ == '__main__':
    test_if_else()
    test_if_without_else_and_elif()
    test_while_with_break_and_continue()
    test_while_else()
    test_nested_loops_and_early_return()
    test_sequential_ifs_are_linear()
    test_module_forward()
    test_fallback_keeps_semantics()
    test_python_and_symbolic_locals()
    test_conditional_expression_and_multiple_returns()
    test_while_true_break()
    test_stock_backend_unaffected()
    test_match_on_python_value()
    test_match_with_tensor_guard()
    test_for_else_falls_back()
    test_match_on_bool_of_tensor_falls_back()
