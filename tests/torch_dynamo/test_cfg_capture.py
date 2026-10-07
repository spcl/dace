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


def _check(fn, inputs, expect_capture=True, expected_compiles=1, **options):
    """Compares ``fn`` compiled with control-flow capture against eager PyTorch on every input."""
    backend = ControlFlowBackend(**options)
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for args in inputs:
            args = args if isinstance(args, tuple) else (args,)
            torch.testing.assert_close(compiled(*args), fn(*args), rtol=1e-5, atol=1e-5)
    if expect_capture:
        assert "cfg" in backend.kinds(), backend.events
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

    _check(f, [torch.full((3,), 0.9), torch.full((4,), 0.3), torch.full((2,), 0.0), torch.rand(5)])


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

    _check(f, [torch.rand(4), torch.full((3,), 20.0), torch.full((2,), 0.1), torch.full((5,), 1000.0)])


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

    _check(f, [torch.rand(3), torch.full((2,), 14.0), torch.full((4,), 100.0)])


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

    _check(f, [torch.rand(3) + 0.5, torch.full((2,), 9.5), torch.full((4,), 300.0)])


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
    blocks = int(next(detail for kind, _, detail in backend.events if kind == "cfg").split()[0])
    assert blocks <= 4 * 6, f"{blocks} blocks for 6 sequential conditionals"


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
def test_branch_in_constant_loop():
    """
    A data-dependent branch inside a loop Dynamo would unroll: the analysis restarts and captures the loop from its
    GET_ITER, with a symbolic counter.
    """

    def f(x):
        y = x
        for _ in range(3):
            if y.sum() > 0:
                y = y - 1
            else:
                y = y + 2
        return y

    _check(f, [torch.rand(4), -torch.rand(4)])


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

    _check(f, [torch.rand(3), torch.full((2,), 60.0)])


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
    with pytest.raises(Exception, match="Data-dependent branching"):
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
def test_for_else():
    """The else clause of a for loop is the edge taken when the loop runs out of items."""

    def f(x):
        y = x
        for _ in range(3):
            y = y * 2
            if y.sum() > 50:
                break
        else:
            y = y - 100
        return y

    _check(f, [torch.rand(3), torch.full((2,), 20.0)])


@pytest.mark.torch
def test_symbolic_range():
    """``range`` over a symbolic size is a loop with a symbolic trip count (one compilation for all sizes)."""

    def f(x):
        y = x[0] * 0
        for i in range(x.shape[0]):
            y = y + x[i] * i
        return y

    _check(f, [torch.randn(4, 3), torch.randn(7, 3), torch.randn(2, 3)])


@pytest.mark.torch
def test_iterate_tensor():

    def f(x):
        y = x[0] * 0
        for row in x:
            y = y * 0.5 + row
        return y

    _check(f, [torch.randn(4, 3), torch.randn(6, 3)])


@pytest.mark.torch
def test_symbolic_range_with_break_continue_else():

    def f(x):
        y = x[0]
        for i in range(1, x.shape[0]):
            if y.sum() > 20:
                break
            if x[i].sum() < 0:
                continue
            y = y * 2 + x[i]
        else:
            y = y - 100
        return y

    _check(f, [torch.rand(5, 2), torch.rand(3, 2) * 0.1, torch.full((4, 2), 30.0), -torch.rand(6, 2)])


@pytest.mark.torch
def test_nested_loops_with_while():

    def f(x):
        y = x[0] * 0
        for row in x:
            for j in range(row.shape[0]):
                y = y + row[j]
            while y.abs().sum() > 100:
                y = y * 0.5
        return y

    _check(f, [torch.randn(3, 4) * 10, torch.randn(5, 4)])


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if x.sum() > 0:
                x = -x
        return x


@pytest.mark.torch
def test_loop_over_modules_falls_back():
    """A loop over a list of modules cannot have a symbolic counter: Dynamo's graph break keeps the semantics."""
    torch.manual_seed(0)
    _check(_Stack().eval(), [torch.randn(2, 4), torch.randn(3, 4)], expect_capture=False)


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
    assert "cfg" not in backend.kinds()


@pytest.mark.torch
def test_symbolic_branches_opt_in():
    """Branches on sizes are guards (one compilation per outcome) unless ``symbolic_branches`` makes them edges."""

    def f(x):
        if x.shape[0] > 4:
            y = x * 2
        else:
            y = x - 1
        return y + 1

    inputs = [torch.randn(6, 2), torch.randn(3, 2), torch.randn(8, 2)]
    backend = _check(f, inputs, expect_capture=False)
    assert "cfg" not in backend.kinds() and backend.compile_count == 2
    _check(f, inputs, symbolic_branches=True)


@pytest.mark.torch
def test_while_counter():
    """A counter compared with a size: with ``symbolic_branches``, the loop has a symbolic trip count."""

    def f(x):
        i = 0
        y = x[0] * 0
        while i < x.shape[0]:
            y = y + x[i]
            i += 1
        return y

    _check(f, [torch.randn(4, 3), torch.randn(6, 3)], symbolic_branches=True)


@pytest.mark.torch
def test_counting_down():
    """A loop variable that may become negative is not assumed nonnegative (the capture falls back)."""

    def f(x):
        i = x.shape[0] - 1
        y = x[0] * 0
        while i >= 0:
            y = y * 2 + x[i]
            i -= 1
        return y

    _check(f, [torch.randn(4, 3), torch.randn(6, 3)], expect_capture=False, symbolic_branches=True)


@pytest.mark.torch
def test_branch_on_item():
    """A branch on a data-dependent scalar (``.item()``) is an edge of the graph."""

    def f(x):
        n = (x > 0).sum().item()
        if n > 2:
            return x * n
        return x - 1

    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        _check(f, [torch.randn(6), -torch.rand(5), torch.rand(3)])


class _Branching(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.alt = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if h.sum() > 0:
            h = torch.tanh(h) * 2
        else:
            h = self.alt(h)
        return h + 1


@pytest.mark.torch
def test_training_through_captured_branch():
    """
    AOTAutograd cannot differentiate captured control flow; DaCe's autodiff does, in one SDFG with a forward and a
    backward phase (the forward phase records the branch taken). Parameters of the branch not taken get zero
    gradients.
    """
    torch.manual_seed(0)
    model, reference = _Branching(), _Branching()
    reference.load_state_dict(model.state_dict())
    backend = ControlFlowBackend()
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for x in (torch.randn(3, 4), -torch.rand(5, 4) * 3):
        x_ref = x.clone().requires_grad_(True)
        x = x.clone().requires_grad_(True)
        out, ref = compiled(x), reference(x_ref)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
        out.square().sum().backward()
        ref.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
        for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
            expected = p_ref.grad if p_ref.grad is not None else torch.zeros_like(p_ref)
            torch.testing.assert_close(p.grad, expected, rtol=1e-4, atol=1e-5, msg=name)
        model.zero_grad()
        reference.zero_grad()
    assert "cfg" in backend.kinds() and backend.compile_count == 1


@pytest.mark.torch
def test_training_interleaved_calls_keep_their_branches():
    """Each forward call records its own tape: two calls that take different branches, backpropagated in reverse."""
    torch.manual_seed(0)
    model, reference = _Branching(), _Branching()
    reference.load_state_dict(model.state_dict())
    compiled = torch.compile(model, backend=ControlFlowBackend(), dynamic=True)
    first, second = torch.randn(3, 4), -torch.rand(5, 4) * 3
    inputs = [t.clone().requires_grad_(True) for t in (first, second)]
    inputs_ref = [t.clone().requires_grad_(True) for t in (first, second)]
    outputs = [compiled(x) for x in inputs]
    outputs_ref = [reference(x) for x in inputs_ref]
    for out, ref in zip(reversed(outputs), reversed(outputs_ref)):
        out.square().sum().backward()
        ref.square().sum().backward()
    for x, x_ref in zip(inputs, inputs_ref):
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)


class _Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        h = x[0] * 0
        for i in range(x.shape[0]):
            h = torch.tanh(self.fc(h)) + x[i]  # ``Linear`` of a vector squeezes its result in place
        return h


@pytest.mark.torch
def test_training_through_captured_loop():
    """A loop over a symbolic size is differentiated by DaCe (as a for loop), for every size with one compilation."""
    torch.manual_seed(0)
    model, reference = _Recurrent(), _Recurrent()
    reference.load_state_dict(model.state_dict())
    backend = ControlFlowBackend()
    compiled = torch.compile(model, backend=backend, dynamic=True)
    for x in (torch.randn(4, 3), torch.randn(6, 3)):
        x_ref = x.clone().requires_grad_(True)
        x = x.clone().requires_grad_(True)
        out, ref = compiled(x), reference(x_ref)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
        out.square().sum().backward()
        ref.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
        for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)
        model.zero_grad()
        reference.zero_grad()
    assert "cfg" in backend.kinds() and backend.compile_count == 1


if __name__ == "__main__":
    test_if_else()
    test_if_without_else_and_elif()
    test_while_with_break_and_continue()
    test_while_else()
    test_nested_loops_and_early_return()
    test_sequential_ifs_are_linear()
    test_module_forward()
    test_branch_in_constant_loop()
    test_python_and_symbolic_locals()
    test_conditional_expression_and_multiple_returns()
    test_while_true_break()
    test_stock_backend_unaffected()
    test_match_on_python_value()
    test_match_with_tensor_guard()
    test_for_else()
    test_symbolic_range()
    test_iterate_tensor()
    test_symbolic_range_with_break_continue_else()
    test_nested_loops_with_while()
    test_loop_over_modules_falls_back()
    test_symbolic_branches_opt_in()
    test_while_counter()
    test_counting_down()
    test_branch_on_item()
    test_training_through_captured_branch()
    test_training_interleaved_calls_keep_their_branches()
    test_match_on_bool_of_tensor_falls_back()
    test_training_through_captured_loop()
