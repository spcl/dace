# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Training a ``@dace.program`` that uses modules with DaCe's automatic differentiation: as one SDFG that computes the
loss and the gradients (``dace.ml.training_step``), or as a differentiable function with a forward and a backward SDFG
(``dace.ml.differentiable``).
"""
import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402

N = dace.symbol('N')


def _models(make_model):
    torch.manual_seed(0)
    model, reference = make_model(), make_model()
    reference.load_state_dict(model.state_dict())
    return model, reference


def _assert_gradients(model, reference):
    for (name, p), p_ref in zip(model.named_parameters(), reference.parameters()):
        if p_ref.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.torch
def test_sgd_steps_compile_once():
    """Optimizer updates are in place, so the program reads the new parameters without recompiling."""
    model, reference = _models(lambda: nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)))

    @dace.program
    def loss_program(x: dace.float32[N, 4], target: dace.float32[N, 2]):
        difference = model(x) - target
        return np.sum(difference * difference)

    step = dace.ml.training_step(loss_program)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.05)
    for n in (5, 7, 5):
        x, target = torch.randn(n, 4), torch.randn(n, 2)
        optimizer.zero_grad()
        optimizer_ref.zero_grad()
        loss = step(x, target)
        loss_ref = (reference(x) - target).square().sum()
        loss_ref.backward()
        torch.testing.assert_close(loss, loss_ref.detach(), rtol=1e-4, atol=1e-5)
        _assert_gradients(model, reference)
        optimizer.step()
        optimizer_ref.step()
    assert step.compile_count == 1


@pytest.mark.torch
def test_argument_gradient_and_accumulation():
    """Arguments that require gradients get them too; gradients accumulate over calls like ``backward()``."""
    model, reference = _models(lambda: nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 3)))

    @dace.program
    def loss_program(x: dace.float32[N, 6]):
        y = model(x)
        return np.sum(y * y)

    step = dace.ml.training_step(loss_program)
    x = torch.randn(4, 6, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    for _ in range(2):
        step(x)
        reference(x_ref).square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    _assert_gradients(model, reference)


@pytest.mark.torch
def test_frozen_parameters():
    """Parameters that do not require gradients are not differentiated."""
    model, reference = _models(lambda: nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1)))
    for module in (model, reference):
        module[0].requires_grad_(False)

    @dace.program
    def loss_program(x: dace.float32[N, 3]):
        return np.sum(model(x))

    x = torch.randn(6, 3)
    dace.ml.training_step(loss_program)(x)
    reference(x).sum().backward()
    assert model[0].weight.grad is None and model[0].bias.grad is None
    _assert_gradients(model, reference)


@pytest.mark.torch
def test_module_in_loop():
    """A module applied repeatedly in a loop of the program (the loop is differentiated by DaCe, not unrolled)."""
    model, reference = _models(lambda: nn.Sequential(nn.Linear(4, 4), nn.Tanh()))

    @dace.program
    def loss_program(x: dace.float32[N, 4], steps: dace.int64):
        h = np.copy(x)
        for _ in range(steps):
            h[:] = model(h)
        return np.sum(h * h)

    x = torch.randn(3, 4)
    loss = dace.ml.training_step(loss_program)(x, 3)
    h = x
    for _ in range(3):
        h = reference(h)
    loss_ref = h.square().sum()
    loss_ref.backward()
    torch.testing.assert_close(loss, loss_ref.detach(), rtol=1e-4, atol=1e-5)
    _assert_gradients(model, reference)


@pytest.mark.torch
def test_loss_must_be_scalar():
    model = nn.Linear(2, 2)

    @dace.program
    def not_a_loss(x: dace.float32[N, 2]):
        return model(x)

    with pytest.raises(ValueError, match='scalar'):
        dace.ml.training_step(not_a_loss)(torch.randn(3, 2))


@pytest.mark.torch
def test_differentiable_composes_with_torch():
    """A program between PyTorch layers: autograd runs its backward SDFG and reaches the layers on both sides."""
    torch.manual_seed(0)
    model, reference = _models(lambda: nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4)))
    before, before_ref = _models(lambda: nn.Linear(3, 4))
    after, after_ref = _models(lambda: nn.Linear(4, 2))

    @dace.program
    def block(x: dace.float32[N, 4]):
        y = model(x)
        return y * y + x

    block = dace.ml.differentiable(block)
    for n in (5, 7):
        x = torch.randn(n, 3)
        target = torch.randn(n, 2)
        loss = nn.functional.mse_loss(after(block(before(x))), target)
        loss.backward()
        h = before_ref(x)
        y = reference(h)
        loss_ref = nn.functional.mse_loss(after_ref(y * y + h), target)
        loss_ref.backward()
        torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-6)
        for pair in ((model, reference), (before, before_ref), (after, after_ref)):
            _assert_gradients(*pair)
            pair[0].zero_grad()
            pair[1].zero_grad()
    assert block.compile_count == 2  # One forward and one backward SDFG for all sizes


@pytest.mark.torch
def test_differentiable_interleaved_and_two_outputs():
    """Saved forward data belongs to each call; outputs that the loss does not use get zero gradients."""
    model, reference = _models(lambda: nn.Linear(3, 3))

    @dace.program
    def two(x: dace.float32[N, 3]):
        y = model(x)
        return np.tanh(y), y * 2

    two = dace.ml.differentiable(two)
    a, b = torch.randn(4, 3), torch.randn(6, 3)
    a_first, _ = two(a)
    b_first, b_second = two(b)
    (b_first.sum() + b_second.square().sum()).backward()
    a_first.square().sum().backward()
    y_b, y_a = reference(b), reference(a)
    (torch.tanh(y_b).sum() + (y_b * 2).square().sum()).backward()
    torch.tanh(y_a).square().sum().backward()
    _assert_gradients(model, reference)


@pytest.mark.torch
def test_differentiable_training_loop():
    model, reference = _models(lambda: nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)))

    @dace.program
    def forward(x: dace.float32[N, 4]):
        return model(x)

    forward = dace.ml.differentiable(forward)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer_ref = torch.optim.SGD(reference.parameters(), lr=0.05)
    for step in range(3):
        x, y = torch.randn(6 + step, 4), torch.randn(6 + step, 1)
        optimizer.zero_grad()
        optimizer_ref.zero_grad()
        nn.functional.mse_loss(forward(x), y).backward()
        nn.functional.mse_loss(reference(x), y).backward()
        _assert_gradients(model, reference)
        optimizer.step()
        optimizer_ref.step()
    assert forward.compile_count == 2


if __name__ == '__main__':
    test_sgd_steps_compile_once()
    test_argument_gradient_and_accumulation()
    test_frozen_parameters()
    test_module_in_loop()
    test_loss_must_be_scalar()
    test_differentiable_composes_with_torch()
    test_differentiable_interleaved_and_two_outputs()
    test_differentiable_training_loop()
