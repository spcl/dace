# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Training a ``@dace.program`` that uses modules with DaCe's automatic differentiation (``dace.ml.training_step``)."""
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


if __name__ == '__main__':
    test_sgd_steps_compile_once()
    test_argument_gradient_and_accumulation()
    test_frozen_parameters()
    test_module_in_loop()
    test_loss_must_be_scalar()
