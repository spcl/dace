# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Random operators (dropout) across every route that runs a ``torch.nn.Module``: a module used inside a ``@dace.program``,
the DaCe autodiff training routes (``dace.ml.training_step`` and ``dace.ml.differentiable``), and ahead-of-time export
(``dace.ml.compile``). Each route draws a fresh seed per call (reproducible under ``torch.manual_seed``) and a training
step shares the seed between its forward and backward, so the recomputed dropout mask matches the one used in the
forward.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402

N = dace.symbol("N")


def _ml_available() -> bool:
    try:
        import dace.libraries.ml  # noqa: F401
    except ImportError:
        return False
    return True


needs_ml = pytest.mark.skipif(not _ml_available(), reason="dace.libraries.ml is not available")


@pytest.mark.torch
@needs_ml
def test_program_dropout_draws_and_reproduces():
    """A training-mode dropout module inside a ``@dace.program``: fresh masks per call, reproducible with manual_seed."""
    p = 0.3
    model = nn.Dropout(p).train()

    @dace.program
    def prog(x: dace.float32[N, 1000]):
        return model(x)

    x = np.random.rand(64, 1000).astype(np.float32) + 0.5
    prog(x)  # warmup: compile once (the compile path draws a different number of seeds than cached calls)

    y1 = np.asarray(prog(x))
    y2 = np.asarray(prog(x))
    kept = y1 != 0
    # Every element is either dropped (0) or kept and scaled by 1 / (1 - p)
    np.testing.assert_allclose(y1[kept], x[kept] / (1 - p), rtol=1e-5, atol=1e-5)
    assert np.all(y1[~kept] == 0)
    # The keep rate is close to 1 - p on a large tensor, and two calls draw different masks
    assert abs(kept.mean() - (1 - p)) < 0.02
    assert not np.array_equal(y1, y2), "two calls must draw different masks"

    torch.manual_seed(5)
    a = np.asarray(prog(x))
    torch.manual_seed(5)
    b = np.asarray(prog(x))
    assert np.array_equal(a, b), "the mask must follow torch's generator"


def _dropout_model(p: float):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 6, bias=False), nn.Dropout(p)).train()


@pytest.mark.torch
@needs_ml
def test_training_step_shares_seed_with_backward():
    """``training_step`` gradient equals torch's, computed with the mask the DaCe forward used (from a same-seed run)."""
    p = 0.4
    model = _dropout_model(p)
    weight = model[0].weight

    @dace.program
    def block(x: dace.float32[N, 6]):
        return model(x)

    @dace.program
    def loss_program(x: dace.float32[N, 6]):
        return np.sum(model(x))

    forward = dace.ml.differentiable(block)
    step = dace.ml.training_step(loss_program)

    x = torch.randn(10, 6)
    forward(x)  # warmup compile (the compile path draws a different number of seeds)
    step(x)
    model.zero_grad()

    # The forward and backward of a step share the seed, so the mask of a same-seed differentiable forward is the mask
    # the training step differentiated through.
    torch.manual_seed(314)
    out = forward(x).detach()
    mask = (out != 0).to(out.dtype)

    torch.manual_seed(314)
    step(x)
    grad = weight.grad.detach().clone()

    scale = 1.0 / (1 - p)
    expected = (mask * scale).T @ x  # d/dW sum(mask * scale * (x @ W.T))
    torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.torch
@needs_ml
def test_differentiable_backward_recomputes_mask():
    """``differentiable`` returns the dropout output and its backward recomputes the same mask (matching torch)."""
    p = 0.25
    model = _dropout_model(p)
    weight = model[0].weight

    @dace.program
    def block(x: dace.float32[N, 6]):
        return model(x)

    forward = dace.ml.differentiable(block)
    x = torch.randn(12, 6, requires_grad=True)
    out = forward(x)
    out.sum().backward()

    mask = (out != 0).to(out.dtype)
    scale = 1.0 / (1 - p)
    # d/dx sum(mask * scale * (x @ W.T)) = (mask * scale) @ W
    torch.testing.assert_close(x.grad, (mask * scale) @ weight, rtol=1e-4, atol=1e-4)
    # d/dW accumulates (mask * scale).T @ x
    torch.testing.assert_close(weight.grad, (mask * scale).T @ x.detach(), rtol=1e-4, atol=1e-4)


@pytest.mark.torch
@needs_ml
def test_export_compiled_module_draws_fresh_masks():
    """``dace.ml.compile`` (ahead of time) of a training-mode module draws a fresh mask on every call."""
    p = 0.3
    model = nn.Dropout(p).train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # training-mode warning is covered by test_training_mode_warning
        compiled = dace.ml.compile(model, dace.float32[N, 1000])

    x = torch.rand(64, 1000) + 0.5
    with torch.no_grad():
        y1 = compiled(x)
        y2 = compiled(x)
    kept = y1 != 0
    torch.testing.assert_close(y1[kept], x[kept] / (1 - p))
    assert torch.all(y1[~kept] == 0)
    assert abs(kept.float().mean().item() - (1 - p)) < 0.02
    assert not torch.equal(y1, y2), "two calls must draw different masks"


@pytest.mark.torch
def test_training_mode_warning():
    """``to_sdfg``/``compile`` warn when the module is in training mode, but not when it is in eval mode."""
    with pytest.warns(UserWarning, match="training mode"):
        dace.ml.to_sdfg(nn.Dropout(0.3).train(), dace.float32[N, 8])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dace.ml.to_sdfg(nn.Dropout(0.3).eval(), dace.float32[N, 8])
    assert not any("training mode" in str(w.message) for w in caught)


if __name__ == "__main__":
    test_training_mode_warning()
    if _ml_available():
        test_program_dropout_draws_and_reproduces()
        test_training_step_shares_seed_with_backward()
        test_differentiable_backward_recomputes_mask()
        test_export_compiled_module_draws_fresh_masks()
