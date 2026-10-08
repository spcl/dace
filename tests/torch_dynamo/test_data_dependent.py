# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Data-dependent scalars (``.item()``): symbols the SDFG assigns from data."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402

N = dace.symbol("N")

# Inputs with different sizes and different data-dependent values (including an empty selection)
INPUTS = [torch.randn(6), torch.randn(9), torch.full((5,), -1.0), torch.tensor([3.0, -1, 2, 5, -2, 1])]


def _check(fn, inputs=INPUTS):
    compiled = dace.ml.compile(fn, fullgraph=True)
    with torch.no_grad():
        for x in inputs:
            torch.testing.assert_close(compiled(x), fn(x))
    assert compiled._dace_backend.compile_count == 1
    return compiled._dace_backend


@pytest.mark.torch
def test_item_in_arithmetic():

    def f(x):
        return x * x.argmax().item() + x.sum().item()

    _check(f)


@pytest.mark.torch
def test_slice_with_data_dependent_bound():
    """``x[:n]`` has a data-dependent extent; intermediates of that size are allocated after ``n`` is known."""

    def f(x):
        n = (x > 0).sum().item()
        return (x[:n] * 2).sum(dim=0, keepdim=True) + x.mean()

    _check(f)


@pytest.mark.torch
def test_negative_data_dependent_slice_end():
    """A data-dependent end of either sign follows Python's slicing rules."""

    def f(x):
        n = x.argmin().item() - 3
        return (x[:n] * 2).sum(dim=0, keepdim=True)

    _check(f)


@pytest.mark.torch
def test_data_dependent_shift():

    def f(x):
        return x.roll(x.argmax().item())

    _check(f)


@pytest.mark.torch
def test_return_item():
    """A data-dependent scalar returned by the graph comes back as a Python number."""

    def f(x):
        return x * 2, (x > 0).sum().item()

    _check(f)


@pytest.mark.torch
def test_data_dependent_output_shape_is_reported():

    def f(x):
        return x[: (x > 0).sum().item()] * 2

    with pytest.raises(Exception, match="data-dependent output shapes"):
        _check(f)


class _Scaled(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        return h * h.argmax().item()


@pytest.mark.torch
def test_item_in_program_module():
    torch.manual_seed(0)
    model = _Scaled()

    @dace.program
    def program(x: dace.float32[N, 4]):
        return model(x)

    for rows in (3, 5):
        x = torch.randn(rows, 4)
        with torch.no_grad():
            np.testing.assert_allclose(program(x.numpy().copy()), model(x).numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_item_in_arithmetic()
    test_slice_with_data_dependent_bound()
    test_negative_data_dependent_slice_end()
    test_data_dependent_shift()
    test_return_item()
    test_data_dependent_output_shape_is_reported()
    test_item_in_program_module()
