# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for runtime/lowering corner cases of the TorchDynamo frontend."""
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


@pytest.mark.torch
def test_bool_fill_constant(backend):
    """zeros_like of a bool scalar is constant-propagated; the numpy bool must unparse as a Python/C++ literal."""

    def f(x):
        return torch.zeros_like(x.sum() > 0) | (x.sum() > 1)

    compiled = torch.compile(f, backend=backend, dynamic=True)
    with torch.no_grad():
        for x in (torch.randn(4, 6), torch.ones(5, 7)):
            assert bool(compiled(x)) == bool(f(x))
    assert backend.compile_count == 1


@pytest.mark.torch
def test_rank0_inputs_and_outputs(backend):

    def f(x, s):
        return x * s, (x * s).sum()

    compiled = torch.compile(f, backend=backend, dynamic=True)
    with torch.no_grad():
        for x in (torch.randn(4, 6), torch.randn(5, 7)):
            out = compiled(x, torch.tensor(2.5))
            ref = f(x, torch.tensor(2.5))
            torch.testing.assert_close(out[0], ref[0])
            torch.testing.assert_close(out[1], ref[1])
    assert backend.compile_count == 1


if __name__ == '__main__':
    for test in (test_bool_fill_constant, test_rank0_inputs_and_outputs):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, 'ok')
