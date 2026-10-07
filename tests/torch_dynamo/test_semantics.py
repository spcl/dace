# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numerical semantics of lowerings that must match ATen exactly (rounding, NaN propagation, identities, constants)."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


def _check(backend, fn, inputs_per_call, expected_compiles=1):
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for inputs in inputs_per_call:
            torch.testing.assert_close(compiled(*inputs), fn(*inputs), equal_nan=True)
    if expected_compiles is not None:
        assert backend.compile_count == expected_compiles


@pytest.mark.torch
def test_amax_amin_identity(backend):
    """
    The output of a max/min reduction must be initialized: all-negative inputs expose a zero-initialized output, and
    repeated calls expose leftover memory.
    """

    def f(x):
        return x.amax(dim=1), x.amin(dim=0), x.amax(), x.amin()

    inputs = [(-torch.rand(4, 6) - 1,), (torch.rand(5, 7) + 1,)] * 3
    _check(backend, f, inputs)


@pytest.mark.torch
def test_amax_integer_and_bool(backend):

    def f(x):
        return x.amax(dim=0), x.amin(dim=1), (x > 0).amax(dim=1), (x > 0).amin(dim=1)

    _check(backend, f, [(torch.randint(-50, -10, (4, 6)),), (torch.randint(10, 50, (5, 7)),)])


@pytest.mark.torch
def test_argmax_argmin_all_negative(backend):

    def f(x):
        return x.argmax(dim=1), x.argmin(dim=0), x.argmax()

    _check(backend, f, [(-torch.rand(4, 6) - 1,), (-torch.rand(5, 7) - 1,)])


@pytest.mark.torch
def test_round_half_to_even(backend):

    def f(x):
        return torch.round(x)

    _check(
        backend,
        f,
        [(torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49]),), (torch.tensor([3.5, 4.5, -3.5, -4.5, 7.2]),)],
    )


@pytest.mark.torch
def test_integer_floor_division(backend):

    def f(a, b):
        return torch.div(a, b, rounding_mode="floor"), torch.div(a, b, rounding_mode="trunc"), a // b

    a = torch.tensor([7, -7, 7, -7, 6, -6, 0])
    b = torch.tensor([2, 2, -2, -2, 3, 3, 5])
    _check(backend, f, [(a, b), (a[:5].clone(), b[:5].clone())])


@pytest.mark.torch
def test_nan_propagation(backend):
    """
    maximum/minimum/clamp/relu propagate NaN from any operand; fmax/fmin ignore it. Requires DaCe's default compiler
    flags (no ``-ffast-math``, which lets the compiler assume that no value is NaN).
    """

    def f(x, y):
        clamped = torch.clamp(x, -0.5, 0.5), torch.clamp(x, min=y)
        return torch.maximum(x, y), torch.minimum(x, y), *clamped, torch.relu(x), torch.fmax(x, y), torch.fmin(x, y)

    nan = float("nan")
    x = torch.tensor([nan, 1.0, -1.0, nan, 0.25])
    y = torch.tensor([0.0, nan, 2.0, nan, -3.0])
    _check(backend, f, [(x, y), (x[1:].clone(), y[1:].clone())])


@pytest.mark.torch
def test_tensor_constants(backend):
    """Tensors created inside the program become compile-time constants of the SDFG (including bool and non-finite)."""

    def f(x):
        scale = torch.tensor([1.0, 2.0, float("inf"), -float("inf")])
        mask = torch.tensor([True, False, True, False])
        floor = torch.tensor([0.0, float("nan"), 1.0, 2.0])
        return torch.where(mask, x * scale, torch.tensor(-1.5)), torch.maximum(x, floor)

    _check(backend, f, [(torch.randn(3, 4),), (torch.randn(6, 4),)])


@pytest.mark.torch
def test_tensor_constant_class_weights(backend):

    def f(x, target):
        return F.cross_entropy(x, target, weight=torch.tensor([0.5, 1.0, 2.0, 1.5]))

    _check(
        backend, f, [(torch.randn(3, 4), torch.tensor([0, 1, 3])), (torch.randn(5, 4), torch.tensor([2, 2, 1, 0, 3]))]
    )


@pytest.mark.torch
def test_as_strided_with_storage_offset(backend):

    def f(x):
        v = x[1:]
        return torch.as_strided(v, (2, 2), (1, 1), v.storage_offset() + 1) * 2

    # Tensor.storage_offset() graph-breaks in Dynamo: the view reaches a second graph as an input with an offset
    _check(backend, f, [(torch.arange(12.0),), (torch.arange(20.0),)], expected_compiles=2)


if __name__ == "__main__":
    for test in (
        test_amax_amin_identity,
        test_amax_integer_and_bool,
        test_argmax_argmin_all_negative,
        test_round_half_to_even,
        test_integer_floor_division,
        test_nan_propagation,
        test_tensor_constants,
        test_tensor_constant_class_weights,
        test_as_strided_with_storage_offset,
    ):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, "ok")
