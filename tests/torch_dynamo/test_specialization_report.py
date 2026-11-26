# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Specialization reporting: a dimension the user named in ``dynamic_shapes`` that Dynamo nonetheless pins to a constant
while tracing (e.g. a Python ``for`` loop over ``x.shape[0]``) must be reported, not silently accepted.
"""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo import capture  # noqa: E402
from dace.frontend.ml.torch.dynamo.capture import CaptureError  # noqa: E402


def _chunked(x):
    """A Python ``for`` loop over a dimension: Dynamo evaluates ``x.shape[0]`` and specializes it to its value."""
    acc = x
    for _ in range(0, x.shape[0], 4):
        acc = acc + 1
    return acc


def _plain(x):
    """Nothing reads the symbolic size, so the dimension stays a symbol."""
    return x * 2


@pytest.mark.torch
def test_named_dimension_specialized_raises():
    torch._dynamo.reset()
    with pytest.raises(CaptureError, match=r"N was specialized to 24"):
        capture(_chunked, torch.randn(24, 5), dynamic_shapes={"x": {0: "N"}})


@pytest.mark.torch
def test_all_mode_specialization_is_reported_not_raised():
    torch._dynamo.reset()
    program = capture(_chunked, torch.randn(24, 5), dynamic_shapes="all")
    assert program.specializations, "the specialized dimension should be reported under 'all'"
    specialized = next(s for s in program.specializations if s.value == 24)
    assert specialized.strict is False
    assert specialized.name == "x_dim0" and specialized.symbol
    assert "chunked" in specialized.origin


@pytest.mark.torch
def test_symbolic_dimension_has_no_specialization():
    torch._dynamo.reset()
    program = capture(_plain, torch.randn(24, 5), dynamic_shapes={"x": {0: "N"}})
    assert program.specializations == []
    assert set(program.symbol_names.values()) == {"N"}


if __name__ == "__main__":
    test_named_dimension_specialized_raises()
    test_all_mode_specialization_is_reported_not_raised()
    test_symbolic_dimension_has_no_specialization()
    print("ok")
