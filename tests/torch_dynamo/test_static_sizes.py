# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Sizes that are static in DaCe stay static while a module is captured from a ``@dace.program`` (only the DaCe symbols
are Dynamo symbols), and tracing failures are reported as short, concrete ``CaptureError`` messages.
"""

import re

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import capture, convertible  # noqa: E402
from dace.frontend.ml.torch.dynamo.capture import CaptureError  # noqa: E402
from dace.frontend.ml.torch.dynamo.symbols import free_symbol_names  # noqa: E402

N = dace.symbol("N")

_DYNAMO_SYMBOL = re.compile(r"\bs\d+\b")


class _ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4 * 30 * 30, 5)

    def forward(self, x):
        return self.fc(self.conv(x).relu().flatten(1))


class _TooSmall(nn.Module):
    """A 7x7 convolution with stride 2 followed by a 3x3 pool: an 8x8 input leaves nothing to pool."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 7, stride=2)

    def forward(self, x):
        return nn.functional.max_pool2d(self.conv(x), 3, 2)


class _Shared(nn.Module):
    def forward(self, a, b, n: int):
        return a + b * n


def _placeholder_sizes(program):
    sizes = []
    for node in program.graph.graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor):
            sizes.extend(node.meta["val"].shape)
    return sizes


@pytest.mark.torch
def test_program_static_sizes_stay_static():
    torch.manual_seed(0)
    model = _ConvNet().eval()

    @dace.program
    def prog(x: dace.float32[N, 3, 32, 32]):
        return model(x)

    for rows in (2, 3):
        x = np.random.rand(rows, 3, 32, 32).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(prog(x), expected, rtol=1e-4, atol=1e-5)

    program = convertible.as_sdfg_convertible(model).program
    assert set(program.symbol_names.values()) == {"N"}
    (batch_symbol,) = program.symbol_names  # Dynamo's name of N
    for size in _placeholder_sizes(program):
        if isinstance(size, torch.SymInt):
            assert str(size.node.expr) == batch_symbol, f"static size traced symbolically: {size}"
    # Only the user's symbol appears in the assumptions and ranges Dynamo reports
    for relation in program.shape_guards:
        assert not isinstance(relation, str), relation
        assert free_symbol_names(relation) <= {"N"}, relation
    assert set(program.symbol_ranges) == {"N"}


@pytest.mark.torch
def test_program_shared_symbol_and_integer_argument():
    module = _Shared()

    @dace.program
    def prog(a: dace.float32[N, 6], b: dace.float32[N, 6], n: dace.int64):
        return module(a, b, n)

    a, b = np.random.rand(4, 6).astype(np.float32), np.random.rand(4, 6).astype(np.float32)
    for n in (3, 5):
        np.testing.assert_allclose(prog(a, b, n), a + b * n, rtol=1e-6)
    program = convertible.as_sdfg_convertible(module).program
    # One symbol for N (both arrays), one symbolic integer for n, nothing for the static 6
    assert sorted(program.symbol_names.values()) == ["N", "n"]
    assert any(ref.kind == "argument" and ref.qualname == "n" for ref in program.inputs)
    assert all(size == 6 for size in _placeholder_sizes(program) if isinstance(size, int))


@pytest.mark.torch
def test_capture_named_specification_keeps_other_sizes_static():
    model = nn.Linear(4, 3)
    program = capture(model, torch.randn(5, 4), dynamic_shapes={"input": {0: "batch"}})
    sizes = _placeholder_sizes(program)
    assert [s for s in sizes if isinstance(s, torch.SymInt)] and all(
        isinstance(s, int) or str(s.node.expr) in program.symbol_names for s in sizes
    )
    assert set(program.symbol_names.values()) == {"batch"}
    assert set(program.symbol_ranges) == {"batch"}


@pytest.mark.torch
def test_capture_error_from_program_is_concrete_and_short():
    model = _TooSmall().eval()

    @dace.program
    def prog(x: dace.float32[N, 3, 8, 8]):
        return model(x)

    with pytest.raises(CaptureError) as info:
        prog.to_sdfg()
    message = str(info.value)
    assert "_TooSmall" in message and "Output size is too small" in message
    assert "(4x1x1)" in message.split("\n")[0]  # Concrete sizes (the 8x8 input after the convolution), no symbols
    assert not _DYNAMO_SYMBOL.search(message), message
    assert "from user code:" in message and "max_pool2d" in message
    assert "TORCHDYNAMO_VERBOSE=1" in message
    assert message.count("\n") < 20
    cause = info.value.__cause__
    assert cause is not None
    if not torch._dynamo.config.verbose:
        assert cause.__traceback__ is None


@pytest.mark.torch
def test_capture_error_names_user_symbols():

    def f(x):
        return x.reshape(x.shape[0] * x.shape[1] + 1)  # Never a valid reshape

    with pytest.raises(CaptureError) as info:
        capture(f, torch.randn(5, 4), dynamic_shapes={"x": {0: "rows", 1: "cols"}})
    message = str(info.value)
    assert "rows" in message and "cols" in message, message
    assert not _DYNAMO_SYMBOL.search(message), message


if __name__ == "__main__":
    test_program_static_sizes_stay_static()
    test_program_shared_symbol_and_integer_argument()
    test_capture_named_specification_keeps_other_sizes_static()
    test_capture_error_from_program_is_concrete_and_short()
    test_capture_error_names_user_symbols()
