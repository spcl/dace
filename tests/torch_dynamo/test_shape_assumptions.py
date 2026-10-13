# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shape assumptions (Dynamo shape guards) of modules captured from a ``@dace.program``: tracing heuristics of torch must
not specialize the user's symbols, and the assumptions a model really makes are reported with their origin.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import capture, convertible  # noqa: E402
from dace.frontend.ml.torch.dynamo.symbols import free_symbol_names  # noqa: E402

N = dace.symbol("N")

#: ATen's CPU convolution backend heuristics compare the batch size with this constant
_CONV_BACKEND_THRESHOLD = 16


class _ConvNet(nn.Module):
    """conv -> relu -> flatten -> linear: nothing in it depends on the batch size."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4 * 6 * 8, 5)

    def forward(self, x):
        return self.fc(self.conv(x).relu().flatten(1))


class _Branchy(nn.Module):
    """A module whose result really depends on the batch size."""

    def forward(self, x):
        if x.shape[0] > 4:
            return x * 2
        return x + 1


def _assumptions_on(program, symbol: str):
    return [
        a for a in program.assumptions if not isinstance(a.relation, str) and symbol in free_symbol_names(a.relation)
    ]


@pytest.mark.torch
def test_capture_conv_records_no_batch_heuristic():
    program = capture(_ConvNet().eval(), torch.randn(2, 3, 8, 10), dynamic_shapes={"x": {0: "N"}})
    assert program.symbol_names and set(program.symbol_names.values()) == {"N"}
    assert not _assumptions_on(program, "N"), [str(a) for a in program.assumptions]
    assert program.symbol_ranges["N"] == (2, None)
    # The backend selection of torch is restored after capturing
    assert getattr(torch._C._select_conv_backend, "__wrapped__", None) is None


@pytest.mark.torch
def test_program_conv_net_runs_on_both_sides_of_the_threshold():
    torch.manual_seed(0)
    model = _ConvNet().eval()

    @dace.program
    def prog(x: dace.float32[N, 3, 8, 10]):
        return model(x)

    programs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for batch in (2, _CONV_BACKEND_THRESHOLD + 4):
            x = np.random.rand(batch, 3, 8, 10).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(prog(x), expected, rtol=1e-4, atol=1e-5)
            programs.append(convertible.as_sdfg_convertible(model).program)
    assert programs[0] is programs[1], "the module was captured again for the larger batch"
    assert not _assumptions_on(programs[0], "N"), [str(a) for a in programs[0].assumptions]
    assert programs[0].symbol_ranges["N"] == (2, None)
    assert not [w for w in caught if "assumes" in str(w.message)], [str(w.message) for w in caught]


@pytest.mark.torch
def test_real_assumption_is_reported_with_origin_and_range():
    model = _Branchy()

    @dace.program
    def prog(x: dace.float32[N, 3]):
        return model(x)

    x = np.random.rand(3, 3).astype(np.float32)
    with pytest.warns(UserWarning, match="assumes N <= 4") as record:
        np.testing.assert_allclose(prog(x), x + 1)
    (message,) = [str(w.message) for w in record if "assumes" in str(w.message)]
    assert "if x.shape[0] > 4" in message and "in forward" in message, message
    assert "N to [2, 4]" in message, message
    program = convertible.as_sdfg_convertible(model).program
    (assumption,) = _assumptions_on(program, "N")
    assert str(assumption.relation) == "N <= 4" and "forward" in assumption.origin
    assert program.symbol_ranges["N"] == (2, 4)


@pytest.mark.torch
def test_compiled_conv_net_does_not_recompile_at_the_threshold():
    """The torch.compile route traces without the backend heuristic too: one SDFG serves both sides of it."""
    import dace.ml

    torch.manual_seed(0)
    model = _ConvNet().eval()
    compiled = dace.ml.compile(model)
    with torch.no_grad():
        for batch in (2, _CONV_BACKEND_THRESHOLD + 4):
            x = torch.randn(batch, 3, 8, 10)
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-5)
    assert compiled._dace_backend.compile_count == 1


if __name__ == "__main__":
    test_capture_conv_records_no_batch_heuristic()
    test_program_conv_net_runs_on_both_sides_of_the_threshold()
    test_real_assumption_is_reported_with_origin_and_range()
    test_compiled_conv_net_does_not_recompile_at_the_threshold()
