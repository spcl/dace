# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Graph capture without compilation: input sources, symbol names, guards, and running the captured graph."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dace.frontend.ml.torch.dynamo import capture  # noqa: E402

OFFSET = torch.tensor([0.25, -0.5, 1.0, 2.0])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.layers = nn.Sequential(nn.ReLU(), nn.Linear(8, 4))
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 4))
        self.alpha = 0.5

    def forward(self, x, n: int):
        return self.layers(self.fc1(x)) * self.scale * self.alpha + n


def _function_with_global(x):
    return x * 2 + OFFSET


def _sources(program):
    return {(ref.kind, ref.qualname, ref.dim) for ref in program.inputs}


@pytest.mark.torch
def test_capture_module_input_sources():
    model = _Model().eval()
    program = capture(model, torch.randn(3, 4), 2, dynamic_shapes={"x": {0: "batch"}})
    sources = _sources(program)
    assert {
        ("parameter", "fc1.weight", None),
        ("parameter", "fc1.bias", None),
        ("parameter", "layers.1.weight", None),
        ("parameter", "layers.1.bias", None),
        ("buffer", "scale", None),
        ("attribute", "alpha", None),
        ("argument", "x", None),
        ("argument", "n", None),
        ("size", "x", 0),
    } <= sources
    # The SDFG arguments are named after the sources
    sdfg = program.to_sdfg("capture_module_names")
    assert {"x", "fc1_weight", "fc1_bias", "layers_1_weight", "layers_1_bias", "scale", "alpha"} <= set(sdfg.arglist())
    assert "batch" in sdfg.free_symbols


@pytest.mark.torch
def test_capture_global_source():
    program = capture(_function_with_global, torch.randn(5, 4))
    assert ("global", "OFFSET", None) in _sources(program)
    assert "OFFSET" in program.to_sdfg("capture_global").arglist()


@pytest.mark.torch
def test_capture_guards_and_ranges():
    model = _Model().eval()
    batch = torch.export.Dim("batch", min=2, max=64)
    program = capture(model, torch.randn(3, 4), 2, dynamic_shapes={"x": {0: batch}})
    tensor_matches = {g.source.qualname for g in program.guards if g.kind == "TENSOR_MATCH"}
    assert {"x", "fc1.weight", "scale"} <= tensor_matches
    assert any(g.kind == "GRAD_MODE" and g.source is None for g in program.guards)
    assert program.symbol_ranges["batch"] == (2, 64)


@pytest.mark.torch
def test_capture_does_not_execute():
    calls = []

    def f(x):
        calls.append(len(calls))
        return torch.sin(x) * 2

    program = capture(f, torch.randn(6))
    assert calls == []
    assert program.graph is not None


@pytest.mark.torch
def test_capture_graph_break_raises():

    def f(x):
        y = x * 2
        torch._dynamo.graph_break()
        return y + 1

    with pytest.raises(Exception, match="graph_break"):
        capture(f, torch.randn(4))


@pytest.mark.torch
def test_capture_compile_and_bind():
    """The captured graph compiles once and runs for other sizes, with inputs bound from their sources."""
    torch.manual_seed(0)
    model = _Model().eval()
    program = capture(model, torch.randn(3, 4), 2, dynamic_shapes={"x": {0: "batch"}})
    run = program.compile("capture_compile_and_bind")
    with torch.no_grad():
        for batch, n in ((3, 2), (7, 5)):
            x = torch.randn(batch, 4)
            (out,) = run(x, n)
            torch.testing.assert_close(out, model(x, n))
        # Parameters are bound by reference: updates are visible without recapturing
        model.fc1.bias.add_(1.0)
        x = torch.randn(4, 4)
        (out,) = run(x, 1)
        torch.testing.assert_close(out, model(x, 1))


if __name__ == "__main__":
    test_capture_module_input_sources()
    test_capture_global_source()
    test_capture_guards_and_ranges()
    test_capture_does_not_execute()
    test_capture_graph_break_raises()
    test_capture_compile_and_bind()
