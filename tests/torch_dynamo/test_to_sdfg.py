# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ahead-of-time export of inference SDFGs from DaCe data descriptors (``dace.ml.to_sdfg`` / ``dace.ml.compile``)."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402

N = dace.symbol("N")


class _ConvNet(nn.Module):
    """A small conv/linear inference model with parameters and a buffer."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(4 * 2 * 2, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.mark.torch
def test_to_sdfg_names_and_symbol():
    torch.manual_seed(0)
    model = _ConvNet().eval()
    sdfg = dace.ml.to_sdfg(model, dace.float32[N, 3, 8, 10])

    # The input keeps the forward() parameter name, parameters and buffers their qualified names, and N its name.
    args = set(sdfg.arglist())
    assert "x" in args
    assert "conv_weight" in args and "conv_bias" in args
    assert "fc_weight" in args and "fc_bias" in args
    assert "N" in {str(s) for s in sdfg.free_symbols}
    assert isinstance(sdfg.name, str) and sdfg.name


@pytest.mark.torch
def test_to_sdfg_numerics_two_batches():
    torch.manual_seed(1)
    model = _ConvNet().eval()
    compiled = dace.ml.compile(model, dace.float32[N, 3, 8, 10])  # ahead-of-time overload
    for n in (2, 7):
        x = torch.randn(n, 3, 8, 10)
        with torch.no_grad():
            ref = model(x)
        out = compiled(x)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    # The ahead-of-time overload exposes the SDFG and the compiled SDFG.
    assert compiled.sdfg is dace.ml.sdfg_of(compiled)
    assert compiled.compiled_sdfg is not None


@pytest.mark.torch
def test_to_sdfg_custom_name_and_save(tmp_path):
    torch.manual_seed(2)
    model = _ConvNet().eval()
    sdfg = dace.ml.to_sdfg(model, dace.float32[N, 3, 8, 10], name="my_inference")
    assert sdfg.name == "my_inference"
    compiled = dace.ml.compile(model, dace.float32[N, 3, 8, 10], name="my_inference")
    path = str(tmp_path / "model.sdfgz")
    compiled.save(path)
    reloaded = dace.SDFG.from_file(path)
    assert reloaded.name == "my_inference"


@pytest.mark.torch
def test_plain_function():
    M = dace.symbol("M")

    def f(x, b):
        return torch.tanh(x) * 2.0 + b

    sdfg = dace.ml.to_sdfg(f, dace.float32[M, 6], dace.float32[6])
    assert "x" in sdfg.arglist() and "b" in sdfg.arglist()
    assert "M" in {str(s) for s in sdfg.free_symbols}

    compiled = dace.ml.compile(f, dace.float32[M, 6], dace.float32[6])
    for m in (3, 8):
        x, b = torch.randn(m, 6), torch.randn(6)
        out = compiled(x, b)
        torch.testing.assert_close(out, torch.tanh(x) * 2.0 + b, rtol=1e-4, atol=1e-5)


@pytest.mark.torch
def test_parameters_bound_at_call_time():
    """Changing a parameter after compiling is reflected without recompiling (parameters are bound per call)."""
    torch.manual_seed(3)
    model = nn.Linear(6, 4).eval()
    M = dace.symbol("M")
    compiled = dace.ml.compile(model, dace.float32[M, 6])
    x = torch.randn(5, 6)
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        model.weight.mul_(-1.0)
        model.bias.add_(2.0)
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_to_sdfg_names_and_symbol()
    test_to_sdfg_numerics_two_batches()
    test_plain_function()
    test_parameters_bound_at_call_time()
    print("ok")
