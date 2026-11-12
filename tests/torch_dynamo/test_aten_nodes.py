# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ATen operators as library nodes: generic nodes for operators without a lowering, convolution, dropout."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import DaceBackend, capture  # noqa: E402
from dace.libraries.torch import ATenOp, Convolution, NativeDropout  # noqa: E402
from dace.sdfg import nodes  # noqa: E402


def _library_nodes(sdfg, kind):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, kind)]


@torch.library.custom_op("dace_test::scaled_square", mutates_args=())
def scaled_square(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * x * scale


@scaled_square.register_fake
def _(x, scale):
    return torch.empty_like(x)


@pytest.mark.torch
def test_unknown_operator_becomes_library_node():
    """An operator DaCe has no lowering for is imported as an ``ATenOp`` and fails only at expansion, by name."""

    def f(x):
        return scaled_square(x + 1, 2.5)

    sdfg = capture(f, torch.randn(4, 6)).to_sdfg("unknown_operator")
    (node,) = _library_nodes(sdfg, ATenOp)
    assert node.target == "dace_test.scaled_square.default"
    assert node.arguments == {"scale": 2.5}
    assert set(node.in_connectors) == {"x"} and set(node.out_connectors) == {"output"}

    # The node survives serialization (also without torch: the node classes do not import it)
    restored = dace.SDFG.from_json(sdfg.to_json())
    (node,) = _library_nodes(restored, ATenOp)
    assert node.target == "dace_test.scaled_square.default" and node.arguments == {"scale": 2.5}

    with pytest.raises(NotImplementedError, match="dace_test.scaled_square.default"):
        sdfg.expand_library_nodes()


@pytest.mark.torch
def test_convolution_node():
    """Convolutions are ``Convolution`` nodes carrying the operator arguments (no padding or maps in the SDFG)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, 3, stride=2, padding=1, dilation=1, groups=1, bias=True).eval()
    sdfg = capture(conv, torch.randn(2, 3, 9, 11), dynamic_shapes={"input": {0: "N"}}).to_sdfg("conv_node")
    (node,) = _library_nodes(sdfg, Convolution)
    assert node.target == "aten.convolution.default"
    assert node.arguments["stride"] == [2, 2] and node.arguments["padding"] == [1, 1]
    assert node.arguments["groups"] == 1 and node.arguments["transposed"] is False
    assert set(node.in_connectors) == {"input", "weight", "bias"} and set(node.out_connectors) == {"output"}
    assert not any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.torch
def test_dropout_node():
    """Training-mode dropout is a ``NativeDropout`` node seeded by the program's random seed symbol."""
    drop = nn.Dropout(0.3).train()
    sdfg = capture(drop, torch.randn(4, 6)).to_sdfg("dropout_node")
    (node,) = _library_nodes(sdfg, NativeDropout)
    assert node.arguments["p"] == 0.3 and node.arguments["train"] is True
    assert str(node.seed) == "dace_random_seed" and "dace_random_seed" in sdfg.symbols
    # The mask is not used by the module, so simplification drops that output
    assert "output_0" in node.out_connectors


@pytest.mark.torch
def test_dropout_eval_is_identity():
    drop = nn.Dropout(0.3).eval()
    sdfg = capture(drop, torch.randn(4, 6)).to_sdfg("dropout_eval")
    assert not _library_nodes(sdfg, NativeDropout)


# ----------------------------------------------------------------------- execution (needs dace.libraries.ml)
def _ml_available() -> bool:
    try:
        import dace.libraries.ml  # noqa: F401
    except ImportError:
        return False
    return True


needs_ml = pytest.mark.skipif(not _ml_available(), reason="dace.libraries.ml is not available")


@pytest.mark.torch
@needs_ml
def test_convolution_runs():
    torch.manual_seed(0)
    conv = nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2, bias=True).eval()
    backend = DaceBackend()
    compiled = torch.compile(conv, backend=backend, dynamic=True)
    with torch.no_grad():
        for x in (torch.randn(2, 4, 10, 12), torch.randn(3, 4, 7, 13)):
            torch.testing.assert_close(compiled(x), conv(x), rtol=1e-4, atol=1e-4)
    assert backend.compile_count == 1
    assert _library_nodes(backend.last_sdfg, Convolution)


@pytest.mark.torch
@needs_ml
def test_dropout_runs():
    p = 0.3
    drop = nn.Dropout(p).train()
    backend = DaceBackend()
    compiled = torch.compile(drop, backend=backend, dynamic=True)
    x = torch.rand(64, 1000) + 0.5
    with torch.no_grad():
        y1 = compiled(x)
        y2 = compiled(x)
    kept = y1 != 0
    assert abs(kept.float().mean().item() - (1 - p)) < 0.02
    torch.testing.assert_close(y1[kept], x[kept] / (1 - p))
    assert not torch.equal(y1, y2), "two calls must draw different masks"
    torch.manual_seed(5)
    with torch.no_grad():
        a = compiled(x)
    torch.manual_seed(5)
    with torch.no_grad():
        b = compiled(x)
    assert torch.equal(a, b), "the mask must follow torch's generator"


@pytest.mark.torch
@needs_ml
def test_dropout_backward():
    p = 0.4
    drop = nn.Dropout(p).train()
    backend = DaceBackend()
    compiled = torch.compile(drop, backend=backend, dynamic=True)
    x = torch.randn(8, 50, requires_grad=True)
    y = compiled(x)
    y.sum().backward()
    expected = torch.where(y != 0, torch.full_like(x, 1 / (1 - p)), torch.zeros_like(x))
    torch.testing.assert_close(x.grad, expected)


if __name__ == "__main__":
    test_unknown_operator_becomes_library_node()
    test_convolution_node()
    test_dropout_node()
    test_dropout_eval_is_identity()
    if _ml_available():
        test_convolution_runs()
        test_dropout_runs()
        test_dropout_backward()
