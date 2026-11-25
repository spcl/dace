# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared index/_unsafe_index lowering: broadcasting, layout, and 3D upsampling."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


@pytest.mark.torch
@pytest.mark.parametrize("unsafe", [False, True], ids=["index", "unsafe_index"])
@pytest.mark.parametrize(
    "layout", ["consecutive", "separated", "scalar_index", "scalar_output", "strided", "singleton_dims", "unit"]
)
def test_integer_advanced_indexing(backend, unsafe, layout):
    op = torch.ops.aten._unsafe_index.Tensor if unsafe else torch.ops.aten.index.Tensor
    x = torch.arange(3 * 4 * 5 * 6, dtype=torch.float32).reshape(3, 4, 5, 6)
    if layout == "consecutive":
        i, j = torch.tensor([[0], [-1]]), torch.tensor([[1, -1, 2]])
        positions = (None, 0, 1)
    elif layout == "separated":
        i, j = torch.tensor([[0], [-1]]), torch.tensor([[1, -1, 2]])
        positions = (0, None, 1)
    elif layout == "scalar_index":
        i, j = torch.tensor(-1, dtype=torch.int32), torch.tensor([0, -1, 2])
        positions = (None, 0, 1)
    elif layout == "scalar_output":
        x = x[0, 0]
        i, j = torch.tensor(-1), torch.tensor(2)
        positions = (0, 1)
    elif layout == "strided":
        x = x.permute(3, 1, 0, 2)
        i, j = torch.tensor([[0], [-1]]), torch.tensor([[1, -1, 2]])
        positions = (0, None, 1)
    elif layout == "singleton_dims":
        x = x[:1, :, :1, :]
        i, j = torch.tensor([[0], [-1]]), torch.tensor([[0, -1, 0]])
        positions = (None, 0, 1)
    else:
        x = torch.tensor([[[3.0]]])
        i, j = torch.tensor([[0], [-1]]), torch.tensor([[0, -1, 0]])
        positions = (None, 0, 1)

    def f(x, i, j):
        tensors = (i, j)
        indices = [None if p is None else tensors[p] for p in positions]
        return op(x, indices)

    compiled = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        reference = torch.ops.aten.index.Tensor(x, [None if p is None else (i, j)[p] for p in positions])
        torch.testing.assert_close(compiled(x, i, j), reference, rtol=0, atol=0)
    assert backend.compile_count == 1


@pytest.mark.torch
def test_nearest_upsample3d_dynamic(backend):
    def f(x):
        return F.interpolate(x, size=(7, 9, 11), mode="nearest")

    compiled = torch.compile(f, backend=backend, dynamic=True, fullgraph=True)
    with torch.no_grad():
        for shape in ((2, 3, 4, 5, 6), (5, 3, 6, 8, 10)):
            x = torch.randn(shape)
            torch.testing.assert_close(compiled(x), f(x), rtol=0, atol=0)
    assert backend.compile_count == 1
