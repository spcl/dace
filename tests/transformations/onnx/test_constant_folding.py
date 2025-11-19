# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("onnx", reason="ONNX not installed")

import torch
import torch.nn as nn
from dace.frontend.ml.torch.module import DaceModule
from dace.transformation.onnx.constant_folding import ConstantFolding
from dace.libraries.onnx.nodes.onnx_op import ONNXOp


def count_onnx_ops(sdfg, op_names=None):
    """Count ONNX operation nodes in the SDFG.

    Args:
        sdfg: The SDFG to count nodes in
        op_names: Optional list of operation names to count. If None, counts all ONNX ops.

    Returns:
        Count of matching ONNX operation nodes
    """
    count = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, ONNXOp):
            if op_names is None or node.schema.name in op_names:
                count += 1
    return count


@pytest.mark.onnx
def test_shape_constant_folding():
    """Test that Shape nodes with fixed input shapes are folded."""

    class ShapeModel(nn.Module):

        def forward(self, x):
            # Create operation that uses Shape internally
            # expand uses Shape -> ConstantOfShape
            return x.unsqueeze(0).expand(2, -1, -1)

    model = ShapeModel()
    x = torch.randn(3, 5)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="shape_test")
    sdfg = dace_module.sdfg

    shape_count_before = count_onnx_ops(sdfg, ["Shape"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    shape_count_after = count_onnx_ops(sdfg, ["Shape"])
    if shape_count_before > 0:
        assert shape_count_after < shape_count_before, f"Expected Shape nodes to decrease after constant folding (before: {shape_count_before}, after: {shape_count_after})"

    result = dace_module(x)
    expected = x.unsqueeze(0).expand(2, -1, -1)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_constantofshape_constant_folding():
    """Test that ConstantOfShape nodes with constant shape inputs are folded."""

    class ConstantOfShapeModel(nn.Module):

        def forward(self, x):
            # Use expand which creates Shape -> ConstantOfShape internally
            return x.unsqueeze(2).expand(-1, -1, 4, -1)

    model = ConstantOfShapeModel()
    x = torch.randn(2, 3, 5)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="constantofshape_test")
    sdfg = dace_module.sdfg

    constantofshape_count_before = count_onnx_ops(sdfg, ["ConstantOfShape"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    constantofshape_count_after = count_onnx_ops(sdfg, ["ConstantOfShape"])
    if constantofshape_count_before > 0:
        assert constantofshape_count_after < constantofshape_count_before, f"Expected ConstantOfShape nodes to decrease after constant folding (before: {constantofshape_count_before}, after: {constantofshape_count_after})"

    result = dace_module(x)
    expected = x.unsqueeze(2).expand(-1, -1, 4, -1)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_range_constant_folding():
    """Test that Range nodes with constant inputs are folded."""

    class RangeModel(nn.Module):

        def forward(self, x):
            indices = torch.arange(0, x.size(0), dtype=torch.int64)
            return x[indices]

    model = RangeModel()
    x = torch.randn(5, 3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="range_test")
    sdfg = dace_module.sdfg

    range_count_before = count_onnx_ops(sdfg, ["Range"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    range_count_after = count_onnx_ops(sdfg, ["Range"])
    if range_count_before > 0:
        assert range_count_after < range_count_before, f"Expected Range nodes to decrease after constant folding (before: {range_count_before}, after: {range_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_mul_constant_folding():
    """Test that Mul nodes with all constant inputs are folded."""

    class MulModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
            self.const_b = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

        def forward(self, x):
            # Multiply two constants, then use result with input
            constant_mul = self.const_a * self.const_b
            return x + constant_mul

    model = MulModel()
    x = torch.randn(2, 2)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="mul_test")
    sdfg = dace_module.sdfg

    mul_count_before = count_onnx_ops(sdfg, ["Mul"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    mul_count_after = count_onnx_ops(sdfg, ["Mul"])
    if mul_count_before > 0:
        assert mul_count_after < mul_count_before, f"Expected Mul nodes to decrease after constant folding (before: {mul_count_before}, after: {mul_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_add_constant_folding():
    """Test that Add nodes with all constant inputs are folded."""

    class AddModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([4.0, 5.0, 6.0])

        def forward(self, x):
            constant_sum = self.const_a + self.const_b
            return x * constant_sum

    model = AddModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="add_test")
    sdfg = dace_module.sdfg

    add_count_before = count_onnx_ops(sdfg, ["Add"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    add_count_after = count_onnx_ops(sdfg, ["Add"])
    if add_count_before > 0:
        assert add_count_after < add_count_before, f"Expected Add nodes to decrease after constant folding (before: {add_count_before}, after: {add_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_sub_constant_folding():
    """Test that Sub nodes with all constant inputs are folded."""

    class SubModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([10.0, 20.0, 30.0])
            self.const_b = torch.tensor([1.0, 2.0, 3.0])

        def forward(self, x):
            constant_diff = self.const_a - self.const_b
            return x + constant_diff

    model = SubModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="sub_test")
    sdfg = dace_module.sdfg

    sub_count_before = count_onnx_ops(sdfg, ["Sub"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    sub_count_after = count_onnx_ops(sdfg, ["Sub"])
    if sub_count_before > 0:
        assert sub_count_after < sub_count_before, f"Expected Sub nodes to decrease after constant folding (before: {sub_count_before}, after: {sub_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_div_constant_folding():
    """Test that Div nodes with all constant inputs are folded."""

    class DivModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([10.0, 20.0, 30.0])
            self.const_b = torch.tensor([2.0, 4.0, 5.0])

        def forward(self, x):
            constant_div = self.const_a / self.const_b
            return x * constant_div

    model = DivModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="div_test")
    sdfg = dace_module.sdfg

    div_count_before = count_onnx_ops(sdfg, ["Div"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    div_count_after = count_onnx_ops(sdfg, ["Div"])
    if div_count_before > 0:
        assert div_count_after < div_count_before, f"Expected Div nodes to decrease after constant folding (before: {div_count_before}, after: {div_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_equal_constant_folding():
    """Test that Equal nodes with all constant inputs are folded."""

    class EqualModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([1.0, 3.0, 3.0])

        def forward(self, x):
            # Compare two constants
            mask = torch.eq(self.const_a, self.const_b)
            return torch.where(mask, x, x * 2)

    model = EqualModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="equal_test")
    sdfg = dace_module.sdfg

    equal_count_before = count_onnx_ops(sdfg, ["Equal"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    equal_count_after = count_onnx_ops(sdfg, ["Equal"])
    if equal_count_before > 0:
        assert equal_count_after < equal_count_before, f"Expected Equal nodes to decrease after constant folding (before: {equal_count_before}, after: {equal_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_greater_constant_folding():
    """Test that Greater nodes with all constant inputs are folded."""

    class GreaterModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([3.0, 2.0, 1.0])
            self.const_b = torch.tensor([1.0, 2.0, 3.0])

        def forward(self, x):
            mask = torch.gt(self.const_a, self.const_b)
            return torch.where(mask, x, x * 2)

    model = GreaterModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="greater_test")
    sdfg = dace_module.sdfg

    greater_count_before = count_onnx_ops(sdfg, ["Greater"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    greater_count_after = count_onnx_ops(sdfg, ["Greater"])
    if greater_count_before > 0:
        assert greater_count_after < greater_count_before, f"Expected Greater nodes to decrease after constant folding (before: {greater_count_before}, after: {greater_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_less_constant_folding():
    """Test that Less nodes with all constant inputs are folded."""

    class LessModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([3.0, 2.0, 1.0])

        def forward(self, x):
            mask = torch.lt(self.const_a, self.const_b)
            return torch.where(mask, x, x * 2)

    model = LessModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="less_test")
    sdfg = dace_module.sdfg

    less_count_before = count_onnx_ops(sdfg, ["Less"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    less_count_after = count_onnx_ops(sdfg, ["Less"])
    if less_count_before > 0:
        assert less_count_after < less_count_before, f"Expected Less nodes to decrease after constant folding (before: {less_count_before}, after: {less_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_greaterorequal_constant_folding():
    """Test that GreaterOrEqual nodes with all constant inputs are folded."""

    class GreaterOrEqualModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([3.0, 2.0, 1.0])
            self.const_b = torch.tensor([1.0, 2.0, 3.0])

        def forward(self, x):
            mask = torch.ge(self.const_a, self.const_b)
            return torch.where(mask, x, x * 2)

    model = GreaterOrEqualModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="greaterorequal_test")
    sdfg = dace_module.sdfg

    greaterorequal_count_before = count_onnx_ops(sdfg, ["GreaterOrEqual"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    greaterorequal_count_after = count_onnx_ops(sdfg, ["GreaterOrEqual"])
    if greaterorequal_count_before > 0:
        assert greaterorequal_count_after < greaterorequal_count_before, f"Expected GreaterOrEqual nodes to decrease after constant folding (before: {greaterorequal_count_before}, after: {greaterorequal_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_lessorequal_constant_folding():
    """Test that LessOrEqual nodes with all constant inputs are folded."""

    class LessOrEqualModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([3.0, 2.0, 1.0])

        def forward(self, x):
            mask = torch.le(self.const_a, self.const_b)
            return torch.where(mask, x, x * 2)

    model = LessOrEqualModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="lessorequal_test")
    sdfg = dace_module.sdfg

    lessorequal_count_before = count_onnx_ops(sdfg, ["LessOrEqual"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    lessorequal_count_after = count_onnx_ops(sdfg, ["LessOrEqual"])
    if lessorequal_count_before > 0:
        assert lessorequal_count_after < lessorequal_count_before, f"Expected LessOrEqual nodes to decrease after constant folding (before: {lessorequal_count_before}, after: {lessorequal_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_where_constant_folding():
    """Test that Where nodes with all constant inputs are folded."""

    class WhereModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.condition = torch.tensor([True, False, True])
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([4.0, 5.0, 6.0])

        def forward(self, x):
            # Where with constant inputs
            constant_result = torch.where(self.condition, self.const_a, self.const_b)
            return x + constant_result

    model = WhereModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="where_test")
    sdfg = dace_module.sdfg

    where_count_before = count_onnx_ops(sdfg, ["Where"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    where_count_after = count_onnx_ops(sdfg, ["Where"])
    if where_count_before > 0:
        assert where_count_after < where_count_before, f"Expected Where nodes to decrease after constant folding (before: {where_count_before}, after: {where_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_unsqueeze_constant_folding():
    """Test that Unsqueeze nodes with constant inputs are folded."""

    class UnsqueezeModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([1.0, 2.0, 3.0])

        def forward(self, x):
            # Unsqueeze a constant
            unsqueezed_const = self.const.unsqueeze(0).unsqueeze(2)
            # Use the result with input
            return x + unsqueezed_const

    model = UnsqueezeModel()
    x = torch.randn(1, 3, 1)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="unsqueeze_test")
    sdfg = dace_module.sdfg

    unsqueeze_count_before = count_onnx_ops(sdfg, ["Unsqueeze"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    unsqueeze_count_after = count_onnx_ops(sdfg, ["Unsqueeze"])
    if unsqueeze_count_before > 0:
        assert unsqueeze_count_after < unsqueeze_count_before, f"Expected Unsqueeze nodes to decrease after constant folding (before: {unsqueeze_count_before}, after: {unsqueeze_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_concat_constant_folding():
    """Test that Concat nodes with all constant inputs are folded."""

    class ConcatModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            self.const_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        def forward(self, x):
            # Concatenate two constants
            concatenated = torch.cat([self.const_a, self.const_b], dim=1)
            # Use result with input
            return x * concatenated

    model = ConcatModel()
    x = torch.randn(2, 4)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="concat_test")
    sdfg = dace_module.sdfg

    concat_count_before = count_onnx_ops(sdfg, ["Concat"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    concat_count_after = count_onnx_ops(sdfg, ["Concat"])
    if concat_count_before > 0:
        assert concat_count_after < concat_count_before, f"Expected Concat nodes to decrease after constant folding (before: {concat_count_before}, after: {concat_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_reshape_constant_folding():
    """Test that Reshape nodes with constant inputs are folded."""

    class ReshapeModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def forward(self, x):
            # Reshape a constant
            reshaped_const = self.const.reshape(3, 2)
            # Use result with input
            return x * reshaped_const

    model = ReshapeModel()
    x = torch.randn(3, 2)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="reshape_test")
    sdfg = dace_module.sdfg

    reshape_count_before = count_onnx_ops(sdfg, ["Reshape"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    reshape_count_after = count_onnx_ops(sdfg, ["Reshape"])
    if reshape_count_before > 0:
        assert reshape_count_after < reshape_count_before, f"Expected Reshape nodes to decrease after constant folding (before: {reshape_count_before}, after: {reshape_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_reshape_with_infer_dimension():
    """Test that Reshape nodes with -1 (infer dimension) are folded correctly."""

    class ReshapeInferModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def forward(self, x):
            # Reshape with -1 to infer dimension
            reshaped_const = self.const.reshape(-1, 2)
            # Use result with input
            return x * reshaped_const

    model = ReshapeInferModel()
    x = torch.randn(3, 2)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="reshape_infer_test")
    sdfg = dace_module.sdfg

    reshape_count_before = count_onnx_ops(sdfg, ["Reshape"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    reshape_count_after = count_onnx_ops(sdfg, ["Reshape"])
    if reshape_count_before > 0:
        assert reshape_count_after < reshape_count_before, f"Expected Reshape nodes to decrease after constant folding (before: {reshape_count_before}, after: {reshape_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_trilu_upper_constant_folding():
    """Test that Trilu nodes with upper triangular and constant inputs are folded."""

    class TriluUpperModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        def forward(self, x):
            # Apply upper triangular to constant
            triu_const = torch.triu(self.const, diagonal=0)
            # Use result with input
            return x * triu_const

    model = TriluUpperModel()
    x = torch.randn(3, 3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="trilu_upper_test")
    sdfg = dace_module.sdfg

    trilu_count_before = count_onnx_ops(sdfg, ["Trilu"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    trilu_count_after = count_onnx_ops(sdfg, ["Trilu"])
    if trilu_count_before > 0:
        assert trilu_count_after < trilu_count_before, f"Expected Trilu nodes to decrease after constant folding (before: {trilu_count_before}, after: {trilu_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_trilu_lower_constant_folding():
    """Test that Trilu nodes with lower triangular and constant inputs are folded."""

    class TriluLowerModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        def forward(self, x):
            # Apply lower triangular to constant
            tril_const = torch.tril(self.const, diagonal=0)
            # Use result with input
            return x + tril_const

    model = TriluLowerModel()
    x = torch.randn(3, 3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="trilu_lower_test")
    sdfg = dace_module.sdfg

    trilu_count_before = count_onnx_ops(sdfg, ["Trilu"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    trilu_count_after = count_onnx_ops(sdfg, ["Trilu"])
    if trilu_count_before > 0:
        assert trilu_count_after < trilu_count_before, f"Expected Trilu nodes to decrease after constant folding (before: {trilu_count_before}, after: {trilu_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_trilu_with_diagonal_offset_constant_folding():
    """Test that Trilu nodes with diagonal offset and constant inputs are folded."""

    class TriluOffsetModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])

        def forward(self, x):
            # Apply upper triangular with diagonal offset
            triu_const = torch.triu(self.const, diagonal=1)
            # Use result with input
            return x * triu_const

    model = TriluOffsetModel()
    x = torch.randn(3, 4)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="trilu_offset_test")
    sdfg = dace_module.sdfg

    trilu_count_before = count_onnx_ops(sdfg, ["Trilu"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    trilu_count_after = count_onnx_ops(sdfg, ["Trilu"])
    if trilu_count_before > 0:
        assert trilu_count_after < trilu_count_before, f"Expected Trilu nodes to decrease after constant folding (before: {trilu_count_before}, after: {trilu_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_cast_float_to_int_constant_folding():
    """Test that Cast nodes converting float to int with constant inputs are folded."""

    class CastFloatToIntModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([1.5, 2.7, 3.2])

        def forward(self, x):
            # Cast constant to int
            int_const = self.const.to(torch.int32)
            # Use result with input (cast back to float for computation)
            return x + int_const.float()

    model = CastFloatToIntModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="cast_float_to_int_test")
    sdfg = dace_module.sdfg

    cast_count_before = count_onnx_ops(sdfg, ["Cast"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    cast_count_after = count_onnx_ops(sdfg, ["Cast"])
    if cast_count_before > 0:
        assert cast_count_after < cast_count_before, f"Expected Cast nodes to decrease after constant folding (before: {cast_count_before}, after: {cast_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_cast_int_to_float_constant_folding():
    """Test that Cast nodes converting int to float with constant inputs are folded."""

    class CastIntToFloatModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([1, 2, 3], dtype=torch.int32)

        def forward(self, x):
            # Cast constant to float
            float_const = self.const.float()
            # Use result with input
            return x * float_const

    model = CastIntToFloatModel()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="cast_int_to_float_test")
    sdfg = dace_module.sdfg

    cast_count_before = count_onnx_ops(sdfg, ["Cast"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    cast_count_after = count_onnx_ops(sdfg, ["Cast"])
    if cast_count_before > 0:
        assert cast_count_after < cast_count_before, f"Expected Cast nodes to decrease after constant folding (before: {cast_count_before}, after: {cast_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_cast_float32_to_float64_constant_folding():
    """Test that Cast nodes converting between float types with constant inputs are folded."""

    class CastFloat32ToFloat64Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.const = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        def forward(self, x):
            # Cast constant to float64
            float64_const = self.const.double()
            # Use result with input (cast back for computation)
            return x + float64_const.float()

    model = CastFloat32ToFloat64Model()
    x = torch.randn(3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="cast_float32_to_float64_test")
    sdfg = dace_module.sdfg

    cast_count_before = count_onnx_ops(sdfg, ["Cast"])

    sdfg.apply_transformations_repeated(ConstantFolding)

    cast_count_after = count_onnx_ops(sdfg, ["Cast"])
    if cast_count_before > 0:
        assert cast_count_after < cast_count_before, f"Expected Cast nodes to decrease after constant folding (before: {cast_count_before}, after: {cast_count_after})"

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


if __name__ == "__main__":
    test_shape_constant_folding()
    test_constantofshape_constant_folding()
    test_range_constant_folding()
    test_mul_constant_folding()
    test_add_constant_folding()
    test_sub_constant_folding()
    test_div_constant_folding()
    test_equal_constant_folding()
    test_greater_constant_folding()
    test_less_constant_folding()
    test_greaterorequal_constant_folding()
    test_lessorequal_constant_folding()
    test_where_constant_folding()
    test_unsqueeze_constant_folding()
    test_concat_constant_folding()
    test_reshape_constant_folding()
    test_reshape_with_infer_dimension()
    test_trilu_upper_constant_folding()
    test_trilu_lower_constant_folding()
    test_trilu_with_diagonal_offset_constant_folding()
    test_cast_float_to_int_constant_folding()
    test_cast_int_to_float_constant_folding()
    test_cast_float32_to_float64_constant_folding()
