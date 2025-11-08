# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("onnx", reason="ONNX not installed")

import numpy as np
import torch
import torch.nn as nn
import dace
from dace.frontend.ml.torch.module import DaceModule
from dace.transformation.onnx.constant_folding import ConstantFolding


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

    # Convert to DaCe and apply constant folding
    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="shape_test")

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

    # Run the model
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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

    result = dace_module(x)
    expected = x.unsqueeze(2).expand(-1, -1, 4, -1)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_range_constant_folding():
    """Test that Range nodes with constant inputs are folded."""

    class RangeModel(nn.Module):

        def forward(self, x):
            # Use arange which creates Range internally
            indices = torch.arange(0, x.size(0), dtype=torch.int64)
            return x[indices]

    model = RangeModel()
    x = torch.randn(5, 3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="range_test")

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

    result = dace_module(x)
    expected = model(x)

    assert torch.allclose(result, expected)


@pytest.mark.onnx
def test_multiple_operations_constant_folding():
    """Test constant folding when multiple operations can be folded."""

    class MultipleOpsModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.const_a = torch.tensor([1.0, 2.0, 3.0])
            self.const_b = torch.tensor([2.0, 2.0, 2.0])

        def forward(self, x):
            # Multiple operations that should all be folded
            mul_result = self.const_a * self.const_b  # Mul
            add_result = mul_result + self.const_b  # Add
            reshaped = add_result.reshape(1, 3)  # Reshape
            unsqueezed = reshaped.unsqueeze(0)  # Unsqueeze
            return x * unsqueezed

    model = MultipleOpsModel()
    x = torch.randn(1, 1, 3)

    dace_module = DaceModule(model, dummy_inputs=(x, ), simplify=False, sdfg_name="multiple_ops_test")

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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

    # Apply constant folding
    sdfg = dace_module.sdfg
    sdfg.apply_transformations_repeated(ConstantFolding)

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
    test_multiple_operations_constant_folding()
    test_trilu_upper_constant_folding()
    test_trilu_lower_constant_folding()
    test_trilu_with_diagonal_offset_constant_folding()
    test_cast_float_to_int_constant_folding()
    test_cast_int_to_float_constant_folding()
    test_cast_float32_to_float64_constant_folding()
