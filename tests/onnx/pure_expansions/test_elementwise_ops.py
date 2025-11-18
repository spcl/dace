# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import dace
import dace.libraries.onnx as donnx
from scipy import special


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.onnx
def test_log(sdfg_name):
    """Test Log operation."""

    @dace.program
    def log_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXLog(input=inp, output=result)
        return result

    log_prog.__name__ = sdfg_name

    inp = np.abs(np.random.randn(5, 5).astype(np.float32)) + 0.1

    sdfg = log_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.log(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_exp(sdfg_name):
    """Test Exp operation."""

    @dace.program
    def exp_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXExp(input=inp, output=result)
        return result

    exp_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32) * 0.5

    sdfg = exp_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.exp(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_sin(sdfg_name):
    """Test Sin operation."""

    @dace.program
    def sin_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXSin(input=inp, output=result)
        return result

    sin_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = sin_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.sin(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_cos(sdfg_name):
    """Test Cos operation."""

    @dace.program
    def cos_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXCos(input=inp, output=result)
        return result

    cos_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = cos_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.cos(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_tanh(sdfg_name):
    """Test Tanh operation."""

    @dace.program
    def tanh_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXTanh(input=inp, output=result)
        return result

    tanh_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = tanh_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.tanh(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_equal():
    """Test Equal operator with DaCe ONNX frontend."""

    @dace
    def equal_prog(a: dace.float32[5], b: dace.float32[5]):
        out = dace.define_local([5], dace.bool_)
        donnx.ONNXEqual(A=a, B=b, C=out)
        return out

    A = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    B = np.array([1, 2, 4, 4, 6], dtype=np.float32)
    result = equal_prog(a=A.copy(), b=B.copy())
    expected = np.equal(A, B)

    np.testing.assert_array_equal(result, expected, err_msg="Equal output mismatch")


@pytest.mark.onnx
def test_erf(sdfg_name):
    """Test Erf operation."""

    @dace.program
    def erf_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXErf(input=inp, output=result)
        return result

    erf_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = erf_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = special.erf(inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_neg(sdfg_name):
    """Test Neg operation."""

    @dace.program
    def neg_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXNeg(X=inp, Y=result)
        return result

    neg_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = neg_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = -inp

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_relu(sdfg_name):
    """Test Relu operation."""

    @dace.program
    def relu_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXRelu(X=inp, Y=result)
        return result

    relu_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = relu_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.maximum(inp, 0)

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3])
def test_leaky_relu(alpha, sdfg_name):
    """Test LeakyRelu operation."""

    @dace.program
    def leaky_relu_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXLeakyRelu(X=inp, Y=result, alpha=alpha)
        return result

    leaky_relu_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = leaky_relu_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.where(inp > 0, inp, alpha * inp)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_sigmoid(sdfg_name):
    """Test Sigmoid operation."""

    @dace.program
    def sigmoid_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXSigmoid(X=inp, Y=result)
        return result

    sigmoid_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = sigmoid_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = 1 / (1 + np.exp(-inp))

    assert_allclose(result, expected, atol=1e-6)


@pytest.mark.onnx
def test_softplus(sdfg_name):
    """Test Softplus operation."""

    @dace.program
    def softplus_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXSoftplus(X=inp, Y=result)
        return result

    softplus_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = softplus_prog.to_sdfg()

    result = sdfg(inp=inp)
    expected = np.log(1 + np.exp(inp))

    assert_allclose(result, expected, atol=1e-6)


@pytest.mark.onnx
@pytest.mark.parametrize("shape_a, shape_b", [
    ([5, 5], [5, 5]),
    ([5, 1], [5, 5]),
    ([1, 5], [5, 5]),
    ([5], [5, 5]),
])
def test_add(shape_a, shape_b, sdfg_name):
    """Test Add operation with broadcasting."""

    @dace.program
    def add_prog(A: dace.float32[shape_a], B: dace.float32[shape_b]):
        result = dace.define_local(shape_b, dace.float32)
        donnx.ONNXAdd(A=A, B=B, C=result)
        return result

    add_prog.__name__ = sdfg_name

    A = np.random.randn(*shape_a).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32)

    sdfg = add_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = A + B

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("shape_a, shape_b", [
    ([5, 5], [5, 5]),
    ([5, 5], [1, 5]),
])
def test_sub(shape_a, shape_b, sdfg_name):
    """Test Sub operation with broadcasting."""

    @dace.program
    def sub_prog(A: dace.float32[shape_a], B: dace.float32[shape_b]):
        result = dace.define_local(shape_a, dace.float32)
        donnx.ONNXSub(A=A, B=B, C=result)
        return result

    sub_prog.__name__ = sdfg_name

    A = np.random.randn(*shape_a).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32)

    sdfg = sub_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = A - B

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("shape_a, shape_b", [
    ([5, 5], [5, 5]),
    ([5, 1], [5, 5]),
])
def test_mul(shape_a, shape_b, sdfg_name):
    """Test Mul operation with broadcasting."""

    @dace.program
    def mul_prog(A: dace.float32[shape_a], B: dace.float32[shape_b]):
        result = dace.define_local(shape_b, dace.float32)
        donnx.ONNXMul(A=A, B=B, C=result)
        return result

    mul_prog.__name__ = sdfg_name

    A = np.random.randn(*shape_a).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32)

    sdfg = mul_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = A * B

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("shape_a, shape_b", [
    ([5, 5], [5, 5]),
    ([5, 5], [1, 5]),
])
def test_div(shape_a, shape_b, sdfg_name):
    """Test Div operation with broadcasting."""

    @dace.program
    def div_prog(A: dace.float32[shape_a], B: dace.float32[shape_b]):
        result = dace.define_local(shape_a, dace.float32)
        donnx.ONNXDiv(A=A, B=B, C=result)
        return result

    div_prog.__name__ = sdfg_name

    A = np.random.randn(*shape_a).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32) + 0.5  # Avoid division by zero

    sdfg = div_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = A / B

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("shape_a, shape_b", [
    ([5, 5], [5, 5]),
    ([5, 1], [1, 5]),
])
def test_pow(shape_a, shape_b, sdfg_name):
    """Test Pow operation with broadcasting."""

    @dace.program
    def pow_prog(A: dace.float32[shape_a], B: dace.float32[shape_b]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXPow(X=A, Y=B, Z=result)
        return result

    pow_prog.__name__ = sdfg_name

    A = np.abs(np.random.randn(*shape_a).astype(np.float32)) + 0.1
    B = np.random.randn(*shape_b).astype(np.float32) * 0.5  # Keep exponents small

    sdfg = pow_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = np.power(A, B)

    assert_allclose(result, expected, atol=1e-5)


@pytest.mark.onnx
@pytest.mark.parametrize("min_val, max_val", [
    (-1.0, 1.0),
    (0.0, 1.0),
    (-5.0, 5.0),
], ids=["neg1_1", "0_1", "neg5_5"])
def test_clip_dynamic(min_val, max_val, sdfg_name):
    """Test Clip operation with dynamic min/max."""

    min_arr = np.array([min_val], dtype=np.float32)
    max_arr = np.array([max_val], dtype=np.float32)

    @dace.program
    def clip_prog(inp: dace.float32[5, 5], min_val: dace.float32[1], max_val: dace.float32[1]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXClip(input=inp, min=min_val, max=max_val, output=result)
        return result

    clip_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32) * 5

    sdfg = clip_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, min_val=min_arr, max_val=max_arr)
    expected = np.clip(inp, min_val, max_val)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_clip_constant_bounds(sdfg_name):
    """Test Clip operation with constant min/max bounds."""

    # Test with constant values embedded
    min_const = -1.0
    max_const = 1.0

    @dace.program
    def clip_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        min_arr = dace.define_local([1], dace.float32)
        max_arr = dace.define_local([1], dace.float32)
        min_arr[0] = min_const
        max_arr[0] = max_const
        donnx.ONNXClip(input=inp, min=min_arr, max=max_arr, output=result)
        return result

    clip_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32) * 5

    sdfg = clip_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)
    expected = np.clip(inp, min_const, max_const)

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize(
    "shape_min, shape_max",
    [
        ([1], [1]),  # Scalar min/max
        ([5, 1], [1, 5]),  # Broadcasting min/max
        ([1, 5], [5, 1]),  # Different broadcasting patterns
    ],
    ids=["scalar", "broadcast_1", "broadcast_2"])
def test_clip_broadcasting(shape_min, shape_max, sdfg_name):
    """Test Clip operation with broadcasting on min/max."""

    min_arr = np.random.randn(*shape_min).astype(np.float32) - 2
    max_arr = np.random.randn(*shape_max).astype(np.float32) + 2

    @dace.program
    def clip_prog(inp: dace.float32[5, 5], min_val: dace.float32[shape_min], max_val: dace.float32[shape_max]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXClip(input=inp, min=min_val, max=max_val, output=result)
        return result

    clip_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32) * 5

    sdfg = clip_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, min_val=min_arr, max_val=max_arr)
    expected = np.clip(inp, min_arr, max_arr)

    assert_allclose(result, expected)


if __name__ == "__main__":
    test_log(sdfg_name="test_log")
    test_exp(sdfg_name="test_exp")
    test_sin(sdfg_name="test_sin")
    test_cos(sdfg_name="test_cos")
    test_tanh(sdfg_name="test_tanh")
    test_erf(sdfg_name="test_erf")
    test_neg(sdfg_name="test_neg")

    test_relu(sdfg_name="test_relu")
    test_leaky_relu(alpha=0.01, sdfg_name="test_leaky_relu_0.01")
    test_leaky_relu(alpha=0.1, sdfg_name="test_leaky_relu_0.1")
    test_leaky_relu(alpha=0.3, sdfg_name="test_leaky_relu_0.3")
    test_sigmoid(sdfg_name="test_sigmoid")
    test_softplus(sdfg_name="test_softplus")

    test_add(shape_a=[5, 5], shape_b=[5, 5], sdfg_name="test_add_1")
    test_add(shape_a=[5, 1], shape_b=[5, 5], sdfg_name="test_add_2")
    test_add(shape_a=[1, 5], shape_b=[5, 5], sdfg_name="test_add_3")
    test_add(shape_a=[5], shape_b=[5, 5], sdfg_name="test_add_4")

    test_sub(shape_a=[5, 5], shape_b=[5, 5], sdfg_name="test_sub_1")
    test_sub(shape_a=[5, 5], shape_b=[1, 5], sdfg_name="test_sub_2")

    test_mul(shape_a=[5, 5], shape_b=[5, 5], sdfg_name="test_mul_1")
    test_mul(shape_a=[5, 1], shape_b=[5, 5], sdfg_name="test_mul_2")

    test_div(shape_a=[5, 5], shape_b=[5, 5], sdfg_name="test_div_1")
    test_div(shape_a=[5, 5], shape_b=[1, 5], sdfg_name="test_div_2")

    test_pow(shape_a=[5, 5], shape_b=[5, 5], sdfg_name="test_pow_1")
    test_pow(shape_a=[5, 1], shape_b=[1, 5], sdfg_name="test_pow_2")

    test_clip_dynamic(min_val=-1.0, max_val=1.0, sdfg_name="test_clip_dynamic_1")
    test_clip_dynamic(min_val=0.0, max_val=1.0, sdfg_name="test_clip_dynamic_2")
    test_clip_dynamic(min_val=-5.0, max_val=5.0, sdfg_name="test_clip_dynamic_3")

    test_clip_constant_bounds(sdfg_name="test_clip_constant_bounds")

    test_clip_broadcasting(shape_min=[1], shape_max=[1], sdfg_name="test_clip_broadcasting_1")
    test_clip_broadcasting(shape_min=[5, 1], shape_max=[1, 5], sdfg_name="test_clip_broadcasting_2")
    test_clip_broadcasting(shape_min=[1, 5], shape_max=[5, 1], sdfg_name="test_clip_broadcasting_3")
