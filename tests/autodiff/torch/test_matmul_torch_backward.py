"""
Comprehensive tests for MatMul backward pass using PyTorch and DaCe modules.

This test suite validates the automatic differentiation of matrix multiplication
operations using DaCe's DaceModule with PyTorch. It covers all major cases:
- Basic 2D matrix multiplication
- Batched matrix multiplication
- Broadcasting scenarios
- 1D vector operations (vector-vector, matrix-vector, vector-matrix)
- Mixed dimensional inputs
- Complex chains of operations
"""

import numpy as np
import pytest
import copy

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
import torch.nn.functional as F

from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close

##################################
# Testing utilities
##################################


def run_matmul_test(
    module: torch.nn.Module,
    sdfg_name: str,
    input_shapes: dict,
    use_sum: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    auto_optimize: bool = False,
):
    """
    Run a MatMul backward test comparing PyTorch and DaCe implementations.

    Args:
        module: PyTorch module to test
        sdfg_name: Name for the generated SDFG
        input_shapes: Dict mapping input names to their shapes
        use_sum: If True, use sum() for scalar loss, otherwise use max()
        rtol: Relative tolerance for gradient comparison
        atol: Absolute tolerance for gradient comparison
        auto_optimize: Whether to auto-optimize the SDFG
    """
    # Create a copy for DaCe
    dace_module_model = copy.deepcopy(module)

    # Generate random inputs
    pytorch_inputs = {}
    dace_inputs = {}

    for name, shape in input_shapes.items():
        input_value = torch.rand(*shape, dtype=torch.float32)

        pytorch_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
        pytorch_input.copy_(input_value)
        pytorch_input.requires_grad = True
        pytorch_inputs[name] = pytorch_input

        dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
        dace_input.copy_(input_value)
        dace_input.requires_grad = True
        dace_inputs[name] = dace_input

    # Run PyTorch forward and backward
    if len(pytorch_inputs) == 1:
        pytorch_output = module(list(pytorch_inputs.values())[0])
    else:
        pytorch_output = module(*pytorch_inputs.values())

    if use_sum:
        pytorch_loss = pytorch_output.sum()
    else:
        pytorch_loss = pytorch_output.max()
    pytorch_loss.backward()

    # Create DaCe module with backward support
    dace_module = DaceModule(
        dace_module_model,
        simplify=True,
        backward=True,
        sdfg_name=sdfg_name,
        auto_optimize=auto_optimize,
    )

    # Run DaCe forward and backward
    if len(dace_inputs) == 1:
        dace_output = dace_module(list(dace_inputs.values())[0])
    else:
        dace_output = dace_module(*dace_inputs.values())

    if use_sum:
        dace_loss = dace_output.sum()
    else:
        dace_loss = dace_output.max()
    dace_loss.backward()

    # Compare gradients for all inputs
    for name in input_shapes.keys():
        torch_tensors_close(f"gradient_{name}", pytorch_inputs[name].grad, dace_inputs[name].grad, rtol=rtol, atol=atol)

    # Compare gradients for all parameters
    for (name, pytorch_param), (dace_name, dace_param) in zip(module.named_parameters(),
                                                              dace_module.named_parameters()):
        assert 'model.' + name == dace_name
        if pytorch_param.grad is not None:
            torch_tensors_close(name, pytorch_param.grad, dace_param.grad, rtol=rtol, atol=atol)


##################################
# Basic 2D Matrix Multiplication Tests
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_2d_basic(sdfg_name: str):
    """Test basic 2D matrix multiplication: (m, k) @ (k, n) -> (m, n)"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (5, 4), "B": (4, 3)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_2d_square(sdfg_name: str):
    """Test square matrix multiplication: (n, n) @ (n, n) -> (n, n)"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (8, 8), "B": (8, 8)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_2d_tall(sdfg_name: str):
    """Test tall matrix multiplication: (m, k) @ (k, n) where m >> n"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (20, 5), "B": (5, 3)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_2d_wide(sdfg_name: str):
    """Test wide matrix multiplication: (m, k) @ (k, n) where n >> m"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (3, 5), "B": (5, 20)})


##################################
# Batched Matrix Multiplication Tests
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_3d_batched(sdfg_name: str):
    """Test 3D batched matrix multiplication: (b, m, k) @ (b, k, n) -> (b, m, n)"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (4, 5, 6), "B": (4, 6, 7)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_4d_batched(sdfg_name: str):
    """Test 4D batched matrix multiplication: (b1, b2, m, k) @ (b1, b2, k, n) -> (b1, b2, m, n)"""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (2, 3, 4, 5), "B": (2, 3, 5, 6)})


##################################
# Broadcasting Tests
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_broadcast_2d_3d(sdfg_name: str):
    """Test broadcasting: (m, k) @ (b, k, n) -> (b, m, n)

    The 2D matrix is broadcast across the batch dimension.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (5, 4), "B": (3, 4, 6)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_broadcast_3d_2d(sdfg_name: str):
    """Test broadcasting: (b, m, k) @ (k, n) -> (b, m, n)

    The 2D matrix is broadcast across the batch dimension.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (3, 5, 4), "B": (4, 6)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_broadcast_3d_4d(sdfg_name: str):
    """Test broadcasting: (b2, m, k) @ (b1, b2, k, n) -> (b1, b2, m, n)

    The 3D tensor is broadcast across the first batch dimension.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (3, 4, 5), "B": (2, 3, 5, 6)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_broadcast_4d_3d(sdfg_name: str):
    """Test broadcasting: (b1, b2, m, k) @ (b2, k, n) -> (b1, b2, m, n)

    The 3D tensor is broadcast across the first batch dimension.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (2, 3, 4, 5), "B": (3, 5, 6)})


##################################
# 1D Vector Operations Tests
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_1d_1d_dot_product(sdfg_name: str):
    """Test 1D dot product: (n,) @ (n,) -> scalar

    This is an inner product of two vectors.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(
        MatMulModule(),
        sdfg_name,
        {
            "A": (10, ),
            "B": (10, )
        },
        use_sum=False  # Output is already a scalar
    )


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_2d_1d_matvec(sdfg_name: str):
    """Test matrix-vector multiplication: (m, n) @ (n,) -> (m,)

    Multiply matrix by vector to get a vector.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (5, 7), "B": (7, )})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_1d_2d_vecmat(sdfg_name: str):
    """Test vector-matrix multiplication: (m,) @ (m, n) -> (n,)

    Multiply vector by matrix to get a vector.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (5, ), "B": (5, 7)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_3d_1d_batched_matvec(sdfg_name: str):
    """Test batched matrix-vector: (b, m, n) @ (n,) -> (b, m)

    The vector is broadcast across all batches.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (3, 5, 7), "B": (7, )})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_1d_3d_batched_vecmat(sdfg_name: str):
    """Test batched vector-matrix: (m,) @ (b, m, n) -> (b, n)

    The vector is broadcast across all batches.
    """

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (5, ), "B": (3, 5, 7)})


##################################
# Mixed Dimensional and Complex Tests
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_complex_chain(sdfg_name: str):
    """Test a complex chain of operations with matmul."""

    class ComplexModule(nn.Module):

        def forward(self, X, Y, W):
            Xt = X.T
            YW = W * Y
            Z = Xt @ YW
            Zl = torch.log(Z + 1)
            return Zl

    run_matmul_test(ComplexModule(), sdfg_name, {"X": (4, 5), "Y": (4, 3), "W": (4, 3)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_sequential(sdfg_name: str):
    """Test sequential matmuls: (A @ B) @ C."""

    class SequentialMatMul(nn.Module):

        def forward(self, A, B, C):
            AB = A @ B
            return AB @ C

    run_matmul_test(SequentialMatMul(), sdfg_name, {"A": (5, 4), "B": (4, 6), "C": (6, 3)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_with_elementwise(sdfg_name: str):
    """Test matmul combined with elementwise operations."""

    class MatMulWithBias(nn.Module):

        def forward(self, A, B, bias):
            Y = A @ B
            Y_biased = Y + bias
            Y_relu = F.relu(Y_biased)
            return Y_relu

    run_matmul_test(MatMulWithBias(), sdfg_name, {"A": (5, 4), "B": (4, 3), "bias": (3, )})


##################################
# Tests with Learnable Parameters
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_with_weight_parameter(sdfg_name: str):
    """Test matmul with a learnable weight parameter."""

    class LinearLayer(nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3))

        def forward(self, x):
            return x @ self.weight

    run_matmul_test(LinearLayer(), sdfg_name, {"x": (5, 4)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_with_batched_weight(sdfg_name: str):
    """Test batched matmul with a learnable batched weight parameter."""

    class BatchedLinear(nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, 5, 3))

        def forward(self, x):
            return self.weight @ x

    run_matmul_test(BatchedLinear(), sdfg_name, {"x": (10, 3, 7)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_linear_stack(sdfg_name: str):
    """Test a stack of linear layers (multiple sequential matmuls with weights)."""

    class LinearStack(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128, bias=False)
            self.fc2 = nn.Linear(128, 64, bias=False)
            self.fc3 = nn.Linear(64, 10, bias=False)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

    run_matmul_test(LinearStack(), sdfg_name, {"x": (4, 784)})


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_multi_input_weighted(sdfg_name: str):
    """Test matmul with multiple inputs and learnable parameters."""

    class MultiInputWeighted(nn.Module):

        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.randn(4, 3))
            self.W2 = nn.Parameter(torch.randn(6, 3))

        def forward(self, A, B):
            # (A @ W1) + (B @ W2)
            return (A @ self.W1) + (B @ self.W2)

    run_matmul_test(
        MultiInputWeighted(),
        sdfg_name,
        {
            "A": (5, 4),
            "B": (5, 6)
        },
        atol=1e-3  # Slightly higher tolerance for complex gradients
    )


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_attention_like(sdfg_name: str):
    """Test an attention-like pattern: Q @ K^T @ V."""

    class AttentionLike(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = 1.0 / np.sqrt(64)

        def forward(self, Q, K, V):
            # Simplified attention: (Q @ K^T) @ V
            scores = (Q @ K.transpose(-2, -1)) * self.scale
            return scores @ V

    run_matmul_test(AttentionLike(), sdfg_name, {"Q": (2, 8, 64), "K": (2, 8, 64), "V": (2, 8, 64)}, atol=1e-3)


##################################
# Edge Cases
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_single_element_matrices(sdfg_name: str):
    """Test matmul with 1x1 matrices (edge case)."""

    class MatMulModule(nn.Module):

        def forward(self, A, B):
            return A @ B

    run_matmul_test(MatMulModule(), sdfg_name, {"A": (1, 1), "B": (1, 1)})


# TODO: somehow uninitialized temporaries are in the graph
@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_with_transpose(sdfg_name: str):
    """Test matmul with transposed inputs."""

    class TransposedMatMul(nn.Module):

        def forward(self, A, B):
            At = A.T
            Bt = B.T
            return At @ Bt

    run_matmul_test(TransposedMatMul(), sdfg_name, {"A": (4, 5), "B": (3, 4)})


##################################
# Llama-Specific Patterns
##################################


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_4d_multihead_attention(sdfg_name: str):
    """Test 4D multi-head attention pattern as used in Llama.

    This tests the exact pattern: (batch, n_heads, seq_len, head_dim)
    which is the standard multi-head attention structure in transformers.
    """

    class MultiHeadAttention(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = 1.0 / np.sqrt(64)

        def forward(self, Q, K, V):
            # Q, K, V: (batch, n_heads, seq_len, head_dim)
            # Compute attention scores: Q @ K^T
            scores = Q @ K.transpose(-2, -1)  # (batch, n_heads, seq_len, seq_len)
            scores = scores * self.scale
            # Apply attention to values: scores @ V
            output = scores @ V  # (batch, n_heads, seq_len, head_dim)
            return output

    run_matmul_test(MultiHeadAttention(),
                    sdfg_name, {
                        "Q": (2, 8, 32, 64),
                        "K": (2, 8, 32, 64),
                        "V": (2, 8, 32, 64)
                    },
                    atol=1e-3)


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_grouped_query_attention(sdfg_name: str):
    """Test Grouped Query Attention (GQA) with broadcasting.

    Llama-2 and Llama-3 use GQA where Q has more heads than K/V.
    For example: 32 Q heads, 8 KV heads (4:1 ratio).
    This requires broadcasting K and V to match Q's head dimension.
    """

    class GroupedQueryAttention(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = 1.0 / np.sqrt(64)
            self.n_q_heads = 8
            self.n_kv_heads = 2  # 4:1 ratio like Llama

        def forward(self, Q, K, V):
            # Q: (batch, n_q_heads, seq, dim) = (1, 8, 32, 64)
            # K, V: (batch, n_kv_heads, seq, dim) = (1, 2, 32, 64)
            # K and V will broadcast to match Q's head dimension

            # Expand K and V to match Q heads (each KV head serves n_q_heads/n_kv_heads Q heads)
            batch, n_kv_heads, seq_len, head_dim = K.shape
            n_rep = self.n_q_heads // self.n_kv_heads

            # Repeat K and V along head dimension
            K = K.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len,
                                      head_dim).reshape(batch, self.n_q_heads, seq_len, head_dim)
            V = V.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len,
                                      head_dim).reshape(batch, self.n_q_heads, seq_len, head_dim)

            # Standard attention
            scores = Q @ K.transpose(-2, -1) * self.scale
            output = scores @ V
            return output

    run_matmul_test(GroupedQueryAttention(),
                    sdfg_name, {
                        "Q": (1, 8, 32, 64),
                        "K": (1, 2, 32, 64),
                        "V": (1, 2, 32, 64)
                    },
                    atol=1e-3)


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_llama_ffn_structure(sdfg_name: str):
    """Test Llama-style SwiGLU FFN with large expansion factor.

    Llama uses a gated feed-forward network with ~3.5x expansion:
    - w_gate and w_up: dim -> ffn_dim (parallel)
    - w_down: ffn_dim -> dim
    - output = w_down(silu(w_gate(x)) * w_up(x))

    We use smaller dimensions for faster testing but maintain the structure.
    """

    class LlamaStyleFFN(nn.Module):

        def __init__(self):
            super().__init__()
            dim = 512
            ffn_dim = int(dim * 3.5)  # ~1792, similar to Llama's expansion

            self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
            self.w_up = nn.Linear(dim, ffn_dim, bias=False)
            self.w_down = nn.Linear(ffn_dim, dim, bias=False)

        def forward(self, x):
            # x: (batch, seq_len, dim)
            gate = F.silu(self.w_gate(x))  # SiLU activation
            up = self.w_up(x)
            return self.w_down(gate * up)  # Element-wise multiply then project

    run_matmul_test(
        LlamaStyleFFN(),
        sdfg_name,
        {"x": (2, 32, 512)},  # (batch, seq_len, dim)
        atol=1e-3)


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_attention_qk_only(sdfg_name: str):
    """Test just the Q @ K^T part of attention separately.

    This isolates the attention score computation which is a critical
    pattern in Llama: (batch, heads, seq, dim) @ (batch, heads, dim, seq)
    """

    class AttentionScores(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = 1.0 / np.sqrt(64)

        def forward(self, Q, K):
            # Q, K: (batch, n_heads, seq_len, head_dim)
            scores = Q @ K.transpose(-2, -1)  # (batch, n_heads, seq_len, seq_len)
            return scores * self.scale

    run_matmul_test(AttentionScores(), sdfg_name, {"Q": (2, 8, 64, 64), "K": (2, 8, 64, 64)}, atol=1e-3)


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_attention_av_only(sdfg_name: str):
    """Test just the Attention_Scores @ V part separately.

    This isolates the attention application which is another critical
    pattern: (batch, heads, seq, seq) @ (batch, heads, seq, dim)
    """

    class AttentionApply(nn.Module):

        def forward(self, scores, V):
            # scores: (batch, n_heads, seq_len, seq_len)
            # V: (batch, n_heads, seq_len, head_dim)
            return scores @ V

    run_matmul_test(AttentionApply(), sdfg_name, {"scores": (2, 8, 64, 64), "V": (2, 8, 64, 64)}, atol=1e-3)


@pytest.mark.torch
@pytest.mark.autodiff
def test_matmul_larger_sequence(sdfg_name: str):
    """Test with larger sequence length closer to Llama's typical use.

    Llama often processes 512-2048+ token sequences. This tests a moderate
    sequence length to ensure the implementation scales.
    """

    class LargeSequenceAttention(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = 1.0 / np.sqrt(64)

        def forward(self, Q, K, V):
            scores = (Q @ K.transpose(-2, -1)) * self.scale
            return scores @ V

    run_matmul_test(LargeSequenceAttention(),
                    sdfg_name, {
                        "Q": (1, 8, 512, 64),
                        "K": (1, 8, 512, 64),
                        "V": (1, 8, 512, 64)
                    },
                    atol=1e-3)


##################################
# Entry point for direct execution
##################################

if __name__ == "__main__":
    # Basic 2D tests
    # test_matmul_2d_basic(sdfg_name="test_matmul_2d_basic")
    # test_matmul_2d_square(sdfg_name="test_matmul_2d_square")
    # test_matmul_2d_tall(sdfg_name="test_matmul_2d_tall")
    # test_matmul_2d_wide(sdfg_name="test_matmul_2d_wide")

    # # Batched tests
    # test_matmul_3d_batched(sdfg_name="test_matmul_3d_batched")
    # test_matmul_4d_batched(sdfg_name="test_matmul_4d_batched")

    # # Broadcasting tests
    # test_matmul_broadcast_2d_3d(sdfg_name="test_matmul_broadcast_2d_3d")
    # test_matmul_broadcast_3d_2d(sdfg_name="test_matmul_broadcast_3d_2d")
    # test_matmul_broadcast_3d_4d(sdfg_name="test_matmul_broadcast_3d_4d")
    # test_matmul_broadcast_4d_3d(sdfg_name="test_matmul_broadcast_4d_3d")

    # # 1D vector tests
    # test_matmul_1d_1d_dot_product(sdfg_name="test_matmul_1d_1d_dot_product")
    # test_matmul_2d_1d_matvec(sdfg_name="test_matmul_2d_1d_matvec")
    # test_matmul_1d_2d_vecmat(sdfg_name="test_matmul_1d_2d_vecmat")
    # test_matmul_3d_1d_batched_matvec(sdfg_name="test_matmul_3d_1d_batched_matvec")
    # test_matmul_1d_3d_batched_vecmat(sdfg_name="test_matmul_1d_3d_batched_vecmat")

    # # Mixed dimensional tests
    # test_matmul_complex_chain(sdfg_name="test_matmul_complex_chain")
    # test_matmul_sequential(sdfg_name="test_matmul_sequential")
    # test_matmul_with_elementwise(sdfg_name="test_matmul_with_elementwise")

    # # Tests with learnable parameters
    # test_matmul_with_weight_parameter(sdfg_name="test_matmul_with_weight_parameter")
    # test_matmul_with_batched_weight(sdfg_name="test_matmul_with_batched_weight")
    # test_matmul_linear_stack(sdfg_name="test_matmul_linear_stack")
    # test_matmul_multi_input_weighted(sdfg_name="test_matmul_multi_input_weighted")
    # test_matmul_attention_like(sdfg_name="test_matmul_attention_like")

    # # Edge cases
    # test_matmul_single_element_matrices(sdfg_name="test_matmul_single_element_matrices")
    # test_matmul_with_transpose(sdfg_name="test_matmul_with_transpose")
    test_matmul_llama_ffn_structure(sdfg_name="test_matmul_llama_ffn_structure")

    print("All MatMul PyTorch backward tests passed!")
