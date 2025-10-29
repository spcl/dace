import numpy as np
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch

import dace
import dace.sdfg.nodes as nd
from dace.autodiff import add_backward_pass


class MatMulBackwardRunner:
    """Runner for MatMul backward pass tests with numerical validation."""

    def __init__(self, sdfg, target, simplify=True):
        if simplify:
            sdfg.simplify()
        self.sdfg: dace.SDFG = sdfg
        self.target = target

        # Find all non-transient floating-point inputs for gradient computation
        assert len(sdfg.nodes()) == 1, f"Expected single-state SDFG, got {len(sdfg.nodes())} states"
        state = sdfg.nodes()[0]
        required_grads = list(
            node for node in state.nodes() if isinstance(node, nd.AccessNode)
            and node.desc(sdfg).dtype in [dace.float32, dace.float64] and not node.desc(sdfg).transient)

        add_backward_pass(sdfg=self.sdfg, outputs=[self.target], inputs=required_grads)

    def run(self, **inputs):
        """Run the backward pass and return gradients."""
        intermediate_arrs = {}
        gradient_target = "gradient_" + self.target

        # Initialize all intermediate and gradient arrays
        for name, arr in self.sdfg.arrays.items():
            # Skip gradient target, dunder names, inputs, and transients
            if (name == gradient_target or name.startswith("__") or name in inputs or arr.transient):
                continue

            dtype = getattr(np, arr.dtype.to_string())
            intermediate_arrs[name] = np.zeros(arr.shape, dtype=dtype)

        inputs.update(intermediate_arrs)

        # Set gradient of target to 1 (for sum reduction)
        inputs["gradient_" + self.target] = np.ones((1, ),
                                                    dtype=getattr(np, self.sdfg.arrays[self.target].dtype.to_string()))

        self.sdfg(**inputs)

        return {name: arr for name, arr in inputs.items()}


def run_correctness(func):
    """Decorator to run correctness test comparing DaCe vs PyTorch gradients."""

    def test_correctness():
        runner, pytorch_func, inputs = func()

        # Prepare DaCe inputs
        sdfg_dict = {name: arr.copy() for name, arr in inputs.items()}

        # Prepare PyTorch inputs with gradient tracking
        torch_dict = {name: torch.tensor(arr.copy(), requires_grad=True) for name, arr in inputs.items()}

        # Run both implementations
        sdfg_results = runner.run(**sdfg_dict)
        torch_results = pytorch_func(**torch_dict)

        # Compare gradients
        for k, v in torch_results.items():
            v = v.detach().numpy()
            diff = np.linalg.norm(sdfg_results[k] - v) / max(np.prod(v.shape), 1)
            assert diff < 1e-5, f"Gradient mismatch for '{k}': normalized difference {diff:.2e} exceeds tolerance 1e-5"

    return test_correctness


##################################
# Basic 2D Matrix Multiplication Tests
##################################


@pytest.mark.autodiff
@run_correctness
def test_matmul_2d_basic():
    """Test basic 2D matrix multiplication: (m, k) @ (k, n) -> (m, n)"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5, 4],
        B: dace.float32[4, 3],
        Y: dace.float32[5, 3],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5, 4).astype(np.float32),
            B=np.random.rand(4, 3).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_2d_square():
    """Test square matrix multiplication: (n, n) @ (n, n) -> (n, n)"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float64[8, 8],
        B: dace.float64[8, 8],
        Y: dace.float64[8, 8],
        S: dace.float64[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:8, 0:8])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(8, 8).astype(np.float64),
            B=np.random.rand(8, 8).astype(np.float64),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_2d_tall():
    """Test tall matrix multiplication: (m, k) @ (k, n) where m >> n"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[20, 5],
        B: dace.float32[5, 3],
        Y: dace.float32[20, 3],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:20, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(20, 5).astype(np.float32),
            B=np.random.rand(5, 3).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_2d_wide():
    """Test wide matrix multiplication: (m, k) @ (k, n) where n >> m"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[3, 5],
        B: dace.float32[5, 20],
        Y: dace.float32[3, 20],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:3, 0:20])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(3, 5).astype(np.float32),
            B=np.random.rand(5, 20).astype(np.float32),
        ),
    )


##################################
# Batched Matrix Multiplication Tests
##################################


@pytest.mark.autodiff
@run_correctness
def test_matmul_3d_batched():
    """Test 3D batched matrix multiplication: (b, m, k) @ (b, k, n) -> (b, m, n)"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[4, 5, 6],
        B: dace.float32[4, 6, 7],
        Y: dace.float32[4, 5, 7],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:4, 0:5, 0:7])
        def summap(i, j, k):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j, k]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(4, 5, 6).astype(np.float32),
            B=np.random.rand(4, 6, 7).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_4d_batched():
    """Test 4D batched matrix multiplication: (b1, b2, m, k) @ (b1, b2, k, n) -> (b1, b2, m, n)"""

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[2, 3, 4, 5],
        B: dace.float32[2, 3, 5, 6],
        Y: dace.float32[2, 3, 4, 6],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:2, 0:3, 0:4, 0:6])
        def summap(i, j, k, l):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j, k, l]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(2, 3, 4, 5).astype(np.float32),
            B=np.random.rand(2, 3, 5, 6).astype(np.float32),
        ),
    )


##################################
# Broadcasting Tests
##################################


@pytest.mark.autodiff
@run_correctness
def test_matmul_broadcast_2d_3d():
    """Test broadcasting: (m, k) @ (b, k, n) -> (b, m, n)

    The 2D matrix is broadcast across the batch dimension.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5, 4],
        B: dace.float32[3, 4, 6],
        Y: dace.float32[3, 5, 6],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:3, 0:5, 0:6])
        def summap(i, j, k):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j, k]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5, 4).astype(np.float32),
            B=np.random.rand(3, 4, 6).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_broadcast_3d_2d():
    """Test broadcasting: (b, m, k) @ (k, n) -> (b, m, n)

    The 2D matrix is broadcast across the batch dimension.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[3, 5, 4],
        B: dace.float32[4, 6],
        Y: dace.float32[3, 5, 6],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:3, 0:5, 0:6])
        def summap(i, j, k):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j, k]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(3, 5, 4).astype(np.float32),
            B=np.random.rand(4, 6).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_broadcast_3d_4d():
    """Test broadcasting: (b2, m, k) @ (b1, b2, k, n) -> (b1, b2, m, n)

    The 3D tensor is broadcast across the first batch dimension.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[3, 4, 5],
        B: dace.float32[2, 3, 5, 6],
        Y: dace.float32[2, 3, 4, 6],
        S: dace.float32[1],
    ):
        Y = A @ B
        S[0] = np.sum(Y)

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(3, 4, 5).astype(np.float32),
            B=np.random.rand(2, 3, 5, 6).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_broadcast_4d_3d():
    """Test broadcasting: (b1, b2, m, k) @ (b2, k, n) -> (b1, b2, m, n)

    The 3D tensor is broadcast across the first batch dimension.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[2, 3, 4, 5],
        B: dace.float32[3, 5, 6],
        Y: dace.float32[2, 3, 4, 6],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:2, 0:3, 0:4, 0:6])
        def summap(i, j, k, l):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j, k, l]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(2, 3, 4, 5).astype(np.float32),
            B=np.random.rand(3, 5, 6).astype(np.float32),
        ),
    )


##################################
# 1D Vector Operations Tests
##################################


@pytest.mark.autodiff
@run_correctness
def test_matmul_1d_1d_dot_product():
    """Test 1D dot product: (n,) @ (n,) -> scalar

    This is an inner product of two vectors.
    """

    def torch_func(*, A, B):
        Y = A @ B
        Y.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[10],
        B: dace.float32[10],
    ):
        return A @ B

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "__return"),
        torch_func,
        dict(
            A=np.random.rand(10).astype(np.float32),
            B=np.random.rand(10).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_2d_1d_matvec():
    """Test matrix-vector multiplication: (m, n) @ (n,) -> (m,)

    Multiply matrix by vector to get a vector.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5, 7],
        B: dace.float32[7],
        Y: dace.float32[5],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:5])
        def summap(i):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5, 7).astype(np.float32),
            B=np.random.rand(7).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_1d_2d_vecmat():
    """Test vector-matrix multiplication: (m,) @ (m, n) -> (n,)

    Multiply vector by matrix to get a vector.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5],
        B: dace.float32[5, 7],
        Y: dace.float32[7],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:7])
        def summap(i):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5).astype(np.float32),
            B=np.random.rand(5, 7).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_3d_1d_batched_matvec():
    """Test batched matrix-vector: (b, m, n) @ (n,) -> (b, m)

    The vector is broadcast across all batches.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[3, 5, 7],
        B: dace.float32[7],
        Y: dace.float32[3, 5],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:3, 0:5])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(3, 5, 7).astype(np.float32),
            B=np.random.rand(7).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_1d_3d_batched_vecmat():
    """Test batched vector-matrix: (m,) @ (b, m, n) -> (b, n)

    The vector is broadcast across all batches.
    """

    def torch_func(*, A, B):
        Y = A @ B
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5],
        B: dace.float32[3, 5, 7],
        Y: dace.float32[3, 7],
        S: dace.float32[1],
    ):
        Y[:] = A @ B

        @dace.map(_[0:3, 0:7])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5).astype(np.float32),
            B=np.random.rand(3, 5, 7).astype(np.float32),
        ),
    )


##################################
# Mixed Dimensional Tests
##################################


@pytest.mark.autodiff
@run_correctness
def test_matmul_complex_chain():
    """Test a complex chain of operations with matmul."""

    def torch_func(*, X, Y, W):
        Xt = X.T
        YW = W * Y
        Z = Xt @ YW
        Zl = torch.log(Z + 1)
        S = Zl.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad, gradient_W=W.grad)

    @dace.program
    def dace_func(
        X: dace.float32[4, 5],
        Y: dace.float32[4, 3],
        W: dace.float32[4, 3],
        S: dace.float32[1],
    ):
        Xt = np.transpose(X)
        YW = W * Y
        Z = Xt @ YW

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = log(z + 1)

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            X=np.random.rand(4, 5).astype(np.float32),
            Y=np.random.rand(4, 3).astype(np.float32),
            W=np.random.rand(4, 3).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_sequential():
    """Test sequential matmuls: (A @ B) @ C."""

    def torch_func(*, A, B, C):
        AB = A @ B
        Y = AB @ C
        S = Y.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad, gradient_C=C.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5, 4],
        B: dace.float32[4, 6],
        C: dace.float32[6, 3],
        Y: dace.float32[5, 3],
        S: dace.float32[1],
    ):
        AB = A @ B
        Y[:] = AB @ C

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            y << Y[i, j]
            s = y

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5, 4).astype(np.float32),
            B=np.random.rand(4, 6).astype(np.float32),
            C=np.random.rand(6, 3).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_matmul_with_elementwise():
    """Test matmul combined with elementwise operations."""

    def torch_func(*, A, B, bias):
        Y = A @ B
        Y_biased = Y + bias
        Y_relu = torch.nn.functional.relu(Y_biased)
        S = Y_relu.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad, gradient_bias=bias.grad)

    @dace.program
    def dace_func(
        A: dace.float32[5, 4],
        B: dace.float32[4, 3],
        bias: dace.float32[3],
        Y: dace.float32[5, 3],
        S: dace.float32[1],
    ):
        Y[:] = A @ B
        Y_biased = Y + bias

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            yb << Y_biased[i, j]
            s = max(yb, 0.0)

    sdfg = dace_func.to_sdfg()

    return (
        MatMulBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            A=np.random.rand(5, 4).astype(np.float32),
            B=np.random.rand(4, 3).astype(np.float32),
            bias=np.random.rand(3).astype(np.float32),
        ),
    )


if __name__ == "__main__":
    # Basic 2D tests
    test_matmul_2d_basic()
    test_matmul_2d_square()
    test_matmul_2d_tall()
    test_matmul_2d_wide()

    test_matmul_3d_batched()
    test_matmul_4d_batched()

    test_matmul_broadcast_2d_3d()
    test_matmul_broadcast_3d_2d()
    test_matmul_broadcast_3d_4d()
    test_matmul_broadcast_4d_3d()

    test_matmul_1d_1d_dot_product()
    test_matmul_2d_1d_matvec()
    test_matmul_1d_2d_vecmat()
    test_matmul_3d_1d_batched_matvec()
    test_matmul_1d_3d_batched_vecmat()

    test_matmul_complex_chain()
    test_matmul_sequential()
    test_matmul_with_elementwise()
