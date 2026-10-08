# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch

import dace
from dace import SDFG, InterstateEdge, Memlet
from dace.autodiff import add_backward_pass
from test_single_state import SDFGBackwardRunner, run_correctness


@pytest.mark.autodiff
@run_correctness
def test_two_state_add_mul():
    """
    Test a two-state SDFG:
    - State 1: Z = X + Y (element-wise addition)
    - State 2: S = sum(Z * Z) (element-wise multiplication then sum)
    """

    sdfg = SDFG("two_state_add_mul")

    sdfg.add_array("X", [3, 3], dace.float32)
    sdfg.add_array("Y", [3, 3], dace.float32)
    sdfg.add_array("Z", [3, 3], dace.float32, transient=False)
    sdfg.add_array("S", [1], dace.float32)

    state1 = sdfg.add_state("state1")
    X_read = state1.add_access("X")
    Y_read = state1.add_access("Y")
    Z_write = state1.add_access("Z")

    map_entry, map_exit = state1.add_map("add_map", dict(i="0:3", j="0:3"))

    tasklet_add = state1.add_tasklet("add", {"x", "y"}, {"z"}, "z = x + y")

    state1.add_memlet_path(X_read, map_entry, tasklet_add, dst_conn="x", memlet=Memlet("X[i, j]"))
    state1.add_memlet_path(Y_read, map_entry, tasklet_add, dst_conn="y", memlet=Memlet("Y[i, j]"))
    state1.add_memlet_path(tasklet_add, map_exit, Z_write, src_conn="z", memlet=Memlet("Z[i, j]"))

    state2 = sdfg.add_state("state2")
    Z_read = state2.add_access("Z")
    S_write = state2.add_access("S")

    map_entry2, map_exit2 = state2.add_map("mul_map", dict(i="0:3", j="0:3"))

    tasklet_mul = state2.add_tasklet("mul", {"z"}, {"s"}, "s = z * z")

    state2.add_memlet_path(Z_read, map_entry2, tasklet_mul, dst_conn="z", memlet=Memlet("Z[i, j]"))
    state2.add_memlet_path(
        tasklet_mul, map_exit2, S_write, src_conn="s", memlet=Memlet("S[0]", wcr="lambda a, b: a + b")
    )

    sdfg.add_edge(state1, state2, InterstateEdge())

    # PyTorch reference implementation
    def torch_func(*, X, Y):
        Z = X + Y
        S = (Z * Z).sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad)

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            X=np.random.rand(3, 3).astype(np.float32),
            Y=np.random.rand(3, 3).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_conditional_simple():
    """
    Test a Python program with a simple conditional in the forward pass:
    if X[0, 0] > 0.5:
        Y = X * 2
    else:
        Y = X * 3
    S = sum(Y)
    """

    @dace.program
    def conditional_program(X: dace.float32[3, 3], Y: dace.float32[3, 3], S: dace.float32[1]):
        if X[0, 0] > 0.5:
            Y[:] = X * 2.0
        else:
            Y[:] = X * 3.0
        S[0] = np.sum(Y)

    sdfg = conditional_program.to_sdfg(simplify=True)

    # PyTorch reference implementation
    def torch_func(*, X):
        Y = torch.where(X[0, 0] > 0.5, X * 2.0, X * 3.0)
        S = Y.sum()
        S.backward()
        return dict(gradient_X=X.grad)

    return (
        SDFGBackwardRunner(sdfg, "S", simplify=False),
        torch_func,
        dict(X=np.random.rand(3, 3).astype(np.float32)),
    )


@pytest.mark.autodiff
@run_correctness
def test_for_loop():
    """
    Test a simple for loop similar to jacobi_1d, but simplified:
    for i in range(3):
        A = A + B
    S = sum(A)
    """

    @dace.program
    def for_loop_program(A: dace.float32[10], B: dace.float32[10]):
        for i in range(3):
            A[:] = A + B
        return np.sum(A)

    sdfg = for_loop_program.to_sdfg()

    # PyTorch reference implementation
    def torch_func(*, A, B):
        A_result = A.clone()
        for i in range(3):
            A_result = A_result + B
        S = A_result.sum()
        S.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    return (
        SDFGBackwardRunner(sdfg, "__return"),
        torch_func,
        dict(
            A=np.random.rand(10).astype(np.float32),
            B=np.random.rand(10).astype(np.float32),
        ),
    )


@pytest.mark.autodiff
@run_correctness
def test_diamond_pattern_conditional():
    """
    Test an SDFG with a diamond pattern control flow using GOTOs.

    Structure:
    state1: Y = X * 2
    if X[0] > 0.5:
        goto state3
    else:
        goto state2
    state2: Y = Y + 1
    state3: S = sum(Y)

    This creates a diamond pattern where both paths can reach state3.
    """

    sdfg = SDFG("irreducible_cf")

    # Add arrays
    sdfg.add_array("X", [5], dace.float32)
    sdfg.add_array("Y", [5], dace.float32, transient=False)
    sdfg.add_array("S", [1], dace.float32)

    # State 1: Y = X * 2
    state1 = sdfg.add_state("state1")
    X_read1 = state1.add_access("X")
    Y_write1 = state1.add_access("Y")

    map_entry1, map_exit1 = state1.add_map("mul_map", dict(i="0:5"))
    tasklet1 = state1.add_tasklet("mul", {"x"}, {"y"}, "y = x * 2.0")

    state1.add_memlet_path(X_read1, map_entry1, tasklet1, dst_conn="x", memlet=Memlet("X[i]"))
    state1.add_memlet_path(tasklet1, map_exit1, Y_write1, src_conn="y", memlet=Memlet("Y[i]"))

    # State 2: Y = Y + 1
    state2 = sdfg.add_state("state2")
    Y_read2 = state2.add_access("Y")
    Y_write2 = state2.add_access("Y")

    map_entry2, map_exit2 = state2.add_map("add_map", dict(i="0:5"))
    tasklet2 = state2.add_tasklet("add", {"y_in"}, {"y_out"}, "y_out = y_in + 1.0")

    state2.add_memlet_path(Y_read2, map_entry2, tasklet2, dst_conn="y_in", memlet=Memlet("Y[i]"))
    state2.add_memlet_path(tasklet2, map_exit2, Y_write2, src_conn="y_out", memlet=Memlet("Y[i]"))

    # State 3: S = sum(Y)
    state3 = sdfg.add_state("state3")
    Y_read3 = state3.add_access("Y")
    S_write3 = state3.add_access("S")

    map_entry3, map_exit3 = state3.add_map("sum_map", dict(i="0:5"))
    tasklet3 = state3.add_tasklet("sum", {"y"}, {"s"}, "s = y")

    state3.add_memlet_path(Y_read3, map_entry3, tasklet3, dst_conn="y", memlet=Memlet("Y[i]"))
    state3.add_memlet_path(tasklet3, map_exit3, S_write3, src_conn="s", memlet=Memlet("S[0]", wcr="lambda a, b: a + b"))

    # Create conditional edges (irreducible control flow)
    # Add condition: if X[0] > 0.5 goto state3, else goto state2
    sdfg.add_edge(state1, state3, InterstateEdge(condition="X[0] > 0.5"))
    sdfg.add_edge(state1, state2, InterstateEdge(condition="X[0] <= 0.5"))
    sdfg.add_edge(state2, state3, InterstateEdge())

    # PyTorch reference implementation
    def torch_func(*, X):
        Y = X * 2.0
        Y = torch.where(X[0] > 0.5, Y, Y + 1.0)
        S = Y.sum()
        S.backward()
        return dict(gradient_X=X.grad)

    return (
        SDFGBackwardRunner(sdfg, "S", simplify=False),
        torch_func,
        dict(X=np.random.rand(5).astype(np.float32)),
    )


@pytest.mark.autodiff
@run_correctness
def test_multi_output_state():
    """
    Test a two-state SDFG where the first state produces multiple outputs:
    State 1: Y = X * 2, Z = X + 1
    State 2: S = sum(Y * Z)
    """

    # Build SDFG using API
    sdfg = SDFG("multi_output_state")

    # Add arrays
    sdfg.add_array("X", [5], dace.float32)
    sdfg.add_array("Y", [5], dace.float32, transient=False)
    sdfg.add_array("Z", [5], dace.float32, transient=False)
    sdfg.add_array("S", [1], dace.float32)

    # State 1: Compute Y and Z
    state1 = sdfg.add_state("state1")
    X_read1 = state1.add_access("X")
    Y_write1 = state1.add_access("Y")
    Z_write1 = state1.add_access("Z")

    map_entry1, map_exit1 = state1.add_map("compute_map", dict(i="0:5"))
    tasklet_y = state1.add_tasklet("compute_y", {"x"}, {"y"}, "y = x * 2.0")
    tasklet_z = state1.add_tasklet("compute_z", {"x"}, {"z"}, "z = x + 1.0")

    state1.add_memlet_path(X_read1, map_entry1, tasklet_y, dst_conn="x", memlet=Memlet("X[i]"))
    state1.add_memlet_path(tasklet_y, map_exit1, Y_write1, src_conn="y", memlet=Memlet("Y[i]"))

    X_read2 = state1.add_access("X")
    state1.add_memlet_path(X_read2, map_entry1, tasklet_z, dst_conn="x", memlet=Memlet("X[i]"))
    state1.add_memlet_path(tasklet_z, map_exit1, Z_write1, src_conn="z", memlet=Memlet("Z[i]"))

    # State 2: Multiply and sum
    state2 = sdfg.add_state("state2")
    Y_read2 = state2.add_access("Y")
    Z_read2 = state2.add_access("Z")
    S_write2 = state2.add_access("S")

    map_entry2, map_exit2 = state2.add_map("mul_sum_map", dict(i="0:5"))
    tasklet_mul = state2.add_tasklet("mul", {"y", "z"}, {"s"}, "s = y * z")

    state2.add_memlet_path(Y_read2, map_entry2, tasklet_mul, dst_conn="y", memlet=Memlet("Y[i]"))
    state2.add_memlet_path(Z_read2, map_entry2, tasklet_mul, dst_conn="z", memlet=Memlet("Z[i]"))
    state2.add_memlet_path(
        tasklet_mul, map_exit2, S_write2, src_conn="s", memlet=Memlet("S[0]", wcr="lambda a, b: a + b")
    )

    # Connect states
    sdfg.add_edge(state1, state2, InterstateEdge())

    # PyTorch reference implementation
    def torch_func(*, X):
        Y = X * 2.0
        Z = X + 1.0
        S = (Y * Z).sum()
        S.backward()
        return dict(gradient_X=X.grad)

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_func,
        dict(X=np.random.rand(5).astype(np.float32)),
    )


@pytest.mark.autodiff
def test_extremal_reduction_in_loop_resets_tie_counter():
    """``max`` reduced once per loop iteration: every window's argmax must get the full gradient.

    The extremal backward splits the gradient between elements tied for the maximum, which needs a
    per-output counter. That counter has to be cleared on every execution of the backward, not once
    where it is allocated -- otherwise it accumulates across iterations and the k-th reversed
    iteration hands back ``grad / k``. Random inputs have no ties, so the exact answer is the 0/1
    indicator of each window's argmax and any division shows up immediately.
    """

    S0, S1, S2, S3 = (dace.symbol(n, dtype=dace.int64) for n in ("S0", "S1", "S2", "S3"))

    @dace.program
    def pooled(x: dace.float32[S0, S1, S2, S3]):
        out = np.ndarray([S0, S1 // 2, S2 // 2, S3], dtype=np.float32)
        for i in range(S1 // 2):
            for j in range(S2 // 2):
                out[:, i, j, :] = np.max(x[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :], axis=(1, 2))
        return np.sum(out)

    sdfg = pooled.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["x"], outputs=["__return"])

    x = (
        np.array(
            [[0.9, 0.1, 0.2, 0.8], [0.3, 0.4, 0.7, 0.6], [0.5, 0.95, 0.15, 0.25], [0.35, 0.45, 0.85, 0.55]],
            dtype=np.float32,
        )
        .reshape(1, 4, 4, 1)
        .copy(order="C")
    )
    gradient_x = np.zeros_like(x)
    sdfg(x, S0=1, S1=4, S2=4, S3=1, gradient_x=gradient_x, gradient___return=np.ones((1,), dtype=np.float32))

    expected = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            window = x[0, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, 0]
            di, dj = np.unravel_index(int(np.argmax(window)), window.shape)
            expected[0, 2 * i + di, 2 * j + dj, 0] = 1.0

    assert np.array_equal(gradient_x, expected), (
        f"expected the argmax of every window to carry gradient 1, got {gradient_x[0, :, :, 0].tolist()}"
    )


if __name__ == "__main__":
    test_two_state_add_mul()
    test_conditional_simple()
    test_for_loop()
    test_diamond_pattern_conditional()
    test_multi_output_state()
    test_extremal_reduction_in_loop_resets_tie_counter()
