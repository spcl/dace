# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed")
import torch

import dace
from tests.autodiff.test_single_state import SDFGBackwardRunner


def run_max_reduction_test(dace_func, torch_func, inputs, rtol=1e-5, atol=1e-5):
    sdfg = dace_func.to_sdfg()
    runner = SDFGBackwardRunner(sdfg, "__return")

    sdfg_dict = {name: arr.copy() for name, arr in inputs.items()}
    torch_dict = {name: torch.tensor(arr.copy(), requires_grad=True) for name, arr in inputs.items()}

    sdfg_results = runner.run(**sdfg_dict)
    torch_results = torch_func(**torch_dict)

    for k, v in torch_results.items():
        v = v.detach().numpy()
        assert np.allclose(sdfg_results[k], v, rtol=rtol, atol=atol), \
            f"Gradient mismatch for '{k}':\n  DaCe:    {sdfg_results[k]}\n  PyTorch: {v}"


@pytest.mark.autodiff
def test_max_single_maximum():
    """Max reduction with single maximum - no ties."""

    def torch_func(*, W):
        Z = torch.amax(W, dim=0)
        S = Z.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def dace_func(W: dace.float32[4]):
        Z = np.max(W, axis=0)
        S = np.sum(Z)
        return S

    inputs = dict(W=np.array([1.0, 3.0, 2.0, 0.0], dtype=np.float32))
    run_max_reduction_test(dace_func, torch_func, inputs)


@pytest.mark.autodiff
def test_max_tied_values_2d():
    """Max reduction with tied values along an axis.

    For input [[1, 3], [3, 2]] with max along axis=0:
    - Column 0: max=3 at row 1 only -> grad [0, 1]
    - Column 1: max=3 at row 0 only -> grad [1, 0]
    """

    def torch_func(*, W):
        Z = torch.amax(W, dim=0)
        S = Z.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def dace_func(W: dace.float32[2, 2]):
        Z = np.max(W, axis=0)
        S = np.sum(Z)
        return S

    inputs = dict(W=np.array([[1.0, 3.0], [3.0, 2.0]], dtype=np.float32))
    run_max_reduction_test(dace_func, torch_func, inputs)


@pytest.mark.autodiff
def test_max_tied_values_same_column():
    """Max reduction with tied values in the same column.

    For input [[3, 1], [3, 2]] with max along axis=0:
    - Column 0: max=3 at rows 0 AND 1 -> split grad equally: [0.5, 0.5]
    - Column 1: max=2 at row 1 only -> grad [0, 1]

    Expected gradient: [[0.5, 0], [0.5, 1]]
    """

    def torch_func(*, W):
        Z = torch.amax(W, dim=0)
        S = Z.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def dace_func(W: dace.float32[2, 2]):
        Z = np.max(W, axis=0)
        S = np.sum(Z)
        return S

    inputs = dict(W=np.array([[3.0, 1.0], [3.0, 2.0]], dtype=np.float32))
    run_max_reduction_test(dace_func, torch_func, inputs)


@pytest.mark.autodiff
def test_max_all_equal_column():
    """Max reduction where entire column has same value.

    For input [[3, 1], [3, 2], [3, 0]] with max along axis=0:
    - Column 0: all values are 3 -> split equally: [1/3, 1/3, 1/3]
    - Column 1: max=2 at row 1 only -> grad [0, 1, 0]

    Expected gradient: [[1/3, 0], [1/3, 1], [1/3, 0]]
    """

    def torch_func(*, W):
        Z = torch.amax(W, dim=0)
        S = Z.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def dace_func(W: dace.float32[3, 2]):
        Z = np.max(W, axis=0)
        S = np.sum(Z)
        return S

    inputs = dict(W=np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    run_max_reduction_test(dace_func, torch_func, inputs)


@pytest.mark.autodiff
def test_min_tied_values():
    """Min reduction with tied values.

    For input [[1, 2], [1, 3], [2, 1]] with min along axis=0:
    - Column 0: min=1 at rows 0 AND 1 -> split: [0.5, 0.5, 0]
    - Column 1: min=1 at row 2 only -> grad [0, 0, 1]

    Expected gradient: [[0.5, 0], [0.5, 0], [0, 1]]
    """

    def torch_func(*, W):
        Z = torch.amin(W, dim=0)
        S = Z.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def dace_func(W: dace.float32[3, 2]):
        Z = np.min(W, axis=0)
        S = np.sum(Z)
        return S

    inputs = dict(W=np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]], dtype=np.float32))
    run_max_reduction_test(dace_func, torch_func, inputs)


if __name__ == "__main__":
    test_max_single_maximum()
    test_max_tied_values_2d()
    test_max_tied_values_same_column()
    test_max_all_equal_column()
    test_min_tied_values()
