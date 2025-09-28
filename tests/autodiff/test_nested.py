import numpy as np
import pytest
import torch

import dace
from dace import nodes as nd
from dace.transformation.interstate import StateFusion

import dace.libraries.onnx as donnx
from test_single_state import SDFGBackwardRunner, run_correctness


@dace.program
def inner_sdfg(Z: dace.float32[3, 3], W: dace.float32[3, 3]):
    W[:] = dace.elementwise(lambda x: log(x), Z)


@dace.program
def inner_sdfg_with_intermediate(Z: dace.float32[3, 3], W: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Z)
    W[:] = dace.elementwise(lambda x: log(x), intermediate)


@dace.program
def middle_sqrt(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg(intermediate, W)
    Z = np.sum(W)
    return Z


@pytest.mark.autodiff
@run_correctness
def test_nested():
    sdfg = middle_sqrt.to_sdfg(simplify=True)

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        W = torch.log(inter)
        Z = torch.sum(W)
        Z.backward()
        return dict(gradient_Y=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return",
                               simplify=False), torch_func, dict(Y=np.random.rand(3, 3).astype(np.float32)))


@dace.program
def middle_sqrt_with_intermediate(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg_with_intermediate(intermediate, W)
    Z = np.sum(W)
    return Z


@pytest.mark.autodiff
@run_correctness
def test_nested_forwarding():
    sdfg = middle_sqrt_with_intermediate.to_sdfg(simplify=True)

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        inter2 = torch.sqrt(inter)
        W = torch.log(inter2)
        Z = torch.sum(W)
        Z.backward()
        return dict(gradient_Y=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return",
                               simplify=False), torch_func, dict(Y=np.random.rand(3, 3).astype(np.float32)))


@dace.program
def middle_sqrt_no_sum(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg_with_intermediate(intermediate, W)
    return W


@dace.program
def outer_sqrt_with_intermediate(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    W[:] = middle_sqrt_no_sum(intermediate)
    Z = np.sum(W)
    return Z


@pytest.mark.autodiff
@run_correctness
def test_triple_nested_forwarding():
    sdfg = outer_sqrt_with_intermediate.to_sdfg(simplify=True)

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        inter2 = torch.sqrt(inter)
        inter3 = torch.sqrt(inter2)
        W = torch.log(inter3)
        Z = torch.sum(W)
        Z.backward()
        return dict(gradient_Y=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return",
                               simplify=False), torch_func, dict(Y=np.random.rand(3, 3).astype(np.float32)))


@pytest.mark.autodiff
@run_correctness
def test_view_forwarding():
    # Prepare the inner sdfg
    old_default = donnx.default_implementation
    donnx.default_implementation = "pure"

    @dace.program
    def add_reshape_grad_test_nested(inp1: dace.float64[9], bias: dace.float64[3], target_shape: dace.int64[2],
                                     result: dace.float64):
        reshaped = dace.define_local([3, 3], dace.float64)
        added = inp1 + 1
        donnx.ONNXReshape(data=added, shape=target_shape, reshaped=reshaped)
        Z = reshaped * bias
        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        result[:] = np.sum(Zl)

    sdfg = add_reshape_grad_test_nested.to_sdfg(simplify=True)

    sdfg.expand_library_nodes()
    del sdfg.arrays["target_shape"]

    donnx.default_implementation = old_default

    # Prepare the outer SDFG

    @dace.program
    def inner_view_forwarding(inp1: dace.float64[9], bias: dace.float64[3]):
        result = dace.define_local_scalar(dace.float64)
        # target shape gets removed by the pure reshape expansion
        sdfg(inp1=inp1, bias=bias, result=result)
        return result + 1

    # This generates a FunctionCallRegion in the current frontned
    # We need to simplify.
    outer_sdfg = inner_view_forwarding.to_sdfg(simplify=True)
    outer_sdfg.apply_transformations_repeated([StateFusion])

    def torch_func(*, inp1, bias):
        reshaped = torch.reshape(inp1 + 1, [3, 3])

        Z = reshaped * bias
        Zl = torch.log(Z + 1)
        S = Zl.sum() + 1

        S.backward()
        return dict(gradient_inp1=inp1.grad, gradient_bias=bias.grad)

    return (SDFGBackwardRunner(outer_sdfg, "__return", simplify=False), torch_func,
            dict(inp1=np.random.rand(9).astype(np.float64), bias=np.random.rand(3).astype(np.float64)))
