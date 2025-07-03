from functools import reduce

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import dace
from dace import data
import dace.sdfg.nodes as nd
from dace.transformation.interstate import StateFusion

import dace.libraries.onnx as donnx
from dace.autodiff import AutoDiffException, add_backward_pass

##################################
# Testing utilities


def run_correctness(func):

    def test_correctness():
        runner, pytorch_func, inputs = func()
        sdfg_dict = {name: arr.copy() for name, arr in inputs.items()}
        torch_dict = {
            name: torch.tensor(arr.copy(), requires_grad=True)
            for name, arr in inputs.items()
        }

        sdfg_results = runner.run(**sdfg_dict)
        torch_results = pytorch_func(**torch_dict)

        for k, v in torch_results.items():
            print("-" * 10, k, "-" * 10)
            v = v.detach().numpy()
            diff = np.linalg.norm(sdfg_results[k] - v) / reduce(
                lambda x, y: x * y, v.shape)

            print("Difference:", diff)

            print("Torch results:", "-" * 10)
            print(v)
            print("SDFG results:", "-" * 10)
            print(sdfg_results[k])
            print("Difference:")
            print(v - sdfg_results[k])

            assert diff < 1e-5

    return test_correctness


class SDFGBackwardRunner:

    def __init__(self, sdfg, target, simplify=True):
        if simplify:
            sdfg.simplify()
        self.sdfg: dace.SDFG = sdfg
        self.target = target

        assert len(sdfg.nodes()) == 1
        state = sdfg.nodes()[0]
        required_grads = list(
            node for node in state.nodes()
            if isinstance(node, nd.AccessNode) and node.desc(sdfg).dtype in
            [dace.float32, dace.float64] and not node.desc(sdfg).transient)

        add_backward_pass(sdfg=self.sdfg,
                          outputs=[self.target],
                          inputs=required_grads)

    def run(self, **inputs):

        # zero out all arrays
        intermediate_arrs = {
            name: np.zeros(arr.shape, dtype=getattr(np, arr.dtype.to_string()))
            for name, arr in self.sdfg.arrays.items()
            if name != "gradient_" + self.target if not name.startswith("__")
            if name not in inputs if not arr.transient
        }
        inputs.update(intermediate_arrs)
        inputs["gradient_" + self.target] = np.ones(
            (1, ),
            dtype=getattr(np, self.sdfg.arrays[self.target].dtype.to_string()))

        print("Pre-execution arrays")
        for k, v in inputs.items():
            print(k, "-" * 10)
            print("\t{}".format(v.dtype))
            print("\t{}".format("is_contiguous:", v.flags["C_CONTIGUOUS"]))
            print("\t{}".format(v))

        self.sdfg(**inputs)

        print("Post-execution arrays")
        for k, v in inputs.items():
            print(k, "-" * 10)
            print("\t{}".format(v.dtype))
            print("\t{}".format("is_contiguous:", v.flags["C_CONTIGUOUS"]))
            print("\t{}".format(v))

        results = {name: arr for name, arr in inputs.items()}
        return results


##################################
# Tests


@run_correctness
def test_gemm():

    def torch_gemm(*, X, Y):
        Z = X @ Y
        S = Z.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad)

    @dace.program
    def dace_gemm(
        X: dace.float32[5, 4],
        Y: dace.float32[4, 3],
        Z: dace.float32[5, 3],
        S: dace.float32[1],
    ):

        Z[:] = X @ Y

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = z

    sdfg = dace_gemm.to_sdfg()

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_gemm,
        dict(
            X=np.random.rand(5, 4).astype(np.float32),
            Y=np.random.rand(4, 3).astype(np.float32),
        ),
    )


@run_correctness
def test_sum():

    def torch_sum(*, X, Y):
        Z = X + Y
        Z = Z * Z
        S = Z.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad)

    @dace.program
    def dace_sum(
        X: dace.float32[3, 3],
        Y: dace.float32[3, 3],
        Z: dace.float32[3, 3],
        S: dace.float32[1],
    ):

        Z[:] = X + Y

        @dace.map(_[0:3, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = z * z

    sdfg = dace_sum.to_sdfg()
    state = sdfg.nodes()[0]

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_sum,
        dict(
            X=np.random.rand(3, 3).astype(np.float32),
            Y=np.random.rand(3, 3).astype(np.float32),
        ),
    )


@run_correctness
def test_complex_tasklet():

    def torch_sum(*, X, Y):
        Z = X + Y
        Z = Z * Z
        S = Z.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad)

    @dace.program
    def dace_sum_complex(
        X: dace.float32[3, 3],
        Y: dace.float32[3, 3],
        Z: dace.float32[3, 3],
        S: dace.float32[1],
    ):

        Z[:] = X + Y

        @dace.map(_[0:3, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]

            z1 = z + 1
            log(3)  # random expr
            z2 = z - 1 * (2 / 2)
            # hello world 1, 2, 3
            s = z1 * z2

    sdfg = dace_sum_complex.to_sdfg()
    state = sdfg.nodes()[0]

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_sum,
        dict(
            X=np.random.rand(3, 3).astype(np.float32),
            Y=np.random.rand(3, 3).astype(np.float32),
        ),
    )


@run_correctness
def test_tasklets_direct_scalar_edges():

    def torch_func(*, A):
        tmp_a = torch.sqrt(A)
        tmp_b = torch.log(tmp_a + 1)
        tmp_c = torch.sin(tmp_b)

        tmp_c.backward()
        return dict(gradient_A=A.grad)

    sdfg = dace.SDFG("tasklets_direct_scalar_edges")
    state = sdfg.add_state()

    sdfg.add_array(
        "A",
        shape=[
            1,
        ],
        dtype=dace.float32,
    )
    sdfg.add_array(
        "C",
        shape=[
            1,
        ],
        dtype=dace.float32,
    )

    tmp_a, tmp_a_desc = sdfg.add_scalar("tmp_a", dace.float32, transient=True)
    tmp_b, tmp_b_desc = sdfg.add_scalar("tmp_b", dace.float32, transient=True)

    A = state.add_access("A")
    C = state.add_access("C")

    task1 = state.add_tasklet("task1", {"inp"}, {"out"}, "out = sqrt(inp)")
    task2 = state.add_tasklet("task2", {"inp"}, {"out"}, "out = log(inp + 1)")
    task3 = state.add_tasklet("task3", {"inp"}, {"out"}, "out = sin(inp)")

    state.add_edge(A, None, task1, "inp", dace.Memlet.simple("A", "0"))
    state.add_edge(task1, "out", task2, "inp", dace.Memlet.simple(tmp_a, "0"))
    state.add_edge(task2, "out", task3, "inp", dace.Memlet.simple(tmp_b, "0"))
    state.add_edge(task3, "out", C, None, dace.Memlet.simple("C", "0"))

    return (
        SDFGBackwardRunner(sdfg, "C"),
        torch_func,
        dict(A=np.random.rand(1).astype(np.float32)),
    )


@run_correctness
def test_tasklets_only_reuse():

    def torch_func(*, A):
        tmp_a = torch.sqrt(A)
        tmp_b = torch.log(A + 1)

        C = tmp_a * tmp_b

        C.backward()
        return dict(gradient_A=A.grad)

    @dace.program
    def tasklets_only_reuse(A: dace.float32[1], C: dace.float32[1]):
        tmp_a = dace.define_local_scalar(dace.float32)
        tmp_b = dace.define_local_scalar(dace.float32)

        with dace.tasklet:
            a << A[0]
            a_out >> tmp_a

            a_out = sqrt(a)

        with dace.tasklet:
            a << A[0]
            a_out >> tmp_b

            a_out = log(a + 1)

        with dace.tasklet:
            a << tmp_a
            b << tmp_b
            c >> C[0]
            c = a * b

    sdfg = tasklets_only_reuse.to_sdfg(simplify=False)
    sdfg.simplify()

    return (
        SDFGBackwardRunner(sdfg, "C"),
        torch_func,
        dict(A=np.random.rand(1).astype(np.float32)),
    )


@run_correctness
def test_tasklets_multioutput():

    def torch_func(*, A, B):
        tmp_a = torch.sqrt(A)
        tmp_b = torch.log(B + 1)

        C = tmp_a * tmp_b * B

        C.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def tasklets_multioutput(A: dace.float32[1], B: dace.float32[1],
                             C: dace.float32[1]):
        tmp_a = dace.define_local_scalar(dace.float32)
        tmp_b = dace.define_local_scalar(dace.float32)
        tmp_d = dace.define_local_scalar(dace.float32)

        with dace.tasklet:
            a << A[0]
            a_out >> tmp_a

            a_out = sqrt(a)

        with dace.tasklet:
            b << B[0]
            b_out >> tmp_b
            d_out >> tmp_d

            b_out = log(b + 1)
            d_out = b

        with dace.tasklet:
            a << tmp_a
            b << tmp_b
            d << tmp_d
            c >> C[0]
            c = a * b * d

    sdfg = tasklets_multioutput.to_sdfg(simplify=False)
    sdfg.simplify()

    return (
        SDFGBackwardRunner(sdfg, "C"),
        torch_func,
        dict(
            A=np.random.rand(1).astype(np.float32),
            B=np.random.rand(1).astype(np.float32),
        ),
    )


@run_correctness
def test_tasklets_only():

    def torch_func(*, A, B):
        tmp_a = torch.sqrt(A)
        tmp_b = torch.log(B + 1)

        C = tmp_a * tmp_b

        C.backward()
        return dict(gradient_A=A.grad, gradient_B=B.grad)

    @dace.program
    def tasklets_only(A: dace.float32[1], B: dace.float32[1],
                      C: dace.float32[1]):
        tmp_a = dace.define_local_scalar(dace.float32)
        tmp_b = dace.define_local_scalar(dace.float32)

        with dace.tasklet:
            a << A[0]
            a_out >> tmp_a

            a_out = sqrt(a)

        with dace.tasklet:
            a << B[0]
            a_out >> tmp_b

            a_out = log(a + 1)

        with dace.tasklet:
            a << tmp_a
            b << tmp_b
            c >> C[0]
            c = a * b

    sdfg = tasklets_only.to_sdfg(simplify=False)
    sdfg.simplify()

    return (
        SDFGBackwardRunner(sdfg, "C"),
        torch_func,
        dict(
            A=np.random.rand(1).astype(np.float32),
            B=np.random.rand(1).astype(np.float32),
        ),
    )


@run_correctness
def test_add_mmul_transpose_log():

    def torch_func(*, X, Y, W):

        Xt = X.T
        YW = W * Y
        Z = Xt @ YW
        Zl = torch.log(Z + 1)

        S = Zl.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad, gradient_W=W.grad)

    @dace.program
    def add_mmul_transpose_log(
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

    sdfg = add_mmul_transpose_log.to_sdfg()

    return (
        SDFGBackwardRunner(sdfg, "S"),
        torch_func,
        dict(
            X=np.random.rand(4, 5).astype(np.float32),
            W=np.random.rand(4, 3).astype(np.float32),
            Y=np.random.rand(4, 3).astype(np.float32),
        ),
    )


@run_correctness
def test_reduce_node_1_axis_and_none_axis():

    def torch_func(*, X, Y, W):

        Xt = X.T
        YW = torch.sum(W, dim=0) * Y
        Z = Xt @ YW
        Zl = torch.log(Z + 1)

        S = Zl.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad, gradient_W=W.grad)

    @dace.program
    def reduce_node_1_axis_and_none_axis(X: dace.float32[4, 5],
                                         Y: dace.float32[4, 3],
                                         W: dace.float32[7, 4, 3]):

        Xt = np.transpose(X)
        YW = np.sum(W, axis=0) * Y
        Z = Xt @ YW

        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        S = np.sum(Zl)
        return S

    sdfg = reduce_node_1_axis_and_none_axis.to_sdfg()

    return (
        SDFGBackwardRunner(sdfg, "__return"),
        torch_func,
        dict(
            X=np.random.rand(4, 5).astype(np.float32),
            W=np.random.rand(7, 4, 3).astype(np.float32),
            Y=np.random.rand(4, 3).astype(np.float32),
        ),
    )


@run_correctness
def test_reduce_max_simple():

    def torch_func(*, W):

        Z = torch.max(W, dim=1)
        S = Z.values.sum()
        S.backward()
        return dict(gradient_W=W.grad)

    @dace.program
    def reduce_max_simple(W: dace.float32[4, 5]):

        Z = np.max(W, axis=1)
        S = np.sum(Z)
        return S

    sdfg = reduce_max_simple.to_sdfg()

    return (
        SDFGBackwardRunner(sdfg, "__return"),
        torch_func,
        dict(W=np.random.rand(4, 5).astype(np.float32)),
    )


@run_correctness
def test_reduce_max_node_1_axis():

    def torch_func(*, X, Y, W):

        Xt = X.T
        YW = torch.min(W, dim=0).values * Y
        Z = Xt @ YW
        Zl = torch.log(Z + 1)

        S = Zl.sum()
        S.backward()
        return dict(gradient_X=X.grad, gradient_Y=Y.grad, gradient_W=W.grad)

    @dace.program
    def dace_func(X: dace.float64[4, 5], Y: dace.float64[4, 3],
                  W: dace.float64[7, 4, 3]):

        Xt = np.transpose(X)
        YW = np.min(W, axis=0) * Y
        Z = Xt @ YW

        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        S = np.sum(Zl)
        return S

    sdfg = dace_func.to_sdfg()

    return (
        SDFGBackwardRunner(sdfg, "__return"),
        torch_func,
        dict(
            X=np.random.rand(4, 5).astype(np.float64),
            W=np.random.rand(7, 4, 3).astype(np.float64),
            Y=np.random.rand(4, 3).astype(np.float64),
        ),
    )


@run_correctness
def test_reshape():

    @dace.program
    def single_state_reshape(inp: dace.float64[9], bias: dace.float64[3],
                             target_shape: dace.int64[2]):
        reshaped = dace.define_local([3, 3], dace.float64)
        donnx.ONNXReshape(data=inp, shape=target_shape, reshaped=reshaped)
        Z = reshaped + bias
        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        S = np.sum(Zl)
        return S

    sdfg = single_state_reshape.to_sdfg(simplify=False)

    sdfg.apply_transformations_repeated([StateFusion])

    def torch_func(*, inp, bias):
        reshaped = torch.reshape(inp, [3, 3])

        Z = reshaped + bias
        Zl = torch.log(Z + 1)
        S = Zl.sum()

        S.backward()
        return dict(gradient_inp=inp.grad, gradient_bias=bias.grad)

    return (
        SDFGBackwardRunner(sdfg, "__return", simplify=False),
        torch_func,
        dict(
            inp=np.random.rand(9).astype(np.float64),
            bias=np.random.rand(3).astype(np.float64),
        ),
    )


@run_correctness
def test_reshape_on_memlet_path():
    old_default = donnx.default_implementation
    donnx.default_implementation = "pure"

    @dace.program
    def single_state_reshape_memlet_path(inp1: dace.float64[9],
                                         bias: dace.float64[3],
                                         target_shape: dace.int64[2]):
        reshaped = dace.define_local([3, 3], dace.float64)
        donnx.ONNXReshape(data=inp1, shape=target_shape, reshaped=reshaped)
        Z = reshaped + bias
        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        S = np.sum(Zl)
        return S

    sdfg = single_state_reshape_memlet_path.to_sdfg(simplify=False)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([StateFusion])

    donnx.default_implementation = old_default

    def torch_func(*, inp1, bias):
        reshaped = torch.reshape(inp1, [3, 3])

        Z = reshaped + bias
        Zl = torch.log(Z + 1)
        S = Zl.sum()

        S.backward()
        return dict(gradient_inp1=inp1.grad, gradient_bias=bias.grad)

    return (
        SDFGBackwardRunner(sdfg, "__return", simplify=False),
        torch_func,
        dict(
            inp1=np.random.rand(9).astype(np.float64),
            bias=np.random.rand(3).astype(np.float64),
        ),
    )


@run_correctness
def test_reshape_reuse_in_same_state():
    old_default = donnx.default_implementation
    donnx.default_implementation = "pure"

    @dace.program
    def single_state_reshape_same_state(inp: dace.float64[9],
                                        target_shape: dace.int64[2]):
        reshaped = dace.define_local([3, 3], dace.float64)
        donnx.ONNXReshape(data=inp, shape=target_shape, reshaped=reshaped)
        Zl = dace.elementwise(lambda x: log(x + 1), reshaped)
        S = np.sum(Zl)
        return S

    sdfg = single_state_reshape_same_state.to_sdfg(simplify=False)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([StateFusion])

    donnx.default_implementation = old_default

    def torch_func(*, inp):
        reshaped = torch.reshape(inp, [3, 3])

        Z = reshaped
        Zl = torch.log(Z + 1)
        S = Zl.sum()

        S.backward()
        return dict(gradient_inp=inp.grad)

    return (
        SDFGBackwardRunner(sdfg, "__return", simplify=False),
        torch_func,
        dict(inp=np.random.rand(9).astype(np.float64), ),
    )


if __name__ == "__main__":
    test_tasklets_direct_scalar_edges()
    test_reshape_on_memlet_path()
    test_reshape_reuse_in_same_state()
    test_reduce_max_node_1_axis()
