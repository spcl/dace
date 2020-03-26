import torch
import numpy as np
import dace
import dace.graph.nodes as nd
from dace.autodiff import add_backward_pass
from dace.libraries.blas import ExpandGemmPure
from dace.transformation import optimizer

all_tests = []


def correctness_test(func):
    all_tests.append(func)
    return func


class SDFGBackwardRunner:
    def __init__(self, sdfg, target, required_grads=None):
        self.sdfg = sdfg
        self.target = target
        self.required_grads = required_grads
        state = sdfg.nodes()[0]

        matches = [
            node for node in state.nodes()
            if type(node) is nd.AccessNode and node.data == target
        ]
        if len(matches) > 1:
            raise ValueError(
                "Found more than one accessnode with data {}".format(target))

        add_backward_pass(sdfg, state, matches[0])
        self.sdfg.apply_strict_transformations()

    def run(self, **inputs):
        # zero out all arrays
        intermediate_arrs = {
            name: np.zeros(arr.shape, dtype=getattr(np, arr.dtype.to_string()))
            for name, arr in self.sdfg.arrays.items()
            if name != self.target + "_grad" if not name.startswith("__")
            if name not in inputs if not arr.transient
        }
        inputs.update(intermediate_arrs)
        inputs[self.target + "_grad"] = np.ones(
            (1, ),
            dtype=getattr(np, self.sdfg.arrays[self.target].dtype.to_string()))
        self.sdfg.save("out.sdfg")
        self.sdfg(**inputs)

        results = {
            name: arr
            for name, arr in inputs.items()
            if name.endswith("_grad") and name != self.target + "_grad"
            if self.required_grads is None or name in self.required_grads
        }
        return results


def check_correctness(runner: SDFGBackwardRunner, pytorch_func, inputs):
    sdfg_dict = {name: arr.copy() for name, arr in inputs.items()}
    torch_dict = {
        name: torch.tensor(arr.copy(), requires_grad=True)
        for name, arr in inputs.items()
    }

    sdfg_results = runner.run(**sdfg_dict)
    torch_results = pytorch_func(**torch_dict)

    for k, v in torch_results.items():
        v = v.detach().numpy()
        diff = np.linalg.norm(sdfg_results[k] - v) / (v.shape[0] * v.shape[1])
        assert diff < 1e-5


@correctness_test
def test_gemm():
    def torch_gemm(*, X, Y):
        Z = X @ Y
        S = Z.sum()
        S.backward()
        return dict(X_grad=X.grad, Y_grad=Y.grad)

    @dace.program
    def dace_gemm(X: dace.float32[5, 4], Y: dace.float32[4, 3],
                  Z: dace.float32[5, 3], S: dace.float32[1]):

        Z[:] = X @ Y

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = z

    sdfg = dace_gemm.to_sdfg()
    state = sdfg.nodes()[0]
    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()

    return SDFGBackwardRunner(sdfg, "S"), torch_gemm, dict(
        X=np.random.rand(5, 4).astype(np.float32),
        Y=np.random.rand(4, 3).astype(np.float32))


@correctness_test
def test_sum():
    def torch_sum(*, X, Y):
        Z = X + Y
        Z = Z * Z
        S = Z.sum()
        S.backward()
        return dict(X_grad=X.grad, Y_grad=Y.grad)

    @dace.program
    def dace_sum(X: dace.float32[3, 3], Y: dace.float32[3, 3],
                 Z: dace.float32[3, 3], S: dace.float32[1]):

        Z[:] = X + Y

        @dace.map(_[0:3, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = z * z

    sdfg = dace_sum.to_sdfg()
    state = sdfg.nodes()[0]

    return SDFGBackwardRunner(sdfg, "S"), torch_sum, dict(
        X=np.random.rand(3, 3).astype(np.float32),
        Y=np.random.rand(3, 3).astype(np.float32))


@correctness_test
def test_add_mmul_transpose_log():
    def torch_func(*, X, Y, W):

        Xt = X.T
        YW = W * Y
        Z = Xt @ YW
        Zl = torch.log(Z)

        S = Zl.sum()
        S.backward()
        return dict(X_grad=X.grad, Y_grad=Y.grad)

    @dace.program
    def dace_func(X: dace.float32[4, 5], Y: dace.float32[4, 3],
                  W: dace.float32[4, 3], Z: dace.float32[5, 3],
                  S: dace.float32[1]):

        Xt[:] = np.transpose(X)
        YW[:] = W * Y
        Z[:] = Xt @ YW

        @dace.map(_[0:5, 0:3])
        def summap(i, j):
            s >> S(1, lambda x, y: x + y)[0]
            z << Z[i, j]
            s = log(z)

    sdfg = dace_func.to_sdfg()
    state = sdfg.nodes()[0]
    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()

    return SDFGBackwardRunner(sdfg, "S"), torch_func, dict(
        X=np.random.rand(4, 5).astype(np.float32),
        W=np.random.rand(4, 3).astype(np.float32),
        Y=np.random.rand(4, 3).astype(np.float32))


def test_all():
    for test in all_tests:
        check_correctness(*test())


if __name__ == "__main__":
    test_all()
