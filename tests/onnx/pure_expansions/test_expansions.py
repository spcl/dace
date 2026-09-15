# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import copy
import numpy as np

import dace
from dace import transformation, data as dt
from dace.libraries import blas
import dace.library

import dace.libraries.onnx as donnx
from dace.transformation.onnx import expand_onnx_nodes

from tests.ml_gpu_utils import DEVICES, run_sdfg


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("a_shape, b_shape", [([2, 4], [4, 3])])
def test_matmul_expansion(a_shape, b_shape, device):
    blas.Gemm.default_implementation = "pure"
    sdfg = dace.SDFG(f"test_matmul_expansion_{device}")

    X = np.random.rand(*a_shape).astype(np.float32)
    Z = np.random.rand(*b_shape).astype(np.float32)
    expected_result = X @ Z
    sdfg.add_array("X", a_shape, dace.float32)
    sdfg.add_array("Z", b_shape, dace.float32)
    sdfg.add_array("__return", expected_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))

    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    with dace.library.change_default(blas, "pure"):
        sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = run_sdfg(sdfg, device, X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_cast_int_to_float(device):
    sdfg = dace.SDFG(f"test_cast_int_to_float_{device}")

    sdfg.add_array("X", [2, 4], dace.int32)
    sdfg.add_array("__return", [2, 4], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = donnx.converters.typeclass_to_onnx_tensor_type_int(dace.float32)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None, sdfg.make_array_memlet("__return"))

    X = np.random.randint(0, 10, size=(2, 4), dtype=np.int32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(X.astype(np.float32), result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_cast_float_to_int(device):
    sdfg = dace.SDFG(f"test_cast_float_to_int_{device}")

    sdfg.add_array("X", [2, 4], dace.float32)
    sdfg.add_array("__return", [2, 4], dace.int32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = donnx.converters.typeclass_to_onnx_tensor_type_int(dace.int32)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None, sdfg.make_array_memlet("__return"))

    X = np.random.normal(scale=10, size=(2, 4)).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(X.astype(np.int32), result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_cast_float_to_long(device):
    sdfg = dace.SDFG(f"test_cast_float_to_long_{device}")

    sdfg.add_array("X", [2, 4], dace.float32)
    sdfg.add_array("__return", [2, 4], dace.int64)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = donnx.converters.typeclass_to_onnx_tensor_type_int(dace.int64)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None, sdfg.make_array_memlet("__return"))

    X = np.random.normal(scale=10, size=(2, 4)).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(X.astype(np.int64), result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
#+yapf: disable
@pytest.mark.parametrize("reduce_type, keepdims, axes",
                         [('Sum',  True,  [0]),
                          ('Sum',  False, [-1]),
                          ('Sum',  True,  [0, -1]),
                          ('Max',  False, [0, -1]),
                          ('Max',  True,  [0]),
                          ('Max',  True,  [-1]),
                          ('Mean', True,  [-1]),
                          ('Mean', True,  [0, -1]),
                          ('Mean', False, [0])])
#+yapf: enable
def test_reduce(keepdims, reduce_type, axes, device):

    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(f"test_reduce_{device}")

    sdfg.add_array("X", [2, 4, 10], dace.float32)

    numpy_func = getattr(np, reduce_type.lower())
    numpy_result = numpy_func(X.copy(), axis=tuple(axes), keepdims=keepdims)

    resulting_shape = numpy_result.shape

    sdfg.add_array("__return", resulting_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = getattr(donnx, "ONNXReduce" + reduce_type)("reduce")
    op_node.axes = axes
    op_node.keepdims = 1 if keepdims else 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())
    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(numpy_result, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_reduce_scalar(device):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(f"test_reduce_scalar_{device}")

    numpy_result = np.mean(X)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_scalar("Y", dace.float32, transient=True)
    sdfg.add_array("__return", [1], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Y = state.add_access("Y")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceMean("mean")
    op_node.keepdims = 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_Y, None, sdfg.make_array_memlet("Y"))

    state.add_edge(access_Y, None, access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(numpy_result, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("new_shape", [[8, 10], [80], [2, 40]])
def test_reshape(new_shape, device):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(f"test_reshape_{device}")

    numpy_result = X.reshape(*new_shape)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_array("shape", [len(new_shape)], dace.int64)
    sdfg.add_array("__return", new_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_shape = state.add_access("shape")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReshape("reshape")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))
    state.add_edge(access_shape, None, op_node, "shape", sdfg.make_array_memlet("shape"))

    state.add_edge(op_node, "reshaped", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    # we don't need shape anymore
    del sdfg.arrays["shape"]

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_flatten(device):

    new_shape = [2, 40]
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(f"test_flatten_{device}")

    numpy_result = X.reshape(*new_shape)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_array("__return", new_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXFlatten("flatten")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_reciprocal(device):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    numpy_result = 1 / X
    sdfg = dace.SDFG(f"test_reciprocal_{device}")

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_array("__return", numpy_result.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReciprocal("reciprocal")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "X", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "Y", access_result, None, sdfg.make_array_memlet("__return"))

    sdfg.expand_library_nodes()

    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = run_sdfg(sdfg, device, X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_einsum(device):

    @dace.program
    def test_einsum(A: dace.float64[5, 4, 3], B: dace.float64[3, 2]):
        Y = dace.define_local([5, 4, 2], dace.float64)
        donnx.ONNXEinsum(Inputs__0=A, Inputs__1=B, Output=Y, equation="bij, jk -> bik")
        return Y

    sdfg = test_einsum.to_sdfg()
    expand_onnx_nodes(sdfg)
    assert any(isinstance(n, blas.Gemm) for n, _ in sdfg.all_nodes_recursive())

    A = np.random.rand(5, 4, 3).astype(np.float64)
    B = np.random.rand(3, 2).astype(np.float64)
    result = run_sdfg(sdfg, device, A=A.copy(), B=B.copy())
    assert_allclose(result, np.einsum("bij ,jk -> bik", A, B))


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_reshape_add(device):

    @dace.program
    def add_reshape(inp: dace.float64[9], bias: dace.float64[3], target_shape: dace.int64[2]):
        reshaped = dace.define_local([3, 3], dace.float64)
        donnx.ONNXReshape(data=inp, shape=target_shape, reshaped=reshaped)

        return reshaped + bias

    sdfg: dace.SDFG = add_reshape.to_sdfg(simplify=False)

    sdfg.apply_transformations_repeated([transformation.interstate.StateFusion])

    inp = np.arange(9).astype(np.float64)
    bias = np.arange(3).astype(np.float64)
    result = run_sdfg(sdfg, device, inp=inp.copy(), bias=bias.copy(), target_shape=np.array([3, 3]).astype(np.int64))

    assert_allclose(result, inp.reshape(3, 3) + bias)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_desc", [dace.float32[2, 3], dace.float32[1], dace.float32])
def test_sum_arrays(input_desc, device):

    if isinstance(input_desc, dt.Array):
        shape = input_desc.shape
    else:
        shape = [1]

    def prog(inp0: copy.deepcopy(input_desc), inp1: copy.deepcopy(input_desc), inp2: copy.deepcopy(input_desc)):
        result = dace.define_local(shape, dace.float32)
        donnx.ONNXSum(data_0__0=inp0, data_0__1=inp1, data_0__2=inp2, sum=result)
        return result

    prog.__name__ = f"test_sum_arrays_{device}"
    prog = dace.program(prog)

    inputs = [np.random.randn(*shape).astype(np.float32) for _ in range(3)]
    if not isinstance(input_desc, dt.Array):
        inputs = [i[0] for i in inputs]
    np_result = (inputs[0] + inputs[1]) + inputs[2]
    sdfg = prog.to_sdfg()
    result = run_sdfg(sdfg, device, inp0=inputs[0], inp1=inputs[1], inp2=inputs[2])

    assert_allclose(result, np_result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_shape(device):

    @dace.program
    def shape(inp: dace.float64[9, 5, 3]):
        shp = dace.define_local([3], dace.int64)
        donnx.ONNXShape(data=inp, shape=shp)
        return shp

    sdfg: dace.SDFG = shape.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    inp = np.random.rand(9, 5, 3).astype(np.float64)
    result = run_sdfg(sdfg, device, inp=inp.copy())
    assert_allclose(result, [9, 5, 3]), result


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_gather_onnx_1(device):
    # gather in ONNX operators.md
    @dace.program
    def gather(inp: dace.float64[3, 2], indices: dace.int64[2, 2]):
        output = dace.define_local([2, 2, 2], dace.float64)
        donnx.ONNXGather(data=inp, output=output, indices=indices, axis=0)
        return output

    sdfg: dace.SDFG = gather.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]])
    indices = np.array([[0, 1], [1, 2]])
    result = run_sdfg(sdfg, device, inp=data.copy(), indices=indices.copy())
    assert_allclose(result, data[indices])


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_gather_bert(device):
    # gather found at start of bert model
    @dace.program
    def gather(embs: dace.float64[64, 8], input_ids: dace.int64[8, 16]):
        output = dace.define_local([8, 16, 8], dace.float64)
        donnx.ONNXGather(data=embs, output=output, indices=input_ids, axis=0)
        return output

    sdfg: dace.SDFG = gather.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    embs = np.random.rand(64, 8).astype(np.float64)
    input_ids = np.random.randint(low=0, high=64, size=(8, 16)).astype(np.int64)
    result = run_sdfg(sdfg, device, embs=embs.copy(), input_ids=input_ids.copy())
    assert_allclose(result, embs[input_ids])


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_gather_scalar(device):
    # gather test 2 in BERT model (third last op)
    @dace.program
    def gather(inp: dace.float64[1, 8, 32], indices: dace.int64):
        output = dace.define_local([1, 32], dace.float64)
        donnx.ONNXGather(data=inp, output=output, indices=indices, axis=1)
        return output

    sdfg: dace.SDFG = gather.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    data = np.random.rand(1, 8, 32)
    indices = np.int64(5)
    result = run_sdfg(sdfg, device, inp=data.copy(), indices=indices.copy())
    np_result = np.take(data, indices, axis=1)

    assert_allclose(result, np_result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_gather_onnx_2(device):
    # gather test 2 in ONNX operators.md
    @dace.program
    def gather(inp: dace.float64[3, 3], indices: dace.int64[1, 2]):
        output = dace.define_local([3, 1, 2], dace.float64)
        donnx.ONNXGather(data=inp, output=output, indices=indices, axis=1)
        return output

    sdfg: dace.SDFG = gather.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    data = np.array([
        [1.0, 1.2, 1.9],
        [2.3, 3.4, 3.9],
        [4.5, 5.7, 5.9],
    ])
    indices = np.array([[0, 2]])
    result = run_sdfg(sdfg, device, inp=data.copy(), indices=indices.copy())
    np_result = np.take(data, indices, axis=1)

    assert_allclose(result, np_result)


@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_unsqueeze(device):

    @dace.program
    def unsqueeze(inp: dace.float64[3, 3]):
        output = dace.define_local([3, 1, 3, 1], dace.float64)
        axes = dace.define_local([2], dace.int64)
        axes[0] = 1
        axes[1] = 3
        donnx.ONNXUnsqueeze(data=inp, expanded=output, axes=axes)
        return output

    sdfg: dace.SDFG = unsqueeze.to_sdfg()

    data = np.array([
        [1.0, 1.2, 1.9],
        [2.3, 3.4, 3.9],
        [4.5, 5.7, 5.9],
    ])

    np_result = np.reshape(data, [3, 1, 3, 1])

    result = run_sdfg(sdfg, device, inp=data.copy())
    assert result.shape == (3, 1, 3, 1)
    assert_allclose(result, np_result)


@pytest.mark.onnx
def test_pure_expansion_reads_gpu_staged_constant():
    """A constant an op reads on the host must stay readable after GPU offloading.

    Offloading puts the model's arrays on the device and stages a host-read one back under a new
    name, so the ``axes`` the pure ReduceMean expansion needs no longer answers to the name the
    model gave it. The expansion then declines, and autodiff meets an ONNX node with no
    differentiable form. Runs on the host: offloading is a graph rewrite, no device needed.
    """
    from onnx import TensorProto, helper, numpy_helper
    from dace.frontend.ml.onnx import ONNXModel
    from dace.libraries.onnx.converters import clean_onnx_name
    from dace.libraries.onnx.nodes.onnx_op import ONNXOp
    from dace.libraries.onnx.op_implementations.reduction_ops import PureReduceMean
    from dace.sdfg.utils import in_edge_with_name

    node = helper.make_node("ReduceMean", ["data", "axes"], ["reduced"], keepdims=1)
    graph = helper.make_graph([node],
                              "reduce_mean", [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 4, 10])],
                              [helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [2, 4, 1])],
                              initializer=[numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")])
    model = ONNXModel("test_pure_expansion_reads_gpu_staged_constant",
                      helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
                      cuda=True,
                      onnx_simplify=False)

    state, reduce_mean = next((state, n) for state in model.sdfg.states() for n in state.nodes()
                              if isinstance(n, ONNXOp) and n.schema.name == "ReduceMean")

    # The point of the test: offloading really did rename the container the op reads ``axes`` from,
    # so the expansion is being asked about a name the ONNX graph never used.
    staged = in_edge_with_name(reduce_mean, state, "axes").src.data
    assert staged != clean_onnx_name("axes")
    assert_allclose(model.clean_weights[staged].numpy(), np.array([2], dtype=np.int64))

    assert PureReduceMean.forward_can_be_applied(reduce_mean, state, model.sdfg)


if __name__ == "__main__":
    test_matmul_expansion(a_shape=[2, 4], b_shape=[4, 3], device="cpu")
    test_cast_int_to_float(device="cpu")
    test_cast_float_to_int(device="cpu")
    test_cast_float_to_long(device="cpu")

    reduce_params = [(True, 'Sum', [0]), (False, 'Sum', [-1]), (True, 'Sum', [0, -1]), (False, 'Max', [0, -1]),
                     (True, 'Max', [0]), (True, 'Max', [-1]), (True, 'Mean', [-1]), (True, 'Mean', [0, -1]),
                     (False, 'Mean', [0])]
    for keepdims, reduce_type, axes in reduce_params:
        test_reduce(keepdims=keepdims, reduce_type=reduce_type, axes=axes, device="cpu")

    test_reduce_scalar(device="cpu")

    for new_shape in [[8, 10], [80], [2, 40]]:
        test_reshape(new_shape=new_shape, device="cpu")

    test_flatten(device="cpu")
    test_reciprocal(device="cpu")
    test_einsum(device="cpu")
    test_reshape_add(device="cpu")

    for input_desc in [dace.float32[2, 3], dace.float32[1], dace.float32]:
        test_sum_arrays(input_desc=input_desc, device="cpu")

    test_shape(device="cpu")
    test_gather_onnx_1(device="cpu")
    test_gather_bert(device="cpu")
    test_gather_scalar(device="cpu")
    test_gather_onnx_2(device="cpu")
    test_unsqueeze(device="cpu")
    test_pure_expansion_reads_gpu_staged_constant()
