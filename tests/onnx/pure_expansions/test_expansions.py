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
from dace.util import utils


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.onnx
@pytest.mark.parametrize("a_shape, b_shape", [([2, 4], [4, 3])])
def test_matmul_expansion(a_shape, b_shape, sdfg_name):
    blas.Gemm.default_implementation = "pure"
    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X, Z=Z)

    assert_allclose(expected_result, result)


@pytest.mark.onnx
def test_cast_int_to_float(sdfg_name):
    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(X.astype(np.float32), result)


@pytest.mark.onnx
def test_cast_float_to_int(sdfg_name):
    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(X.astype(np.int32), result)


@pytest.mark.onnx
def test_cast_float_to_long(sdfg_name):
    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(X.astype(np.int64), result)


@pytest.mark.onnx
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
def test_reduce(keepdims, reduce_type, axes, sdfg_name):

    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

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
    result = sdfg(X=X)

    assert_allclose(numpy_result, result, rtol=1e-5, atol=1e-5)


@pytest.mark.onnx
def test_reduce_scalar(sdfg_name):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

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
    # check that the expansion worked. The default ORT expansion contains a Tasklet with suffix _onnx_code
    assert not any(
        isinstance(n, dace.nodes.Tasklet) and n.name.endswith("_onnx_code") for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
@pytest.mark.parametrize("new_shape", [[8, 10], [80], [2, 40]])
def test_reshape(new_shape, sdfg_name):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
def test_flatten(sdfg_name):

    new_shape = [2, 40]
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
def test_reciprocal(sdfg_name):
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    numpy_result = 1 / X
    sdfg = dace.SDFG(sdfg_name)

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

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


@pytest.mark.onnx
def test_einsum():

    @dace.program
    def test_einsum(A: dace.float64[5, 4, 3], B: dace.float64[3, 2]):
        Y = dace.define_local([5, 4, 2], dace.float64)
        donnx.ONNXEinsum(Inputs__0=A, Inputs__1=B, Output=Y, equation="bij, jk -> bik")
        return Y

    sdfg = test_einsum.to_sdfg()
    utils.expand_onnx_nodes(sdfg)
    assert any(isinstance(n, blas.Gemm) for n, _ in sdfg.all_nodes_recursive())

    A = np.random.rand(5, 4, 3).astype(np.float64)
    B = np.random.rand(3, 2).astype(np.float64)
    result = test_einsum(A.copy(), B.copy())
    assert_allclose(result, np.einsum("bij ,jk -> bik", A, B))


@pytest.mark.onnx
def test_reshape_add():

    @dace.program
    def add_reshape(inp: dace.float64[9], bias: dace.float64[3], target_shape: dace.int64[2]):
        reshaped = dace.define_local([3, 3], dace.float64)
        donnx.ONNXReshape(data=inp, shape=target_shape, reshaped=reshaped)

        return reshaped + bias

    sdfg: dace.SDFG = add_reshape.to_sdfg(simplify=False)

    sdfg.apply_transformations_repeated([transformation.interstate.StateFusion])

    inp = np.arange(9).astype(np.float64)
    bias = np.arange(3).astype(np.float64)
    result = sdfg(inp=inp.copy(), bias=bias.copy(), target_shape=np.array([3, 3]).astype(np.int64))

    assert_allclose(result, inp.reshape(3, 3) + bias)


@pytest.mark.onnx
@pytest.mark.parametrize("input_desc", [dace.float32[2, 3], dace.float32[1], dace.float32])
def test_sum_arrays(input_desc, sdfg_name):

    if isinstance(input_desc, dt.Array):
        shape = input_desc.shape
    else:
        shape = [1]

    def prog(inp0: copy.deepcopy(input_desc), inp1: copy.deepcopy(input_desc), inp2: copy.deepcopy(input_desc)):
        result = dace.define_local(shape, dace.float32)
        donnx.ONNXSum(data_0__0=inp0, data_0__1=inp1, data_0__2=inp2, sum=result)
        return result

    prog.__name__ = sdfg_name
    prog = dace.program(prog)

    inputs = [np.random.randn(*shape).astype(np.float32) for _ in range(3)]
    if not isinstance(input_desc, dt.Array):
        inputs = [i[0] for i in inputs]
    np_result = (inputs[0] + inputs[1]) + inputs[2]
    result = prog(*inputs)

    assert_allclose(result, np_result)


@pytest.mark.onnx
def test_shape():

    @dace.program
    def shape(inp: dace.float64[9, 5, 3]):
        shp = dace.define_local([3], dace.int64)
        donnx.ONNXShape(data=inp, shape=shp)
        return shp

    sdfg: dace.SDFG = shape.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    inp = np.random.rand(9, 5, 3).astype(np.float64)
    result = sdfg(inp=inp.copy())
    assert_allclose(result, [9, 5, 3]), result


@pytest.mark.onnx
def test_gather_onnx_1():
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
    result = sdfg(inp=data.copy(), indices=indices.copy())
    assert_allclose(result, data[indices])


@pytest.mark.onnx
def test_gather_bert():
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
    result = sdfg(embs=embs.copy(), input_ids=input_ids.copy())
    assert_allclose(result, embs[input_ids])


@pytest.mark.onnx
def test_gather_scalar():
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
    result = sdfg(inp=data.copy(), indices=indices.copy())
    np_result = np.take(data, indices, axis=1)

    assert_allclose(result, np_result)


@pytest.mark.onnx
def test_gather_onnx_2():
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
    result = sdfg(inp=data.copy(), indices=indices.copy())
    np_result = np.take(data, indices, axis=1)

    assert_allclose(result, np_result)


@pytest.mark.onnx
def test_unsqueeze():

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

    result = sdfg(inp=data.copy())
    assert result.shape == (3, 1, 3, 1)
    assert_allclose(result, np_result)
