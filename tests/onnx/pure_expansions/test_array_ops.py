# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import dace
import dace.libraries.onnx as donnx


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# ==============================================================================
# Concatenation Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_concat_3_inputs(axis, sdfg_name):
    """Test Concat operation with 3 inputs along different axes."""

    shapes = {0: [6, 3, 4], 1: [2, 9, 4], 2: [2, 3, 12]}
    result_shape = shapes[axis]

    @dace.program
    def concat_prog(A: dace.float32[2, 3, 4], B: dace.float32[2, 3, 4], C: dace.float32[2, 3, 4]):
        result = dace.define_local(result_shape, dace.float32)
        donnx.ONNXConcat(inputs__0=A, inputs__1=B, inputs__2=C, concat_result=result, axis=axis)
        return result

    concat_prog.__name__ = sdfg_name

    A = np.random.randn(2, 3, 4).astype(np.float32)
    B = np.random.randn(2, 3, 4).astype(np.float32)
    C = np.random.randn(2, 3, 4).astype(np.float32)

    sdfg = concat_prog.to_sdfg()

    result = sdfg(A=A, B=B, C=C)
    expected = np.concatenate([A, B, C], axis=axis)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_concat_2_inputs(sdfg_name):
    """Test Concat operation with 2 inputs of different sizes."""

    @dace.program
    def concat_prog(A: dace.float32[2, 3], B: dace.float32[2, 5]):
        result = dace.define_local([2, 8], dace.float32)
        donnx.ONNXConcat(inputs__0=A, inputs__1=B, concat_result=result, axis=1)
        return result

    concat_prog.__name__ = sdfg_name

    A = np.random.randn(2, 3).astype(np.float32)
    B = np.random.randn(2, 5).astype(np.float32)

    sdfg = concat_prog.to_sdfg()

    result = sdfg(A=A, B=B)
    expected = np.concatenate([A, B], axis=1)

    assert_allclose(result, expected)


# ==============================================================================
# Squeeze Tests
# ==============================================================================


@pytest.mark.onnx
def test_squeeze_single_axis(sdfg_name):
    """Test Squeeze operation removing a single dimension."""

    @dace.program
    def squeeze_prog(inp: dace.float32[2, 1, 3, 4]):
        result = dace.define_local([2, 3, 4], dace.float32)
        axes = dace.define_local([1], dace.int64)
        axes[0] = 1
        donnx.ONNXSqueeze(data=inp, squeezed=result, axes=axes)
        return result

    squeeze_prog.__name__ = sdfg_name

    inp = np.random.randn(2, 1, 3, 4).astype(np.float32)

    sdfg = squeeze_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp)
    expected = np.squeeze(inp, axis=1)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_squeeze_multiple_axes(sdfg_name):
    """Test Squeeze operation removing multiple dimensions."""

    @dace.program
    def squeeze_prog(inp: dace.float32[1, 2, 1, 3, 1]):
        result = dace.define_local([2, 3], dace.float32)
        axes = dace.define_local([3], dace.int64)
        axes[0] = 0
        axes[1] = 2
        axes[2] = 4
        donnx.ONNXSqueeze(data=inp, squeezed=result, axes=axes)
        return result

    squeeze_prog.__name__ = sdfg_name

    inp = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)

    sdfg = squeeze_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)
    expected = np.squeeze(inp, axis=(0, 2, 4))

    assert_allclose(result, expected)


# ==============================================================================
# Expand Tests
# ==============================================================================


@pytest.mark.onnx
def test_expand_broadcast(sdfg_name):
    """Test Expand operation broadcasting to larger shape."""

    @dace.program
    def expand_prog(inp: dace.float32[3, 1], shape_arr: dace.int64[2]):
        result = dace.define_local([3, 4], dace.float32)
        donnx.ONNXExpand(input=inp, shape=shape_arr, output=result)
        return result

    expand_prog.__name__ = sdfg_name

    inp = np.random.randn(3, 1).astype(np.float32)
    shape_arr = np.array([3, 4], dtype=np.int64)

    sdfg = expand_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, shape_arr=shape_arr)
    expected = np.broadcast_to(inp, (3, 4))

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_expand_higher_rank(sdfg_name):
    """Test Expand operation adding dimensions."""

    @dace.program
    def expand_prog(inp: dace.float32[3], shape_arr: dace.int64[3]):
        result = dace.define_local([2, 1, 3], dace.float32)
        donnx.ONNXExpand(input=inp, shape=shape_arr, output=result)
        return result

    expand_prog.__name__ = sdfg_name

    inp = np.random.randn(3).astype(np.float32)
    shape_arr = np.array([2, 1, 3], dtype=np.int64)

    sdfg = expand_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp, shape_arr=shape_arr)
    expected = np.broadcast_to(inp, (2, 1, 3))

    assert_allclose(result, expected)


# ==============================================================================
# Transpose Tests
# ==============================================================================


@pytest.mark.onnx
@pytest.mark.parametrize("perm", [[0, 2, 1], [1, 0, 2], [2, 1, 0]])
def test_transpose(perm, sdfg_name):
    """Test Transpose operation with different permutations."""

    inp_shape = [2, 3, 4]
    out_shape = [inp_shape[i] for i in perm]

    @dace.program
    def transpose_prog(inp: dace.float32[2, 3, 4]):
        result = dace.define_local(out_shape, dace.float32)
        donnx.ONNXTranspose(data=inp, transposed=result, perm=perm)
        return result

    transpose_prog.__name__ = sdfg_name

    inp = np.random.randn(2, 3, 4).astype(np.float32)

    sdfg = transpose_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)
    expected = np.transpose(inp, perm)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_transpose_2d(sdfg_name):
    """Test Transpose operation on 2D matrix."""

    @dace.program
    def transpose_prog(inp: dace.float32[4, 5]):
        result = dace.define_local([5, 4], dace.float32)
        donnx.ONNXTranspose(data=inp, transposed=result, perm=[1, 0])
        return result

    transpose_prog.__name__ = sdfg_name

    inp = np.random.randn(4, 5).astype(np.float32)

    sdfg = transpose_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)
    expected = np.transpose(inp)

    assert_allclose(result, expected)


# ==============================================================================
# Slice Tests
# ==============================================================================


@pytest.mark.onnx
def test_slice_constant_all_axes(sdfg_name):
    """Test Slice operation with constant parameters on all axes."""

    @dace.program
    def slice_prog(inp: dace.float32[10, 20, 30]):
        starts = dace.define_local([3], dace.int64)
        ends = dace.define_local([3], dace.int64)
        axes = dace.define_local([3], dace.int64)
        steps = dace.define_local([3], dace.int64)
        result = dace.define_local([5, 10, 15], dace.float32)

        starts[0] = 0
        starts[1] = 5
        starts[2] = 10
        ends[0] = 5
        ends[1] = 15
        ends[2] = 25
        axes[0] = 0
        axes[1] = 1
        axes[2] = 2
        steps[0] = 1
        steps[1] = 1
        steps[2] = 1

        donnx.ONNXSlice(data=inp, starts=starts, ends=ends, axes=axes, steps=steps, output=result)
        return result

    slice_prog.__name__ = sdfg_name

    inp = np.random.randn(10, 20, 30).astype(np.float32)

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp)
    expected = inp[0:5, 5:15, 10:25]

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_slice_dynamic_params(sdfg_name):
    """Test Slice operation with dynamic (runtime) parameters."""

    @dace.program
    def slice_prog(inp: dace.float32[10, 20], starts: dace.int64[2], ends: dace.int64[2], axes: dace.int64[2],
                   steps: dace.int64[2]):
        # Output shape must match the expected slice result
        result = dace.define_local([5, 10], dace.float32)
        donnx.ONNXSlice(data=inp, starts=starts, ends=ends, axes=axes, steps=steps, output=result)
        return result

    slice_prog.__name__ = sdfg_name

    inp = np.random.randn(10, 20).astype(np.float32)
    starts = np.array([0, 5], dtype=np.int64)
    ends = np.array([5, 15], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp, starts=starts, ends=ends, axes=axes, steps=steps)
    expected = inp[0:5, 5:15]

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_slice_single_axis(sdfg_name):
    """Test Slice operation on a single axis with constant params."""

    @dace.program
    def slice_prog(inp: dace.float32[10, 20]):
        starts = dace.define_local([1], dace.int64)
        ends = dace.define_local([1], dace.int64)
        axes = dace.define_local([1], dace.int64)
        steps = dace.define_local([1], dace.int64)
        result = dace.define_local([5, 20], dace.float32)

        starts[0] = 2
        ends[0] = 7
        axes[0] = 0
        steps[0] = 1

        donnx.ONNXSlice(data=inp, starts=starts, ends=ends, axes=axes, steps=steps, output=result)
        return result

    slice_prog.__name__ = sdfg_name

    inp = np.random.randn(10, 20).astype(np.float32)

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp)
    expected = inp[2:7, :]

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_slice_with_steps(sdfg_name):
    """Test Slice operation with step > 1."""

    @dace.program
    def slice_prog(inp: dace.float32[20]):
        starts = dace.define_local([1], dace.int64)
        ends = dace.define_local([1], dace.int64)
        axes = dace.define_local([1], dace.int64)
        steps = dace.define_local([1], dace.int64)
        result = dace.define_local([5], dace.float32)

        starts[0] = 0
        ends[0] = 15
        axes[0] = 0
        steps[0] = 3

        donnx.ONNXSlice(data=inp, starts=starts, ends=ends, axes=axes, steps=steps, output=result)
        return result

    slice_prog.__name__ = sdfg_name

    inp = np.random.randn(20).astype(np.float32)

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp)
    expected = inp[0:15:3]

    assert_allclose(result, expected)


@pytest.mark.onnx
@pytest.mark.parametrize("start, end", [(0, 5), (2, 8), (5, 10)])
def test_slice_different_ranges(start, end, sdfg_name):
    """Test Slice operation with different ranges."""

    @dace.program
    def slice_prog(inp: dace.float32[20]):
        starts = dace.define_local([1], dace.int64)
        ends = dace.define_local([1], dace.int64)
        axes = dace.define_local([1], dace.int64)
        steps = dace.define_local([1], dace.int64)
        result = dace.define_local([end - start], dace.float32)

        starts[0] = start
        ends[0] = end
        axes[0] = 0
        steps[0] = 1

        donnx.ONNXSlice(data=inp, starts=starts, ends=ends, axes=axes, steps=steps, output=result)
        return result

    slice_prog.__name__ = sdfg_name

    inp = np.random.randn(20).astype(np.float32)

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    sdfg.simplify()

    result = sdfg(inp=inp)
    expected = inp[start:end]

    assert_allclose(result, expected)


# ==============================================================================
# Split Tests
# ==============================================================================


@pytest.mark.onnx
def test_split_equal(sdfg_name):
    """Test Split operation with equal splits."""

    sdfg = dace.SDFG(sdfg_name)

    inp = np.random.randn(6, 4).astype(np.float32)
    split = np.array([2, 2, 2], dtype=np.int64)

    sdfg.add_array("inp", inp.shape, dace.float32)
    sdfg.add_array("split", split.shape, dace.int64)
    sdfg.add_array("out0", [2, 4], dace.float32)
    sdfg.add_array("out1", [2, 4], dace.float32)
    sdfg.add_array("out2", [2, 4], dace.float32)

    state = sdfg.add_state()

    op_node = donnx.ONNXSplit("split", axis=0, optional={'split'})
    state.add_node(op_node)

    # Add variadic output connectors
    op_node.add_out_connector("outputs__0")
    op_node.add_out_connector("outputs__1")
    op_node.add_out_connector("outputs__2")

    state.add_edge(state.add_read("inp"), None, op_node, "input", sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("split"), None, op_node, "split", sdfg.make_array_memlet("split"))
    state.add_edge(op_node, "outputs__0", state.add_write("out0"), None, sdfg.make_array_memlet("out0"))
    state.add_edge(op_node, "outputs__1", state.add_write("out1"), None, sdfg.make_array_memlet("out1"))
    state.add_edge(op_node, "outputs__2", state.add_write("out2"), None, sdfg.make_array_memlet("out2"))

    sdfg.expand_library_nodes()

    out0, out1, out2 = sdfg(inp=inp, split=split)

    expected = np.split(inp, [2, 4], axis=0)

    assert_allclose(out0, expected[0])
    assert_allclose(out1, expected[1])
    assert_allclose(out2, expected[2])


# ==============================================================================
# Where Tests
# ==============================================================================


@pytest.mark.onnx
def test_where(sdfg_name):
    """Test Where operation for conditional selection."""

    @dace.program
    def where_prog(cond: dace.bool_[5, 5], X: dace.float32[5, 5], Y: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXWhere(condition=cond, X=X, Y=Y, output=result)
        return result

    where_prog.__name__ = sdfg_name

    cond = (np.random.randn(5, 5) > 0)
    X = np.random.randn(5, 5).astype(np.float32)
    Y = np.random.randn(5, 5).astype(np.float32)

    sdfg = where_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(cond=cond, X=X, Y=Y)
    expected = np.where(cond, X, Y)

    assert_allclose(result, expected)


@pytest.mark.onnx
def test_where_broadcast(sdfg_name):
    """Test Where operation with broadcasting."""

    @dace.program
    def where_prog(cond: dace.bool_[5, 1], X: dace.float32[5, 3], Y: dace.float32[5, 3]):
        result = dace.define_local([5, 3], dace.float32)
        donnx.ONNXWhere(condition=cond, X=X, Y=Y, output=result)
        return result

    where_prog.__name__ = sdfg_name

    cond = (np.random.randn(5, 1) > 0)
    X = np.random.randn(5, 3).astype(np.float32)
    Y = np.random.randn(5, 3).astype(np.float32)

    sdfg = where_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(cond=cond, X=X, Y=Y)
    expected = np.where(cond, X, Y)

    assert_allclose(result, expected)


# ==============================================================================
# Identity Tests
# ==============================================================================


@pytest.mark.onnx
def test_identity(sdfg_name):
    """Test Identity operation (pass-through)."""

    @dace.program
    def identity_prog(inp: dace.float32[5, 5]):
        result = dace.define_local([5, 5], dace.float32)
        donnx.ONNXIdentity(input=inp, output=result)
        return result

    identity_prog.__name__ = sdfg_name

    inp = np.random.randn(5, 5).astype(np.float32)

    sdfg = identity_prog.to_sdfg()
    sdfg.expand_library_nodes()

    result = sdfg(inp=inp)

    assert_allclose(result, inp)


@pytest.mark.onnx
def test_greater_or_equal():
    """Test GreaterOrEqual operator with DaCe ONNX frontend."""

    @dace
    def gte_prog(a: dace.float32[5], b: dace.float32[5]):
        out = dace.define_local([5], dace.bool_)
        donnx.ONNXGreaterOrEqual(A=a, B=b, C=out)
        return out

    A = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    B = np.array([1, 2, 4, 3, 6], dtype=np.float32)
    result = gte_prog(a=A.copy(), b=B.copy())
    expected = np.greater_equal(A, B)

    np.testing.assert_array_equal(result, expected, err_msg="GreaterOrEqual output mismatch")


@pytest.mark.onnx
def test_topk():
    """Test TopK operator with DaCe ONNX frontend."""

    import onnx
    from onnx import helper, TensorProto

    # Create a TopK model
    input_tensor = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 5])
    k_tensor = helper.make_tensor('K', TensorProto.INT64, [], [3])
    values_tensor = helper.make_tensor_value_info('Values', TensorProto.FLOAT, [2, 3])
    indices_tensor = helper.make_tensor_value_info('Indices', TensorProto.INT64, [2, 3])

    node = helper.make_node('TopK', inputs=['X', 'K'], outputs=['Values', 'Indices'], axis=-1, largest=1, sorted=1)

    graph = helper.make_graph([node], 'topk_test', [input_tensor], [values_tensor, indices_tensor], [k_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    # Test with DaCe
    from dace.libraries.onnx import ONNXModel
    dace_model = ONNXModel("test_topk", model)

    X = np.array([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]], dtype=np.float32)
    values, indices = dace_model(X=X)

    # Check if values are correct (indices might differ for ties)
    for i in range(X.shape[0]):
        sorted_row = np.sort(X[i])[::-1][:3]
        np.testing.assert_allclose(np.sort(values[i])[::-1],
                                   sorted_row,
                                   atol=1e-5,
                                   rtol=1e-5,
                                   err_msg=f"TopK values mismatch for row {i}")


@pytest.mark.onnx
def test_range():
    """Test Range operator with DaCe ONNX frontend."""

    import onnx
    from onnx import helper, TensorProto

    # Create a Range model
    start_tensor = helper.make_tensor('start', TensorProto.INT64, [], [2])
    limit_tensor = helper.make_tensor('limit', TensorProto.INT64, [], [10])
    delta_tensor = helper.make_tensor('delta', TensorProto.INT64, [], [2])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [4])

    node = helper.make_node('Range', inputs=['start', 'limit', 'delta'], outputs=['output'])

    graph = helper.make_graph([node], 'range_test', [], [output_tensor], [start_tensor, limit_tensor, delta_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    # Test with DaCe
    from dace.libraries.onnx import ONNXModel
    dace_model = ONNXModel("range", model)

    result = dace_model()
    expected = np.arange(2, 10, 2, dtype=np.int64)

    np.testing.assert_array_equal(result, expected, err_msg="Range output mismatch")


@pytest.mark.onnx
def test_constant_of_shape():
    """Test ConstantOfShape operator with DaCe ONNX frontend."""

    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Create a ConstantOfShape model
    shape_tensor = helper.make_tensor('shape', TensorProto.INT64, [2], [3, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4])

    # Create value attribute (scalar tensor)
    value_attr = numpy_helper.from_array(np.array([5.0], dtype=np.float32), name='value')

    node = helper.make_node('ConstantOfShape', inputs=['shape'], outputs=['output'], value=value_attr)

    graph = helper.make_graph([node], 'constant_of_shape_test', [], [output_tensor], [shape_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    # Test with DaCe
    from dace.libraries.onnx import ONNXModel
    dace_model = ONNXModel("constant_of_shape", model)

    result = dace_model()
    expected = np.full([3, 4], 5.0, dtype=np.float32)

    np.testing.assert_array_equal(result, expected, err_msg="ConstantOfShape output mismatch")


@pytest.mark.onnx
def test_gather_nd():
    """Test GatherND operator with DaCe ONNX frontend."""

    import onnx
    from onnx import helper, TensorProto

    # Create a GatherND model
    data_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2])
    indices_tensor = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 2])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2])

    node = helper.make_node('GatherND', inputs=['data', 'indices'], outputs=['output'], batch_dims=0)

    graph = helper.make_graph([node], 'gather_nd_test', [data_tensor, indices_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    # Test with DaCe
    from dace.libraries.onnx import ONNXModel
    dace_model = ONNXModel("gather_nd", model)

    data = np.array([[0, 1], [2, 3]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    result = dace_model(data=data, indices=indices)
    expected = np.array([0, 3], dtype=np.float32)

    np.testing.assert_array_equal(result, expected, err_msg="GatherND output mismatch")


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

    result = sdfg(X=X)

    assert_allclose(X.astype(np.int64), result)


@pytest.mark.onnx
def test_scatter_nd():
    """Test ScatterND operator with DaCe ONNX frontend."""

    import onnx
    from onnx import helper, TensorProto

    # Create a ScatterND model
    data_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [4, 4])
    indices_tensor = helper.make_tensor_value_info('indices', TensorProto.INT64, [3, 2])
    updates_tensor = helper.make_tensor_value_info('updates', TensorProto.FLOAT, [3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 4])

    node = helper.make_node('ScatterND', inputs=['data', 'indices', 'updates'], outputs=['output'])

    graph = helper.make_graph([node], 'scatter_nd_test', [data_tensor, indices_tensor, updates_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # Use IR version 8 for compatibility

    # Test with DaCe
    from dace.libraries.onnx import ONNXModel
    dace_model = ONNXModel("scatter_nd", model)

    data = np.zeros((4, 4), dtype=np.float32)
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    updates = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = dace_model(data=data, indices=indices, updates=updates)

    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    expected[2, 2] = 3.0

    np.testing.assert_array_equal(result, expected, err_msg="ScatterND output mismatch")


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

    result = sdfg(X=X)

    assert_allclose(numpy_result, result)


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


if __name__ == "__main__":
    for axis in [0, 1, 2]:
        test_concat_3_inputs(axis=axis, sdfg_name=f"test_concat_3_inputs_axis_{axis}")

    test_concat_2_inputs(sdfg_name="test_concat_2_inputs")
    test_squeeze_single_axis(sdfg_name="test_squeeze_single_axis")
    test_squeeze_multiple_axes(sdfg_name="test_squeeze_multiple_axes")
    test_expand_broadcast(sdfg_name="test_expand_broadcast")
    test_expand_higher_rank(sdfg_name="test_expand_higher_rank")

    for perm in [[0, 2, 1], [1, 0, 2], [2, 1, 0]]:
        test_transpose(perm=perm, sdfg_name=f"test_transpose_{'_'.join(map(str, perm))}")

    test_transpose_2d(sdfg_name="test_transpose_2d")

    # # Slice tests
    test_slice_constant_all_axes(sdfg_name="test_slice_constant_all_axes")
    test_slice_dynamic_params(sdfg_name="test_slice_dynamic_params")
    test_slice_single_axis(sdfg_name="test_slice_single_axis")
    test_slice_with_steps(sdfg_name="test_slice_with_steps")
    for i, (start, end) in enumerate([(0, 5), (2, 8), (5, 10)]):
        test_slice_different_ranges(start=start, end=end, sdfg_name=f"test_slice_different_ranges_{i}")

    test_split_equal(sdfg_name="test_split_equal")
    test_where(sdfg_name="test_where")
    test_where_broadcast(sdfg_name="test_where_broadcast")
    test_identity(sdfg_name="test_identity")

    test_slice_different_ranges(start=0, end=5, sdfg_name="test_slice_range_0_5")
