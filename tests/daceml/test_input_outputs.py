"""
Testing input and output combinations for onnx Ops

| Output / Input | Scalar CPU | Scalar GPU | Array CPU | Array GPU |
|----------------+------------+------------+-----------+-----------|
| Scalar CPU     | Add        |            | Shape     |           |
| Scalar GPU     |            | Add        |           | Squeeze   |
| Array CPU      | Unsqueeze  |            | Add       | Shape     |
| Array GPU      |            | Unsqueeze  |           | Add       |

ALSO: test CPU fallback for all combinations of GPU ops
"""
import types
from contextlib import suppress

import numpy as np
import pytest

import dace
import dace.libraries.onnx as donnx
from dace.libraries.ort_api import ORTAPIError
from dace.libraries.ort_api.python_bindings import ExecutableKernelContext


class BreakOpChecker:
    def __enter__(self):
        # monkey patch try_create to always fail
        self.old_try_create = ExecutableKernelContext.try_create_kernel

        def fail_create(self, provider_id):
            raise ORTAPIError("oh no :(")

        ExecutableKernelContext.try_create_kernel = types.MethodType(
            fail_create, ExecutableKernelContext)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # undo the change
        ExecutableKernelContext.try_create_kernel = self.old_try_create


@pytest.mark.ort
@pytest.mark.parametrize("break_opchecker", [True, False])
@pytest.mark.parametrize("simplify", [True, False])
def test_squeeze(gpu, simplify, break_opchecker, sdfg_name):

    with BreakOpChecker() if break_opchecker else suppress():
        sdfg = dace.SDFG(sdfg_name)

        sdfg.add_array("X_arr", [1], dace.float32)
        sdfg.add_array("axes", [1], dace.int64, transient=True)
        sdfg.add_scalar("scalar", dace.float32, transient=True)
        sdfg.add_array("__return", [1], dace.float32)

        state = sdfg.add_state()
        access_X = state.add_access("X_arr")
        access_axes = state.add_access("axes")
        access_scalar = state.add_access("scalar")

        access_result = state.add_access("__return")

        # Tasklet to initialize axes
        init_axes = state.add_tasklet("init_axes", inputs={}, outputs={"__axes": dace.pointer(dace.int64)}, code="__axes[0] = 0;", language=dace.Language.CPP)

        state.add_edge(init_axes, "__axes", access_axes, None,
                       sdfg.make_array_memlet("axes"))

        op_node = donnx.ONNXSqueeze("Squeeze")

        state.add_node(op_node)
        state.add_edge(access_X, None, op_node, "data",
                       sdfg.make_array_memlet("X_arr"))

        state.add_edge(op_node, "squeezed", access_scalar, None,
                       sdfg.make_array_memlet("scalar"))
        
        unsqueeze_op = donnx.ONNXUnsqueeze("Unsqueeze")
        state.add_node(unsqueeze_op)
        state.add_edge(access_scalar, None, unsqueeze_op, "data",
                       sdfg.make_array_memlet("scalar"))
        state.add_edge(access_axes, None, unsqueeze_op, "axes",
                       sdfg.make_array_memlet("axes"))
        state.add_edge(unsqueeze_op, "expanded", access_result, None,
                       sdfg.make_array_memlet("__return"))

        X = np.random.rand(1).astype(np.float32)

        if gpu:
            sdfg.apply_gpu_transformations()

        if simplify:
            sdfg.expand_library_nodes()
            sdfg.simplify()

        result = sdfg(X_arr=X)

        assert result.shape == (1, )
        assert result[0] == X


@pytest.mark.ort
@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize("break_opchecker", [True, False])
def test_shape(gpu, simplify, break_opchecker, sdfg_name):
    with BreakOpChecker() if break_opchecker else suppress():
        sdfg = dace.SDFG(sdfg_name)

        sdfg.add_array("X_arr", [2, 4], dace.float32)
        sdfg.add_array("__return", [2], dace.int64)

        state = sdfg.add_state()
        access_X = state.add_access("X_arr")

        access_result = state.add_access("__return")

        op_node = donnx.ONNXShape("Shape")

        state.add_node(op_node)
        state.add_edge(access_X, None, op_node, "data",
                       sdfg.make_array_memlet("X_arr"))

        state.add_edge(op_node, "shape", access_result, None,
                       sdfg.make_array_memlet("__return"))

        X = np.random.rand(2, 4).astype(np.float32)

        if gpu:
            sdfg.apply_gpu_transformations()

        if simplify:
            sdfg.expand_library_nodes()
            sdfg.simplify()

        result = sdfg(X_arr=X)

        assert np.all(result == (2, 4))


@pytest.mark.ort
@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize("break_opchecker", [True, False])
def test_unsqueeze(gpu, simplify, break_opchecker, sdfg_name):
    with BreakOpChecker() if break_opchecker else suppress():
        sdfg = dace.SDFG(sdfg_name)

        sdfg.add_scalar("X_arr", dace.float32)
        sdfg.add_array("axes", [1], dace.int64, transient=True)
        sdfg.add_array("__return", [1], dace.float32)

        state = sdfg.add_state()
        access_X = state.add_access("X_arr")
        access_axes = state.add_access("axes")

        access_result = state.add_access("__return")

        # Tasklet to initialize axes
        init_axes = state.add_tasklet("init_axes", inputs={}, outputs={"__axes": dace.pointer(dace.int64)}, code="__axes[0] = 0;", language=dace.Language.CPP)

        state.add_edge(init_axes, "__axes", access_axes, None,
                       sdfg.make_array_memlet("axes"))

        op_node = donnx.ONNXUnsqueeze("Unsqueeze")

        state.add_node(op_node)
        state.add_edge(access_X, None, op_node, "data",
                       sdfg.make_array_memlet("X_arr"))
        state.add_edge(access_axes, None, op_node, "axes",
                       sdfg.make_array_memlet("axes"))

        state.add_edge(op_node, "expanded", access_result, None,
                       sdfg.make_array_memlet("__return"))

        X = np.float32(np.random.rand())

        if gpu:
            sdfg.apply_gpu_transformations()

        if simplify:
            sdfg.expand_library_nodes()
            sdfg.simplify()

        result = sdfg(X_arr=X)

        assert result.shape == (1, )
        assert X == result[0]


@pytest.mark.ort
@pytest.mark.parametrize("scalars", [True, False])
@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize("break_opchecker", [True, False])
def test_add(gpu, scalars, simplify, break_opchecker, sdfg_name):
    with BreakOpChecker() if break_opchecker else suppress():
        sdfg = dace.SDFG(sdfg_name)

        if scalars:
            sdfg.add_scalar("X_arr", dace.float32)
            sdfg.add_scalar("W_arr", dace.float32)
            sdfg.add_scalar("Z_arr", dace.float32, transient=True)
            sdfg.add_array("axes", [1], dace.int64, transient=True)
            sdfg.add_array("__return", [1], dace.float32)
        else:
            sdfg.add_array("X_arr", [2, 2], dace.float32)
            sdfg.add_array("W_arr", [2, 2], dace.float32)
            sdfg.add_array("__return", [2, 2], dace.float32)

        state = sdfg.add_state()
        access_X = state.add_access("X_arr")
        access_W = state.add_access("W_arr")

        if scalars:
            access_Z = state.add_access("Z_arr")
            access_axes = state.add_access("axes")

        access_result = state.add_access("__return")

        op_node = donnx.ONNXAdd("Add")

        state.add_node(op_node)
        state.add_edge(access_X, None, op_node, "A",
                       sdfg.make_array_memlet("X_arr"))
        state.add_edge(access_W, None, op_node, "B",
                       sdfg.make_array_memlet("W_arr"))

        if scalars:
            state.add_edge(op_node, "C", access_Z, None,
                           sdfg.make_array_memlet("Z_arr"))
        else:
            state.add_edge(op_node, "C", access_result, None,
                           sdfg.make_array_memlet("__return"))

        if scalars:
            # Tasklet to initialize axes
            init_axes = state.add_tasklet("init_axes", inputs={}, outputs={"__axes": dace.pointer(dace.int64)}, code="__axes[0] = 0;", language=dace.Language.CPP)

            state.add_edge(init_axes, "__axes", access_axes, None,
                           sdfg.make_array_memlet("axes"))

            unsqueeze_op = donnx.ONNXUnsqueeze("Unsqueeze")
            state.add_node(unsqueeze_op)
            state.add_edge(access_Z, None, unsqueeze_op, "data",
                           sdfg.make_array_memlet("Z_arr"))
            state.add_edge(access_axes, None, unsqueeze_op, "axes",
                           sdfg.make_array_memlet("axes"))
            state.add_edge(unsqueeze_op, "expanded", access_result, None,
                           sdfg.make_array_memlet("__return"))

        shapes = [] if scalars else [2, 2]
        X = np.random.rand(*shapes)
        W = np.random.rand(*shapes)
        if not scalars:
            X = X.astype(np.float32)
            W = W.astype(np.float32)

        if gpu:
            sdfg.apply_gpu_transformations()

        if simplify:
            sdfg.expand_library_nodes()
            sdfg.simplify()

        result = sdfg(X_arr=X, W_arr=W)

        numpy_result = X + W

        assert np.allclose(result, numpy_result)
