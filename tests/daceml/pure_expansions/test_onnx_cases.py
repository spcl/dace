# """
# Runs the onnx backend tests for every operator that has a pure implementation

# Resources:
# https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md
# https://github.com/onnx/onnx-coreml/blob/master/tests/onnx_backend_node_test.py
# """

# import inspect
# import pytest

# import onnx.backend.test
# import onnx.backend.test.report

# from dace.libraries.onnx import DaCeMLBackend
# from dace.libraries.onnx.nodes.onnx_op import ONNXOp
# from dace.libraries.onnx.forward_implementation_abc import ONNXForward

# ALL_PURE_IMPLS = {}
# for impl, args in ONNXForward.extensions().items():
#     if "op" in args and "name" in args and args["name"] == "pure":
#         ALL_PURE_IMPLS[args["op"]] = impl


# class DaCeMLPureBackend(DaCeMLBackend):
#     @classmethod
#     def is_compatible(cls, model, device='CPU', **kwargs):
#         ops = {n.op_type for n in model.graph.node}

#         # empty difference means all ops are compatible
#         if ops.difference(ALL_PURE_IMPLS):
#             return False

#         # further, check that every pure op can be expanded
#         sdfg = cls.prepare(model).model.sdfg
#         for n, parent in sdfg.all_nodes_recursive():
#             if isinstance(n, ONNXOp):
#                 parent_sdfg = parent.parent
#                 if not ALL_PURE_IMPLS[n.schema.name].forward_can_be_applied(
#                         n, parent, parent_sdfg):
#                     return False

#         return True


# backend_test = onnx.backend.test.runner.Runner(DaCeMLPureBackend, __name__)
# backend_test.enable_report()

# EXCLUDED = [
#     'test_cast_DOUBLE_to_FLOAT16_cpu',
#     'test_cast_BFLOAT16_to_FLOAT_cpu',
#     'test_cast_FLOAT16_to_DOUBLE_cpu',
#     'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
#     'test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu',
#     'test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
#     'test_cast_FLOAT16_to_FLOAT8E5M2_cpu',
#     'test_cast_FLOAT16_to_INT4_cpu',
#     'test_cast_FLOAT16_to_UINT4_cpu',
#     'test_cast_FLOAT8E4M3FNUZ_to_FLOAT16_cpu',
#     'test_cast_FLOAT8E4M3FNUZ_to_FLOAT_cpu',
#     'test_cast_FLOAT8E4M3FN_to_FLOAT16_cpu',
#     'test_cast_FLOAT8E4M3FN_to_FLOAT_cpu',
#     'test_cast_FLOAT8E5M2FNUZ_to_FLOAT16_cpu',
#     'test_cast_FLOAT8E5M2FNUZ_to_FLOAT_cpu',
#     'test_cast_FLOAT8E5M2_to_FLOAT16_cpu',
#     'test_cast_FLOAT8E5M2_to_FLOAT_cpu',
#     'test_cast_FLOAT_to_BFLOAT16_cpu',
#     'test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
#     'test_cast_FLOAT_to_FLOAT8E4M3FN_cpu',
#     'test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
#     'test_cast_FLOAT_to_FLOAT8E5M2_cpu',
#     'test_cast_FLOAT_to_INT4_cpu',
#     'test_cast_FLOAT_to_UINT4_cpu',
#     'test_cast_INT4_to_FLOAT16_cpu',
#     'test_cast_INT4_to_FLOAT_cpu',
#     'test_cast_INT4_to_INT8_cpu',
#     'test_cast_UINT4_to_FLOAT16_cpu',
#     'test_cast_UINT4_to_FLOAT_cpu',
#     'test_cast_UINT4_to_UINT8_cpu',
#     'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
#     'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu',
#     'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
#     'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu',
#     'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
#     'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu',
#     'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
#     'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu',
#     'test_castlike_BFLOAT16_to_FLOAT_expanded_cpu',
#     'test_castlike_DOUBLE_to_FLOAT16_expanded_cpu',
#     'test_castlike_FLOAT16_to_DOUBLE_expanded_cpu',
#     'test_castlike_FLOAT16_to_FLOAT_expanded_cpu',
#     'test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded_cpu',
#     'test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded_cpu',
#     'test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded_cpu',
#     'test_castlike_FLOAT8E5M2_to_FLOAT_expanded_cpu',
#     'test_castlike_FLOAT_to_BFLOAT16_expanded_cpu',
#     'test_castlike_FLOAT_to_FLOAT16_expanded_cpu',
#     'test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded_cpu',
#     'test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded_cpu',
#     'test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded_cpu',
#     'test_castlike_FLOAT_to_FLOAT8E5M2_expanded_cpu',
#     'test_castlike_FLOAT_to_STRING_expanded_cpu',
#     'test_castlike_STRING_to_FLOAT_expanded_cpu',
#     'test_cast_FLOAT16_to_FLOAT_cpu',
#     'test_cast_FLOAT_to_FLOAT16_cpu',
#     'test_cast_FLOAT_to_STRING_cpu',
#     'test_cast_STRING_to_FLOAT_cpu',
#     'test_clip_default_inbounds_cpu',
#     'test_clip_default_int8_inbounds_cpu',
#     'test_clip_default_int8_max_cpu',
#     'test_clip_default_int8_min_cpu',
#     'test_clip_default_max_cpu',
#     'test_clip_default_min_cpu',
#     'test_einsum_batch_diagonal_cpu',
#     'test_einsum_batch_matmul_cpu',
#     'test_einsum_inner_prod_cpu',
#     'test_expand_dim_changed_cpu',
#     'test_expand_dim_unchanged_cpu',
#     'test_gather_negative_indices_cpu',
#     'test_gemm_all_attributes_cpu',
#     'test_gemm_alpha_cpu',
#     'test_gemm_beta_cpu',
#     'test_gemm_default_no_bias_cpu',
#     'test_gemm_default_scalar_bias_cpu',
#     'test_gemm_default_single_elem_vector_bias_cpu',
#     'test_gemm_default_vector_bias_cpu',
#     'test_gemm_default_zero_bias_cpu',
#     'test_gemm_transposeA_cpu',
#     'test_gemm_transposeB_cpu',
#     'test_reshape_extended_dims_cpu',
#     'test_reshape_negative_dim_cpu',
#     'test_reshape_negative_extended_dims_cpu',
#     'test_reshape_one_dim_cpu',
#     'test_reshape_reduced_dims_cpu',
#     'test_reshape_reordered_all_dims_cpu',
#     'test_reshape_reordered_last_dims_cpu',
#     'test_reshape_zero_and_negative_dim_cpu',
#     'test_reshape_zero_dim_cpu',
#     'test_split_equal_parts_1d_cpu',
#     'test_split_equal_parts_2d_cpu',
#     'test_split_equal_parts_default_axis_cpu',
# ]
# for test in EXCLUDED:
#     backend_test.exclude(test)

# cases = backend_test.test_cases['OnnxBackendNodeModelTest']
# for name, func in inspect.getmembers(cases):
#     if name.startswith("test"):
#         setattr(cases, name, pytest.mark.onnx(func))
#     if name.endswith("cuda"):
#         setattr(cases, name, pytest.mark.gpu(func))
