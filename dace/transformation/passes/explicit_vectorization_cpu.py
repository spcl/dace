# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import IntegerPowerToMult, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.explicit_vectorization import ExplicitVectorization
from dace.transformation.passes.fuse_branches_pass import FuseBranchesPass


class ExplicitVectorizationPipelineCPU(ppl.Pipeline):
    _cpu_global_code = """
template<typename T>
inline void vector_mult(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}

template<typename T>
inline void vector_mult_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] * cReg[i];
    }}
}}

template<typename T>
inline void vector_add(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}

template<typename T>
inline void vector_add_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] + cReg[i];
    }}
}}

template<typename T>
inline void vector_sub(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] - b[i];
    }}
}}

template<typename T>
inline void vector_sub_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] - cReg[i];
    }}
}}

template<typename T>
inline void vector_sub_w_scalar_c(T * __restrict__ b, const T constant, const T * __restrict__ a) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = cReg[i] - a[i];
    }}
}}

template<typename T>
inline void vector_div(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] / b[i];
    }}
}}

template<typename T>
inline void vector_div_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] / cReg[i];
    }}
}}

template<typename T>
inline void vector_div_w_scalar_c(T * __restrict__ b, const T constant, const T * __restrict__ a) {{
    T cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = cReg[i] / a[i];
    }}
}}

template<typename T>
inline void vector_copy(T * __restrict__ dst, const T * __restrict__ src) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        dst[i] = src[i];
    }}
}}
"""

    def __init__(self, vector_width):
        passes = [
            FuseBranchesPass(),
            RemoveFPTypeCasts(),
            RemoveIntTypeCasts(),
            IntegerPowerToMult(),
            SplitTasklets(),
            CleanDataToScalarSliceToTaskletPattern(),
            InlineSDFGs(),
            ExplicitVectorization(
                templates={
                    "*": "vector_mult({lhs}, {rhs1}, {rhs2});",
                    "+": "vector_add({lhs}, {rhs1}, {rhs2});",
                    "-": "vector_sub({lhs}, {rhs1}, {rhs2});",
                    "/": "vector_div({lhs}, {rhs1}, {rhs2});",
                    "=": "vector_copy({lhs}, {rhs1});",
                    "c+": "vector_add_w_scalar({lhs}, {rhs1}, {constant});",
                    "c/": "vector_div_w_scalar({lhs}, {rhs1}, {constant});",
                    "c*": "vector_mult_w_scalar({lhs}, {rhs1}, {constant});",
                    "c-": "vector_sub_w_scalar({lhs}, {rhs1}, {constant});",
                    "/c": "vector_div_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "-c": "vector_sub_w_scalar_c({lhs}, {constant}, {rhs1});",
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.Register,
                vector_output_storage=dace.dtypes.StorageType.Register,
                global_code=ExplicitVectorizationPipelineCPU._cpu_global_code.format(vector_width=vector_width),
                global_code_location="frame",
                vector_op_numeric_type=dace.float64)
        ]
        super().__init__(passes)
