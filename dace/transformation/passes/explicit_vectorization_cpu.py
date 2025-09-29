# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import IntegerPowerToMult, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.explicit_vectorization import ExplicitVectorization

class ExplicitVectorizationPipelineCPU(ppl.Pipeline):
    _cpu_global_code = """
inline void vector_mult(double * __restrict__ c, const double * __restrict__ a, const double * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}
inline void vector_mult_w_scalar(double * __restrict__ b, const double * __restrict__ a, const double constant) {{
    double cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] * cReg[i];
    }}
}}
inline void vector_add(double * __restrict__ c, const double * __restrict__ a, const double * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}
inline void vector_add_w_scalar(double * __restrict__ b, const double * __restrict__ a, const double constant) {{
    double cReg[{vector_width}];
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        cReg[i] = constant;
    }}
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] + cReg[i];
    }}
}}
inline void vector_copy(double * __restrict__ dst, const double * __restrict__ src) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        dst[i] = src[i];
    }}
}}
"""

    def __init__(self, vector_width):
        passes = [
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
                    "=": "vector_copy({lhs}, {rhs1});",
                    "c+": "vector_add_w_scalar({lhs}, {rhs1}, {constant});",
                    "c*": "vector_mult_w_scalar({lhs}, {rhs1}, {constant});",
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.Register,
                vector_output_storage=dace.dtypes.StorageType.Register,
                global_code=ExplicitVectorizationPipelineCPU._cpu_global_code.format(vector_width=vector_width),
                global_code_location="frame")
        ]
        super().__init__(passes)
