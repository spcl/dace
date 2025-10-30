# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Iterator
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.duplicate_all_memlets_sharing_same_in_connector import DuplicateAllMemletsSharingSingleMapOutConnector
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.vectorization.vectorize import Vectorize
from dace.transformation.passes.eliminate_branches import EliminateBranches


class VectorizeCPU(ppl.Pipeline):
    _cpu_global_code = """
#include <cmath>


#if defined(__clang__)
    #define _dace_vectorize_hint
  #define _dace_vectorize "clang loop vectorize(enable) vectorize_width({vector_width}8)"
#elif defined(__GNUC__)
  #define _dace_vectorize_hint
  #define _dace_vectorize "omp simd simdlen({vector_width})"
#else
    #define _dace_vectorize_hint
  #define _dace_vectorize "omp simd simdlen({vector_width})"
#endif


template<typename T>
inline void vector_mult(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}

template<typename T>
inline void vector_mult_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] * constant;
    }}
}}

template<typename T>
inline void vector_add(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}

template<typename T>
inline void vector_add_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] + constant;
    }}
}}

template<typename T>
inline void vector_sub(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] - b[i];
    }}
}}

template<typename T>
inline void vector_sub_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] - constant;
    }}
}}

template<typename T>
inline void vector_sub_w_scalar_c(T * __restrict__ b, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = constant - a[i];
    }}
}}

template<typename T>
inline void vector_div(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] / b[i];
    }}
}}

template<typename T>
inline void vector_div_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = a[i] / constant;
    }}
}}

template<typename T>
inline void vector_div_w_scalar_c(T * __restrict__ b, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        b[i] = constant / a[i];
    }}
}}

template<typename T>
inline void vector_copy(T * __restrict__ dst, const T * __restrict__ src) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        dst[i] = src[i];
    }}
}}

// ---- Additional elementwise ops ----

template<typename T>
inline void vector_exp(T * __restrict__ out, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::exp(a[i]);
    }}
}}

template<typename T>
inline void vector_log(T * __restrict__ out, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::log(a[i]);
    }}
}}

template<typename T>
inline void vector_min(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::min(a[i], b[i]);
    }}
}}

template<typename T>
inline void vector_min_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::min(a[i], constant);
    }}
}}

template<typename T>
inline void vector_max(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::max(a[i], b[i]);
    }}
}}

template<typename T>
inline void vector_max_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = std::max(a[i], constant);
    }}
}}

template<typename T>
inline void vector_gt(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] > b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_gt_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] > constant) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_gt_w_scalar_c(T * __restrict__ out, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (constant > a[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_lt(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] < b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_lt_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] < constant) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_lt_w_scalar_c(T * __restrict__ out, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (constant < a[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_ge(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] >= b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_ge_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] >= constant) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_ge_w_scalar_c(T * __restrict__ out, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (constant >= a[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_le(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] <= b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_le_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] <= constant) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_le_w_scalar_c(T * __restrict__ out, const T constant, const T * __restrict__ a) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (constant <= a[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_eq(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] == b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_eq_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] == constant) ? 1.0 : 0.0;
    }}
}}


template<typename T>
inline void vector_ne(T * __restrict__ out, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] != b[i]) ? 1.0 : 0.0;
    }}
}}

template<typename T>
inline void vector_ne_w_scalar(T * __restrict__ out, const T * __restrict__ a, const T constant) {{
    #pragma _dace_vectorize_hint
    #pragma _dace_vectorize
    for (int i = 0; i < {vector_width}; i++) {{
        out[i] = (a[i] != constant) ? 1.0 : 0.0;
    }}
}}
"""

    def __init__(self, vector_width):
        passes = [
            EliminateBranches(),
            RemoveFPTypeCasts(),
            RemoveIntTypeCasts(),
            PowerOperatorExpansion(),
            SplitTasklets(),
            CleanDataToScalarSliceToTaskletPattern(),
            InlineSDFGs(),
            DuplicateAllMemletsSharingSingleMapOutConnector(),
            Vectorize(
                templates={
                    "*": "vector_mult({lhs}, {rhs1}, {rhs2});",
                    "+": "vector_add({lhs}, {rhs1}, {rhs2});",
                    "-": "vector_sub({lhs}, {rhs1}, {rhs2});",
                    "/": "vector_div({lhs}, {rhs1}, {rhs2});",
                    "=": "vector_copy({lhs}, {rhs1});",
                    "log": "vector_log({lhs}, {rhs1});",
                    "exp": "vector_exp({lhs}, {rhs1});",
                    "min": "vector_min({lhs}, {rhs1}, {rhs2});",
                    "max": "vector_max({lhs}, {rhs1}, {rhs2});",
                    ">": "vector_gt({lhs}, {rhs1}, {rhs2});",
                    "<": "vector_lt({lhs}, {rhs1}, {rhs2});",
                    ">=": "vector_ge({lhs}, {rhs1}, {rhs2});",
                    "<=": "vector_le({lhs}, {rhs1}, {rhs2});",
                    "==": "vector_eq({lhs}, {rhs1}, {rhs2});",
                    "!=": "vector_ne({lhs}, {rhs1}, {rhs2});",
                    # scalar variants type 1
                    "*c": "vector_mult_w_scalar({lhs}, {rhs1}, {constant});",
                    "+c": "vector_add_w_scalar({lhs}, {rhs1}, {constant});",
                    "-c": "vector_sub_w_scalar({lhs}, {rhs1}, {constant});",
                    "/c": "vector_div_w_scalar({lhs}, {rhs1}, {constant});",
                    "minc": "vector_min_w_scalar({lhs}, {rhs1}, {constant});",
                    "maxc": "vector_max_w_scalar({lhs}, {rhs1}, {constant});",
                    ">c": "vector_gt_w_scalar({lhs}, {rhs1}, {constant});",
                    "<c": "vector_lt_w_scalar({lhs}, {rhs1}, {constant});",
                    ">=c": "vector_ge_w_scalar({lhs}, {rhs1}, {constant});",
                    "<=c": "vector_le_w_scalar({lhs}, {rhs1}, {constant});",
                    "==c": "vector_eq_w_scalar({lhs}, {rhs1}, {constant});",
                    "!=c": "vector_ne_w_scalar({lhs}, {rhs1}, {constant});",
                    # scalar variants type 2 for non-commutative ops
                    "c-": "vector_sub_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "c/": "vector_div_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "c>": "vector_gt_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "c<": "vector_lt_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "c>=": "vector_ge_w_scalar_c({lhs}, {constant}, {rhs1});",
                    "c<=": "vector_le_w_scalar_c({lhs}, {constant}, {rhs1});",
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.Register,
                vector_output_storage=dace.dtypes.StorageType.Register,
                global_code=VectorizeCPU._cpu_global_code.format(vector_width=vector_width),
                global_code_location="frame",
                vector_op_numeric_type=dace.float64)
        ]
        super().__init__(passes)

    def iterate_over_passes(self, sdfg: dace.SDFG) -> Iterator[Pass]:
        """
            Iterates over passes in the pipeline, potentially multiple times based on which elements were modified
            in the pass.
            Vectorization pipeline needs to run only once!

            :param sdfg: The SDFG on which the pipeline is currently being applied
        """
        for p in self.passes:
            p: Pass
            yield p
