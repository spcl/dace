# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Iterator, List, Optional
import dace
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.vectorization.vectorize import Vectorize
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads


class VectorizeGPU(ppl.Pipeline):
    _gpu_global_code = """

#define _dace_vectorize

template<typename T>
__host__ __device__ __forceinline__ void vector_mult(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] * b[i];
    }}
}}

template<typename T>
__host__ __device__ __forceinline__ void vector_mult_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
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
__host__ __device__ __forceinline__ void vector_add(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] + b[i];
    }}
}}

template<typename T>
__host__ __device__ __forceinline__ void vector_add_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
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
__host__ __device__ __forceinline__ void vector_div(T * __restrict__ c, const T * __restrict__ a, const T * __restrict__ b) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        c[i] = a[i] / b[i];
    }}
}}

template<typename T>
__host__ __device__ __forceinline__ void vector_div_w_scalar(T * __restrict__ b, const T * __restrict__ a, const T constant) {{
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
__host__ __device__ __forceinline__ void vector_copy(T * __restrict__ dst, const T * __restrict__ src) {{
    #pragma omp unroll
    for (int i = 0; i < {vector_width}; i++) {{
        dst[i] = src[i];
    }}
}}
"""

    def __init__(self,
                 vector_width: str,
                 try_to_demote_symbols_in_nsdfgs: bool = False,
                 fuse_overlapping_loads: bool = False,
                 apply_on_maps: Optional[List[str]] = None,
                 insert_copies: bool = False):
        passes = [
            EliminateBranches(),
            RemoveFPTypeCasts(),
            RemoveIntTypeCasts(),
            PowerOperatorExpansion(),
            SplitTasklets(),
            CleanDataToScalarSliceToTaskletPattern(),
            InlineSDFGs(),
            Vectorize(
                templates={
                    "*": "vector_mult({lhs}, {rhs1}, {rhs2});",
                    "+": "vector_add({lhs}, {rhs1}, {rhs2});",
                    "=": "vector_copy({lhs}, {rhs1});",
                    "c+": "vector_add({lhs}, {rhs1}, {constant});",
                    "c*": "vector_mult({lhs}, {rhs1}, {constant});",
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.Register,
                vector_output_storage=dace.dtypes.StorageType.Register,
                global_code=VectorizeGPU._gpu_global_code.format(vector_width=vector_width),
                global_code_location="frame",
                vector_op_numeric_type=dace.float64,
                try_to_demote_symbols_in_nsdfgs=try_to_demote_symbols_in_nsdfgs,
                apply_on_maps=apply_on_maps,
                insert_copies=insert_copies,
            )
        ]
        if fuse_overlapping_loads:
            passes.append(FuseOverlappingLoads())
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
