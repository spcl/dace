# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.tasklet_preprocessing_passes import IntegerPowerToMult, RemoveFPTypeCasts, RemoveIntTypeCasts
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.explicit_vectorization import ExplicitVectorization
from dace.transformation.passes.fuse_branches_pass import FuseBranchesPass


class ExplicitVectorizationPipelineSoftHier(ppl.Pipeline):
    _softhier_global_code = """
#ifndef _SOFTHIER_MACROS_DEFINED
#define _SOFTHIER_MACROS_DEFINED
#define STR(x) #x
#define XSTR(x) STR(x)
#endif _SOFTHIER_MACROS_DEFINED

inline void _softhier_vi_vadd_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    uint32_t vlen = {vector_width};
    uint32_t avl;
    while(vlen > 0){{
        asm volatile("vsetvli %0, %1, e" XSTR(16) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(16) ".v v8,  (%0)" ::"r"(va_addr));
        asm volatile("vle" XSTR(16) ".v v0,  (%0)" ::"r"(vb_addr));
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse" XSTR(16) ".v v8,  (%0)" ::"r"(vc_addr));
        vlen -= avl;
        va_addr += 2*avl;
        vb_addr += 2*avl;
        vc_addr += 2*avl;
    }}
}}

/*vc = va * vb*/
inline void _softhier_vi_vmul_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    uint32_t vlen = {vector_width};
    uint32_t avl;
    while(vlen > 0){{
        asm volatile("vsetvli %0, %1, e" XSTR(16) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(16) ".v v8,  (%0)" ::"r"(va_addr));
        asm volatile("vle" XSTR(16) ".v v0,  (%0)" ::"r"(vb_addr));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse" XSTR(16) ".v v8,  (%0)" ::"r"(vc_addr));
        vlen -= avl;
        va_addr += 2*avl;
        vb_addr += 2*avl;
        vc_addr += 2*avl;
    }}
}}

/*vc = va - vb*/
inline void _softhier_vi_vsub_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    uint32_t vlen = {vector_width};
    uint32_t avl;
    while(vlen > 0){{
        asm volatile("vsetvli %0, %1, e" XSTR(16) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(16) ".v v8,  (%0)" ::"r"(va_addr));
        asm volatile("vle" XSTR(16) ".v v0,  (%0)" ::"r"(vb_addr));
        asm volatile("vfsub.vv v8, v8, v0");
        asm volatile("vse" XSTR(16) ".v v8,  (%0)" ::"r"(vc_addr));
        vlen -= avl;
        va_addr += 2*avl;
        vb_addr += 2*avl;
        vc_addr += 2*avl;
    }}
}}


/*vc = va / vb*/
inline void _softhier_vi_vdiv_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    uint32_t vlen = {vector_width};
    uint32_t avl;
    while(vlen > 0){{
        asm volatile("vsetvli %0, %1, e" XSTR(16) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(16) ".v v8,  (%0)" ::"r"(va_addr));
        asm volatile("vle" XSTR(16) ".v v0,  (%0)" ::"r"(vb_addr));
        asm volatile("vfdiv.vv v8, v8, v0");
        asm volatile("vse" XSTR(16) ".v v8,  (%0)" ::"r"(vc_addr));
        vlen -= avl;
        va_addr += 2*avl;
        vb_addr += 2*avl;
        vc_addr += 2*avl;
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
                    "+": "_softhier_vi_vadd_((uint32_t){lhs}, (uint32_t){rhs1}, (uint32_t){rhs2});",
                    "=": "vector_copy({lhs}, {rhs1});"
                },
                vector_width=vector_width,
                vector_input_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                vector_output_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                global_code=ExplicitVectorizationPipelineSoftHier._softhier_global_code.format(
                    vector_width=vector_width),
                global_code_location="soft_hier",
                vector_op_numeric_type=dace.float16,
            )
        ]
        super().__init__(passes)
