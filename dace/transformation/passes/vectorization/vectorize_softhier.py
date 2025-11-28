# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import dace
from typing import Iterator, List, Optional
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveFPTypeCasts, RemoveIntTypeCasts, RemoveMathCall
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import LowerInterstateConditionalAssignmentsToTasklets
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.vectorization.vectorize import Vectorize
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
from dace.transformation.passes.vectorization.remove_reduntant_assignments import RemoveRedundantAssignments
from dace.transformation.passes.vectorization.remove_vector_maps import RemoveVectorMaps

class PartialDict(defaultdict):
    def __missing__(self, key):
        return "{" + key + "}"
    
class VectorizeSoftHier(ppl.Pipeline):
    _softhier_global_code = f'''
#ifndef _SOFTHIER_MACROS_DEFINED
#define _SOFTHIER_MACROS_DEFINED
#define STR(x) #x
#define XSTR(x) STR(x)
#endif _SOFTHIER_MACROS_DEFINED

inline void _softhier_vadd_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width)
{{
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {{
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){{
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfadd.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
    //flex_intra_cluster_sync();
}}

/*vc = va * vb*/
inline void _softhier_vmul_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {{
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){{
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfmul.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
    //flex_intra_cluster_sync();
}}

/*vc = va - vb*/
inline void _softhier_vsub_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    //flex_intra_cluster_sync();
    if (flex_is_first_core()) {{
        uint32_t vlen = vector_width;
        uint32_t avl;
        while(vlen > 0){{
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfsub.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
    //flex_intra_cluster_sync();
}}


// Vector addition with scalar: vc = va + scalar
inline void _softhier_vadd_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    if (flex_is_first_core()) {{
        uint32_t avl;
        while(vector_width > 0) {{
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma": "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfadd.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
}}


// Vector multiplication with scalar: vc = va * scalar
inline void _softhier_vmul_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    if (flex_is_first_core()) {{
        uint32_t avl;
        while(vector_width > 0) {{
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfmul.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
}}

// Vector division: vc = va / vb
inline void _softhier_vdiv_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    if (flex_is_first_core()) {{
        uint32_t avl;
        while(vector_width > 0) {{
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vle32.v v0, (%0)" ::"r"(vb_addr));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
}}

// Vector division with scalar: vc = va / scalar
inline void _softhier_vdiv_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    if (flex_is_first_core()) {{
        uint32_t avl;
        while(vector_width > 0) {{
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
            asm volatile("vle32.v v8, (%0)" ::"r"(va_addr));
            asm volatile("vfmv.v.f v0, %0" ::"f"(scalar));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse32.v v8, (%0)" ::"r"(vc_addr));
            vector_width -= avl;
            va_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    }}
}}

static const float MAXLOGF = 88.72283905206835f;
static const float MINLOGF = -88.0f;
static const float C1F = 0.693359375f;
static const float C2F = -2.12194440e-4f;
static const float PX1expf = 1.9875691500E-4f;
static const float PX2expf = 1.3981999507E-3f;
static const float PX3expf = 8.3334519073E-3f;
static const float PX4expf = 4.1665795894E-2f;
static const float PX5expf = 1.6666665459E-1f;
static const float PX6expf = 5.0000001201E-1f;
static const float LOG2EF = 1.44269504088896341f;

// Vector exponent: vb = exp(va)
static inline void _softhier_vexp_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vector_width)
{{
    if (flex_is_first_core()) {{
        uint32_t vlen = vector_width;
        uint32_t avl;
        
        /* Constants */
        const float const_log2ef = LOG2EF;
        const float const_05 = 0.5f;
        const float const_c1f = C1F;
        const float const_c2f = C2F;
        const float const_px1 = PX1expf;
        const float const_px2 = PX2expf;
        const float const_px3 = PX3expf;
        const float const_px4 = PX4expf;
        const float const_px5 = PX5expf;
        const float const_px6 = PX6expf;
        const float const_1 = 1.0f;
        const float const_maxlogf = MAXLOGF;
        const float const_minlogf = MINLOGF;
        const float const_inf = INFINITY;
        const float const_zero = 0.0f;
        const float const_nan = NAN;
        const uint32_t const_bias = 0x7f;
        const uint32_t const_shift = 23;
        
        while(vlen > 0) {{
            /* Set vector length */
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            
            /* Load input vector into v0 */
            asm volatile("vle" XSTR(32) ".v v0, (%0)" :: "r"(va_addr));
            
            /* v1 = LOG2EF * v0 */
            /* v1 = v1 + 0.5 */
            /* v1 = floor(v1) */
            asm volatile("vfmul.vf v1, v0, %0" :: "f"(const_log2ef));
            asm volatile("vfadd.vf v1, v1, %0" :: "f"(const_05));
            asm volatile("vfcvt.x.f.v v1, v1");
            asm volatile("vfcvt.f.x.v v1, v1");  // z in v1
            
            /* v2 = v1 (z) * C1F */
            /* v3 -= v2 */
            asm volatile("vfmul.vf v2, v1, %0" :: "f"(const_c1f));
            asm volatile("vfsub.vv v3, v3, v2");
            
            /* v2 = v1 (z) * C2F */
            /* v3 (x) -= v2 */
            asm volatile("vfmul.vf v2, v1, %0" :: "f"(const_c2f));
            asm volatile("vfsub.vv v3, v3, v2");
            
            /* v4 (n) = (int)z */
            asm volatile("vfcvt.x.f.v v4, v1");
            
            /* x2 (v4) = x * x (v3) */
            asm volatile("vfmul.vv v4, v3, v3");
            
            /* Polynomial in v5 (z) = x*PX1 + PX2 */
            asm volatile("vfmul.vf v5, v0, %0" :: "f"(const_px1));
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_px2));
            asm volatile("vfmul.vv v5, v5, v0");  // z *= x
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_px3));
            asm volatile("vfmul.vv v5, v5, v0");  // z *= x
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_px4));
            asm volatile("vfmul.vv v5, v5, v0");  // z *= x
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_px5));
            asm volatile("vfmul.vv v5, v5, v0");  // z *= x
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_px6));
            asm volatile("vfmul.vv v5, v5, v4");  // z *= x2
            
            /* v5 += x + 1.0 */
            asm volatile("vfadd.vf v5, v5, %0" :: "f"(const_1));  // z += 1.0
            asm volatile("vfadd.vv v5, v5, v3");                   // z += x
            
            /* Build 2^n in v4: (n (v4) + 0x7f) << 23 */
            asm volatile("vadd.vx v4, v4, %0" :: "r"(const_bias));
            asm volatile("vsll.vx v4, v4, %0" :: "r"(const_shift));

            asm volatile("vfcvt.f.x.v v4, v4");  // cast n (v4) to float

            /* z *= v4 */
            asm volatile("vfmul.vv v5, v5, v4");
            
            /* Store result from v5 */
            asm volatile("vse" XSTR(32) ".v v5, (%0)" :: "r"(vb_addr));
            
            /* Update pointers and length */
            vlen -= avl;
            va_addr += 4 * avl;
            vb_addr += 4 * avl;
        }}
    }}
}}
'''

    def __init__(self,
                 vector_width: str,
                 try_to_demote_symbols_in_nsdfgs: bool = False,
                 fuse_overlapping_loads: bool = False,
                 apply_on_maps: Optional[List[dace.nodes.MapEntry]] = None,
                 insert_copies: bool = False,
                 only_apply_vectorization_pass: bool = False,
                 no_inline: bool = False,
                 fail_on_unvectorizable: bool = False,
                 eliminate_trivial_vector_map: bool = True,
                 no_copy_out: bool = True,
                 dtype: dace.typeclass = dace.float32):
        vectorizer = Vectorize(templates={
            "+": f"_softhier_vadd_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "*": f"_softhier_vmul_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "-": f"_softhier_vsub_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "+c": f"_softhier_vadd_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "*c": f"_softhier_vmul_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "/c": f"_softhier_vdiv_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "exp": f"_softhier_vexp_vv_({{rhs1}}, {{lhs}}, {vector_width});",
        },
                               vector_width=vector_width,
                               vector_input_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                               vector_output_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                               global_code=VectorizeSoftHier._softhier_global_code,
                               global_code_location="soft_hier",
                               vector_op_numeric_type=dtype,
                               try_to_demote_symbols_in_nsdfgs=try_to_demote_symbols_in_nsdfgs,
                               apply_on_maps=apply_on_maps,
                               insert_copies=insert_copies,
                               fail_on_unvectorizable=fail_on_unvectorizable,
                               eliminate_trivial_vector_map=eliminate_trivial_vector_map,
                               no_copy_out=no_copy_out,
                               tasklet_prefix="_softhier_",
                               )
        if not only_apply_vectorization_pass:
            passes = [
                EliminateBranches(),
                RemoveRedundantAssignments(),
                LowerInterstateConditionalAssignmentsToTasklets(),
                RemoveEmptyStates(),
                RemoveFPTypeCasts(),
                RemoveIntTypeCasts(),
                PowerOperatorExpansion(),
                SplitTasklets(),
                CleanDataToScalarSliceToTaskletPattern(),
            ]
            if not no_inline:
                passes.append(InlineSDFGs())
            passes.append(vectorizer)
        else:
            passes = [RemoveMathCall(), vectorizer]
        if fuse_overlapping_loads:
            passes.append(FuseOverlappingLoads())
        if eliminate_trivial_vector_map:
            passes.append(RemoveVectorMaps())
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
