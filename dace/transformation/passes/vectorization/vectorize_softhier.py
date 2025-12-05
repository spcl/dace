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
import os
class PartialDict(defaultdict):
    def __missing__(self, key):
        return "{" + key + "}"

skip_scalar = os.getenv('SOFTHIER_SKIP_SCALAR_FALLBACK', '0') == '1'

vgt_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(32) ".v v8, (%0)" :: "r"(va_addr));
         asm volatile("vfmv.v.f v0, %0" :: "f"(threshold));
        // v8 = va - vb  (comparison via sign test)
        asm volatile("vfsub.vv v8, v8, v0");
        // mask: v8 < 0  → true -> illegal op we make it into anither sub
        asm volatile("vfsub.vv v8, v8, v0");
        //asm volatile("vmslt.vx v0, v8, zero");
        // fill vector registers with true_val / false_val
         asm volatile("vfadd.vv v16, v8, v0");
        asm volatile("vfmv.v.f v16, %0" :: "f"(true_val));
        asm volatile("vfmv.v.f v24, %0" :: "f"(false_val));
        asm volatile("vse" XSTR(32) ".v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vgt_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)};
uint32_t core_id = flex_get_core_id();

if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;

    va_addr += vu_offset;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;

    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(32) ".v v8, (%0)" :: "r"(va_addr));
         asm volatile("vfmv.v.f v0, %0" :: "f"(threshold));
        // v8 = va - vb  (comparison via sign test)
        asm volatile("vfsub.vv v8, v8, v0");
        // mask: v8 < 0  → true -> illegal op we make it into anither sub
        asm volatile("vfsub.vv v8, v8, v0");
        //asm volatile("vmslt.vx v0, v8, zero");
        // fill vector registers with true_val / false_val
         asm volatile("vfadd.vv v16, v8, v0");
        asm volatile("vfmv.v.f v16, %0" :: "f"(true_val));
        asm volatile("vfmv.v.f v24, %0" :: "f"(false_val));
        asm volatile("vse" XSTR(32) ".v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''


vlt_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(32) ".v v8, (%0)" :: "r"(va_addr));
         asm volatile("vfmv.v.f v0, %0" :: "f"(threshold));
        // v8 = va - vb  (comparison via sign test)
        asm volatile("vfsub.vv v8, v8, v0");
        // mask: v8 < 0  → true -> illegal op we make it into anither sub
        asm volatile("vfsub.vv v8, v8, v0");
        //asm volatile("vmslt.vx v0, v8, zero");
        // fill vector registers with true_val / false_val
         asm volatile("vfadd.vv v16, v8, v0");
        asm volatile("vfmv.v.f v16, %0" :: "f"(true_val));
        asm volatile("vfmv.v.f v24, %0" :: "f"(false_val));
        asm volatile("vse" XSTR(32) ".v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vlt_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)};
uint32_t core_id = flex_get_core_id();

if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;

    va_addr += vu_offset;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;

    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(32) ".v v8, (%0)" :: "r"(va_addr));
         asm volatile("vfmv.v.f v0, %0" :: "f"(threshold));
        // v8 = va - vb  (comparison via sign test)
        asm volatile("vfsub.vv v8, v8, v0");
        // mask: v8 < 0  → true -> illegal op we make it into anither sub
        asm volatile("vfsub.vv v8, v8, v0");
        //asm volatile("vmslt.vx v0, v8, zero");
        // fill vector registers with true_val / false_val
         asm volatile("vfadd.vv v16, v8, v0");
        asm volatile("vfmv.v.f v16, %0" :: "f"(true_val));
        asm volatile("vfmv.v.f v24, %0" :: "f"(false_val));
        asm volatile("vse" XSTR(32) ".v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vgt_vs_scalar = f'''
// Uncommenting this 36% to 1.5%
float* va_ptr = (float*)va_addr;
float* vc_ptr = (float*)vc_addr;

uint32_t num_vector_cores = {os.getenv('SOFTHIER_CORES_PER_CLUSTER', 1)};
uint32_t core_id = flex_get_core_id();
uint32_t per_core_length = vector_width / num_vector_cores;
if (core_id < num_vector_cores){{
    #pragma GCC unroll 32
    for (uint32_t i = core_id * per_core_length; i < (core_id + 1) * per_core_length; i++) {{
        vc_ptr[i] = (va_ptr[i] > threshold) ? 1.0f : 0.0f;
    }}
}}
'''

vlt_vs_scalar = f'''
// Uncommenting this 36% to 1.5%
float* va_ptr = (float*)va_addr;
float* vc_ptr = (float*)vc_addr;

uint32_t num_vector_cores = {os.getenv('SOFTHIER_CORES_PER_CLUSTER', 1)};
uint32_t core_id = flex_get_core_id();
uint32_t per_core_length = vector_width / num_vector_cores;
if (core_id < num_vector_cores){{
    #pragma GCC unroll 32
    for (uint32_t i = core_id * per_core_length; i < (core_id + 1) * per_core_length; i++) {{
        vc_ptr[i] = (va_ptr[i] < threshold) ? 1.0f : 0.0f;
    }}
}}
'''

vadd_vv_single_core = f'''
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
'''

vadd_vv_multi_core = f'''
    uint32_t num_vector_cores = {os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)};
    uint32_t core_id = flex_get_core_id();
    if (core_id < num_vector_cores) {{
        uint32_t length_per_vu = (vector_width / num_vector_cores);
        uint32_t vu_offset = core_id * length_per_vu * 4;
        va_addr += vu_offset;
        vb_addr += vu_offset;
        vc_addr += vu_offset;
        uint32_t vlen = length_per_vu;
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
'''

vmul_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vsub_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfsub.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vadd_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t avl;
    while(vector_width > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vector_width -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vmul_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t avl;
    while(vector_width > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vector_width -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vdiv_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t avl;
    while(vector_width > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vector_width));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfdiv.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vector_width -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''


# ---------------------------
# Multi-core implementations
# ---------------------------

vmul_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vsub_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfsub.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vadd_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vmul_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vdiv_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = (vector_width / num_vector_cores);
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfdiv.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vdiv_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load va
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // divide
        asm volatile("vfdiv.vv v8, v8, v0");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''


vdiv_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load va
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // divide
        asm volatile("vfdiv.vv v8, v8, v0");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''



vmin_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load vb
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // min
        asm volatile("vfmin.vv v8, v8, v0");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vdiv_sv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));

        // load vector va into v0
        asm volatile("vle32.v v0, (%0)" :: "r"(va_addr));

        // broadcast scalar into v8
        asm volatile("vfmv.v.f v8, %0" :: "f"(scalar));

        // v8 = v8 / v0   -->   scalar / va[i]
        asm volatile("vfdiv.vv v8, v8, v0");

        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));

        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vdiv_sv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();

if (core_id < num_vector_cores) {{

    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;

    va_addr += vu_offset;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;

    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));

        // load vector va into v0
        asm volatile("vle32.v v0, (%0)" :: "r"(va_addr));

        // broadcast scalar into v8
        asm volatile("vfmv.v.f v8, %0" :: "f"(scalar));

        // v8 = v8 / v0   --> scalar / va[i]
        asm volatile("vfdiv.vv v8, v8, v0");

        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));

        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''



vmin_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load vb
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // min
        asm volatile("vfmin.vv v8, v8, v0");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vmax_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load vb
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // max
        asm volatile("vfmax.vv v8, v8, v0");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vmax_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfmax.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vsub_sv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // load vb
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        // broadcast scalar
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        // subtract scalar - vector
        asm volatile("vfsub.vv v8, v0, v8");
        // store result
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vsub_sv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));
        asm volatile("vfsub.vv v8, v0, v8");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''
vsub_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));

        // load vb (vector)
        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));

        // broadcast scalar into v0
        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));

        // vector - scalar: v8 = v8 - v0
        asm volatile("vfsub.vv v8, v8, v0");

        // store
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));

        vlen -= avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''
vsub_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    vb_addr += vu_offset;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;

    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(avl) : "r"(vlen));

        asm volatile("vle32.v v8, (%0)" :: "r"(vb_addr));

        asm volatile("vfmv.v.f v0, %0" :: "f"(scalar));

        asm volatile("vfsub.vv v8, v8, v0");

        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));

        vlen -= avl;
        vb_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vmin_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmin.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vmin_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmin.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vmax_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmax.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vmax_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    vc_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vle32.v v0, (%0)" :: "r"(vb_addr));
        asm volatile("vfmax.vv v8, v8, v0");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4*avl;
        vb_addr += 4*avl;
        vc_addr += 4*avl;
    }}
}}
'''

vexp_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile(".word %0"::"i"(0x32041857));  // vfexp.vv v16, v8
        asm volatile("vse32.v v16, (%0)" :: "r"(vb_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
    }}
}}
'''

vexp_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vb_addr += vu_offset;
    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while(vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile(".word %0"::"i"(0x32041857));  // vfexp.vv v16, v8
        asm volatile("vse32.v v16, (%0)" :: "r"(vb_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vb_addr += 4 * avl;
    }}
}}
'''

vsqrt_vv_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfsqrt.v v8, v8");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vsqrt_vv_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    va_addr += vu_offset;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle32.v v8, (%0)" :: "r"(va_addr));
        asm volatile("vfsqrt.v v8, v8");
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        va_addr += 4 * avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vfill_vs_single_core = f'''
if (flex_is_first_core()) {{
    uint32_t vlen = vector_width;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // broadcast constant
        asm volatile("vfmv.v.f v8, %0" :: "f"(value));
        // store
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vc_addr += 4 * avl;
    }}
}}
'''

vfill_vs_multi_core = f'''
uint32_t num_vector_cores = {os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)};
uint32_t core_id = flex_get_core_id();
if (core_id < num_vector_cores) {{
    uint32_t length_per_vu = vector_width / num_vector_cores;
    uint32_t vu_offset = core_id * length_per_vu * 4;
    vc_addr += vu_offset;

    uint32_t vlen = length_per_vu;
    uint32_t avl;
    while (vlen > 0) {{
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        // broadcast constant
        asm volatile("vfmv.v.f v8, %0" :: "f"(value));
        // store
        asm volatile("vse32.v v8, (%0)" :: "r"(vc_addr));
        vlen -= avl;
        vc_addr += 4 * avl;
    }}
}}
'''



class VectorizeSoftHier(ppl.Pipeline):
    _softhier_global_code = f'''
#ifndef _SOFTHIER_MACROS_DEFINED
#define _SOFTHIER_MACROS_DEFINED
#define STR(x) #x
#define XSTR(x) STR(x)
#endif _SOFTHIER_MACROS_DEFINED

#include <stdlib.h>

inline void _softhier_vfill_vs_(float value, uint32_t vc_addr, uint32_t vector_width)
{{
    {vfill_vs_single_core if int(os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)) == 1 else vfill_vs_multi_core }
}}

inline void _softhier_vfill_vv_(float value, uint32_t vc_addr, uint32_t vector_width)
{{
    float scalar = 0.0;
    {vfill_vs_single_core if int(os.getenv("SOFTHIER_NUM_VECTOR_UNITS", 1)) == 1 else vfill_vs_multi_core }
}}

inline void _softhier_vsqrt_vv_(uint32_t vc_addr, uint32_t va_addr, uint32_t vector_width)
{{
    {vsqrt_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vsqrt_vv_multi_core}
}}

inline void _softhier_vadd_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width)
{{
    {vadd_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vadd_vv_multi_core}
}}

/*vc = va * vb*/
inline void _softhier_vmul_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    {vmul_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmul_vv_multi_core}
}}

/*vc = va - vb*/
inline void _softhier_vsub_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    {vsub_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vsub_vv_multi_core}

}}


// Vector addition with scalar: vc = va + scalar
inline void _softhier_vadd_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vadd_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vadd_vs_multi_core}
}}


// Vector multiplication with scalar: vc = va * scalar
inline void _softhier_vmul_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vmul_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmul_vs_multi_core}
}}

// Vector division: vc = va / vb
inline void _softhier_vdiv_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    {vdiv_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vdiv_vv_multi_core}
}}


inline void _softhier_vgt_vs_(uint32_t va_addr, uint32_t vc_addr, uint32_t vector_width, float threshold, float true_val, float false_val) {{
    {((vgt_vs_scalar) if not skip_scalar else (vgt_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vgt_vs_multi_core))}
}}

inline void _softhier_vlt_vs_(uint32_t va_addr, uint32_t vc_addr, uint32_t vector_width, float threshold, float true_val, float false_val) {{
    {((vlt_vs_scalar) if not skip_scalar else (vlt_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vlt_vs_multi_core))}
}}


// Vector division with scalar: vc = va / scalar
inline void _softhier_vdiv_vs_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vdiv_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vdiv_vs_multi_core}
}}

inline void _softhier_vdiv_sv_(uint32_t va_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vdiv_sv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vdiv_sv_multi_core}
}}


inline void _softhier_vmin_vs_(uint32_t vb_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vmin_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmin_vs_multi_core}
}}

inline void _softhier_vmax_vs_(uint32_t vb_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vmax_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmax_vs_multi_core}
}}

inline void _softhier_vsub_sv_(uint32_t vb_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vsub_sv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vsub_sv_multi_core}
}}

inline void _softhier_vsub_vs_(uint32_t vb_addr, float scalar, uint32_t vc_addr, uint32_t vector_width) {{
    {vsub_vs_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vsub_vs_multi_core}
}}

inline void _softhier_vmin_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    {vmin_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmin_vv_multi_core}
}}

inline void _softhier_vmax_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width) {{
    {vmax_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmax_vv_multi_core}
}}

inline void _softhier_vgt_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float true_val, float false_val) {{
    // TODO
    {vmin_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmin_vv_multi_core}

}}

inline void _softhier_vlt_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vc_addr, uint32_t vector_width, float true_val, float false_val) {{
    // TODO
    {vmax_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vmax_vv_multi_core}

}}

// Vector exponent: vb = exp(va)
inline void _softhier_vexp_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vector_width) {{
    {vexp_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vexp_vv_multi_core}
}}

// Vector exponent: vb = log(va)
inline void _softhier_vlog_vv_(uint32_t va_addr, uint32_t vb_addr, uint32_t vector_width)
{{
    // TODO
    {vexp_vv_single_core if int(os.getenv('SOFTHIER_NUM_VECTOR_UNITS', 1)) == 1 else vexp_vv_multi_core}
}}

inline void _softhier_vgather_fp32_int32_(uint32_t va_addr, int32_t* idx, uint32_t vc_addr, uint32_t vector_width) {{
    float* va_ptr = (float*)va_addr;
    int32_t* vb_ptr = idx;
    float* vc_ptr = (float*)vc_addr;

    uint32_t num_vector_cores = {int(os.getenv("SOFTHIER_NUM_CORE_PER_CLUSTER", 1))};
    uint32_t core_id = flex_get_core_id();

    uint32_t per_core_length = vector_width / num_vector_cores;
    uint32_t start = core_id * per_core_length;
    uint32_t end = start + per_core_length;

    for (uint32_t i = start; i < end; i++) {{
        vc_ptr[i] = va_ptr[vb_ptr[i]];
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
            "/": f"_softhier_vdiv_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "+c": f"_softhier_vadd_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "*c": f"_softhier_vmul_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "c*": f"_softhier_vmul_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "/c": f"_softhier_vdiv_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "c/": f"_softhier_vdiv_sv_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "-c": f"_softhier_vsub_vs_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "c-": f"_softhier_vsub_sv_({{rhs1}}, {{constant}}, {{lhs}}, {vector_width});",
            "minc": f"_softhier_vmin_vs_({{rhs1}}, {{lhs}}, {{constant}}, {vector_width});",
            "cmin": f"_softhier_vmin_vs_({{rhs1}}, {{lhs}}, {{constant}}, {vector_width});",
            "maxc": f"_softhier_vmax_vs_({{rhs1}}, {{lhs}}, {{constant}}, {vector_width});",
            "cmax": f"_softhier_vmax_vs_({{rhs1}}, {{lhs}}, {{constant}}, {vector_width});",
            ">c": f"_softhier_vgt_vs_({{rhs1}}, {{lhs}}, {vector_width}, {{constant}}, 1.0, 0.0);",
            "c>": f"_softhier_vlt_vs_({{rhs1}}, {{lhs}}, {vector_width}, {{constant}}, 1.0, 0.0);",
            ">": f"_softhier_vgt_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width}, 1.0, 0.0);",
            "<c": f"_softhier_vlt_vs_({{rhs1}}, {{lhs}}, {vector_width}, {{constant}}, 1.0, 0.0);",
            "c<": f"_softhier_vgt_vs_({{rhs1}}, {{lhs}}, {vector_width}, {{constant}}, 1.0, 0.0);",
            "<": f"_softhier_vlt_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width}, 1.0, 0.0);",
            "exp": f"_softhier_vexp_vv_({{rhs1}}, {{lhs}}, {vector_width});",
            "log": f"_softhier_vlog_vv_({{rhs1}}, {{lhs}}, {vector_width});",
            "min": f"_softhier_vmin_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "max": f"_softhier_vmax_vv_({{rhs1}}, {{rhs2}}, {{lhs}}, {vector_width});",
            "sqrt": f"_softhier_vsqrt_vv_({{rhs1}}, {{lhs}}, {vector_width});",
            "fillc": f"_softhier_vfill_vs_({{constant}}, {{lhs}}, {vector_width});",
            "cfill": f"_softhier_vfill_vs_({{constant}}, {{lhs}}, {vector_width});",
            "fill": f"_softhier_vfill_vv_({{rhs1}}, {{lhs}}, {vector_width});",
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
                LowerInterstateConditionalAssignmentsToTasklets(),
                RemoveEmptyStates(),
                RemoveFPTypeCasts(),
                RemoveIntTypeCasts(),
                PowerOperatorExpansion(),
                SplitTasklets(),
                CleanDataToScalarSliceToTaskletPattern(),
                RemoveRedundantAssignments(permissive=True),
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
