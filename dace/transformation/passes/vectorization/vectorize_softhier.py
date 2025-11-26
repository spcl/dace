# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
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


class VectorizeSoftHier(ppl.Pipeline):
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
    //flex_intra_cluster_sync();
    //if (flex_is_first_core()) {{
        uint32_t vlen = {vector_width};
        uint32_t avl;
        while(vlen > 0){{
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vadd.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    //}}
    //flex_intra_cluster_sync();
}}

/*vc = va * vb*/
inline void _softhier_vi_vmul_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    //flex_intra_cluster_sync();
    //if (flex_is_first_core()) {{
        uint32_t vlen = {vector_width};
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
    //}}
    //flex_intra_cluster_sync();
}}

/*vc = va - vb*/
inline void _softhier_vi_vsub_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    //flex_intra_cluster_sync();
    //if (flex_is_first_core()) {{
        uint32_t vlen = {vector_width};
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
    //}}
    //flex_intra_cluster_sync();
}}


/*vc = va / vb*/
inline void _softhier_vi_vdiv_(
    uint32_t va_addr,
    uint32_t vb_addr,
    uint32_t vc_addr)
{{
    //flex_intra_cluster_sync();
    //if (flex_is_first_core()) {{
        uint32_t vlen = {vector_width};
        uint32_t avl;
        while(vlen > 0){{
            asm volatile("vsetvli %0, %1, e" XSTR(32) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
            asm volatile("vle" XSTR(32) ".v v8,  (%0)" ::"r"(va_addr));
            asm volatile("vle" XSTR(32) ".v v0,  (%0)" ::"r"(vb_addr));
            asm volatile("vfdiv.vv v8, v8, v0");
            asm volatile("vse" XSTR(32) ".v v8,  (%0)" ::"r"(vc_addr));
            vlen -= avl;
            va_addr += 4*avl;
            vb_addr += 4*avl;
            vc_addr += 4*avl;
        }}
    //}}
    //flex_intra_cluster_sync();
}}
"""

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
                 no_copy_out: bool = True):
        vectorizer = Vectorize(templates={
            "+": "_softhier_vi_vadd_({rhs1}, {rhs2}, {lhs});",
        },
                               vector_width=vector_width,
                               vector_input_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                               vector_output_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                               global_code=VectorizeSoftHier._softhier_global_code.format(vector_width=vector_width),
                               global_code_location="soft_hier",
                               vector_op_numeric_type=dace.float32,
                               try_to_demote_symbols_in_nsdfgs=try_to_demote_symbols_in_nsdfgs,
                               apply_on_maps=apply_on_maps,
                               insert_copies=insert_copies,
                               fail_on_unvectorizable=fail_on_unvectorizable,
                               eliminate_trivial_vector_map=eliminate_trivial_vector_map,
                               no_copy_out=no_copy_out,
                               tasklet_prefix="_softhier_")
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
