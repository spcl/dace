# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Iterator, List, Optional, Set
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.vectorization.remove_reduntant_assignments import RemoveRedundantAssignments
from dace.transformation.passes.clean_data_to_scalar_slice_to_tasklet_pattern import CleanDataToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveFPTypeCasts, RemoveIntTypeCasts, RemoveMathCall
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import LowerInterstateConditionalAssignmentsToTasklets
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.vectorization.vectorize import Vectorize
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
from dace.transformation.passes.vectorization.remove_vector_maps import RemoveVectorMaps
from dace.transformation.passes.vectorization.same_write_set_if_else_to_merge_cfg import SameWriteSetIfElseToMergeCFG
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


class VectorizeCPU(ppl.Pipeline):
    _cpu_global_code = """
#include "dace/cpu_vectorizable_math.h"
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
                 user_skip_nsdfg_arrays: Optional[Set[str]] = None,
                 use_fp_factor: bool = True,
                 branch_normalization: bool = False):
        if use_fp_factor and branch_normalization:
            raise ValueError("VectorizeCPU: use_fp_factor and branch_normalization are mutually exclusive; "
                             "choose one branch-lowering strategy")
        vectorizer = Vectorize(
            templates={
                "*": "vector_mult<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "-": "vector_sub<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "/": "vector_div<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "=": "vector_copy<{dtype}, {vector_width}>({lhs}, {rhs1});",
                "=c": "vector_copy_w_scalar<{dtype}, {vector_width}>({lhs}, {constant});",
                "log": "vector_log<{dtype}, {vector_width}>({lhs}, {rhs1});",
                "exp": "vector_exp<{dtype}, {vector_width}>({lhs}, {rhs1});",
                "min": "vector_min<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "max": "vector_max<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                ">": "vector_gt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "<": "vector_lt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                ">=": "vector_ge<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "<=": "vector_le<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "==": "vector_eq<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "!=": "vector_ne<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
                "merge": "vector_select<{dtype}, {vector_width}>({lhs}, {cond}, {then_arm}, {else_arm});",
                # scalar variants type 1
                "*c": "vector_mult_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "+c": "vector_add_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "-c": "vector_sub_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "/c": "vector_div_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "minc": "vector_min_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "maxc": "vector_max_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                ">c": "vector_gt_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "<c": "vector_lt_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                ">=c": "vector_ge_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "<=c": "vector_le_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "==c": "vector_eq_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                "!=c": "vector_ne_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
                # scalar variants type 2 for non-commutative ops
                "c-": "vector_sub_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
                "c/": "vector_div_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
                "c>": "vector_gt_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
                "c<": "vector_lt_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
                "c>=": "vector_ge_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
                "c<=": "vector_le_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            },
            vector_width=vector_width,
            vector_input_storage=dace.dtypes.StorageType.Register,
            vector_output_storage=dace.dtypes.StorageType.Register,
            global_code=VectorizeCPU._cpu_global_code.format(vector_width=vector_width),
            global_code_location="frame",
            vector_op_numeric_type=dace.float64,
            try_to_demote_symbols_in_nsdfgs=try_to_demote_symbols_in_nsdfgs,
            apply_on_maps=apply_on_maps,
            insert_copies=insert_copies,
            fail_on_unvectorizable=fail_on_unvectorizable,
            eliminate_trivial_vector_map=eliminate_trivial_vector_map,
            user_skip_nsdfg_arrays=user_skip_nsdfg_arrays)
        if not only_apply_vectorization_pass:
            # Pick the branch-lowering front of the pipeline. ``use_fp_factor``
            # keeps today's behaviour, ``EliminateBranches`` collapses if/else
            # to the FP-factor form ``a = c*x + (1-c)*y``. ``branch_normalization``
            # opts into the new M3 pipeline, ``SameWriteSetIfElseToMergeCFG``
            # rewrites same-write-set arms to a 3-CFG merge form, then
            # ``BranchNormalization`` flattens the rest into merge tasklets.
            # In merge mode, ``LowerInterstateConditionalAssignmentsToTasklets``
            # runs before the M3 passes so the condition variable lives in a
            # tasklet plus a transient array (the vectorizer can then drive
            # it per-lane through the merge tasklet's ``_c`` connector).
            if branch_normalization:
                passes = [
                    LowerInterstateConditionalAssignmentsToTasklets(),
                    SameWriteSetIfElseToMergeCFG(),
                    BranchNormalization(),
                    RemoveRedundantAssignments(),
                ]
            else:
                passes = [
                    EliminateBranches(),
                    RemoveRedundantAssignments(),
                    LowerInterstateConditionalAssignmentsToTasklets(),
                ]
            passes += [
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
        self._applied_before = False
        super().__init__(passes)

    def iterate_over_passes(self, sdfg: dace.SDFG) -> Iterator[Pass]:
        """
            Iterates over passes in the pipeline, potentially multiple times based on which elements were modified
            in the pass.
            Vectorization pipeline needs to run only once!

            :param sdfg: The SDFG on which the pipeline is currently being applied
        """
        if self._applied_before is False:
            CleanDataToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
            self._applied_before = True
        for p in self.passes:
            p: Pass
            yield p


"""
N = dace.symbol("N", dtype=dace.int64)
@dace.program
def if_add(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i,j in dace.map[0:N, 0:N]:
        if A[i, j] > 0:
            C[i, j] = C[i, j] + B[i, j]
        else:
            C[i, j] = C[i, j] - B[i, j]

N = dace.symbol("N", dtype=dace.int64)
@dace.program
def if_add(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i,j in dace.map[0:N, 0:N]:
        if A[i, j] > 0:
            B[i, j] = B[i, j] - 1.0
        else:
            B[i, j] = B[i, j] + 1.0

# How to tile this, the "if-else" requires a state change therefore the
# you need to handle the nested SDFGs, in the end for mask the code needs to look like this:

# First you need to flatten if branch (I have a function for this)
# you can use dace/sdfg/construction_utils.py -> move_branch_cfg_up_discard_conditions
# It moves the branch condition up (but only keeps one body)
# So you can do this by duplicating the CFG, pasting it behind the original (add_state_after)
# then run this function separately then you have someting like
@dace.program
def if_add(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i,j in dace.map[0:N, 0:N]:
        if A[i, j] > 0:
            B[i, j] = B[i, j] - 1.0
        if A[i, j] <= 0:
            B[i, j] = B[i, j] + 1.0


# You can assume there is only 1 nested SDFG node within the map for the analysis
# Like, either 1 nsdfg node and no other nodes, or only non-nsdfg nodes
# You can use: dace/sdfg/transformation/passes/vectorization/vectorization_utils:: map_consists_of_single_nsdfg_or_no_nsdfg
# You can copy paste the function for now, after the pass is working we need to slowly PR these functions, we can do it together

# Then in the end it should look like this where you have 3 lib nodes to generate a mask and for the operations of each branch
N = dace.symbol("N", dtype=dace.int64)
T = 8
@dace.program
def if_add(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i,j in dace.map[0:N//T, 0:N//T]:
        for ii, jj in dace.map[0:T, 0:T]: # In the transformation you need to tile the map outside (and then keep the symbols known while you process the nested SDFG)
            mask[0:T, 0:T] = gen_mask(A[i*T:(i+1)*T, j*T:(j+1)*T], lambda: A[i] > 0)
            apply(mask, B[i*T:(i+1)*T, j*T:(j+1)*T], lambda b, m: b-1.0 if m else b+1.0)
            apply(~mask, B[i*T:(i+1)*T, j*T:(j+1)*T], lambda b, m: b-1.0 if m else b+1.0)
"""
