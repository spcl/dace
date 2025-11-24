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
                 fail_on_unvectorizable: bool = False):
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
