# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU vectorization pipeline: branch lowering, map preparation, then ``Vectorize``."""
import dace
from typing import Iterator, List, Optional, Set
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.clean_tasklet_to_scalar_slice_to_access_node_pattern import CleanTaskletToScalarSliceToAccessNodePattern
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import CleanAccessNodeToScalarSliceToTaskletPattern
from dace.transformation.passes.vectorization.remove_reduntant_assignments import RemoveRedundantAssignments
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveMathCall, RewriteModuloToPyMod
from dace.transformation.passes import InlineSDFGs
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import LowerInterstateConditionalAssignmentsToTasklets
from dace.transformation.passes.vectorization.remove_empty_states import RemoveEmptyStates
from dace.transformation.passes.vectorization.vectorize import Vectorize
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.remove_vector_maps import RemoveVectorMaps
from dace.transformation.passes.vectorization.fuse_overlapping_loads import FuseOverlappingLoads
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalizationPipeline
from dace.transformation.passes.vectorization.detect_gather import DetectGather
from dace.transformation.passes.vectorization.detect_scatter import DetectScatter
from dace.transformation.passes.vectorization.detect_strided_load import DetectStridedLoad
from dace.transformation.passes.vectorization.detect_strided_store import DetectStridedStore
from dace.transformation.passes.vectorization.detect_multi_dim_strided_load import DetectMultiDimStridedLoad
from dace.transformation.passes.vectorization.detect_multi_dim_strided_store import DetectMultiDimStridedStore
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
from dace.transformation.passes.vectorization.split_map_for_vector_remainder import SplitMapForVectorRemainder
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.utils.iteration import assert_no_lane_memlet_reads


class _EarlyPureWCRReductionLift(LiftMapReductionToReduce):
    """Distinct pipeline type for the early ``pure_wcr_only`` reduction lift.

    Pipeline forbids two passes of the same type. This subclass lets the early pure-WCR lift (before
    ``_WCRToAugAssignPass`` strips the WCR) coexist with the full RMW-shape
    :class:`LiftMapReductionToReduce` that runs later, after ``InlineSDFGs`` finalises the opaque body.
    """

    def __init__(self):
        super().__init__(pure_wcr_only=True)


class _WCRToAugAssignPass(ppl.Pass):
    """Run :class:`WCRToAugAssign` to a fixed point.

    Strips every WCR that survived into the body (in-place ``a[i] = a[i] + b[i]`` canonicalised /
    ``LoopToMap``-lowered to a ``... -(+=)-> a`` WCR, incl. AN->AN copy shape). Vectorizer assumes NO
    inner WCR -- a stray one is silently mis-vectorised (reduction dropped, computes ``a = b``). Runs
    after ``LoopToMap`` so WCRs the parallelisation minted are converted too. Mirrors multi-dim
    ``_RunWCRToAugAssign``.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _pipeline_results):
        from dace.transformation.dataflow import WCRToAugAssign
        from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                                    no_wcr_in_map_body)
        applied = sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
        # Post-condition / vectorize pre-condition: no WCR inside any map body (in-place vectorizer
        # doesn't resolve loop-carried reductions; boundary MapExit -> AN edge lives outside the body).
        assert_invariant(no_wcr_in_map_body(sdfg), "VectorizeCPU", "no WCR inside the map body before vectorizing")
        return applied


class _LoopToMapPass(ppl.Pass):
    """Run :class:`LoopToMap` to a fixed point.

    Runs after branch normalization (removes the ``ConditionalBlock``s blocking conversion) so a
    data-parallel ``for i in range(...)`` body becomes a Map the vectorizer can stride. No-op for
    ``dace.map`` kernels (no ``LoopRegion``s); non-provably-parallel loops left alone by
    ``LoopToMap``'s safety analysis.
    """

    def __init__(self, permissive: bool = False):
        super().__init__()
        self._permissive = permissive

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _pipeline_results):
        from dace.transformation.interstate import LoopToMap
        return sdfg.apply_transformations_repeated(LoopToMap, permissive=self._permissive)


class _AssertNoBodyWCRPass(ppl.Pass):
    """Vectorizer-entry precondition: NO loose WCR in the region about to be vectorized. Read-only.

    Asserts BOTH checkers after the body is nested: ``no_wcr_in_map_body`` AND
    ``no_wcr_inside_nested_sdfgs`` (a self-contained WCR hiding inside the body NSDFG, invisible to the
    map-body checker). Only WCR allowed past here = a genuine reduction in lifted form (``Reduce``
    libnode from ``LiftMapReductionToReduce`` / boundary ``MapExit -> AN`` edge). Every self-contained
    / in-place RMW must already be an explicit aug-assign (``WCRToAugAssign``).
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _pipeline_results):
        from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                                    no_wcr_in_map_body,
                                                                                    no_wcr_inside_nested_sdfgs)
        assert_invariant(no_wcr_in_map_body(sdfg), "VectorizeCPU", "no loose WCR in the map body before vectorizing")
        assert_invariant(no_wcr_inside_nested_sdfgs(sdfg), "VectorizeCPU",
                         "no loose WCR inside the body NSDFG before vectorizing")
        return None


class _AssertNoLaneMemletReadsPass(ppl.Pass):
    """Run ``assert_no_lane_memlet_reads`` after the vectorizer. In pipeline only when
    ``remainder_strategy='masked'`` (loud-failure contract, Option B in the plan)."""

    def __init__(self, vector_width: int):
        super().__init__()
        self._vector_width = vector_width

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _pipeline_results):
        assert_no_lane_memlet_reads(sdfg, self._vector_width)


class VectorizeCPU(ppl.Pipeline):
    """Ordered pipeline lowering innermost maps to CPU SIMD; see :meth:`__init__` for knobs."""

    _CPU_GLOBAL_CODE = """
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
                 branch_normalization: bool = False,
                 lower_to_intrinsics: bool = False,
                 gather_intrinsic: bool = True,
                 scatter_intrinsic: bool = True,
                 collapse_laneid_index_loads: bool = False,
                 loop_to_map_permissive: bool = False,
                 force_autovec_ops: Optional[Set[str]] = None,
                 force_pscalar_ops: Optional[Set[str]] = None,
                 remainder_strategy: str = "scalar"):
        """Build the pipeline.

        :param vector_width: SIMD lane count.
        :param try_to_demote_symbols_in_nsdfgs: demote NSDFG symbols to scalars when safe.
        :param fuse_overlapping_loads: fuse an array read at overlapping subsets into one shared
            union window. ``insert_copies=True`` bakes it into the NSDFG staging copy (one union
            buffer vs one ``vector_copy`` per subset); ``insert_copies=False`` uses the standalone
            ``FuseOverlappingLoads`` pass on the post-vectorization between-maps load fan.
        :param apply_on_maps: restrict vectorization to these map entries.
        :param insert_copies: insert copy-in/out around NSDFG boundaries.
        :param only_apply_vectorization_pass: run only ``Vectorize`` (skip prep/cleanup).
        :param no_inline: skip ``InlineSDFGs``.
        :param fail_on_unvectorizable: raise instead of skipping a map that cannot be vectorized.
        :param eliminate_trivial_vector_map: append ``RemoveVectorMaps``.
        :param user_skip_nsdfg_arrays: NSDFG array names to exclude from copy-in/out.
        :param use_fp_factor: branch lowering via ``c*x + (1-c)*y`` (exclusive with ``branch_normalization``).
        :param branch_normalization: branch lowering via the M3 ``ITE`` normalization.
        :param lower_to_intrinsics: also collapse strided / multi-dim-strided per-lane fans to
            intrinsics (gather/scatter collapse always on — see ``gather_intrinsic`` / ``scatter_intrinsic``).
        :param gather_intrinsic: ``True`` (default) emits the ``gather`` intrinsic for the main loop;
            ``False`` keeps its per-lane scalar gather fan. Masked vector remainder always uses the
            intrinsic (per-lane scalar fan faults on inactive lanes).
        :param scatter_intrinsic: as ``gather_intrinsic`` but for scatter.
        :param collapse_laneid_index_loads: ``True`` collapses a recognised per-lane laneid index fan
            (W laneid symbols bound to a contiguous ``<idxarr>[0:W]`` slice) so the gather intrinsic
            reads the index array directly via an ``_idx`` connector; dead laneid symbols + their
            interstate-edge assignments removed. Default ``False`` keeps the per-lane laneid-symbol form.
        :param loop_to_map_permissive: run the internal ``LoopToMap`` permissive. A scatter
            ``for i in range(...): a[idx[i]] = ...`` loop's data-dependent write index reads as a
            possible conflict to the non-permissive safety analysis (refuses to parallelize);
            permissive accepts it (caller asserts indices conflict-free). Default ``False``.
        :param force_autovec_ops: ops to emit as ``vector_<op>_av`` (autovec hint).
        :param force_pscalar_ops: ops to emit as ``vector_<op>_pscalar`` (no autovec hint).
        :param remainder_strategy: ``"scalar"`` (scalar postamble), ``"masked"`` (iter-mask
            remainder) or ``"full_loop_mask"`` (R3, not yet wired).
        :raises ValueError: on a rejected knob combination (see body).
        :raises NotImplementedError: for ``remainder_strategy="full_loop_mask"`` (R3, queued).
        """
        if use_fp_factor and branch_normalization:
            raise ValueError("VectorizeCPU: use_fp_factor and branch_normalization are mutually exclusive; "
                             "choose one branch-lowering strategy")
        # ``remainder_strategy``: maps whose trip is not *provably* divisible by W. P2
        # (``SplitMapForVectorRemainder``) always runs + does the symbolic divisibility analysis:
        # if ``simplify(ub-lb+1) % W == 0`` provably true (e.g. ``0 : 8*N``) it does NOT split (main
        # path tiles cleanly); else it splits and the strategy picks the remainder shape:
        #   "scalar"          - R1 (default): main step-W map + step-1 ``ScheduleType.Sequential``
        #                       scalar postamble. No mask.
        #   "masked"          - R2: main step-W (no mask) + step-W remainder body with a P3
        #                       ``_iter_mask`` gating trailing OOB lanes.
        #   "full_loop_mask"  - R3 (TODO): no split; one step-W map, ``_iter_mask`` wired everywhere.
        _VALID_REMAINDER = {"scalar", "masked", "full_loop_mask"}
        if remainder_strategy not in _VALID_REMAINDER:
            raise ValueError(f"VectorizeCPU: remainder_strategy must be one of "
                             f"{sorted(_VALID_REMAINDER)}, got {remainder_strategy!r}")
        if remainder_strategy == "full_loop_mask":
            raise NotImplementedError("VectorizeCPU: remainder_strategy='full_loop_mask' is queued (R3); "
                                      "currently only 'scalar' and 'masked' are wired end-to-end.")
        # fp_factor + masked rejected (locked decision): masked path emits ITE tasklets / iter_mask
        # blends fp-factor can't combine cleanly (would need a per-iteration bool-to-float cast). Use
        # branch_normalization for masked, or remainder_strategy="scalar" for fp_factor.
        if use_fp_factor and remainder_strategy == "masked":
            raise ValueError("VectorizeCPU: use_fp_factor=True is incompatible with "
                             "remainder_strategy='masked'; choose branch_normalization=True or "
                             "remainder_strategy='scalar'")
        # ``force_autovec_ops`` / ``force_pscalar_ops`` (Option F overlay): per-op override of the
        # emitter's implementation choice. Keys = templates-dict op ids (``"+"``, ``"ITE"``, ``"+c"``,
        # ``"c-"``, ``"log"``, ``"=c"`` ...).
        #   force_pscalar_ops={"div"}  -> ``vector_div_pscalar`` (pure scalar loop, no autovec hint)
        #   force_autovec_ops={"exp"}  -> ``vector_exp_av`` (scalar loop + _dace_vectorize hint)
        # Default: unsuffixed names (best for the dispatcher-selected arch).
        force_autovec_ops = set(force_autovec_ops) if force_autovec_ops else set()
        force_pscalar_ops = set(force_pscalar_ops) if force_pscalar_ops else set()
        overlap = force_autovec_ops & force_pscalar_ops
        if overlap:
            raise ValueError(f"VectorizeCPU: force_autovec_ops and force_pscalar_ops "
                             f"overlap on {sorted(overlap)}; an op can only be in one set")
        templates = {
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
            "int_floor": "vector_int_floor<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "int_ceil": "vector_int_ceil<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            ">": "vector_gt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "<": "vector_lt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            ">=": "vector_ge<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "<=": "vector_le<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "==": "vector_eq<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "!=": "vector_ne<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "ITE": "vector_select<{dtype}, {vector_width}>({lhs}, {cond}, {then_arm}, {else_arm});",
            "ITE_masked":
            "vector_select_av_masked<{dtype}, {vector_width}>({lhs}, {cond}, {then_arm}, {else_arm}, {mask});",
            # scalar variants type 1
            "*c": "vector_mult_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "+c": "vector_add_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "-c": "vector_sub_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "/c": "vector_div_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "minc": "vector_min_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "maxc": "vector_max_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "int_floorc": "vector_int_floor_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "int_ceilc": "vector_int_ceil_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            ">c": "vector_gt_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "<c": "vector_lt_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            ">=c": "vector_ge_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "<=c": "vector_le_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "==c": "vector_eq_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            "!=c": "vector_ne_w_scalar<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant});",
            # scalar variants type 2 for non-commutative ops
            "c-": "vector_sub_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "c/": "vector_div_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "cint_floor": "vector_int_floor_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "cint_ceil": "vector_int_ceil_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "c>": "vector_gt_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "c<": "vector_lt_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "c>=": "vector_ge_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            "c<=": "vector_le_w_scalar_c<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1});",
            # Masked variants (Option F overlay): emitted when the tasklet has an ``_iter_mask`` input
            # connector wired from a P3 ``_iter_mask: bool[W]`` array. Bound to ``_av_masked`` from
            # ``cpu_vectorizable_math_common.h`` (uniform across arches via macros; arch-specific masked
            # specializations can be added later where a measured win).
            "*_masked": "vector_mult_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "+_masked": "vector_add_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "-_masked": "vector_sub_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "/_masked": "vector_div_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "=_masked": "vector_copy_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {mask});",
            "=c_masked": "vector_copy_w_scalar_av_masked<{dtype}, {vector_width}>({lhs}, {constant}, {mask});",
            "log_masked": "vector_log_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {mask});",
            "exp_masked": "vector_exp_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {mask});",
            "min_masked": "vector_min_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "max_masked": "vector_max_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
            "*c_masked": "vector_mult_w_scalar_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant}, {mask});",
            "+c_masked": "vector_add_w_scalar_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant}, {mask});",
            "-c_masked": "vector_sub_w_scalar_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant}, {mask});",
            "/c_masked": "vector_div_w_scalar_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {constant}, {mask});",
            "c-_masked": "vector_sub_w_scalar_c_av_masked<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1}, {mask});",
            "c/_masked": "vector_div_w_scalar_c_av_masked<{dtype}, {vector_width}>({lhs}, {constant}, {rhs1}, {mask});",
        }
        # Per-op variant overrides: rewrite ``vector_<name><...>`` → ``vector_<name>_<suffix><...>``.
        # Suffixed functions defined in cpu_vectorizable_math_common.h identically across arches.
        import re as _re
        _vec_name_pat = _re.compile(r"\bvector_(\w+)<")
        for op in force_pscalar_ops | force_autovec_ops:
            if op.endswith("_masked"):
                raise ValueError(f"VectorizeCPU: override key {op!r} references a masked-variant "
                                 f"entry; force_*_ops applies to the default-path emission only. "
                                 f"Use the base op identifier (without _masked).")
            if op not in templates:
                raise KeyError(f"VectorizeCPU: force_*_ops references unknown op {op!r}; "
                               f"valid op identifiers: {sorted(k for k in templates if not k.endswith('_masked'))}")
        for op in force_pscalar_ops:
            templates[op] = _vec_name_pat.sub(r"vector_\1_pscalar<", templates[op], count=1)
        for op in force_autovec_ops:
            templates[op] = _vec_name_pat.sub(r"vector_\1_av<", templates[op], count=1)
        vectorizer = Vectorize(templates=templates,
                               vector_width=vector_width,
                               vector_input_storage=dace.dtypes.StorageType.Register,
                               vector_output_storage=dace.dtypes.StorageType.Register,
                               global_code=VectorizeCPU._CPU_GLOBAL_CODE.format(vector_width=vector_width),
                               global_code_location="frame",
                               vector_op_numeric_type=dace.float64,
                               try_to_demote_symbols_in_nsdfgs=try_to_demote_symbols_in_nsdfgs,
                               apply_on_maps=apply_on_maps,
                               insert_copies=insert_copies,
                               fail_on_unvectorizable=fail_on_unvectorizable,
                               eliminate_trivial_vector_map=eliminate_trivial_vector_map,
                               user_skip_nsdfg_arrays=user_skip_nsdfg_arrays,
                               fuse_overlapping_loads=fuse_overlapping_loads)
        if not only_apply_vectorization_pass:
            # Branch-lowering front. ``use_fp_factor``: ``EliminateBranches`` collapses if/else to
            # FP-factor ``a = c*x + (1-c)*y``. ``branch_normalization`` (M3): ``SameWriteSetIfElseToITECFG``
            # rewrites same-write-set arms to a 3-CFG ITE form, then ``BranchNormalization`` flattens the
            # rest into ITE tasklets; ``LowerInterstateConditionalAssignmentsToTasklets`` runs first so
            # the condition lives in a tasklet + transient array (vectorizer drives it per-lane via the
            # ITE tasklet's ``_c`` connector).
            # Clean passes fold the frontend ``A -> A_slice(scalar) -> tasklet`` reads while
            # ``A -> A_slice`` is still a direct AccessNode edge: after ``LoopToMap`` the MapEntry sits
            # between ``A`` and its slice scalar, the pattern no longer matches, and the scalar would be
            # widened along the wrong (contiguous) dim for a non-contiguous access (C-layout ``bb[i, j]``,
            # ``i`` innermost).
            # RewriteModuloToPyMod first: normalise every ``%`` (tasklet bodies, loop/map ranges, branch
            # conditions, memlet subsets, interstate edges) to ``py_mod`` (C's ``%`` miscompiles negatives).
            if branch_normalization:
                passes = [
                    RewriteModuloToPyMod(),
                    CleanAccessNodeToScalarSliceToTaskletPattern(),
                    CleanTaskletToScalarSliceToAccessNodePattern(),
                    LowerInterstateConditionalAssignmentsToTasklets(),
                    BranchNormalizationPipeline(),
                    RemoveRedundantAssignments(),
                ]
            else:
                passes = [
                    RewriteModuloToPyMod(),
                    CleanAccessNodeToScalarSliceToTaskletPattern(),
                    CleanTaskletToScalarSliceToAccessNodePattern(),
                    EliminateBranches(),
                    RemoveRedundantAssignments(),
                    LowerInterstateConditionalAssignmentsToTasklets(),
                ]
            passes += [
                RemoveEmptyStates(),
                # loop->map (branch normalization already removed the blocking ConditionalBlocks).
                # See _LoopToMapPass.
                _LoopToMapPass(permissive=loop_to_map_permissive),
                # Genuine scalar reductions (``dot += a[i]*b[i]``, ``acc = sum(A)``) survive as a scalar
                # ``CR:+`` / ``CR:*`` MapExit edge ``WCRToAugAssign`` can't strip (not a sequential
                # in-place RMW). Lift to product buffer + ``Reduce`` libnode HERE, before
                # ``_WCRToAugAssignPass`` asserts no map-body WCR. RMW-shape reductions lifted later by
                # the full pass (needs final-shape body after ``InlineSDFGs``).
                _EarlyPureWCRReductionLift(),
                # Strip inner WCR → explicit aug-assign. See _WCRToAugAssignPass.
                _WCRToAugAssignPass(),
                # dtype casts NOT stripped: kept as 1-input cast tasklets → ``TileUnop(op='cast')``
                # (explicit per-lane convert), so integer arithmetic feeding a cast keeps natural width.
                PowerOperatorExpansion(),
                SplitTasklets(),
                # Normalise direct ``MapEntry -> AccessNode`` / ``AccessNode -> MapExit`` staging edges
                # (from frontend shifted reads like ``b[i + 1]``) into 3-node chains with a plain
                # ``_out = _in`` tasklet. Vectorize refuses memlets with ``other_subset`` set; this
                # eliminates them.
                InsertAssignTaskletsAtMapBoundary(),
            ]
            if not no_inline:
                passes.append(InlineSDFGs())
            # Reduction lift: an innermost map carrying a scalar RMW reduction across iterations (spmv
            # row ``acc = acc + data[idx]*x[indices[idx]]``) is mis-vectorized by 1-D ``Vectorize`` once
            # the reduced trip exceeds W (per-chunk partials never folded). Rewrite to product-map +
            # ``Reduce`` libnode: product map vectorizes as gather + product (scalar remainder keeps the
            # gather tail in range), ``Reduce`` carries its own vectorized horizontal fold. No-op without
            # a map-carried reduction. After ``InlineSDFGs`` (needs final-shape indirect-access NSDFG).
            passes.append(LiftMapReductionToReduce())
            # P1 (NestInnermostMapBodyIntoNSDFG) + P2 (SplitMapForVectorRemainder) ALWAYS run
            # (divisibility analysis above; ``remainder_strategy`` only selects the remainder *shape*).
            #
            # R1: scalar postamble. Main step-1 (vectorize tiles to step-W normally) + a step-1
            # ScheduleType.Sequential remainder the vectorize pass leaves alone (codegen emits a plain
            # scalar tail loop).
            if remainder_strategy == "scalar":
                passes.extend([
                    NestInnermostMapBodyIntoNSDFG(vector_width=vector_width),
                    SplitMapForVectorRemainder(vector_width=vector_width, mode="scalar"),
                ])
            # R2: masked remainder. Main step-W (no mask) + step-W remainder body NSDFG; P3 attaches
            # _iter_mask so lane-fanout collapse (R2-b) picks the _masked template variant.
            elif remainder_strategy == "masked":
                # P2 emits remainder as step-1 length-R with a ``__masked_rem`` label marker; P3 detects
                # it and attaches _iter_mask using the map's static lb in the fill formula (mask survives
                # Vectorize's tiling). Vectorize tiles step-1 length-R → outer step-W trip-1 + inner
                # step-1 length-W (standard form); C.2-b wiring routes every body tasklet to its _masked
                # variant, gating trailing OOB lanes.
                passes.extend([
                    NestInnermostMapBodyIntoNSDFG(vector_width=vector_width),
                    SplitMapForVectorRemainder(vector_width=vector_width, mode="masked"),
                    GenerateIterationMask(vector_width=vector_width,
                                          mode="masked",
                                          lower_to_intrinsics=lower_to_intrinsics),
                ])
            # Vectorizer-entry precondition: nested body carries no loose WCR. See _AssertNoBodyWCRPass.
            passes.append(_AssertNoBodyWCRPass())
            passes.append(vectorizer)
        else:
            passes = [RemoveMathCall(), vectorizer]
        # ``fuse_overlapping_loads`` covers two patterns (mutually exclusive on ``insert_copies``, so
        # fusion never runs twice; GPU pipeline always uses the standalone pass):
        # 1. ``insert_copies=True``: array read at overlapping subsets fused into one union-window
        #    staging buffer inside ``add_copies_before_and_after_nsdfg`` (composes with movable/unmovable
        #    classification).
        # 2. ``insert_copies=False``: post-vectorization between-maps ``MapEntry -> [AccessNode x N] ->
        #    MapEntry`` load fan (not from the copy-staging path) → standalone ``FuseOverlappingLoads``,
        #    appended only when there's no copy-staging path to bake into.
        # TODO: collect the optional optimisation knobs (``lower_to_intrinsics``, ``insert_copies``,
        # ``try_to_demote_symbols_in_nsdfgs``, ...) under an ``optimisations`` group/preset; could fully
        # bake pattern (2) to retire the standalone pass.
        if fuse_overlapping_loads and not insert_copies:
            passes.append(FuseOverlappingLoads())
        # Lower the scalar ``assign_<i>`` fans around each ``_packed`` access node into single
        # ``gather_double`` / ``scatter_double`` / ``strided_{load,store}_double`` intrinsic calls (four
        # passes share ``utils/lane_fanout.py``; only dispatch direction + pattern differ).
        # Gather/scatter collapse runs unconditionally: the masked vector remainder MUST use the
        # intrinsic (per-lane scalar fan faults on inactive lanes). ``gather_intrinsic`` /
        # ``scatter_intrinsic`` control only the *main* loop — ``False`` collapses the masked remainder
        # only, main loop keeps its per-lane scalar fan (C++ may still auto-vectorize).
        passes.append(
            DetectGather(only_masked=not gather_intrinsic, collapse_laneid_index_loads=collapse_laneid_index_loads))
        passes.append(
            DetectScatter(only_masked=not scatter_intrinsic, collapse_laneid_index_loads=collapse_laneid_index_loads))
        # Strided / multi-dim-strided collapse stays opt-in.
        if lower_to_intrinsics:
            passes.extend([
                DetectStridedLoad(),
                DetectStridedStore(),
                DetectMultiDimStridedLoad(),
                DetectMultiDimStridedStore(),
            ])
        if eliminate_trivial_vector_map:
            passes.append(RemoveVectorMaps())
        # Masked-remainder verifier (Option B): with _iter_mask in scope, every per-lane memlet must
        # have been collapsed to a masked intrinsic by the detect_*.py passes. Raises a named
        # RuntimeError on any uncollapsed ``_laneid_<i>`` in a subset (prevents runtime SIGABRT/SIGSEGV
        # on inactive lanes).
        if remainder_strategy == "masked":
            passes.append(_AssertNoLaneMemletReadsPass(vector_width=vector_width))
        self._applied_before = False
        super().__init__(passes)

    def iterate_over_passes(self, sdfg: dace.SDFG) -> Iterator[Pass]:
        """Iterate over pipeline passes. Vectorization pipeline runs only once.

            :param sdfg: SDFG the pipeline is applied to.
        """
        if self._applied_before is False:
            CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, {})
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
# You can use: dace.transformation.passes.vectorization.utils.map_predicates::map_consists_of_single_nsdfg_or_no_nsdfg
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
