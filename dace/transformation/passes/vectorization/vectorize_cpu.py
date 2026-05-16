# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU vectorization pipeline: branch lowering, map preparation, then ``Vectorize``."""
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
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalizationPipeline
from dace.transformation.passes.vectorization.detect_gather import DetectGather
from dace.transformation.passes.vectorization.detect_scatter import DetectScatter
from dace.transformation.passes.vectorization.detect_strided_load import DetectStridedLoad
from dace.transformation.passes.vectorization.detect_strided_store import DetectStridedStore
from dace.transformation.passes.vectorization.detect_multi_dim_strided_load import DetectMultiDimStridedLoad
from dace.transformation.passes.vectorization.detect_multi_dim_strided_store import DetectMultiDimStridedStore
from dace.transformation.passes.vectorization.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.split_map_for_vector_remainder import SplitMapForVectorRemainder
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.utils.iteration import assert_no_lane_memlet_reads


class _AssertNoLaneMemletReadsPass(ppl.Pass):
    """Thin Pass wrapper that runs ``assert_no_lane_memlet_reads`` after
    the vectorizer. Lands in the pipeline only when ``remainder_strategy='masked'``
    so the loud failure is exactly the locked-decision contract — see
    Option B in the plan."""

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
                 force_autovec_ops: Optional[Set[str]] = None,
                 force_pscalar_ops: Optional[Set[str]] = None,
                 remainder_strategy: str = "scalar"):
        """Build the pipeline.

        :param vector_width: SIMD lane count.
        :param try_to_demote_symbols_in_nsdfgs: demote NSDFG symbols to scalars when safe.
        :param fuse_overlapping_loads: append ``FuseOverlappingLoads``.
        :param apply_on_maps: restrict vectorization to these map entries.
        :param insert_copies: insert copy-in/out around NSDFG boundaries.
        :param only_apply_vectorization_pass: run only ``Vectorize`` (skip prep/cleanup).
        :param no_inline: skip ``InlineSDFGs``.
        :param fail_on_unvectorizable: raise instead of skipping a map that cannot be vectorized.
        :param eliminate_trivial_vector_map: append ``RemoveVectorMaps``.
        :param user_skip_nsdfg_arrays: NSDFG array names to exclude from copy-in/out.
        :param use_fp_factor: branch lowering via ``c*x + (1-c)*y`` (mutually exclusive with ``branch_normalization``).
        :param branch_normalization: branch lowering via the M3 ``merge`` normalization.
        :param lower_to_intrinsics: also collapse strided / multi-dim-strided
            per-lane fans to intrinsics (gather/scatter collapse is always on
            — see ``gather_intrinsic``/``scatter_intrinsic``).
        :param gather_intrinsic: ``True`` (default) emits the ``gather``
            intrinsic for the main loop; ``False`` keeps the main loop's
            per-lane scalar gather fan. The masked vector remainder always
            uses the intrinsic regardless (per-lane scalar fan faults on
            inactive lanes).
        :param scatter_intrinsic: as ``gather_intrinsic`` but for scatter.
        :param force_autovec_ops: ops to emit as ``vector_<op>_av`` (autovec hint).
        :param force_pscalar_ops: ops to emit as ``vector_<op>_pscalar`` (no autovec hint).
        :param remainder_strategy: ``"scalar"`` (scalar postamble), ``"masked"`` (iter-mask
            remainder) or ``"full_loop_mask"`` (R3, not yet wired).
        :raises ValueError: on a rejected knob combination (see body).
        :raises NotImplementedError: for ``remainder_strategy="full_loop_mask"``.
        """
        if use_fp_factor and branch_normalization:
            raise ValueError("VectorizeCPU: use_fp_factor and branch_normalization are mutually exclusive; "
                             "choose one branch-lowering strategy")
        # ``remainder_strategy`` controls how the pipeline handles maps whose
        # iteration count is not *provably* divisible by ``vector_width``.
        # There is no longer a ``"divides_evenly"`` mode: P2
        # (``SplitMapForVectorRemainder``) always runs and does the symbolic
        # divisibility analysis itself — if ``simplify(ub-lb+1) % W == 0`` is
        # provably true (e.g. a map over ``0 : 8*N``) it does NOT split, so
        # the SDFG carries no remainder map and the main path tiles cleanly.
        # When divisibility cannot be proven, P2 always splits and the
        # strategy below picks the remainder shape:
        #   "scalar"          - R1 (default): main step-W map + a step-1
        #                       ``ScheduleType.Sequential`` scalar postamble.
        #                       No mask anywhere.
        #   "masked"          - R2: main step-W (no mask) + step-W remainder
        #                       body with a P3 ``_iter_mask`` so the trailing
        #                       OOB lanes are gated.
        #   "full_loop_mask"  - R3 (TODO): no remainder split; one step-W map
        #                       with ``_iter_mask`` wired everywhere
        #                       (SVE-style).
        _VALID_REMAINDER = {"scalar", "masked", "full_loop_mask"}
        if remainder_strategy not in _VALID_REMAINDER:
            raise ValueError(f"VectorizeCPU: remainder_strategy must be one of "
                             f"{sorted(_VALID_REMAINDER)}, got {remainder_strategy!r}")
        if remainder_strategy == "full_loop_mask":
            raise NotImplementedError("VectorizeCPU: remainder_strategy='full_loop_mask' is queued (R3); "
                                      "currently only 'scalar' and 'masked' are wired end-to-end.")
        # K1=fp_factor + K2=masked is rejected per the locked plan decision:
        # the masked path emits canonical merge tasklets / iter_mask blends
        # that fp-factor lowering can't combine with cleanly (would need a
        # bool-to-float cast on every iteration). Use branch_normalization
        # for the masked path, or remainder_strategy="scalar" for fp_factor.
        if use_fp_factor and remainder_strategy == "masked":
            raise ValueError("VectorizeCPU: use_fp_factor=True is incompatible with "
                             "remainder_strategy='masked'; choose branch_normalization=True or "
                             "remainder_strategy='scalar'")
        # ``force_autovec_ops`` / ``force_pscalar_ops`` (Option F overlay)
        # let callers override per-op which implementation the emitter
        # selects. Keys are templates-dict op identifiers (``"+"``,
        # ``"merge"``, ``"+c"``, ``"c-"``, ``"log"``, ``"=c"`` etc.).
        #   force_pscalar_ops={"div"}  -> emit ``vector_div_pscalar``
        #                                  (pure scalar loop, no autovec hint)
        #   force_autovec_ops={"exp"}  -> emit ``vector_exp_av``
        #                                  (scalar loop + _dace_vectorize hint)
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
            ">": "vector_gt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "<": "vector_lt<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            ">=": "vector_ge<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "<=": "vector_le<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "==": "vector_eq<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "!=": "vector_ne<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
            "merge": "vector_select<{dtype}, {vector_width}>({lhs}, {cond}, {then_arm}, {else_arm});",
            "merge_masked":
            "vector_select_av_masked<{dtype}, {vector_width}>({lhs}, {cond}, {then_arm}, {else_arm}, {mask});",
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
            # Masked variants (Option F overlay): emitted when the tasklet has
            # an ``_iter_mask`` input connector wired from a P3-generated
            # ``_iter_mask: bool[W]`` array. Bound to ``_av_masked`` from
            # ``cpu_vectorizable_math_common.h`` — uniformly available across
            # arches via the macros; arch-specific masked specializations can
            # be added later for ops where they're a measured win.
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
        # Apply per-op variant overrides. Rewrites ``vector_<name><...>`` to
        # ``vector_<name>_<suffix><...>`` in the affected template string;
        # the suffixed functions are defined in cpu_vectorizable_math_common.h
        # for every op identically across arches.
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
                    BranchNormalizationPipeline(),
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
                # Normalise direct ``MapEntry -> AccessNode`` / ``AccessNode -> MapExit``
                # staging edges (produced by python-frontend shifted reads like
                # ``b[i + 1]``) into 3-node chains with a plain ``_out = _in``
                # tasklet in the middle. The vectorize pass refuses memlets with
                # ``other_subset`` set; this pass eliminates them as a precursor.
                InsertAssignTaskletsAtMapBoundary(),
            ]
            if not no_inline:
                passes.append(InlineSDFGs())
            # P1 (NestInnermostMapBodyIntoNSDFG) + P2
            # (SplitMapForVectorRemainder) ALWAYS run.  P2 does the symbolic
            # divisibility analysis: when ``simplify(ub-lb+1) % W == 0`` is
            # provably true (e.g. a map over ``0 : 8*N``) it returns without
            # splitting, so no remainder map is emitted and the main path
            # tiles cleanly.  When divisibility cannot be proven P2 always
            # splits; ``remainder_strategy`` only selects the remainder
            # *shape*.
            #
            # R1: scalar postamble. Main step-1 (which the vectorize pass
            # tiles to step-W in the normal way) plus a step-1
            # ScheduleType.Sequential remainder that the vectorize pass
            # leaves alone — the codegen emits it as a plain scalar tail
            # loop.
            if remainder_strategy == "scalar":
                passes.extend([
                    NestInnermostMapBodyIntoNSDFG(vector_width=vector_width),
                    SplitMapForVectorRemainder(vector_width=vector_width, mode="scalar"),
                ])
            # R2: masked remainder. Split into main step-W (no mask) + step-W
            # remainder body NSDFG; P3 attaches _iter_mask to the remainder
            # body so the lane-fanout collapse (R2-b) picks the _masked
            # template variant on remainder-side fanouts.
            elif remainder_strategy == "masked":
                # P2 emits remainder as step-1 length-R with a ``__masked_rem``
                # label marker; P3 detects the marker and attaches _iter_mask
                # using the map's static lb in the fill formula (so the mask
                # survives Vectorize's tiling unchanged). Vectorize then tiles
                # the step-1 length-R remainder to outer step-W trip-1 +
                # inner step-1 length-W (the standard form), and the C.2-b
                # wiring routes every body tasklet to its _masked runtime
                # variant so the trailing OOB lanes are gated.
                passes.extend([
                    NestInnermostMapBodyIntoNSDFG(vector_width=vector_width),
                    SplitMapForVectorRemainder(vector_width=vector_width, mode="masked"),
                    GenerateIterationMask(vector_width=vector_width, mode="masked"),
                ])
            passes.append(vectorizer)
        else:
            passes = [RemoveMathCall(), vectorizer]
        # TODO: ``fuse_overlapping_loads`` is one of the optional optimisation
        # knobs the pipeline exposes (alongside ``lower_to_intrinsics``,
        # ``insert_copies``, ``try_to_demote_symbols_in_nsdfgs``, etc.). The
        # plan should record these as opt-in optimisation passes the user can
        # enable per-call; they default ``False`` to keep the baseline
        # pipeline minimal. A future slice should collect them under a
        # dedicated ``optimisations`` group / preset for ergonomics.
        if fuse_overlapping_loads:
            passes.append(FuseOverlappingLoads())
        # Lower the scalar ``assign_<i>`` fans the vectorizer leaves around
        # each ``_packed`` access node into single ``gather_double`` /
        # ``scatter_double`` / ``strided_{load,store}_double`` intrinsic
        # calls. The four passes share a single implementation in
        # ``utils/lane_fanout.py``; only the dispatch direction and
        # pattern differ. Default ``False`` keeps the scalar shape (the
        # ``assign_<i>`` fan stays in the SDFG and the C++ compiler
        # auto-vectorizes it if it can).
        # Gather/scatter fan collapse runs unconditionally: the masked
        # vector remainder MUST use the intrinsic (a per-lane scalar fan
        # faults on inactive lanes). ``gather_intrinsic``/``scatter_intrinsic``
        # only control the *main* loop — when ``False`` the pass collapses
        # the masked remainder only and the main loop keeps its per-lane
        # scalar fan (which the C++ compiler may still auto-vectorize).
        passes.append(DetectGather(only_masked=not gather_intrinsic))
        passes.append(DetectScatter(only_masked=not scatter_intrinsic))
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
        # Masked-remainder verifier (Option B in the plan): when _iter_mask
        # is in scope, every per-lane memlet must have been collapsed by the
        # detect_*.py passes to a masked intrinsic. The verifier raises a
        # named NotImplementedError-style RuntimeError if it finds any
        # uncollapsed _laneid_<i> in a subset — preventing the SIGABRT/SIGSEGV
        # that would otherwise happen at runtime on inactive lanes.
        if remainder_strategy == "masked":
            passes.append(_AssertNoLaneMemletReadsPass(vector_width=vector_width))
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
