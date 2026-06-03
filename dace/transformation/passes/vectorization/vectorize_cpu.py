# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU vectorization pipeline: branch lowering, map preparation, then ``Vectorize``."""
import dace
from typing import Iterator, List, Optional, Set
from dace.transformation import Pass, pass_pipeline as ppl
from dace.transformation.passes.vectorization.remove_reduntant_assignments import RemoveRedundantAssignments
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import CleanAccessNodeToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import PowerOperatorExpansion, RemoveFPTypeCasts, RemoveIntTypeCasts, RemoveMathCall
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
from dace.transformation.passes.vectorization.split_map_for_vector_remainder import SplitMapForVectorRemainder
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.utils.iteration import assert_no_lane_memlet_reads


class _LoopToMapPass(ppl.Pass):
    """Thin Pass wrapper running :class:`LoopToMap` to a fixed point.

    Runs after branch normalization (which removes the
    ``ConditionalBlock`` s that block the conversion) so a data-parallel
    ``for i in range(...)`` loop body becomes a Map the vectorizer can
    stride. Kernels written with ``dace.map`` have no ``LoopRegion`` s so
    this is a no-op for them; loops that are not provably parallel are
    left alone by ``LoopToMap``'s own safety analysis.
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
                 collapse_laneid_index_loads: bool = False,
                 loop_to_map_permissive: bool = False,
                 force_autovec_ops: Optional[Set[str]] = None,
                 force_pscalar_ops: Optional[Set[str]] = None,
                 remainder_strategy: str = "scalar",
                 num_cores: int = 1,
                 sve_style: Optional[str] = None):
        """Build the pipeline.

        :param vector_width: SIMD lane count.
        :param try_to_demote_symbols_in_nsdfgs: demote NSDFG symbols to scalars when safe.
        :param fuse_overlapping_loads: fuse an array read at multiple
            overlapping subsets into one shared union window. With
            ``insert_copies=True`` this is baked into the NSDFG staging copy
            (one union buffer instead of one ``vector_copy`` per subset);
            with ``insert_copies=False`` the standalone ``FuseOverlappingLoads``
            pass fuses the post-vectorization between-maps load fan.
        :param apply_on_maps: restrict vectorization to these map entries.
        :param insert_copies: insert copy-in/out around NSDFG boundaries.
        :param only_apply_vectorization_pass: run only ``Vectorize`` (skip prep/cleanup).
        :param no_inline: skip ``InlineSDFGs``.
        :param fail_on_unvectorizable: raise instead of skipping a map that cannot be vectorized.
        :param eliminate_trivial_vector_map: append ``RemoveVectorMaps``.
        :param user_skip_nsdfg_arrays: NSDFG array names to exclude from copy-in/out.
        :param use_fp_factor: branch lowering via ``c*x + (1-c)*y`` (mutually exclusive with ``branch_normalization``).
        :param branch_normalization: branch lowering via the M3 ``ITE`` normalization.
        :param lower_to_intrinsics: also collapse strided / multi-dim-strided
            per-lane fans to intrinsics (gather/scatter collapse is always on
            — see ``gather_intrinsic``/``scatter_intrinsic``).
        :param gather_intrinsic: ``True`` (default) emits the ``gather``
            intrinsic for the main loop; ``False`` keeps the main loop's
            per-lane scalar gather fan. The masked vector remainder always
            uses the intrinsic regardless (per-lane scalar fan faults on
            inactive lanes).
        :param scatter_intrinsic: as ``gather_intrinsic`` but for scatter.
        :param collapse_laneid_index_loads: when ``True``, a recognised
            per-lane laneid index fan (W laneid symbols bound to a
            contiguous ``<idxarr>[0:W]`` slice) is collapsed so the
            gather intrinsic reads the index array directly via an
            ``_idx`` connector; the dead laneid symbols and their
            interstate-edge assignments are removed. Default ``False``
            keeps the per-lane laneid-symbol form.
        :param loop_to_map_permissive: run the internal ``LoopToMap`` in
            permissive mode. A scatter ``for i in range(...): a[idx[i]]
            = ...`` loop has a data-dependent write index that the
            non-permissive safety analysis treats as a possible write
            conflict and refuses to parallelize; permissive mode accepts
            it (the caller asserts the indices are conflict-free).
            Default ``False``.
        :param force_autovec_ops: ops to emit as ``vector_<op>_av`` (autovec hint).
        :param force_pscalar_ops: ops to emit as ``vector_<op>_pscalar`` (no autovec hint).
        :param remainder_strategy: ``"scalar"`` (scalar postamble), ``"masked"`` (iter-mask
            remainder) or ``"full_loop_mask"`` (R3, not yet wired).
        :param num_cores: number of contiguous core blocks the innermost
            data-parallel map is tiled into. Only meaningful with
            ``sve_style`` set (the SVE-style model tiles the map across
            ``num_cores`` cores, then turns each per-core chunk into a
            masked while-loop). ``<= 1`` is the inert default; ``sve_style``
            requires ``> 1``.
        :param sve_style: select the **SVE-style always-mask emission
            model** (``None`` = off, today's pipeline). Every operation is
            emitted with an ``_iter_mask`` so the trailing partial block
            needs no remainder loop — the mask gates the inactive lanes.
            Two lengths:

            - ``"fixed"`` — compile-time ``vector_width`` lanes. Runs on
              AVX-512 / portable x86: the architecture's mask register IS
              the iteration mask. This is *SVE-style on a fixed-width ISA*,
              not the ARM SVE backend.
            - ``"variable"`` — ARM SVE runtime vector length
              (``svcntd()`` / ``svwhilelt_b64``); the D2 Map→SVE-while
              lowering. Queued as the final SVE task — raises
              ``NotImplementedError`` for now.

            **Knob interaction under ``sve_style`` (the always-mask model
            makes several tuning knobs forced, irrelevant, or
            conflicting):**

            - *Forced on* (set internally; their default is fine, you do
              not pass them): branch lowering forced to the ITE path
              (``use_fp_factor`` is ignored — the always-mask model
              requires SIMD blends, so ``branch_normalization`` is on),
              ``lower_to_intrinsics=True`` (locked Option B — masked
              indirection safety).
            - *Rejected if non-default* (raise ``ValueError``):
              ``only_apply_vectorization_pass=True`` (the SVE-style prep
              chain is mandatory), ``remainder_strategy`` other than the
              default (SVE-style has no remainder loop — the mask covers
              the tail), ``gather_intrinsic=False`` / ``scatter_intrinsic
              =False`` (a per-lane scalar fan faults on inactive lanes),
              non-empty ``force_autovec_ops`` / ``force_pscalar_ops``
              (they only affect the non-masked default path, which
              SVE-style never takes — they would silently no-op).
            - *Required*: ``num_cores > 1``.
            - *Orthogonal — silently allowed, no SVE-specific meaning*:
              ``fuse_overlapping_loads``, ``collapse_laneid_index_loads``,
              ``apply_on_maps``, ``insert_copies``, ``no_inline``,
              ``fail_on_unvectorizable``, ``eliminate_trivial_vector_map``,
              ``user_skip_nsdfg_arrays``, ``loop_to_map_permissive``,
              ``try_to_demote_symbols_in_nsdfgs``.

        :raises ValueError: on a rejected knob combination (see body).
        :raises NotImplementedError: for ``remainder_strategy="full_loop_mask"``,
            for ``sve_style="variable"`` (queued), and for ``sve_style``
            set at all until the S-SVE5b pipeline assembly lands (the
            knob contract / validation / documentation are in place now;
            ``ForLoopToMaskedWhile`` lands in S-SVE5a first).
        """
        # SVE-style always-mask emission is validated FIRST so an sve_style
        # caller gets SVE-specific messages, not a downstream legacy mutex
        # (e.g. the use_fp_factor/remainder check — use_fp_factor defaults
        # True and is *ignored* under sve_style, so that check must not
        # fire here). Policy: orthogonal-harmless knobs are silently
        # allowed; relevant-but-conflicting or no-effect-under-SVE knobs
        # set to a non-default value raise. Full taxonomy: the
        # ``sve_style`` docstring entry.
        _VALID_SVE_STYLE = {None, "fixed", "variable"}
        if sve_style not in _VALID_SVE_STYLE:
            raise ValueError(f"VectorizeCPU: sve_style must be one of "
                             f"{sorted(s for s in _VALID_SVE_STYLE if s is not None)} or None, "
                             f"got {sve_style!r}")
        sve_fixed = False
        if sve_style is not None:
            if sve_style == "variable":
                # Per design pivot 2026-05-20: variable-VL emission
                # (one opaque CPP tasklet per map with a svwhilelt-
                # driven while-loop body) is deferred. For SVE hardware
                # use ``sve_style="fixed"`` with ``vector_width`` matched
                # to the target SVE register width (W=8 for SVE-512, W=4
                # for SVE-256, etc.) — the existing fixed chain already
                # emits svwhilelt + svcntd internally per W-chunk via
                # cpu_vectorizable_math_arm_sve.h. The exploratory
                # ``SveStyleVariableFinalize`` class is retained in
                # vectorize_sve.py as a prototype of the future
                # "whole map as CPP tasklet" approach (SpMV + axpy +
                # triad recognisers implemented), but not user-reachable
                # via this knob.
                raise NotImplementedError("VectorizeCPU: sve_style='variable' is deferred (open task). For SVE "
                                          "hardware use sve_style='fixed' with vector_width matched to the target "
                                          "SVE register width (W=8 for SVE-512, W=4 for SVE-256, etc.); the SVE "
                                          "arch header (cpu_vectorizable_math_arm_sve.h) already uses svwhilelt + "
                                          "svcntd per W-chunk internally. The variable-VL whole-map-to-CPP-tasklet "
                                          "approach is parked in SveStyleVariableFinalize as a prototype.")
            # Branch lowering is forced to the ITE path: ``use_fp_factor``
            # defaults True (legacy), so rejecting it would force every
            # sve_style caller to also pass use_fp_factor=False. The
            # always-mask model requires ITE blends, so use_fp_factor is
            # ignored and branch_normalization is forced on (documented;
            # the forced assignment lands with the S-SVE5b pipeline).
            if only_apply_vectorization_pass:
                raise ValueError("VectorizeCPU: sve_style needs the full tile->mask->for->while "
                                 "prep chain; only_apply_vectorization_pass must be False")
            if remainder_strategy != "scalar":
                raise ValueError("VectorizeCPU: sve_style has no remainder loop (the iteration "
                                 "mask covers the trailing partial block); remainder_strategy is "
                                 "N/A under sve_style — leave it at the default")
            if not gather_intrinsic or not scatter_intrinsic:
                raise ValueError("VectorizeCPU: sve_style forces gather/scatter intrinsics "
                                 "(a per-lane scalar fan faults on inactive lanes); "
                                 "gather_intrinsic and scatter_intrinsic must stay True")
            if force_autovec_ops or force_pscalar_ops:
                raise ValueError("VectorizeCPU: sve_style emits the masked intrinsic path only; "
                                 "force_autovec_ops / force_pscalar_ops affect the non-masked "
                                 "default path and would silently no-op under sve_style")
            if num_cores <= 1:
                raise ValueError("VectorizeCPU: sve_style tiles the innermost map across "
                                 "num_cores contiguous core blocks; pass num_cores > 1")
            # sve_style='fixed' forces the ITE branch front and the
            # masked intrinsic path. ``use_fp_factor`` defaults True
            # (legacy) — override silently to the ITE front as
            # documented; force lower_to_intrinsics so masked gather/
            # scatter is safe (a per-lane scalar fan faults on inactive
            # lanes). The chain itself is the SveStyleFinalize
            # orchestrator appended after the shared front below.
            sve_fixed = (sve_style == "fixed")
            use_fp_factor = False
            branch_normalization = True
            lower_to_intrinsics = True
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
        # the masked path emits canonical ITE tasklets / iter_mask blends
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
        # ``"ITE"``, ``"+c"``, ``"c-"``, ``"log"``, ``"=c"`` etc.).
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
                               user_skip_nsdfg_arrays=user_skip_nsdfg_arrays,
                               fuse_overlapping_loads=fuse_overlapping_loads)
        if not only_apply_vectorization_pass:
            # Pick the branch-lowering front of the pipeline. ``use_fp_factor``
            # keeps today's behaviour, ``EliminateBranches`` collapses if/else
            # to the FP-factor form ``a = c*x + (1-c)*y``. ``branch_normalization``
            # opts into the new M3 pipeline, ``SameWriteSetIfElseToITECFG``
            # rewrites same-write-set arms to a 3-CFG ITE form, then
            # ``BranchNormalization`` flattens the rest into ITE tasklets.
            # In ITE mode, ``LowerInterstateConditionalAssignmentsToTasklets``
            # runs before the M3 passes so the condition variable lives in a
            # tasklet plus a transient array (the vectorizer can then drive
            # it per-lane through the ITE tasklet's ``_c`` connector).
            # Fold the frontend ``A -> A_slice(scalar) -> tasklet`` reads up
            # front, while ``A -> A_slice`` is still a direct AccessNode edge.
            # After ``LoopToMap`` the MapEntry sits between ``A`` and its
            # slice scalar, so the pattern no longer matches — the scalar
            # would survive into the vectorizer and be widened along the
            # wrong (contiguous) array dim for a non-contiguous vectorised
            # access (e.g. C-layout ``bb[i, j]`` with ``i`` innermost).
            if branch_normalization:
                passes = [
                    CleanAccessNodeToScalarSliceToTaskletPattern(),
                    LowerInterstateConditionalAssignmentsToTasklets(),
                    BranchNormalizationPipeline(),
                    RemoveRedundantAssignments(),
                ]
            else:
                passes = [
                    CleanAccessNodeToScalarSliceToTaskletPattern(),
                    EliminateBranches(),
                    RemoveRedundantAssignments(),
                    LowerInterstateConditionalAssignmentsToTasklets(),
                ]
            passes += [
                RemoveEmptyStates(),
                # Branch normalization has removed the ConditionalBlocks
                # that block loop->map conversion; a data-parallel
                # ``for i in range(...)`` body now becomes a Map the
                # vectorizer can stride. Runs before the Vectorize / prep
                # passes. No-op for ``dace.map`` kernels.
                _LoopToMapPass(permissive=loop_to_map_permissive),
                RemoveFPTypeCasts(),
                RemoveIntTypeCasts(),
                PowerOperatorExpansion(),
                SplitTasklets(),
                # Normalise direct ``MapEntry -> AccessNode`` / ``AccessNode -> MapExit``
                # staging edges (produced by python-frontend shifted reads like
                # ``b[i + 1]``) into 3-node chains with a plain ``_out = _in``
                # tasklet in the middle. The vectorize pass refuses memlets with
                # ``other_subset`` set; this pass eliminates them as a precursor.
                InsertAssignTaskletsAtMapBoundary(),
            ]
            if not no_inline:
                passes.append(InlineSDFGs())
            if sve_fixed:
                # SVE-style 'fixed': the whole tile -> mask -> vectorize ->
                # detect -> MapToForLoop -> ForLoopToMaskedWhile chain is a
                # single coordinating pass that captures the global trip
                # bound at tile time (the M3.2 pattern). It owns the
                # vectorizer + detect + RemoveVectorMaps internally, so the
                # normal prep / post-vectorizer blocks are bypassed.
                from dace.transformation.passes.vectorization.vectorize_sve import SveStyleFinalize
                passes.append(
                    SveStyleFinalize(vectorizer,
                                     vector_width=vector_width,
                                     num_cores=num_cores,
                                     lower_to_intrinsics=lower_to_intrinsics,
                                     eliminate_trivial_vector_map=eliminate_trivial_vector_map))
                self._applied_before = False
                super().__init__(passes)
                return
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
                    GenerateIterationMask(vector_width=vector_width,
                                          mode="masked",
                                          lower_to_intrinsics=lower_to_intrinsics),
                ])
            passes.append(vectorizer)
        else:
            passes = [RemoveMathCall(), vectorizer]
        # ``fuse_overlapping_loads`` covers two distinct patterns:
        #
        # 1. ``insert_copies=True``: an array read at multiple overlapping
        #    subsets is fused into one shared union-window staging buffer
        #    *inside* ``add_copies_before_and_after_nsdfg`` (baked here so it
        #    composes with the movable / unmovable classification).
        # 2. ``insert_copies=False``: the post-vectorization between-maps
        #    ``MapEntry -> [AccessNode x N] -> MapEntry`` load fan, which is
        #    not produced by the copy-staging path. This is still handled by
        #    the standalone post-vectorizer ``FuseOverlappingLoads`` pass,
        #    appended only when there is no copy-staging path to bake into.
        #
        # The two are mutually exclusive on ``insert_copies`` so fusion
        # never runs twice. (The GPU pipeline always uses the standalone
        # pass.)
        #
        # TODO: ``fuse_overlapping_loads`` is one of the optional
        # optimisation knobs the pipeline exposes (alongside
        # ``lower_to_intrinsics``, ``insert_copies``,
        # ``try_to_demote_symbols_in_nsdfgs``, etc.). A future slice should
        # collect them under a dedicated ``optimisations`` group / preset,
        # and could fully bake pattern (2) to retire the standalone pass.
        if fuse_overlapping_loads and not insert_copies:
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
