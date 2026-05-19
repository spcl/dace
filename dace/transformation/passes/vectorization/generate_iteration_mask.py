# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Attach a per-iteration boolean lane mask to innermost map bodies.

A transient ``_iter_mask`` with ``mask[l] = (lb + l <= ub)`` is added to
the body NestedSDFG of every targeted innermost map; the downstream
emitter recognises the name and switches to masked intrinsic variants.
The mask is filled by a CPP tasklet in a prepended start state.

Precondition: every innermost map body is already a single NestedSDFG
(produced by ``NestInnermostMapBodyIntoNSDFG``); bare-tasklet bodies are
rejected.
"""
from typing import Optional

import dace
from dace import properties
from dace.data import add_mask
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.lane_access import classify_lane_access
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_innermost_map,
)


@properties.make_properties
class GenerateIterationMask(ppl.Pass):
    """Attach ``_iter_mask`` to the body of every target innermost map."""

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    mode = properties.Property(dtype=str,
                               default="step_w_only",
                               allow_none=False,
                               desc="``step_w_only`` masks only maps with step==vector_width "
                               "(legacy step-W detection); ``all_innermost`` masks every innermost "
                               "map (used by full_loop_mask strategy); ``masked`` masks maps tagged "
                               "``__masked_rem`` by SplitMapForVectorRemainder(mode='masked'); "
                               "``global`` (SVE-style) masks every innermost map with the fill keyed "
                               "to the *running loop variable* against the *global* trip "
                               "(``mask[l] = (iter_var + l < global_ub)``) rather than the map's own "
                               "static ``lb``/``ub`` — required when the per-core block spans multiple "
                               "W-tiles so the cutoff differs per tile. The first three mode names "
                               "match the ``VectorizeCPU.remainder_strategy`` knob.")
    global_ub = properties.Property(dtype=str,
                                    default=None,
                                    allow_none=True,
                                    desc="``mode='global'`` only: the original (pre-tile) *exclusive* "
                                    "upper bound of the innermost trip (e.g. ``\"N\"``). The mask fill "
                                    "is ``iter_var + l < global_ub``. Required for ``mode='global'``.")

    lower_to_intrinsics = properties.Property(dtype=bool,
                                              default=False,
                                              desc="Whether the pipeline collapses per-lane gather/scatter/strided "
                                              "fans to masked intrinsics. When False, a masked remainder whose "
                                              "body has an access that fans out per-lane (transposed / strided "
                                              "over the lane var) cannot be made safe — the per-lane reads would "
                                              "fault on inactive tail lanes. Such a map is auto-degraded to a "
                                              "scalar (Sequential, step-1) remainder instead of being masked.")

    def __init__(self,
                 vector_width: int = 8,
                 mode: str = "step_w_only",
                 lower_to_intrinsics: bool = False,
                 global_ub: Optional[str] = None):
        super().__init__()
        self.vector_width = vector_width
        self.mode = mode
        self.lower_to_intrinsics = lower_to_intrinsics
        self.global_ub = global_ub

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Attach the iteration mask to every targeted innermost map body.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of masks attached, or ``None`` if none.
        :raises ValueError: If ``mode`` is not a recognised mode.
        :raises NotImplementedError: If a targeted map body is not a single NestedSDFG.
        """
        if self.mode not in ("step_w_only", "all_innermost", "masked", "global"):
            raise ValueError(f"GenerateIterationMask.mode must be 'step_w_only', 'all_innermost', "
                             f"'masked', or 'global', got {self.mode!r}")
        if self.mode == "global" and self.global_ub is None:
            raise ValueError("GenerateIterationMask.mode='global' requires global_ub (the original "
                             "pre-tile exclusive upper bound of the innermost trip)")
        W = self.vector_width
        applied = 0
        for n, g in [(n, g) for n, g in sdfg.all_nodes_recursive()
                     if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)]:
            if not is_innermost_map(g, n):
                continue
            if not n.map.range.ranges:
                continue
            lb, ub, step = n.map.range[-1]
            if self.mode == "step_w_only" and (step != W) and (str(step) != str(W)):
                continue
            if self.mode == "masked" and not n.map.label.endswith("__masked_rem"):
                continue
            # 'global' (SVE-style) and 'all_innermost' target every
            # innermost map; 'global' additionally keys the fill to the
            # running loop variable against the global trip.
            nsdfg_node = get_single_nsdfg_inside_map(g, n)
            if nsdfg_node is None:
                raise NotImplementedError(f"GenerateIterationMask requires every innermost map's body to be a single "
                                          f"NestedSDFG (run NestInnermostMapBodyIntoNSDFG first); map {n.label!r} has "
                                          f"a bare-tasklet body")
            # Auto-degrade an unsafe masked remainder to a scalar one.
            # A masked remainder runs the W-wide body and relies on each
            # per-lane gather/scatter/strided fan collapsing to a *masked*
            # intrinsic that honours ``_iter_mask`` (so inactive tail lanes
            # never dereference an out-of-bounds address). Without
            # ``lower_to_intrinsics`` that collapse cannot happen, so a
            # body whose access fans out per-lane (transposed / strided
            # over the lane variable) would OOB-fault on inactive lanes.
            # Rather than mask it (and have ``assert_no_lane_memlet_reads``
            # raise downstream), degrade THIS map to a scalar remainder:
            # strip the ``__masked_rem`` marker and force ``Sequential`` so
            # the vectorizer skips it and codegen emits a plain scalar
            # tail. The main (vectorised) map is unaffected. The scalar
            # postamble is universally correct for any access pattern.
            if (self.mode == "masked" and not self.lower_to_intrinsics
                    and self._body_fans_out_per_lane(g, n, nsdfg_node, n.map.params[-1])):
                if n.map.label.endswith("__masked_rem"):
                    n.map.label = n.map.label[:-len("__masked_rem")]
                n.map.schedule = dace.dtypes.ScheduleType.Sequential
                continue
            global_ub = self.global_ub if self.mode == "global" else None
            if self._attach_mask(nsdfg_node, n.map.params[-1], lb, ub, W, global_ub):
                applied += 1
        return applied or None

    @staticmethod
    def _subset_fans_out(sub, strides, iter_var: str) -> bool:
        """Whether one ``(subset, array-strides)`` pair fans out per lane.

        ``True`` iff the access is strided (``A[2*i]``), transposed
        (``iter_var`` in a non-stride-1 dim, e.g. ``zqx[z1, i, j]``) or
        diagonal (``iter_var`` in >1 dims). Each lowers to a per-lane
        gather fan that is unsafe under a masked remainder without a
        masked intrinsic. Delegates to the canonical
        :func:`classify_lane_access`.
        """
        if strides is None or sub is None or len(strides) != len(sub):
            return False
        return classify_lane_access(sub, strides, iter_var).fans_out_per_lane

    @staticmethod
    def _lane_variant_indirect_dim_count(inner_sdfg, sub, iter_var: str) -> int:
        """Count subset dims that are LANE-VARIANT indirect gather components.

        A dim counts iff its ``begin`` reaches an array read whose index
        depends on ``iter_var`` (the vectorised lane), either:

        - directly: ``begin`` contains an array-subscript ``A[... iter_var
          ...]`` (a SymPy ``Function`` named after an array in
          ``inner_sdfg.arrays`` whose args mention ``iter_var``); or
        - via a symbol: ``begin`` references a symbol whose interstate-
          edge assignment is ``<sym> = A[... iter_var ...]``.

        ``>= 2`` such dims in one array access is a multi-variable
        gather — no hardware gather intrinsic takes two per-lane index
        vectors — so a masked remainder containing it must degrade to a
        scalar tail. A LOOP-INVARIANT array-indexed index (no
        ``iter_var`` in its source) does NOT count: a single-index
        gather still collapses to the R2 masked-gather intrinsic.
        Direct affine ``iter_var`` indices are handled separately by
        :func:`classify_lane_access`; they are not counted here.

        :param inner_sdfg: The body NestedSDFG's inner SDFG.
        :param sub: The memlet subset.
        :param iter_var: The vectorised loop parameter.
        :returns: Number of lane-variant indirect gather dims.
        """
        import sympy
        from dace.transformation.passes.vectorization.utils.lane_expansion import find_symbol_assignment
        arrays = set(inner_sdfg.arrays)

        def _reads_array_with_iter(expr) -> bool:
            if not hasattr(expr, "free_symbols"):
                return False
            fnames = {getattr(f.func, "__name__", str(f.func)) for f in expr.atoms(sympy.Function)}
            fsyms = {str(s) for s in expr.free_symbols}
            return bool((fnames | fsyms) & arrays) and (iter_var in fsyms)

        count = 0
        for dim in sub:
            b = dim[0]
            if not hasattr(b, "free_symbols"):
                continue
            dim_is_lane_indirect = False
            # Direct nested array subscript in the begin expression.
            if _reads_array_with_iter(b):
                dim_is_lane_indirect = True
            else:
                for s in {str(x) for x in b.free_symbols}:
                    if s == iter_var or s in arrays:
                        continue
                    try:
                        assign = find_symbol_assignment(inner_sdfg, s)
                    except Exception:
                        assign = None
                    if not assign:
                        continue
                    try:
                        rhs = dace.symbolic.pystr_to_symbolic(assign)
                    except Exception:
                        continue
                    if _reads_array_with_iter(rhs):
                        dim_is_lane_indirect = True
                        break
            if dim_is_lane_indirect:
                count += 1
        return count

    @classmethod
    def _body_fans_out_per_lane(cls, state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                nsdfg_node: dace.nodes.NestedSDFG, iter_var: str) -> bool:
        """Whether the remainder map accesses an array transposed/strided over ``iter_var``.

        Such an access lowers to a per-lane gather/scatter fan, only safe
        under a masked remainder if collapsed to a masked intrinsic
        (``lower_to_intrinsics``). The per-lane index can live in two
        places, so both are checked:

        - the NSDFG **boundary** edges in the parent state (outer array
          strides) — e.g. ``t[i, 1]`` passed sliced into the body; or
        - the **inner** NSDFG memlets (inner array strides) — e.g.
          cloudsc_one passes ``zqx`` whole and indexes ``zqx[z1, i, j]``
          inside.

        :param state: Parent state containing ``nsdfg_node``.
        :param map_entry: The remainder map entry (its body is the NSDFG).
        :param nsdfg_node: The body NestedSDFG of the remainder map.
        :param iter_var: The remainder map's innermost loop parameter.
        :returns: ``True`` if any boundary or inner memlet fans out per lane.
        """
        for e in state.in_edges(nsdfg_node) + state.out_edges(nsdfg_node):
            if e.data.data is None or e.data.data not in state.sdfg.arrays:
                continue
            arr = state.sdfg.arrays[e.data.data]
            if cls._subset_fans_out(e.data.subset, getattr(arr, "strides", None), iter_var):
                return True
        inner = nsdfg_node.sdfg
        for st in inner.all_states():
            for e in st.edges():
                if e.data.data is None or e.data.data not in inner.arrays:
                    continue
                arr = inner.arrays[e.data.data]
                if cls._subset_fans_out(e.data.subset, getattr(arr, "strides", None), iter_var):
                    return True
                # Multi-variable indirect gather: >=2 lane-variant
                # gathered index components in one access -> no gather
                # intrinsic exists, so a masked remainder must degrade
                # to a scalar tail (a single-index gather still
                # collapses to the R2 masked-gather intrinsic and is
                # left alone).
                if e.data.subset is not None and cls._lane_variant_indirect_dim_count(inner, e.data.subset,
                                                                                      iter_var) >= 2:
                    return True
        return False

    def _attach_mask(self,
                     nsdfg_node: dace.nodes.NestedSDFG,
                     iter_var: str,
                     lb,
                     ub,
                     W: int,
                     global_ub: Optional[str] = None) -> bool:
        """Add and fill the ``_iter_mask`` transient inside one body NestedSDFG.

        :param nsdfg_node: The NestedSDFG node whose inner SDFG receives the mask.
        :param iter_var: The map's innermost loop parameter name.
        :param lb: Symbolic lower bound of the innermost range.
        :param ub: Symbolic upper bound of the innermost range.
        :param W: Number of lanes in the mask.
        :param global_ub: ``mode='global'`` (SVE-style) only. When given,
            the fill is keyed to the *running loop variable* against the
            *global* exclusive bound — ``mask[l] = (iter_var + l < global_ub)``
            — rather than the map's static ``lb``/``ub``. The per-core
            block spans multiple W-tiles, so after the vectorizer tiles
            the map (W-step outer + length-W inner) the body NSDFG runs
            once per W-tile and ``iter_var`` (the W-step outer param) is
            in scope: the mask correctly re-evaluates per tile so the
            ragged last-block cutoff is honoured. (The static-``lb``
            form below is for the single-trailing-block masked
            remainder, where a per-tile refill would be wrong.)
        :returns: ``True`` if a mask was added, ``False`` if one already existed.
        """
        inner: dace.SDFG = nsdfg_node.sdfg
        # Idempotency, skip if a mask is already attached.
        if any(name.startswith("_iter_mask") for name in inner.arrays):
            return False

        mask_name = add_mask(inner, "_iter_mask", W)

        # Mask fill formula uses the map's STATIC start value (``lb``) rather
        # than the dynamic ``iter_var``. Reason: after Vectorize tiles the
        # map (W-step outer + step-1 length-W inner), the body NSDFG runs
        # per-inner-iteration; if the formula referenced the loop param it
        # would re-fill the mask 8x with shifting values. Using ``lb`` (a
        # symbolic expression in the outer-scope symbols, e.g. ``8*floor(N/8)``
        # for the masked remainder) makes the formula invariant — same value
        # on every inner iteration. For step-W trip-1 maps (the legacy
        # ``step_w_only`` path), ``lb == iter_var`` at runtime so this is
        # backward compatible.
        if global_ub is not None:
            # SVE-style: per-tile cutoff against the global trip, keyed to
            # the running loop variable (see the global_ub docstring).
            key_str = str(iter_var)
            bound_str = str(global_ub)
            cmp = "<"
            bound_syms = [str(s) for s in dace.symbolic.symlist(dace.symbolic.pystr_to_symbolic(bound_str)).values()]
        else:
            # Map's STATIC start value (``lb``) rather than the dynamic
            # ``iter_var``. Reason: after Vectorize tiles the map (W-step
            # outer + step-1 length-W inner), the body NSDFG runs
            # per-inner-iteration; referencing the loop param would re-fill
            # the mask 8x with shifting values. ``lb`` (e.g. ``8*floor(N/8)``
            # for the masked remainder) is invariant across inner
            # iterations. For step-W trip-1 maps (legacy ``step_w_only``),
            # ``lb == iter_var`` at runtime so this is backward compatible.
            key_str = str(lb)
            bound_str = str(ub)
            cmp = "<="
            bound_syms = [str(s) for s in dace.symbolic.symlist(ub).values()]

        # Ensure every free symbol in the key/bound is visible inside the
        # inner NSDFG. The map's own ``iter_var`` is also kept in the symbol
        # set for backward compatibility with callers that might rely on it.
        symbols_to_ensure = ([iter_var] + bound_syms + [str(s) for s in dace.symbolic.symlist(lb).values()])
        for sname in symbols_to_ensure:
            if sname not in inner.symbols and sname in nsdfg_node.sdfg.parent_sdfg.symbols:
                inner.add_symbol(sname, nsdfg_node.sdfg.parent_sdfg.symbols[sname])
            elif sname not in inner.symbols:
                # iter_var is typically not in parent_sdfg.symbols (it's a map
                # parameter, scoped to the map entry). Default to int64.
                inner.add_symbol(sname, dace.int64)
            if sname not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[sname] = sname

        # Capture the current start block BEFORE prepending the new prep
        # state, otherwise ``inner.start_block`` becomes ambiguous between
        # the existing start and the new node.
        old_start = inner.start_block

        # Each lane ``l`` sets ``mask[l] = (key + l <cmp> bound)``: the
        # default form is ``(lb + l <= ub)``; the SVE-style global form is
        # ``(iter_var + l < global_ub)``.
        body = "\n".join([f"_o[{l}] = (({key_str}) + {l} {cmp} ({bound_str}));" for l in range(W)])
        prep = inner.add_state("_iter_mask_init", is_start_block=True)
        an = prep.add_access(mask_name)
        t = prep.add_tasklet(
            name="_iter_mask_fill",
            inputs=set(),
            outputs={"_o"},
            code=body,
            language=dace.dtypes.Language.CPP,
        )
        prep.add_edge(t, "_o", an, None, dace.Memlet(f"{mask_name}[0:{W}]"))
        if old_start is not None and old_start is not prep:
            inner.add_edge(prep, old_start, dace.InterstateEdge())
        return True
