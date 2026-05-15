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
                               "``__masked_rem`` by SplitMapForVectorRemainder(mode='masked'). The "
                               "mode names match the ``VectorizeCPU.remainder_strategy`` knob.")

    def __init__(self, vector_width: int = 8, mode: str = "step_w_only"):
        """Initialize the pass.

        :param vector_width: Number of lanes in the mask.
        :param mode: ``step_w_only`` masks maps with step==``vector_width``;
            ``all_innermost`` masks every innermost map; ``masked`` masks
            maps tagged ``__masked_rem``.
        """
        super().__init__()
        self.vector_width = vector_width
        self.mode = mode

    def modifies(self) -> ppl.Modifies:
        """Return the set of SDFG elements this pass may modify."""
        return ppl.Modifies.Descriptors | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Return whether the pass should run again after modifications."""
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Attach the iteration mask to every targeted innermost map body.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of masks attached, or ``None`` if none.
        :raises ValueError: If ``mode`` is not a recognised mode.
        :raises NotImplementedError: If a targeted map body is not a single NestedSDFG.
        """
        if self.mode not in ("step_w_only", "all_innermost", "masked"):
            raise ValueError(f"GenerateIterationMask.mode must be 'step_w_only', 'all_innermost', or "
                             f"'masked', got {self.mode!r}")
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
            nsdfg_node = get_single_nsdfg_inside_map(g, n)
            if nsdfg_node is None:
                raise NotImplementedError(f"GenerateIterationMask requires every innermost map's body to be a single "
                                          f"NestedSDFG (run NestInnermostMapBodyIntoNSDFG first); map {n.label!r} has "
                                          f"a bare-tasklet body")
            if self._attach_mask(nsdfg_node, n.map.params[-1], lb, ub, W):
                applied += 1
        return applied or None

    def _attach_mask(self, nsdfg_node: dace.nodes.NestedSDFG, iter_var: str, lb, ub, W: int) -> bool:
        """Add and fill the ``_iter_mask`` transient inside one body NestedSDFG.

        :param nsdfg_node: The NestedSDFG node whose inner SDFG receives the mask.
        :param iter_var: The map's innermost loop parameter name.
        :param lb: Symbolic lower bound of the innermost range.
        :param ub: Symbolic upper bound of the innermost range.
        :param W: Number of lanes in the mask.
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
        lb_str = str(lb)
        ub_str = str(ub)

        # Ensure every free symbol in ``lb`` and ``ub`` is visible inside the
        # inner NSDFG. The map's own ``iter_var`` is also kept in the symbol
        # set for backward compatibility with callers that might rely on it.
        symbols_to_ensure = ([iter_var] + [str(s) for s in dace.symbolic.symlist(ub).values()] +
                             [str(s) for s in dace.symbolic.symlist(lb).values()])
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

        # Each lane ``l`` sets ``mask[l] = (lb + l <= ub)``.
        body = "\n".join([f"_o[{l}] = (({lb_str}) + {l} <= ({ub_str}));" for l in range(W)])
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
