# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize an SDFG into the body shape the loop-lifting passes match on.

:class:`~dace.transformation.passes.loop_to_scan.LoopToScan` and
:class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce` are pure matchers: they
rewrite only what they recognise and report ``None`` -- "did not modify the SDFG" -- when
they recognise nothing. Their matchers nevertheless need a canonical body: WCR edges
spelled as in-body augmented assignment, no frontend ``__out = __inp`` copy tasklet
between an accumulator and its update, forward iteration, and one content state per loop
body. This pass performs exactly that normalization, so the lifters themselves stay
side-effect free on a no-match.

The canonicalization pipeline wires it in ahead of every lift stage. A direct caller that
builds an SDFG straight from the Python frontend and wants the lifts must run it first.
"""
from typing import Optional, Set, Type

from dace import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


@xf.explicit_cf_compatible
class LiftPreprocess(ppl.Pass):
    """Canonicalize loop bodies for ``LoopToScan`` / ``LoopToReduce`` matching."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Normalize every loop body in ``sdfg`` (and nested SDFGs).

        :returns: The number of normalization rewrites, or ``None`` if the SDFG was
                  left untouched.
        """
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
        from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride
        from dace.transformation.passes.loop_to_scan import _collect_loops, _fuse_body_states
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.symbol_propagation import SymbolPropagation

        count = 0
        # Reductions written as WCR edges -> in-body augmented assignment, so the matchers
        # see one uniform tasklet shape. No-op on already-augassign bodies.
        applied = PatternMatchAndApplyRepeated([WCRToAugAssign()]).apply_pass(sdfg, {})
        count += sum(len(v) for v in applied.values()) if applied else 0
        # Strip frontend ``__out = __inp`` copy tasklets so the matcher sees the bare
        # ``out[i+1] = out[i] + delta[i]`` shape instead of an ``assign_NN`` copy node.
        applied = PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
        count += sum(len(v) for v in applied.values()) if applied else 0

        # NOTE: D4 (CleanAccessNode + CleanTasklet) is deliberately NOT applied here.
        # ``LoopToScan``'s matcher already handles the frontend's scalar-slice intermediates
        # via ``_chase_forward_to_accum`` and friends; the WCR/TrivialTasklet folds above are
        # sufficient. Running the clean folds in addition (in either order vs WCR) is
        # redundant work and previously regressed the ``for_1133_shape_reverse_engineered``
        # case by stripping intermediates the matcher relies on.

        # ``LoopToScan`` only handles ``stride == 1``; flip backward iteration up front rather
        # than teaching every gate about sign. ``NormalizeNegativeStride`` rebinds the old
        # iterator on the body entry iedge (``jm = N - _loop_pos_X``) rather than rewriting
        # body memlets, so follow up with ``SymbolPropagation`` to express the body subsets in
        # the new positive-stride iterator.
        flipped = NormalizeNegativeStride().apply_pass(sdfg, {})
        if flipped:
            count += flipped
            propagated = SymbolPropagation().apply_pass(sdfg, {})
            count += len(propagated) if propagated else 0

        # Fold adjacent content states inside a loop body: whole-SDFG ``StateFusion`` does not
        # reach into LoopRegion bodies via ``MatchPatterns``, and the matchers want one.
        for loop, _, _ in _collect_loops(sdfg):
            count += _fuse_body_states(loop)
        return count or None
