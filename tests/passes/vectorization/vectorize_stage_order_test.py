# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering invariants of the vectorizer pipeline, and its entry canonicalization.

The canonicalize recipe used to own ``LowerITEToFpFactor`` and ``SplitTasklets``; the ordering
constraints against them lived in ``tests/canonicalize/canonicalize_pipeline_stage_order_test.py``. Both
passes now live HERE, so the constraints do too -- otherwise they are asserted in a file where
their subject no longer exists, which is a test that passes by skipping.
"""
import pytest

from dace.transformation.passes.canonicalize.pipeline import IvSubstitutionFissionFixpoint
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_multi_dim import (VectorizeCPUMultiDim,
                                                                          vectorization_prep_units)


def _pass_names(**knobs) -> list:
    """The vectorizer's pipeline as a flat list of pass class names."""
    config = VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR, **knobs)
    return [type(p).__name__ for p in VectorizeCPUMultiDim(config).passes]


@pytest.mark.parametrize('branch_mode', [BranchMode.MERGE, BranchMode.FP_FACTOR])
def test_split_tasklets_runs_in_every_branch_mode(branch_mode):
    """``SplitTasklets`` is the tile emitter's precondition -- one op per tasklet, so
    ``ConvertTaskletsToTileOps`` can classify each body statement. Asserted as PRESENCE: it is
    what the canonicalize recipe hands off, and a drop here would silently leave multi-op
    tasklets for the converter to refuse."""
    assert 'SplitTasklets' in _pass_names(branch_mode=branch_mode)


def test_fp_factor_lowering_precedes_the_tasklet_split():
    """``LowerITEToFpFactor`` folds ``ITE(c, t, e)`` into the multi-op ``c*t + (1-c)*e``, which
    ``SplitTasklets`` then breaks into single-op binops the tile emitter turns into ``TileBinop``.
    Splitting first would leave the fp-factor arithmetic fused in one tasklet."""
    names = _pass_names(branch_mode=BranchMode.FP_FACTOR, remainder_strategy=RemainderStrategy.MASKED_TAIL)
    assert 'LowerITEToFpFactor' in names, 'fp_factor mode without its ITE lowering'
    assert names.index('LowerITEToFpFactor') < names.index('SplitTasklets')


def test_merge_mode_has_no_fp_factor_lowering():
    """Merge mode lowers a same-write-set if/else to a per-lane ``TileITE`` select, so the
    fp-factor fold must NOT run -- it would rewrite the ITE the select is built from."""
    assert 'LowerITEToFpFactor' not in _pass_names(branch_mode=BranchMode.MERGE)


def _prep_pass_names() -> list:
    """Every pass the entry prep runs, composites expanded to their members."""
    names = []
    for unit in vectorization_prep_units():
        names.append(type(unit).__name__)
        members = unit.round_units() if isinstance(unit, IvSubstitutionFissionFixpoint) else unit.units()
        names += [type(m).__name__ for m in members]
    return names


def test_the_entry_prep_substitutes_induction_variables():
    """The prep must close induction variables. While an IV is live every statement in the body
    reads the same counter, so the body is one dependence component: the statement fission the
    tile emitter needs is illegal and no per-lane widening can proceed. Asserted as presence --
    a caller arriving from a bare ``LoopToMap`` + ``simplify`` has run no such pass."""
    assert 'InductionVariableSubstitution' in _prep_pass_names()


def test_the_entry_prep_does_not_canonicalize():
    """The prep is STRUCTURAL, not the canonicalize recipe. The documented order is canonicalize
    (or ParallelizeLoops) -> vectorize, so re-deriving the canonical shape here pays for it twice.

    Two members of the recipe would additionally be wrong to run: a semantic lift hands the tiler
    a library node with no per-lane body to widen, and ``ShortLoopUnroll`` straight-lines a short
    constant-trip loop, deleting the very map the tiler was called to widen."""
    names = _prep_pass_names()
    for recipe_only in ('ShortLoopUnroll', 'LiftInv', 'LoopToSymm', 'ParallelizeLoops',
                        'AssignmentAndCopyKernelToMemsetAndMemcpy'):
        assert recipe_only not in names, f'{recipe_only} is the caller\'s recipe, not the entry prep'


def test_the_semantic_lifts_do_not_run_inside_the_vectorizer():
    """Whatever canon runs, the vectorizer's own pipeline contains no map -> library-node lift:
    the tile path needs raw maps."""
    names = _pass_names()
    for lift in ('LiftInv', 'LoopToSymm', 'AssignmentAndCopyKernelToMemsetAndMemcpy'):
        assert lift not in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
