# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering invariants of the vectorizer pipeline, and its entry canonicalization.

The canonicalize recipe used to own ``LowerITEToFpFactor`` and ``SplitTasklets``; the ordering
constraints against them lived in ``tests/canonicalize/canonicalize_pipeline_stage_order_test.py``. Both
passes now live HERE, so the constraints do too -- otherwise they are asserted in a file where
their subject no longer exists, which is a test that passes by skipping.
"""
import pytest

from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_multi_dim import (ENTRY_CANONICALIZE_KWARGS,
                                                                          VectorizeCPUMultiDim)


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


def test_entry_canonicalization_skips_the_semantic_lifts():
    """The vectorizer canonicalizes at its own entry, and must do so with
    ``semantic_lifting=False``.

    Two reasons, both hard requirements. A lifted ``Einsum`` / ``Copy`` / ``Memset`` library node
    has no per-lane body for the tiler to widen. And ``LiftInv`` -- one of the lifts -- matches the
    ``1 if i == j else 0`` identity tasklet that this pipeline's ``LowerITEToFpFactor`` rewrites
    into arithmetic; running the lift from inside the vectorizer would put it on the wrong side of
    that rewrite, the exact ordering the canonicalize recipe documents."""
    assert ENTRY_CANONICALIZE_KWARGS['semantic_lifting'] is False


def test_the_semantic_lifts_do_not_run_inside_the_vectorizer():
    """Whatever canon runs, the vectorizer's own pipeline contains no map -> library-node lift:
    the tile path needs raw maps."""
    names = _pass_names()
    for lift in ('LiftInv', 'LoopToSymm', 'AssignmentAndCopyKernelToMemsetAndMemcpy'):
        assert lift not in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
