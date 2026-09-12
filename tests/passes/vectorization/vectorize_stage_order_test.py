# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering invariants of the vectorizer pipeline, and its entry canonicalization.

The canonicalize recipe used to own ``LowerITEToFpFactor`` and ``SplitTasklets``; the ordering
constraints against them lived in ``tests/canonicalize/canonicalize_pipeline_stage_order_test.py``. Both
passes now live HERE, so the constraints do too -- otherwise they are asserted in a file where
their subject no longer exists, which is a test that passes by skipping.
"""
import warnings

import pytest

import dace
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.transformation.passes.canonicalize.pipeline import IvSubstitutionFissionFixpoint
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode, ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_multi_dim import (EMITTABLE_TILE_NODE_TYPES,
                                                                          VectorizeCPUMultiDim,
                                                                          vectorization_prep_units)

N_SYM = dace.symbol('N')


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


def fill_lifted_memset(name: str) -> dace.SDFG:
    """``for i: A[i, 0:N] = 0.0`` with the row store already a ``FillLibraryNode`` -- the shape
    canonicalize's ``lift_copy`` stage leaves behind for a pure zero-init nest."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N_SYM, N_SYM], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('rows', {'i': '0:N'})
    node = FillLibraryNode(name='memsetLib_A_0')
    state.add_node(node)
    state.add_nedge(entry, node, dace.Memlet())
    state.add_memlet_path(node,
                          exit_node,
                          state.add_write('A'),
                          src_conn=FillLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet('A[i, 0:N]'))
    return sdfg


def elementwise_copy(name: str) -> dace.SDFG:
    """``for i: B[i] = A[i] * 2`` -- an ordinary per-lane body the tiler does widen."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N_SYM], dace.float64)
    sdfg.add_array('B', [N_SYM], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('lanes', {'i': '0:N'})
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    return sdfg


def vectorize_and_collect_warnings(sdfg: dace.SDFG) -> list[str]:
    """Run the CPU tile orchestrator over ``sdfg`` and return the warning messages it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    return [str(w.message) for w in caught]


def test_a_kernel_the_tiler_leaves_untouched_says_so():
    """A map whose body is an opaque library node passes no tile-candidate gate, so every tile
    pass skips it and the orchestrator emits nothing -- and it must SAY it emitted nothing. The
    silence read exactly like a successful vectorization, which is how a caller ended up comparing
    a kernel against itself."""
    messages = vectorize_and_collect_warnings(fill_lifted_memset('untiled_memset'))
    assert any('tiled nothing' in m for m in messages), messages
    # Not a refusal: nothing was restored, and the callers that audit refusals grep that phrase.
    assert not any('refusing to vectorize' in m for m in messages), messages


def test_a_kernel_that_does_tile_stays_quiet():
    """The empty-bracket control for the assertion above: bracket a kernel the tiler DOES widen
    and the same counter must read zero, or 'tiled nothing' proves nothing."""
    sdfg = elementwise_copy('tiled_scale')
    messages = vectorize_and_collect_warnings(sdfg)
    assert not any('tiled nothing' in m for m in messages), messages
    assert any(isinstance(node, EMITTABLE_TILE_NODE_TYPES) for node, _ in sdfg.all_nodes_recursive())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
