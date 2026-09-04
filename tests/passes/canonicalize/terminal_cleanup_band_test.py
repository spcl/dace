# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonicalization pipeline's terminal band -- what replaced the terminal ``SimplifyPass``.

The recipe used to end with a full ``SimplifyPass``, which meant every stage after it could lean on
re-inlining, state fusion and copy-chain folding without saying so. What the recipe actually needed
from it is now wired explicitly, in two halves.

The reclaimers, as one fixpoint pipeline:

* ``DeadDataflowElimination`` -- ``SplitStatements`` replicates a statement per independent output,
  and the fission / parallelize stages downstream of the ``reduce`` simplifies orphan a replica
  whose output nothing consumes. The chain (an uninitialized read feeding a write nobody reads
  back) reaches codegen as a real allocation plus real tasklets.
* ``ArrayElimination`` -- ``MapFusionVertical`` mints a fresh ``__map_fusion_<x>`` carrier per fused
  edge, so a value consumed by two fused consumers gets a SECOND carrier that is a plain copy of the
  first. The post-fuse ``RedundantArray`` cannot fold it (it redirects a transient's WRITERS, which
  is impossible while the source still has another reader), and the mirrored ``RedundantSecondArray``
  is unsafe to run bare.

Then the state-machine tidy-up -- ``_inline_single_state`` + ``_structural_cleanup`` plus
``PruneEmptyConditionalBranches``, which splice out the scaffolding earlier stages leave behind,
such as the empty ``else`` arm ``LoopToScan`` leaves when it splits a masked scan.

Every behavioural test here is A/B in one test -- with the half under test and without it -- so none
can go vacuous if a later change makes the residue disappear for some other reason.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import List

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.canonicalize import pipeline as canon_pipeline
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches
from dace.transformation.passes.simplify import SimplifyPass

LEN_1D = dace.symbol('LEN_1D')


@dace.program
def _fuse_diamond(out: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    """One producer ``t = a*a`` feeding two consumers that rejoin -- the shape that mints the
    duplicate ``__map_fusion_t`` carrier when the diamond collapses to one map."""
    t = np.empty(LEN_1D, dtype=np.float64)
    u = np.empty(LEN_1D, dtype=np.float64)
    v = np.empty(LEN_1D, dtype=np.float64)
    for i in dace.map[0:LEN_1D]:
        t[i] = a[i] * a[i]
    for i in dace.map[0:LEN_1D]:
        u[i] = t[i] + 1.0
    for i in dace.map[0:LEN_1D]:
        v[i] = t[i] - 1.0
    for i in dace.map[0:LEN_1D]:
        out[i] = u[i] * v[i]


@dace.program
def _fission_dep_then_indep(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], x: dace.float64[LEN_1D],
                            y: dace.float64[LEN_1D]):
    """A carried prefix sum beside an independent body -- fission splits them, and the split leaves
    a replica of the ``a`` staging chain that nothing downstream consumes."""
    a[0] = x[0]
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + x[i]
        b[i] = y[i] * 2.0


@dace.program
def _guarded_scan(out: dace.float64[LEN_1D], delta: dace.float64[LEN_1D], mask: dace.int64[LEN_1D]):
    """TSVC-2.5 ``scan_conditional``: a masked prefix sum, where the running total advances only
    where the mask is set. ``LoopToScan`` splits the guard into a delta build plus a scan, which
    leaves the ``else`` arm -- ``out[i] = out[i - 1]``, the additive identity -- with nothing left
    to do. Caller seeds ``out[0]``."""
    for i in range(1, LEN_1D):
        if mask[i] > 0:
            out[i] = out[i - 1] + delta[i]
        else:
            out[i] = out[i - 1]


def _is_reclaim_stage(unit: ppl.Pass) -> bool:
    """The stage under test: the reclaimers' fixpoint pipeline, never a ``SimplifyPass``."""
    if not isinstance(unit, ppl.FixedPointPipeline) or isinstance(unit, SimplifyPass):
        return False
    return unit._pass_names == {DeadDataflowElimination.__name__, ArrayElimination.__name__}


#: Unit types of :func:`_structural_cleanup`, taken from the helper itself rather than transcribed:
#: the A/B below has to skip exactly the cleanup, and a transcribed list goes stale silently.
_CLEANUP_TYPES = frozenset(type(p).__name__ for _lbl, p in canon_pipeline._structural_cleanup('probe'))


def _band_start() -> int:
    """Index of the reclaim pipeline in the flat recipe -- the head of the terminal band."""
    at = [i for i, (_lbl, p) in enumerate(canon_pipeline._build_stages()) if _is_reclaim_stage(p)]
    assert len(at) == 1, f'expected exactly one reclaim stage, found {at}'
    return at[0]


def _leads_a_cleanup(unit) -> bool:
    """Whether ``unit`` is the ``PruneConnectors`` + ``InlineSDFG`` fixpoint that leads a cleanup
    site (an un-inlined body hides its per-element memlets behind a whole-array boundary memlet, so
    the cleanup would read the wrong shape)."""
    return (isinstance(unit, PatternMatchAndApplyRepeated)
            and any(isinstance(t, InlineSDFG) for t in unit.transformations))


def _cleanup_slots() -> List[int]:
    """Recipe indices of the terminal cleanup: the structural-cleanup helper, the inline that leads
    it, and the empty-arm prune.

    Selected by ROLE, not as a slice after the reclaimers. The terminal optimization tail --
    ``LoopToMap``, the fusions, ``RedundantArray``, remat -- now sits between the reclaimers and the
    cleanup they precede, so a fixed-width window would skip that tail instead and the A/B would be
    measuring something else entirely while still passing.
    """
    stages = canon_pipeline._build_stages()
    start = _band_start()
    names = [type(p).__name__ for _lbl, p in stages]
    want = [type(p).__name__ for _lbl, p in canon_pipeline._structural_cleanup('probe')]
    # The FIRST full occurrence of the helper after the reclaimers -- matched as a run, not by
    # membership: the recipe's tail holds further symbol passes of the same types, and picking
    # those up would make the A/B skip work that is not cleanup at all.
    at = [i for i in range(start + 1, len(names) - len(want) + 1) if names[i:i + len(want)] == want]
    assert at, 'no structural cleanup after the reclaimers -- re-home this A/B with it'
    helper = list(range(at[0], at[0] + len(want)))
    slots = set(helper)
    slots.update(i for i, (_lbl, p) in enumerate(stages)
                 if start < i <= helper[-1] and isinstance(p, PruneEmptyConditionalBranches))
    leading = [i for i, (_lbl, p) in enumerate(stages) if start < i < helper[0] and _leads_a_cleanup(p)]
    assert leading, 'the terminal cleanup is no longer led by an inline'
    slots.add(leading[-1])
    return sorted(slots)


def _canonicalize(sdfg: dace.SDFG,
                  with_reclaim: bool = True,
                  with_cleanup: bool = True,
                  with_optional: bool = True) -> dace.SDFG:
    """Run the real recipe, optionally with one part of the terminal work skipped (the A/B
    reference). ``with_reclaim`` drops the two reclaimers; ``with_cleanup`` drops the inline, the
    structural cleanup and the empty-arm prune that follow them; ``with_optional`` drops the
    terminal ``OptionalArrayInference``, which sits later, beside the symbol cleanups."""
    canon_pipeline.disable_openmp_sections(sdfg)
    start = _band_start()
    cleanup_slots = _cleanup_slots()
    for index, (_label, unit) in enumerate(canon_pipeline._build_stages()):
        if not with_reclaim and index == start:
            continue
        if not with_cleanup and index in cleanup_slots:
            continue
        if not with_optional and isinstance(unit, OptionalArrayInference):
            continue
        unit.apply_pass(sdfg, {})
    canon_pipeline.disable_openmp_sections(sdfg)
    sdfg.validate()
    return sdfg


def _transients(sdfg: dace.SDFG) -> List[str]:
    return sorted(name for nested in sdfg.all_sdfgs_recursive() for name, desc in nested.arrays.items()
                  if desc.transient)


def _access_nodes(sdfg: dace.SDFG) -> List[str]:
    return sorted(n.data for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode))


def _tasklets(sdfg: dace.SDFG) -> List[str]:
    return sorted(n.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))


def _maps(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _units_of(unit) -> List[object]:
    """``unit`` and the passes it composes, so a nested ``SimplifyPass`` is not invisible here."""
    if isinstance(unit, canon_pipeline.IvSubstitutionFissionFixpoint):
        return [unit, *unit.round_units()]
    return [unit]


def test_the_terminal_band_is_wired_in_once_and_holds_no_simplify():
    stages = canon_pipeline._build_stages()
    labels = [label for label, unit in stages if _is_reclaim_stage(unit)]
    assert labels == ['end'], labels
    # Look INSIDE a composite stage as well: the IV/fission fixpoint owns a ``SimplifyPass`` of
    # its own, and counting only top-level entries would let a simplify move into the terminal
    # band inside a wrapper without this noticing -- which is the whole property under test.
    simplifies = [label for label, unit in stages for _p in _units_of(unit) if isinstance(_p, SimplifyPass)]
    assert simplifies == ['clean', 'reduce', 'reduce'], simplifies

    start = _band_start()
    slots = _cleanup_slots()
    assert all(stages[i][0] == 'end' for i in slots), [stages[i][0] for i in slots]
    assert stages[start][0] == 'end', stages[start][0]
    # Nothing from the reclaimers onward may re-introduce a whole pipeline: the band exists
    # precisely because the recipe used to end with one.
    after = [type(p).__name__ for _lbl, unit in stages[start:] for p in _units_of(unit) if isinstance(p, SimplifyPass)]
    assert not after, after
    want = [type(p).__name__ for _lbl, p in canon_pipeline._structural_cleanup('probe')]
    run = [type(stages[i][1]).__name__ for i in slots if type(stages[i][1]).__name__ in _CLEANUP_TYPES]
    assert run == want, run


def test_fused_diamond_loses_the_duplicate_map_fusion_carrier():
    """Without the stage the collapsed diamond carries ``__map_fusion_t`` AND a copy of it."""
    reference = _canonicalize(_fuse_diamond.to_sdfg(simplify=False), with_reclaim=False)
    dupes = [n for n in _transients(reference) if n.startswith('__map_fusion_t')]
    assert len(dupes) == 2, f'expected the duplicated carrier in the reference, got {_transients(reference)}'
    copy_edges = [(e.src.data, e.dst.data) for nested in reference.all_sdfgs_recursive() for state in nested.states()
                  for e in state.edges() if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)]
    assert copy_edges, 'reference should still hold the AccessNode->AccessNode copy'

    cleaned = _canonicalize(_fuse_diamond.to_sdfg(simplify=False), with_reclaim=True)
    assert len([n for n in _transients(cleaned) if n.startswith('__map_fusion_t')]) == 1, _transients(cleaned)
    assert not [(e.src.data, e.dst.data) for nested in cleaned.all_sdfgs_recursive() for state in nested.states()
                for e in state.edges() if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)]
    # The reclaim is a pure removal: it must not cost the fusion the diamond earned.
    assert _maps(cleaned) == _maps(reference), 'reclaiming must not change the map structure'


#: The orphaned replica the fission leaves behind: the staging array it reads uninitialized, the two
#: per-expression descriptors that chain stages through, and the tasklets that write it for nobody.
#: All three descriptors carry the replica's ``nested_sdfg_`` prefix; the live half of the split
#: works on ``a_slice_plus_x_slice`` and the ``_scan_*`` pair, so the sets never overlap.
_DEAD_REPLICA_ARRAY = 'nested_sdfg_a'
_DEAD_REPLICA_TRANSIENTS = {
    'nested_sdfg_a',
    'nested_sdfg_a_index',
    'nested_sdfg_a_slice_plus_x_slice',
}
_DEAD_REPLICA_TASKLETS = {
    '_Add_',
    '_assign_in_nested_sdfg_a_to_nested_sdfg_a_index',
    '_assign_out_nested_sdfg_a_slice_plus_x_slice_to_nested_sdfg_a',
}
#: What the split's live half computes -- named so the removal below cannot quietly take it too.
_LIVE_TRANSIENTS = {'_scan_in_a', '_scan_seed_a', 'a_slice_plus_x_slice'}


def test_fission_replica_is_absent_from_the_canonical_form():
    """No orphaned ``SplitStatements`` replica survives -- and the live half is untouched.

    This used to assert that the reclaim band REMOVED the replica, with the un-reclaimed reference
    carrying it as the precondition. The split no longer emits it: the reference now holds exactly
    ``_LIVE_TRANSIENTS`` and none of the dead names, so the reclaimers have nothing to take here
    and the old precondition is unreachable. The property worth pinning is the end state, asserted
    on BOTH arms so it holds whether the chain is never created or created and collected -- and
    paired with the live set so it cannot pass by deleting everything. The reclaim band keeps its
    own coverage in ``test_fused_diamond_loses_the_duplicate_map_fusion_carrier``, where a
    duplicate carrier IS still produced.
    """
    for with_reclaim in (False, True):
        sdfg = _canonicalize(_fission_dep_then_indep.to_sdfg(simplify=False), with_reclaim=with_reclaim)
        transients, nodes_, tasklets = set(_transients(sdfg)), _access_nodes(sdfg), set(_tasklets(sdfg))
        assert not (_DEAD_REPLICA_TRANSIENTS & transients), (with_reclaim, sorted(transients))
        assert _DEAD_REPLICA_ARRAY not in nodes_, (with_reclaim, nodes_)
        assert not (_DEAD_REPLICA_TASKLETS & tasklets), (with_reclaim, sorted(tasklets))
        assert _LIVE_TRANSIENTS <= transients, (with_reclaim, sorted(transients))
    # Exact, not a subset: the canonical form is the live half and nothing else.
    assert set(_transients(sdfg)) == _LIVE_TRANSIENTS, sorted(_transients(sdfg))


def _states(sdfg: dace.SDFG) -> List[str]:
    return sorted(state.label for nested in sdfg.all_sdfgs_recursive() for state in nested.states())


def _workless_branches(sdfg: dace.SDFG) -> List[str]:
    """Conditional arms that carry no dataflow at all.

    Named by the arm rather than by a state, because that is the whole point of the case: an arm is
    a ControlFlowRegion, so ``DeadStateElimination`` walks past it however empty its states are.
    """
    return sorted(branch.label for nested in sdfg.all_sdfgs_recursive() for block in nested.all_control_flow_blocks()
                  if isinstance(block, ConditionalBlock) for _condition, branch in block.branches
                  if not any(state.nodes() for state in branch.all_states()))


def test_spent_conditional_arm_is_spliced_out():
    """The cleanup half of the band: state-machine scaffolding, not dataflow.

    The terminal ``SimplifyPass`` was the last thing that ran ``FuseStates`` / ``DeadStateElimination``
    over the recipe's output. ``_structural_cleanup`` plus ``PruneEmptyConditionalBranches`` at ``end``
    is what takes its place, and the empty arm is the half only the latter reaches -- the state-level
    passes see a ControlFlowRegion and step over it.
    """
    reference = _canonicalize(_guarded_scan.to_sdfg(simplify=False), with_cleanup=False)
    spent = _workless_branches(reference)
    assert spent, f'expected the split scan to leave an empty arm, got {_states(reference)}'

    cleaned = _canonicalize(_guarded_scan.to_sdfg(simplify=False), with_cleanup=True)
    assert _workless_branches(cleaned) == [], _states(cleaned)
    assert len(_states(cleaned)) < len(_states(reference))
    # The split itself must survive the tidy-up: the scan is still a Map, and the arm that was
    # pruned really was the identity.
    assert _maps(cleaned) == _maps(reference) and _maps(cleaned) > 0

    length = 64
    rng = np.random.default_rng(11)
    delta, mask = rng.random(length), (rng.random(length) > 0.5).astype(np.int64)
    expected = np.zeros(length)
    for i in range(1, length):
        expected[i] = expected[i - 1] + delta[i] if mask[i] > 0 else expected[i - 1]

    out = np.zeros(length)
    cleaned(out=out, delta=delta, mask=mask, LEN_1D=length)
    assert np.allclose(out, expected)


def test_the_stage_is_idempotent():
    """Re-entering the stage must find nothing -- canonicalize's output is already its fixed point."""
    sdfg = _canonicalize(_fuse_diamond.to_sdfg(simplify=False))
    before = (_transients(sdfg), _access_nodes(sdfg), _tasklets(sdfg))
    stage = [unit for _lbl, unit in canon_pipeline._build_stages() if _is_reclaim_stage(unit)][0]
    stage.apply_pass(sdfg, {})
    assert (_transients(sdfg), _access_nodes(sdfg), _tasklets(sdfg)) == before
    sdfg.validate()


@pytest.mark.parametrize('program,kwargs', [
    (_fuse_diamond, 'diamond'),
    (_fission_dep_then_indep, 'fission'),
])
def test_reclaim_is_bit_exact(program, kwargs):
    """A reclaimer that changes results is a miscompile: same pipeline with and without the stage."""
    size = 64
    rng = np.random.default_rng(20260812)
    # Drawn ONCE, before the loop: both runs must see byte-identical inputs or the comparison
    # measures the RNG, not the stage.
    base = {name: rng.random(size) + 0.5 for name in ('a', 'x', 'y')}

    results = []
    for with_reclaim in (False, True):
        sdfg = _canonicalize(program.to_sdfg(simplify=False), with_reclaim=with_reclaim)
        sdfg.name = f'{kwargs}_reclaim' if with_reclaim else f'{kwargs}_reference'
        if kwargs == 'diamond':
            out = np.zeros(size)
            sdfg.compile()(out=out, a=base['a'].copy(), LEN_1D=size)
            results.append(out)
        else:
            a, b = np.zeros(size), np.zeros(size)
            sdfg.compile()(a=a, b=b, x=base['x'].copy(), y=base['y'].copy(), LEN_1D=size)
            results.append(np.concatenate([a, b]))
    assert np.array_equal(results[1].view(np.uint64), results[0].view(np.uint64)), \
        'reclaiming changed the result'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
