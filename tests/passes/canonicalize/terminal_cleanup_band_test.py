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

Then the state-machine tidy-up -- ``_inline_single_state`` + ``_structural_cleanup``, the recipe's
own between-phase helpers -- which splices out the scaffolding earlier stages leave behind, such as
``BreakAntiDependence``'s spent ``*_antidep_prologue`` state.

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
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.canonicalize import pipeline as canon_pipeline
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
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
def _war_unit(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    """TSVC ``s121``: ``a[i] = a[i+1] + b[i]``. ``BreakAntiDependence`` snapshot-renames ``a`` to
    lift it, and leaves a spent ``*_antidep_prologue`` state behind."""
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


def _is_reclaim_stage(unit: ppl.Pass) -> bool:
    """The stage under test: the reclaimers' fixpoint pipeline, never a ``SimplifyPass``."""
    if not isinstance(unit, ppl.FixedPointPipeline) or isinstance(unit, SimplifyPass):
        return False
    return unit._pass_names == {DeadDataflowElimination.__name__, ArrayElimination.__name__}


#: The terminal band, in order: the reclaimers, then the inline that leads every cleanup site, then
#: the three units of :func:`_structural_cleanup`. Spelled out so the A/B below cannot silently start
#: skipping the wrong passes if the band is reordered.
_TERMINAL_BAND = ('FixedPointPipeline', 'PatternMatchAndApplyRepeated', 'PatternMatchAndApplyRepeated',
                  'EmptyStateElimination', 'DeadStateElimination')


def _band_start() -> int:
    """Index of the reclaim pipeline in the flat recipe -- the head of the terminal band."""
    at = [i for i, (_lbl, p) in enumerate(canon_pipeline._build_stages()) if _is_reclaim_stage(p)]
    assert len(at) == 1, f'expected exactly one reclaim stage, found {at}'
    return at[0]


def _canonicalize(sdfg: dace.SDFG, with_reclaim: bool = True, with_cleanup: bool = True) -> dace.SDFG:
    """Run the real recipe, optionally with either half of the terminal band skipped (the A/B
    reference). ``with_reclaim`` drops the two reclaimers; ``with_cleanup`` drops the inline plus
    the structural cleanup that follows them."""
    canon_pipeline.disable_openmp_sections(sdfg)
    start = _band_start()
    cleanup_slots = range(start + 1, start + len(_TERMINAL_BAND))
    for index, (_label, unit) in enumerate(canon_pipeline._build_stages()):
        if not with_reclaim and index == start:
            continue
        if not with_cleanup and index in cleanup_slots:
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


def test_the_terminal_band_is_wired_in_once_and_holds_no_simplify():
    stages = canon_pipeline._build_stages()
    labels = [label for label, unit in stages if _is_reclaim_stage(unit)]
    assert labels == ['end'], labels
    simplifies = [label for label, unit in stages if isinstance(unit, SimplifyPass)]
    assert simplifies == ['clean', 'reduce', 'reduce'], simplifies

    start = _band_start()
    band = [type(unit).__name__ for _label, unit in stages[start:start + len(_TERMINAL_BAND)]]
    assert tuple(band) == _TERMINAL_BAND, band
    assert all(label == 'end' for label, _unit in stages[start:start + len(_TERMINAL_BAND)])


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


#: The orphaned replica the fission leaves behind: the staging array, and the tasklets that read it
#: uninitialized and write it back for nobody.
_DEAD_REPLICA_ARRAY = 'nested_sdfg_a'
_DEAD_REPLICA_TASKLETS = {
    '_Add_',
    '_assign_in_nested_sdfg_a_to_nested_sdfg_a_index',
    '_assign_out_nested_sdfg_a_slice_plus_x_slice_to_nested_sdfg_a',
}


def test_fission_replica_dead_chain_is_removed():
    """The orphaned ``SplitStatements`` replica -- array, its access nodes, and its tasklets.

    Named exactly, not by prefix: the LIVE half of the split keeps ``nested_sdfg_a_index`` and
    ``nested_sdfg_a_slice_plus_x_slice``, so a prefix test would demand the reclaimers delete
    working dataflow.
    """
    reference = _canonicalize(_fission_dep_then_indep.to_sdfg(simplify=False), with_reclaim=False)
    assert _DEAD_REPLICA_ARRAY in _transients(reference), \
        f'expected the orphaned replica in the reference, got {_transients(reference)}'
    assert _DEAD_REPLICA_ARRAY in _access_nodes(reference)
    assert _DEAD_REPLICA_TASKLETS <= set(_tasklets(reference)), _tasklets(reference)

    cleaned = _canonicalize(_fission_dep_then_indep.to_sdfg(simplify=False), with_reclaim=True)
    assert _DEAD_REPLICA_ARRAY not in _transients(cleaned), _transients(cleaned)
    assert _DEAD_REPLICA_ARRAY not in _access_nodes(cleaned)
    assert not (_DEAD_REPLICA_TASKLETS & set(_tasklets(cleaned))), _tasklets(cleaned)
    # Only the dead chain goes: the live half of the split is untouched.
    assert set(_transients(reference)) - set(_transients(cleaned)) == {_DEAD_REPLICA_ARRAY}
    assert set(_tasklets(reference)) - set(_tasklets(cleaned)) == _DEAD_REPLICA_TASKLETS
    assert _maps(cleaned) == _maps(reference), 'reclaiming must not change the map structure'


def _states(sdfg: dace.SDFG) -> List[str]:
    return sorted(state.label for nested in sdfg.all_sdfgs_recursive() for state in nested.states())


def test_spent_antidependence_prologue_state_is_spliced_out():
    """The cleanup half of the band: state-machine scaffolding, not dataflow.

    ``BreakAntiDependence`` stages the snapshot in its own ``*_antidep_prologue`` state; once the
    lift is done that state is spent, and the terminal ``SimplifyPass`` was the last thing that ran
    ``FuseStates`` over it. ``_structural_cleanup`` at ``end`` is what takes its place.
    """
    reference = _canonicalize(_war_unit.to_sdfg(simplify=False), with_cleanup=False)
    prologues = [s for s in _states(reference) if 'antidep_prologue' in s]
    assert prologues, f'expected the spent prologue in the reference, got {_states(reference)}'

    cleaned = _canonicalize(_war_unit.to_sdfg(simplify=False), with_cleanup=True)
    assert [s for s in _states(cleaned) if 'antidep_prologue' in s] == [], _states(cleaned)
    assert len(_states(cleaned)) < len(_states(reference))
    # The lift itself must survive the tidy-up: the WAR loop is still a Map.
    assert _maps(cleaned) == _maps(reference) and _maps(cleaned) > 0


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
