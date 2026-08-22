# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the coalescing phase in :mod:`tests.corpus.cloudsc.pipelines`.

Fast and in-process. The phase-plan tests build plans only (no CloudSC parse, no pipeline run). The
effect tests use a small CloudSC-shaped fixture -- four element-wise ``(jl, jk)`` loop nests over
shared arrays -- driven through the ``parallelize`` variant's own phases up to, but not including,
``coalesce``. That is exactly the graph the phase is meant to see: every lifted loop body sits in its
own NestedSDFG, so the maps cannot fuse until the walls come down.

    pytest tests/corpus/cloudsc/cloudsc_pipeline_coalesce_test.py -v
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from tests.corpus.cloudsc.pipelines import _parallelize_phases as parallelize_phases
from tests.corpus.cloudsc.pipelines import (COALESCE_PHASE, COALESCE_VARIANTS, VARIANTS, pretreat_stages,
                                            specialize_stage, variant_phases)

KLON = dace.symbol('KLON')
KLEV = dace.symbol('KLEV')

#: Fixture shape -- small enough to compile in seconds, big enough that a wrong fusion shows up.
FIXTURE_KLON, FIXTURE_KLEV = 12, 7


@dace.program
def elementwise_chain(pt: dace.float64[KLON, KLEV], pq: dace.float64[KLON, KLEV], pa: dace.float64[KLON, KLEV],
                      tend_t: dace.float64[KLON, KLEV], tend_q: dace.float64[KLON, KLEV]):
    """Four CloudSC-shaped element-wise sweeps over the same buffers. The 1st and 3rd are a
    read-after-write chain on ``tend_t``, and the 3rd reads ``tend_q`` that the 4th overwrites -- so a
    fusion that ignores ordering would silently change the answer."""
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_t[jl, jk] = pt[jl, jk] * 2.0 + pq[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_q[jl, jk] = pq[jl, jk] * 0.5 - pa[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_t[jl, jk] = tend_t[jl, jk] + tend_q[jl, jk] * pa[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_q[jl, jk] = tend_q[jl, jk] * pt[jl, jk] + 1.0


def graph_counts(sdfg: dace.SDFG):
    """``(nested SDFGs, map entries)`` over the whole graph, nestings included."""
    nested = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.NestedSDFG))
    maps = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    return nested, maps


def phase_names(variant: str):
    return [name for name, _ in variant_phases(variant)]


def apply_phase(sdfg: dace.SDFG, name: str):
    """Apply the named phase of the ``parallelize`` plan; return the per-stage ``apply_pass`` returns."""
    stages = dict(variant_phases('parallelize'))[name]
    return [apply_fn(sdfg) for _label, apply_fn in stages]


@pytest.fixture(scope='module')
def parallelized():
    """The ``parallelize`` variant driven up to -- and stopping before -- the coalescing phase."""
    sdfg = elementwise_chain.to_sdfg(simplify=False)
    for name, stages in variant_phases('parallelize'):
        if name == COALESCE_PHASE:
            break
        for _label, apply_fn in stages:
            apply_fn(sdfg)
    sdfg.validate()
    return sdfg


def run_fixture(sdfg: dace.SDFG, tag: str):
    """Run a private copy of ``sdfg`` on fixed seeded inputs; return the mutated output buffers."""
    candidate = copy.deepcopy(sdfg)
    candidate.name = f'coalesce_fixture_{tag}'
    rng = np.random.default_rng(0)
    shape = (FIXTURE_KLON, FIXTURE_KLEV)
    args = {
        'pt': rng.random(shape),
        'pq': rng.random(shape),
        'pa': rng.random(shape),
        'tend_t': np.zeros(shape),
        'tend_q': np.zeros(shape),
    }
    candidate(KLON=FIXTURE_KLON, KLEV=FIXTURE_KLEV, **args)
    return args['tend_t'], args['tend_q']


@pytest.mark.parametrize('variant', COALESCE_VARIANTS)
def test_phase_present_for_its_variants(variant):
    """The coalescing phase is appended after the map-producing phase -- it has nothing to fuse until
    the loops are maps -- and holds the inline-then-fuse recipe in that order."""
    phases = variant_phases(variant)
    names = [name for name, _ in phases]
    assert names.count(COALESCE_PHASE) == 1, names
    assert names.index(COALESCE_PHASE) > names.index('parallelize'), names
    assert [label for label, _ in dict(phases)[COALESCE_PHASE]
            ] == ['inline_nsdfgs', 'fuse_states', 'collapse', 'fuse_maps', 'fuse_states']


@pytest.mark.parametrize('variant', [v for v in VARIANTS if v not in COALESCE_VARIANTS])
def test_phase_absent_for_the_canon_variants(variant):
    """Canon already coalesces inside its own recipe, so it must not get a second round here."""
    assert COALESCE_PHASE not in phase_names(variant)


@pytest.mark.parametrize('variant', COALESCE_VARIANTS)
def test_rest_of_the_plan_is_unchanged(variant):
    """Only one phase is added, and it is appended: every earlier phase keeps its name and stage count
    against the plan's own sources of truth, so the per-phase numeric checks before it still run on
    exactly the graphs they used to."""
    phases = variant_phases(variant)
    expected = [('start', len(specialize_stage(None) + pretreat_stages()))]
    expected += [(name, len(stages)) for name, stages in parallelize_phases()]
    assert [(name, len(stages)) for name, stages in phases[:-1]] == expected
    assert phases[-1][0] == COALESCE_PHASE


def test_coalesce_stays_before_the_terminal_offload_phase():
    """Offload is terminal by construction; coalescing must not displace it."""
    names = [name for name, _ in variant_phases('parallelize', offload=True)]
    assert names[-1] == 'offload'
    assert names.index(COALESCE_PHASE) == len(names) - 2, names


def test_coalescing_removes_the_nsdfg_walls_and_fuses_maps(parallelized):
    """The measurement: NSDFG and map counts strictly drop, and the graph still validates."""
    sdfg = copy.deepcopy(parallelized)
    before = graph_counts(sdfg)
    assert before[0] > 0, 'fixture is not NSDFG-wrapped -- the phase would have nothing to inline'
    assert before[1] > 1, 'fixture has no adjacent maps -- the phase would have nothing to fuse'
    apply_phase(sdfg, COALESCE_PHASE)
    sdfg.validate()
    after = graph_counts(sdfg)
    assert after[0] == 0, f'nested SDFGs left after inlining: {before} -> {after}'
    assert after[1] < before[1], f'no maps fused: {before} -> {after}'


def test_coalescing_does_not_change_the_numbers(parallelized):
    """Inlining and fusion are value-preserving: bit-identical outputs, not merely close."""
    sdfg = copy.deepcopy(parallelized)
    reference = run_fixture(sdfg, 'before')
    apply_phase(sdfg, COALESCE_PHASE)
    sdfg.validate()
    coalesced = run_fixture(sdfg, 'after')
    for name, ref, out in zip(('tend_t', 'tend_q'), reference, coalesced):
        assert np.array_equal(ref, out), f'{name} changed: max |diff| = {np.max(np.abs(ref - out))}'


def test_reapplying_the_phase_is_a_no_op(parallelized):
    """Idempotence -- the ``FixedPointPipeline`` spin hazard. ONE application coalesces fully: the
    second leaves the graph bit-identical and every stage reports "did not modify"."""
    sdfg = copy.deepcopy(parallelized)
    apply_phase(sdfg, COALESCE_PHASE)
    settled = sdfg.hash_sdfg()
    returns = apply_phase(sdfg, COALESCE_PHASE)
    assert sdfg.hash_sdfg() == settled, 'the phase kept mutating a graph it had already coalesced'
    inline, fuse_states, collapse, fuse_maps, fuse_states_again = returns
    assert (inline, fuse_states, collapse, fuse_states_again) == (None, None, None, None)
    # ``fuse_maps`` is a ``Pipeline``, whose return always carries its ``FindSingleUseData``
    # dependency's result -- so it is never ``None``. What must be absent is the fusion itself.
    assert 'FullMapFusion' not in fuse_maps


def test_every_coalesce_stage_actually_fires(parallelized):
    """No dead stages: on the first application each one reports a modification. A stage that never
    matches costs a graph scan and hides the fact that nothing was coalesced."""
    sdfg = copy.deepcopy(parallelized)
    inline, fuse_states, collapse, fuse_maps, fuse_states_again = apply_phase(sdfg, COALESCE_PHASE)
    assert inline, 'nothing inlined'
    assert fuse_states, 'no states fused before map fusion'
    assert collapse, 'no map nest collapsed'
    assert 'FullMapFusion' in fuse_maps, f'no maps fused: {sorted(fuse_maps)}'
    assert fuse_states_again, 'the trailing state fusion is dead -- drop it'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
