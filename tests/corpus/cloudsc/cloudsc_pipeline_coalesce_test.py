# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the coalescing band of the canonicalization recipe
(:func:`dace.transformation.passes.canonicalize.pipeline._coalesce`).

The band used to be an extra phase that ``tests.corpus.cloudsc.pipelines`` appended to the
``parallelize`` plan. It now lives in the canonicalize recipe itself, as one contiguous run of
``coalesce``-labelled stages between ``post_l2m`` and ``loop_fuse``, and the CloudSC plan files those
labels into the ``parallelize`` super-phase instead of opening a checkpoint boundary of their own.

The placement tests read the recipe only (no CloudSC parse, no pipeline run). The effect tests use a
small CloudSC-shaped fixture -- four element-wise ``(jl, jk)`` loop nests over shared arrays -- driven
through the canon recipe up to, but not including, the band. That is the graph the band is meant to
see: the loops have just become maps and nothing has fused them yet.

    pytest tests/corpus/cloudsc/cloudsc_pipeline_coalesce_test.py -v
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import _build_stages, _coalesce
from tests.corpus.cloudsc.pipelines import variant_phases

#: The recipe label the band carries, and the stages it must sit between.
BAND = 'coalesce'
BEFORE_BAND, AFTER_BAND = 'post_l2m', 'loop_fuse'

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


def nmaps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def band_bounds(target: str):
    """``(labels, first, last)`` for the coalescing band in the ``target`` recipe."""
    labels = [label for label, _ in _build_stages(target=target)]
    at = [i for i, label in enumerate(labels) if label == BAND]
    assert at, f'no {BAND!r} stage in the {target} recipe'
    return labels, at[0], at[-1]


@pytest.mark.parametrize('target', ['cpu', 'gpu'])
def test_band_is_one_contiguous_run_between_l2m_and_loop_fuse(target):
    """The band is a single uninterrupted run of the length ``_coalesce`` declares, and it sits after
    the loops have become maps but before loop fusion -- it has nothing to fuse any earlier, and
    ``loop_fuse`` would see un-coalesced maps any later."""
    labels, first, last = band_bounds(target)
    assert last - first + 1 == labels.count(BAND), f'{BAND} band is split: {labels.count(BAND)} stages, {first}..{last}'
    assert labels.count(BAND) == len(_coalesce()), 'recipe and _coalesce() disagree on the band length'
    assert labels[first - 1] == BEFORE_BAND, labels[first - 1]
    assert labels[last + 1] == AFTER_BAND, labels[last + 1]


@pytest.mark.parametrize('variant', ['canon_cpu', 'canon_gpu'])
def test_band_folds_into_the_parallelize_super_phase(variant):
    """The CloudSC plan must not open a checkpoint boundary for the band: its stages ride inside the
    ``parallelize`` super-phase, next to the ``post_l2m`` stages they follow."""
    phases = variant_phases(variant)
    assert BAND not in [name for name, _ in phases], 'the band must not be its own phase any more'
    holders = {name for name, stages in phases if any(label == BAND for label, _ in stages)}
    assert holders == {'parallelize'}, holders


def test_the_parallelize_variant_has_no_band():
    """``parallelize`` runs ``ParallelizePipeline``, which coalesces in its own ``fuse`` phase; a
    second band here would be dead work."""
    labels = [label for _, stages in variant_phases('parallelize') for label, _ in stages]
    assert BAND not in labels, labels


@pytest.fixture(scope='module')
def premapped():
    """The fixture driven through the canon recipe up to -- and stopping before -- the band."""
    stages = _build_stages(target='cpu')
    _, first, _ = band_bounds('cpu')
    sdfg = elementwise_chain.to_sdfg(simplify=False)
    for _label, unit in stages[:first]:
        unit.apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def apply_band(sdfg: dace.SDFG):
    """Apply the coalescing band to ``sdfg``; return the per-stage ``apply_pass`` returns."""
    stages = _build_stages(target='cpu')
    _, first, last = band_bounds('cpu')
    return [unit.apply_pass(sdfg, {}) for _label, unit in stages[first:last + 1]]


def fused_maps(returns) -> bool:
    """Did any stage of the band report a map fusion? ``FuseMaps`` runs inside a ``Pipeline``, whose
    return always carries its ``FindSingleUseData`` dependency -- so presence of the key, not
    truthiness of the return, is what says the fusion happened."""
    return any(isinstance(ret, dict) and 'FuseMaps' in ret for ret in returns)


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


def test_coalescing_fuses_the_maps(premapped):
    """The measurement: the map count strictly drops and the graph still validates."""
    sdfg = copy.deepcopy(premapped)
    before = nmaps(sdfg)
    assert before > 1, 'fixture has no adjacent maps -- the band would have nothing to fuse'
    returns = apply_band(sdfg)
    sdfg.validate()
    assert fused_maps(returns), 'the band reported no map fusion'
    assert nmaps(sdfg) < before, f'no maps fused: {before} -> {nmaps(sdfg)}'


def test_coalescing_does_not_change_the_numbers(premapped):
    """Inlining and fusion are value-preserving: bit-identical outputs, not merely close."""
    sdfg = copy.deepcopy(premapped)
    reference = run_fixture(sdfg, 'before')
    apply_band(sdfg)
    sdfg.validate()
    coalesced = run_fixture(sdfg, 'after')
    for name, ref, out in zip(('tend_t', 'tend_q'), reference, coalesced):
        assert np.array_equal(ref, out), f'{name} changed: max |diff| = {np.max(np.abs(ref - out))}'


def test_reapplying_the_band_is_a_no_op(premapped):
    """Idempotence -- the ``FixedPointPipeline`` spin hazard. ONE application coalesces fully: the
    second leaves the graph bit-identical and reports no further fusion."""
    sdfg = copy.deepcopy(premapped)
    apply_band(sdfg)
    settled = sdfg.hash_sdfg()
    again = apply_band(sdfg)
    assert sdfg.hash_sdfg() == settled, 'the band kept mutating a graph it had already coalesced'
    assert not fused_maps(again), 'the band fused maps a second time'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
