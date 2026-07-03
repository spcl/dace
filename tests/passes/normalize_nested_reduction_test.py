# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for
:class:`~dace.transformation.passes.normalize_nested_reduction.NormalizeNestedReduction`.

The pass rewrites a masked/nested scalar reduction emitted as an in-nsdfg WCR into a
write-only output connector (the ``azimint_naive`` shape) into the seeded-body-local +
AccessNode-sourced map-exit-WCR shape that survives downstream canonicalization. It must
be value-preserving on the untransformed baseline for every kernel it touches, and a
no-op on a second run.
"""
import os

# Pin a deterministic single-threaded run before DaCe/OpenMP initialize, so the
# value-preserving assertions don't flake on thread races.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import uuid

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.normalize_nested_reduction import NormalizeNestedReduction
from tests.corpus import corpus_suite as CS


def _write_only_scalar_wcr_conns(sdfg: dace.SDFG):
    """Every write-only nsdfg output connector still fed by an in-body scalar WCR."""
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for st in sd.all_states():
            for n in st.nodes():
                if not isinstance(n, nodes.NestedSDFG):
                    continue
                for oc in n.out_connectors:
                    if oc in n.in_connectors:
                        continue
                    for ist in n.sdfg.all_states():
                        for e in ist.edges():
                            if (e.data is not None and e.data.wcr is not None and e.data.data == oc
                                    and e.data.subset is not None and e.data.subset.num_elements() == 1):
                                out.append(oc)
    return out


# ---------------------------------------------------------------------------
# Corpus tests: value-preservation on the untransformed baseline (normalize only).
# ---------------------------------------------------------------------------
# azimint_* carry the target write-only in-nsdfg reduction; the polybench kernels carry a
# nested reduction the pass must leave value-preserving (symm/covariance/gramschmidt are
# already-normalized or symbol-guarded, floyd_warshall exercises the ``min`` op/identity).
_NESTED_REDUCTION_KERNELS = [
    ('np', 'azimint_naive'),
    ('np', 'azimint_hist'),
    ('poly', 'symm'),
    ('poly', 'covariance'),
    ('poly', 'gramschmidt'),
    ('poly', 'floyd_warshall'),
]


def _normalize_only(sdfg: dace.SDFG) -> dace.SDFG:
    NormalizeNestedReduction().apply_pass(sdfg, {})
    return sdfg


@pytest.mark.parametrize('suite,name', _NESTED_REDUCTION_KERNELS)
def test_normalize_only_preserves_semantics(suite, name):
    """``NormalizeNestedReduction`` alone (no canonicalize) is value-preserving."""
    ctx = CS.make(suite, name, 'S')
    sdfg = CS.build(ctx, _normalize_only, 'nnr_' + uuid.uuid4().hex[:8])
    assert CS.run_matches(ctx, sdfg), f'normalize changed {suite}:{name} output vs reference'


@pytest.mark.parametrize('suite,name', _NESTED_REDUCTION_KERNELS)
def test_normalize_is_idempotent(suite, name):
    """A second apply is a no-op (returns ``None``) and the output still matches."""
    ctx = CS.make(suite, name, 'S')

    second = {}

    def twice(sdfg):
        NormalizeNestedReduction().apply_pass(sdfg, {})
        second['res'] = NormalizeNestedReduction().apply_pass(sdfg, {})
        return sdfg

    sdfg = CS.build(ctx, twice, 'nnr2_' + uuid.uuid4().hex[:8])
    assert second['res'] is None, f'second apply on {suite}:{name} should be a no-op; got {second["res"]}'
    assert CS.run_matches(ctx, sdfg), f'double-normalize changed {suite}:{name} output vs reference'


def test_azimint_naive_structure_after_normalize():
    """On ``azimint_naive`` the pass fires: no write-only in-nsdfg scalar WCR remains, and
    the map-exit edge feeding the accumulator ``tmp`` now carries a WCR."""
    ctx = CS.make('np', 'azimint_naive', 'S')

    captured = {}

    def tf(sdfg):
        captured['res'] = NormalizeNestedReduction().apply_pass(sdfg, {})
        return sdfg

    sdfg = CS.build(ctx, tf, 'nnrstruct_' + uuid.uuid4().hex[:8])

    assert captured['res'] is not None, 'pass should rewrite azimint_naive (has the target pattern)'
    assert not _write_only_scalar_wcr_conns(sdfg), 'no write-only in-nsdfg scalar WCR should remain'

    tmp_exit_wcr = [
        e.data.wcr for st in sdfg.all_states() for e in st.edges()
        if e.data is not None and e.data.data == 'tmp' and isinstance(e.dst, nodes.MapExit)
    ]
    assert tmp_exit_wcr, 'the map-exit edge feeding tmp should exist'
    assert all(w is not None for w in tmp_exit_wcr), f'map-exit edge feeding tmp must carry a WCR; got {tmp_exit_wcr}'


# ---------------------------------------------------------------------------
# Hand-built micro SDFGs: the single-accumulator ``+`` and ``min`` cases.
# ---------------------------------------------------------------------------
def _build_masked_reduction(op_wcr: str, seed_outer: bool) -> dace.SDFG:
    """``acc OP= data[i]`` guarded by ``mask[i] > 0``, with the body wrapped in a
    NestedSDFG whose accumulator is a WRITE-ONLY output connector carrying an in-body
    scalar WCR -- the shape the pass normalizes.

    :param op_wcr: The WCR lambda string (e.g. ``'lambda x, y: x + y'``).
    :param seed_outer: When True, seed ``acc`` to 0 in a state before the map (the ``+``
        shape); when False, leave it unseeded so the pass must insert a ``min`` seed.
    """
    N = 24
    sdfg = dace.SDFG('masked_' + uuid.uuid4().hex[:8])
    sdfg.add_array('data', [N], dace.float64)
    sdfg.add_array('mask', [N], dace.int64)
    sdfg.add_array('acc', [1], dace.float64)

    if seed_outer:
        seed = sdfg.add_state('seed', is_start_block=True)
        tz = seed.add_tasklet('z', {}, {'__out'}, '__out = 0.0')
        seed.add_edge(tz, '__out', seed.add_write('acc'), None, dace.Memlet('acc[0]'))
        state = sdfg.add_state_after(seed, 'map')
    else:
        state = sdfg.add_state('map', is_start_block=True)

    body = dace.SDFG('body')
    body.add_array('d', [1], dace.float64)
    body.add_array('m', [1], dace.int64)
    body.add_scalar('acc_out', dace.float64)  # WRITE-ONLY output connector
    cond = ConditionalBlock('ifb')
    body.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion('then', sdfg=body)
    tstate = branch.add_state('t', is_start_block=True)
    t = tstate.add_tasklet('acc', {'_d'}, {'__out'}, '__out = _d')
    tstate.add_edge(tstate.add_read('d'), None, t, '_d', dace.Memlet('d[0]'))
    tstate.add_edge(t, '__out', tstate.add_write('acc_out'), None, dace.Memlet(data='acc_out', subset='0', wcr=op_wcr))
    cond.add_branch(dace.properties.CodeBlock('m[0] > 0'), branch)

    me, mx = state.add_map('m', {'i': f'0:{N}'})
    n = state.add_nested_sdfg(body, {'d', 'm'}, {'acc_out'})
    state.add_memlet_path(state.add_read('data'), me, n, dst_conn='d', memlet=dace.Memlet('data[i]'))
    state.add_memlet_path(state.add_read('mask'), me, n, dst_conn='m', memlet=dace.Memlet('mask[i]'))
    state.add_memlet_path(n, mx, state.add_write('acc'), src_conn='acc_out', memlet=dace.Memlet('acc[0]'))
    sdfg.validate()
    return sdfg


def test_micro_single_accumulator_sum():
    """The single-scalar ``+`` reduction is bit-exact after normalization (the bug drops
    the cross-iteration accumulation -> ~one lane's value)."""
    sdfg = _build_masked_reduction('lambda x, y: x + y', seed_outer=True)
    res = NormalizeNestedReduction().apply_pass(sdfg, {})
    assert res is not None
    assert not _write_only_scalar_wcr_conns(sdfg)
    sdfg.validate()

    rng = np.random.default_rng(0)
    data = rng.uniform(-1.0, 1.0, 24)
    mask = (rng.uniform(0.0, 1.0, 24) > 0.5).astype(np.int64)
    acc = np.zeros(1)
    sdfg(data=data.copy(), mask=mask.copy(), acc=acc, N=24)
    assert np.isclose(acc[0], data[mask > 0].sum())


def test_micro_single_accumulator_min_seeds_outer_accumulator():
    """The ``min`` reduction seeds the outer accumulator to ``+inf`` when it is not already
    seeded (a ``min`` reduction started from 0, or from garbage, is silently wrong)."""
    sdfg = _build_masked_reduction('lambda x, y: min(x, y)', seed_outer=False)
    res = NormalizeNestedReduction().apply_pass(sdfg, {})
    assert res is not None
    sdfg.validate()

    rng = np.random.default_rng(3)
    data = rng.uniform(-1.0, 1.0, 24)
    mask = (rng.uniform(0.0, 1.0, 24) > 0.4).astype(np.int64)
    acc = np.full(1, 1234.5)  # garbage; the inserted seed must overwrite it with +inf
    sdfg(data=data.copy(), mask=mask.copy(), acc=acc, N=24)
    assert np.isclose(acc[0], data[mask > 0].min())


def test_micro_no_op_when_map_exit_edge_already_has_wcr():
    """A NestedSDFG whose ``NestedSDFG -> MapExit`` edge already carries a WCR (the
    already-normalized / native ``symm`` shape) is left untouched."""
    sdfg = _build_masked_reduction('lambda x, y: x + y', seed_outer=True)
    NormalizeNestedReduction().apply_pass(sdfg, {})  # first run normalizes it
    res = NormalizeNestedReduction().apply_pass(sdfg, {})  # already has map-exit WCR
    assert res is None


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
