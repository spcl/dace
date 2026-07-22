# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for
:class:`~dace.transformation.passes.normalize_wcr.NormalizeWCR`.

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
from dace.transformation.passes.normalize_wcr import NormalizeWCR
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


def _wcr_edges(sdfg: dace.SDFG, single: bool):
    """WCR edges whose subset is single-element (``single=True``) or multi-element."""
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for st in sd.all_states():
            for e in st.edges():
                if e.data is not None and e.data.wcr is not None and e.data.subset is not None:
                    if (e.data.subset.num_elements() == 1) == single:
                        out.append(e)
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
    NormalizeWCR().apply_pass(sdfg, {})
    return sdfg


@pytest.mark.parametrize('suite,name', _NESTED_REDUCTION_KERNELS)
def test_normalize_only_preserves_semantics(suite, name):
    """``NormalizeWCR`` alone (no canonicalize) is value-preserving."""
    ctx = CS.make(suite, name, 'S')
    sdfg = CS.build(ctx, _normalize_only, 'nnr_' + uuid.uuid4().hex[:8])
    assert CS.run_matches(ctx, sdfg), f'normalize changed {suite}:{name} output vs reference'


@pytest.mark.parametrize('suite,name', _NESTED_REDUCTION_KERNELS)
def test_normalize_is_idempotent(suite, name):
    """A second apply is a no-op (returns ``None``) and the output still matches."""
    ctx = CS.make(suite, name, 'S')

    second = {}

    def twice(sdfg):
        NormalizeWCR().apply_pass(sdfg, {})
        second['res'] = NormalizeWCR().apply_pass(sdfg, {})
        return sdfg

    sdfg = CS.build(ctx, twice, 'nnr2_' + uuid.uuid4().hex[:8])
    assert second['res'] is None, f'second apply on {suite}:{name} should be a no-op; got {second["res"]}'
    assert CS.run_matches(ctx, sdfg), f'double-normalize changed {suite}:{name} output vs reference'


def test_symm_slice_wcr_extracted_to_single_element():
    """symm's ``C[0:i,j]`` slice WCR (a per-instance scatter trapped in a 2-state nsdfg,
    surfacing as a multi-element WCR at the boundary) is extracted to an outer
    single-element ``C[k,j]`` WCR; no multi-element WCR survives and the value matches."""
    ctx = CS.make('poly', 'symm', 'S')
    seen = {}

    def norm(sdfg):
        seen['before'] = len(_wcr_edges(sdfg, single=False))
        NormalizeWCR(extract_slice_wcr=True).apply_pass(sdfg, {})
        # The trapped nsdfg output is redirected to a dead transient; its inner producer is
        # then dead. simplify's DCE prunes it (as the canonicalize pipeline does downstream).
        sdfg.simplify()
        seen['after_slice'] = len(_wcr_edges(sdfg, single=False))
        seen['after_single'] = len(_wcr_edges(sdfg, single=True))
        return sdfg

    sdfg = CS.build(ctx, norm, 'nnrslice_' + uuid.uuid4().hex[:8])
    assert seen['before'] >= 1, 'symm baseline should carry the C[0:i,j] slice WCR'
    assert seen['after_slice'] == 0, f'slice WCR not fully extracted: {seen["after_slice"]} remain'
    assert seen['after_single'] >= 1, 'expected a single-element scatter WCR after extraction'
    assert CS.run_matches(ctx, sdfg), 'slice-WCR extraction changed symm output vs reference'


def test_azimint_naive_structure_after_normalize():
    """On ``azimint_naive`` the pass fires: no write-only in-nsdfg scalar WCR remains, and
    the map-exit edge feeding the accumulator ``tmp`` now carries a WCR."""
    ctx = CS.make('np', 'azimint_naive', 'S')

    captured = {}

    def tf(sdfg):
        captured['res'] = NormalizeWCR().apply_pass(sdfg, {})
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
    res = NormalizeWCR().apply_pass(sdfg, {})
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
    res = NormalizeWCR().apply_pass(sdfg, {})
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
    NormalizeWCR().apply_pass(sdfg, {})  # first run normalizes it
    res = NormalizeWCR().apply_pass(sdfg, {})  # already has map-exit WCR
    assert res is None


# ---------------------------------------------------------------------------
# Two INDEPENDENT reductions in one map + the NestInnermostMapBodyIntoNSDFG (vectorizer
# entry) / ExpandNestedSDFGInputs interaction. The supported shape is ``nsdfg -> AN
# -[wcr]-> MapExit -> [wcr]`` for a single-element / scalar accumulator; nesting the map
# body one level deeper and expanding the boundary must keep BOTH reductions correct.
# ---------------------------------------------------------------------------
def _build_two_independent_reductions() -> dace.SDFG:
    """A map with TWO INDEPENDENT scalar reductions ``acc1 += data1[i]`` and
    ``acc2 += data2[i]`` in one body NestedSDFG -- each a WRITE-ONLY output connector
    carrying its own in-body scalar ``+`` WCR. ``NormalizeWCR`` must normalize BOTH,
    independently, without cross-contaminating the two accumulators."""
    N = 24
    sdfg = dace.SDFG('two_red_' + uuid.uuid4().hex[:8])
    sdfg.add_array('data1', [N], dace.float64)
    sdfg.add_array('data2', [N], dace.float64)
    sdfg.add_array('acc1', [1], dace.float64)
    sdfg.add_array('acc2', [1], dace.float64)

    seed = sdfg.add_state('seed', is_start_block=True)
    for acc in ('acc1', 'acc2'):
        tz = seed.add_tasklet('z_' + acc, {}, {'__out'}, '__out = 0.0')
        seed.add_edge(tz, '__out', seed.add_write(acc), None, dace.Memlet(acc + '[0]'))
    state = sdfg.add_state_after(seed, 'map')

    body = dace.SDFG('body')
    body.add_array('d1', [1], dace.float64)
    body.add_array('d2', [1], dace.float64)
    body.add_scalar('acc1_out', dace.float64)  # WRITE-ONLY output connectors
    body.add_scalar('acc2_out', dace.float64)
    cstate = body.add_state('c', is_start_block=True)
    for d, oc in (('d1', 'acc1_out'), ('d2', 'acc2_out')):
        t = cstate.add_tasklet('a_' + oc, {'_d'}, {'__out'}, '__out = _d')
        cstate.add_edge(cstate.add_read(d), None, t, '_d', dace.Memlet(d + '[0]'))
        cstate.add_edge(t, '__out', cstate.add_write(oc), None,
                        dace.Memlet(data=oc, subset='0', wcr='lambda x, y: x + y'))

    me, mx = state.add_map('m', {'i': f'0:{N}'})
    n = state.add_nested_sdfg(body, {'d1', 'd2'}, {'acc1_out', 'acc2_out'})
    state.add_memlet_path(state.add_read('data1'), me, n, dst_conn='d1', memlet=dace.Memlet('data1[i]'))
    state.add_memlet_path(state.add_read('data2'), me, n, dst_conn='d2', memlet=dace.Memlet('data2[i]'))
    state.add_memlet_path(n, mx, state.add_write('acc1'), src_conn='acc1_out', memlet=dace.Memlet('acc1[0]'))
    state.add_memlet_path(n, mx, state.add_write('acc2'), src_conn='acc2_out', memlet=dace.Memlet('acc2[0]'))
    sdfg.validate()
    return sdfg


def _run_two_reductions(sdfg):
    rng = np.random.default_rng(7)
    d1 = rng.uniform(-1.0, 1.0, 24)
    d2 = rng.uniform(-1.0, 1.0, 24)
    a1, a2 = np.zeros(1), np.zeros(1)
    sdfg(data1=d1.copy(), data2=d2.copy(), acc1=a1, acc2=a2, N=24)
    return a1, a2, d1, d2


def test_micro_two_independent_reductions_normalize():
    """``NormalizeWCR`` normalizes BOTH write-only scalar reductions; each stays bit-exact
    and independent (``acc1 == data1.sum()``, ``acc2 == data2.sum()``)."""
    sdfg = _build_two_independent_reductions()
    res = NormalizeWCR().apply_pass(sdfg, {})
    assert res is not None
    assert not _write_only_scalar_wcr_conns(sdfg), 'both reductions must be normalized off the connectors'
    sdfg.validate()
    a1, a2, d1, d2 = _run_two_reductions(sdfg)
    assert np.isclose(a1[0], d1.sum()) and np.isclose(a2[0], d2.sum())


def test_two_independent_reductions_through_nest_body_and_expand():
    """The vectorizer entry nests the innermost map body one level deeper
    (``NestInnermostMapBodyIntoNSDFG``) and later widens the boundary
    (``ExpandNestedSDFGInputs``). Running ``NormalizeWCR`` in between must keep BOTH scalar
    reductions valid + bit-exact -- the ``nsdfg -> AN -[wcr]-> MapExit -> [wcr]`` shape must
    survive the extra nesting + boundary expansion."""
    from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
    from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
    sdfg = _build_two_independent_reductions()
    NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    NormalizeWCR().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
    sdfg.validate()
    # No self-contained WCR must survive inside a body NSDFG (the tiler drops it) for either acc.
    assert not _write_only_scalar_wcr_conns(sdfg), 'nest+expand must not strand a body-local scalar WCR'
    a1, a2, d1, d2 = _run_two_reductions(sdfg)
    assert np.isclose(a1[0], d1.sum()), f'acc1 wrong after nest+normalize+expand: {a1[0]} vs {d1.sum()}'
    assert np.isclose(a2[0], d2.sum()), f'acc2 wrong after nest+normalize+expand: {a2[0]} vs {d2.sum()}'


# ---------------------------------------------------------------------------
# Currently-BREAKING pattern (documented gap): an in-place aug-assign ``b[i] += addend``
# lifted to a WCR-map makes ``b`` a READ-WRITE output connector (in BOTH in and out
# connectors) with a redundant interior WCR alongside the already-surfaced boundary WCR.
# NormalizeWCR only handles WRITE-ONLY connectors, so the interior WCR is left in the body;
# the tiler's "no WCR inside the body NSDFG" invariant then trips (TSVC s212). A naive
# drop-to-plain of the interior WCR mis-scopes the boundary index in codegen, so the fix is
# deferred -- this test pins the current behavior so a future fix flips the xfail.
# ---------------------------------------------------------------------------
def _build_readwrite_redundant_interior_wcr() -> dace.SDFG:
    """``b[i] += data[i]`` as a WCR-map: ``b`` is BOTH an in and out connector of the body
    NestedSDFG, the interior writes the addend into ``b`` via a scalar ``+`` WCR (write-only
    sink in the body), and the boundary ``nsdfg -> MapExit -> b`` edge ALSO carries a ``+``
    WCR. The interior WCR is redundant and should collapse to a plain write."""
    N = 24
    sdfg = dace.SDFG('rw_wcr_' + uuid.uuid4().hex[:8])
    sdfg.add_array('data', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state('map', is_start_block=True)

    body = dace.SDFG('body')
    body.add_array('d', [1], dace.float64)
    body.add_array('b_io', [1], dace.float64)  # read-write connector (in AND out)
    cstate = body.add_state('c', is_start_block=True)
    t = cstate.add_tasklet('acc', {'_d'}, {'__out'}, '__out = _d')
    cstate.add_edge(cstate.add_read('d'), None, t, '_d', dace.Memlet('d[0]'))
    cstate.add_edge(t, '__out', cstate.add_write('b_io'), None,
                    dace.Memlet(data='b_io', subset='0', wcr='lambda x, y: x + y'))

    me, mx = state.add_map('m', {'i': f'0:{N}'})
    n = state.add_nested_sdfg(body, {'d', 'b_io'}, {'b_io'})  # b_io is in AND out
    state.add_memlet_path(state.add_read('data'), me, n, dst_conn='d', memlet=dace.Memlet('data[i]'))
    state.add_memlet_path(state.add_read('b'), me, n, dst_conn='b_io', memlet=dace.Memlet('b[i]'))
    state.add_memlet_path(n, mx, state.add_write('b'), src_conn='b_io',
                          memlet=dace.Memlet(data='b', subset='i', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def _body_wcr_edges(sdfg):
    """Self-contained WCR edges INSIDE a body NestedSDFG (what the tiler forbids)."""
    return [(ist.label, e) for _n, _p in sdfg.all_nodes_recursive() if isinstance(_n, nodes.NestedSDFG)
            for ist in _n.sdfg.all_states() for e in ist.edges() if e.data is not None and e.data.wcr is not None]


def test_readwrite_redundant_interior_wcr_current_gap():
    """DOCUMENTS the current gap (TSVC s212): NormalizeWCR does NOT clear the redundant
    interior WCR on a read-write connector, so a body-local WCR survives. When the fix lands
    (drop the redundant interior WCR to a plain write), flip this to assert it is cleared."""
    sdfg = _build_readwrite_redundant_interior_wcr()
    NormalizeWCR().apply_pass(sdfg, {})
    surviving = _body_wcr_edges(sdfg)
    # Current behavior: the interior WCR is left in place (the `write_only` filter skips a
    # read-write connector). This is the s212 break; asserting it here so a future fix is noticed.
    assert surviving, ('read-write redundant interior WCR unexpectedly cleared -- if this is the '
                       'intended fix, replace this assertion with `assert not surviving`')


@pytest.mark.xfail(reason='s212: the read-write aug-assign ``b[i] += addend`` is a DOUBLE-'
                   'accumulation encoding (interior WCR into the body-local ``b`` sink + boundary '
                   'WCR), and DaCe WCR first-write semantics reset a location to the op identity, '
                   'so the body-WCR cannot simply be dropped to a plain write -- that discards the '
                   'incoming ``b_old`` (verified divergence). The fix must rewrite to a single '
                   'accumulation the tiler accepts: wire the input ``b`` into the body as an '
                   'explicit read-modify-write (``b_out = b_in + addend``, plain) with a PLAIN '
                   'boundary, OR have canon emit a single-WCR form. Neither is a NormalizeWCR '
                   'one-liner; pinned here so the fix flips this xfail.',
                   strict=True)
def test_readwrite_redundant_interior_wcr_should_be_cleared():
    """The DESIRED behavior once s212 is fixed: no self-contained WCR survives in the body,
    achieved by an explicit read-modify-write (NOT a naive drop-to-plain, which loses ``b_old``)."""
    sdfg = _build_readwrite_redundant_interior_wcr()
    NormalizeWCR().apply_pass(sdfg, {})
    assert not _body_wcr_edges(sdfg), 'interior WCR should become a read-modify-write, no body WCR'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
