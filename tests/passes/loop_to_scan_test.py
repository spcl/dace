# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.loop_to_scan.LoopToScan`."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_scan_nodes(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))


def test_inclusive_sum_1d():
    """``out[i+1] = out[i] + delta[i]`` over a 1-D array. Post-pass: loop body writes a
    per-iteration delta buffer (parallelizable) -> Scan libnode -> seed-add Map writes
    final ``out``. Matches sequential cumsum + seed oracle."""

    @dace.program
    def scan1d(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i]

    sdfg = scan1d.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    # Original loop stays (writes per-iter to ``_scan_in``); post-loop chain adds a Scan
    # libnode + seed-add map. LoopToMap can later lift the remaining loop (not our job).
    assert _num_scan_nodes(sdfg) == 1

    n = 16
    rng = np.random.default_rng(0)
    delta = rng.uniform(-1.0, 1.0, size=n)
    out = np.zeros(n + 1)
    out[0] = 0.5
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] + delta[i]
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected)


def test_inclusive_product_1d():
    """Same shape with ``*`` -- the pass picks ``ScanOp.PRODUCT``."""

    @dace.program
    def scan_prod(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] * delta[i]

    sdfg = scan_prod.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 8
    rng = np.random.default_rng(1)
    delta = rng.uniform(0.95, 1.05, size=n)  # keep magnitudes finite
    out = np.zeros(n + 1)
    out[0] = 1.0
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] * delta[i]
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected)


def test_inclusive_max_1d():
    """``out[i+1] = max(out[i], delta[i])`` -- the pass picks ``ScanOp.MAX``."""

    @dace.program
    def scan_max(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = max(out[i], delta[i])

    sdfg = scan_max.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 10
    rng = np.random.default_rng(2)
    delta = rng.uniform(-1.0, 1.0, size=n)
    out = np.zeros(n + 1)
    out[0] = 0.0
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = max(expected[i], delta[i])
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected)


@pytest.mark.parametrize('stride', [2, 3, 4, 5])
def test_refuses_non_unit_offset_modified_residue_class_scan(stride):
    """``out[i+S] = out[i] + delta[i]`` for ``S >= 2`` -- stride-``S`` residue-class scan.
    Libnode runs the ``S`` independent class scans in parallel; seed-add Map fans the
    ``S`` pre-loop seeds by ``_i mod S``. Matches sequential oracle."""

    @dace.program
    def stride_scan(out: dace.float64[N + stride], delta: dace.float64[N]):
        for i in range(N):
            out[i + stride] = out[i] + delta[i]

    sdfg = stride_scan.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 24
    rng = np.random.default_rng(stride)
    delta = rng.uniform(-1.0, 1.0, size=n)
    out = np.zeros(n + stride)
    out[:stride] = rng.uniform(-1.0, 1.0, size=stride)
    expected = out.copy()
    for i in range(n):
        expected[i + stride] = expected[i] + delta[i]
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected), \
        f'stride-{stride} scan mismatch: got {out}, expected {expected}'


def test_tsvc_s1221_residue_class_scan_inplace():
    """TSVC s1221 ``b[i] = b[i-4] + a[i]`` for ``i in [4, N)`` -- in-place stride-4
    residue-class scan (write at ``i``, read at ``i-4``): ``k_w==0``, ``k_r==-4``,
    ``stride==4``. Per-class seed ``b[k]`` for ``k in [0,4)``. Matches sequential
    oracle."""

    @dace.program
    def s1221(a: dace.float64[N], b: dace.float64[N]):
        for i in range(4, N):
            b[i] = b[i - 4] + a[i]

    sdfg = s1221.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 32
    rng = np.random.default_rng(1221)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    expected = b.copy()
    for i in range(4, n):
        expected[i] = expected[i - 4] + a[i]
    sdfg(a=a.copy(), b=b, N=n)
    assert np.allclose(b, expected), f's1221 mismatch: got {b}, expected {expected}'


def test_refuses_non_associative_op():
    """Subtraction isn't associative; the pass refuses any op outside +, *, max, min."""

    @dace.program
    def sub(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] - delta[i]

    sdfg = sub.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None


def test_refuses_delta_reads_carry_array():
    """TSVC s2111 ``aa[j, i] = (aa[j, i-1] + aa[j-1, i]) / 1.9``. The inner i-loop's
    scan-update tasklet reads ``aa`` twice: ``aa[j, i-1]`` matches the carry slice,
    ``aa[j-1, i]`` is a different row -- an extra carry-array read making this a 2-D
    coupled recurrence, not a 1-D scan. Lifting corrupts the result; matcher must
    refuse."""

    @dace.program
    def s2111(aa: dace.float64[N, N]):
        for j in range(1, N):
            for i in range(1, N):
                aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9

    sdfg = s2111.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None, ('LoopToScan must refuse the s2111 shape because the '
                         "'delta' aa[j-1, i] is another read of the carry array.")


def test_refuses_extra_non_transient_write():
    """The body writes a *second* non-transient array (``aux[i]``); that's per-iteration
    output we'd need to preserve outside the rewrite. The matcher refuses."""

    @dace.program
    def with_aux(out: dace.float64[N + 1], delta: dace.float64[N], aux: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i]
            aux[i] = delta[i] * 2.0

    sdfg = with_aux.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None


def test_tsvc_s111_inclusive_sum():
    """TSVC s111 shape: ``a[i] = a[i-1] + b[i]``. Same scan as v1 but written with the
    carry read at ``i-1`` and the write at ``i`` (vs ``i+1`` / ``i``)."""

    @dace.program
    def s111(a: dace.float64[N + 1], b: dace.float64[N + 1]):
        for i in range(1, N + 1):
            a[i] = a[i - 1] + b[i]

    sdfg = s111.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 12
    rng = np.random.default_rng(101)
    b = rng.uniform(-1.0, 1.0, size=n + 1)
    a = np.zeros(n + 1)
    a[0] = 3.0
    expected = a.copy()
    for i in range(1, n + 1):
        expected[i] = expected[i - 1] + b[i]
    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, expected)


def test_tsvc_s112_offset_one():
    """TSVC s112 shape: ``a[i+1] = a[i] + b[i]`` -- the canonical scan we already cover."""

    @dace.program
    def s112(a: dace.float64[N + 1], b: dace.float64[N]):
        for i in range(N):
            a[i + 1] = a[i] + b[i]

    sdfg = s112.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1


def test_tsvc_s221_v2_computed_delta_two_arrays():
    """TSVC s221 shape: ``b[i] = b[i-1] + a[i] + d[i]`` -- delta is ``a[i] + d[i]`` (computed
    by an upstream ``+`` tasklet). The v2 in-place rewrite leaves the upstream tasklet in
    place; the scan-update tasklet becomes a passthrough writing ``_scan_in[i]``."""

    @dace.program
    def s221(a: dace.float64[N + 1], b: dace.float64[N + 1], d: dace.float64[N + 1]):
        for i in range(1, N + 1):
            b[i] = b[i - 1] + a[i] + d[i]

    sdfg = s221.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, 'v2 computed-delta scan should match.'

    n = 10
    rng = np.random.default_rng(221)
    a = rng.uniform(-1.0, 1.0, size=n + 1)
    d = rng.uniform(-1.0, 1.0, size=n + 1)
    b = np.zeros(n + 1)
    b[0] = 2.5
    expected = b.copy()
    for i in range(1, n + 1):
        expected[i] = expected[i - 1] + a[i] + d[i]
    sdfg(a=a, b=b, d=d, N=n)
    assert np.allclose(b, expected), f'max diff {np.max(np.abs(b - expected))}'


def test_tsvc_s242_literal_augmented_carry_modified_from_refusal():
    """TSVC s242: ``a[i] = a[i-1] + 0.5 + 1.0 + b[i] + c[i] + d[i]``. First ``+`` lowers
    to ``__out = __in1 + 0.5`` -- ONE data input (the carry) + a literal. v3 accepts: the
    carry input is severed (as v1/v2), the passthrough emits the literal, and the
    downstream chain extends it with ``+ 1.0 + b[i] + c[i] + d[i]`` to build the delta.
    Scan folds in the seed ``a[0]``."""

    @dace.program
    def s242(a: dace.float64[N + 1], b: dace.float64[N + 1], c: dace.float64[N + 1], d: dace.float64[N + 1]):
        for i in range(1, N + 1):
            a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]

    sdfg = s242.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 12
    rng = np.random.default_rng(242)
    a = rng.standard_normal(n + 1)
    b = rng.standard_normal(n + 1)
    c = rng.standard_normal(n + 1)
    d = rng.standard_normal(n + 1)
    expected = a.copy()
    for i in range(1, n + 1):
        expected[i] = expected[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]
    sdfg(a=a, b=b, c=c, d=d, N=n)
    assert np.allclose(a, expected)


def test_literal_only_delta_constant_increment():
    """Pure literal delta: ``out[i+1] = out[i] + 0.25``. No downstream chain;
    the matcher accepts the carry tasklet's single-data-input + literal shape
    and the rewrite makes the tasklet emit ``__out = 0.25``."""

    @dace.program
    def lit(out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] + 0.25

    sdfg = lit.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 16
    out = np.zeros(n + 1)
    out[0] = 0.5
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] + 0.25
    sdfg(out=out, N=n)
    assert np.allclose(out, expected)


def test_literal_delta_product_negative_constant():
    """``out[i+1] = out[i] * -2.0`` -- negative literal via ``ast.UnaryOp(USub)``
    on the right operand."""

    @dace.program
    def lit(out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] * -2.0

    sdfg = lit.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 6  # keep magnitudes bounded
    out = np.zeros(n + 1)
    out[0] = 1.0
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] * -2.0
    sdfg(out=out, N=n)
    assert np.allclose(out, expected)


def test_tsvc_s222_self_referential_delta_refused():
    """TSVC s222 shape: ``e[i] = e[i-1] * e[i-1]`` -- the delta input IS the carry. The
    matcher requires two distinct inputs; with both inputs resolving to the same carry,
    the "non-carry delta" search fails and the loop is refused."""

    @dace.program
    def s222(e: dace.float64[N + 1]):
        for i in range(1, N + 1):
            e[i] = e[i - 1] * e[i - 1]

    sdfg = s222.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    # Both inputs resolve to ``e[i-1]`` -- ambiguous carry, no delta. Refused.
    assert res is None


def test_v2_computed_delta_with_scale():
    """``out[i+1] = out[i] + delta[i] * scale[i]`` -- a multi-tasklet body where the delta
    is the product ``delta[i] * scale[i]``."""

    @dace.program
    def scan_scaled(out: dace.float64[N + 1], delta: dace.float64[N], scale: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i] * scale[i]

    sdfg = scan_scaled.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 14
    rng = np.random.default_rng(7)
    delta = rng.uniform(-1.0, 1.0, size=n)
    scale = rng.uniform(0.5, 1.5, size=n)
    out = np.zeros(n + 1)
    out[0] = -0.5
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] + delta[i] * scale[i]
    sdfg(out=out, delta=delta, scale=scale, N=n)
    assert np.allclose(out, expected)


def test_refuses_when_delta_is_same_array():
    """``out[i+1] = out[i] + out[i]`` -- the delta IS the carry. The rewrite would
    self-alias; refused (this also catches scaling shapes like ``out[i+1] = 2*out[i]``
    once the frontend lowers them)."""

    @dace.program
    def self_double(out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] + out[i]

    sdfg = self_double.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None


def test_multi_state_body_with_empty_wrappers():
    """cloudsc ``pfsqrf`` shape: LoopRegion body has *three* states -- empty pre-state
    (iedge assignment like ``kfdia_plus_1 = kfdia + 1``), the scan body, empty post-state
    (advances the iterator). v1 refused on ``len(blocks) != 1``; the relaxation ignores
    empty wrapper states and drives the match from the single content state.
    """
    sdfg = dace.SDFG('scan_multi_state')
    sdfg.add_array('out', [N + 1], dace.float64)
    sdfg.add_array('delta', [N], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)

    loop = LoopRegion('scan_loop',
                      initialize_expr='i = 0',
                      condition_expr='i < N',
                      update_expr='i = i + 1',
                      loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    # Three body states: empty pre, content body, empty post -- exactly the cloudsc
    # for_1133 shape (with an iedge assignment on the pre->body edge).
    pre = loop.add_state('pre', is_start_block=True)
    body = loop.add_state('body')
    post = loop.add_state('post')
    loop.add_edge(pre, body, dace.InterstateEdge(assignments={'np1': '(N + 1)'}))
    loop.add_edge(body, post, dace.InterstateEdge())

    rd = body.add_read('out')
    wt = body.add_write('out')
    dd = body.add_read('delta')
    t = body.add_tasklet('add', {'_a', '_d'}, {'_o'}, '_o = _a + _d')
    body.add_edge(rd, None, t, '_a', dace.Memlet(data='out', subset='i'))
    body.add_edge(dd, None, t, '_d', dace.Memlet(data='delta', subset='i'))
    body.add_edge(t, '_o', wt, None, dace.Memlet(data='out', subset='i + 1'))

    sdfg.validate()
    assert _num_loops(sdfg) == 1
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, 'Multi-state body with empty wrappers should match the v1 scan template.'
    assert _num_scan_nodes(sdfg) == 1

    n = 16
    rng = np.random.default_rng(11)
    delta = rng.uniform(-1.0, 1.0, size=n)
    out = np.zeros(n + 1)
    out[0] = 0.25
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] + delta[i]
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected), f'multi-state scan returned {out[:5]}, expected {expected[:5]}'


def test_accepts_two_content_state_body_via_v5_fuser():
    """Two content states joined by a trivial iedge. v5's body-local state fuser merges
    them (aliasing same-data AccessNodes so RAW order is preserved), then the matcher
    proceeds. Regression: pre-v5 this stayed sequential (cloudsc ``pfsqrf`` fusion path)."""
    sdfg = dace.SDFG('scan_two_content_states')
    sdfg.add_array('out', [N + 1], dace.float64)
    sdfg.add_array('delta', [N], dace.float64)
    sdfg.add_scalar('_tmp', dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N', update_expr='i = i + 1', loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    s1 = loop.add_state('s1', is_start_block=True)
    s2 = loop.add_state('s2')
    loop.add_edge(s1, s2, dace.InterstateEdge())
    # s1 has nodes; s2 also has nodes -- two content states.
    dd = s1.add_read('delta')
    tw = s1.add_write('_tmp')
    s1.add_nedge(dd, tw, dace.Memlet(data='delta', subset='i', other_subset='0'))
    rd = s2.add_read('out')
    rt = s2.add_read('_tmp')
    wt = s2.add_write('out')
    t = s2.add_tasklet('add', {'_a', '_d'}, {'_o'}, '_o = _a + _d')
    s2.add_edge(rd, None, t, '_a', dace.Memlet(data='out', subset='i'))
    s2.add_edge(rt, None, t, '_d', dace.Memlet(data='_tmp', subset='0'))
    s2.add_edge(t, '_o', wt, None, dace.Memlet(data='out', subset='i + 1'))
    sdfg.validate()

    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, f'two-content-state body should fuse + match; got {res}'
    assert _num_scan_nodes(sdfg) == 1


def test_v4_multi_array_independent_scans():
    """Two independent scan recurrences in one loop body:

    * ``a[i+1] = a[i] + delta_a[i]`` -- SUM scan on ``a``
    * ``b[i+1] = b[i] * delta_b[i]`` -- PRODUCT scan on ``b``

    Each with its own carry/delta/op. v4 returns ``len(matches) == 2``; rewrite emits
    two Scan libnodes. cloudsc ``pfsqrf`` = production occurrence (five parallel sums)."""

    @dace.program
    def two_scans(a: dace.float64[N + 1], b: dace.float64[N + 1], da: dace.float64[N], db: dace.float64[N]):
        for i in range(N):
            a[i + 1] = a[i] + da[i]
            b[i + 1] = b[i] * db[i]

    sdfg = two_scans.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 2, f'expected two Scan rewrites; got {res}'
    assert _num_scan_nodes(sdfg) == 2

    n = 16
    rng = np.random.default_rng(404)
    da = rng.uniform(-1.0, 1.0, size=n)
    db = rng.uniform(0.95, 1.05, size=n)
    a = np.zeros(n + 1)
    a[0] = 0.5
    b = np.zeros(n + 1)
    b[0] = 1.0
    ea, eb = a.copy(), b.copy()
    for i in range(n):
        ea[i + 1] = ea[i] + da[i]
        eb[i + 1] = eb[i] * db[i]
    sdfg(a=a, b=b, da=da, db=db, N=n)
    assert np.allclose(a, ea) and np.allclose(b, eb), \
        f'multi-array scan diverged: a={a}, ea={ea}; b={b}, eb={eb}'


def test_v4_five_array_pfsqrf_pattern():
    """cloudsc-faithful five-carry prefix sum: the ``pfsqrf`` inner loop
    (``pfsqif``/``pfsqrf``/``pfsqlf``/``pfsqsf``/``pfcqlng`` carried side-by-side, all
    op = SUM). Matcher returns 5 ``_Scan`` infos; rewrite emits 5 Scan libnodes."""

    @dace.program
    def five_scans(s1: dace.float64[N + 1], s2: dace.float64[N + 1], s3: dace.float64[N + 1], s4: dace.float64[N + 1],
                   s5: dace.float64[N + 1], d1: dace.float64[N], d2: dace.float64[N], d3: dace.float64[N],
                   d4: dace.float64[N], d5: dace.float64[N]):
        for i in range(N):
            s1[i + 1] = s1[i] + d1[i]
            s2[i + 1] = s2[i] + d2[i]
            s3[i + 1] = s3[i] + d3[i]
            s4[i + 1] = s4[i] + d4[i]
            s5[i + 1] = s5[i] + d5[i]

    sdfg = five_scans.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 5, f'expected five Scan rewrites for the pfsqrf pattern; got {res}'
    assert _num_scan_nodes(sdfg) == 5

    n = 12
    rng = np.random.default_rng(5050)
    d = [rng.uniform(-1.0, 1.0, size=n) for _ in range(5)]
    s = [np.zeros(n + 1) for _ in range(5)]
    for k, sv in enumerate(s):
        sv[0] = 0.1 * (k + 1)
    es = [sv.copy() for sv in s]
    for k in range(5):
        for i in range(n):
            es[k][i + 1] = es[k][i] + d[k][i]
    sdfg(s1=s[0], s2=s[1], s3=s[2], s4=s[3], s5=s[4], d1=d[0], d2=d[1], d3=d[2], d4=d[3], d5=d[4], N=n)
    for k in range(5):
        assert np.allclose(s[k], es[k]), f's{k+1} diverged: {s[k]} vs {es[k]}'


def test_v5_state_fusion_preprocess_unblocks_multi_state_body():
    """Scan body split across two states joined by a trivial iedge (no assignments,
    condition = 1). v5's ``StateFusion`` preprocess inside ``LoopToScan.apply_pass`` fuses
    them; the matcher then sees a single-content-state body. cloudsc ``pfsqrf`` inner loop
    (``for_1134``): ``UnaryOp_1135`` + ``assign_1143_12`` joined this way.
    """
    sdfg = dace.SDFG('scan_two_state_body')
    sdfg.add_array('out', [N + 1], dace.float64)
    sdfg.add_array('delta', [N], dace.float64)
    sdfg.add_scalar('_tmp', dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N', update_expr='i = i + 1', loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    # State 1: copy delta[i] into a transient scalar -- StateFusion can fuse this
    # with state 2 because the iedge is trivial and the transient is per-iteration.
    s1 = loop.add_state('compute_tmp', is_start_block=True)
    s2 = loop.add_state('apply')
    loop.add_edge(s1, s2, dace.InterstateEdge())

    dd = s1.add_read('delta')
    tw = s1.add_write('_tmp')
    s1.add_nedge(dd, tw, dace.Memlet(data='delta', subset='i', other_subset='0'))

    rd = s2.add_read('out')
    rt = s2.add_read('_tmp')
    wt = s2.add_write('out')
    t = s2.add_tasklet('scan_step', {'_a', '_d'}, {'_o'}, '_o = _a + _d')
    s2.add_edge(rd, None, t, '_a', dace.Memlet(data='out', subset='i'))
    s2.add_edge(rt, None, t, '_d', dace.Memlet(data='_tmp', subset='0'))
    s2.add_edge(t, '_o', wt, None, dace.Memlet(data='out', subset='i + 1'))
    sdfg.validate()

    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, ('v5: StateFusion preprocess should fuse the two-state body, then '
                      f'the scan matcher accepts; got {res}')
    assert _num_scan_nodes(sdfg) == 1

    n = 14
    rng = np.random.default_rng(50)
    delta = rng.uniform(-1.0, 1.0, size=n)
    out = np.zeros(n + 1)
    out[0] = 0.3
    expected = out.copy()
    for i in range(n):
        expected[i + 1] = expected[i] + delta[i]
    sdfg(out=out, delta=delta, N=n)
    assert np.allclose(out, expected), f'two-state-body scan diverged: {out} vs {expected}'


def test_v5_multi_write_an_per_carrier_in_fused_body():
    """cloudsc ``pfsqrf`` after v5 fusion: body has TWO write ANs for the same carrier
    ``out`` (from the pre-fuse state-1 and state-2 writes, carried as separate sink
    nodes). ``_find_unique_write_edge`` refuses the multi-write-AN case, so the match
    fails even though the carry is structurally valid. state 1 writes ``out[i]``
    (side-effect), state 2 writes ``out[i+1]`` (scan carry). The matcher must pick the
    scan-carry write (loop-var-indexed subset matching the recurrence) and ignore the
    side-effect write.
    """
    sdfg = dace.SDFG('multi_write_an_fused')
    sdfg.add_array('out', [N + 1], dace.float64)
    sdfg.add_array('delta', [N], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N', update_expr='i = i + 1', loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    s1 = loop.add_state('side_effect', is_start_block=True)
    s2 = loop.add_state('scan_step')
    loop.add_edge(s1, s2, dace.InterstateEdge())

    # State 1: write a side-effect to out[i] (mimics the cloudsc pre-scan compute).
    t1 = s1.add_tasklet('side', {'_d'}, {'_o'}, '_o = _d * 0.5')
    dd = s1.add_read('delta')
    ow1 = s1.add_write('out')
    s1.add_edge(dd, None, t1, '_d', dace.Memlet(data='delta', subset='i'))
    s1.add_edge(t1, '_o', ow1, None, dace.Memlet(data='out', subset='i'))

    # State 2: the scan step. Reads out[i], writes out[i+1].
    rd = s2.add_read('out')
    wt = s2.add_write('out')
    dd2 = s2.add_read('delta')
    t2 = s2.add_tasklet('scan_step', {'_a', '_d'}, {'_o'}, '_o = _a + _d')
    s2.add_edge(rd, None, t2, '_a', dace.Memlet(data='out', subset='i'))
    s2.add_edge(dd2, None, t2, '_d', dace.Memlet(data='delta', subset='i'))
    s2.add_edge(t2, '_o', wt, None, dace.Memlet(data='out', subset='i + 1'))
    sdfg.validate()

    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, ('Multi-write-AN per carrier in the fused body: matcher should pick the scan-'
                      f'carry write (out[i+1]) and ignore the side-effect write (out[i]); got {res}')
    assert _num_scan_nodes(sdfg) == 1


def test_v6_negative_write_offset_scan_with_outer_axis():
    """cloudsc ``for_1134``: 2-D scan ``arr[outer, inner-1] = arr[outer, inner-2] +
    delta[outer, inner]``. Scan-axis write offset NEGATIVE (k_w = -1), carry-read offset
    -2; outer axis loop-invariant. The synthetic frontend version matches today; the
    cloudsc-specific blocker (likely a delta-chain peculiarity) is tracked separately.
    """

    @dace.program
    def neg_offset_scan(arr: dace.float64[5, N + 2], delta: dace.float64[5, N + 2]):
        for j in range(5):
            for i in range(2, N + 2):
                arr[j, i - 1] = arr[j, i - 2] + delta[j, i]

    sdfg = neg_offset_scan.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, (f'k_w = -1 scan with outer axis should match; got {res}')
    assert _num_scan_nodes(sdfg) == 1


KLEV = dace.symbol('KLEV')
KLON = dace.symbol('KLON')


@dace.program
def _pfsqrf_2d_nested(pfsqrf: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
    """The cloudsc ``for_1133`` shape: outer scan over levels ``jk``, inner data-parallel
    column loop ``jl``, prefix-sum recurrence on ``pfsqrf`` along ``jk``."""
    for jk in range(1, KLEV):
        for jl in range(KLON):
            pfsqrf[jk, jl] = pfsqrf[jk - 1, jl] + delta[jk, jl]


def test_cloudsc_for_1133_shape_nested_inner_loopregion():
    """cloudsc ``for_1133``: outer scan over ``jk`` containing an inner data-parallel
    ``jl`` column loop, prefix-sum on ``pfsqrf``. ``_match_all`` descends one level into
    the inner ``LoopRegion`` for the scan-update tasklet; the rewrite emits one ``Scan``
    libnode with ``stride = inner_size`` (residue-class) over a contiguous
    ``[trip, inner_size]`` delta buffer + a ``Map[(i, j)]`` seed-add. Matches oracle.
    """
    import numpy as np
    sdfg = _pfsqrf_2d_nested.to_sdfg(simplify=True)
    # Nested (vector) scan lift is opt-in (the default keeps the inner map -- see
    # the ``lift_nested_scan`` Property); this test exercises the lift path.
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the for_1133 prefix-sum shape; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1

    KL, KO = 5, 4
    rng = np.random.default_rng(42)
    p_init = rng.random((KL, KO))
    d = rng.random((KL, KO))
    p_ref = p_init.copy()
    for jk in range(1, KL):
        for jl in range(KO):
            p_ref[jk, jl] = p_ref[jk - 1, jl] + d[jk, jl]
    p_test = p_init.copy()
    sdfg(pfsqrf=p_test, delta=d, KLEV=KL, KLON=KO)
    assert np.allclose(p_test, p_ref), (f'Vector-scan rewrite must match the sequential oracle; max diff '
                                        f'{np.abs(p_test - p_ref).max()}')


def _build_for_1133_post_l2m_sdfg():
    """Build the post-``LoopToMap`` for_1133 fixture: outer carry
    ``LoopRegion[jk]`` containing a single state with the inner column
    ``Map[jl]`` that ``LoopToMap`` has already lifted."""
    from dace.transformation.interstate.loop_to_map import LoopToMap
    sdfg = _pfsqrf_2d_nested.to_sdfg(simplify=True)
    inner = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable == 'jl')
    xform = LoopToMap()
    xform.loop = inner
    xform.expr_index = 0
    assert xform.can_be_applied(inner.parent_graph, 0, sdfg, permissive=False), \
        'inner jl-loop must be parallel for this test fixture'
    xform.apply(inner.parent_graph, sdfg)
    sdfg.validate()
    return sdfg


def _for_1133_oracle(klev, klon, p_init, d):
    """Sequential oracle for the for_1133 prefix sum."""
    expected = p_init.copy()
    for jk in range(1, klev):
        for jl in range(klon):
            expected[jk, jl] = expected[jk - 1, jl] + d[jk, jl]
    return expected


def test_cloudsc_for_1133_detection_off_keeps_post_l2m_shape():
    """With ``interchange_carry_with_map=False`` (the default), ``LoopToScan``
    must NOT touch the post-L2M shape: the outer carry ``LoopRegion`` and the
    inner parallel ``Map`` stay in place, and no Scan libnode is emitted.
    Numerics still match (the SDFG just executes the original nested loop)."""
    import numpy as np
    sdfg = _build_for_1133_post_l2m_sdfg()
    LoopToScan(interchange_carry_with_map=False).apply_pass(sdfg, {})
    sdfg.validate()
    n_loops = sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    n_maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry))
    assert n_loops == 1, f'outer carry LoopRegion must be preserved when knob is off; got {n_loops} LoopRegions'
    assert n_maps == 1, f'inner parallel Map must be preserved when knob is off; got {n_maps} MapEntries'
    assert _num_scan_nodes(sdfg) == 0, 'no Scan libnode must be emitted when knob is off'

    KL, KO = 6, 4
    rng = np.random.default_rng(1133)
    p_init = rng.standard_normal((KL, KO))
    d = rng.standard_normal((KL, KO))
    p_ref = _for_1133_oracle(KL, KO, p_init, d)
    p_got = p_init.copy()
    sdfg(pfsqrf=p_got, delta=d.copy(), KLEV=KL, KLON=KO)
    assert np.allclose(p_got, p_ref), \
        f'post-L2M execution must match the oracle even without the interchange; max diff {np.abs(p_got - p_ref).max()}'


def test_cloudsc_for_1133_detection_on_interchanges_to_map_over_scan():
    """``interchange_carry_with_map=True``: interchange by relocation -- the outer carry
    ``LoopRegion[jk]`` moves from the top SDFG INTO the NestedSDFG inside the parallel
    ``Map[jl]``. Each Map thread runs its own sequential ``for jk`` reading/writing
    ``pfsqrf``/``delta`` straight from global memory -- no buffers, no Scan libnode, no
    copies.

    Post-conditions:
      * 0 LoopRegions in the top SDFG (carry now one level down).
      * exactly 1 MapEntry (the interchanged parallel axis).
      * 0 Scan libnodes.
      * 0 transients introduced by the rewrite.
      * exactly 1 LoopRegion inside the Map-body NSDFG, with the original carry var.
      * numeric match vs the sequential oracle.
    """
    import numpy as np
    sdfg = _build_for_1133_post_l2m_sdfg()
    LoopToScan(interchange_carry_with_map=True).apply_pass(sdfg, {})
    sdfg.validate()
    n_loops_top = sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    n_maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry))
    assert n_loops_top == 0, f'outer carry LoopRegion must be gone from the top SDFG; got {n_loops_top}'
    assert n_maps == 1, f'exactly one MapEntry (the interchanged parallel axis) must remain; got {n_maps}'
    assert _num_scan_nodes(sdfg) == 1 or _num_scan_nodes(sdfg) == 0
    assert _num_scan_nodes(sdfg) == 0, \
        f'NO Scan libnode must be emitted by the buffer-free interchange path; got {_num_scan_nodes(sdfg)}'
    # No interchange-introduced buffers anywhere in the SDFG.
    bad_transients = [(sd.name, name) for sd in sdfg.all_sdfgs_recursive() for name, desc in sd.arrays.items()
                      if getattr(desc, 'transient', False) and name.startswith('_interchange_')]
    assert not bad_transients, f'no per-column buffer must be introduced; got {bad_transients}'
    # Inner sequential LoopRegion must now live inside the Map-body NSDFG.
    inner_loops = []
    for sd in sdfg.all_sdfgs_recursive():
        if sd is sdfg:
            continue
        inner_loops += [n for n in sd.all_control_flow_regions() if isinstance(n, LoopRegion)]
    assert len(inner_loops) == 1, f'exactly one inner LoopRegion (sequential carry) expected; got {len(inner_loops)}'

    KL, KO = 6, 4
    rng = np.random.default_rng(1133)
    p_init = rng.standard_normal((KL, KO))
    d = rng.standard_normal((KL, KO))
    p_ref = _for_1133_oracle(KL, KO, p_init, d)
    p_got = p_init.copy()
    sdfg(pfsqrf=p_got, delta=d.copy(), KLEV=KL, KLON=KO)
    assert np.allclose(p_got, p_ref), \
        f'interchange (buffer-free) must match the oracle; max diff {np.abs(p_got - p_ref).max()}'


def test_cloudsc_for_1133_shape_after_inner_l2m():
    """Original blocker test, historical name. Calls ``LoopToScan()`` with the knob
    opted in -- previously the default refused this (xfail-strict); the interchange path
    now lifts it (no Scan libnode; the carry loop relocates into the per-thread NSDFG).
    """
    sdfg = _build_for_1133_post_l2m_sdfg()
    res = LoopToScan(interchange_carry_with_map=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the for_1133 prefix-sum shape once the inner '
                                          f'column loop is a Map; got res={res}.')
    # The buffer-free interchange path emits 0 Scan libnodes.
    assert _num_scan_nodes(sdfg) == 0


# -----------------------------------------------------------------------------
# Refusal-mode probes for the cloudsc pfsqXf shapes. Each exercises ONE failure gate
# in ``LoopToScan._match_all`` that the cloudsc-actual bodies trip. Marked
# ``xfail(strict=True)`` -> when the matcher is extended the test XPASSes and forces
# removing the marker, locking in the extension.
# -----------------------------------------------------------------------------


def test_outer_body_with_extra_content_state_alongside_inner_loop():
    """Outer scan ``jk`` body has both the inner column loop AND a separate state
    preparing a transient slice (cloudsc frontend materializes per-iter slices in their
    own state). The extra state writes only a transient (``tmp``), so the matcher accepts
    once descent tolerates extra content states."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])
    import numpy as _np

    @dace.program
    def with_slice_state(arr: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
        for jk in range(1, KLEV):
            # Transient row-slice: lives only across this jk iteration.
            tmp = _np.empty(KLON, dtype=_np.float64)
            for jl in range(KLON):
                tmp[jl] = delta[jk, jl] * 2.0
            for jl in range(KLON):
                arr[jk, jl] = arr[jk - 1, jl] + tmp[jl]

    sdfg = with_slice_state.to_sdfg(simplify=True)
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the for_1133 shape even with extra body states; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_inner_loop_with_multi_state_body():
    """Inner column loop body has multiple content statements at the Python
    level (split into separate frontend states pre-simplify). ``_fuse_body_states``
    + the existing transient-chain walk together handle the post-simplify shape.
    Positive lockdown: confirms the matcher accepts the multi-statement inner."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])

    @dace.program
    def multi_inner(arr: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
        for jk in range(1, KLEV):
            for jl in range(KLON):
                # Two-statement inner body. The frontend may keep two states.
                tmp_val: dace.float64 = arr[jk - 1, jl] + delta[jk, jl]
                arr[jk, jl] = tmp_val

    sdfg = multi_inner.to_sdfg(simplify=True)
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the shape even when the inner body is multi-state; '
                                          f'got res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_carrier_with_extra_constant_axis_besides_inner_var():
    """3-D carrier where one axis is constant in the loop (e.g. species
    index), one is the outer scan axis (``jk``), one is the inner Map axis
    (``jl``). The cloudsc pfsqXf are scoped per-species; the species index
    is a constant non-scan axis the matcher must accept."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])

    @dace.program
    def per_species(arr: dace.float64[3, KLEV, KLON], delta: dace.float64[3, KLEV, KLON]):
        # Hard-coded species index 1 (the constant non-scan axis).
        for jk in range(1, KLEV):
            for jl in range(KLON):
                arr[1, jk, jl] = arr[1, jk - 1, jl] + delta[1, jk, jl]

    sdfg = per_species.to_sdfg(simplify=True)
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift a 3-D carrier with one constant non-scan axis; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_carrier_read_through_two_hop_transient_chain():
    """Synthesize a 2-hop transient slice on the carrier read side. Hard to
    force naturally from the Python frontend; this test uses an explicit
    SDFG-level helper to plant the second-hop transient."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])

    @dace.program
    def two_hop(arr: dace.float64[KLEV, KLON], delta: dace.float64[KLEV, KLON]):
        for jk in range(1, KLEV):
            for jl in range(KLON):
                # The double-slice forces two intermediates: row slice then column dereference
                row = arr[jk - 1, :]
                arr[jk, jl] = row[jl] + delta[jk, jl]

    sdfg = two_hop.to_sdfg(simplify=True)
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should walk through a 2-hop transient slice chain; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_carrier_with_computed_delta_chain():
    """Delta is a computed expression (multiply + add) of two arrays rather than a direct
    single-array read. cloudsc pfsqXf: ``delta = (zqxn2d[i] - zqx0[i]) * zgdph_r``."""
    KLEV = dace.symbol('KLEV')

    @dace.program
    def computed_delta(out: dace.float64[KLEV], a: dace.float64[KLEV], b: dace.float64[KLEV], c: dace.float64[KLEV]):
        for jk in range(1, KLEV):
            out[jk] = out[jk - 1] + (a[jk] - b[jk]) * c[jk]

    sdfg = computed_delta.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should accept a computed-expression delta; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_cloudsc_for_1133_shape_reverse_engineered_from_fortran():
    """Mimics the Fortran cloudsc ``DO JK ...`` body producing ``for_1133``: per-level
    slice-copy then per-species accumulation. FaCe-emitted cloudsc SDFG shape -- a
    5-array prefix-sum on (jl, jk) with a computed delta over jm."""
    KLEV, KLON, NCLV = (dace.symbol(s) for s in ['KLEV', 'KLON', 'NCLV'])

    @dace.program
    def cloudsc_like(pfsqif: dace.float64[KLON, KLEV + 1], zqxn2d: dace.float64[KLON, NCLV],
                     zqx0: dace.float64[KLON, NCLV], zgdph_r: dace.float64):
        for jk in range(1, KLEV + 1):
            # Slice-copy: carry level-1 value forward
            for jl in range(KLON):
                pfsqif[jl, jk] = pfsqif[jl, jk - 1]
            # Per-species accumulation (the actual delta computation)
            for jm in range(NCLV):
                for jl in range(KLON):
                    pfsqif[jl, jk] = pfsqif[jl, jk] + (zqxn2d[jl, jm] - zqx0[jl, jm]) * zgdph_r

    sdfg = cloudsc_like.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the cloudsc-like for_1133 body shape; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1

    klev, klon, nclv = 8, 4, 3
    rng = np.random.default_rng(1133)
    pfsqif = rng.standard_normal((klon, klev + 1))
    zqxn2d = rng.standard_normal((klon, nclv))
    zqx0 = rng.standard_normal((klon, nclv))
    zgdph_r = 0.137
    expected = pfsqif.copy()
    for jk in range(1, klev + 1):
        for jl in range(klon):
            expected[jl, jk] = expected[jl, jk - 1]
        for jm in range(nclv):
            for jl in range(klon):
                expected[jl, jk] = expected[jl, jk] + (zqxn2d[jl, jm] - zqx0[jl, jm]) * zgdph_r
    sdfg(pfsqif=pfsqif, zqxn2d=zqxn2d, zqx0=zqx0, zgdph_r=zgdph_r, KLEV=klev, KLON=klon, NCLV=nclv)
    assert np.allclose(pfsqif, expected), (f'Composite-body scan numerics must match the sequential oracle; '
                                           f'max diff = {np.abs(pfsqif - expected).max()}')


def test_backward_stride_minus_one_prefix_sum():
    """The cloudsc ``for_1079`` shape: backward iteration with carry on the
    next-higher index. ``jm`` runs from ``NCLV - 1`` down to ``1`` (stride
    -1); ``acc[jm-1] = acc[jm] + delta[jm]`` is a right-to-left prefix sum.
    LoopToScan must accept this by reversing iteration semantics."""

    @dace.program
    def backward_scan(acc: dace.float64[N + 1], delta: dace.float64[N + 1]):
        for jm in range(N, 0, -1):
            acc[jm - 1] = acc[jm] + delta[jm]

    sdfg = backward_scan.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should accept a backward-stride (-1) prefix sum; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1

    n = 16
    rng = np.random.default_rng(1079)
    acc = rng.standard_normal(n + 1)
    delta = rng.standard_normal(n + 1)
    expected = acc.copy()
    for jm in range(n, 0, -1):
        expected[jm - 1] = expected[jm] + delta[jm]
    sdfg(acc=acc, delta=delta, N=n)
    assert np.allclose(acc, expected), (f'Backward-stride scan numerics must match the sequential oracle; '
                                        f'max diff = {np.abs(acc - expected).max()}')


def test_multi_slot_carrier_is_refused_not_miscompiled():
    """cloudsc ``for_430``: multiple constant-index slots on the SAME carrier in a level
    loop (``zvqx[0/1/2, jk, jl]``). Each slot is an independent prefix recurrence along
    the level axis, but the multi-slot rewrite chains a per-slot ``_rewrite`` over the
    shared loop and fails to capture each slot's external seed (``zvqx[r, 0, jl]``),
    reading an uninitialised buffer -> lift is numerically WRONG (maxdiff ~0.85 vs
    oracle). LoopToScan must REFUSE the multi-slot shape (leave sequential) even with
    ``lift_nested_scan=True``. Guards the silent miscompile: old code lifted (3 Scan
    libnodes) and diverged; correct = no lift + values equal the oracle."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])

    @dace.program
    def multi_slot(zvqx: dace.float64[5, KLEV, KLON], delta: dace.float64[KLEV, KLON]):
        for jk in range(1, KLEV):
            for jl in range(KLON):
                zvqx[0, jk, jl] = zvqx[0, jk - 1, jl] + delta[jk, jl] * 0.1
                zvqx[1, jk, jl] = zvqx[1, jk - 1, jl] + delta[jk, jl] * 0.2
                zvqx[2, jk, jl] = zvqx[2, jk - 1, jl] + delta[jk, jl] * 0.3

    sdfg = multi_slot.to_sdfg(simplify=True)
    # Even with nested-scan lifting enabled, the multi-slot carrier is refused.
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert not res, f'multi-slot carrier must be refused (unsound rewrite), got res={res}'
    assert _num_scan_nodes(sdfg) == 0

    # Numerical correctness: the un-lifted (sequential) SDFG matches the oracle.
    klev, klon = 6, 4
    rng = np.random.default_rng(430)
    zvqx = rng.random((5, klev, klon))
    delta = rng.random((klev, klon))
    ref = zvqx.copy()
    for jk in range(1, klev):
        for jl in range(klon):
            ref[0, jk, jl] = ref[0, jk - 1, jl] + delta[jk, jl] * 0.1
            ref[1, jk, jl] = ref[1, jk - 1, jl] + delta[jk, jl] * 0.2
            ref[2, jk, jl] = ref[2, jk - 1, jl] + delta[jk, jl] * 0.3
    got = zvqx.copy()
    sdfg(zvqx=got, delta=delta, KLEV=klev, KLON=klon)
    assert np.allclose(got[:3], ref[:3]), f'max-diff {np.abs(got[:3] - ref[:3]).max()}'


def test_scan_with_conditional_body_descends_into_if():
    """``out[i+1] = out[i] + delta[i]`` guarded by a mask -- the carry update sits inside
    a ``ConditionalBlock``. LoopToScan must descend into the if/else branches to see the
    recurrence. Verifies structural lift (Scan emitted) AND numeric match, so both the
    matched-branch scan-update and the OTHER branch's carrier writes are handled.
    """

    @dace.program
    def conditional_scan(out: dace.float64[N + 1], delta: dace.float64[N], mask: dace.int32[N]):
        for i in range(N):
            if mask[i] > 0:
                out[i + 1] = out[i] + delta[i]
            else:
                out[i + 1] = out[i]

    sdfg = conditional_scan.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (
        f'LoopToScan should lift the conditional scan shape by descending into '
        f'the if/else branches; got res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1

    n = 16
    rng = np.random.default_rng(99)
    delta = rng.uniform(-1.0, 1.0, size=n)
    mask = rng.integers(0, 2, size=n).astype(np.int32)
    out = np.zeros(n + 1)
    out[0] = 0.5
    expected = out.copy()
    for i in range(n):
        if mask[i] > 0:
            expected[i + 1] = expected[i] + delta[i]
        else:
            expected[i + 1] = expected[i]
    sdfg(out=out, delta=delta, mask=mask, N=n)
    assert np.allclose(out, expected), (f'Conditional-carry numerics must match the sequential oracle; got {out}, '
                                        f'expected {expected}, max-diff {np.abs(out - expected).max()}')


# -----------------------------------------------------------------------------
# Scalar-carry prefix scan (TSVC s3112 family).
# -----------------------------------------------------------------------------


def test_scalar_carry_inclusive_sum_s3112():
    """``sum=0; for i: sum+=a[i]; b[i]=sum`` -- the canonical scalar-carry shape.
    After the rewrite the loop is gone, replaced by a delta-build Map -> Scan ->
    out-write Map chain. ``b`` matches the running prefix sum of ``a``."""

    @dace.program
    def s3112(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
            b[i] = sum_val

    sdfg = s3112.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1
    assert _num_loops(sdfg) == 0

    n = 16
    rng = np.random.default_rng(3112)
    a = rng.uniform(-1.0, 1.0, size=n)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, np.cumsum(a))


def test_scalar_carry_inclusive_product():
    """Multiplicative variant: ``prod *= a[i]; b[i] = prod``."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        prod = 1.0
        for i in range(N):
            prod = prod * a[i]
            b[i] = prod

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 8
    rng = np.random.default_rng(0)
    a = rng.uniform(0.95, 1.05, size=n)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, np.cumprod(a))


def test_scalar_carry_inclusive_max():
    """Max variant: ``m = max(m, a[i]); b[i] = m``. Matcher recognises
    ``max(__acc, __delta)`` as ``ScanOp.MAX``."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        m = -1e300
        for i in range(N):
            m = max(m, a[i])
            b[i] = m

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 12
    rng = np.random.default_rng(7)
    a = rng.uniform(-3.0, 3.0, size=n)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, np.maximum.accumulate(a))


def test_scalar_carry_inclusive_min():
    """Min variant: ``m = min(m, a[i]); b[i] = m``."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        m = 1e300
        for i in range(N):
            m = min(m, a[i])
            b[i] = m

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 9
    rng = np.random.default_rng(11)
    a = rng.uniform(-2.0, 2.0, size=n)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, np.minimum.accumulate(a))


def test_scalar_carry_refuses_extra_non_prefix_use():
    """``sum+=a[i]; b[i]=sum; c[i]=sum*2.0`` -- the accumulator is read for a
    NON-prefix purpose (the ``c[i]=sum*2.0`` write). Refuse: the matcher
    requires the post-RMW accumulator to feed exactly one per-iter output."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
            b[i] = sum_val
            c[i] = sum_val * 2.0

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    # Refused -- the original loop survives, no Scan emitted.
    assert res is None or res == 0
    assert _num_scan_nodes(sdfg) == 0
    assert _num_loops(sdfg) == 1


def test_scalar_carry_refuses_overwrite_not_rmw():
    """``sum = a[i]; b[i] = sum`` -- the accumulator is OVERWRITTEN each iter,
    not RMW-updated. No recurrence, so no scan; refuse."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            sum_val = a[i]
            b[i] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None or res == 0
    assert _num_scan_nodes(sdfg) == 0


def test_scalar_carry_acc_used_post_loop_emits_writeback():
    """Scalar accumulator read AFTER the loop (TSVC s319-style: ``b[N-1] = sum_val``):
    the rewrite emits a writeback state copying the last scan-output element into
    ``acc[0]`` so downstream readers see the final running value (matching the
    pre-rewrite sequential post-loop scalar)."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], result: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
            b[i] = sum_val
        result[0] = sum_val  # post-loop use forces the writeback

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_scan_nodes(sdfg) == 1

    n = 10
    rng = np.random.default_rng(319)
    a = rng.uniform(-1.0, 1.0, size=n)
    b = np.zeros(n)
    result = np.zeros(1)
    sdfg(a=a, b=b, result=result, N=n)
    assert np.allclose(b, np.cumsum(a))
    assert np.isclose(result[0], a.sum())


def test_scalar_carry_acc_not_used_post_loop_no_writeback():
    """When the accumulator is NOT read after the loop, the rewrite skips
    the writeback state -- fewer ops, no dead writes."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
            b[i] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    # Inspect: the rewrite added s_build, s_scan, s_write. The writeback state
    # name suffix is ``_scan_acc_post`` -- absent here.
    state_labels = {s.label for s in sdfg.all_states()}
    assert not any('_scan_acc_post' in lbl for lbl in state_labels)


def test_scalar_carry_preserves_iedge_assignments_on_loop_boundary():
    """The rewriter must preserve interstate-edge assignments on the loop's in/out edges
    (symbol bindings the canonicalize pipeline cascades onto loop boundaries).
    Synthesizes a pre-loop iedge assignment and checks it survives.
    """
    import dace

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
            b[i] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    # Attach a trivial assignment to the iedge feeding the loop.
    loop = next(r for r in sdfg.all_control_flow_regions()
                if isinstance(r, dace.sdfg.state.LoopRegion) and r.loop_variable)
    parent = loop.parent_graph
    in_edges = list(parent.in_edges(loop))
    assert in_edges, 'test fixture: loop should have at least one in-edge'
    # Add a marker assignment to the first in-edge.
    in_edges[0].data.assignments['_marker_pre_loop'] = '42'

    LoopToScan().apply_pass(sdfg, {})

    # After the rewrite, the marker must still be on an iedge feeding the new
    # head state (``*_scan_build``).
    new_head = next(s for s in sdfg.all_states() if s.label.endswith('_scan_build'))
    head_in_edges = list(sdfg.in_edges(new_head))
    found = any(e.data.assignments.get('_marker_pre_loop') == '42' for e in head_in_edges)
    assert found, 'iedge assignment ``_marker_pre_loop=42`` lost during rewrite'


_LS_N = dace.symbol('LS_N')
_LS_K = dace.symbol('LS_K')


@dace.program
def _prefix_scan_with_outside_seed(fall: dace.float64[_LS_N, _LS_K], flux: dace.float64[_LS_N, _LS_K]):
    """Forward prefix scan ``flux[k] = flux[k-1] * 0.9 + fall[k]`` where the
    SEED ``flux[0] = fall[0]`` is initialised in a sibling block of the inner
    ``k`` loop. cloudsc ZPFPLSX sedimentation shape."""
    for i in range(_LS_N):
        flux[i, 0] = fall[i, 0]
        for k in range(1, _LS_K):
            flux[i, k] = flux[i, k - 1] * 0.9 + fall[i, k]


@dace.program
def _backward_recurrence_with_outside_seed(rhs: dace.float64[_LS_N, _LS_K], x: dace.float64[_LS_N, _LS_K]):
    """Backward recurrence ``x[k] = rhs[k] - 0.5 * x[k+1]`` where the seed
    ``x[K-1] = rhs[K-1]`` is initialised in a sibling block of the inner
    ``k`` loop. Mirrors Thomas-solver backward substitution."""
    for i in range(_LS_N):
        x[i, _LS_K - 1] = rhs[i, _LS_K - 1]
        for k in range(_LS_K - 2, -1, -1):
            x[i, k] = rhs[i, k] - 0.5 * x[i, k + 1]


def _scan_libnodes(sdfg: dace.SDFG) -> int:
    """Count ``Scan`` library nodes anywhere in ``sdfg``. ``LoopToScan`` lowers
    each accepted recurrence into one such node."""
    n = 0
    for node, _ in sdfg.all_nodes_recursive():
        if type(node).__name__ == 'Scan':
            n += 1
    return n


def test_refuses_when_carrier_has_sibling_seed_write_forward():
    """``LoopToScan`` must refuse a recurrence whose carrier is also written in a SIBLING
    block (the seed init ``flux[0] = fall[0]`` just before the inner ``for k``
    prefix-scan). Pre-fix: the pass matched the inner ``k`` loop and emitted a ``Scan``;
    codegen built a carry buffer whose seed slot wasn't populated from the sibling write,
    so output diverged from the oracle. Asserts (a) no ``Scan`` libnode, (b) end-to-end
    match vs the sequential oracle."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    n, k = 7, 8
    sdfg = _prefix_scan_with_outside_seed.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert _scan_libnodes(sdfg) == 0, ('LoopToScan must refuse a recurrence whose carrier has a sibling seed write; '
                                       f'got {_scan_libnodes(sdfg)} Scan libnodes')

    rng = np.random.default_rng(21)
    fall = rng.uniform(0.0, 1.0, (n, k))
    flux = np.zeros((n, k))
    sdfg(fall=fall.copy(), flux=flux, LS_N=n, LS_K=k)

    expected = np.zeros((n, k))
    for ii in range(n):
        expected[ii, 0] = fall[ii, 0]
        for kk in range(1, k):
            expected[ii, kk] = expected[ii, kk - 1] * 0.9 + fall[ii, kk]
    assert np.allclose(flux, expected), (f'value mismatch: max diff {np.abs(flux - expected).max():.2e}')


def test_refuses_when_carrier_has_sibling_seed_write_backward():
    """Forward-case contract on a backward recurrence (Thomas back-substitution).
    ``NormalizeNegativeStride`` rewrites to forward iteration; the matcher sees a coef=-1
    carry-read; the sibling-seed-write check on the original ``x[K-1] = ...`` initialiser
    also covers this."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    n, k = 4, 6
    sdfg = _backward_recurrence_with_outside_seed.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert _scan_libnodes(sdfg) == 0, (f'LoopToScan must refuse the backward recurrence with outside seed; '
                                       f'got {_scan_libnodes(sdfg)} Scan libnodes')

    rng = np.random.default_rng(31)
    rhs = rng.standard_normal((n, k))
    x = np.zeros((n, k))
    sdfg(rhs=rhs.copy(), x=x, LS_N=n, LS_K=k)

    expected = np.zeros((n, k))
    for ii in range(n):
        expected[ii, k - 1] = rhs[ii, k - 1]
        for kk in range(k - 2, -1, -1):
            expected[ii, kk] = rhs[ii, kk] - 0.5 * expected[ii, kk + 1]
    assert np.allclose(x, expected), f'value mismatch: max diff {np.abs(x - expected).max():.2e}'


@dace.program
def _prefix_scan_with_in_kernel_1d_seed(a: dace.float64[N], x: dace.float64[N]):
    """Forward FLAT 1-D prefix sum ``a[i] = a[i-1] + x[i]`` whose seed ``a[0] = x[0]`` is
    set in a sibling block (TSVC ``fission_dep_then_indep`` after ``LoopFission`` isolates
    the carried body). This lifts (vs the per-row 2-D seed shapes): the seed-add reads
    ``a[0]`` straight from the live array; the scan writes only ``a[1..]``, never the seed
    slot."""
    a[0] = x[0]
    for i in range(1, N):
        a[i] = a[i - 1] + x[i]


@dace.program
def _stride2_prefix_scan_with_in_kernel_1d_seed(a: dace.float64[N], x: dace.float64[N]):
    """Stride-2 sibling of the above (``a[i] = a[i-2] + x[i]``, seeds ``a[0]``,
    ``a[1]``; TSVC ``fission_dep_const_offset``). Two residue-class scans; the
    two seed slots are again read live and never overwritten."""
    a[0] = x[0]
    a[1] = x[1]
    for i in range(2, N):
        a[i] = a[i - 2] + x[i]


@pytest.mark.parametrize('prog,seed_len', [(_prefix_scan_with_in_kernel_1d_seed, 1),
                                           (_stride2_prefix_scan_with_in_kernel_1d_seed, 2)])
def test_lifts_forward_flat_1d_scan_with_in_kernel_seed(prog, seed_len):
    """A forward FLAT 1-D scan whose seed is written in a sibling block DOES lift
    (contrast the refused per-row 2-D seed shapes): the flat seed-add reads the
    untouched seed slot from the live array. Asserts a single Scan libnode (the
    stride-``seed_len`` residue classes are encoded in one Scan's ``stride``) and
    end-to-end numeric match vs the sequential oracle."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    sdfg = prog.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert _scan_libnodes(sdfg) == 1, (f'expected one (stride-{seed_len}) Scan libnode, got {_scan_libnodes(sdfg)}')

    n = 16
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    a = np.zeros(n)
    expected = np.zeros(n)
    for s in range(seed_len):
        expected[s] = x[s]
    for i in range(seed_len, n):
        expected[i] = expected[i - seed_len] + x[i]
    sdfg(a=a, x=x.copy(), N=n)
    assert np.allclose(a, expected), f'value mismatch: max diff {np.abs(a - expected).max():.2e}'


@dace.program
def _two_step_descending_recurrence(a: dace.float64[_LS_N], b: dace.float64[_LS_N]):
    """``b[i] = 0.5 * b[i+1] + 0.25 * b[i+2] + a[i]`` -- two-deep loop-carried
    descending recurrence. The matcher accepts the 1-step ``b[i+1]`` carry
    but the body ALSO reads ``b[i+2]`` directly; that second read isn't
    routed through the scan's carry buffer and reads the wrong value once
    iterations are reordered."""
    for i in range(_LS_N - 3, -1, -1):
        b[i] = 0.5 * b[i + 1] + 0.25 * b[i + 2] + a[i]


def test_refuses_multi_step_recurrence_with_multiple_carrier_reads():
    """``LoopToScan`` must refuse a recurrence where the carrier is read at
    more than one offset (multi-step dependency). The scan rewrite emits a
    single 1-step carry buffer; a second read of the carrier at a different
    subset stays as a direct array load and sees the wrong value once the
    scan reorders iterations."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    n = 21
    sdfg = _two_step_descending_recurrence.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert _scan_libnodes(sdfg) == 0, (f'LoopToScan must refuse a 2-step recurrence; '
                                       f'got {_scan_libnodes(sdfg)} Scan libnodes')

    rng = np.random.default_rng(7)
    a = rng.standard_normal(n)
    b = np.zeros(n)
    b[n - 1] = 1.0
    b[n - 2] = 0.7
    sdfg(a=a.copy(), b=b, LS_N=n)
    expected = np.zeros(n)
    expected[n - 1] = 1.0
    expected[n - 2] = 0.7
    for i in range(n - 3, -1, -1):
        expected[i] = 0.5 * expected[i + 1] + 0.25 * expected[i + 2] + a[i]
    assert np.allclose(b, expected), f'value mismatch: max diff {np.abs(b - expected).max():.2e}'


def test_fuse_body_states_refuses_carry_through_state_boundary():
    """``_fuse_body_states`` must NOT merge two adjacent body states when the first reads
    a scalar the second writes (TSVC s252: ``a[i] = s + t`` in state 1 then ``t = s`` in
    state 2). The iedge orders the carry read before its write; collapsing removes that
    ordering, so codegen schedules ``t = s`` before ``s + t`` and the add sees the NEW
    ``t`` -- carrier RAW order lost. Fix: delegate the safety check to
    ``StateFusionExtended.can_be_applied`` (cross-state RAW/WAW hazards); slice 2.4b-B.
    Pinned with the real s252 kernel, verified end-to-end.
    """
    from tests.corpus.tsvc.tsvc import s252_d_single
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    sdfg = s252_d_single.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()

    n = 32
    rng = np.random.default_rng(252)
    a0 = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    a_exp = a0.copy()
    t = 0.0
    for i in range(n):
        s = b[i] * c[i]
        a_exp[i] = s + t
        t = s

    sa = a0.copy()
    sdfg(a=sa, b=b.copy(), c=c.copy(), LEN_1D=n)
    assert np.allclose(sa, a_exp), ('s252 carry broke after canonicalize: ``_fuse_body_states`` must '
                                    'refuse the body merge when the inter-state edge orders a carrier '
                                    'read before its write')


@dace.program
def nested_scan_parallel_inner(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    # j carries the recurrence (outer), i is the data-parallel inner axis -- the
    # shape LoopStridePermutation produces by moving the unit-stride axis inner.
    for j in range(1, N):
        for i in range(N):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def test_nested_scan_keeps_map_inside_by_default():
    """Default (``lift_nested_scan=False``): a carry loop wrapping a DOALL inner
    loop is NOT lifted -- it stays a sequential loop + (later) parallel map, so
    no ``Scan`` libnode is emitted. Running the un-lifted SDFG still matches."""
    sdfg = nested_scan_parallel_inner.to_sdfg(simplify=True)
    rng = np.random.default_rng(1)
    aa0 = rng.standard_normal((16, 16))
    bb = rng.standard_normal((16, 16))
    ref = aa0.copy()
    sdfg(aa=ref, bb=bb.copy(), N=16)

    LoopToScan().apply_pass(sdfg, {})
    assert _num_scan_nodes(sdfg) == 0, ('the carry loop wrapping a parallelizable inner loop must NOT '
                                        'be lifted by default (keep the map inside)')
    got = aa0.copy()
    sdfg(aa=got, bb=bb.copy(), N=16)
    assert np.allclose(got, ref)


def test_nested_scan_lifts_with_knob():
    """``lift_nested_scan=True``: the vector scan is still lifted (a ``Scan``
    libnode with a Map over the inner axis)."""
    sdfg = nested_scan_parallel_inner.to_sdfg(simplify=True)
    res = LoopToScan(lift_nested_scan=True).apply_pass(sdfg, {})
    assert res, "lift_nested_scan=True should lift the vector scan"
    assert _num_scan_nodes(sdfg) >= 1, "a Scan libnode is emitted when the knob opts in"


N_SS = dace.symbol('N_SS')
K_SS = dace.symbol('K_SS')


@dace.program
def _symbolic_stride_scan(a: dace.float64[N_SS], x: dace.float64[N_SS]):
    """Symbolic-stride prefix sum ``a[i] = a[i-K] + x[i]`` (TSVC-2.5
    ``scan_strided_sym``). ``K`` is a runtime symbol of unknown sign; the caller
    seeds ``a[0..K-1]``."""
    for i in range(K_SS, N_SS):
        a[i] = a[i - K_SS] + x[i]


def test_symbolic_stride_scan_specializes_if_scan_else_seq():
    """Recurrence with SYMBOLIC stride ``K`` (compile-time unknown sign) lifts to a
    residue-class ``Scan`` under an ``if K >= 1: scan else: sequential`` specialization.
    The decomposition into ``K`` independent prefix scans is valid only for ``K >= 1``;
    a violating value must degrade to the original loop -- hence the conditional."""
    sdfg = _symbolic_stride_scan.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()

    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f'expected exactly one specialization conditional, got {len(cbs)}'
    (cond, par_region), (else_cond, seq_region) = cbs[0].branches
    assert cond is not None and else_cond is None, 'branches must be (K >= 1 -> scan, else -> sequential)'
    assert 'K_SS' in cond.as_string and '>= 1' in cond.as_string, f'guard must be K >= 1: {cond.as_string!r}'
    # True branch: the residue-class Scan pipeline. Else branch: the pinned loop.
    assert _num_scan_nodes(par_region) == 1, 'the K >= 1 branch must hold exactly one residue-class Scan libnode'
    assert _num_scan_nodes(seq_region) == 0, 'the else branch must keep the sequential loop (no Scan)'
    seq_loops = [
        r for r in seq_region.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and r.loop_variable
    ]
    assert seq_loops and all(l.pinned_sequential for l in seq_loops), \
        'the else-branch fallback loop must be pinned sequential'


@pytest.mark.parametrize('k', [0, 1, 2, 3, 5])
def test_symbolic_stride_scan_value_exact(k):
    """Bit-exact vs the sequential recurrence for several strides -- including the
    degenerate ``K = 0`` that must take the sequential else-branch (a residue-class
    scan with stride 0 is undefined; the fallback computes ``a[i] += x[i]``)."""
    sdfg = _symbolic_stride_scan.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})

    n = 64
    rng = np.random.default_rng(100 + k)
    a0 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    got = a0.copy()
    sdfg(a=got, x=x, N_SS=n, K_SS=k)

    exp = a0.copy()
    for i in range(k, n):
        exp[i] = exp[i - k] + x[i]
    assert np.allclose(got, exp), f'value mismatch at K={k}: max diff {np.abs(got - exp).max():.2e}'


def _carried_writes_in_loops(sdfg, name):
    """Count AccessNodes of ``name`` that are WRITTEN (have an in-edge) inside any
    LoopRegion body -- i.e. a surviving sequential recurrence write."""
    total = 0
    for loop in sdfg.all_control_flow_regions():
        if not (isinstance(loop, LoopRegion) and loop.loop_variable):
            continue
        for st in loop.all_states():
            total += sum(1 for n in st.data_nodes() if n.data == name and st.in_degree(n) > 0)
    return total


def test_masked_conditional_scan_lifts_and_neutralizes_else_branch():
    """Masked prefix scan (TSVC-2.5 ``scan_conditional``): ``if mask[i]>0: out[i] =
    out[i-1] + delta[i] else: out[i] = out[i-1]``. The rewrite folds the masked
    delta into a pre-zeroed buffer + Scan; the else-branch's ``out[i]=out[i-1]``
    hold must be NEUTRALIZED (else the loop stays a live sequential recurrence,
    correct only by redundant recompute). Post-pass: one Scan, no carried ``out``
    write left in the loop body, and bit-exact vs the sequential oracle."""

    @dace.program
    def masked_scan(out: dace.float64[N], delta: dace.float64[N], mask: dace.int64[N]):
        for i in range(1, N):
            if mask[i] > 0:
                out[i] = out[i - 1] + delta[i]
            else:
                out[i] = out[i - 1]

    sdfg = masked_scan.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, f'expected one masked-scan rewrite; got {res}'
    assert _num_scan_nodes(sdfg) == 1
    # The fix: the loop body no longer carries a write to ``out`` (delta-build
    # only). ``out`` is written solely by the post-loop seed-add Map.
    assert _carried_writes_in_loops(sdfg, 'out') == 0, \
        'else-branch recurrence write to `out` was not neutralized'

    n = 32
    rng = np.random.default_rng(7)
    delta = rng.standard_normal(n)
    mask = (rng.standard_normal(n) > 0.0).astype(np.int64)
    out = rng.standard_normal(n)
    exp = out.copy()
    for i in range(1, n):
        exp[i] = exp[i - 1] + (delta[i] if mask[i] > 0 else 0.0)
    sdfg(out=out, delta=delta, mask=mask, N=n)
    assert np.allclose(out, exp), f'masked scan diverged: max diff {np.abs(out - exp).max():.2e}'


def test_multi_slot_same_array_five_carries():
    """Five INDEPENDENT prefix sums carried in one loop body writing distinct
    constant slots of ONE array (TSVC-2.5 ``scan_multi_5carry`` / cloudsc
    ``pfsqrf`` flat shape): ``acc[r, i] = acc[r, i-1] + delta[r, i]`` for
    ``r=0..4``. Lifts to five Scan libnodes (one per slot) via the dedicated
    multi-slot rewrite -- bit-exact vs the sequential oracle."""

    @dace.program
    def multi_slot(acc: dace.float64[5, N], delta: dace.float64[5, N]):
        for i in range(1, N):
            acc[0, i] = acc[0, i - 1] + delta[0, i]
            acc[1, i] = acc[1, i - 1] + delta[1, i]
            acc[2, i] = acc[2, i - 1] + delta[2, i]
            acc[3, i] = acc[3, i - 1] + delta[3, i]
            acc[4, i] = acc[4, i - 1] + delta[4, i]

    sdfg = multi_slot.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, f'expected one multi-slot loop rewrite; got {res}'
    assert _num_scan_nodes(sdfg) == 5, f'expected five per-slot Scan libnodes; got {_num_scan_nodes(sdfg)}'
    assert _carried_writes_in_loops(sdfg, 'acc') == 0, 'a per-slot recurrence write to `acc` survived'

    n = 24
    rng = np.random.default_rng(909)
    acc = rng.standard_normal((5, n))
    delta = rng.standard_normal((5, n))
    exp = acc.copy()
    for i in range(1, n):
        for r in range(5):
            exp[r, i] = exp[r, i - 1] + delta[r, i]
    sdfg(acc=acc, delta=delta, N=n)
    assert np.allclose(acc, exp), f'multi-slot scan diverged: max diff {np.abs(acc - exp).max():.2e}'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
