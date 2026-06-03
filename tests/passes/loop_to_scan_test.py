# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.loop_to_scan.LoopToScan`."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_scan_nodes(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))


def test_inclusive_sum_1d():
    """``out[i+1] = out[i] + delta[i]`` over a 1-D array. After the pass the loop body
    writes a per-iteration delta buffer (now parallelizable), a Scan libnode follows,
    and a seed-add Map writes the final ``out`` values. Numeric result matches the
    sequential cumsum + seed oracle."""

    @dace.program
    def scan1d(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i]

    sdfg = scan1d.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    # The original loop stays (now writing per-iteration to ``_scan_in``); the post-loop
    # chain contributes a Scan libnode + a seed-add map. LoopToMap can later lift the
    # remaining loop, but that's not LoopToScan's responsibility.
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
    """``out[i+S] = out[i] + delta[i]`` for ``S >= 2`` -- a stride-``S``
    residue-class scan. The libnode runs the ``S`` independent class scans
    in parallel; the seed-add Map fans the ``S`` pre-loop seeds out by
    ``_i mod S``. Numerically matches the sequential oracle."""

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
    """TSVC s1221 ``b[i] = b[i-4] + a[i]`` for ``i in [4, N)`` -- the in-place
    spelling of a stride-4 residue-class scan: the write is at ``i`` and the
    read at ``i-4``. ``k_w == 0``, ``k_r == -4``, ``stride == 4``.
    Per-class seed at ``b[k]`` for ``k in [0, 4)`` (the pre-loop values).
    Numerically matches the sequential oracle."""

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
    """TSVC s2111 shape: ``aa[j, i] = (aa[j, i-1] + aa[j-1, i]) / 1.9``. The
    inner i-loop's scan-update tasklet has two reads of ``aa``: one matches
    the carry slice (``aa[j, i-1]`` -- same j, previous i), the other is at a
    *different* row (``aa[j-1, i]``). The second is NOT a non-out delta; it's
    an extra read of the carry array that makes the recurrence a 2-D coupled
    one, not a 1-D scan. Lifting it as a scan corrupts the result. The
    matcher must refuse."""

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
    """TSVC s242: ``a[i] = a[i-1] + 0.5 + 1.0 + b[i] + c[i] + d[i]``. The first ``+``
    lowers to a tasklet ``__out = __in1 + 0.5`` -- ONE data input (the carry) +
    a literal. v3 of the matcher accepts this shape: the carry input is severed
    as in v1/v2 and the tasklet's passthrough emits the literal directly; the
    downstream chain extends that literal with ``+ 1.0 + b[i] + c[i] + d[i]``
    to build the per-iteration delta. Scan then folds in the seed ``a[0]``."""

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
    """The cloudsc ``pfsqrf`` shape: the LoopRegion body has *three* SDFGStates --
    an empty pre-state (carrying an iedge assignment like ``kfdia_plus_1 = kfdia + 1``),
    the actual scan body, and an empty post-state (advancing the iterator symbol).
    The v1 matcher used to refuse this on the ``len(blocks) != 1`` check; with the
    relaxation, empty wrapper states are ignored and the single content state drives
    the scan match.
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
    """Two content states inside the body, joined by a trivial iedge. v5's body-local
    state fuser merges them into one state (aliasing same-data AccessNodes so
    read-after-write order is preserved), then the matcher proceeds normally.

    Before v5 this loop stayed sequential -- regression target for the body-local
    fusion path used by the cloudsc ``pfsqrf`` shape."""
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
    """Two independent scan recurrences in the same loop body:

    * ``a[i+1] = a[i] + delta_a[i]`` -- a SUM scan on ``a``
    * ``b[i+1] = b[i] * delta_b[i]`` -- a PRODUCT scan on ``b``

    Each has its own carry, its own delta, its own associative op. The v4
    matcher returns ``len(matches) == 2``; the rewrite emits two Scan libnodes
    side-by-side. Cloudsc ``pfsqrf`` is the production occurrence of this
    pattern (five parallel sums in one body)."""

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
    """Cloudsc-faithful five-carry prefix sum: matches the structure of the
    surviving ``pfsqrf`` inner loop (``pfsqif``/``pfsqrf``/``pfsqlf``/
    ``pfsqsf``/``pfcqlng`` all carried side-by-side, all with op = SUM).

    The matcher returns 5 ``_Scan`` infos; the rewrite emits 5 Scan libnodes."""

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
    """A scan whose body is split across two SDFGStates joined by a trivial
    interstate edge (no assignments, condition = 1). The v5 ``StateFusion``
    preprocess inside ``LoopToScan.apply_pass`` fuses the two states; the
    matcher then sees a single-content-state body and proceeds.

    The cloudsc ``pfsqrf`` inner loop (``for_1134``) has this shape: ``UnaryOp_1135``
    and ``assign_1143_12`` joined by an iedge with no assignments and condition 1.
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
    """The actual cloudsc ``pfsqrf`` shape after v5 fusion: the body has TWO write
    AccessNodes for the same carrier ``out`` (one from the pre-fuse state-1 write,
    one from the pre-fuse state-2 write; they get carried over by the merge as
    separate sink nodes). ``_find_unique_write_edge`` refuses on the multi-write-AN
    case, so the scan match fails even though the carry is structurally valid.

    Each pre-fuse state writes a DIFFERENT subset of ``out``: state 1 writes
    ``out[i]`` (an intermediate side-effect), state 2 writes ``out[i+1]`` (the
    scan carry). Both write nodes survive the fuse; the matcher needs to identify
    the scan-carry write node (the one whose subset is the loop-variable-indexed
    one matching the scan recurrence) and ignore the side-effect write.
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
    """Cloudsc ``for_1134`` shape: a 2-D scan ``arr[outer, inner - 1] = arr[outer,
    inner - 2] + delta[outer, inner]``. The write offset on the scan axis is
    NEGATIVE (k_w = -1), and the carry-read offset is -2. The other (outer) axis
    is loop-invariant.

    Documents that the synthetic frontend version of this shape DOES match
    correctly today; cloudsc-specific blocker (which still refuses) is elsewhere
    (likely a delta-chain peculiarity in the actual cloudsc body) and tracked
    separately.
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
    """The cloudsc ``for_1133`` shape: outer scan over ``jk`` containing an inner
    data-parallel ``jl`` column loop, prefix-sum on ``pfsqrf``. ``_match_all``
    descends one level into the inner ``LoopRegion`` to find the scan-update
    tasklet; the rewrite emits a single ``Scan`` libnode with
    ``stride = inner_size`` (residue-class) over a contiguous ``[trip, inner_size]``
    delta buffer, followed by a ``Map[(i, j)]`` seed-add. End-to-end numerics
    match the sequential oracle.
    """
    import numpy as np
    sdfg = _pfsqrf_2d_nested.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
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
    """With ``interchange_carry_with_map=True``, ``LoopToScan`` interchanges
    the loops by relocation: the outer carry ``LoopRegion[jk]`` is moved
    from the top SDFG INTO the NestedSDFG sitting inside the parallel
    ``Map[jl]``. Each Map thread then runs its own sequential ``for jk``
    body that reads/writes ``pfsqrf``/``delta`` directly out of global
    memory -- NO buffers, NO Scan libnode, NO copies.

    Post-conditions:
      * 0 LoopRegions in the top SDFG (the carry now lives one level down).
      * exactly 1 MapEntry (the interchanged parallel axis).
      * 0 Scan libnodes (we don't synthesise one anymore).
      * 0 transients introduced by the rewrite (no per-thread buffer).
      * exactly 1 LoopRegion inside the Map-body NestedSDFG, with the
        original carry variable.
      * numeric match against the sequential oracle.
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
    """The original blocker test, kept under its historical name. Calls
    ``LoopToScan()`` with the knob explicitly opted in -- previously the
    default refused this shape and the test was xfail-strict; the
    interchange path now lifts it (without emitting a Scan libnode; the
    rewrite simply relocates the carry loop into the per-thread NSDFG).
    """
    sdfg = _build_for_1133_post_l2m_sdfg()
    res = LoopToScan(interchange_carry_with_map=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift the for_1133 prefix-sum shape once the inner '
                                          f'column loop is a Map; got res={res}.')
    # The buffer-free interchange path emits 0 Scan libnodes.
    assert _num_scan_nodes(sdfg) == 0


# -----------------------------------------------------------------------------
# Refusal-mode probes for the cloudsc pfsqXf shapes. Each test below exercises
# ONE failure gate in ``LoopToScan._match_all`` that the cloudsc-actual bodies
# trip. They are marked ``xfail(strict=True)`` so that when the matcher is
# extended to accept the shape, the test flips to ``XPASS`` and forces the
# author to remove the marker, locking in the extension.
# -----------------------------------------------------------------------------


def test_outer_body_with_extra_content_state_alongside_inner_loop():
    """Outer scan ``jk`` body has both the inner column loop AND a separate
    SDFGState that prepares a transient slice. Mirrors the cloudsc frontend's
    tendency to materialize per-iteration slices in their own state. The extra
    state writes only to a transient (``tmp``) so the matcher should accept
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
    res = LoopToScan().apply_pass(sdfg, {})
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
    res = LoopToScan().apply_pass(sdfg, {})
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
    res = LoopToScan().apply_pass(sdfg, {})
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
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should walk through a 2-hop transient slice chain; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_carrier_with_computed_delta_chain():
    """Delta is a computed expression (multiply + add) of two arrays, not a
    direct read of a single delta array. The cloudsc pfsqXf shape has
    ``delta = (zqxn2d[i] - zqx0[i]) * zgdph_r``."""
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
    """Mimics the Fortran cloudsc ``DO JK ...`` body that produces ``for_1133``:
    per-level slice-copy then per-species accumulation. This is what the
    actual cloudsc SDFG looks like when emitted by FaCe -- a 5-array
    prefix-sum on (jl, jk) with a non-trivial computed delta over jm."""
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


def test_level_indexed_write_with_multiple_constant_slots():
    """The cloudsc ``for_430`` shape: multiple constant-index slots on the
    same carrier in a level loop. Each slot's update is independent and
    forms its own prefix recurrence along the level axis."""
    KLEV, KLON = (dace.symbol(s) for s in ['KLEV', 'KLON'])

    @dace.program
    def multi_slot(zvqx: dace.float64[5, KLEV, KLON], delta: dace.float64[KLEV, KLON]):
        for jk in range(1, KLEV):
            for jl in range(KLON):
                zvqx[0, jk, jl] = zvqx[0, jk - 1, jl] + delta[jk, jl] * 0.1
                zvqx[1, jk, jl] = zvqx[1, jk - 1, jl] + delta[jk, jl] * 0.2
                zvqx[2, jk, jl] = zvqx[2, jk - 1, jl] + delta[jk, jl] * 0.3

    sdfg = multi_slot.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is not None and res >= 1, (f'LoopToScan should lift multi-slot level-indexed prefix sums; got '
                                          f'res={res}, scan libnodes={_num_scan_nodes(sdfg)}.')
    assert _num_scan_nodes(sdfg) >= 1


def test_scan_with_conditional_body_descends_into_if():
    """``out[i+1] = out[i] + delta[i]`` guarded by a mask. The carry update is
    inside a ``ConditionalBlock`` in the loop body. LoopToScan needs to descend
    into the if/else branches to see the recurrence.

    Verifies BOTH structural lift (Scan libnode emitted) AND numerical
    correctness against the sequential oracle so that the rewrite handles each
    branch consistently (the matched-branch's scan-update + the OTHER
    branch's writes to the carrier).
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
    """When the scalar accumulator is read AFTER the loop (TSVC s319-style:
    ``b[N-1] = sum_val``), the rewrite emits a writeback state that copies
    the last scan-output element into ``acc[0]`` so downstream readers see
    the final running value -- matching the pre-rewrite sequential
    post-loop scalar value."""

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
    """The rewriter must preserve any interstate-edge assignments on the loop's
    in-edge and out-edge (the symbol bindings the canonicalize pipeline
    cascades onto loop boundaries). Synthesizes a contrived case by adding a
    pre-loop iedge assignment ``offset = 0`` and checks it survives.
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
    """``LoopToScan`` must refuse a recurrence whose carrier is also written in
    a SIBLING block of the loop (typically the seed initialisation
    ``flux[0] = fall[0]`` immediately before the inner ``for k`` prefix-scan).

    Pre-fix the pass matched the inner ``k`` loop as a scan and emitted a
    ``Scan`` libnode; codegen built a carry buffer whose seed slot was not
    populated from the sibling write, so the parallel output diverged from
    the sequential oracle.

    Asserts (a) no ``Scan`` libnode is produced, (b) end-to-end the SDFG
    matches the sequential numpy oracle."""
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
    """Same contract as the forward case but on a backward recurrence (Thomas
    back-substitution shape). ``NormalizeNegativeStride`` rewrites the loop
    to forward iteration; the matcher then sees a coef=-1 carry-read; the
    refusal predicate also covers this case via the sibling-seed-write
    check on the original ``x[K-1] = ...`` initialiser."""
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
    """``_fuse_body_states`` must NOT merge two adjacent body states when the
    first reads a scalar that the second writes (the TSVC s252 pattern
    ``a[i] = s + t`` in state 1 then ``t = s`` in state 2). The inter-state
    edge orders the carry's read before its write; collapsing the two
    states into one removes that ordering and codegen schedules the write
    (``t = s``) before the read (``s + t``), so the addition sees the NEW
    ``t`` instead of the previous iteration's value -- carrier RAW order
    lost.

    Delegating the safety check to ``StateFusionExtended.can_be_applied``
    (which understands cross-state RAW / WAW hazards) is the disposition
    landed in slice 2.4b-B. Pinned with the actual TSVC s252 kernel and
    verified end-to-end against the sequential reference.
    """
    from tests.corpus.tsvc import s252_d_single
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


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
