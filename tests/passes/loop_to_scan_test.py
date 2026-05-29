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
    def s242(a: dace.float64[N + 1], b: dace.float64[N + 1], c: dace.float64[N + 1],
             d: dace.float64[N + 1]):
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

    loop = LoopRegion('scan_loop', initialize_expr='i = 0', condition_expr='i < N',
                      update_expr='i = i + 1', loop_var='i')
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
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N',
                      update_expr='i = i + 1', loop_var='i')
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
    def two_scans(a: dace.float64[N + 1], b: dace.float64[N + 1],
                  da: dace.float64[N], db: dace.float64[N]):
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
    a = np.zeros(n + 1); a[0] = 0.5
    b = np.zeros(n + 1); b[0] = 1.0
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
    def five_scans(s1: dace.float64[N + 1], s2: dace.float64[N + 1],
                   s3: dace.float64[N + 1], s4: dace.float64[N + 1],
                   s5: dace.float64[N + 1],
                   d1: dace.float64[N], d2: dace.float64[N], d3: dace.float64[N],
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
    sdfg(s1=s[0], s2=s[1], s3=s[2], s4=s[3], s5=s[4],
         d1=d[0], d2=d[1], d3=d[2], d4=d[3], d5=d[4], N=n)
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
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N',
                      update_expr='i = i + 1', loop_var='i')
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
    out = np.zeros(n + 1); out[0] = 0.3
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
    loop = LoopRegion('loop', initialize_expr='i = 0', condition_expr='i < N',
                      update_expr='i = i + 1', loop_var='i')
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
    assert res == 1, (
        'Multi-write-AN per carrier in the fused body: matcher should pick the scan-'
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
    assert res == 1, (
        f'k_w = -1 scan with outer axis should match; got {res}')
    assert _num_scan_nodes(sdfg) == 1


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
