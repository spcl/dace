# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`WavefrontSkew`. Classical 2-D wavefront pattern (TSVC s2111)."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.wavefront_skew import (WavefrontSkew, _SKEW_T_PREFIX, _SKEW_P_PREFIX)

N = dace.symbol('N')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


@dace.program
def wavefront_2d(aa: dace.float64[N, N]):
    """s2111: classical 2-D wavefront."""
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


def test_wavefront_skew_rewrites_to_skewed_iterators_modified_inner_lifted_to_map():
    """**Contract changed.** :class:`WavefrontSkew` now lifts the inner
    ``p``-loop to a Map directly inside the pass (via the ``LoopToMap.apply``
    utility -- sound by construction of the skew). After the pass:

    * Exactly one LoopRegion remains -- the outer ``t``-loop, sequential.
    * Its body contains a Map whose iteration symbol carries the ``p``-prefix.

    Previous contract expected two LoopRegions (both ``t`` and ``p``) and
    relied on a later ``LoopToMap`` stage to lift the inner; the in-pass
    lift makes the parallel structure visible immediately, simplifies later
    stages, and prevents a global permissive ``LoopToMap`` from accidentally
    racing the outer ``t``-loop downstream.
    """
    from dace.sdfg import nodes

    sdfg = wavefront_2d.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    loops = _loops(sdfg)
    assert len(loops) == 1, f"expected 1 outer t-loop after skew + inner-map; got {len(loops)}"
    assert loops[0].loop_variable.startswith(_SKEW_T_PREFIX), \
        f"surviving loop should be the diagonal ``t``; got {loops[0].loop_variable}"

    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1, f"expected exactly 1 inner Map; got {len(map_entries)}"
    map_node = map_entries[0].map
    assert len(map_node.params) == 1 and map_node.params[0].startswith(_SKEW_P_PREFIX), \
        f"inner Map should iterate over ``p``; got params={map_node.params}"


def test_wavefront_skew_value_preserving():
    """End-to-end: the skewed nest produces the same final ``aa`` as the
    original Python reference (iteration ORDER changes -- elements on one
    diagonal are visited in a different sequence -- but each element's
    semantic source values are the same, so the numerics match)."""
    n = 8
    rng = np.random.default_rng(2111)
    aa0 = rng.standard_normal((n, n))
    ref = aa0.copy()
    for i in range(1, n):
        for j in range(1, n):
            ref[i, j] = (ref[i, j - 1] + ref[i - 1, j]) / 1.9

    sdfg = wavefront_2d.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    got = aa0.copy()
    sdfg(aa=got, N=n)
    assert np.allclose(got, ref)


def test_wavefront_skew_then_l2m_parallelises_inner():
    """After skewing, the inner loop has no loop-carried dependence, so
    ``LoopToMap`` lifts it to a parallel Map."""
    sdfg = wavefront_2d.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    from dace.sdfg import nodes
    n_maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    n_loops = len(_loops(sdfg))
    assert n_maps >= 1, f"expected at least one parallel Map after skewing + LoopToMap; got maps={n_maps}"
    # The outer ``t`` loop stays sequential.
    assert n_loops <= 1


@dace.program
def wavefront_2d_symbolic(aa: dace.float64[N, N], sym1: dace.int64, sym2: dace.int64):
    """A wavefront whose dependence vectors are *symbolic*: ``(0, -sym1)``
    (read at ``aa[i, j - sym1]``) and ``(-sym2, 0)`` (``aa[i - sym2, j]``).
    Polyhedral schedulers without an oracle for symbol signs typically give
    up here; DaCe's symbolic positivity is enough to recognise the case."""
    for i in range(sym2, N):
        for j in range(sym1, N):
            aa[i, j] = (aa[i, j - sym1] + aa[i - sym2, j]) / 1.9


def test_wavefront_skew_accepts_symbolic_offsets():
    """The matcher should now lift symbolic-offset wavefronts when the
    offset symbols are declared positive (``dace.symbol`` with ``positive=True``
    via the function argument types)."""
    sym1 = dace.symbol('sym1', positive=True)
    sym2 = dace.symbol('sym2', positive=True)

    @dace.program
    def prog(aa: dace.float64[N, N]):
        for i in range(sym2, N):
            for j in range(sym1, N):
                aa[i, j] = (aa[i, j - sym1] + aa[i - sym2, j]) / 1.9

    sdfg = prog.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1


def test_wavefront_skew_emits_runtime_guard_for_unannotated_symbol():
    """When the offset symbol is *not* declared ``positive=True`` the matcher
    still accepts via the optimistic fall-through, but a ``__builtin_trap``
    runtime guard is planted in a pre-state to catch a runtime sym <= 0
    violation. A positive runtime value passes the guard and produces the
    correct skewed result.
    """
    sym = dace.symbol('sym_unannot')  # no ``positive=True``

    @dace.program
    def prog(aa: dace.float64[N, N]):
        for i in range(sym, N):
            for j in range(sym, N):
                aa[i, j] = (aa[i, j - sym] + aa[i - sym, j]) / 1.9

    sdfg = prog.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    # A pre-state ``_skew_guard_*`` with a single zero-connector tasklet was
    # planted before the (now skewed) loop. Exactly one such tasklet exists.
    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label.startswith('_skew_guard_')]
    assert len(guard_states) == 1, f'expected 1 guard state, got {len(guard_states)}'
    guards = [
        n for n in guard_states[0].nodes() if isinstance(n, dace.nodes.Tasklet) and n.label.startswith('_skew_guard_')
    ]
    assert len(guards) == 1
    assert '__builtin_trap' in guards[0].code.as_string

    # Runtime check: a positive ``sym_unannot`` value passes the guard and
    # the result matches the un-skewed sequential oracle.
    n_concrete, s_concrete = 12, 2
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((n_concrete, n_concrete))
    expected = aa.copy()
    for i in range(s_concrete, n_concrete):
        for j in range(s_concrete, n_concrete):
            expected[i, j] = (expected[i, j - s_concrete] + expected[i - s_concrete, j]) / 1.9
    out = aa.copy()
    sdfg(aa=out, N=n_concrete, sym_unannot=s_concrete)
    assert np.allclose(out, expected)


def test_wavefront_skew_refuses_when_inner_already_parallel():
    """TSVC ``s1119``: ``aa[i, j] = aa[i-1, j] + bb[i, j]``. The dep ``(1, 0)``
    is only on the outer ``i`` axis; the inner ``j`` is already parallel.
    A skew would gain nothing -- a direct ``LoopToMap`` on the inner produces
    the same parallel structure axis-aligned. The pass must refuse so the
    later ``LoopToMap`` stage handles ``j`` directly without the skew detour.
    """

    @dace.program
    def s1119(aa: dace.float64[N, N], bb: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(N):
                aa[i, j] = aa[i - 1, j] + bb[i, j]

    sdfg = s1119.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    assert res is None, "skew must refuse when only the outer axis carries"
    sdfg.validate()
    # Confirm the inner-j is still a LoopRegion (untouched by skew); the
    # later ``LoopToMap`` stage will lift it.
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert not any(l.loop_variable.startswith(_SKEW_T_PREFIX) for l in loops)
    assert not any(l.loop_variable.startswith(_SKEW_P_PREFIX) for l in loops)


def test_wavefront_skew_runtime_guard_traps_on_violation():
    """Negative ``sym_unannot`` violates the wavefront-dep assumption; the
    planted ``__builtin_trap`` fires and the program aborts (subprocess
    isolation prevents the trap from killing the test runner)."""
    import subprocess
    import textwrap
    src = textwrap.dedent('''
        import sys
        sys.path.insert(0, '/home/primrose/Work/yakup-dev')
        import numpy as np
        import dace
        from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
        N = dace.symbol('N')
        sym = dace.symbol('sym_unannot')

        @dace.program
        def prog(aa: dace.float64[N, N]):
            for i in range(sym, N):
                for j in range(sym, N):
                    aa[i, j] = (aa[i, j - sym] + aa[i - sym, j]) / 1.9

        sdfg = prog.to_sdfg(simplify=True)
        assert WavefrontSkew().apply_pass(sdfg, {}) == 1
        sdfg(aa=np.zeros((8, 8)), N=8, sym_unannot=-1)  # negative -> trap
    ''')
    res = subprocess.run(['/home/primrose/.pyenv/versions/py13/bin/python', '-c', src],
                         capture_output=True,
                         timeout=120)
    # ``__builtin_trap`` -> SIGILL; the python interpreter exits with a non-zero
    # status (typically -SIGILL or similar). A normal exit means no guard fired.
    assert res.returncode != 0, ('runtime guard did not trap on a violating sym '
                                 f'(stdout={res.stdout!r}, stderr={res.stderr[-400:]!r})')


@dace.program
def seidel_perfect(aa: dace.float64[N, N]):
    """Gauss-Seidel 2-D stencil as an explicit perfect nest. Its stored deps
    ``{(0,-1),(-1,0),(-1,-1),(-1,1)}`` need a skew ``tau`` with ``a > b > 0``:
    both 45-degree diagonals ``(1, 1)`` / ``(1, -1)`` are illegal, so the pass
    must reach the steeper ``(2, 1)`` candidate."""
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j] + aa[i - 1, j - 1] + aa[i - 1, j + 1]) / 4.0


def test_wavefront_skew_steep_gauss_seidel_lifts_inner_to_map():
    """The steep ``tau = (2, 1)`` case: neither axis is parallel and neither
    45-degree diagonal is legal, so the pass skews on the steeper diagonal and
    lifts the inner ``p``-loop to a Map (one sequential ``t``-loop remains)."""
    from dace.sdfg import nodes

    sdfg = seidel_perfect.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    loops = _loops(sdfg)
    assert len(loops) == 1 and loops[0].loop_variable.startswith(_SKEW_T_PREFIX)
    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1
    assert map_entries[0].map.params[0].startswith(_SKEW_P_PREFIX)


def test_wavefront_skew_steep_gauss_seidel_value_preserving():
    """End-to-end: the steep-skewed Gauss-Seidel nest reproduces the sequential
    reference exactly (the int-division ``p`` bounds and the ISL ``dim_min`` /
    ``dim_max`` ``t`` range must be right for this to hold)."""
    n = 10
    rng = np.random.default_rng(4711)
    aa0 = rng.standard_normal((n, n))
    ref = aa0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ref[i, j] = (ref[i, j - 1] + ref[i - 1, j] + ref[i - 1, j - 1] + ref[i - 1, j + 1]) / 4.0

    sdfg = seidel_perfect.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    got = aa0.copy()
    sdfg(aa=got, N=n)
    assert np.allclose(got, ref)


def test_wavefront_skew_steep_then_l2m_keeps_one_sequential_loop():
    """After the steep skew a subsequent ``LoopToMap`` finds the inner already a
    Map and leaves the diagonal ``t``-loop sequential (pinned)."""
    from dace.sdfg import nodes

    sdfg = seidel_perfect.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    n_maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert n_maps >= 1
    assert len(_loops(sdfg)) <= 1


def test_dependence_kind_classifies_backward_flow_forward_anti():
    """A distance ``(du, dv) = writer - current`` is flow when lexicographically
    backward (writer earlier -> read sees the new value) and anti when forward
    (writer later -> read sees the soon-overwritten old value). Symbolic
    components stay conservatively flow."""
    from dace import symbolic
    from dace.transformation.passes.canonicalize.wavefront_skew import dependence_kind
    p = symbolic.pystr_to_symbolic
    assert dependence_kind(p('0'), p('-1')) == 'flow'   # aa[i, j-1]
    assert dependence_kind(p('-1'), p('0')) == 'flow'   # aa[i-1, j]
    assert dependence_kind(p('-1'), p('1')) == 'flow'   # aa[i-1, j+1] (du<0 dominates)
    assert dependence_kind(p('0'), p('1')) == 'anti'    # aa[i, j+1] (old)
    assert dependence_kind(p('1'), p('0')) == 'anti'    # aa[i+1, j] (old)
    assert dependence_kind(p('1'), p('-1')) == 'anti'   # aa[i+1, j-1] (du>0 dominates)
    assert dependence_kind(p('0'), p('-sym1')) == 'flow'  # symbolic -> conservative flow


@dace.program
def gauss_seidel_5pt(aa: dace.float64[N, N]):
    """Classic 5-point in-place Gauss-Seidel. It reads FORWARD neighbours
    ``aa[i, j+1]`` and ``aa[i+1, j]`` (the still-old values a later iteration
    overwrites) as well as backward ``aa[i, j-1]`` / ``aa[i-1, j]``. The forward
    reads are ANTI dependences whose legality constraint is opposite-signed to a
    flow dependence's; modelling them as flow makes the backward and forward reads
    demand contradictory skews, so the pass would refuse. With the flow/anti split
    the anti-diagonal skew ``tau = (1, 1)`` is legal."""
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j] + aa[i, j + 1] + aa[i + 1, j]) / 4.0


def test_wavefront_skew_five_point_gauss_seidel_forward_reads_lifts_to_map():
    """The 5-point in-place Gauss-Seidel skews on the anti-diagonal despite its
    forward (anti-dependence) reads -- one sequential ``t``-loop + a parallel
    inner ``p``-Map."""
    from dace.sdfg import nodes

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    loops = _loops(sdfg)
    assert len(loops) == 1 and loops[0].loop_variable.startswith(_SKEW_T_PREFIX)
    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1 and map_entries[0].map.params[0].startswith(_SKEW_P_PREFIX)


def test_wavefront_skew_five_point_gauss_seidel_value_preserving():
    """The anti-diagonal skew of the 5-point Gauss-Seidel reproduces the
    sequential reference exactly -- the forward reads must be scheduled before
    the overwrite for this to hold."""
    n = 12
    rng = np.random.default_rng(51)
    aa0 = rng.standard_normal((n, n))
    ref = aa0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ref[i, j] = (ref[i, j - 1] + ref[i - 1, j] + ref[i, j + 1] + ref[i + 1, j]) / 4.0

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    got = aa0.copy()
    sdfg(aa=got, N=n)
    assert np.allclose(got, ref)


def _load_corpus_nussinov():
    """Load the polybench ``nussinov`` program object from the corpus, or ``None`` if the
    corpus tree is not present. Its ``@dace.tasklet`` bodies lower to the exact 2-D
    wavefront ``WavefrontSkew`` exposes -- the one real corpus beneficiary of the skew."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'corpus', 'polybench', 'medley', 'nussinov.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('corpus_nussinov', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod)['nussinov']


def _nussinov_oracle(seq, table):
    n = table.shape[0]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < n:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < n:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] + (1 if seq[i] + seq[j] == 3 else 0))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])
    return table


def test_wavefront_skew_fires_on_nussinov_through_full_pipeline():
    """Regression guard: the full ``canonicalize`` pipeline (not just the isolated pass)
    must skew the corpus ``nussinov`` -- ``WavefrontSkew`` fires at least once. Pins the
    one real wavefront beneficiary against a pass-ordering regression (the class of change
    that silently serialised nussinov before)."""
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize import wavefront_skew as ws

    nussinov = _load_corpus_nussinov()
    if nussinov is None:
        pytest.skip('corpus nussinov not available')

    fired = [0]
    original = ws.WavefrontSkew.apply_pass

    def counting(self, sdfg, res):
        out = original(self, sdfg, res)
        fired[0] += 0 if out is None else (len(out) if hasattr(out, '__len__') else int(out))
        return out

    ws.WavefrontSkew.apply_pass = counting
    try:
        sdfg = nussinov.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True, target='cpu')
    finally:
        ws.WavefrontSkew.apply_pass = original
    assert fired[0] >= 1, 'WavefrontSkew did not fire on nussinov through the full pipeline'


def test_wavefront_skew_nussinov_value_preserving_through_full_pipeline():
    """The skewed, canonicalized ``nussinov`` reproduces the sequential reference exactly."""
    from dace.transformation.passes.canonicalize import canonicalize

    nussinov = _load_corpus_nussinov()
    if nussinov is None:
        pytest.skip('corpus nussinov not available')

    n = 40
    seq = np.array([(i + 1) % 4 for i in range(n)], dtype=np.int32)
    ref = _nussinov_oracle(seq, np.zeros((n, n), dtype=np.int32))

    sdfg = nussinov.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='cpu')
    got = np.zeros((n, n), dtype=np.int32)
    sdfg(seq=seq.copy(), table=got, N=n)
    assert np.array_equal(got, ref)


def test_wavefront_skew_five_point_absorbs_split_snapshot_through_full_pipeline():
    """Regression guard for the snapshot-absorb path. Through the FULL pipeline
    (not the isolated pass) ``SplitStatements`` breaks the inner anti-dependence
    ``aa[i, j+1]`` with a per-iteration snapshot ``aa_split_snap = aa`` in the
    outer body -- an imperfect nest that ``extract_two_level_nest`` used to reject,
    silently serialising the kernel. :func:`absorb_split_snapshots` now folds the
    snapshot back into the live array so the diagonal skew still fires: no
    non-pinned residual loop survives and a parallel ``p``-Map is present."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize import canonicalize

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='cpu')

    nonpinned = [l for l in _loops(sdfg) if not getattr(l, 'pinned_sequential', False)]
    assert not nonpinned, f"expected no non-pinned residual loop; got {[l.loop_variable for l in nonpinned]}"
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert any(m.map.params[0].startswith(_SKEW_P_PREFIX) for m in maps), \
        f"expected a parallel wavefront p-Map; got maps={[m.map.params for m in maps]}"
    # The absorbed snapshot must be gone -- no ``_split_snap`` access node, copy,
    # nor descriptor survives (the terminal SimplifyPass runs ArrayElimination).
    snap_nodes = [n for n, _ in sdfg.all_nodes_recursive()
                  if isinstance(n, nodes.AccessNode) and n.data.endswith('_split_snap')]
    assert not snap_nodes, f"snapshot copy not eliminated: {[n.data for n in snap_nodes]}"
    assert not any(name.endswith('_split_snap') for name in sdfg.arrays), \
        f"snapshot descriptor not eliminated: {[n for n in sdfg.arrays if n.endswith('_split_snap')]}"


def test_wavefront_skew_five_point_snapshot_absorb_value_preserving():
    """The snapshot-absorbed, fully-canonicalized 5-point Gauss-Seidel reproduces
    the sequential reference exactly (the forward reads must still see the old
    value -- the diagonal schedule guarantees the writer runs on a later
    diagonal). Checked under multiple sizes."""
    from dace.transformation.passes.canonicalize import canonicalize

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='cpu')
    csdfg = sdfg.compile()
    for n in (8, 13, 32):
        rng = np.random.default_rng(n)
        aa0 = rng.standard_normal((n, n))
        ref = aa0.copy()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                ref[i, j] = (ref[i, j - 1] + ref[i - 1, j] + ref[i, j + 1] + ref[i + 1, j]) / 4.0
        got = aa0.copy()
        csdfg(aa=got, N=n)
        assert np.allclose(got, ref), f"N={n} mismatch (max {np.max(np.abs(got - ref)):.2e})"


def _p(s):
    return dace.symbolic.pystr_to_symbolic(s)


def test_snapshot_reads_forward_classifies_in_iteration_space_not_array_offset():
    """The snapshot forward-safety gate must reason in ITERATION space (invert the
    write map), NOT by a raw array-index offset. For a reflected write map
    ``a[N-1-i, j]`` the two spaces disagree in sign, so an array-offset check would
    accept a backward (flow) read and reject a forward (anti) one -- exactly
    inverted, a silent miscompile. This pins the iteration-space classification."""
    from dace.transformation.passes.canonicalize.wavefront_skew import (WriteMap, snapshot_reads_forward)

    # Reflected row map: row = (N-1) - i  (c1 = -1), col = j.
    reflected = ('a', WriteMap('i', 'j', c0=_p('N - 1'), c1=-1, d0=_p('0'), d2=1, transposed=False), [])
    # Array cell [N-i, j] is written by iteration (i-1, j) -> BACKWARD (flow). Its raw
    # array offset vs the write [N-1-i, j] is [+1, 0] (would look "forward"); iteration
    # space says backward -> MUST refuse.
    backward = [(None, None, None, [_p('N - i'), _p('j')], 'a')]
    assert snapshot_reads_forward(backward, reflected, 'i', 'j') is False
    # Array cell [N-2-i, j] is written by iteration (i+1, j) -> FORWARD (anti). Raw
    # array offset is [-1, 0] (would look "backward"); iteration space says forward -> accept.
    forward = [(None, None, None, [_p('N - 2 - i'), _p('j')], 'a')]
    assert snapshot_reads_forward(forward, reflected, 'i', 'j') is True

    # Identity map sanity: a[i, j+1] forward (anti), a[i, j-1] backward (flow).
    identity = ('a', WriteMap('i', 'j', c0=_p('0'), c1=1, d0=_p('0'), d2=1, transposed=False), [])
    assert snapshot_reads_forward([(None, None, None, [_p('i'), _p('j + 1')], 'a')], identity, 'i', 'j') is True
    assert snapshot_reads_forward([(None, None, None, [_p('i'), _p('j - 1')], 'a')], identity, 'i', 'j') is False
    # A snapshot on a non-carrier array can never be reasoned about -> refuse.
    assert snapshot_reads_forward([(None, None, None, [_p('i'), _p('j + 1')], 'b')], identity, 'i', 'j') is False


def test_is_split_snapshot_state_requires_full_array_copy():
    """``is_split_snapshot_state`` must accept only a WHOLE-array copy; a partial
    slice would let the absorb redirect a read onto data the snapshot never held."""
    from dace.transformation.passes.canonicalize.wavefront_skew import is_split_snapshot_state

    for subset, expected in (('a[0:N, 0:N]', True), ('a[0:2, 0:N]', False)):
        sdfg = dace.SDFG(f'snap_copy_{expected}')
        sdfg.add_array('a', [N, N], dace.float64)
        sdfg.add_array('a_split_snap', [N, N], dace.float64, transient=True)
        st = sdfg.add_state('copy', is_start_block=True)
        st.add_edge(st.add_access('a'), None, st.add_access('a_split_snap'), None, dace.Memlet(subset))
        assert is_split_snapshot_state(st) is expected, f"{subset} -> expected {expected}"


def _snapshot_nest(external_reader: bool):
    """A minimal 2-level nest with a per-iteration snapshot ``a_split_snap = a`` in
    the outer body and an inner read ``a_split_snap[i, j+1]``. With
    ``external_reader`` a second outer-body state also reads the snapshot."""
    sdfg = dace.SDFG('snap_nest')
    sdfg.add_array('a', [N, N], dace.float64)
    sdfg.add_array('a_split_snap', [N, N], dace.float64, transient=True)
    outer = LoopRegion('outer', 'i < N - 1', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(outer, is_start_block=True)
    cp = outer.add_state('cp', is_start_block=True)
    cp.add_edge(cp.add_access('a'), None, cp.add_access('a_split_snap'), None, dace.Memlet('a[0:N, 0:N]'))
    inner = LoopRegion('inner', 'j < N - 1', 'j', 'j = 1', 'j = j + 1')
    outer.add_node(inner)
    outer.add_edge(cp, inner, dace.InterstateEdge())
    body = inner.add_state('body', is_start_block=True)
    r, w = body.add_access('a_split_snap'), body.add_access('a')
    tk = body.add_tasklet('c', {'inp'}, {'out'}, 'out = inp')
    body.add_edge(r, None, tk, 'inp', dace.Memlet('a_split_snap[i, j + 1]'))
    body.add_edge(tk, 'out', w, None, dace.Memlet('a[i, j]'))
    if external_reader:
        ext = outer.add_state('ext')
        etk = ext.add_tasklet('e', {'inp'}, {'out'}, 'out = inp')
        ext.add_edge(ext.add_access('a_split_snap'), None, etk, 'inp', dace.Memlet('a_split_snap[i, 0]'))
        ext.add_edge(etk, 'out', ext.add_access('a'), None, dace.Memlet('a[i, 0]'))
        outer.add_edge(inner, ext, dace.InterstateEdge())
    return sdfg, outer, inner


def test_plan_split_snapshots_refuses_external_snapshot_reader():
    """A snapshot array read OUTSIDE the inner loop must abort the absorb: dropping
    the copy would leave that external read consuming a dead transient."""
    from dace.transformation.passes.canonicalize.wavefront_skew import plan_split_snapshots

    sdfg, outer, inner = _snapshot_nest(external_reader=True)
    assert plan_split_snapshots(outer, inner, sdfg) is None


def test_plan_split_snapshots_is_non_mutating_then_commit_applies():
    """Planning must not touch the SDFG (so a later skew refusal is a no-op);
    committing then redirects the read onto the live array and empties the copy."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize.wavefront_skew import (plan_split_snapshots, commit_split_snapshots)

    sdfg, outer, inner = _snapshot_nest(external_reader=False)
    cp = next(b for b in outer.nodes() if isinstance(b, SDFGState) and b.label == 'cp')
    body = next(inner.all_states())

    plan = plan_split_snapshots(outer, inner, sdfg)
    assert plan is not None
    snap_src, snap_reads, copy_states = plan
    assert snap_src == {'a_split_snap': 'a'} and len(snap_reads) == 1 and copy_states == [cp]
    # Planning is non-mutating: copy state + snapshot read still present.
    assert len(list(cp.nodes())) == 2
    assert any(n.data == 'a_split_snap' for n in body.data_nodes())

    commit_split_snapshots(snap_reads, copy_states)
    # Copy emptied; the inner read now comes from the live array, no snapshot node.
    assert len(list(cp.nodes())) == 0
    assert not any(n.data == 'a_split_snap' for n in body.data_nodes())
    a_readers = [n for n in body.data_nodes() if n.data == 'a' and body.in_degree(n) == 0]
    assert a_readers and any(body.out_degree(n) > 0 for n in a_readers)


def test_dependence_kind_symbolic_forward_positive_is_anti():
    """Soundness of :func:`dependence_kind` on symbolic distances. A forward read
    at a *declared-positive* symbolic distance ``aa[i, j + S]`` (``du, dv`` =
    ``(0, S)``, ``S > 0``) is a genuine ANTI dependence and MUST classify as
    ``'anti'`` -- treating it as flow lets the pass pick the difference-diagonal
    ``tau = (1, -1)`` and schedule the overwrite before the read (silent
    miscompile). A backward symbolic read ``aa[i, j - S]`` (``(0, -S)``) stays
    ``'flow'``; an unannotated / unprovable-sign symbol also stays conservatively
    ``'flow'`` (the optimistic retry pins it with a runtime guard)."""
    from dace import symbolic
    from dace.transformation.passes.canonicalize.wavefront_skew import dependence_kind
    p = symbolic.pystr_to_symbolic
    S = dace.symbol('S', positive=True)
    assert dependence_kind(p('0'), S) == 'anti'          # aa[i, j+S], S>0 -> forward anti
    assert dependence_kind(S, p('0')) == 'anti'          # aa[i+S, j], S>0 -> forward anti
    assert dependence_kind(p('0'), -S) == 'flow'         # aa[i, j-S] -> backward flow
    assert dependence_kind(p('-1'), S) == 'flow'         # du=-1 backward dominates lexicographically
    assert dependence_kind(p('0'), p('-sym1')) == 'flow'  # unprovable sign -> conservative flow


def _hand_built_forward_symbolic_nest(fwd_col):
    """A perfect 2-D nest ``aa[i,j] = aa[i-1,j] + aa[i, <fwd_col>]`` built directly
    (not via ``@dace.program``): the frontend lowers a two-read integer body
    through ``aa_index`` slice transients that ``simplify`` collapses into
    disconnected symbol refs, hiding the reads from ``collect_carrier`` so the pass
    would refuse and never engage. Building the memlets from the positive ``S``
    symbol OBJECT (not a parsed string, which strips ``positive=True``) is what
    drives the genuine forward-anti dependence into the pass."""
    from dace import subsets
    N_ = dace.symbol('N')
    i, j = dace.symbol('i'), dace.symbol('j')
    sdfg = dace.SDFG('wf_fwd_sym')
    sdfg.add_array('aa', [N_, N_], dace.int64)
    sdfg.add_symbol('S', dace.int64)
    outer = LoopRegion('outer', 'i < N - 1', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion('inner', 'j < N - 1', 'j', 'j = 1', 'j = j + 1')
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state('body', is_start_block=True)
    rb, rf, w = body.add_access('aa'), body.add_access('aa'), body.add_access('aa')
    tk = body.add_tasklet('c', {'a', 'b'}, {'o'}, 'o = a + b')

    def point(e0, e1):
        return subsets.Range([(e0, e0, 1), (e1, e1, 1)])

    body.add_edge(rb, None, tk, 'a', dace.Memlet(data='aa', subset=point(i - 1, j)))
    body.add_edge(rf, None, tk, 'b', dace.Memlet(data='aa', subset=point(i, fwd_col)))
    body.add_edge(tk, 'o', w, None, dace.Memlet(data='aa', subset=point(i, j)))
    return sdfg


def test_wavefront_skew_symbolic_positive_forward_read_value_preserving():
    """Regression for the symbolic-forward-read soundness bug. ``aa[i, j + S]``
    (``S`` declared positive) is a forward ANTI dependence; the pre-fix
    ``dependence_kind`` classified every symbolic distance as flow, so the pass
    committed the difference-diagonal ``tau = (1, -1)`` UNGUARDED and scheduled the
    overwrite before the read -- a silent miscompile (verified: 182 wrong cells).
    With the fix the forward distance classifies ``'anti'`` and the pass picks the
    sum-diagonal ``tau = (1, 1)`` (which schedules the overwrite on a strictly
    later ``t``), reproducing the sequential reference bit-for-bit. Integer
    arithmetic keeps the check exact."""
    S = dace.symbol('S', positive=True)
    sdfg = _hand_built_forward_symbolic_nest(dace.symbol('j') + S)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, "the hand-built forward-symbolic nest must engage the skew"
    # No runtime guard is planted for a declared-positive offset, so the schedule
    # must be correct outright (not merely trap-safe).
    guards = [s for s in sdfg.all_states() if s.label.startswith('_skew_guard_')]
    assert not guards, f"a declared-positive forward read must not need a runtime guard; got {len(guards)}"

    n, s = 16, 1
    rng = np.random.default_rng(2111)
    aa0 = rng.integers(0, 7, size=(n, n), dtype=np.int64)
    ref = aa0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ref[i, j] = ref[i - 1, j] + ref[i, j + s]
    got = aa0.copy()
    sdfg(aa=got, N=n, S=s)
    assert np.array_equal(got, ref), f"mismatch: got\n{got}\nref\n{ref}"


def test_wavefront_skew_symbolic_backward_read_not_over_refused():
    """Guard against the fix over-refusing: a BACKWARD symbolic read ``aa[i, j - S]``
    is a genuine flow (RAW) recurrence and must still skew correctly (``tau = (1, 1)``
    reads the freshly produced value), reproducing the sequential reference."""
    S = dace.symbol('S', positive=True)
    sdfg = _hand_built_forward_symbolic_nest(dace.symbol('j') - S)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n, s = 16, 1
    rng = np.random.default_rng(2112)
    aa0 = rng.integers(0, 7, size=(n, n), dtype=np.int64)
    ref = aa0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ref[i, j] = ref[i - 1, j] + ref[i, j - s]
    got = aa0.copy()
    sdfg(aa=got, N=n, S=s)
    assert np.array_equal(got, ref), f"mismatch: got\n{got}\nref\n{ref}"


def test_wavefront_skew_non_2d_carried_dependence_value_preserving():
    """A 3-D array ``bb`` carries a dependence ``bb[i-1, j+1, 0]`` that lies exactly
    ON the chosen ``tau = (1, 1)`` wavefront (``tau . (-1, 1) == 0``), so its source
    and sink fall in the SAME parallel ``p``-wavefront. ``collect_carrier`` skips
    every non-2-D array, so the skew is decided from the 2-D ``aa`` alone and the
    forced inner->Map lift races ``bb``. The result must match the sequential
    reference bit-for-bit."""
    @dace.program
    def prog(aa: dace.int64[N, N], bb: dace.int64[N, N, 1]):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                aa[i, j] = aa[i, j - 1] + aa[i - 1, j]
                bb[i, j, 0] = bb[i - 1, j + 1, 0] + aa[i, j]

    sdfg = prog.to_sdfg(simplify=True)
    WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()

    n = 10
    rng = np.random.default_rng(4711)
    aa0 = rng.integers(0, 5, size=(n, n), dtype=np.int64)
    bb0 = rng.integers(0, 5, size=(n, n, 1), dtype=np.int64)
    aref, bref = aa0.copy(), bb0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            aref[i, j] = aref[i, j - 1] + aref[i - 1, j]
            bref[i, j, 0] = bref[i - 1, j + 1, 0] + aref[i, j]
    agot, bgot = aa0.copy(), bb0.copy()
    sdfg(aa=agot, bb=bgot, N=n)
    assert np.array_equal(agot, aref) and np.array_equal(bgot, bref), \
        f"bb mismatch: got\n{bgot[..., 0]}\nref\n{bref[..., 0]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
