# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`WavefrontSkew`. Classical 2-D wavefront pattern (TSVC s2111)."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
