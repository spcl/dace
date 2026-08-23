# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`WavefrontSkew`. Classical 2-D wavefront pattern (TSVC s2111)."""
import itertools
import random
import sys
from fractions import Fraction

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.wavefront_skew import (WavefrontSkew, _SKEW_T_PREFIX, _SKEW_P_PREFIX)

# The corpus program itself, imported as a package: its ``@dace.tasklet`` bodies lower to the
# exact 2-D wavefront ``WavefrontSkew`` exposes -- the one real corpus beneficiary of the skew.
from tests.corpus.polybench.medley.nussinov import nussinov as corpus_nussinov

N = dace.symbol('N')
tsteps = dace.symbol('tsteps')


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
    still accepts via the optimistic fall-through, but a ``std::abort``
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
    assert 'std::abort' in guards[0].code.as_string

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


def test_wavefront_skew_runtime_guard_traps_on_violation(tmp_path):
    """Negative ``sym_unannot`` violates the wavefront-dep assumption; the
    planted ``std::abort`` fires and the program aborts (subprocess
    isolation prevents the trap from killing the test runner)."""
    import subprocess
    import textwrap
    src = textwrap.dedent('''
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
        print('BUILT', flush=True)
        sdfg(aa=np.zeros((8, 8)), N=8, sym_unannot=-1)  # negative -> trap
    ''')
    # A FILE, not ``python -c``: the child builds a ``@dace.program``, and the frontend reads it back
    # with ``inspect.getsource``, which has no source to find for a ``-c`` string -- the child then
    # died on that TypeError before ever reaching the guarded call.
    script = tmp_path / 'wavefront_guard_child.py'
    script.write_text(src)
    res = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    ctx = f'(rc={res.returncode}, stdout={res.stdout!r}, stderr={res.stderr[-400:]!r})'
    # Non-vacuity: without BUILT any child failure -- an import error, a skew that did not apply --
    # would satisfy the returncode check and the test could never fail.
    assert b'BUILT' in res.stdout, f'child died before the guarded call {ctx}'
    assert res.returncode != 0, f'runtime guard did not trap on a violating sym {ctx}'


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
    assert dependence_kind(p('0'), p('-1')) == 'flow'  # aa[i, j-1]
    assert dependence_kind(p('-1'), p('0')) == 'flow'  # aa[i-1, j]
    assert dependence_kind(p('-1'), p('1')) == 'flow'  # aa[i-1, j+1] (du<0 dominates)
    assert dependence_kind(p('0'), p('1')) == 'anti'  # aa[i, j+1] (old)
    assert dependence_kind(p('1'), p('0')) == 'anti'  # aa[i+1, j] (old)
    assert dependence_kind(p('1'), p('-1')) == 'anti'  # aa[i+1, j-1] (du>0 dominates)
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

    fired = [0]
    original = ws.WavefrontSkew.apply_pass

    def counting(self, sdfg, res):
        out = original(self, sdfg, res)
        fired[0] += 0 if out is None else (len(out) if hasattr(out, '__len__') else int(out))
        return out

    ws.WavefrontSkew.apply_pass = counting
    try:
        sdfg = corpus_nussinov.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True, target='cpu')
    finally:
        ws.WavefrontSkew.apply_pass = original
    assert fired[0] >= 1, 'WavefrontSkew did not fire on nussinov through the full pipeline'


def test_wavefront_skew_nussinov_value_preserving_through_full_pipeline():
    """The skewed, canonicalized ``nussinov`` reproduces the sequential reference exactly."""
    from dace.transformation.passes.canonicalize import canonicalize

    n = 40
    seq = np.array([(i + 1) % 4 for i in range(n)], dtype=np.int32)
    ref = _nussinov_oracle(seq, np.zeros((n, n), dtype=np.int32))

    sdfg = corpus_nussinov.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='cpu')
    got = np.zeros((n, n), dtype=np.int32)
    sdfg(seq=seq.copy(), table=got, N=n)
    assert np.array_equal(got, ref)


def test_wavefront_skew_five_point_absorbs_split_snapshot_through_full_pipeline():
    """Regression guard for the snapshot-absorb path. Through the FULL pipeline
    (not the isolated pass) ``BreakAntiDependence`` breaks the inner anti-dependence
    ``aa[i, j+1]`` with a per-iteration snapshot ``aa_split_snap = aa`` in the
    outer body -- an imperfect nest that ``extract_two_level_nest`` used to reject,
    silently serialising the kernel. :func:`commit_split_snapshots` now folds the
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
    snap_nodes = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode) and n.data.endswith('_split_snap')
    ]
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


def _win(spec: str):
    """A snapshot copy window, the last field of a ``SnapRead``."""
    return dace.subsets.Range.from_string(spec)


def test_snapshot_reads_forward_classifies_in_iteration_space_not_array_offset():
    """The snapshot forward-safety gate must reason in ITERATION space (invert the
    write map), NOT by a raw array-index offset. For a reflected write map
    ``a[N-1-i, j]`` the two spaces disagree in sign, so an array-offset check would
    accept a backward (flow) read and reject a forward (anti) one -- exactly
    inverted, a silent miscompile. This pins the iteration-space classification."""
    from dace.transformation.passes.canonicalize.wavefront_skew import (WriteMap, snapshot_reads_forward)

    # Reflected row map: row = -i + (N-1), col = j.
    reflected = ('a', WriteMap('i', 'j', m=(-1, 0, 0, 1), c=(_p('N - 1'), _p('0'))), [])
    # Array cell [N-i, j] is written by iteration (i-1, j) -> BACKWARD (flow). Its raw
    # array offset vs the write [N-1-i, j] is [+1, 0] (would look "forward"); iteration
    # space says backward -> MUST refuse.
    backward = [(None, None, None, [_p('N - i'), _p('j')], 'a', _win('0:N, 0:N'))]
    assert snapshot_reads_forward(backward, reflected, 'i', 'j') is False
    # Array cell [N-2-i, j] is written by iteration (i+1, j) -> FORWARD (anti). Raw
    # array offset is [-1, 0] (would look "backward"); iteration space says forward -> accept.
    forward = [(None, None, None, [_p('N - 2 - i'), _p('j')], 'a', _win('0:N, 0:N'))]
    assert snapshot_reads_forward(forward, reflected, 'i', 'j') is True

    # Identity map sanity: a[i, j+1] forward (anti), a[i, j-1] backward (flow).
    identity = ('a', WriteMap('i', 'j', m=(1, 0, 0, 1), c=(_p('0'), _p('0'))), [])
    anti = [(None, None, None, [_p('i'), _p('j + 1')], 'a', _win('0:N, 0:N'))]
    assert snapshot_reads_forward(anti, identity, 'i', 'j') is True
    flow = [(None, None, None, [_p('i'), _p('j - 1')], 'a', _win('0:N, 0:N'))]
    assert snapshot_reads_forward(flow, identity, 'i', 'j') is False
    # A snapshot on a non-carrier array can never be reasoned about -> refuse.
    other = [(None, None, None, [_p('i'), _p('j + 1')], 'b', _win('0:N, 0:N'))]
    assert snapshot_reads_forward(other, identity, 'i', 'j') is False


def _snap_copy_state(memlet: str):
    """A one-edge ``a -> a_split_snap`` copy state carrying ``memlet``."""
    sdfg = dace.SDFG('snap_copy')
    sdfg.add_array('a', [N, N], dace.float64)
    sdfg.add_array('a_split_snap', [N, N], dace.float64, transient=True)
    st = sdfg.add_state('copy', is_start_block=True)
    st.add_edge(st.add_access('a'), None, st.add_access('a_split_snap'), None, dace.Memlet(memlet))
    return st


def test_split_snapshot_window_accepts_a_narrowed_copy_but_demands_identity_indexing():
    """**Contract changed.** ``BreakAntiDependence`` narrows the snapshot to the window its
    redirected edges read (a single row on the 5-point Gauss-Seidel), so the recognizer hands
    the WINDOW back instead of insisting on a whole-array copy -- containment of the reads is
    discharged against that window by ``snapshot_reads_in_window``. What must still be refused
    is a copy that is not an identity index map: a shifted destination or a strided copy makes
    ``snap[idx]`` and ``a[idx]`` different cells, so redirecting a read would MOVE it."""
    from dace.transformation.passes.canonicalize.wavefront_skew import split_snapshot_window

    whole = split_snapshot_window(_snap_copy_state('a[0:N, 0:N]'))
    assert whole == _win('0:N, 0:N')
    # The shape BreakAntiDependence emits: one row, the read window of the redirected edges.
    row = split_snapshot_window(_snap_copy_state('a[i, 2:N]'))
    assert row == _win('i, 2:N')
    # Shifted destination -- snap[1] holds a[0], so a redirected read would move a row.
    assert split_snapshot_window(_snap_copy_state('a[0:2, 0:N] -> [1:3, 0:N]')) is None
    # Strided copy -- the odd rows were never captured, and bounds alone would not see it.
    assert split_snapshot_window(_snap_copy_state('a[0:N:2, 0:N]')) is None


def test_snapshot_reads_outside_the_copied_window_refuse_the_absorb():
    """The protection the window relaxation moved: a read that can leave the copied window
    reads a cell the snapshot never held, so the absorb must refuse. Proved over the iteration
    domain, not by shape -- an in-window read at a symbolic index is still accepted."""
    from dace.transformation.passes.canonicalize.wavefront_skew import (domain_constraints, snapshot_reads_in_window)

    # i, j both in [1, N-2] -- the 5-point Gauss-Seidel domain.
    domain = domain_constraints('i', 'j', (_p('1'), _p('N - 2')), (_p('1'), _p('N - 2')))

    def reads(index, window):
        return [(None, None, None, index, 'a', _win(window))]

    # a_split_snap[i, j+1] against the row window a[i, 2:N]: j+1 in [2, N-1] -- contained.
    assert snapshot_reads_in_window(reads([_p('i'), _p('j + 1')], 'i, 2:N'), 'i', 'j', domain) is True
    # The same window with a BACKWARD read a[i, j-1]: j-1 reaches 0, below the window.
    assert snapshot_reads_in_window(reads([_p('i'), _p('j - 1')], 'i, 2:N'), 'i', 'j', domain) is False
    # A single-column window cannot cover the whole forward read range.
    assert snapshot_reads_in_window(reads([_p('i'), _p('j + 1')], 'i, 5:6'), 'i', 'j', domain) is False
    # Wrong row: the snapshot captured row i, the read wants row i+1.
    assert snapshot_reads_in_window(reads([_p('i + 1'), _p('j + 1')], 'i, 2:N'), 'i', 'j', domain) is False
    # A whole-array window covers everything the domain can index.
    assert snapshot_reads_in_window(reads([_p('i'), _p('j + 1')], '0:N, 0:N'), 'i', 'j', domain) is True


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
    assert dependence_kind(p('0'), S) == 'anti'  # aa[i, j+S], S>0 -> forward anti
    assert dependence_kind(S, p('0')) == 'anti'  # aa[i+S, j], S>0 -> forward anti
    assert dependence_kind(p('0'), -S) == 'flow'  # aa[i, j-S] -> backward flow
    assert dependence_kind(p('-1'), S) == 'flow'  # du=-1 backward dominates lexicographically
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


def _canon_structure(prog):
    """``(sequential LoopRegions, MapEntries)`` of ``prog`` after full canonicalization."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False)
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    return sdfg, _loops(sdfg), maps


@dace.program
def row_stencil_forward_read(a: dace.float64[N, N]):
    """tsvc_2_5 ``wf_diff_skew``: ``a[i, j] = a[i, j] + a[i-1, j] + a[i-1, j+1]``."""
    for i in range(1, N):
        for j in range(0, N - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]


@dace.program
def row_stencil_diagonal_read(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    """TSVC ``s119``: ``aa[i, j] = aa[i-1, j-1] + bb[i, j]``."""
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


@pytest.mark.parametrize('prog', [row_stencil_forward_read, row_stencil_diagonal_read])
def test_sequential_outer_parallel_inner_row_stencil_lifts_inner_to_map(prog):
    """Counter-cases of the wavefront family: an in-place stencil that writes row ``i`` and
    reads only row ``i-1`` carries on the OUTER axis alone, so the inner ``j`` is DOALL and an
    axis-aligned schedule already extracts all the parallelism -- ``WavefrontSkew`` correctly
    refuses (``tau = (1, 0)`` is legal). The pipeline must then actually deliver that Map: the
    per-dimension disjointness proof has to see that read row ``i-1`` can never be write row
    ``i``, which needs the enclosing iterator to count as FIXED while the inner loop runs.
    Both shapes previously canonicalized to two fully sequential loops and ZERO maps.
    """
    sdfg, loops, maps = _canon_structure(prog)
    assert len(maps) >= 1, f"inner j must lift to a parallel Map; got maps={len(maps)} loops={len(loops)}"
    assert len(loops) == 1, f"only the carried outer i may stay sequential; got {[l.loop_variable for l in loops]}"


def test_row_stencil_forward_read_value_preserving():
    """The lane-crossing read ``a[i-1, j+1]`` must keep its sequential outer axis: hoisting the
    ``j`` Map OUT of the ``i`` loop would let lane ``j`` read what lane ``j+1`` writes one
    iteration earlier."""
    n = 12
    rng = np.random.default_rng(1719)
    a0 = rng.standard_normal((n, n))
    ref = a0.copy()
    for i in range(1, n):
        for j in range(0, n - 1):
            ref[i, j] = ref[i, j] + ref[i - 1, j] + ref[i - 1, j + 1]

    sdfg, _loops_, maps = _canon_structure(row_stencil_forward_read)
    assert len(maps) >= 1
    got = a0.copy()
    sdfg(a=got, N=n)
    assert np.allclose(got, ref)


def test_row_stencil_diagonal_read_value_preserving():
    n = 12
    rng = np.random.default_rng(119)
    aa0 = rng.standard_normal((n, n))
    bb0 = rng.standard_normal((n, n))
    ref = aa0.copy()
    for i in range(1, n):
        for j in range(1, n):
            ref[i, j] = ref[i - 1, j - 1] + bb0[i, j]

    sdfg, _loops_, maps = _canon_structure(row_stencil_diagonal_read)
    assert len(maps) >= 1
    got = aa0.copy()
    sdfg(aa=got, bb=bb0.copy(), N=n)
    assert np.allclose(got, ref)


@dace.program
def same_lane_carry(a: dace.float64[N, N]):
    """s231/s233 recurrence sweep: the carry stays inside column ``j``."""
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            a[i, j] = a[i, j] + a[i - 1, j]


@dace.program
def lane_crossing_carry(a: dace.float64[N, N]):
    """The carry moves one column: lane ``j`` reads what lane ``j+1`` wrote."""
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            a[i, j] = a[i, j] + a[i - 1, j + 1]


def _loop_over_map(prog):
    """``prog`` with only its INNER loop lifted, so an ``i`` LoopRegion wraps one ``j`` Map."""
    from dace.transformation.interstate.loop_to_map import LoopToMap

    sdfg = prog.to_sdfg(simplify=True)
    for cfg in list(sdfg.all_control_flow_regions()):
        if not isinstance(cfg, LoopRegion) or not cfg.loop_variable:
            continue
        if any(isinstance(c, LoopRegion) and c is not cfg for c in cfg.all_control_flow_regions()):
            continue  # not the innermost loop
        xform = LoopToMap()
        xform.loop = cfg
        assert xform.can_be_applied(cfg.parent_graph, 0, sdfg), "inner j must be DOALL"
        xform.apply(cfg.parent_graph, sdfg)
        break
    return sdfg


@pytest.mark.parametrize('prog, interchangeable', [(same_lane_carry, True), (lane_crossing_carry, False)])
def test_move_loop_into_map_refuses_lane_crossing_carry(prog, interchangeable):
    """``MoveLoopIntoMap`` makes the Map the OUTER parallel axis, which is only value-preserving
    when every loop-carried dependence stays inside ONE map lane. ``a[i-1, j]`` does (same
    column, the s231/s233 recurrence-sweep shape this interchange exists for); ``a[i-1, j+1]``
    does not -- lane ``j`` would read what lane ``j+1`` writes on the previous iteration.
    """
    from dace.sdfg import nodes
    from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap

    sdfg = _loop_over_map(prog)
    outer = [l for l in _loops(sdfg)]
    assert len(outer) == 1, "fixture must leave exactly the outer i-loop"
    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), \
        "fixture must leave the inner j as a Map"
    xform = MoveLoopIntoMap()
    xform.loop = outer[0]
    assert xform.can_be_applied(outer[0].parent_graph, 0, sdfg) is interchangeable


# =========================================================================== #
#  Wavefronts the corpora carry that the pipeline does NOT find today.        #
# =========================================================================== #
#
# A kernel can only hide a wavefront where TWO sequential axes survive canonicalize. Over the four
# corpora that is a short list, and on it are four nests whose diagonals are genuinely parallel and
# that the pass nevertheless refuses. Each one gets a PAIR of tests:
#
#   * ``..._is_a_genuine_wavefront``  -- a PASSING test carrying the evidence: the dependence
#     distances taken from the index sets the kernel actually touches, which schedules those
#     distances admit, and an EXACT-RATIONAL re-run of the kernel on the legal diagonal with the
#     iterations SHUFFLED inside each diagonal. A shuffle cannot change the answer if the diagonal
#     is parallel, so equality with the sequential reference is a direct executable proof rather
#     than an appeal to the reader's dependence intuition. These must always pass: they are claims
#     about the KERNEL, not about DaCe.
#   * ``..._is_detected``            -- the tripwire, ``xfail(strict=True)``. It asserts the state
#     we want (the nest is skewed) and therefore fails today. ``strict`` is load-bearing: the day
#     someone extends the refusing guard, the xfail turns into a FAILURE and forces this file to be
#     revisited, so the gap cannot rot into a silently-passing no-op.

#: Skews probed by the evidence tests: ``wavefront_skew._SKEW_CANDIDATES`` plus the two
#: axis-aligned schedules. A nest is a GENUINE wavefront exactly when ``(1, 0)`` and ``(0, 1)`` are
#: both illegal (neither loop parallelises on its own) and some diagonal is legal.
CANDIDATE_TAUS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2))

#: The CPU knob set the corpus parallelism gate uses (``tests/corpus/measure_parallelization.py``).
CPU_PARAMS = dict(target='cpu',
                  peel_limit=4,
                  break_anti_dependence=True,
                  interchange_carry_with_map=True,
                  scatter_to_guarded_maps=True)


def dependence_distances(iters, write_set, read_set):
    """``{(d_outer, d_inner): {kinds}}`` for a nest, from the cells its iterations actually touch.

    ``iters`` must be in EXECUTION order, so for ``p`` before ``q`` a hit of ``write(p)`` against
    ``read(q)`` is a flow dependence, ``read(p)`` against ``write(q)`` an anti dependence, and
    ``write(p)`` against ``write(q)`` an output dependence. The index sets are only ever asked
    whether an intersection is EMPTY, so no answer here depends on set iteration order.
    """
    out = {}
    for p, q in itertools.combinations(iters, 2):
        for kind, hit in (('flow', write_set(p) & read_set(q)), ('anti', read_set(p) & write_set(q)),
                          ('output', write_set(p) & write_set(q))):
            if hit:
                out.setdefault((q[0] - p[0], q[1] - p[1]), set()).add(kind)
    return out


def tau_legal(tau, dists):
    """Lamport's condition: ``tau`` orders every dependence strictly on the sequential axis."""
    return all(tau[0] * a + tau[1] * b > 0 for (a, b) in dists)


def legal_taus(dists):
    return [t for t in CANDIDATE_TAUS if tau_legal(t, dists)]


def run_sequentially(iters, body, make):
    state = make()
    for p in iters:
        body(state, p)
    return state


def run_on_diagonals(iters, tau, body, make, seed):
    """``(result, diagonals)`` from executing ``body`` diagonal by diagonal under ``tau``, with the
    iterations SHUFFLED inside each diagonal -- the shuffle is what makes a match evidence of
    parallelism rather than of a coincidentally compatible order."""
    rng = random.Random(seed)
    diagonals = {}
    for p in iters:
        diagonals.setdefault(tau[0] * p[0] + tau[1] * p[1], []).append(p)
    state = make()
    for level in sorted(diagonals):
        points = list(diagonals[level])
        rng.shuffle(points)
        for p in points:
            body(state, p)
    return state, diagonals


def residual_loops(sdfg):
    return [
        c for sd in sdfg.all_sdfgs_recursive() for c in sd.all_control_flow_regions()
        if isinstance(c, LoopRegion) and c.loop_variable
    ]


def skew_diagonals(sdfg):
    """The ``_skew_t_`` diagonal loops a successful :class:`WavefrontSkew` leaves behind."""
    return [c for c in residual_loops(sdfg) if c.loop_variable.startswith(_SKEW_T_PREFIX)]


# --------------------------------------------------------------------------- #
# C1 -- polybench seidel_2d, the ``(i, j)`` nest at fixed ``t``.               #
# --------------------------------------------------------------------------- #


@dace.program
def seidel_2d_npbench(A: dace.float64[N, N], tsteps: dace.int32):
    """polybench ``seidel_2d`` exactly as ``tests/corpus/polybench/stencils/seidel_2d.py`` carries
    it: the npbench formulation, a slice-vectorized neighbour sum over each row followed by a
    sequential in-row Gauss-Seidel scan."""
    for t in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def seidel_reference(A, nsteps, n):
    """The corpus formulation in numpy -- the value oracle for the skewed SDFG."""
    for _t in range(0, nsteps - 1):
        for i in range(1, n - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, n - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0
    return A


def test_seidel_2d_ij_is_a_genuine_wavefront():
    """seidel_2d's ``(i, j)`` nest is the classical Gauss-Seidel wavefront, ``tau = (2, 1)``.

    Three things are established in EXACT RATIONALS, so no float rounding can be mistaken for a
    difference in algorithm:

    1. the corpus's npbench formulation computes the same values as the textbook element-wise
       9-point Gauss-Seidel triple loop;
    2. so does the FUSED point nest -- the slice statement distributed into the scan's ``j`` loop,
       which is the single perfectly-nested ``LoopRegion`` :class:`WavefrontSkew` requires;
    3. that fused nest run on the ``t = 2i + j`` diagonals, shuffled within each diagonal,
       reproduces the sequential answer.

    Together they say the wavefront is a property of the KERNEL, not of any particular way of
    writing it -- which is why refusing to skew the corpus spelling is a miss and not a verdict.
    """
    n, steps = 9, 3

    def make():
        return [[Fraction(i * (j + 2) + 2, n) for j in range(n)] for i in range(n)]

    def classic(A):
        for _t in range(steps - 1):
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] +
                               A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9
        return A

    def npbench(A):
        for _t in range(steps - 1):
            for i in range(1, n - 1):
                # numpy slice semantics: the whole right-hand side is evaluated before assignment.
                row = [
                    A[i][j] + (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j + 1] + A[i + 1][j - 1] +
                               A[i + 1][j] + A[i + 1][j + 1]) for j in range(1, n - 1)
                ]
                for j in range(1, n - 1):
                    A[i][j] = row[j - 1]
                for j in range(1, n - 1):
                    A[i][j] = (A[i][j] + A[i][j - 1]) / 9
        return A

    def point_body(A, p):
        """The slice statement and the scan statement for ONE point, in program order."""
        i, j = p
        A[i][j] = A[i][j] + (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j + 1] + A[i + 1][j - 1] +
                             A[i + 1][j] + A[i + 1][j + 1])
        A[i][j] = (A[i][j] + A[i][j - 1]) / 9

    reference = classic(make())
    assert npbench(make()) == reference, 'the corpus formulation is the classic 9-point kernel'

    iters = [(i, j) for i in range(1, n - 1) for j in range(1, n - 1)]

    def write_set(p):
        return {p}

    def read_set(p):
        return {(p[0] + di, p[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)}

    dists = dependence_distances(iters, write_set, read_set)
    assert sorted(dists) == [(0, 1), (1, -1), (1, 0), (1, 1)], f'unexpected distances {sorted(dists)}'
    assert legal_taus(dists) == [(2, 1)], 'only the steep Gauss-Seidel diagonal is legal'
    assert not tau_legal((1, 0), dists) and not tau_legal((0, 1), dists), \
        'neither axis parallelises on its own -- this is what makes it a wavefront and not a scan'

    # Per time step the fused nest is the classic kernel, and its (2, 1) diagonals are parallel.
    fused = make()
    for _t in range(steps - 1):
        for p in iters:
            point_body(fused, p)
    assert fused == reference, 'the fused point nest is the same kernel'

    shuffled = make()
    for _t in range(steps - 1):
        level_of = {}
        for p in iters:
            level_of.setdefault(2 * p[0] + p[1], []).append(p)
        rng = random.Random(17)
        for level in sorted(level_of):
            points = list(level_of[level])
            rng.shuffle(points)
            for p in points:
                point_body(shuffled, p)
    assert shuffled == reference, 'shuffling within each t = 2i + j diagonal must not change the answer'


def test_seidel_2d_ij_wavefront_skews_under_reconstruct_plus_origin_knobs():
    """seidel_2d's ``(i, j)`` wavefront IS reachable today -- but only with two non-default knobs.

    ``reconstruct_wavefront_nest`` rebuilds the imperfect body (the slice-vectorized Map sitting
    beside the scan ``LoopRegion``) into the single ``LoopRegion`` the skew requires, and
    ``normalize_loop_and_map_origin`` rebases both to a 0-based begin so their ranges line up.
    With both on, ``WavefrontSkew`` fires and the diagonal is ``_skew_t_ in [0 .. 3*N - 9]`` --
    exactly ``t = 2i + j`` over the rebased ``[0, N-3]`` box, the ``tau = (2, 1)`` proved legal in
    :func:`test_seidel_2d_ij_is_a_genuine_wavefront`.

    **This falsifies a comment in the pipeline.** ``pipeline.py`` lines 366-374 justify the
    ``reconstruct_wavefront_nest=False`` default with: "on the corpus kernel it targets, the Map's
    slice-normalized range and the scan's direct-index range do not line up (a real offset, not
    just a DOALL refusal), so the reconstruction never actually fires there yet -- ON is safe
    (mutate-on-provable-win only) but unproven to help; flip once a corpus win is measured." The
    ranges DO line up once ``normalize_loop_and_map_origin`` rebases them; the two knobs were only
    ever evaluated in ISOLATION, and jointly they fire. This test is that measurement.

    The assertion is value-preserving on purpose: a structural count alone would also be satisfied
    by a rewrite that parallelised something unsound.
    """
    n, steps = 12, 4
    rng = np.random.default_rng(2026)
    a0 = rng.standard_normal((n, n))

    on = seidel_2d_npbench.to_sdfg(simplify=True)
    canonicalize(on, validate=False, reconstruct_wavefront_nest=True, normalize_loop_and_map_origin=True, **CPU_PARAMS)
    diagonals = skew_diagonals(on)
    assert len(diagonals) == 1, f'expected the (i, j) nest to be skewed; residual loops were ' \
                                f'{[c.loop_variable for c in residual_loops(on)]}'
    assert len(residual_loops(on)) == 2, 'only the t time-step loop and the diagonal may remain'
    end = symbolic.simplify(loop_analysis.get_loop_end(diagonals[0]) - symbolic.pystr_to_symbolic('3*N - 9'))
    assert end == 0, f'diagonal should run to 3*N - 9 (= 2i + j over [0, N-3]); got {end} more'

    finalized = finalize_for_target(on, 'cpu')
    finalized.name = 'seidel_2d_wavefront_knobs_on'
    got = a0.copy()
    finalized(A=got, tsteps=steps, N=n)
    assert np.allclose(got, seidel_reference(a0.copy(), steps, n)), 'the skew must be value-preserving'

    # Non-vacuity: with the knobs at their defaults the very same kernel is NOT skewed, so the
    # assertions above are testing the knobs and not something the pipeline does anyway.
    off = seidel_2d_npbench.to_sdfg(simplify=True)
    canonicalize(off, validate=False, **CPU_PARAMS)
    assert skew_diagonals(off) == [], 'default knobs must leave the (i, j) nest unskewed'
    assert len(residual_loops(off)) == 3, 'default knobs leave t, i and the in-row scan sequential'


# --------------------------------------------------------------------------- #
# C2 -- the ``(t, i)`` ROW-granularity wavefront behind a slice-shaped read.   #
# --------------------------------------------------------------------------- #


@dace.program
def row_sweep_3pt(A: dace.float64[N, N], tsteps: dace.int32):
    """In-place 3-point row sweep: row ``i`` is rebuilt from rows ``i-1`` (already updated this
    time step) and ``i+1`` (still holding the previous step's values).

    This is seidel_2d's ``(t, i)`` shape with the in-row scan removed, which is the point: there is
    no ``(i, j)`` point nest here for ``ReconstructWavefrontNest`` to rebuild, so the ONLY wavefront
    in the kernel is the row-granularity one and the only thing refusing it is
    ``collect_carrier``'s non-point-read guard.
    """
    for t in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] = (A[i - 1, 1:-1] + A[i, 1:-1] + A[i + 1, 1:-1]) / 3.0


def row_sweep_iteration_space(n, steps):
    """``(iters, write_set, read_set, body, make)`` for :func:`row_sweep_3pt` at ROW granularity --
    one iteration is one ``(t, i)`` pair, writing a whole row and reading three."""
    cols = list(range(1, n - 1))
    iters = [(t, i) for t in range(steps - 1) for i in range(1, n - 1)]

    def make():
        return [[Fraction(i * (j + 2) + 2, n) for j in range(n)] for i in range(n)]

    def body(A, p):
        i = p[1]
        # The whole right-hand side is evaluated before the row is written back (slice semantics).
        updated = [(A[i - 1][j] + A[i][j] + A[i + 1][j]) / 3 for j in cols]
        for at, j in enumerate(cols):
            A[i][j] = updated[at]

    def write_set(p):
        return {(p[1], j) for j in cols}

    def read_set(p):
        return {(p[1] + d, j) for d in (-1, 0, 1) for j in cols}

    return iters, write_set, read_set, body, make


def test_row_sweep_ti_is_a_genuine_wavefront():
    """The ``(t, i)`` nest of :func:`row_sweep_3pt` is a wavefront with ``tau = (2, 1)``.

    Row ``i`` reads row ``i-1`` (written earlier in the same time step) and row ``i+1`` (written in
    the previous one), which puts distances on BOTH axes: ``(0, 1)`` forbids ``tau = (1, 0)`` and
    ``(1, -1)`` forbids ``tau = (0, 1)``, so neither the time loop nor the row loop parallelises.
    Of the whole candidate family only the steep ``t = 2t + i`` diagonal survives, and running the
    rows of each diagonal in shuffled order in exact rationals reproduces the sequential answer.
    """
    n, steps = 9, 4
    iters, write_set, read_set, body, make = row_sweep_iteration_space(n, steps)

    dists = dependence_distances(iters, write_set, read_set)
    assert sorted(dists) == [(0, 1), (1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1)], \
        f'unexpected row-granularity distances {sorted(dists)}'
    assert legal_taus(dists) == [(2, 1)], f'only tau=(2, 1) should be legal; got {legal_taus(dists)}'
    assert not tau_legal((1, 0), dists) and not tau_legal((0, 1), dists), 'neither axis is parallel on its own'

    reference = run_sequentially(iters, body, make)
    got, diagonals = run_on_diagonals(iters, (2, 1), body, make, seed=11)
    assert len(diagonals) == 11 and max(len(v) for v in diagonals.values()) == 3
    assert got == reference, 'shuffling the rows within a 2t + i diagonal must not change the answer'


def test_row_sweep_ti_wavefront_is_detected():
    """The row-granularity wavefront, found through a RANGE read rather than a point one.

    Two things had to give. The column axis is the same ``1:N-1`` span in the write and in every
    read and never mentions ``t`` or ``i``, so it carries no dependence and is dropped
    (:func:`~dace.transformation.passes.canonicalize.wavefront_skew.uniform_axes`). What is left
    is a ONE-axis carrier written at ``A[i]`` independently of ``t``, so the array cell no longer
    names the iteration that wrote it -- the distances come from program order instead, and the
    repeated per-step write becomes an output dependence that is what keeps ``t`` sequential.
    Memlet consolidation had also folded the three neighbour reads into the single
    ``A[i-1 : i+2, ...]``; expanding that constant-width range back into its three points is what
    recovers ``(0, -1)``, ``(-1, 0)`` and ``(-1, 1)``, hence ``tau = (2, 1)`` -- the schedule
    :func:`test_row_sweep_ti_is_a_genuine_wavefront` proves correct by execution."""
    sdfg = row_sweep_3pt.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=False, **CPU_PARAMS)
    assert len(skew_diagonals(sdfg)) == 1, \
        f'(t, i) wavefront not found; residual loops {[c.loop_variable for c in residual_loops(sdfg)]}'


# --------------------------------------------------------------------------- #
# C3 / C4 -- polybench lu and ludcmp: an outer loop with TWO sibling inner     #
# loops, which ``extract_two_level_nest`` refuses outright.                    #
# --------------------------------------------------------------------------- #


@dace.program
def lu_factorization(A: dace.float64[N, N]):
    """polybench ``lu``: the ``j < i`` column update followed by the ``j >= i`` row update. Written
    with explicit ``k`` loops rather than the corpus's ``@dace.map`` tasklets; the read/write sets
    on ``A``, and hence every dependence, are identical."""
    for i in range(0, N):
        for j in range(0, i):
            for k in range(0, j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            for k in range(0, i):
                A[i, j] -= A[i, k] * A[k, j]


@dace.program
def ludcmp_factorization(A: dace.float64[N, N]):
    """polybench ``ludcmp``'s factorization phase -- ``lu`` accumulating into the scalar ``w``
    before the single store, which is the shape that leaves the residual nest."""
    for i in range(0, N):
        for j in range(0, i):
            w = A[i, j]
            for k in range(0, j):
                w -= A[i, k] * A[k, j]
            A[i, j] = w / A[j, j]
        for j in range(i, N):
            w = A[i, j]
            for k in range(0, i):
                w -= A[i, k] * A[k, j]
            A[i, j] = w


def lu_iteration_space(n, with_scalar_accumulator):
    """``(iters, write_set, read_set, body, make)`` for the merged ``(i, j)`` space of lu /
    ludcmp. ``with_scalar_accumulator`` models ludcmp's ``w`` as the per-iteration temporary it is
    (i.e. after privatisation); it cannot change the dependences on ``A``, and the test asserts
    that both variants give the same distances."""
    iters = [(i, j) for i in range(n) for j in range(n)]

    def make():
        # Strongly diagonally dominant, so every pivot A[j][j] stays non-zero and the exact
        # rational factorization never divides by zero.
        return [[Fraction((i * j) % 7 + 1) + (Fraction(20 * n) if i == j else Fraction(0)) for j in range(n)]
                for i in range(n)]

    def body(A, p):
        """The two spellings are genuinely different programs -- ``lu`` accumulates IN PLACE into
        ``A[i, j]``, ``ludcmp`` into the scalar ``w`` and stores once -- so asserting that both
        yield the same distances is a real check, not a restatement."""
        i, j = p
        limit = j if j < i else i
        if with_scalar_accumulator:
            acc = A[i][j]
            for k in range(0, limit):
                acc -= A[i][k] * A[k][j]
            A[i][j] = (acc / A[j][j]) if j < i else acc
        else:
            for k in range(0, limit):
                A[i][j] -= A[i][k] * A[k][j]
            if j < i:
                A[i][j] = A[i][j] / A[j][j]

    def write_set(p):
        return {p}

    def read_set(p):
        i, j = p
        limit = j if j < i else i
        cells = {(i, j)}
        for k in range(0, limit):
            cells.add((i, k))
            cells.add((k, j))
        if j < i:
            cells.add((j, j))
        return cells

    return iters, write_set, read_set, body, make


@pytest.mark.parametrize('with_scalar_accumulator, label', [(False, 'lu'), (True, 'ludcmp')])
def test_lu_family_ij_is_a_genuine_wavefront(with_scalar_accumulator, label):
    """The merged ``(i, j)`` space of lu / ludcmp is a wavefront with ``tau = (1, 1)``.

    ``A[i, j]`` reads ``A[i, k]`` for ``k < j`` (same row, earlier column -> distances ``(0, d)``)
    and ``A[k, j]`` for ``k < i`` (earlier row, same column -> distances ``(d, 0)``). Both axes
    therefore carry, so neither loop is parallel, but every distance is non-negative in both
    components -- which is exactly the condition that makes the anti-diagonal free.

    ludcmp is proved here in its OWN right rather than inherited from lu: its scalar ``w`` is a
    per-iteration temporary, so once privatised the dependences on ``A`` are the same, and the test
    asserts that equality instead of assuming it.
    """
    n = 8
    iters, write_set, read_set, body, make = lu_iteration_space(n, with_scalar_accumulator)
    dists = dependence_distances(iters, write_set, read_set)

    assert sorted({d[0] for d in dists}) == list(range(n)), f'{label}: expected i-distances 0..{n - 1}'
    assert sorted({d[1] for d in dists if d[0] == 0}) == list(range(1, n)), \
        f'{label}: the j axis must carry too, or this would be a map-of-scans and not a wavefront'
    assert not tau_legal((1, 0), dists) and not tau_legal((0, 1), dists), f'{label}: neither axis is parallel'
    assert legal_taus(dists) == [(1, 1), (2, 1), (1, 2)], f'{label}: got {legal_taus(dists)}'

    reference = run_sequentially(iters, body, make)
    got, diagonals = run_on_diagonals(iters, (1, 1), body, make, seed=5)
    assert len(diagonals) == 2 * n - 1 and max(len(v) for v in diagonals.values()) == n
    assert got == reference, f'{label}: shuffling within an i + j diagonal must not change the answer'


def test_lu_ij_wavefront_is_detected():
    """lu's outer ``i`` loop holds TWO sibling ``j`` loops (``j < i`` and ``j >= i``), which
    ``extract_two_level_nest`` refuses outright. ``plan_guarded_fusion`` recognises them as one
    iteration space split in the source: adjacent, complementary ranges under a common iterator.
    Analysed jointly -- each sibling's range becoming a guard on its own reads, so the ``A[j, j]``
    read carries only where ``j < i`` -- ``tau = (1, 1)`` is legal, which
    ``test_lu_family_ij_is_a_genuine_wavefront`` proves by shuffling within each diagonal."""
    sdfg = lu_factorization.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=False, **CPU_PARAMS)
    assert len(skew_diagonals(sdfg)) == 1, \
        f'lu (i, j) wavefront not found; residual loops {[c.loop_variable for c in residual_loops(sdfg)]}'


def test_ludcmp_ij_wavefront_is_detected():
    """ludcmp's factorization carries the same two sibling ``j`` loops as lu, and is found the
    same way. Asserted in its OWN right rather than inherited: its scalar ``w`` accumulator makes
    it a different program, and the ludcmp parametrisation of
    ``test_lu_family_ij_is_a_genuine_wavefront`` proves ``tau = (1, 1)`` for it independently."""
    sdfg = ludcmp_factorization.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=False, **CPU_PARAMS)
    assert len(skew_diagonals(sdfg)) == 1, \
        f'ludcmp (i, j) wavefront not found; residual loops {[c.loop_variable for c in residual_loops(sdfg)]}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
