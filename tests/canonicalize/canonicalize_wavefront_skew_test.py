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
    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState)
                    and s.label.startswith('_skew_guard_')]
    assert len(guard_states) == 1, f'expected 1 guard state, got {len(guard_states)}'
    guards = [n for n in guard_states[0].nodes() if isinstance(n, dace.nodes.Tasklet)
              and n.label.startswith('_skew_guard_')]
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
                         capture_output=True, timeout=120)
    # ``__builtin_trap`` -> SIGILL; the python interpreter exits with a non-zero
    # status (typically -SIGILL or similar). A normal exit means no guard fired.
    assert res.returncode != 0, ('runtime guard did not trap on a violating sym '
                                 f'(stdout={res.stdout!r}, stderr={res.stderr[-400:]!r})')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
