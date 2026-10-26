# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`WavefrontSkew`. Classical 2-D wavefront pattern (TSVC s2111)."""

import pathlib
import sys
import tempfile
from fractions import Fraction

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.parallelization.wavefront.skew import SKEW_P_PREFIX, SKEW_T_PREFIX, WavefrontSkew

# The corpus program itself, imported as a package: its ``@dace.tasklet`` bodies lower to the
# exact 2-D wavefront ``WavefrontSkew`` exposes -- the one real corpus beneficiary of the skew.
from tests.corpus.polybench.medley.nussinov import nussinov as corpus_nussinov

N = dace.symbol("N")
tsteps = dace.symbol("tsteps")


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


@dace.program
def wavefront_with_accumulator(a: dace.float64[12, 12], s: dace.float64[1]):
    for i in range(1, 12):
        for j in range(1, 12):
            a[i, j] = 0.5 * a[i - 1, j] + 0.25 * a[i, j - 1] + 0.001 * s[0]
            s[0] = 0.5 * s[0] + a[i, j]


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
    assert loops[0].loop_variable.startswith(SKEW_T_PREFIX), (
        f"surviving loop should be the diagonal ``t``; got {loops[0].loop_variable}"
    )

    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1, f"expected exactly 1 inner Map; got {len(map_entries)}"
    map_node = map_entries[0].map
    assert len(map_node.params) == 1 and map_node.params[0].startswith(SKEW_P_PREFIX), (
        f"inner Map should iterate over ``p``; got params={map_node.params}"
    )


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
    # The diagonal ``t`` loop stays sequential -- and stays, full stop: zero loops here would mean
    # ``LoopToMap`` lifted the sequential diagonal, which is the race this test exists to forbid.
    loops = _loops(sdfg)
    assert n_loops == 1, f"the diagonal t-loop must survive LoopToMap; got {[c.loop_variable for c in loops]}"
    assert loops[0].loop_variable.startswith(SKEW_T_PREFIX), (
        f"the surviving loop should be the diagonal ``t``; got {loops[0].loop_variable}"
    )


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
    sym1 = dace.symbol("sym1", positive=True)
    sym2 = dace.symbol("sym2", positive=True)

    @dace.program
    def prog_wavefront_skew_accepts_symbolic_offsets(aa: dace.float64[N, N]):
        for i in range(sym2, N):
            for j in range(sym1, N):
                aa[i, j] = (aa[i, j - sym1] + aa[i - sym2, j]) / 1.9

    sdfg = prog_wavefront_skew_accepts_symbolic_offsets.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1


def test_wavefront_skew_emits_runtime_guard_for_unannotated_symbol():
    """The diagonal read ``aa[i-1, j-sym]`` gives the distance ``(1, sym)``: ``tau = (1, 1)`` orders it only for
    ``sym >= 0``, and no candidate orders it for every sign. With ``sym`` not declared non-negative the pass
    assumes it and plants a ``std::abort`` guard on ``sym < 0`` before the skewed loop; a non-negative value
    passes and produces the sequential result."""
    sym = dace.symbol("sym_unannot")  # no ``positive=True``

    @dace.program
    def prog_wavefront_skew_emits_runtime_guard_for_unannotated_symbol(aa: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(sym, N):
                aa[i, j] = (aa[i, j - 1] + aa[i - 1, j - sym]) / 1.9

    sdfg = prog_wavefront_skew_emits_runtime_guard_for_unannotated_symbol.to_sdfg(simplify=True)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    # A pre-state ``_skew_guard_*`` with a single zero-connector tasklet was
    # planted before the (now skewed) loop. Exactly one such tasklet exists.
    guard_states = [s for s in sdfg.nodes() if isinstance(s, dace.SDFGState) and s.label.startswith("_skew_guard_")]
    assert len(guard_states) == 1, f"expected 1 guard state, got {len(guard_states)}"
    guards = [
        n for n in guard_states[0].nodes() if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("_skew_guard_")
    ]
    assert len(guards) == 1
    assert guards[0].language == dace.dtypes.Language.Python
    assert "abort()" in guards[0].code.as_string

    # Runtime check: a positive ``sym_unannot`` value passes the guard and
    # the result matches the un-skewed sequential oracle.
    n_concrete, s_concrete = 12, 2
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((n_concrete, n_concrete))
    expected = aa.copy()
    for i in range(1, n_concrete):
        for j in range(s_concrete, n_concrete):
            expected[i, j] = (expected[i, j - 1] + expected[i - 1, j - s_concrete]) / 1.9
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
    assert not any(l.loop_variable.startswith(SKEW_T_PREFIX) for l in loops)
    assert not any(l.loop_variable.startswith(SKEW_P_PREFIX) for l in loops)


def test_wavefront_skew_runtime_guard_traps_on_violation(tmp_path):
    """Negative ``sym_unannot`` violates the assumption the skew is legal under; the
    planted ``std::abort`` fires and the program aborts (subprocess
    isolation prevents the trap from killing the test runner)."""
    import subprocess
    import textwrap

    src = textwrap.dedent("""
        import numpy as np
        import dace
        from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew
        N = dace.symbol('N')
        sym = dace.symbol('sym_unannot')

        @dace.program
        def prog(aa: dace.float64[N, N]):
            for i in range(1, N):
                for j in range(sym, N):
                    aa[i, j] = (aa[i, j - 1] + aa[i - 1, j - sym]) / 1.9

        sdfg = prog.to_sdfg(simplify=True)
        assert WavefrontSkew().apply_pass(sdfg, {}) == 1
        print('BUILT', flush=True)
        sdfg(aa=np.zeros((8, 8)), N=8, sym_unannot=-1)  # negative -> trap
    """)
    # A FILE, not ``python -c``: the child builds a ``@dace.program``, and the frontend reads it back
    # with ``inspect.getsource``, which has no source to find for a ``-c`` string -- the child then
    # died on that TypeError before ever reaching the guarded call.
    script = tmp_path / "wavefront_guard_child.py"
    script.write_text(src)
    res = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    ctx = f"(rc={res.returncode}, stdout={res.stdout!r}, stderr={res.stderr[-400:]!r})"
    # Non-vacuity: without BUILT any child failure -- an import error, a skew that did not apply --
    # would satisfy the returncode check and the test could never fail.
    assert b"BUILT" in res.stdout, f"child died before the guarded call {ctx}"
    assert res.returncode != 0, f"runtime guard did not trap on a violating sym {ctx}"


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
    assert len(loops) == 1 and loops[0].loop_variable.startswith(SKEW_T_PREFIX)
    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1
    assert map_entries[0].map.params[0].startswith(SKEW_P_PREFIX)


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
    loops = _loops(sdfg)
    assert len(loops) == 1, f"the diagonal t-loop must survive LoopToMap; got {[c.loop_variable for c in loops]}"
    assert loops[0].loop_variable.startswith(SKEW_T_PREFIX), (
        f"the surviving loop should be the diagonal ``t``; got {loops[0].loop_variable}"
    )


@pytest.mark.parametrize("di, dj", [(0, 1), (1, 0), (1, -1), (-1, 1), (-1, -1)])
def test_a_third_neighbour_read_in_either_direction_keeps_the_values(di, dj):
    """Beside the backward reads ``aa[i, j-1]`` / ``aa[i-1, j]``, a third read forward of the write (the old value a
    later iteration overwrites) or backward of it (the new value) must be ordered by its own direction."""
    sdfg = dace.SDFG(f"third_read_{di + 1}_{dj + 1}")
    sdfg.add_array("aa", [N, N], dace.float64)
    outer = LoopRegion("outer", "i < N - 1", "i", "i = 1", "i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "j < N - 1", "j", "j = 1", "j = j + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("body", is_start_block=True)
    tk = body.add_tasklet("c", {"w", "n", "x"}, {"o"}, "o = 0.25 * w + 0.5 * n + 0.125 * x")
    body.add_edge(body.add_read("aa"), None, tk, "w", dace.Memlet("aa[i, j - 1]"))
    body.add_edge(body.add_read("aa"), None, tk, "n", dace.Memlet("aa[i - 1, j]"))
    body.add_edge(body.add_read("aa"), None, tk, "x", dace.Memlet(f"aa[i + {di}, j + {dj}]"))
    body.add_edge(tk, "o", body.add_write("aa"), None, dace.Memlet("aa[i, j]"))

    n = 12
    aa0 = np.random.default_rng(331).standard_normal((n, n))
    ref = aa0.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ref[i, j] = 0.25 * ref[i, j - 1] + 0.5 * ref[i - 1, j] + 0.125 * ref[i + di, j + dj]
    assert WavefrontSkew().apply_pass(sdfg, {}) == 1
    got = aa0.copy()
    sdfg(aa=got, N=n)
    assert np.allclose(got, ref)


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
    assert len(loops) == 1 and loops[0].loop_variable.startswith(SKEW_T_PREFIX)
    map_entries = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1 and map_entries[0].map.params[0].startswith(SKEW_P_PREFIX)


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
    from dace.transformation.passes.parallelization.wavefront import skew as ws

    fired = [0]
    original = ws.WavefrontSkew.apply_pass

    def counting(self, sdfg, res):
        out = original(self, sdfg, res)
        fired[0] += 0 if out is None else (len(out) if hasattr(out, "__len__") else int(out))
        return out

    ws.WavefrontSkew.apply_pass = counting
    try:
        sdfg = corpus_nussinov.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True, target="cpu")
    finally:
        ws.WavefrontSkew.apply_pass = original
    assert fired[0] >= 1, "WavefrontSkew did not fire on nussinov through the full pipeline"


def test_wavefront_skew_nussinov_value_preserving_through_full_pipeline():
    """The skewed, canonicalized ``nussinov`` reproduces the sequential reference exactly."""
    from dace.transformation.passes.canonicalize import canonicalize

    n = 40
    seq = np.array([(i + 1) % 4 for i in range(n)], dtype=np.int32)
    ref = _nussinov_oracle(seq, np.zeros((n, n), dtype=np.int32))

    sdfg = corpus_nussinov.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target="cpu")
    got = np.zeros((n, n), dtype=np.int32)
    sdfg(seq=seq.copy(), table=got, N=n)
    assert np.array_equal(got, ref)


def test_wavefront_skew_five_point_absorbs_split_snapshot_through_full_pipeline():
    """Regression guard for the snapshot-absorb path. Through the FULL pipeline
    (not the isolated pass) ``BreakAntiDependence`` breaks the inner anti-dependence
    ``aa[i, j+1]`` with a per-iteration snapshot ``aa_split_snap = aa`` in the
    outer body. The skew folds the snapshot back into the live array and still
    fires: no non-pinned residual loop survives and a parallel ``p``-Map is present."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize import canonicalize

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target="cpu")

    nonpinned = [l for l in _loops(sdfg) if not getattr(l, "pinned_sequential", False)]
    assert not nonpinned, f"expected no non-pinned residual loop; got {[l.loop_variable for l in nonpinned]}"
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert any(m.map.params[0].startswith(SKEW_P_PREFIX) for m in maps), (
        f"expected a parallel wavefront p-Map; got maps={[m.map.params for m in maps]}"
    )
    # The absorbed snapshot must be gone -- no ``_split_snap`` access node, copy,
    # nor descriptor survives (the terminal SimplifyPass runs ArrayElimination).
    snap_nodes = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode) and n.data.endswith("_split_snap")
    ]
    assert not snap_nodes, f"snapshot copy not eliminated: {[n.data for n in snap_nodes]}"
    assert not any(name.endswith("_split_snap") for name in sdfg.arrays), (
        f"snapshot descriptor not eliminated: {[n for n in sdfg.arrays if n.endswith('_split_snap')]}"
    )


def test_wavefront_skew_five_point_snapshot_absorb_value_preserving():
    """The snapshot-absorbed, fully-canonicalized 5-point Gauss-Seidel reproduces
    the sequential reference exactly (the forward reads must still see the old
    value -- the diagonal schedule guarantees the writer runs on a later
    diagonal). Checked under multiple sizes."""
    from dace.transformation.passes.canonicalize import canonicalize

    sdfg = gauss_seidel_5pt.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target="cpu")
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


def _snap_copy_state(memlet: str):
    """A one-edge ``a -> a_split_snap`` copy state carrying ``memlet``."""
    sdfg = dace.SDFG("snap_copy")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("a_split_snap", [N, N], dace.float64, transient=True)
    st = sdfg.add_state("copy", is_start_block=True)
    st.add_edge(st.add_access("a"), None, st.add_access("a_split_snap"), None, dace.Memlet(memlet))
    return st


def test_split_snapshot_window_accepts_a_narrowed_copy_but_demands_identity_indexing():
    """A snapshot may cover only the window its reads use, but must be an identity copy: a shifted or strided one
    makes ``snap[idx]`` and ``a[idx]`` different cells."""
    from dace.transformation.passes.parallelization.wavefront.statements import split_snapshot_window

    assert split_snapshot_window(_snap_copy_state("a[0:N, 0:N]")) == dace.subsets.Range.from_string("0:N, 0:N")
    assert split_snapshot_window(_snap_copy_state("a[i, 2:N]")) == dace.subsets.Range.from_string("i, 2:N")
    assert split_snapshot_window(_snap_copy_state("a[0:2, 0:N] -> [1:3, 0:N]")) is None
    assert split_snapshot_window(_snap_copy_state("a[0:N:2, 0:N]")) is None


def _snapshot_nest(
    snap_read: str,
    live_read: str = "i - 1, j",
    write: str = "i, j",
    window: str = "0:N, 0:N",
    external_reader: bool = False,
):
    """``a[write] = a_split_snap[snap_read] + a[live_read]`` over ``1 <= i, j < N - 1``, the snapshot copied from
    ``a[window]`` at the top of every ``i``. With ``external_reader`` a second outer-body state reads it too."""
    sdfg = dace.SDFG("snap_nest")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("a_split_snap", [N, N], dace.float64, transient=True)
    outer = LoopRegion("outer", "i < N - 1", "i", "i = 1", "i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    cp = outer.add_state("cp", is_start_block=True)
    cp.add_edge(cp.add_access("a"), None, cp.add_access("a_split_snap"), None, dace.Memlet(f"a[{window}]"))
    inner = LoopRegion("inner", "j < N - 1", "j", "j = 1", "j = j + 1")
    outer.add_node(inner)
    outer.add_edge(cp, inner, dace.InterstateEdge())
    body = inner.add_state("body", is_start_block=True)
    tk = body.add_tasklet("c", {"s", "l"}, {"out"}, "out = 0.5 * s + 0.25 * l")
    body.add_edge(body.add_access("a_split_snap"), None, tk, "s", dace.Memlet(f"a_split_snap[{snap_read}]"))
    body.add_edge(body.add_access("a"), None, tk, "l", dace.Memlet(f"a[{live_read}]"))
    body.add_edge(tk, "out", body.add_access("a"), None, dace.Memlet(f"a[{write}]"))
    if external_reader:
        ext = outer.add_state("ext")
        etk = ext.add_tasklet("e", {"inp"}, {"out"}, "out = inp")
        ext.add_edge(ext.add_access("a_split_snap"), None, etk, "inp", dace.Memlet("a_split_snap[i, 0]"))
        ext.add_edge(etk, "out", ext.add_access("a"), None, dace.Memlet("a[i, 0]"))
        outer.add_edge(inner, ext, dace.InterstateEdge())
    return sdfg


def _snapshot_reference(a, snap_read, live_read, write):
    """The sequential meaning of :func:`_snapshot_nest` (whole-array window) in numpy."""
    n = a.shape[0]
    for i in range(1, n - 1):
        snap = a.copy()
        for j in range(1, n - 1):
            env = {"i": i, "j": j, "N": n}
            idx = lambda spec: tuple(eval(part, {}, env) for part in spec.split(","))  # noqa: E731
            a[idx(write)] = 0.5 * snap[idx(snap_read)] + 0.25 * a[idx(live_read)]
    return a


def _structure(sdfg):
    return [(r.label, r.loop_variable, r.init_statement.as_string) for r in _loops(sdfg)], len(list(sdfg.states()))


@pytest.mark.parametrize(
    "snap_read, live_read, write",
    [
        ("i, j + 1", "i - 1, j", "i, j"),  # the old value one column ahead
        ("N - 2 - i, j", "N - 1 - i, j - 1", "N - 1 - i, j"),  # reflected rows: the row the NEXT iteration writes
        ("N - i, j + 1", "N - 1 - i, j - 1", "N - 1 - i, j"),  # reflected rows: the row the previous one wrote
    ],
)
def test_a_snapshot_read_no_earlier_write_of_the_same_step_reaches_is_absorbed(snap_read, live_read, write):
    """Reading the live array in place of the snapshot keeps the value when no write between the copy and the read
    reaches the cell -- decided in iteration space, so a reflected write map is classified by the iteration that
    writes, not by the array offset. The snapshot is gone and the skewed nest reproduces the sequential one."""
    sdfg = _snapshot_nest(snap_read, live_read, write)
    assert WavefrontSkew().apply_pass(sdfg, {}) == 1
    assert not any(
        isinstance(n, dace.nodes.AccessNode) and n.data == "a_split_snap" for n, _ in sdfg.all_nodes_recursive()
    )
    n = 12
    a0 = np.random.default_rng(518).standard_normal((n, n))
    got = a0.copy()
    sdfg(a=got, N=n)
    assert np.allclose(got, _snapshot_reference(a0.copy(), snap_read, live_read, write))


@pytest.mark.parametrize(
    "snap_read, window",
    [
        ("i, j - 1", "0:N, 0:N"),  # written one column earlier in the same step: the snapshot holds the old value
        ("i, j + 1", "i, 5:6"),  # outside the copied window
        ("i + 1, j + 1", "i, 2:N"),  # the wrong row of a one-row window
        ("i, j", "i, 2:N"),  # column 1 is not in the window
    ],
)
def test_a_snapshot_read_the_live_array_cannot_stand_in_for_refuses_and_leaves_the_sdfg(snap_read, window):
    sdfg = _snapshot_nest(snap_read, window=window)
    before = _structure(sdfg)
    assert WavefrontSkew().apply_pass(sdfg, {}) is None
    assert _structure(sdfg) == before
    assert any(isinstance(n, dace.nodes.AccessNode) and n.data == "a_split_snap" for n, _ in sdfg.all_nodes_recursive())


def test_a_snapshot_with_another_reader_is_not_absorbed():
    """Dropping the copy would leave the outer-body reader consuming a dead transient."""
    sdfg = _snapshot_nest("i, j + 1", external_reader=True)
    before = _structure(sdfg)
    assert WavefrontSkew().apply_pass(sdfg, {}) is None
    assert _structure(sdfg) == before


def _hand_built_forward_symbolic_nest(fwd_col):
    """A perfect 2-D nest ``aa[i,j] = aa[i-1,j] + aa[i, <fwd_col>]`` built directly
    (not via ``@dace.program``): the frontend lowers a two-read integer body
    through ``aa_index`` slice transients that ``simplify`` collapses into
    disconnected symbol refs, hiding the reads from the pass. Building the memlets from the positive ``S``
    symbol, declared positive on the SDFG, is what drives the genuine forward-anti dependence into the pass."""
    from dace import subsets

    N_ = dace.symbol("N")
    i, j = dace.symbol("i"), dace.symbol("j")
    sdfg = dace.SDFG("wf_fwd_sym")
    sdfg.add_array("aa", [N_, N_], dace.int64)
    # Declared positive on the SDFG: its facts are what the pass proves the forward sign from
    sdfg.add_symbol("S", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    outer = LoopRegion("outer", "i < N - 1", "i", "i = 1", "i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "j < N - 1", "j", "j = 1", "j = j + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("body", is_start_block=True)
    rb, rf, w = body.add_access("aa"), body.add_access("aa"), body.add_access("aa")
    tk = body.add_tasklet("c", {"a", "b"}, {"o"}, "o = a + b")

    def point(e0, e1):
        return subsets.Range([(e0, e0, 1), (e1, e1, 1)])

    body.add_edge(rb, None, tk, "a", dace.Memlet(data="aa", subset=point(i - 1, j)))
    body.add_edge(rf, None, tk, "b", dace.Memlet(data="aa", subset=point(i, fwd_col)))
    body.add_edge(tk, "o", w, None, dace.Memlet(data="aa", subset=point(i, j)))
    return sdfg


def test_wavefront_skew_symbolic_positive_forward_read_value_preserving():
    """Regression for the symbolic-forward-read soundness bug. ``aa[i, j + S]``
    (``S`` declared positive) is a forward ANTI dependence; the pre-fix
    classification treated every symbolic distance as flow, so the pass
    committed the difference-diagonal ``tau = (1, -1)`` UNGUARDED and scheduled the
    overwrite before the read -- a silent miscompile (verified: 182 wrong cells).
    With the fix the forward distance classifies ``'anti'`` and the pass picks the
    sum-diagonal ``tau = (1, 1)`` (which schedules the overwrite on a strictly
    later ``t``), reproducing the sequential reference bit-for-bit. Integer
    arithmetic keeps the check exact."""
    S = dace.symbol("S", positive=True)
    sdfg = _hand_built_forward_symbolic_nest(dace.symbol("j") + S)
    res = WavefrontSkew().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, "the hand-built forward-symbolic nest must engage the skew"
    # No runtime guard is planted for a declared-positive offset, so the schedule
    # must be correct outright (not merely trap-safe).
    guards = [s for s in sdfg.states() if s.label.startswith("_skew_guard_")]
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
    S = dace.symbol("S", positive=True)
    sdfg = _hand_built_forward_symbolic_nest(dace.symbol("j") - S)
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
    and sink fall in the SAME parallel ``p``-wavefront unless the skew orders every
    array's dependences, not only the 2-D ``aa``'s. The result must match the
    sequential reference bit-for-bit."""

    @dace.program
    def prog_wavefront_skew_non_2d_carried_dependence_value_preserving(aa: dace.int64[N, N], bb: dace.int64[N, N, 1]):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                aa[i, j] = aa[i, j - 1] + aa[i - 1, j]
                bb[i, j, 0] = bb[i - 1, j + 1, 0] + aa[i, j]

    sdfg = prog_wavefront_skew_non_2d_carried_dependence_value_preserving.to_sdfg(simplify=True)
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
    assert np.array_equal(agot, aref) and np.array_equal(bgot, bref), (
        f"bb mismatch: got\n{bgot[..., 0]}\nref\n{bref[..., 0]}"
    )


def test_wavefront_skew_refuses_nest_threading_a_one_element_accumulator():
    sdfg = wavefront_with_accumulator.to_sdfg(simplify=True)
    a0 = np.random.default_rng(0).random((12, 12))
    ref_a, ref_s = a0.copy(), np.zeros(1)
    wavefront_with_accumulator.f(ref_a, ref_s)

    result = WavefrontSkew().apply_pass(sdfg, {})

    assert result is None
    got_a, got_s = a0.copy(), np.zeros(1)
    sdfg(a=got_a, s=got_s)
    assert np.allclose(got_s, ref_s) and np.allclose(got_a, ref_a)


@pytest.mark.parametrize("assignments", [{"y": "a[i - 1, j + 2]"}, {"y": "x", "x": "a[i, j]"}])
def test_wavefront_skew_refuses_nest_carrying_a_value_through_an_interstate_edge(assignments):
    sdfg = dace.SDFG("interstate_carry")
    sdfg.add_array("a", [12, 12], dace.float64)
    outer = LoopRegion("o", "i < 10", "i", "i = 1", "i = i + 1")
    inner = LoopRegion("n", "j < 10", "j", "j = 1", "j = j + 1")
    sdfg.add_node(outer, is_start_block=True)
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state_after(inner.add_state("head", is_start_block=True), assignments=assignments)
    t = body.add_tasklet("t", {"p": None, "q": None}, {"o": None}, "o = p + q + y")
    body.add_edge(body.add_read("a"), None, t, "p", dace.Memlet("a[i - 1, j]"))
    body.add_edge(body.add_read("a"), None, t, "q", dace.Memlet("a[i, j - 1]"))
    body.add_edge(t, "o", body.add_write("a"), None, dace.Memlet("a[i, j]"))

    result = WavefrontSkew().apply_pass(sdfg, {})

    assert result is None


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


@pytest.mark.parametrize("prog", [row_stencil_forward_read, row_stencil_diagonal_read])
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


@pytest.mark.parametrize("prog, interchangeable", [(same_lane_carry, True), (lane_crossing_carry, False)])
def test_move_loop_into_map_refuses_lane_crossing_carry(prog, interchangeable):
    """``MoveLoopIntoMap`` makes the Map the OUTER parallel axis, which is only value-preserving
    when every loop-carried dependence stays inside ONE map lane. ``a[i-1, j]`` does (same
    column, the s231/s233 recurrence-sweep shape this interchange exists for); ``a[i-1, j+1]``
    does not -- lane ``j`` would read what lane ``j+1`` writes on the previous iteration.
    """
    from dace.sdfg import nodes
    from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap

    sdfg = _loop_over_map(prog)
    outer = list(_loops(sdfg))
    assert len(outer) == 1, "fixture must leave exactly the outer i-loop"
    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), (
        "fixture must leave the inner j as a Map"
    )
    xform = MoveLoopIntoMap()
    xform.loop = outer[0]
    assert xform.can_be_applied(outer[0].parent_graph, 0, sdfg) is interchangeable


# #
#  Wavefronts the corpora carry, found through the full pipeline.             #
# #
#
# A kernel can only hide a wavefront where TWO sequential axes survive canonicalize. Over the four
# corpora that is a short list, and on it are four nests whose diagonals are genuinely parallel.
# Each ``..._is_detected`` test canonicalizes the kernel, asserts the diagonal is there, and then
# RUNS the result against the sequential meaning of the same kernel: a structural count alone is
# also satisfied by a rewrite that parallelised something unsound, and the failure mode of a wrong
# skew is a silent miscompile rather than an exception.
#
# That the diagonals are legal is a property of the KERNELS, argued once here rather than re-derived
# per test. seidel_2d's (i, j) nest and row_sweep's (t, i) nest both carry distances on both axes --
# (0, 1) forbids tau = (1, 0) and (1, -1) forbids tau = (0, 1) -- leaving only the steep
# tau = (2, 1); lu / ludcmp read A[i, k] for k < j and A[k, j] for k < i, so every distance is
# non-negative in both components and the anti-diagonal tau = (1, 1) is free. An earlier revision
# carried three exact-rational simulations of those claims; they re-implemented the kernels a second
# time in the test file and no change to DaCe could redden them.

#: The CPU knob set the corpus parallelism gate uses (``tests/corpus/measure_parallelization.py``).
CPU_PARAMS = dict(
    target="cpu",
    peel_limit=4,
    interchange_carry_with_map=True,
    scatter_to_guarded_maps=True,
)


def run_sequentially(iters, body, make):
    state = make()
    for p in iters:
        body(state, p)
    return state


def residual_loops(sdfg):
    return [
        c
        for sd in sdfg.all_sdfgs_recursive()
        for c in sd.all_control_flow_regions()
        if isinstance(c, LoopRegion) and c.loop_variable
    ]


def assert_tiled_seidel_wavefront(sdfg, target: str) -> None:
    """The ``(i, j)`` nest is skewed and TILED: its rectangular tiling is illegal (``tau = (2, 1)`` with the flow
    distance ``(-1, +1)``), so the pass tiles ``(i, w = j + i)``. The CPU keeps the time loop, one tile diagonal, the
    tile row and the in-row scan clipped to the tile's ``w`` window; the GPU also skews the tile interior, leaving
    the time loop and two diagonals."""
    loops = residual_loops(sdfg)
    names = [c.loop_variable for c in loops]
    diagonals = skew_diagonals(sdfg)
    if target == "cpu":
        assert len(diagonals) == 1 and len(loops) == 4, f"expected t, a tile diagonal and two tile loops; got {names}"
        rows = [c for c in loops if not c.loop_variable.startswith(SKEW_T_PREFIX) and c.pinned_sequential]
        inner = [c for c in rows if not any(isinstance(b, LoopRegion) for b in c.all_control_flow_blocks())]
        assert any(row.loop_variable in c.init_statement.as_string for row in rows for c in inner if c is not row), (
            f"the innermost loop's tile window follows the tile row (w = j + i); got {names}"
        )
    else:
        assert len(diagonals) == 2 and len(loops) == 3, f"expected t and two diagonals; got {names}"


def skew_diagonals(sdfg):
    """The ``_skew_t_`` diagonal loops a successful :class:`WavefrontSkew` leaves behind."""
    return [c for c in residual_loops(sdfg) if c.loop_variable.startswith(SKEW_T_PREFIX)]


# #
# C1 -- polybench seidel_2d, the ``(i, j)`` nest at fixed ``t``.               #
# #


@dace.program
def seidel_2d_npbench(A: dace.float64[N, N], tsteps: dace.int32):
    """polybench ``seidel_2d`` exactly as ``tests/corpus/polybench/stencils/seidel_2d.py`` carries
    it: the npbench formulation, a slice-vectorized neighbour sum over each row followed by a
    sequential in-row Gauss-Seidel scan."""
    for t in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]
            )
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def seidel_reference(A, nsteps, n):
    """The corpus formulation in numpy -- the value oracle for the skewed SDFG."""
    for _t in range(0, nsteps - 1):
        for i in range(1, n - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]
            )
            for j in range(1, n - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0
    return A


def test_seidel_2d_ij_wavefront_skews_with_rebased_origins():
    """seidel_2d's ``(i, j)`` nest -- the slice-vectorized row update (maps) beside the in-row scan (a loop) --
    skews as the statement chain it is, with ``normalize_loop_and_map_origin`` rebasing every range to 0 first.
    Run against the sequential reference: a structural count alone is also met by an unsound rewrite."""
    n, steps = 12, 4
    rng = np.random.default_rng(2026)
    a0 = rng.standard_normal((n, n))

    on = seidel_2d_npbench.to_sdfg(simplify=True)
    canonicalize(on, validate=False, normalize_loop_and_map_origin=True, **CPU_PARAMS)
    assert_tiled_seidel_wavefront(on, "cpu")

    finalized = finalize_for_target(on, "cpu")
    finalized.name = "seidel_2d_wavefront_knobs_on"
    got = a0.copy()
    finalized(A=got, tsteps=steps, N=n)
    assert np.allclose(got, seidel_reference(a0.copy(), steps, n)), "the skew must be value-preserving"


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_seidel_2d_unsimplified_row_copy_skews(target):
    """seidel_2d lowered WITHOUT frontend simplification, as the benchmark harness hands it to canon, skews on both
    targets. ``A[i, 1:-1] += expr`` computes ``expr`` into row transients and stores it back; the maps, the row
    store and the scan (``0:N-3`` against ``1:N-2``) are separate statements, aligned by their writes of ``A``.
    The CPU result is RUN against the sequential reference."""
    n, steps = 12, 4
    rng = np.random.default_rng(2026)
    a0 = rng.standard_normal((n, n))

    sdfg = seidel_2d_npbench.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=False, target=target)
    assert_tiled_seidel_wavefront(sdfg, target)
    if target != "cpu":
        return
    finalized = finalize_for_target(sdfg, "cpu")
    finalized.name = "seidel_2d_unsimplified_wavefront"
    got = a0.copy()
    finalized(A=got, tsteps=steps, N=n)
    assert np.allclose(got, seidel_reference(a0.copy(), steps, n)), "the skew must be value-preserving"


# #
# C2 -- the ``(t, i)`` ROW-granularity wavefront behind a slice-shaped read.   #
# #


@dace.program
def row_sweep_3pt(A: dace.float64[N, N], tsteps: dace.int32):
    """In-place 3-point row sweep: row ``i`` is rebuilt from rows ``i-1`` (already updated this
    time step) and ``i+1`` (still holding the previous step's values).

    This is seidel_2d's ``(t, i)`` shape with the in-row scan removed: its only diagonal runs along the time step.
    """
    for t in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] = (A[i - 1, 1:-1] + A[i, 1:-1] + A[i + 1, 1:-1]) / 3.0


def row_sweep_reference(A, nsteps, n):
    """:func:`row_sweep_3pt` in numpy -- the sequential meaning the skewed SDFG has to reproduce."""
    for _t in range(nsteps - 1):
        for i in range(1, n - 1):
            A[i, 1:-1] = (A[i - 1, 1:-1] + A[i, 1:-1] + A[i + 1, 1:-1]) / 3.0
    return A


def test_the_time_step_loop_of_a_row_sweep_is_not_skewed():
    """``t`` indexes nothing: every step re-applies the same sweep to the same rows, a loop over kernel
    invocations, which the skew leaves sequential although ``(t, i)`` has a legal diagonal. The result is RUN
    against the sequential reference."""
    n, steps = 12, 4
    rng = np.random.default_rng(2026)
    a0 = rng.standard_normal((n, n))

    sdfg = row_sweep_3pt.to_sdfg(simplify=True)
    canonicalize(sdfg, **CPU_PARAMS)
    assert skew_diagonals(sdfg) == [], f"the time step was skewed: {[c.loop_variable for c in residual_loops(sdfg)]}"

    finalized = finalize_for_target(sdfg, "cpu")
    finalized.name = "row_sweep_3pt_time_step"
    got = a0.copy()
    finalized(A=got, tsteps=steps, N=n)
    assert np.allclose(got, row_sweep_reference(a0.copy(), steps, n))


# #
# C3 / C4 -- polybench lu and ludcmp: an outer loop with TWO sibling inner     #
# loops, two statements of one outer body.                                    #
# #


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
    """``(iters, body, make)`` for the merged ``(i, j)`` space of lu / ludcmp, in exact rationals.

    ``with_scalar_accumulator`` picks ludcmp's spelling: it accumulates into the scalar ``w`` and
    stores once where lu updates ``A[i, j]`` in place. Both are run here in their own right, since
    it is a genuinely different program and not a restatement of the other.
    """
    iters = [(i, j) for i in range(n) for j in range(n)]

    def make():
        # Strongly diagonally dominant, so every pivot A[j][j] stays non-zero and the exact
        # rational factorization never divides by zero.
        return [
            [Fraction((i * j) % 7 + 1) + (Fraction(20 * n) if i == j else Fraction(0)) for j in range(n)]
            for i in range(n)
        ]

    def body(A, p):
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

    return iters, body, make


def lu_family_matrices(n, with_scalar_accumulator):
    """``(start, reference)`` as float64 arrays: the fixture matrix and the sequential answer.

    The reference is the exact-rational run of :func:`lu_iteration_space`, so it is the kernel's
    meaning rather than a second float implementation that could round the same way a wrong
    schedule does. The fixture is strongly diagonally dominant, so the float factorization the SDFG
    computes stays well-conditioned and the two agree to round-off.
    """
    iters, body, make = lu_iteration_space(n, with_scalar_accumulator)
    start = np.array([[float(v) for v in row] for row in make()])
    reference = np.array([[float(v) for v in row] for row in run_sequentially(iters, body, make)])
    return start, reference


@pytest.mark.parametrize(
    "program, with_scalar_accumulator, label", [(lu_factorization, False, "lu"), (ludcmp_factorization, True, "ludcmp")]
)
def test_lu_family_ij_wavefront_is_detected(program, with_scalar_accumulator, label):
    """lu's outer ``i`` loop holds TWO sibling ``j`` loops (``j < i`` and ``j >= i``): two statements, each over
    its own range, so the ``A[j, j]`` read carries only where ``j < i`` and ``tau = (1, 1)`` is legal.

    ludcmp is asserted in its OWN right rather than inherited: its scalar ``w`` accumulator makes it
    a different program. Both are RUN against their sequential answer, because a diagonal that is
    found but wrongly scheduled raises nothing."""
    n = 8
    start, reference = lu_family_matrices(n, with_scalar_accumulator)

    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, **CPU_PARAMS)
    assert len(skew_diagonals(sdfg)) == 1, (
        f"{label} (i, j) wavefront not found; residual loops {[c.loop_variable for c in residual_loops(sdfg)]}"
    )

    finalized = finalize_for_target(sdfg, "cpu")
    finalized.name = f"{label}_ij_wavefront"
    got = start.copy()
    finalized(A=got, N=n)
    assert np.allclose(got, reference), f"{label}: the (i, j) skew is not value-preserving"


NL = dace.symbol("NL")
NP = dace.symbol("NP")
nst = dace.symbol("nst")


@dace.program
def scan_with_reseed(a: dace.float64[NL, NP], c: dace.float64[NL, NP], x: dace.float64[NL, NP]):
    """``scan_levels_k32``: each step sweeps the levels, then reseeds the first 32 from the last 32."""
    for step in range(nst):
        for k in range(32, NL):
            a[k, :] = c[k, :] * a[k - 32, :] + x[k, :]
        if NL >= 64:
            a[0:32, :] = a[NL - 32 : NL, :]


def test_wavefront_skew_refuses_a_step_loop_whose_body_also_holds_a_conditional_reseed():
    """The step loop's body is the level loop AND a conditional block that reseeds ``a`` between steps. Only the
    level loop is a nest; skewing it left the reseed in the new diagonal loop, once per diagonal instead of once
    per step, and every element came out wrong (``scan_levels_k32``)."""
    nl, np_, steps = 96, 8, 3
    rng = np.random.default_rng(32)
    a0 = rng.random((nl, np_))
    c = 0.2 + 0.7 * rng.random((nl, np_))
    x = rng.random((nl, np_))
    ref = a0.copy()
    for _ in range(steps):
        for k in range(32, nl):
            ref[k, :] = c[k, :] * ref[k - 32, :] + x[k, :]
        if nl >= 64:
            ref[0:32, :] = ref[nl - 32 : nl, :]

    sdfg = scan_with_reseed.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target="cpu")
    assert not any(
        isinstance(r, LoopRegion) and r.loop_variable.startswith(SKEW_T_PREFIX)
        for r in sdfg.all_control_flow_regions(recursive=True)
    ), "the step loop with a reseed beside its level loop was skewed"
    got = a0.copy()
    sdfg(a=got, c=c.copy(), x=x.copy(), NL=nl, NP=np_, nst=steps)
    assert np.allclose(got, ref)


if __name__ == "__main__":
    test_wavefront_skew_rewrites_to_skewed_iterators_modified_inner_lifted_to_map()
    test_wavefront_skew_value_preserving()
    test_wavefront_skew_then_l2m_parallelises_inner()
    test_wavefront_skew_accepts_symbolic_offsets()
    test_wavefront_skew_emits_runtime_guard_for_unannotated_symbol()
    test_wavefront_skew_refuses_when_inner_already_parallel()
    with tempfile.TemporaryDirectory() as tmp:
        test_wavefront_skew_runtime_guard_traps_on_violation(pathlib.Path(tmp))
    test_wavefront_skew_steep_gauss_seidel_lifts_inner_to_map()
    test_wavefront_skew_steep_gauss_seidel_value_preserving()
    test_wavefront_skew_steep_then_l2m_keeps_one_sequential_loop()
    for di, dj in ((0, 1), (1, 0), (1, -1), (-1, 1), (-1, -1)):
        test_a_third_neighbour_read_in_either_direction_keeps_the_values(di, dj)
    test_wavefront_skew_five_point_gauss_seidel_forward_reads_lifts_to_map()
    test_wavefront_skew_five_point_gauss_seidel_value_preserving()
    test_wavefront_skew_fires_on_nussinov_through_full_pipeline()
    test_wavefront_skew_nussinov_value_preserving_through_full_pipeline()
    test_wavefront_skew_five_point_absorbs_split_snapshot_through_full_pipeline()
    test_wavefront_skew_five_point_snapshot_absorb_value_preserving()
    test_split_snapshot_window_accepts_a_narrowed_copy_but_demands_identity_indexing()
    test_a_snapshot_with_another_reader_is_not_absorbed()
    test_wavefront_skew_symbolic_positive_forward_read_value_preserving()
    test_wavefront_skew_symbolic_backward_read_not_over_refused()
    test_wavefront_skew_non_2d_carried_dependence_value_preserving()
    test_wavefront_skew_refuses_a_step_loop_whose_body_also_holds_a_conditional_reseed()
    test_wavefront_skew_refuses_nest_threading_a_one_element_accumulator()
    for assignments in ({"y": "a[i - 1, j + 2]"}, {"y": "x", "x": "a[i, j]"}):
        test_wavefront_skew_refuses_nest_carrying_a_value_through_an_interstate_edge(assignments)
    for prog in (row_stencil_forward_read, row_stencil_diagonal_read):
        test_sequential_outer_parallel_inner_row_stencil_lifts_inner_to_map(prog)
    test_row_stencil_forward_read_value_preserving()
    test_row_stencil_diagonal_read_value_preserving()
    for prog, interchangeable in ((same_lane_carry, True), (lane_crossing_carry, False)):
        test_move_loop_into_map_refuses_lane_crossing_carry(prog, interchangeable)
    test_seidel_2d_ij_wavefront_skews_with_rebased_origins()
    for target in ("cpu", "gpu"):
        test_seidel_2d_unsimplified_row_copy_skews(target)
    test_the_time_step_loop_of_a_row_sweep_is_not_skewed()
    for program, with_scalar_accumulator, label in (
        (lu_factorization, False, "lu"),
        (ludcmp_factorization, True, "ludcmp"),
    ):
        test_lu_family_ij_wavefront_is_detected(program, with_scalar_accumulator, label)
