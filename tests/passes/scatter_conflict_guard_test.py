# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the scatter-conflict guard utility / pass.

Covers the three TSVC scatter patterns (``s4113``, ``s491``, ``vas``), plus an
abort-detection test that runs the SDFG with a duplicate index and verifies the
program traps. Permutation-index runs are expected to terminate cleanly with the
correct numerical result (the scatter Map executes after the guard).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from dace.transformation.passes.scatter_conflict_guard import (GuardScatterConflicts, insert_scatter_guard)

N = dace.symbol('N')

# -- TSVC scatter kernels (1-D, integer index ``ip``) ------------------------


@dace.program
def tsvc_s4113(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], ip: dace.int32[N]):
    """``a[ip[i]] = b[ip[i]] + c[i]`` -- the TSVC s4113 scatter shape."""
    for i in range(N):
        a[ip[i]] = b[ip[i]] + c[i]


@dace.program
def tsvc_s491(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], ip: dace.int32[N]):
    """``a[ip[i]] = b[i] + c[i] * d[i]`` -- the TSVC s491 scatter shape."""
    for i in range(N):
        a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def tsvc_vas(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N]):
    """``a[ip[i]] = b[i]`` -- the simplest 1-D scatter (TSVC vas)."""
    for i in range(N):
        a[ip[i]] = b[i]


# -- Helpers ------------------------------------------------------------------


def _has_integer_sort(sdfg: dace.SDFG) -> bool:
    return any(isinstance(n, IntegerSort) for n, _ in sdfg.all_nodes_recursive())


def _has_sorted_transient(sdfg: dace.SDFG, idx_name: str) -> bool:
    return f'_scatter_guard_sorted_{idx_name}' in sdfg.arrays


def _make_permutation(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int32)


# -- Per-TSVC tests -----------------------------------------------------------


def test_s4113_permutation_runs_cleanly():
    """s4113 with a permutation idx: guard runs (sort + compare), no abort, correct result."""
    sdfg = tsvc_s4113.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()
    assert _has_integer_sort(sdfg)
    assert _has_sorted_transient(sdfg, 'ip')

    n = 64
    ip = _make_permutation(n, seed=0)
    rng = np.random.default_rng(1)
    b = rng.random(n)
    c = rng.random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[ip[i]] + c[i]
    sdfg(a=a, b=b, c=c, ip=ip, N=n)
    assert np.allclose(a, a_ref)


def test_s491_permutation_runs_cleanly():
    """s491 with a permutation idx: guard runs, no abort, correct result."""
    sdfg = tsvc_s491.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()
    assert _has_integer_sort(sdfg)

    n = 48
    ip = _make_permutation(n, seed=2)
    rng = np.random.default_rng(3)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[i] + c[i] * d[i]
    sdfg(a=a, b=b, c=c, d=d, ip=ip, N=n)
    assert np.allclose(a, a_ref)


def test_vas_permutation_runs_cleanly():
    """vas (simplest 1-D scatter) with permutation idx: guard runs, no abort."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()
    assert _has_integer_sort(sdfg)

    n = 32
    ip = _make_permutation(n, seed=4)
    b = np.random.default_rng(5).random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[i]
    sdfg(a=a, b=b, ip=ip, N=n)
    assert np.allclose(a, a_ref)


# -- Structural checks --------------------------------------------------------


def test_guard_states_inserted_before_scatter():
    """The guard's four states (init flag, sort, compare-with-race-write, trap)
    are reachable from the SDFG start and precede every state that reads ``ip``."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    states_before = set(sdfg.states())
    insert_scatter_guard(sdfg, 'ip')
    new_states = set(sdfg.states()) - states_before

    assert len(new_states) == 4, (f"Expected exactly 4 new states (init+sort+compare+trap); got {len(new_states)}.")
    # Each new state's label carries the guard tag.
    new_labels = sorted(s.label for s in new_states)
    assert any('_scatter_guard_init_' in l for l in new_labels), new_labels
    assert any('_scatter_guard_sort_' in l for l in new_labels), new_labels
    assert any('_scatter_guard_compare_' in l for l in new_labels), new_labels
    assert any('_scatter_guard_trap_' in l for l in new_labels), new_labels

    # All four guard states sit at the head of the CFG.
    reachable_before_original = set()
    cur = sdfg.start_block
    while cur in new_states:
        reachable_before_original.add(cur)
        out = list(sdfg.out_edges(cur))
        if not out:
            break
        cur = out[0].dst
    assert reachable_before_original == new_states, (
        f"All four guard states should sit at the head of the CFG; reached {reachable_before_original}")


def test_guard_pass_emits_for_each_named_idx():
    """``GuardScatterConflicts(['ip'])`` emits one guard per named array."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    res = GuardScatterConflicts(['ip']).apply_pass(sdfg, {})
    assert res == 1
    assert _has_integer_sort(sdfg)


def test_guard_refuses_non_integer_idx():
    """Refuse a float ``idx`` (the libnode would refuse downstream too, but we want
    a clean error from the pass itself)."""
    sdfg = dace.SDFG('refuses_float_idx')
    sdfg.add_array('ip', [8], dace.float64)
    sdfg.add_state('s0')
    with pytest.raises(ValueError, match='integer dtype'):
        insert_scatter_guard(sdfg, 'ip')


def test_guard_refuses_unknown_idx_name():
    sdfg = dace.SDFG('refuses_unknown')
    sdfg.add_state('s0')
    with pytest.raises(ValueError, match='not a data descriptor'):
        insert_scatter_guard(sdfg, 'nonexistent')


def test_guard_refuses_double_emit():
    """Calling the helper twice for the same idx raises -- guards are not re-emitted."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    with pytest.raises(ValueError, match='already exists'):
        insert_scatter_guard(sdfg, 'ip')


# -- Abort-on-duplicate (subprocess; SIGABRT/SIGILL is expected) --------------

_DUPLICATE_ABORT_SCRIPT = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})

    import numpy as np
    import dace
    from dace.transformation.passes.scatter_conflict_guard import insert_scatter_guard

    N = dace.symbol('N')

    @dace.program
    def vas(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N]):
        for i in range(N):
            a[ip[i]] = b[i]

    sdfg = vas.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')

    n = 8
    ip = np.array([0, 1, 2, 3, 3, 5, 6, 7], dtype=np.int32)  # duplicate at index 3
    b = np.arange(n, dtype=np.float64)
    a = np.zeros(n)
    sdfg(a=a, b=b, ip=ip, N=n)

    print('UNEXPECTEDLY_SURVIVED', flush=True)
    sys.exit(0)
""")


def test_duplicate_idx_aborts_the_process():
    """Running the guarded SDFG with a duplicate ``ip`` traps before returning.

    Spawns a fresh Python subprocess so the SIGABRT/SIGILL/SIGTRAP from
    ``__builtin_trap()`` doesn't kill the test runner. The subprocess prints
    a marker only if the abort *didn't* fire; we check the marker is absent
    AND the subprocess exited abnormally (non-zero return / signal).
    """
    proc = subprocess.run([sys.executable, '-c', _DUPLICATE_ABORT_SCRIPT], capture_output=True, text=True, timeout=120)
    assert 'UNEXPECTEDLY_SURVIVED' not in proc.stdout, (
        f"Guard failed to abort on duplicate idx. stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}")
    assert proc.returncode != 0, (f"Expected non-zero exit on trap; got returncode={proc.returncode}. "
                                  f"stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}")


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
