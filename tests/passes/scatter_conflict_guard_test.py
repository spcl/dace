# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the scatter-conflict guard utility / pass.

Covers the three TSVC scatter patterns (``s4113``, ``s491``, ``vas``), plus an
abort-detection test that runs the SDFG with a duplicate index and verifies the
program traps. Permutation-index runs are expected to terminate cleanly with the
correct numerical result (the scatter Map executes after the guard).
"""
import copy
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.scatter_conflict_guard import (GuardScatterConflicts, insert_scatter_guard,
                                                               names_are_free_symbols, scatter_index_domain,
                                                               scatter_index_is_provably_injective)
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent


def assert_cfg_list_matches_reset(sdfg: dace.SDFG) -> None:
    """The kept CFG list and ids equal a fresh copy's after ``reset_cfg_list``, and every parent pointer holds."""
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]
    assert_tree_consistent(sdfg)


N = dace.symbol('N')

# TSVC scatter kernels (1-D, integer index ``ip``)


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


# Helpers


def _has_conflict_check(sdfg: dace.SDFG) -> bool:
    return any(isinstance(n, ScatterConflictCheck) for n, _ in sdfg.all_nodes_recursive())


def _make_permutation(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int32)


def _function_body(code: str, header: str) -> str:
    """The brace-matched body of the first function in ``code`` whose text starts with ``header``."""
    start = code.index(header)
    open_brace = code.index('{', start)
    depth = 0
    for i in range(open_brace, len(code)):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return code[open_brace:i + 1]
    raise AssertionError(f"Unbalanced braces after {header!r}")


def _lines_inside_a_function(code: str, needle: str) -> list:
    """Lines containing ``needle`` that sit at nonzero brace depth (i.e. inside some function)."""
    depth = 0
    found = []
    for line in code.splitlines():
        if depth > 0 and needle in line:
            found.append(line.strip())
        depth += line.count('{') - line.count('}')
    return found


# Per-TSVC tests


def test_s4113_permutation_runs_cleanly():
    """s4113 with a permutation idx: guard runs (conflict check), no abort, correct result."""
    sdfg = tsvc_s4113.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()
    assert _has_conflict_check(sdfg)
    assert_cfg_list_matches_reset(sdfg)

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
    assert _has_conflict_check(sdfg)

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
    assert _has_conflict_check(sdfg)

    n = 32
    ip = _make_permutation(n, seed=4)
    b = np.random.default_rng(5).random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[i]
    sdfg(a=a, b=b, ip=ip, N=n)
    assert np.allclose(a, a_ref)


# Structural checks


def test_guard_states_inserted_before_scatter():
    """The guard's two states (conflict check, trap) are reachable from the SDFG
    start and precede every state that reads ``ip``."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    states_before = set(sdfg.states())
    insert_scatter_guard(sdfg, 'ip')
    new_states = set(sdfg.states()) - states_before

    assert len(new_states) == 2, (f"Expected exactly 2 new states (check+trap); got {len(new_states)}.")
    # Each new state's label carries the guard tag.
    new_labels = sorted(s.label for s in new_states)
    assert any('_scatter_guard_check_' in l for l in new_labels), new_labels
    assert any('_scatter_guard_trap_' in l for l in new_labels), new_labels

    # Both guard states sit at the head of the CFG.
    reachable_before_original = set()
    cur = sdfg.start_block
    while cur in new_states:
        reachable_before_original.add(cur)
        out = list(sdfg.out_edges(cur))
        if not out:
            break
        cur = out[0].dst
    assert reachable_before_original == new_states, (
        f"Both guard states should sit at the head of the CFG; reached {reachable_before_original}")


def test_guard_pass_emits_for_each_named_idx():
    """``GuardScatterConflicts(['ip'])`` emits one guard per named array."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    res = GuardScatterConflicts(['ip']).apply_pass(sdfg, {})
    assert res == 1
    assert _has_conflict_check(sdfg)


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


# Tag array: DaCe-owned transient, allocated outside the program body


def test_tag_array_is_a_persistent_transient_sized_by_the_scatter_domain():
    """The conflict check's tag array is a real descriptor sized by ``a``'s domain, not a
    runtime-sized buffer the libnode ``new``s behind DaCe's back."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    assert str(scatter_index_domain(sdfg, 'ip')) == 'N'
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()

    owner = sdfg.arrays['_scatter_guard_owner_ip']
    assert owner.transient
    assert owner.dtype == dace.int64
    assert str(owner.shape[0]) == 'N'  # the scattered array's domain, no runtime max(ip)
    assert owner.lifetime == dace.dtypes.AllocationLifetime.Persistent
    assert owner.storage == dace.dtypes.StorageType.CPU_Heap  # the check is host code everywhere


def every_way_a_symbol_is_defined() -> dace.SDFG:
    """``N`` a plain parameter; ``K`` assigned on an edge, ``M`` inside an if branch, ``i`` a loop variable."""
    sdfg = dace.SDFG('symbol_definitions')
    for name in ('N', 'K', 'M', 'i'):
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_array('a', [N], dace.float64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'K': 'N'}))
    pick = ConditionalBlock('pick')
    sdfg.add_node(pick)
    sdfg.add_edge(second, pick, dace.InterstateEdge())
    branch = ControlFlowRegion('then', sdfg=sdfg)
    head = branch.add_state('head', is_start_block=True)
    branch.add_edge(head, branch.add_state('tail'), dace.InterstateEdge(assignments={'M': 'K + 1'}))
    pick.add_branch(CodeBlock('N > 0'), branch)
    loop = LoopRegion('sweep', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(pick, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('w', {}, {'o'}, 'o = K + M')
    body.add_edge(tasklet, 'o', body.add_write('a'), None, dace.Memlet('a[i]'))
    return sdfg


@pytest.mark.parametrize('names', [{'N'}, {'K'}, {'M'}, {'i'}, {'a'}, {'undeclared'}, {'N', 'K'}, set()])
def test_free_symbol_shortcut_agrees_with_the_walk(names):
    """The tag lifetime hangs on this answer, so it must equal ``names <= sdfg.free_symbols`` for every
    way a name can be defined -- including an assignment hidden inside a conditional branch."""
    sdfg = every_way_a_symbol_is_defined()
    assert names_are_free_symbols(sdfg, names) == (names <= set(sdfg.free_symbols))


def count_sdfg_walks(monkeypatch) -> dict:
    """Count every ``SDFG.free_symbols`` evaluation from here on."""
    calls = {'walks': 0}
    inherited = dace.SDFG.free_symbols

    def counted(self):
        calls['walks'] += 1
        return inherited.fget(self)

    monkeypatch.setattr(dace.SDFG, 'free_symbols', property(counted))
    return calls


def test_free_symbol_shortcut_answers_a_plain_parameter_without_walking(monkeypatch):
    sdfg = every_way_a_symbol_is_defined()
    calls = count_sdfg_walks(monkeypatch)
    assert names_are_free_symbols(sdfg, {'N'})
    assert calls['walks'] == 0, 'a declared parameter nothing defines must not cost a whole-SDFG walk'


def test_guard_insertion_decides_the_tag_lifetime_without_walking_the_sdfg(monkeypatch):
    """One walk per guard was 23.7 s of ls3df_scf's scatter stage: 45 guards, each walking the whole SDFG
    only to learn that the domain ``Lb`` is a parameter."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    calls = count_sdfg_walks(monkeypatch)
    insert_scatter_guard(sdfg, 'ip')
    assert calls['walks'] == 0
    assert sdfg.arrays['_scatter_guard_owner_ip'].lifetime == dace.dtypes.AllocationLifetime.Persistent


def test_generated_guard_has_no_raw_new_and_no_include_in_the_program_body():
    """The timed program body holds no allocation, no preprocessor include and no sweep of its own:
    the tag array is allocated once in ``__dace_init``, and the two passes over ``ip`` live in
    :cpp:func:`dace::detect_collision`, which the body only CALLS -- one implementation to tune,
    and the caller-sized tag array means no ``max(ip)`` sizing sweep either."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    insert_scatter_guard(sdfg, 'ip')
    # Every assertion below reads the emitted body, and the two CPU generators spell a native
    # tasklet's operands differently: ``legacy`` keeps them in connector locals, so the tag array
    # would appear as ``_owner_out`` and the name checks would pass without testing anything.
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        code = sdfg.generate_code()[0].clean_code

    body = _function_body(code, f'void __program_{sdfg.name}_internal')
    assert 'new ' not in body, body
    assert '#include' not in body, body
    calls = [ln for ln in body.splitlines() if 'dace::detect_collision(' in ln]
    assert len(calls) == 1, body
    # A caller-sized tag array: the 5-argument form, so the runtime skips its own max(ip) sweep.
    assert '__0__scatter_guard_owner_ip' in calls[0], calls[0]
    # Neither pass is emitted as text any more: no tag write and no OR-reduce pragma in the body.
    # (A blanket ``for (`` check would trip on the scatter loop the guard exists to protect.)
    assert '__0__scatter_guard_owner_ip[' not in body, f'the tag write must live in the runtime, not here: {body}'
    assert 'reduction(|' not in body, f'the verify pass must live in the runtime, not here: {body}'

    assert not _lines_inside_a_function(code, '#include'), _lines_inside_a_function(code, '#include')
    init = _function_body(code, f'__dace_init_{sdfg.name}(')
    assert '__0__scatter_guard_owner_ip = new' in init, init


def test_runtime_or_reduce_pass_carries_simd():
    """The OR-reduce verify pass is bitwise-or, which is simd-safe (see
    dace/runtime/include/dace/reduction.h); its pragma must carry simd, and any ``if`` clause must
    NAME the directive it belongs to -- a bare ``if()`` on a combined construct binds to simd in
    GCC and silently devectorizes the loop.

    Asserted against the runtime header, because that is where the pass lives now: the expansions
    call :cpp:func:`dace::detect_collision` instead of each emitting a copy of the loop."""
    header = (pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'detect.h').read_text()
    reduce_lines = [ln.strip() for ln in header.splitlines() if 'reduction(| : c)' in ln]
    assert reduce_lines, header
    for line in reduce_lines:
        assert line.startswith('#pragma omp parallel for simd'), line
        assert 'if (parallel : parallel)' in line, f'an unqualified if() would bind to simd: {line}'


def test_tag_array_omitted_when_no_scatter_target_is_visible():
    """No derivable domain (no scatter loop to read a target extent from) -> no tag descriptor;
    the libnode keeps its runtime-sized buffer rather than guessing a bound."""
    sdfg = dace.SDFG('no_scatter_target')
    sdfg.add_array('ip', [8], dace.int32)
    sdfg.add_state('s0')
    assert scatter_index_domain(sdfg, 'ip') is None
    insert_scatter_guard(sdfg, 'ip')
    sdfg.validate()
    assert _has_conflict_check(sdfg)
    assert '_scatter_guard_owner_ip' not in sdfg.arrays


# Lever 1: static-injective elision


@dace.program
def scatter_affine_identity(a: dace.float64[N], b: dace.float64[N]):
    """``ip[i] = i`` produced in-SDFG (identity permutation), then ``a[ip[i]] = b[i]``."""
    ip = np.empty(N, np.int64)
    for i in range(N):
        ip[i] = i
    for i in range(N):
        a[ip[i]] = b[i]


@dace.program
def scatter_affine_strided(a: dace.float64[2 * N], b: dace.float64[N]):
    """``ip[i] = 2*i + 1`` (injective affine over ``[0, N)``), then ``a[ip[i]] = b[i]``."""
    ip = np.empty(N, np.int64)
    for i in range(N):
        ip[i] = 2 * i + 1
    for i in range(N):
        a[ip[i]] = b[i]


@dace.program
def scatter_mod_producer(a: dace.float64[N], b: dace.float64[N]):
    """``ip[i] = i % 3`` (non-injective) -- genuinely conflicts; the guard must be kept."""
    ip = np.empty(N, np.int64)
    for i in range(N):
        ip[i] = i % 3
    for i in range(N):
        a[ip[i]] = b[i]


def build_constant_idx_scatter_sdfg(values) -> dace.SDFG:
    """Build a minimal SDFG whose ``ip`` array is also a compile-time constant.

    Used to exercise the constant-array branch of
    :func:`scatter_index_is_provably_injective` without a producer loop.

    :param values: The integer values baked into the ``ip`` constant.
    :returns: An SDFG with an ``ip`` :class:`~dace.data.Array` descriptor whose contents
              are registered as a compile-time constant.
    """
    sdfg = dace.SDFG('const_idx_scatter')
    sdfg.add_array('ip', [len(values)], dace.int64)
    sdfg.add_constant('ip', np.asarray(values, dtype=np.int64))
    sdfg.add_state('s0')
    return sdfg


def test_affine_identity_producer_elides_guard():
    """An in-SDFG identity producer ``ip[i] = i`` is provably injective: the guard is elided
    (no ``ScatterConflictCheck`` node) and the plain scatter is value-correct vs numpy."""
    sdfg = scatter_affine_identity.to_sdfg(simplify=True)
    assert scatter_index_is_provably_injective(sdfg, 'ip')
    assert insert_scatter_guard(sdfg, 'ip') is None  # elided -> no guard symbol
    assert not _has_conflict_check(sdfg)
    sdfg.validate()

    n = 40
    b = np.random.default_rng(11).random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[i] = b[i]
    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, a_ref)


def test_strided_affine_producer_elides_guard():
    """A strided injective producer ``ip[i] = 2*i + 1`` is provably injective: guard elided,
    value-correct vs numpy."""
    sdfg = scatter_affine_strided.to_sdfg(simplify=True)
    assert scatter_index_is_provably_injective(sdfg, 'ip')
    assert insert_scatter_guard(sdfg, 'ip') is None
    assert not _has_conflict_check(sdfg)
    sdfg.validate()

    n = 16
    b = np.random.default_rng(12).random(n)
    a = np.zeros(2 * n)
    a_ref = np.zeros(2 * n)
    for i in range(n):
        a_ref[2 * i + 1] = b[i]
    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, a_ref)


def test_conflicting_param_idx_keeps_guard():
    """A parameter ``ip`` (unknown runtime contents) is NOT provably injective: the guard is
    kept, and with a permutation the guarded scatter is value-correct vs numpy."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    assert not scatter_index_is_provably_injective(sdfg, 'ip')
    insert_scatter_guard(sdfg, 'ip')
    assert _has_conflict_check(sdfg)
    sdfg.validate()

    n = 32
    ip = _make_permutation(n, seed=13)
    b = np.random.default_rng(14).random(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[i]
    sdfg(a=a, b=b, ip=ip, N=n)
    assert np.allclose(a, a_ref)


def test_non_injective_producer_not_provably_injective():
    """A non-affine producer ``ip[i] = i % 3`` can collide: the analysis refuses to prove
    injectivity (soundness first), so the guard would be kept rather than elided."""
    sdfg = scatter_mod_producer.to_sdfg(simplify=True)
    assert not scatter_index_is_provably_injective(sdfg, 'ip')


def test_constant_permutation_idx_is_injective():
    """A compile-time constant ``ip`` holding a permutation is provably injective."""
    sdfg = build_constant_idx_scatter_sdfg([3, 0, 2, 1])
    assert scatter_index_is_provably_injective(sdfg, 'ip')


def test_constant_duplicate_idx_not_injective():
    """A compile-time constant ``ip`` with a repeated value is NOT injective."""
    sdfg = build_constant_idx_scatter_sdfg([0, 1, 1, 2])
    assert not scatter_index_is_provably_injective(sdfg, 'ip')


# Abort-on-duplicate (subprocess; SIGABRT/SIGILL is expected)

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
    ``std::abort()`` doesn't kill the test runner. The subprocess prints
    a marker only if the abort *didn't* fire; we check the marker is absent
    AND the subprocess exited abnormally (non-zero return / signal).
    """
    proc = subprocess.run([sys.executable, '-c', _DUPLICATE_ABORT_SCRIPT], capture_output=True, text=True, timeout=120)
    assert 'UNEXPECTEDLY_SURVIVED' not in proc.stdout, (
        f"Guard failed to abort on duplicate idx. stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}")
    assert proc.returncode != 0, (f"Expected non-zero exit on trap; got returncode={proc.returncode}. "
                                  f"stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}")


if __name__ == '__main__':
    test_s4113_permutation_runs_cleanly()
    test_s491_permutation_runs_cleanly()
    test_vas_permutation_runs_cleanly()
    test_guard_states_inserted_before_scatter()
    test_guard_pass_emits_for_each_named_idx()
    test_guard_refuses_non_integer_idx()
    test_guard_refuses_unknown_idx_name()
    test_guard_refuses_double_emit()
    test_tag_array_is_a_persistent_transient_sized_by_the_scatter_domain()
    test_generated_guard_has_no_raw_new_and_no_include_in_the_program_body()
    test_generated_guard_or_reduce_pass_carries_simd()
    test_tag_array_omitted_when_no_scatter_target_is_visible()
    test_affine_identity_producer_elides_guard()
    test_strided_affine_producer_elides_guard()
    test_conflicting_param_idx_keeps_guard()
    test_non_injective_producer_not_provably_injective()
    test_constant_permutation_idx_is_injective()
    test_constant_duplicate_idx_not_injective()
    test_duplicate_idx_aborts_the_process()
