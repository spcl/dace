# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.scatter_to_guarded_maps.ScatterToGuardedMaps`.

End-to-end coverage: TSVC scatter kernels are detected, guarded, and parallelized
by a single pass invocation. Per-array detection is checked (multiple distinct
``idx`` arrays each get their own guard) and idempotence (re-running the pass on
an already-processed SDFG is a no-op for the guard step).
"""
import numpy as np
import pytest

import dace
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.scatter_to_guarded_maps import (ScatterToGuardedMaps, detect_scatter_idx_arrays)

N = dace.symbol('N')

# -- TSVC scatter kernels -----------------------------------------------------


@dace.program
def tsvc_s4113(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], ip: dace.int32[N]):
    for i in range(N):
        a[ip[i]] = b[ip[i]] + c[i]


@dace.program
def tsvc_s491(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], ip: dace.int32[N]):
    for i in range(N):
        a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def tsvc_vas(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N]):
    for i in range(N):
        a[ip[i]] = b[i]


@dace.program
def two_distinct_scatters(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                          ip: dace.int32[N], jp: dace.int32[N]):
    """Two scatter loops, each using a different ``idx`` array (``ip`` and ``jp``).

    Tests that the pass detects both ``idx`` arrays and guards each one separately.
    """
    for i in range(N):
        a[ip[i]] = b[i]
    for i in range(N):
        c[jp[i]] = d[i]


@dace.program
def no_scatter_just_elementwise(a: dace.float64[N], b: dace.float64[N]):
    """A plain elementwise loop -- no scatter. The pass should detect nothing
    but still parallelize via permissive (here equivalent to non-permissive) L2M.
    """
    for i in range(N):
        a[i] = b[i] * 2.0


SSYM = dace.symbol('SSYM')


@dace.program
def vas_ssym(a: dace.float64[N], b: dace.float64[N], ip: dace.int64[N]):
    """Symbolic-stride scatter ``a[ip[i*SSYM]] = b[i]`` (TSVC-2.5 ``vas_ssym``).
    The index array is inline-subscripted inside the write memlet subset
    (``a[ip[SSYM*i]]``) rather than bound on an interstate edge."""
    for i in range(N // SSYM):
        a[ip[i * SSYM]] = b[i]


@dace.program
def s4113_ssym(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], ip: dace.int64[N]):
    """Symbolic-stride gather+scatter ``a[ip[i*SSYM]] = b[ip[i*SSYM]] + c[i]``
    (TSVC-2.5 ``s4113_ssym``). The strided index ``ip[SSYM*i]`` is bound on the
    loop's interstate edge (``sym := ip[SSYM*i]``) and used in the write subset."""
    for i in range(N // SSYM):
        a[ip[i * SSYM]] = b[ip[i * SSYM]] + c[i]


def _build_nested_map_scatter_sdfg():
    """Hand-build the shape a ``dace.map`` scatter (``dst[idx[i]] = src[i]``) is
    lowered to inside the canonicalize pipeline: a ``LoopRegion`` whose body has a
    ``NestedSDFG`` writing ``dst`` with a data-dependent (whole-array, unit-volume)
    memlet, the write index living inside the nested SDFG and fed by ``idx[i]``.

    This is the intermediate form of TSVC-2.5 ``ext_scatter_store`` after the map
    is serialized; a fresh ``to_sdfg`` keeps it a Map instead, so we construct the
    shape directly to unit-test the nested-SDFG detection path.
    """
    from dace import Memlet

    sdfg = dace.SDFG('nested_map_scatter')
    sdfg.add_array('src', [N], dace.float64)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('dst', [N], dace.float64)

    loop = LoopRegion('scatter_loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)

    nsdfg = dace.SDFG('scatter_body')
    nsdfg.add_scalar('src_in', dace.float64)
    nsdfg.add_scalar('idx_in', dace.int64)
    nsdfg.add_array('dst_out', [N], dace.float64)
    nstate = nsdfg.add_state('s', is_start_block=True)
    read = nstate.add_read('src_in')
    write = nstate.add_write('dst_out')
    tasklet = nstate.add_tasklet('cp', {'inp'}, {'outp'}, 'outp = inp')
    nstate.add_edge(read, None, tasklet, 'inp', Memlet('src_in[0]'))
    # The write index ``idx_in`` (an input connector) drives the data-dependent write.
    nstate.add_edge(tasklet, 'outp', write, None, Memlet(data='dst_out', subset='idx_in'))

    nnode = body.add_nested_sdfg(nsdfg, {'src_in', 'idx_in'}, {'dst_out'}, symbol_mapping={'N': N})
    a_src = body.add_read('src')
    a_idx = body.add_read('idx')
    a_dst = body.add_write('dst')
    body.add_edge(a_src, None, nnode, 'src_in', Memlet('src[i]'))
    body.add_edge(a_idx, None, nnode, 'idx_in', Memlet('idx[i]'))
    # Data-dependent write: a single element (volume 1) scattered into 0:N.
    body.add_edge(nnode, 'dst_out', a_dst, None, Memlet(data='dst', subset='0:N', volume=1))
    sdfg.validate()
    return sdfg


# -- Helpers ------------------------------------------------------------------


def _count_integer_sort_nodes(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, IntegerSort))


def _count_map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _count_loop_regions(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _make_permutation(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int32)


# -- Detection tests ----------------------------------------------------------


def test_detect_finds_single_scatter():
    """``ip`` is the sole scatter idx; the detector returns ``{'ip'}``."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    assert detect_scatter_idx_arrays(sdfg) == {'ip'}


def test_detect_finds_each_distinct_idx_array():
    """Two scatter loops with different idx arrays -> both detected."""
    sdfg = two_distinct_scatters.to_sdfg(simplify=True)
    assert detect_scatter_idx_arrays(sdfg) == {'ip', 'jp'}


def test_detect_returns_empty_for_elementwise():
    """A plain elementwise loop has no indirect writes; detector returns empty."""
    sdfg = no_scatter_just_elementwise.to_sdfg(simplify=True)
    assert detect_scatter_idx_arrays(sdfg) == set()


# -- End-to-end tests ---------------------------------------------------------


@pytest.mark.parametrize('kernel,inputs_fn', [
    (tsvc_s4113, lambda n: {
        'b': np.random.default_rng(10).random(n),
        'c': np.random.default_rng(11).random(n),
    }),
    (tsvc_s491, lambda n: {
        'b': np.random.default_rng(20).random(n),
        'c': np.random.default_rng(21).random(n),
        'd': np.random.default_rng(22).random(n),
    }),
    (tsvc_vas, lambda n: {
        'b': np.random.default_rng(30).random(n)
    }),
])
def test_scatter_kernel_guarded_and_parallelized(kernel, inputs_fn):
    """Each TSVC scatter is guarded (1 IntegerSort node) and parallelized (>=1 MapEntry)."""
    sdfg = kernel.to_sdfg(simplify=True)
    loops_before = _count_loop_regions(sdfg)
    maps_before = _count_map_entries(sdfg)
    sort_before = _count_integer_sort_nodes(sdfg)

    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()

    assert rewritten == 1, f"Expected exactly 1 guarded idx for {kernel.name}; got {rewritten}."
    assert _count_integer_sort_nodes(sdfg) == sort_before + 1, "IntegerSort guard not emitted."
    assert _count_map_entries(sdfg) > maps_before, "Scatter loop was not parallelized to a Map."
    assert _count_loop_regions(sdfg) < loops_before, "Original LoopRegion was not lifted."

    n = 32
    ip = _make_permutation(n, seed=int(kernel.name.encode().hex(), 16) & 0xFFFFFF)
    kw = inputs_fn(n)
    a = np.zeros(n)
    # Reference: run the un-transformed Python kernel directly via the f.f() handle.
    a_ref = np.zeros(n)
    if kernel is tsvc_s4113:
        for i in range(n):
            a_ref[ip[i]] = kw['b'][ip[i]] + kw['c'][i]
    elif kernel is tsvc_s491:
        for i in range(n):
            a_ref[ip[i]] = kw['b'][i] + kw['c'][i] * kw['d'][i]
    else:  # vas
        for i in range(n):
            a_ref[ip[i]] = kw['b'][i]

    sdfg(a=a, ip=ip, N=n, **kw)
    assert np.allclose(a, a_ref), f"Numerical mismatch on {kernel.name}; max diff {np.max(np.abs(a - a_ref))}"


def test_two_distinct_scatters_get_individual_guards():
    """Two scatter loops with separate ``ip``/``jp`` arrays each get their own guard;
    both run cleanly under permutation indices."""
    sdfg = two_distinct_scatters.to_sdfg(simplify=True)
    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten == 2, f"Expected both ip and jp to be guarded; got {rewritten}."
    assert _count_integer_sort_nodes(sdfg) == 2, "Need one IntegerSort per idx array."

    n = 24
    ip = _make_permutation(n, seed=100)
    jp = _make_permutation(n, seed=101)
    rng = np.random.default_rng(102)
    b, d = rng.random(n), rng.random(n)
    a = np.zeros(n)
    c = np.zeros(n)
    a_ref = np.zeros(n)
    c_ref = np.zeros(n)
    for i in range(n):
        a_ref[ip[i]] = b[i]
        c_ref[jp[i]] = d[i]
    sdfg(a=a, b=b, c=c, d=d, ip=ip, jp=jp, N=n)
    assert np.allclose(a, a_ref)
    assert np.allclose(c, c_ref)


# -- Symbolic-stride scatter forms (TSVC-2.5) ---------------------------------


def test_detect_inline_subscript_symbolic_stride_scatter():
    """``vas_ssym`` writes ``a[ip[SSYM*i]]`` -- the index array is embedded inline
    in the write-memlet subset (no interstate binding). The detector must still
    resolve ``ip``."""
    sdfg = vas_ssym.to_sdfg(simplify=True)
    assert detect_scatter_idx_arrays(sdfg) == {'ip'}


def test_detect_strided_interstate_binding_scatter():
    """``s4113_ssym`` binds ``sym := ip[SSYM*i]`` (a *strided* index, not the bare
    loop var) on the loop's interstate edge. The detector must accept the affine
    index and resolve ``ip``."""
    sdfg = s4113_ssym.to_sdfg(simplify=True)
    assert detect_scatter_idx_arrays(sdfg) == {'ip'}


def _ref_vas_ssym(a, ip, ssym, kw):
    for i in range(a.shape[0] // ssym):
        a[ip[i * ssym]] = kw['b'][i]


def _ref_s4113_ssym(a, ip, ssym, kw):
    for i in range(a.shape[0] // ssym):
        a[ip[i * ssym]] = kw['b'][ip[i * ssym]] + kw['c'][i]


@pytest.mark.parametrize('kernel,inputs_fn,ref', [
    (vas_ssym, lambda n: {
        'b': np.random.default_rng(40).random(n)
    }, _ref_vas_ssym),
    (s4113_ssym, lambda n: {
        'b': np.random.default_rng(41).random(n),
        'c': np.random.default_rng(42).random(n),
    }, _ref_s4113_ssym),
])
def test_symbolic_stride_scatter_guarded_and_parallelized(kernel, inputs_fn, ref):
    """Each symbolic-stride scatter is guarded (1 IntegerSort) and lifted (>=1
    Map), and reproduces the sequential result bit-for-bit under a permutation
    ``ip``."""
    sdfg = kernel.to_sdfg(simplify=True)
    loops_before = _count_loop_regions(sdfg)
    maps_before = _count_map_entries(sdfg)

    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()

    assert rewritten == 1, f"Expected exactly 1 guarded idx for {kernel.name}; got {rewritten}."
    assert _count_integer_sort_nodes(sdfg) == 1, "IntegerSort guard not emitted."
    assert _count_map_entries(sdfg) > maps_before, "Scatter loop was not parallelized to a Map."
    assert _count_loop_regions(sdfg) < loops_before, "Original LoopRegion was not lifted."

    n, ssym = 30, 3
    ip = _make_permutation(n, seed=int(kernel.name.encode().hex(), 16) & 0xFFFFFF).astype(np.int64)
    kw = inputs_fn(n)
    a = np.zeros(n)
    a_ref = np.zeros(n)
    ref(a_ref, ip, ssym, kw)
    sdfg(a=a, ip=ip, N=n, SSYM=ssym, **kw)
    assert np.allclose(a, a_ref), f"Numerical mismatch on {kernel.name}; max diff {np.max(np.abs(a - a_ref))}"


def test_detect_and_lift_nested_map_scatter():
    """A ``dace.map`` scatter lowers to a nested-SDFG data-dependent write inside a
    loop (TSVC-2.5 ``ext_scatter_store``). The detector traces the write index
    through the nested SDFG back to ``idx``; the pass guards it and lifts the loop,
    preserving values bit-for-bit under a permutation ``idx``."""
    sdfg = _build_nested_map_scatter_sdfg()
    assert detect_scatter_idx_arrays(sdfg) == {'idx'}

    loops_before = _count_loop_regions(sdfg)
    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()

    assert rewritten == 1, f"Expected exactly one guarded idx (``idx``); got {rewritten}."
    assert _count_integer_sort_nodes(sdfg) == 1, "IntegerSort guard not emitted for the nested-SDFG scatter."
    assert _count_map_entries(sdfg) >= 1, "Nested-SDFG scatter loop was not parallelized to a Map."
    assert _count_loop_regions(sdfg) < loops_before, "Original LoopRegion was not lifted."

    n = 16
    idx = _make_permutation(n, seed=55).astype(np.int64)
    src = np.random.default_rng(56).standard_normal(n)
    dst = np.zeros(n)
    dst_ref = np.zeros(n)
    for i in range(n):
        dst_ref[idx[i]] = src[i]
    sdfg(src=src, idx=idx, dst=dst, N=n)
    assert np.allclose(dst,
                       dst_ref), f"Numerical mismatch on nested-map scatter; max diff {np.max(np.abs(dst - dst_ref))}"


def test_no_scatter_elementwise_no_op_modified_skips_global_permissive_lift():
    """**Contract changed.** A plain elementwise loop has no scatter idx; the
    pass now leaves it untouched (the previous behaviour ran a global permissive
    ``LoopToMap`` which corrupted unrelated loops with genuine carried deps --
    e.g. a ``WavefrontSkew``-rewritten outer ``t`` loop). The non-permissive
    ``LoopToMap`` in the ``canonicalize`` ``parallelize`` stage handles
    elementwise loops earlier in the pipeline, so this pass no longer needs to.
    Numerics on the elementwise loop are preserved via the original LoopRegion.
    """
    sdfg = no_scatter_just_elementwise.to_sdfg(simplify=True)
    maps_before = _count_map_entries(sdfg)
    loops_before = _count_loop_regions(sdfg)
    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()

    assert rewritten is None, "No idx array should be detected for a plain elementwise loop."
    assert _count_integer_sort_nodes(sdfg) == 0, "No guard should be emitted for non-scatter loops."
    assert _count_map_entries(sdfg) == maps_before, (
        "Non-scatter loops must NOT be permissively lifted here; the parallelize "
        "stage handles them.")
    assert _count_loop_regions(sdfg) == loops_before, (
        "Non-scatter LoopRegion must stay intact so a later strict ``LoopToMap`` "
        "can handle it on its own terms.")

    n = 16
    b = np.random.default_rng(0).random(n)
    a = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(a, b * 2.0)


def test_carry_loop_not_permissively_lifted_by_scatter_pass():
    """**New regression** for the bug this fix addresses. A loop with a genuine
    loop-carried dependence (``a[i] = a[i - 1] + 1``) that lives alongside a
    scatter loop must NOT be lifted by ``ScatterToGuardedMaps``: only the
    scatter loop gets the permissive lift, the carry loop stays a LoopRegion.
    """

    @dace.program
    def mixed(a: dace.float64[N], b: dace.float64[N], ip: dace.int32[N]):
        # Carry-only loop -- LoopToMap must refuse this even with permissive=True
        # (the i-1 read aliases the i-th write).
        for i in range(1, N):
            a[i] = a[i - 1] + 1.0
        # Scatter loop -- this is the one ScatterToGuardedMaps should lift.
        for j in range(N):
            b[ip[j]] = a[j]

    sdfg = mixed.to_sdfg(simplify=True)
    loops_before = _count_loop_regions(sdfg)
    rewritten = ScatterToGuardedMaps().apply_pass(sdfg, {})
    sdfg.validate()

    assert rewritten == 1, "Expected exactly one scatter idx (``ip``)."
    # The carry loop survives as a LoopRegion; the scatter loop is gone (lifted).
    assert _count_loop_regions(sdfg) == loops_before - 1, (
        "Only the scatter loop should have been lifted; the carry loop must "
        "stay as a LoopRegion.")

    # Numerical correctness: the carry chain still resolves correctly.
    n = 12
    rng = np.random.default_rng(0)
    a = np.zeros(n)
    a[0] = 3.0
    b = np.zeros(n)
    ip = rng.permutation(n).astype(np.int32)
    expected_a = a.copy()
    for i in range(1, n):
        expected_a[i] = expected_a[i - 1] + 1.0
    expected_b = np.zeros(n)
    for j in range(n):
        expected_b[ip[j]] = expected_a[j]
    sdfg(a=a, b=b, ip=ip, N=n)
    assert np.allclose(a, expected_a), f"carry chain corrupted: {a} vs {expected_a}"
    assert np.allclose(b, expected_b), f"scatter wrong: {b} vs {expected_b}"


def test_idempotent_on_already_guarded_sdfg():
    """Re-running the pass on a previously-guarded SDFG does not double-emit guards."""
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    ScatterToGuardedMaps().apply_pass(sdfg, {})
    sort_after_first = _count_integer_sort_nodes(sdfg)
    ScatterToGuardedMaps().apply_pass(sdfg, {})
    assert _count_integer_sort_nodes(sdfg) == sort_after_first, ("Re-running the pass must not duplicate the guard.")


# -- emit_unparallelized_else_branch=True: runtime dispatcher tests ------------


def _vas_sequential(b: np.ndarray, ip: np.ndarray, n: int) -> np.ndarray:
    """Sequential reference for ``tsvc_vas``: ``a[ip[i]] = b[i]``. With
    duplicates in ``ip`` the last write wins -- this is the semantics the
    else-branch (collision fallback) must preserve."""
    a = np.zeros(n)
    for i in range(n):
        a[ip[i]] = b[i]
    return a


def test_else_branch_permutation_takes_parallel_path():
    """``emit_unparallelized_else_branch=True`` + ``ip`` is a permutation:
    the dup-count is 0, the conditional routes to the parallel branch, and
    the result matches the sequential reference (because no collision means
    sequential and parallel produce the same values).
    """
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    ScatterToGuardedMaps(emit_unparallelized_else_branch=True).apply_pass(sdfg, {})
    sdfg.validate()

    n = 32
    b = np.random.default_rng(0).random(n)
    ip = _make_permutation(n, seed=42)
    a = np.zeros(n)
    sdfg(a=a, ip=ip.astype(np.int32), b=b, N=n)
    assert np.allclose(a, _vas_sequential(b, ip, n)), (f'parallel branch must compute the same values as the '
                                                       f'sequential reference for a permutation idx')


def test_else_branch_duplicate_idx_takes_sequential_path():
    """``emit_unparallelized_else_branch=True`` + ``ip`` has duplicates: the
    dup-count is positive, the conditional routes to the sequential clone,
    and the result matches the deterministic last-write-wins semantics.

    Without the else-branch (default trap mode) this same input would invoke
    ``__builtin_trap()`` and abort the process; the dispatcher mode degrades
    to a correct sequential run instead.
    """
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    ScatterToGuardedMaps(emit_unparallelized_else_branch=True).apply_pass(sdfg, {})
    sdfg.validate()

    n = 32
    b = np.random.default_rng(1).random(n)
    # Intentional collisions: two distinct ``i`` map to the same destination.
    ip = np.arange(n, dtype=np.int32)
    ip[5] = ip[10]  # duplicate destination -- the parallel write would race
    ip[20] = ip[10]  # another duplicate at the same destination
    a = np.zeros(n)
    sdfg(a=a, ip=ip, b=b, N=n)
    assert np.allclose(a, _vas_sequential(b, ip, n)), (f'sequential fallback must produce the last-write-wins '
                                                       f'result on duplicated idx')


def test_else_branch_dispatcher_emits_both_branches():
    """Structural pin: after ``emit_unparallelized_else_branch=True``, the
    SDFG contains a ``ConditionalBlock`` with BOTH a sequential branch
    (carrying a clone of the original ``LoopRegion``) and a parallel branch
    (where ``LoopToMap`` has lifted the loop to a Map).
    """
    from dace.sdfg.state import ConditionalBlock
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    ScatterToGuardedMaps(emit_unparallelized_else_branch=True).apply_pass(sdfg, {})
    sdfg.validate()

    cond_blocks = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_blocks) == 1, f'expected one ConditionalBlock dispatcher, got {len(cond_blocks)}'
    cb = cond_blocks[0]
    assert len(cb.branches) == 2, f'dispatcher must have both branches (seq + par), got {len(cb.branches)}'
    seq_cond, seq_body = cb.branches[0]
    par_cond, par_body = cb.branches[1]
    assert seq_cond is not None and '> 0' in seq_cond.as_string, (f'sequential branch guard must dispatch on '
                                                                  f'``dup_count > 0``; got {seq_cond}')
    assert par_cond is None, 'parallel branch must be the unconditional ``else``'
    # Sequential branch keeps a LoopRegion (the clone).
    seq_loops = [n for n in seq_body.nodes() if isinstance(n, LoopRegion)]
    assert seq_loops, 'sequential branch must contain the cloned LoopRegion'
    # Parallel branch's loop has been lifted (no LoopRegion left, at least one MapEntry).
    par_loops = [n for n in par_body.nodes() if isinstance(n, LoopRegion)]
    par_maps = sum(1 for n, _ in par_body.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert not par_loops and par_maps >= 1, (f'parallel branch must have been lifted to a Map; '
                                             f'got loops={len(par_loops)}, maps={par_maps}')


# -- assume_no_conflicts=True: skip the guard entirely -----------------------


def test_assume_no_conflicts_skips_guard_and_lifts_unconditionally():
    """``assume_no_conflicts=True``: the caller asserts ``ip`` is a permutation, so the
    pass emits NO sort/dup-count guard and NO ConditionalBlock -- the scatter loop is
    lifted straight to an unconditional parallel Map. Values match the sequential
    reference when ``ip`` really is a permutation (the caller's contract)."""
    from dace.sdfg.state import ConditionalBlock
    sdfg = tsvc_vas.to_sdfg(simplify=True)
    loops_before = _count_loop_regions(sdfg)
    maps_before = _count_map_entries(sdfg)

    ScatterToGuardedMaps(assume_no_conflicts=True).apply_pass(sdfg, {})
    sdfg.validate()

    assert _count_integer_sort_nodes(sdfg) == 0, 'assume mode must emit NO sort guard'
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)], \
        'assume mode must emit NO ConditionalBlock (no if-else fallback)'
    assert _count_map_entries(sdfg) > maps_before, 'the scatter loop must be lifted to a Map'
    assert _count_loop_regions(sdfg) < loops_before, 'the original LoopRegion must be lifted'

    n = 32
    b = np.random.default_rng(0).random(n)
    ip = _make_permutation(n, seed=42)
    a = np.zeros(n)
    sdfg(a=a, ip=ip.astype(np.int32), b=b, N=n)
    assert np.allclose(a, _vas_sequential(b, ip, n)), \
        'assume-mode parallel Map must match the sequential reference for a permutation idx'


@pytest.mark.parametrize('kernel', [tsvc_s4113, tsvc_s491, tsvc_vas])
def test_no_conflict_guard_survives_full_canonicalize(kernel):
    """The runtime no-conflict guard (IntegerSort + __builtin_trap) must survive
    the WHOLE canonicalize pipeline -- including the terminal SimplifyPass. The
    trap tasklet has no data output, so unless it is marked side-effecting,
    DeadDataflowElimination prunes it and the sort/count chain feeding it looks
    dead too, silently dropping the guard and leaving the scatter Map unguarded.
    """
    from dace.transformation.passes.canonicalize.pipeline import canonicalize
    sdfg = kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()

    has_sort = any(isinstance(n, IntegerSort) for n, _ in sdfg.all_nodes_recursive())
    has_trap = any(
        isinstance(n, nodes.Tasklet) and '__builtin_trap' in n.code.as_string for st in sdfg.all_states()
        for n in st.nodes())
    maps = sum(1 for st in sdfg.all_states() for n in st.nodes()
               if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None)
    assert maps >= 1, 'the scatter must parallelize into a Map'
    assert has_sort and has_trap, ('the no-conflict guard (IntegerSort + trap) must survive full '
                                   f'canonicalize; got sort={has_sort} trap={has_trap}')


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
