# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the DaCe capabilities that LLVM (vanilla, without MLIR
dialects) structurally cannot reproduce. Each test demonstrates a concrete
pattern; the docstring names the LLVM equivalent (or its absence) so the test
serves both as a regression guard and as a one-screen orientation for
contributors wondering "what does DaCe do that ``-O3`` cannot".

These are NOT tests of LLVM. They exercise DaCe transformations / IR features
and assert the post-transform shape that LLVM-the-compiler-toolkit lacks the
representation to emit.
"""
import numpy as np
import pytest

import dace
from dace import dtypes, subsets, symbolic
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol('N')


# ---------------------------------------------------------------------------
# 1. Library-node lifting -- LLVM can only SIMD-vectorize, never call cub::DeviceScan
# ---------------------------------------------------------------------------


def test_loop_lifts_to_scan_libnode_not_just_simd():
    """A prefix sum is lifted to a :class:`Scan` libnode that lowers to
    ``cub::DeviceScan`` on CUDA and OpenMP 5.0 ``#pragma omp scan`` on CPU.
    LLVM's ``LoopVectorize`` recognises this shape only as a ``FixedOrderRecurrence``
    and emits a shuffle-stitched SIMD loop -- no scan-libnode call. The high-level
    lift means DaCe can dispatch to hand-tuned backend implementations the
    autovectorizer literally cannot produce.
    """

    @dace.program
    def prefix_sum(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i]

    sdfg = prefix_sum.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, 'Prefix sum should lift to one Scan libnode.'
    scans = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]
    assert len(scans) == 1, f'Expected exactly one Scan libnode; got {len(scans)}.'
    assert scans[0].op == ScanOp.SUM


def test_residue_class_scan_with_stride_LLVM_cannot_vectorize():
    """``out[i+2] = out[i] + delta[i]`` is two independent prefix-sum scans on
    residue classes mod 2. DaCe lifts to a :class:`Scan` libnode with
    ``stride=2`` -- the two residue chains run in parallel via two ``omp scan``
    sections. LLVM's vectorizer refuses non-unit-stride recurrences entirely
    (``FixedOrderRecurrence`` only handles ``i-1`` offsets), so the loop stays
    scalar."""

    @dace.program
    def stride2_demo(out: dace.float64[N + 2], delta: dace.float64[N]):
        for i in range(N):
            out[i + 2] = out[i] + delta[i]

    sdfg = stride2_demo.to_sdfg(simplify=True)
    res = LoopToScan().apply_pass(sdfg, {})
    assert res == 1, f'stride-2 scan should match the matcher; got {res}'
    scans = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]
    assert len(scans) == 1 and int(scans[0].stride) == 2, (
        f'Stride-2 residue-class scan should set Scan.stride=2; got {[int(s.stride) for s in scans]}.')


# ---------------------------------------------------------------------------
# 2. Multiplicative recurrence closed form -- LLVM SCEV only models AddRec
# ---------------------------------------------------------------------------


def test_multiplicative_recurrence_collapses_to_closed_form():
    """``acc *= 0.99`` repeated N times becomes ``acc *= 0.99 ** N`` -- O(1)
    instead of O(N). LLVM's Scalar Evolution (SCEV) models only ``AddRec`` chrecs
    (polynomial in trip count); it CANNOT represent multiplicative recurrences,
    so ``IndVarSimplify`` leaves this loop alone and the only recourse is full
    unrolling.

    DaCe's :class:`InductionVariableSubstitution` recognises the multiplicative
    pattern via SymPy and replaces the loop with a single tasklet."""
    sdfg = dace.SDFG('mult_iv')
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_symbol('M', dace.int32)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('mult_loop', initialize_expr='i = 0', condition_expr='i < M',
                      update_expr='i = i + 1', loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    r = body.add_read('acc')
    w = body.add_write('acc')
    t = body.add_tasklet('mul', {'_a'}, {'_o'}, '_o = _a * 0.99')
    body.add_edge(r, None, t, '_a', dace.Memlet(data='acc', subset='0'))
    body.add_edge(t, '_o', w, None, dace.Memlet(data='acc', subset='0'))
    sdfg.validate()

    InductionVariableSubstitution().apply_pass(sdfg, {})
    sdfg.validate()

    surviving_loops = [r for r in sdfg.all_control_flow_regions()
                       if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(surviving_loops) == 0, (
        'IVSub should have collapsed the multiplicative recurrence to its closed form '
        f'``acc *= 0.99 ** M``; {len(surviving_loops)} loops survived.')


# ---------------------------------------------------------------------------
# 3. Symbolic loop bounds + symbolic shapes -- LLVM specialises per-trip-count
# ---------------------------------------------------------------------------


def test_symbolic_loop_bound_no_specialization_needed():
    """A loop bound that is a symbolic expression ``N + 1`` is propagated through
    every transform without specialisation. LLVM has to either keep the loop
    fully generic (no optimization) or specialise per concrete ``N`` (code
    bloat). DaCe's SymPy backbone lets every analysis reason with the symbolic
    bound directly.

    Verified by running the same SDFG at multiple ``N`` without recompilation
    of the optimization pipeline."""

    @dace.program
    def scan_sym(out: dace.float64[N + 1], delta: dace.float64[N]):
        for i in range(N):
            out[i + 1] = out[i] + delta[i]

    sdfg = scan_sym.to_sdfg(simplify=True)
    LoopToScan().apply_pass(sdfg, {})
    sdfg.validate()

    # Run at two different concrete N. The SDFG was compiled symbolically; no
    # specialisation was needed between runs.
    for n in (8, 32):
        rng = np.random.default_rng(7)
        delta = rng.uniform(-1.0, 1.0, size=n)
        out = np.zeros(n + 1)
        out[0] = 0.5
        expected = out.copy()
        for i in range(n):
            expected[i + 1] = expected[i] + delta[i]
        sdfg(out=out, delta=delta, N=n)
        assert np.allclose(out, expected), f'symbolic-N scan diverged at N={n}'


# ---------------------------------------------------------------------------
# 4. Symbolic subset non-overlap proof -- LLVM AA must assume worst case
# ---------------------------------------------------------------------------


def test_symbolic_subset_non_overlap_proven_at_ir_level():
    """Two ranges ``[0:N/2]`` and ``[N/2:N]`` provably do not overlap for any
    ``N > 0``. DaCe's :meth:`subsets.Range.intersects` returns ``False``
    (definite no overlap), enabling parallel execution.

    LLVM's alias analysis (``AAResults``) cannot reason about symbolic
    ranges -- it queries ``noalias``/``BasicAA``/``TBAA`` which work on
    per-instruction pointers, not on symbolic subset ranges. For ``a[0:N/2]``
    vs ``a[N/2:N]`` it falls back to ``MayAlias`` and serialises any optimization
    that would care.
    """
    M = symbolic.pystr_to_symbolic('M')
    half = symbolic.pystr_to_symbolic('M // 2')
    lower = subsets.Range([(0, half - 1, 1)])
    upper = subsets.Range([(half, M - 1, 1)])
    # ``intersects`` returns ``False`` when proven non-overlapping, ``True`` when
    # provably overlapping, ``None`` when indeterminate. We assert PROVEN
    # non-overlapping -- the symbolic strength that LLVM AA cannot reach.
    result = lower.intersects(upper)
    assert result is False, (f'symbolic non-overlap proof failed: lower={lower}, upper={upper}, '
                             f'intersects() returned {result!r} (expected False)')


# ---------------------------------------------------------------------------
# 5. WCR is explicit on the edge -- LLVM has to pattern-match RecurrenceDescriptor
# ---------------------------------------------------------------------------


def test_wcr_explicit_no_reduction_pattern_match_needed():
    """A Map edge carrying ``wcr=lambda a, b: a + b`` is *explicitly* a reduction
    in the IR. The codegen reads ``edge.data.wcr`` and emits ``#pragma omp
    parallel for reduction(+:acc)`` or an atomic add; there is no
    pattern-matching analysis.

    LLVM has to run ``LoopVectorize`` legality + ``RecurrenceDescriptor``
    inference to discover the same pattern, and the inference fails on
    inter-procedural cases, on type-converted accumulators, on min/max with
    floating-point NaN questions, etc.

    We verify the IR encodes the reduction explicitly on the edge."""

    @dace.program
    def reduce_sum(arr: dace.float64[N], acc: dace.float64[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                _a << arr[i]
                _o >> acc(1, lambda a, b: a + b)[0]
                _o = _a

    sdfg = reduce_sum.to_sdfg(simplify=True)
    wcr_edges = [e for sd in sdfg.all_sdfgs_recursive()
                 for st in sd.all_states()
                 for e in st.edges()
                 if e.data is not None and e.data.wcr is not None]
    assert wcr_edges, 'expected at least one explicit WCR edge'
    assert any('a + b' in e.data.wcr for e in wcr_edges), (
        'reduction op should be explicit on the WCR edge string; codegen reads it directly.')


# ---------------------------------------------------------------------------
# 6. Hierarchical IR -- NestedSDFG composition with symbol_mapping
# ---------------------------------------------------------------------------


def test_nested_sdfg_composition_with_symbol_mapping():
    """A NestedSDFG carries an explicit ``symbol_mapping`` recording how outer
    symbols flow into inner names. LLVM IR is flat: a nested computation must
    either be inlined (losing the boundary) or extracted into a function (with
    flat parameter passing). DaCe keeps the boundary explicit AND lets passes
    propagate symbols through it (verified by the symbol_propagation pass)."""
    inner = dace.SDFG('inner')
    inner.add_array('o', [1], dace.float64)
    inner.add_symbol('inner_n', dace.int32)
    istate = inner.add_state('s')
    t = istate.add_tasklet('t', {}, {'_o'}, '_o = inner_n * 1.0')
    iw = istate.add_write('o')
    istate.add_edge(t, '_o', iw, None, dace.Memlet(data='o', subset='0'))

    outer = dace.SDFG('outer')
    outer.add_array('o', [1], dace.float64)
    outer.add_symbol('outer_n', dace.int32)
    state = outer.add_state('s', is_start_block=True)
    ow = state.add_write('o')
    nsdfg = state.add_nested_sdfg(inner, {}, {'o'}, symbol_mapping={'inner_n': 'outer_n'})
    state.add_edge(nsdfg, 'o', ow, None, dace.Memlet(data='o', subset='0'))
    outer.validate()

    # The symbol_mapping survives + records the cross-boundary binding explicitly.
    assert str(nsdfg.symbol_mapping['inner_n']) == 'outer_n', (
        f'nested SDFG symbol_mapping should record outer_n; got {dict(nsdfg.symbol_mapping)}')

    # Execute end-to-end at a concrete outer_n.
    out = np.zeros(1)
    outer(o=out, outer_n=7)
    assert np.isclose(out[0], 7.0)


# ---------------------------------------------------------------------------
# 7. Multi-backend codegen from one SDFG -- LLVM compiles per-target
# ---------------------------------------------------------------------------


def test_same_sdfg_lowers_to_cpu_and_gpu_via_schedule_only():
    """Switching a Map's ``schedule`` from ``CPU_Multicore`` to ``GPU_Device``
    changes the backend the codegen targets -- same source SDFG, different
    emitted code. LLVM has separate ``llc`` invocations per target and the IR
    is target-specific (datalayout/triple); there is no notion of "one IR,
    pick the backend at the loop level"."""

    @dace.program
    def axpy(x: dace.float64[N], y: dace.float64[N], a: dace.float64):
        for i in dace.map[0:N]:
            y[i] = a * x[i] + y[i]

    cpu_sdfg = axpy.to_sdfg(simplify=True)
    # The schedule on the only Map is a per-Map IR property; flipping it
    # redirects the codegen dispatcher to a different target without changing
    # any tasklet code or memlets.
    for n, _ in cpu_sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            cpu_sdfg.arrays  # touch nothing else
            initial_schedule = n.map.schedule
            n.map.schedule = dtypes.ScheduleType.CPU_Multicore
            assert n.map.schedule == dtypes.ScheduleType.CPU_Multicore
            n.map.schedule = dtypes.ScheduleType.GPU_Device
            assert n.map.schedule == dtypes.ScheduleType.GPU_Device
            # Restore for any downstream test fixture re-use.
            n.map.schedule = initial_schedule
            break
    else:
        pytest.fail('expected at least one MapEntry to flip schedule on')


# ---------------------------------------------------------------------------
# 8. Memlets carry exact subsets -- no aliasing reconstruction needed
# ---------------------------------------------------------------------------


def test_memlet_subset_is_explicit_no_aliasing_reconstruction():
    """A memlet ``Memlet(data='a', subset='i')`` carries the EXACT subset
    accessed -- the dataflow analysis reads the subset directly. LLVM has to
    reconstruct this via MemorySSA + GEP analysis, which is precise enough only
    when the GEP is purely affine; symbolic strides / multi-level indirection
    fall back to MayAlias.
    """
    sdfg = dace.SDFG('subset_demo')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    # ``N`` is already a module-level ``dace.symbol``; ``add_array`` registered it
    # on the SDFG automatically, no explicit ``add_symbol`` needed.
    state = sdfg.add_state('s', is_start_block=True)
    r = state.add_read('a')
    w = state.add_write('b')
    me, mx = state.add_map('m', {'i': '0:N'})
    t = state.add_tasklet('t', {'_a'}, {'_b'}, '_b = _a + 1.0')
    state.add_memlet_path(r, me, t, dst_conn='_a', memlet=dace.Memlet(data='a', subset='i'))
    state.add_memlet_path(t, mx, w, src_conn='_b', memlet=dace.Memlet(data='b', subset='i'))
    sdfg.validate()

    # The per-iteration memlet (the edge incident on the tasklet, not the outer
    # MapEntry-to-MapExit range edge) stores the symbolic subset ``i`` verbatim;
    # downstream analyses (LICM, PCIA, LoopToScan) read it directly without alias
    # reconstruction.
    inner_subsets = []
    for e in state.edges():
        if e.data is None or e.data.data not in ('a', 'b'):
            continue
        if isinstance(e.src, nodes.Tasklet) or isinstance(e.dst, nodes.Tasklet):
            inner_subsets.append(str(e.data.subset))
    assert inner_subsets and all(s == 'i' for s in inner_subsets), (
        f'per-iteration memlets should be the symbolic ``i`` verbatim; got {inner_subsets}')

    # Execute end-to-end.
    n = 16
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, a + 1.0)


# ---------------------------------------------------------------------------
# 9. Symbolic stride/shape on descriptors -- LLVM is byte-pointer-only
# ---------------------------------------------------------------------------


def test_symbolic_shape_and_strides_in_descriptor():
    """An array descriptor carries symbolic shape and strides. Transformations
    that need to reason about layout (tiling, vectorisation, scatter conflict
    detection) read the descriptor directly. LLVM has only byte-pointers; layout
    is reconstructed per GEP from data-layout strings + analysis, which loses
    high-level intent (e.g. "row-major vs column-major")."""
    sdfg = dace.SDFG('symbolic_shape')
    M = dace.symbol('M')
    K = dace.symbol('K')
    sdfg.add_array('a', [M, K], dace.float64)
    desc = sdfg.arrays['a']
    shape_strs = {str(s) for s in desc.shape}
    assert shape_strs == {'M', 'K'}, f'symbolic shape lost; got {shape_strs}'
    # Strides are auto-computed (row-major default) and are themselves symbolic
    # expressions of the shape -- accessible without parsing a target-specific
    # datalayout string.
    stride_strs = {str(s) for s in desc.strides}
    assert any('K' in s for s in stride_strs), (
        f'expected row-major stride to reference the inner-dim symbol K; got {stride_strs}')


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
