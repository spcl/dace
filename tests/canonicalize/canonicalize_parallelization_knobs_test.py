# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize's parallelization knobs, exercised end-to-end through the
pipeline: ``break_anti_dependence`` (snapshot-rename a read-ahead WAR) and
``peel_limit`` (best-effort loop peeling). Both target loops that ``LoopToMap``
would otherwise refuse; the per-target presets (see
``dace.transformation.passes.canonicalize.pipeline._CPU_DEFAULTS`` /
``_GPU_DEFAULTS``) now turn both on by default. Each test pins the OFF
baseline explicitly (``peel_limit=0, break_anti_dependence=False``) so the
"without the knob the loop stays sequential" contract is robust against
future default changes; the ON case enables the relevant knob and asserts
the loop becomes a Map AND stays value-preserving."""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


@dace.program
def _s121(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = a[i + 1] + b[i]


def test_break_anti_dependence_knob_parallelizes():
    """``a[i] = a[i+1] + b[i]`` is a pure read-ahead WAR (TSVC s121): off by
    default it stays a sequential loop (LoopToMap refuses the read-write
    conflict); with ``break_anti_dependence=True`` the array is snapshot-renamed
    so the loop becomes a Map, value-preserving."""
    off = _s121.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'WAR loop must stay sequential without the knob'

    on = _s121.to_sdfg(simplify=True)
    canonicalize(on, validate=True, break_anti_dependence=True)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'break_anti_dependence must parallelize the WAR loop'

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(7):
        ref[i] = a[i + 1] + b[i]  # reads the ORIGINAL a (read-ahead)
    got = a.copy()
    on(a=got, b=b.copy(), N=8)
    assert np.allclose(got, ref)


@dace.program
def _front_conflict(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == 0:
            A[N - 1] = A[N - 1] + 1.0


def test_loop_peeling_front_conflict_knob_parallelizes():
    """A first-iteration guard writes a conflicting extra location
    (``if i==0: A[N-1]+=1``): off by default the write-write conflict keeps the
    loop sequential; with ``peel_limit>0`` the front iteration is peeled off and
    the now-dead guard pruned, leaving a disjoint-write remainder that maps,
    value-preserving."""
    off = _front_conflict.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'boundary-conflict loop must stay sequential without the knob'

    on = _front_conflict.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling must unblock the boundary-conflict loop'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _front_conflict.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _back_conflict(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == N - 1:
            A[0] = A[0] + 1.0


def test_loop_peeling_back_conflict_knob_parallelizes():
    """A last-iteration guard writes a conflicting extra location
    (``if i==N-1: A[0]+=1``): off by default the write-write conflict keeps the
    loop sequential; with ``peel_limit>0`` the final iteration is peeled off
    (anchored on the concrete loop end, not the loop variable) and the now-dead
    guard pruned, leaving a disjoint-write remainder that maps, value-preserving.
    Exercises the back-peel path, which must substitute ``end - k*stride`` so no
    loop-defined symbol survives past the loop to block LoopToMap."""
    off = _back_conflict.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'boundary-conflict loop must stay sequential without the knob'

    on = _back_conflict.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling must unblock the boundary-conflict loop'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _back_conflict.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _multi_front(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == 0:
            A[N - 1] = A[N - 1] + 1.0
        elif i == 1:
            A[N - 2] = A[N - 2] + 1.0


def test_loop_peeling_multi_front_iter_knob_parallelizes():
    """Two first-iteration guards each write a conflicting extra location
    (``if i==0: A[N-1]+=1 elif i==1: A[N-2]+=1``): peeling must take off the first
    *two* iterations and prune both now-dead boundary guards (the if/elif lowers to
    nested conditionals) so the disjoint-write remainder maps. Exercises peeling at
    count>1 plus the multi-branch dead-guard collapse."""
    off = _multi_front.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'multi-iteration boundary conflict must stay sequential off'

    on = _multi_front.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling two front iterations must unblock the loop'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _multi_front.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _multi_front_else(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        if i == 0:
            A[N - 1] = B[i] + 1.0
        elif i == 1:
            A[N - 2] = B[i] + 1.0
        else:
            A[i] = B[i] * 2.0


def test_loop_peeling_multi_front_else_iter_knob_parallelizes():
    """The if/elif/**else** form (``if i==0 ... elif i==1 ... else: A[i]=...``): after
    peeling two front iterations the special-cased arms are dead contradictions and
    the else body is a range tautology -- the guards collapse, leaving the bare
    ``A[i]=B[i]*2`` remainder that maps. Exercises tautology unwrapping of the else
    arm, not just dead-branch removal."""
    off = _multi_front_else.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'if/elif/else boundary conflict must stay sequential off'

    on = _multi_front_else.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling must collapse the if/elif/else and map the remainder'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _multi_front_else.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _multi_back(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == N - 1:
            A[0] = A[0] + 1.0
        elif i == N - 2:
            A[1] = A[1] + 1.0


def test_loop_peeling_multi_back_iter_knob_parallelizes():
    """The final-iterations analogue (``if i==N-1 ... elif i==N-2 ...``): peeling must
    take off the last *two* iterations (anchored on the loop end, no loop symbol
    leaking) and prune both dead guards so the remainder maps."""
    off = _multi_back.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'multi-iteration tail conflict must stay sequential off'

    on = _multi_back.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling two tail iterations must unblock the loop'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _multi_back.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


@dace.program
def _fixed_read(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        a[i] = a[0] + b[i]


def test_loop_peeling_fixed_read_first_iter_knob_parallelizes():
    """``a[i] = a[1] + b[i]`` (textbook): every iteration reads the fixed ``a[1]``,
    which iteration 1 itself writes -- a loop-carried flow dependence (0-indexed:
    iteration 0 writes ``a[0]``, the rest read it). Off by default it stays a
    sequential loop; with ``peel_limit>0`` iteration 0 is peeled off and the
    remainder reads a now-fixed ``a[0]`` (disjoint from the ``a[1:N]`` writes), so
    it maps and runs, value-preserving. Exercises the LoopToMap conflict-analysis
    fix (a loop-invariant read disjoint from the ranged write is not a conflict)
    plus the LICM map-scope fix (the read of a map-written array is not hoisted)."""
    off = _fixed_read.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'the carried fixed-read loop must stay sequential without the knob'

    on = _fixed_read.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling iteration 0 must unblock the fixed-read remainder'

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) + 0.5
    ref_a = a.copy()
    _fixed_read.to_sdfg(simplify=True)(a=ref_a, b=b.copy(), N=8)
    got = a.copy()
    on(a=got, b=b.copy(), N=8)
    assert np.allclose(got, ref_a)


@dace.program
def _mod_wrap_plus1(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[(i + 1) % N] = B[i] * 2.0


@dace.program
def _mod_wrap_minus1(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[(i - 1) % N] = B[i] * 2.0


@dace.program
def _mod_wrap_plus3(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[(i + 3) % N] = B[i] * 2.0


@pytest.mark.parametrize('prog,offset', [(_mod_wrap_plus1, 1), (_mod_wrap_minus1, -1), (_mod_wrap_plus3, 3)])
def test_loop_peeling_modulo_wraparound_knob_parallelizes(prog, offset):
    """A wrap-around write ``A[(i + k) % N] = ...`` is non-affine (the ``% N`` wraps
    at the boundary), so LoopToMap refuses it. With ``peel_limit>0`` the wrapping
    iterations are peeled off (the back for a positive offset, the front for a
    negative one) and the modulo over the remainder is folded to the plain affine
    index ``i + k`` -- so the body maps. The fold (band reduction) makes both the
    remainder and the peeled iterations affine, so the result never depends on C's
    truncated ``%`` (correct for any sign of ``k`` and symbolic ``N``)."""
    off = prog.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'wrap-around modulo write must stay sequential without the knob'

    on = prog.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling + modulo fold must parallelize the wrap-around write'

    B = np.arange(8, dtype=np.float64) + 0.5
    ref = np.zeros(8)
    for i in range(8):
        ref[(i + offset) % 8] = B[i] * 2.0
    got = np.zeros(8)
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref)


@dace.program
def _mod_read_wrap_minus1(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = (B[i] + B[(i - 1) % N]) * 0.5


@dace.program
def _mod_read_wrap_plus1(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = (B[i] + B[(i + 1) % N]) * 0.5


@pytest.mark.parametrize('prog,offset', [(_mod_read_wrap_minus1, -1), (_mod_read_wrap_plus1, 1)])
def test_loop_peeling_modulo_read_wraparound_knob_is_correct(prog, offset):
    """A wrap-around READ ``A[i] = (B[i] + B[(i + k) % N]) * 0.5`` (TSVC s291) already
    maps as-is -- the modulo is a pure read, no parallelization blocker -- but the
    surviving ``% N`` lowers to C's truncated ``%``, which reads the wrong element at
    the wrapping boundary (``(0 - 1) % N`` is ``-1`` in C, ``N - 1`` in Python): the
    no-knob result is WRONG. With ``peel_limit>0`` the wrapping boundary iteration is
    peeled off (front for a negative offset, back for a positive one) and the modulo
    folded to the plain affine index over both the remainder and the peeled iteration,
    so no C ``%`` survives and the result is floor-correct -- without changing the
    global modulo codegen. The correctness peel runs even though LoopToMap already
    maps the loop."""
    on = prog.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling + modulo fold must keep the wrap-around read mapped'

    B = np.arange(8, dtype=np.float64) + 0.5
    ref = np.zeros(8)
    for i in range(8):
        ref[i] = (B[i] + B[(i + offset) % 8]) * 0.5  # Python floor-mod oracle
    got = np.zeros(8)
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref)  # the OFF (no-knob) result is pre-existing-WRONG; only the peeled form is correct


@dace.program
def _interior_guard(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i == 4:
            A[0] = A[0] + 1.0


def test_index_set_split_interior_guard_knob_parallelizes():
    """An interior special-case iteration (``if i == 4: A[0] += 1``) writes a
    conflicting fixed location, so the loop is not parallel and -- being interior --
    no bounded boundary peel reaches it. With ``peel_limit>0`` the loop is index-set
    split into ``[0, 3] + {4} + [5, N-1]``; the guard is a contradiction in the two
    range segments (which map) and a tautology in the single middle iteration.
    Value-preserving."""
    off = _interior_guard.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'interior-guard loop must stay sequential without the knob'

    on = _interior_guard.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'index-set split must parallelize the range segments'

    A = np.arange(1, 11, dtype=np.float64)
    B = np.arange(10, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _interior_guard.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=10)
    got = A.copy()
    on(A=got, B=B.copy(), N=10)
    assert np.allclose(got, ref_A)


@dace.program
def _interior_guard_surrounded(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0  # body before the guard
        if i == 3:
            A[N - 1] = A[N - 1] + 5.0  # the special-iteration body
        A[i] = A[i] + 1.0  # body after the guard


def test_index_set_split_body_around_guard_knob_parallelizes():
    """The ``body1; if i == x: body2; body3`` shape: unconditional work surrounds an
    interior special-case guard. The index-set split puts ``body1; body3`` in the two
    range segments (guard-free, mapping) and ``body1; body2; body3`` in the single
    middle iteration. Value-preserving."""
    off = _interior_guard_surrounded.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'surrounded interior-guard loop must stay sequential off'

    on = _interior_guard_surrounded.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'index-set split must parallelize the surrounding work'

    A = np.arange(1, 11, dtype=np.float64)
    B = np.arange(10, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _interior_guard_surrounded.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=10)
    got = A.copy()
    on(A=got, B=B.copy(), N=10)
    assert np.allclose(got, ref_A)


@dace.program
def _front_range_guard(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        if i < 2:
            A[N - 1] = A[N - 1] + 1.0


def test_loop_peeling_front_range_guard_knob_parallelizes():
    """A *range* guard firing only on the first iterations (``if i < 2``) -- LLVM's
    canonical 'peel to eliminate the comparison' case (distinct from the ``i == x``
    equality split). With ``peel_limit>0`` the front iterations are peeled and the
    now-dead ``i < 2`` guard pruned (a range contradiction over the remainder), so
    the disjoint-write body maps. Value-preserving."""
    off = _front_range_guard.to_sdfg(simplify=True)
    canonicalize(off, validate=True, peel_limit=0, break_anti_dependence=False)
    assert _nmaps(off) == 0 and _nloops(off) == 1, 'range-guard loop must stay sequential without the knob'

    on = _front_range_guard.to_sdfg(simplify=True)
    canonicalize(on, validate=True, peel_limit=8)
    assert _nmaps(on) >= 1 and _nloops(on) == 0, 'peeling must prune the i<2 guard and map the remainder'

    A = np.arange(1, 9, dtype=np.float64)
    B = np.arange(8, dtype=np.float64) + 0.5
    ref_A = A.copy()
    _front_range_guard.to_sdfg(simplify=True)(A=ref_A, B=B.copy(), N=8)
    got = A.copy()
    on(A=got, B=B.copy(), N=8)
    assert np.allclose(got, ref_A)


if __name__ == '__main__':
    test_break_anti_dependence_knob_parallelizes()
    test_loop_peeling_front_range_guard_knob_parallelizes()
    test_loop_peeling_front_conflict_knob_parallelizes()
    test_loop_peeling_back_conflict_knob_parallelizes()
    test_loop_peeling_multi_front_iter_knob_parallelizes()
    test_loop_peeling_multi_front_else_iter_knob_parallelizes()
    test_loop_peeling_multi_back_iter_knob_parallelizes()
    test_loop_peeling_fixed_read_first_iter_knob_parallelizes()
    for _p, _o in [(_mod_wrap_plus1, 1), (_mod_wrap_minus1, -1), (_mod_wrap_plus3, 3)]:
        test_loop_peeling_modulo_wraparound_knob_parallelizes(_p, _o)
    for _p, _o in [(_mod_read_wrap_minus1, -1), (_mod_read_wrap_plus1, 1)]:
        test_loop_peeling_modulo_read_wraparound_knob_is_correct(_p, _o)
    test_index_set_split_interior_guard_knob_parallelizes()
    test_index_set_split_body_around_guard_knob_parallelizes()
