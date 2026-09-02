# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""E2e: induction-variable chains that free a loop to parallelize.

``s128`` is the headline case: a DERIVED-IV chain where the primary IV increment
``j := j + 2`` sits *between* two content blocks and a derived IV ``k := j + 1``
feeds the array gathers ``b[k]`` / ``c[k]``. Substituting the between-blocks
``j`` rewrites the derived iedge to ``k := 2 * i`` (affine), which symbol
propagation folds into the gathers, so ``LoopToMap`` parallelizes the whole loop.
The kernel is run to a numpy reference to confirm the substitution is correct,
not just that it parallelizes.

``s453`` and ``s122`` are the two shapes that need the closed form expanded at the
IV's USE SITES rather than the loop eliminated: ``s453``'s accumulator is READ by
the statement next to it (so it is neither eliminable nor fissionable), and
``s122``'s counter lives in a loop whose start AND stride are symbolic.
"""

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def _canonicalize_counts(name):
    kernel = [k for k in tsvc.collect() if k.name == name][0]
    sdfg = tsvc.to_sdfg(kernel, 'iv_' + name, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)

    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f"{name}: value mismatch on {n}"

    nloops = sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    nmaps = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    return nloops, nmaps, sdfg


def test_s128_derived_iv_chain_parallelizes():
    """s128: derived-IV chain (k := j+1 before j := j+2, between content blocks).
    Must reduce to affine k=2i and fully parallelize, value-preserving."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s128_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s128 (derived-IV chain) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s124_branch_uniform_iv_parallelizes():
    """s124: ``j += 1`` in BOTH branches of the conditional (branch-uniform IV).
    Hoisting the common increment out of the conditional lets IV substitution
    close it to ``j = i``, so ``a[j]`` becomes the parallel ``a[i]``."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s124_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s124 (branch-uniform IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s453_use_site_iv_parallelizes():
    """s453: ``s = s + 2.0; a[i] = s * b[i]`` -- the IV is a DATA accumulator that the other
    statement READS, so it is neither eliminable nor fissionable. Expanding its closed form at
    the use site (``s == s_entry + 2.0*(i+1)`` after this iteration's update) leaves a pure
    per-element body."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s453_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s453 (use-site data IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s122_symbolic_start_and_stride_iv_parallelizes():
    """s122: ``for i in range(n1-1, LEN_1D, n3): k = k + j; a[i] = a[i] + b[LEN_1D-k]`` -- BOTH
    the start and the stride are symbolic, so the counter's closed form needs the trip index
    ``t = int_floor(i - start, stride)`` rather than ``i - start``."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s122_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s122 (symbolic start/stride IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s318_staged_counter_iv_lifts_to_an_arg_reduction():
    """s318: ``k += inc``, where ``inc`` is a scalar ARGUMENT rather than a literal.

    The frontend cannot put that increment on an interstate edge -- the step is data -- so it emits
    a tasklet writing a transient scalar and binds ``k`` to that slot. Promoting the read-only
    argument to a symbol and reading the binding through the one staging hop closes ``k`` to
    ``inc * i``, which turns the guarded body into the strided gather ``a[inc*i]`` the arg-reduction
    lift already handles. Both halves are load-bearing: without the closure the loop keeps a
    per-iteration tasklet and the lift refuses the body outright.
    """
    from dace.libraries.standard.nodes import ArgReduce
    nloops, _nmaps, sdfg = _canonicalize_counts('s318_d_single')
    # A map count is the wrong proxy for "parallelized" here: the lift now folds the gather into
    # the ArgReduce's own strided _in memlet, so the correct answer has ZERO maps. What has to hold
    # is that no sequential loop survives and the strided argmax became one library node.
    assert nloops == 0, f"s318 (staged counter IV) should leave no sequential loop, got loops={nloops}"
    assert sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ArgReduce)) == 1, \
        "the strided argmax must lift to a single ArgReduce, not to a value-only reduction"


def test_s126_two_level_counter_closes_one_loop_at_a_time():
    """s126: ``k`` is stepped once per INNER iteration and once more per OUTER iteration.

    There is no single closed form for a step the loop does not own, so the inner loop is closed
    first -- its whole contribution becomes one assignment on the way out -- which leaves the outer
    loop with a single step per iteration for the next round of the fixed point. Refusing the
    counter outright (the old rule for "stepped in another loop too") left BOTH loops sequential.
    """
    nloops, nmaps, _sdfg = _canonicalize_counts('s126_d_single')
    assert nloops <= 1 and nmaps >= 2, \
        f"s126 (two-level counter) should leave at most the j recurrence, got loops={nloops} maps={nmaps}"


def test_s126_two_level_counter_leaves_no_k_and_an_affine_subset():
    """s126: after both levels close, ``k`` is gone and the gather is affine in (i, j).

    The closed form is ``k(i, j) = i*LEN_2D + j``, so the read ``flat_2d_array[k - 1]`` must land
    on ``i*LEN_2D + j - 1``. The row stride is ``LEN_2D`` and not ``LEN_2D - 1`` because of the
    trailing ``k += 1`` in the outer body, which skips one element per row -- getting that wrong
    still parallelizes, so the loop/map counts above cannot catch it and this subset can.

    The two indices are pinned by the ranges of the maps that ENCLOSE the read (``i`` from 0,
    ``j`` from 1), not by name, and the comparison is symbolic: the symbols are taken from the
    expression itself so both sides carry the same assumptions (two ``LEN_2D`` objects that
    differ only in assumptions do not cancel).
    """
    _nloops, _nmaps, sdfg = _canonicalize_counts('s126_d_single')

    for sd in sdfg.all_sdfgs_recursive():
        leftovers = ({'k'} & set(sd.symbols)) | ({'k'} & set(sd.arrays)) | ({'k'} & {str(s) for s in sd.free_symbols})
        assert not leftovers, f"the counter survived canonicalization in {sd.label} as {leftovers}"

    reads = [(state, e) for state in sdfg.all_states() for e in state.edges() if e.data is not None
             and not e.data.is_empty() and e.data.data == 'flat_2d_array' and e.data.subset.num_elements() == 1]
    assert len(reads) == 1, f"expected one single-element flat_2d_array read, got {len(reads)}"
    state, edge = reads[0]
    (expr, ) = edge.data.subset.min_element()

    scope = state.scope_dict()
    enclosing, node = {}, edge.dst
    while node is not None:
        node = scope[node]
        if isinstance(node, nodes.MapEntry):
            enclosing.update(zip(node.map.params, node.map.range))
    (outer, ) = [p for p, rng in enclosing.items() if symbolic.simplify(rng[0]) == 0]
    (inner, ) = [p for p, rng in enclosing.items() if symbolic.simplify(rng[0]) == 1]

    by_name = {str(s): s for s in expr.free_symbols}
    i, j, n = by_name[outer], by_name[inner], by_name['LEN_2D']
    assert symbolic.simplify(expr - (i * n + j - 1)) == 0, \
        f"s126 gather must be i*LEN_2D + j - 1, got {expr} (i={outer}, j={inner})"


def test_two_level_counter_exit_value_is_exact():
    """The value a two-level counter LEAVES BEHIND must survive the substitution exactly.

    Closing the inner loop folds its whole contribution into the enclosing loop's exit
    assignment rather than splicing a separate state (that is what lets the outer level close in
    turn). This is the composed arithmetic, so it needs its own check: everything else about
    s126 would still pass if the exit value were wrong, because s126 never reads ``k`` again.

    ``k`` starts at 1 and advances ``LEN`` per outer iteration (``LEN - 1`` inner steps plus the
    trailing one), so after ``LEN`` outer iterations it is ``1 + LEN*LEN``.
    """
    LEN = dace.symbol('LEN', dtype=dace.int64, positive=True)

    @dace.program
    def two_level_counter_exit(out: dace.int64[1], flat: dace.float64[LEN * LEN], src: dace.float64[LEN]):
        k = 1
        for i in range(LEN):
            for j in range(1, LEN):
                flat[k - 1] = src[j]
                k = k + 1
            k = k + 1
        out[0] = k

    sdfg = two_level_counter_exit.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)

    for n in (3, 5, 8):
        out = np.zeros(1, dtype=np.int64)
        flat = np.zeros(n * n)
        src = np.copy(np.arange(n, dtype=np.float64)) + 1.0
        sdfg(out=out, flat=flat, src=src, LEN=n)
        assert out[0] == 1 + n * n, f"post-loop counter must be 1 + LEN*LEN; LEN={n} gave {out[0]}"
        # and the gather the closed form drives must still have hit the right slots
        want = np.zeros(n * n)
        k = 1
        for i in range(n):
            for j in range(1, n):
                want[k - 1] = src[j]
                k += 1
            k += 1
        assert np.allclose(flat, want), f"LEN={n}: gather landed on the wrong elements"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
