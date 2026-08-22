# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strengthening test for
:class:`~dace.transformation.passes.canonicalize.loop_to_conditional_reduce.LoopToConditionalReduce`.

Former soundness hole: when the accumulated addend is a COMPUTED expression (not
a bare array element), ``_addend_gather`` traces the addend to the transient
holding the product, so the guard's ``a[i]`` could not be remapped to the
``__addend`` connector. The folded mask tasklet then read a BARE ``a[i]`` that
was not wired as an input -- the guard was always false and the reduction
silently returned 0.0.

The fix is a LIFT, not a refusal: the guard's own array read is WIRED as an extra
``__guardN`` input to the mask tasklet, and the mask is spliced into the branch's
own state (which carries the addend's producer subgraph with it). So the shape
below now LIFTS -- the ConditionalBlock is gone, the loop is left in the parallel
masked-reduction shape, and the numerics are bit-exact with the sequential
reference. Only a guard read that cannot be expressed as a memlet subset (an
indirection ``a[b[i]]``) is still refused.
"""
import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalBlock
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce

N = dace.symbol('N')


def _has_conditional_block(sdfg) -> bool:
    return any(
        isinstance(r, ConditionalBlock) for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions())


def _mask_tasklet(sdfg):
    """The spliced-in mask tasklet -- the one whose body is a ternary."""
    masks = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nd.Tasklet) and 'if ' in n.code.as_string and 'else' in n.code.as_string
    ]
    assert len(masks) == 1, f"expected exactly one mask tasklet, got {len(masks)}"
    return masks[0]


def test_guarded_sum_of_squares_lifts_and_is_bit_exact():
    """``if a[i] > 0: s += a[i]*a[i]`` -- the addend is a COMPUTED expression, so
    the guard's element cannot be routed to ``__addend``. The pass must LIFT this
    anyway by wiring the guard's ``a[i]`` as its own mask input, and the result
    must be BIT-EXACT with the sequential reference (the old bug returned 0.0).
    """

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if a[i] > 0.0:
                s = s + a[i] * a[i]
        b[0] = s

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res == 1, "computed-addend guarded reduction must LIFT, not refuse"
    sdfg.validate()

    # The guard is folded into the accumulated value: no ConditionalBlock left.
    assert not _has_conditional_block(sdfg)
    # The guard's array read is a REAL input of the mask tasklet, not an unbound
    # name -- this is exactly what the old miscompile got wrong.
    mask = _mask_tasklet(sdfg)
    assert '__addend' in mask.in_connectors
    guard_conns = sorted(c for c in mask.in_connectors if c.startswith('__guard'))
    assert guard_conns == ['__guard0'], f"expected one wired guard input, got {sorted(mask.in_connectors)}"
    assert '__guard0' in mask.code.as_string, "the folded cond must read the WIRED connector"

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    expected = float(np.sum(a[a > 0.0]**2))  # 1 + 9 + 25 = 35
    b = np.zeros(1)
    sdfg(a=a.copy(), b=b, N=a.size)
    assert b[0] == expected, f"got {b[0]}, expected {expected}"


def test_guarded_sum_of_squares_bit_exact_against_sequential_reference():
    """The lift is value-exact against an explicit SEQUENTIAL reference over a
    non-trivial input (mixed signs, exact zeros -- the boundary of the ``> 0``
    guard). Summation order is unchanged by the mask, so this is ``==``, never a
    tolerance."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if a[i] > 0.0:
                s = s + a[i] * a[i]
        b[0] = s

    sdfg = kernel.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    rng = np.random.default_rng(31110)
    a = rng.standard_normal(64)
    a[::7] = 0.0  # exact zeros: the guard must exclude them (``> 0``, not ``>= 0``)

    ref = 0.0  # sequential reference, same order as the loop
    for x in a:
        if x > 0.0:
            ref = ref + x * x

    b = np.zeros(1)
    sdfg(a=a.copy(), b=b, N=a.size)
    assert b[0] == ref, f"got {b[0]!r}, expected {ref!r}"


def test_guarded_product_of_computed_addend_lifts():
    """The same lift for a ``*`` reduction: the identity is ``1.0``, so a
    masked-out iteration leaves the product unchanged. Bit-exact vs sequential."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        p = 1.0
        for i in range(N):
            if a[i] > 0.0:
                p = p * (a[i] + a[i])
        b[0] = p

    sdfg = kernel.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert not _has_conditional_block(sdfg)

    a = np.array([1.5, -2.0, 3.25, -4.0, 0.5])
    ref = 1.0
    for x in a:
        if x > 0.0:
            ref = ref * (x + x)

    b = np.zeros(1)
    sdfg(a=a.copy(), b=b, N=a.size)
    assert b[0] == ref, f"got {b[0]!r}, expected {ref!r}"


def test_guard_reads_a_different_element_than_the_addend_lifts():
    """The guard reads ``b[i]`` while the addend is ``a[i]*a[i]`` -- two distinct
    arrays. Both are wireable, so this lifts with the guard read as its own mask
    input."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if b[i] > 0.0:
                s = s + a[i] * a[i]
        out[0] = s

    sdfg = kernel.to_sdfg(simplify=True)
    assert LoopToConditionalReduce().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert not _has_conditional_block(sdfg)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    ref = 0.0
    for i in range(a.size):
        if b[i] > 0.0:
            ref = ref + a[i] * a[i]

    out = np.zeros(1)
    sdfg(a=a.copy(), b=b.copy(), out=out, N=a.size)
    assert out[0] == ref, f"got {out[0]!r}, expected {ref!r}"


def test_lifts_through_the_full_canonicalize_pipeline():
    """The lift must fire where it MATTERS -- inside the canonicalize pipeline,
    not just when the pass is driven standalone. Mid-pipeline the frontend's
    ``i`` has been renamed to a region-scoped ``_loop_it_0`` that is bound by the
    LoopRegion and is absent from ``sdfg.symbols``; a wireability check keyed
    only on ``sdfg.symbols`` silently refuses every guard read there, so the
    guarded atomic survives to codegen. Pin the end state: guard folded away, the
    reduction lowered WITHOUT a per-passing-thread ``reduce_atomic``, bit-exact.
    """
    from dace.transformation.passes.canonicalize.pipeline import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    @dace.program
    def sumsq(a: dace.float64[N], out: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if a[i] > 0.0:
                s = s + a[i] * a[i]
        out[0] = s

    sdfg = sumsq.to_sdfg(simplify=True)
    canonicalize(sdfg)
    assert not _has_conditional_block(sdfg), "the guard must be folded by the pipeline, not left as control flow"
    # The loop is gone -- the body is a parallel map.
    assert not any(isinstance(r, LoopRegion) and r.loop_variable for r in sdfg.all_control_flow_regions())
    assert any(isinstance(n, nd.MapEntry) for n, _ in sdfg.all_nodes_recursive())

    finalize_for_target(sdfg, 'cpu')
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert 'reduce_atomic' not in code, "the guarded per-thread atomic must be gone -- that is the point of the lift"

    # The lifted reduction is a TREE reduction, so it may sum in a different order
    # than the sequential reference. Use integer-valued float64 inputs: every summand
    # and every partial sum is exactly representable, so the reference holds
    # BIT-EXACTLY under any reduction order -- no tolerance to hide a discrepancy.
    rng = np.random.default_rng(981)
    a = rng.integers(-32, 32, size=256).astype(np.float64)
    ref = 0.0
    for x in a:
        if x > 0.0:
            ref = ref + x * x

    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=a.size)
    assert out[0] == ref, f"got {out[0]!r}, expected {ref!r}"


def _build_iedge_symbol_guard_sdfg() -> dace.SDFG:
    """Hand-build the pre-canonicalize IR for::

        s = 0.0
        for i in range(N - 1):
            if a[i + off] > 0.0:
                s = s + a[i] * a[i]
        out[0] = s

    where ``off`` is bound ONLY by an interstate-edge assignment on the edge into the
    loop -- so it is a genuinely-defined symbol at the guard that is absent from
    ``sdfg.symbols``. The python frontend cannot express this shape (it registers its
    gather symbols globally), but the Fortran frontend and hand-built IR do.
    """
    sdfg = dace.SDFG('iedge_symbol_guard')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    sdfg.add_scalar('prod', dace.float64, transient=True)

    init = sdfg.add_state('init', is_start_block=True)
    zero = init.add_tasklet('zero', {}, {'__out'}, '__out = 0.0')
    init.add_edge(zero, '__out', init.add_write('s'), None, dace.Memlet('s[0]'))

    loop = LoopRegion('L', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'off': '1'}))

    cb = ConditionalBlock('cb')
    loop.add_node(cb, is_start_block=True)
    branch = ControlFlowRegion('br', sdfg=sdfg)
    cb.add_branch(CodeBlock('a[i + off] > 0.0'), branch)

    body = branch.add_state('body', is_start_block=True)
    mul = body.add_tasklet('mul', {'__x'}, {'__p'}, '__p = __x * __x')
    body.add_edge(body.add_read('a'), None, mul, '__x', dace.Memlet('a[i]'))
    prod_an = body.add_access('prod')
    body.add_edge(mul, '__p', prod_an, None, dace.Memlet('prod[0]'))
    upd = body.add_tasklet('upd', {'__lhs', '__rhs'}, {'__out'}, '__out = (__lhs + __rhs)')
    body.add_edge(body.add_read('s'), None, upd, '__lhs', dace.Memlet('s[0]'))
    body.add_edge(prod_an, None, upd, '__rhs', dace.Memlet('prod[0]'))
    body.add_edge(upd, '__out', body.add_write('s'), None, dace.Memlet('s[0]'))

    fin = sdfg.add_state('fin')
    sdfg.add_edge(loop, fin, dace.InterstateEdge())
    store = fin.add_tasklet('store', {'__i'}, {'__o'}, '__o = __i')
    fin.add_edge(fin.add_read('s'), None, store, '__i', dace.Memlet('s[0]'))
    fin.add_edge(store, '__o', fin.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def test_guard_symbol_bound_by_an_interstate_edge_lifts():
    """The wireability gate must ask for the symbols DEFINED at the update tasklet,
    not for membership in ``sdfg.symbols``. ``sdfg.symbols`` holds only the SDFG's
    GLOBAL symbols, so it sees none of the binders a guard index is written in -- a
    region-scoped LoopRegion iterator, a map parameter, or (here) an interstate-edge
    assignment. ``off`` below is bound on the edge into the loop and is genuinely
    defined at the guard, yet is absent from ``sdfg.symbols``: the old gate rejected
    ``a[i + off]`` as unwireable and refused the whole lift, leaving the guarded
    per-passing-thread atomic in place.
    """
    sdfg = _build_iedge_symbol_guard_sdfg()
    assert 'off' not in sdfg.symbols, "premise: the guard's symbol is NOT a global SDFG symbol"

    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res == 1, "an interstate-edge-bound guard symbol is defined at the guard -- must LIFT, not refuse"
    sdfg.validate()
    assert not _has_conditional_block(sdfg)

    # ``a[i + off]`` is wired as a real mask input, with the iedge symbol kept in the subset.
    mask = _mask_tasklet(sdfg)
    assert '__guard0' in mask.in_connectors, f"expected the guard read wired, got {sorted(mask.in_connectors)}"
    guard_edges = [e for st in sdfg.states() for e in st.edges() if e.dst is mask and e.dst_conn == '__guard0']
    assert len(guard_edges) == 1
    assert 'off' in str(guard_edges[0].data.subset), f"iedge symbol lost from the subset: {guard_edges[0].data}"

    # Integer-valued float64 inputs: every summand and partial sum is exactly
    # representable, so the reference holds bit-exactly under ANY reduction order.
    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 6.0])
    ref = 0.0
    for i in range(a.size - 1):
        if a[i + 1] > 0.0:
            ref = ref + a[i] * a[i]

    out = np.zeros(1)
    sdfg(a=a.copy(), out=out, N=a.size)
    assert out[0] == ref, f"got {out[0]!r}, expected {ref!r}"


def test_refuses_indirect_guard_read():
    """The one genuinely-unliftable shape: the guard's index is itself read from
    memory (``a[idx[i]]``). A memlet subset is a static symbolic range, so the
    element the mask must read is not knowable without materialising the
    indirection -- which this pass does not do. Refuse and leave the loop for the
    gather passes; numerics stay bit-exact via the untouched sequential loop.
    """

    @dace.program
    def kernel(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[1]):
        s = 0.0
        for i in range(N):
            if a[idx[i]] > 0.0:
                s = s + a[i] * a[i]
        b[0] = s

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None, "indirect guard read is not expressible as a memlet subset -- must refuse"
    sdfg.validate()

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    idx = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    ref = 0.0
    for i in range(a.size):
        if a[idx[i]] > 0.0:
            ref = ref + a[i] * a[i]

    b = np.zeros(1)
    sdfg(a=a.copy(), idx=idx.copy(), b=b, N=a.size)
    assert b[0] == ref, f"got {b[0]!r}, expected {ref!r}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
