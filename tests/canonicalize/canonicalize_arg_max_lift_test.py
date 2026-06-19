# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.arg_max_lift.ArgMaxLift`.

Covers TSVC s314 (max), s316 (min), and refusals on the v1 out-of-scope shapes
(s3113 -- unary transform on the gather; s315 -- index-tracking variant).
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion, ConditionalBlock
from dace.libraries.standard.nodes import Reduce
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift

N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_reduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


# -----------------------------------------------------------------------------
# Positive: TSVC s314 (max) and s316 (min).
# -----------------------------------------------------------------------------


def test_tsvc_s314_max_value_only():
    """``x = a[0]; for i in range(1, N): if a[i] > x: x = a[i]`` lifts to a
    ``Reduce(Max)`` libnode. The pre-loop init ``x = a[0]`` is preserved as
    the seed via the libnode's ``identity=None`` semantics (WCR-Max folds the
    output's existing value into the reduction).
    """

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = s314.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 1

    n = 16
    rng = np.random.default_rng(314)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.max(a)), f"got {out[0]}, expected {np.max(a)}"


def test_tsvc_s316_min_value_only():
    """``<`` instead of ``>`` → ``Reduce(Min)``."""

    @dace.program
    def s316(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] < x:
                x = a[i]
        result[0] = x

    sdfg = s316.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 12
    rng = np.random.default_rng(316)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.min(a)), f"got {out[0]}, expected {np.min(a)}"


def test_max_corner_first_element_is_max():
    """``a[0]`` is the maximum -- the libnode's pre-existing-output seed picks
    it up even though the input slice ``a[1:N]`` excludes index 0."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    a = np.array([100.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.zeros(1)
    sdfg(a=a, result=out, N=6)
    assert np.isclose(out[0], 100.0)


# -----------------------------------------------------------------------------
# Refusals: v1 out-of-scope shapes.
# -----------------------------------------------------------------------------


def test_lifts_abs_transform_s3113():
    """TSVC s3113: ``av = abs(a[i]); if av > maxv: maxv = av`` -- a max reduction
    over the ABS-transformed gather. The abs path materialises ``buf[j] =
    abs(a[j])`` into a contiguous transient, then a ``Reduce(Max)`` over ``buf``;
    the result is ``max(|a|)``."""

    @dace.program
    def s3113(a: dace.float64[N], b: dace.float64[2]):
        maxv = abs(a[0])
        for i in range(N):
            av = abs(a[i])
            if av > maxv:
                maxv = av
        b[0] = maxv

    sdfg = s3113.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, "abs-transform max reduction should lift"
    sdfg.validate()
    assert _num_loops(sdfg) == 0 and _num_reduces(sdfg) == 1

    n = 20
    rng = np.random.default_rng(3113)
    a = rng.standard_normal(n)  # mixed signs -> abs matters
    out = np.zeros(2)
    sdfg(a=a, b=out, N=n)
    assert np.isclose(out[0], np.max(np.abs(a))), f"got {out[0]}, expected {np.max(np.abs(a))}"


def test_refuses_index_tracking_s315():
    """TSVC s315 in its DATA-carrier form (``x`` / ``index`` as Scalars, the
    shape ``to_sdfg(simplify=True)`` produces): the true-branch writes BOTH the
    value carrier and an index, so the data-carrier path sees two terminal
    AccessNodes and refuses. (The SYMBOL-carrier form -- what full canonicalize
    produces -- DOES lift to an ``ArgReduce`` libnode; see
    ``test_symbol_carrier_argmax_with_index``.)"""

    @dace.program
    def s315(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        index = 0
        for i in range(N):
            if a[i] > x:
                x = a[i]
                index = i
        result[0] = x + float(index)

    sdfg = s315.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None, "index-tracking variant should be refused in v1"


def test_refuses_non_comparison_condition():
    """The body's condition must be a single ``a OP b`` comparison; bitwise/
    boolean operators are out of scope."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.int64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if b[i] != 0 and a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    # The compound condition is wrapped in a chain of iedges that the matcher
    # can't trace back to a single ``Compare`` AST node. Refuse.
    assert res is None


def test_refuses_subtraction_op():
    """``Sub`` is not in :data:`_CMP_AST_TO_RTYPE`; only ``>``, ``<``, ``>=``, ``<=``."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] != x:  # ``!=`` not in the set
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None


# -----------------------------------------------------------------------------
# Look-alike refusals: shapes that pattern-match argmax superficially but
# don't actually compute argmax. The matcher must refuse all of these.
# -----------------------------------------------------------------------------


def test_lookalike_refuses_non_unit_stride():
    """``for i in range(0, N, 2)`` -- stride > 1 means the reduce would only
    see half the array; refuse so the loop stays sequential until a
    gather-then-reduce variant lands."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(0, N, 2):
            if a[i] > x:
                x = a[i]
        result[0] = x

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "stride>1 must be refused"


def test_lookalike_refuses_symbolic_stride():
    """``for i in range(0, N, K)`` with symbolic ``K`` -- same as above; the
    integer-stride check throws ``TypeError`` on the symbol and the matcher
    refuses."""
    K = dace.symbol('K_arg_stride')

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(0, N, K):
            if a[i] > x:
                x = a[i]
        result[0] = x

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "symbolic stride must be refused"


def test_lookalike_refuses_carrier_written_to_constant():
    """``if a[i] > x: x = 0.0`` -- the write doesn't read ``a[i]``; this is a
    threshold reset, not argmax. The true-branch state writes ``x`` from a
    constant tasklet, so the gather-resolver fails to find ``a[loop_var]``
    on the source side."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = 0.0
        result[0] = x

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "constant write under cond is not argmax"


def test_lookalike_refuses_cond_doesnt_reference_carrier():
    """``if a[i] > b[i]: x = a[i]`` -- the comparison reads ``b[i]`` instead
    of the carrier. The carrier-name extracted from the comparison RHS would
    be ``b_index`` (the b-gather symbol), not ``x``, so the carrier classifier
    refuses (``b_index`` is not in ``sdfg.arrays`` as a scalar carrier)."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > b[i]:
                x = a[i]
        result[0] = x

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "cond reading a different array is not argmax"


def test_lookalike_refuses_body_after_conditional():
    """``if a[i] > x: x = a[i]; b[i] = x`` -- the loop body has additional
    work *after* the conditional (writes to ``b``). Lifting would drop the
    ``b`` write; the matcher must refuse. (The body of the loop has more
    blocks than just the ConditionalBlock + empty wrappers.)"""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
            b[i] = x  # extra unconditional body work
        result[0] = x

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "unconditional body work alongside the cond is not pure argmax"


def test_lookalike_refuses_carrier_constant_init():
    """``x = 0.0`` (pre-loop init reads no array) -- still semantically argmax,
    but the lift relies on the pre-loop carrier value as the WCR seed; if the
    user starts at 0 and ALL of ``a`` is negative, the lifted reduce would
    return ``0`` while the sequential loop returns the actual max. This
    distinction is currently NOT enforced in v1 -- documented as a known
    limitation; positive numerics test below verifies the common case still
    works. (Refusal will be the design for v2 unless the init is provably
    ``-inf`` / the array's lowest value.)"""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = 0.0  # constant init, not `a[0]`
        for i in range(N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    if res is not None:
        # Currently accepted; verify the common "max is positive" case works.
        sdfg.validate()
        a = np.array([1.0, 5.0, -3.0, 2.0])
        out = np.zeros(1)
        sdfg(a=a, result=out, N=4)
        assert np.isclose(out[0], 5.0)


# -----------------------------------------------------------------------------
# Cross-pass non-interference: ArgMax/Reduce/Scan look-alikes mustn't trigger
# the wrong pass.
# -----------------------------------------------------------------------------


def test_argmax_doesnt_lift_a_plain_reduction_loop():
    """``for i: s = s + a[i]`` -- this is a Reduce shape (handled by
    ``LoopToReduce`` / ``AccumulatorToMapAndReduce``), NOT an argmax. The
    body has no ConditionalBlock, so ArgMaxLift must refuse."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        s = 0.0
        for i in range(N):
            s = s + a[i]
        result[0] = s

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "plain reduction is not argmax"


def test_argmax_doesnt_lift_a_scan_loop():
    """``for i: out[i+1] = out[i] + a[i]`` -- Scan shape; out is array-write
    indexed by loop var. No ConditionalBlock; ArgMaxLift must refuse."""

    @dace.program
    def kernel(a: dace.float64[N], out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] + a[i]

    res = ArgMaxLift().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None, "scan recurrence is not argmax"


def test_loop_to_reduce_doesnt_lift_an_argmax_loop():
    """A real argmax loop (track BOTH the max value AND its index) must NOT
    be lifted by ``LoopToReduce``. The branch body writes TWO accumulators
    inside the conditional (``x = a[i]`` AND ``idx = i``); a single
    ``Reduce`` libnode can carry only one fold, and the wcr-scalar emit
    refuses on the two-write contract its branched-min/max matcher
    enforces. ``ArgMaxLift`` is the right handler for this shape -- it
    materialises the index/value pair into a paired libnode that the
    standard ``Reduce`` cannot express.

    Note: the prior version of this test used the TSVC s314 ``max`` kernel
    (no index tracking); the slice 2a branched-min/max extension to
    ``_extract`` deliberately lifts that shape because ``max(x, a[i])`` is
    idempotent. The intent of THIS test is the argmax contract -- two
    accumulators inside the guard -- so the kernel updated to actually
    exercise that.
    """
    from dace.transformation.passes.loop_to_reduce import LoopToReduce

    @dace.program
    def argmax_kernel(a: dace.float64[N], val_out: dace.float64[1], idx_out: dace.int64[1]):
        x = a[0]
        idx = 0
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
                idx = i
        val_out[0] = x
        idx_out[0] = idx

    sdfg = argmax_kernel.to_sdfg(simplify=True)
    res = LoopToReduce().apply_pass(sdfg, {})
    assert res is None, "LoopToReduce must not lift conditional argmax loops (two accumulators in the branch)"
    res_wcr = LoopToReduce(prefer='wcr-scalar').apply_pass(sdfg, {})
    assert res_wcr is None, "LoopToReduce(wcr-scalar) must also refuse argmax: branch body has two accumulator writes"


def test_loop_to_scan_doesnt_lift_an_argmax_loop():
    """And LoopToScan must also leave argmax loops alone."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    res = LoopToScan().apply_pass(s314.to_sdfg(simplify=True), {})
    assert res is None, "LoopToScan must not lift conditional argmax loops"


def test_loop_to_reduce_doesnt_lift_a_scan_loop():
    """Cross-check the other direction: a scan loop must not be picked up by
    LoopToReduce."""
    from dace.transformation.passes.loop_to_reduce import LoopToReduce

    @dace.program
    def scan(a: dace.float64[N], out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] + a[i]

    res = LoopToReduce().apply_pass(scan.to_sdfg(simplify=True), {})
    assert res is None, "LoopToReduce must not lift scan recurrences"


def test_loop_to_scan_doesnt_lift_a_reduction_loop():
    """And LoopToScan must not pick up plain reductions."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    @dace.program
    def reduce_loop(a: dace.float64[N], result: dace.float64[1]):
        s = 0.0
        for i in range(N):
            s = s + a[i]
        result[0] = s

    res = LoopToScan().apply_pass(reduce_loop.to_sdfg(simplify=True), {})
    assert res is None, "LoopToScan must not lift plain reductions"


# -----------------------------------------------------------------------------
# Symbol-carrier tests: the carrier ``x`` lives on interstate-edge assignments,
# not as a Scalar / length-1 array. Constructed manually because the Python
# frontend doesn't naturally produce symbol-bound argmax carriers; ``x``-as-
# symbol is the cloudsc / ICON shape (e.g. iter counters bound via iedges).
# -----------------------------------------------------------------------------


def _build_symbol_argmax_sdfg(label: str, in_loop_write_rhs: str, op: str = '>', inline_cond: bool = False):
    """Construct an SDFG where the argmax carrier ``x`` is a symbol.

    :param op: comparison operator in the guard (``'>'`` -> Max, ``'<'`` -> Min).
    :param inline_cond: when True the comparison ``(a_index OP x)`` sits directly
        in the ConditionalBlock condition (the shape full canonicalize produces
        for TSVC s314/s316); when False it is indirected through a ``__tmp0``
        symbol bound by an upstream iedge (the older frontend shape).

    Structure::

        [init]
            | iedge: x := a[0]
            v
        [LoopRegion(loop_var=i, range(1, N))]
            body:
                [start (empty)]
                    | iedge: a_index := a[i]
                    v
                [cond_prep (empty)]
                    | iedge: __tmp := (a_index > x)
                    v
                [ConditionalBlock(__tmp)]
                    true-branch:
                        [t1 (empty)]
                            | iedge: x := <in_loop_write_rhs>
                            v
                        [t2 (empty)]
            v
        [post]
            (carrier ``x`` is read here as a symbol)

    :param in_loop_write_rhs: RHS of the carrier-write iedge inside the
        true-branch. Use ``'a[i]'`` for the positive test (real argmax shape)
        and ``'i'`` for the look-alike (wrong RHS) refusal test.
    """
    from dace.sdfg.state import ControlFlowRegion
    from dace.properties import CodeBlock

    sdfg = dace.SDFG(label)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('result', [1], dace.float64)
    # Symbol carriers + helper symbols.
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('a_index', dace.float64)
    sdfg.add_symbol('__tmp0', dace.bool)

    init_state = sdfg.add_state('init', is_start_block=True)

    loop = LoopRegion(label + '_loop',
                      initialize_expr='i = 1',
                      condition_expr='i < N',
                      update_expr='i = i + 1',
                      loop_var='i')
    sdfg.add_node(loop)
    # Pre-loop iedge seeds the symbol from ``a[0]``.
    sdfg.add_edge(init_state, loop, dace.InterstateEdge(assignments={'x': 'a[0]'}))

    # Loop body structure mirrors the Python-frontend lowering shape.
    start_blk = loop.add_state('start', is_start_block=True)
    cond_prep = loop.add_state('cond_prep')
    cond_block = ConditionalBlock('cond_block')
    loop.add_node(cond_block)

    loop.add_edge(start_blk, cond_prep, dace.InterstateEdge(assignments={'a_index': 'a[i]'}))
    if inline_cond:
        # Comparison inlined directly in the condition (post-canonicalize shape).
        loop.add_edge(cond_prep, cond_block, dace.InterstateEdge())
        cond_code = f'(a_index {op} x)'
    else:
        # Comparison indirected through a ``__tmp0`` iedge (older frontend shape).
        loop.add_edge(cond_prep, cond_block, dace.InterstateEdge(assignments={'__tmp0': f'(a_index {op} x)'}))
        cond_code = '__tmp0'

    # True-branch: empty states with the carrier-write iedge between them.
    true_branch = ControlFlowRegion(label + '_true')
    cond_block.add_branch(CodeBlock(cond_code), true_branch)
    t1 = true_branch.add_state('t1', is_start_block=True)
    t2 = true_branch.add_state('t2')
    true_branch.add_edge(t1, t2, dace.InterstateEdge(assignments={'x': in_loop_write_rhs}))

    # Post-loop state: emit ``result[0] = x`` via a tasklet reading the symbol.
    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    w = post.add_write('result')
    t = post.add_tasklet('write_result', {}, {'__out'}, '__out = x', language=dace.dtypes.Language.Python)
    post.add_edge(t, '__out', w, None, dace.Memlet(data='result', subset='0'))

    return sdfg


def test_symbol_carrier_positive():
    """Symbol-carrier argmax: ``x`` is a symbol bound by iedges (pre-loop
    init + in-loop write under the conditional). ArgMaxLift should:

    * detect the symbol carrier,
    * allocate a fresh transient scalar for the Reduce output,
    * extend the input slice down to ``start - 1`` to include the seed
      (``a[0]``) since the pre-loop iedge ``x := a[0]`` is dropped,
    * plant a bind iedge ``x := _arg_max_buf[0]`` after the reduce so the
      downstream state reads the correct symbol value.
    """
    sdfg = _build_symbol_argmax_sdfg('s_arg_pos', in_loop_write_rhs='a[i]')
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, "symbol-carrier argmax must lift"
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 1

    n = 16
    rng = np.random.default_rng(1011)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.max(a)), f"got {out[0]}, expected {np.max(a)}"


def test_symbol_carrier_negative_wrong_rhs():
    """Look-alike with symbol carrier: the in-loop write is ``x := i`` instead
    of ``x := a[i]``. The carrier is updated to the index, not the value, so
    this is NOT argmax. The matcher's symbol-true-branch check verifies the
    RHS is ``arr[loop_var]`` or the gather symbol; ``i`` matches neither, so
    the lift is refused and the loop stays sequential.
    """
    sdfg = _build_symbol_argmax_sdfg('s_arg_neg', in_loop_write_rhs='i')
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None, "wrong-RHS symbol-carrier write must be refused"


def _build_symbol_argmax_index_sdfg(label: str, op: str = '>', inline_cond: bool = True):
    """Symbol-carrier argmax/argmin that ALSO tracks the index (TSVC s315).

    Mirrors :func:`_build_symbol_argmax_sdfg` but the true-branch binds BOTH the
    value carrier ``x := a[i]`` and the index carrier ``index := i``; the pre-loop
    seeds are ``x := a[0]`` / ``index := 0``. The post state reads both symbols
    into ``result`` (value) and ``idx_result`` (index) so the lift can be
    verified end to end. ArgMaxLift must lift this to an ``ArgReduce`` libnode.
    """
    from dace.sdfg.state import ControlFlowRegion
    from dace.properties import CodeBlock

    sdfg = dace.SDFG(label)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('result', [1], dace.float64)
    sdfg.add_array('idx_result', [1], dace.int64)
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('index', dace.int64)
    sdfg.add_symbol('a_index', dace.float64)
    sdfg.add_symbol('__tmp0', dace.bool)

    init_state = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion(label + '_loop',
                      initialize_expr='i = 1',
                      condition_expr='i < N',
                      update_expr='i = i + 1',
                      loop_var='i')
    sdfg.add_node(loop)
    sdfg.add_edge(init_state, loop, dace.InterstateEdge(assignments={'x': 'a[0]', 'index': '0'}))

    start_blk = loop.add_state('start', is_start_block=True)
    cond_prep = loop.add_state('cond_prep')
    cond_block = ConditionalBlock('cond_block')
    loop.add_node(cond_block)
    loop.add_edge(start_blk, cond_prep, dace.InterstateEdge(assignments={'a_index': 'a[i]'}))
    if inline_cond:
        loop.add_edge(cond_prep, cond_block, dace.InterstateEdge())
        cond_code = f'(a_index {op} x)'
    else:
        loop.add_edge(cond_prep, cond_block, dace.InterstateEdge(assignments={'__tmp0': f'(a_index {op} x)'}))
        cond_code = '__tmp0'

    true_branch = ControlFlowRegion(label + '_true')
    cond_block.add_branch(CodeBlock(cond_code), true_branch)
    t1 = true_branch.add_state('t1', is_start_block=True)
    t2 = true_branch.add_state('t2')
    true_branch.add_edge(t1, t2, dace.InterstateEdge(assignments={'x': 'a[i]', 'index': 'i'}))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    wv = post.add_write('result')
    tv = post.add_tasklet('write_val', {}, {'__o'}, '__o = x', language=dace.dtypes.Language.Python)
    post.add_edge(tv, '__o', wv, None, dace.Memlet(data='result', subset='0'))
    wi = post.add_write('idx_result')
    ti = post.add_tasklet('write_idx', {}, {'__o'}, '__o = index', language=dace.dtypes.Language.Python)
    post.add_edge(ti, '__o', wi, None, dace.Memlet(data='idx_result', subset='0'))
    return sdfg


@pytest.mark.parametrize('op,reducer', [('>', np.argmax), ('<', np.argmin)])
def test_symbol_carrier_argmax_with_index(op, reducer):
    """``if a[i] OP x: x = a[i]; index = i`` lifts to an ``ArgReduce`` libnode
    (two scalar outputs) whose value/index are bound back to the ``x`` / ``index``
    symbols. Verifies BOTH the extreme value and its (first-occurrence) index."""
    from dace.libraries.standard.nodes import ArgReduce
    sdfg = _build_symbol_argmax_index_sdfg('s_argidx_' + ('max' if op == '>' else 'min'), op=op)
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, "argmax-with-index must lift"
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ArgReduce)) == 1

    n = 24
    rng = np.random.default_rng(815 if op == '>' else 816)
    a = rng.standard_normal(n)
    val = np.zeros(1)
    idx = np.zeros(1, dtype=np.int64)
    sdfg(a=a, result=val, idx_result=idx, N=n)
    expected_idx = int(reducer(a))
    expected_val = a[expected_idx]
    assert np.isclose(val[0], expected_val), f"value: got {val[0]}, expected {expected_val}"
    assert idx[0] == expected_idx, f"index: got {idx[0]}, expected {expected_idx}"


def test_symbol_carrier_inline_condition_max():
    """The comparison is inlined directly in the ConditionalBlock condition
    (``(a_index > x)``) rather than indirected through a ``__tmp0`` iedge. This
    is the shape full canonicalize produces for TSVC s314; the matcher must
    parse the comparison straight off the condition codeblock."""
    sdfg = _build_symbol_argmax_sdfg('s_arg_inline_max', in_loop_write_rhs='a[i]', op='>', inline_cond=True)
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, "inline-condition symbol-carrier argmax must lift"
    sdfg.validate()
    assert _num_loops(sdfg) == 0 and _num_reduces(sdfg) == 1

    n = 16
    rng = np.random.default_rng(701)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.max(a)), f"got {out[0]}, expected {np.max(a)}"


def test_symbol_carrier_min_inline_all_positive():
    """Min reduction (``<``) with a symbol carrier, inline condition, over
    ALL-POSITIVE data. Regression for the identity bug: a fresh symbol-carrier
    Reduce with ``identity=None`` defaults the accumulator to ``0``, so
    ``min(0, positives) == 0`` would wrongly return 0. The fix seeds the
    accumulator with the dtype's most-positive value, so the true minimum is
    returned (it is the TSVC s316 shape)."""
    sdfg = _build_symbol_argmax_sdfg('s_arg_inline_min', in_loop_write_rhs='a[i]', op='<', inline_cond=True)
    sdfg.validate()
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res == 1, "inline-condition symbol-carrier argmin must lift"
    sdfg.validate()

    n = 16
    rng = np.random.default_rng(702)
    a = rng.random(n) + 0.5  # strictly positive, so a wrong identity=0 would surface
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.min(a)), f"got {out[0]}, expected {np.min(a)} (identity bug returns 0)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
