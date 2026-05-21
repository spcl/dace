# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Hard / adversarial unit tests for the :class:`SymbolPropagation` pass.

The pass propagates symbols that were assigned to a single value forward through the
SDFG, substituting them into downstream blocks and edges to reduce the symbol count.
Its load-bearing assumption is that symbols only change on ``InterStateEdge``
assignments. These tests stress that assumption with the patterns that tend to break
big real-world SDFGs:

* chained inter-dependent symbols feeding array indices,
* conditional (branch-divergent) symbol values feeding indirection,
* interstate-edge conditions that themselves read a propagated symbol,
* loop-carried index symbols (in both ``LoopRegion`` loops and ``dace.map`` maps),
* double indirection / gather (a symbol read out of an array, then used as an index),
* the same symbol name reused with different values in sibling scopes.

Every test builds a small deterministic SDFG, computes the reference result *before*
the pass, applies :meth:`SymbolPropagation.apply_pass`, asserts the SDFG is still valid,
re-runs it, and checks the results match. A test that exposes a genuine pass limitation
is marked :func:`pytest.mark.xfail` with a precise reason.
"""

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.transformation.passes import SymbolPropagation

# ---------------------------------------------------------------------------
# Python-frontend kernels (must be module-level: the frontend reads source).
# ---------------------------------------------------------------------------


@dace.program
def chained_index_range(B: dace.float64[64], C: dace.float64[1], idx: dace.int64):
    """
    Chained inter-dependent index symbols feeding an array access (range form).

    ``idx2 = idx + 1; idx3 = idx2 + 1; C = B[idx3]`` expressed so the symbols
    chain across interstate edges.

    :param B: Source array to gather from.
    :param C: One-element output.
    :param idx: Base index symbol.
    """
    idx2 = idx + 1
    idx3 = idx2 + 1
    C[0] = B[idx3]


@dace.program
def chained_index_deep(B: dace.float64[64], C: dace.float64[1], idx: dace.int64):
    """
    Longer chain of inter-dependent index symbols feeding an array access.

    :param B: Source array to gather from.
    :param C: One-element output.
    :param idx: Base index symbol.
    """
    a = idx + 1
    b = a + 2
    c = b - 1
    d = c + a
    C[0] = B[d]


@dace.program
def cond_index_diverge(A: dace.int64[1], B: dace.float64[64], C: dace.float64[1], idx: dace.int64):
    """
    Conditional symbol assignment feeding indirection (the divergent case).

    ``idx3`` takes a different value on each branch, so no single value may be
    propagated past the join point.

    :param A: One-element selector array (branch taken on ``A[0] > 0``).
    :param B: Source array to gather from.
    :param C: One-element output.
    :param idx: Base index symbol.
    """
    idx2 = idx + 1
    if A[0] > 0:
        idx3 = idx2 + 1
        C[0] = B[idx3]
    else:
        idx3 = idx2 + 4
        C[0] = B[idx3]


@dace.program
def cond_index_diverge_join(A: dace.int64[1], B: dace.float64[64], C: dace.float64[1], idx: dace.int64):
    """
    Branch-divergent symbol that is *used after* the join, not inside the branches.

    The use of ``idx3`` happens after the if/else merge, so the pass must not
    have propagated either branch's value.

    :param A: One-element selector array.
    :param B: Source array to gather from.
    :param C: One-element output.
    :param idx: Base index symbol.
    """
    idx2 = idx + 1
    if A[0] > 0:
        idx3 = idx2 + 1
    else:
        idx3 = idx2 + 4
    C[0] = B[idx3]


@dace.program
def nested_cond_index(A: dace.int64[1], D: dace.int64[1], B: dace.float64[64], C: dace.float64[1], idx: dace.int64):
    """
    Nested conditionals (``if A: if D: ...``) each diverging an index symbol.

    :param A: Outer selector array.
    :param D: Inner selector array.
    :param B: Source array to gather from.
    :param C: One-element output.
    :param idx: Base index symbol.
    """
    idx2 = idx + 1
    if A[0] > 0:
        if D[0] > 0:
            idx3 = idx2 + 1
        else:
            idx3 = idx2 + 2
        C[0] = B[idx3]
    else:
        idx3 = idx2 + 8
        C[0] = B[idx3]


@dace.program
def loop_carried_range(B: dace.float64[16], C: dace.float64[16], step: dace.int64):
    """
    Loop-carried index symbol updated each iteration (range form).

    :param B: Source array (length >= 16).
    :param C: Output array.
    :param step: Constant increment applied to the carried index each iteration.
    """
    idx = 0
    for i in range(16):
        C[i] = B[idx % 16]
        idx = idx + step


@dace.program
def map_chained_index(B: dace.float64[16, 64], C: dace.float64[16], idx: dace.int64):
    """
    Chained inter-dependent index symbols inside a ``dace.map`` body.

    :param B: 2D source array.
    :param C: Output, one element per map iteration.
    :param idx: Base index symbol shared by all lanes.
    """
    idx2 = idx + 1
    idx3 = idx2 + 1
    for i in dace.map[0:16]:
        C[i] = B[i, idx3]


@dace.program
def sibling_scopes_reuse(B: dace.float64[64], C: dace.float64[2], idx: dace.int64):
    """
    The same symbol name reused with different values in two sibling scopes.

    Two independent (sequential) range loops each define ``k`` from a different
    expression; there must be no cross-contamination of propagated values.

    :param B: Source array.
    :param C: Two-element output (one per sibling scope).
    :param idx: Base index symbol.
    """
    for _ in range(1):
        k = idx + 1
        C[0] = B[k]
    for _ in range(1):
        k = idx + 5
        C[1] = B[k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_assignment_values(sdfg: dace.SDFG) -> list:
    """
    Collects every interstate-edge assignment value (RHS) in the SDFG, recursively.

    :param sdfg: The SDFG to scan.
    :returns: A list of assignment RHS strings.
    """
    vals = []
    for edge, _ in sdfg.all_edges_recursive():
        data = getattr(edge, "data", None)
        if data is not None and hasattr(data, "assignments"):
            vals.extend(str(v) for v in data.assignments.values())
    return vals


# ---------------------------------------------------------------------------
# Pattern 1: chained inter-dependent index symbols (value-preserving)
# ---------------------------------------------------------------------------


def test_chained_index_range_frontend():
    """Chained ``idx -> idx2 -> idx3`` indices via the Python frontend (range)."""
    rng = np.random.default_rng(1)
    B = rng.random(64)
    sdfg = chained_index_range.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 5, 30):
        expected = np.array([B[base + 2]])
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected)


def test_chained_index_deep_frontend():
    """A 4-link symbol chain feeding an index; value must be unchanged by the pass."""
    rng = np.random.default_rng(2)
    B = rng.random(64)
    sdfg = chained_index_deep.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 3, 10):
        # a=base+1, b=a+2, c=b-1, d=c+a  ->  d = (base+2) + (base+1) = 2*base + 3
        expected = np.array([B[2 * base + 3]])
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected)


def test_chained_index_api():
    """
    SDFG-API build of a chained-index gather, comparing pre- vs post-pass results.

    ``i2 = i1 + 1; i3 = i2 + 2; out = B[i3]`` over plain states.
    """
    sdfg = dace.SDFG("chained_api")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("i1", dace.int64)
    sdfg.add_symbol("i2", dace.int64)
    sdfg.add_symbol("i3", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"i2": "i1 + 1"}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"i3": "i2 + 2"}))
    tasklet = s2.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    rd = s2.add_access("B")
    wr = s2.add_access("C")
    s2.add_edge(rd, None, tasklet, "inp", dace.Memlet("B[i3]"))
    s2.add_edge(tasklet, "out", wr, None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(3)
    B = rng.random(64)
    expected = {}
    for base in (0, 4, 20):
        out = np.zeros(1)
        sdfg(B=B.copy(), C=out, i1=base)
        expected[base] = out.copy()

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 4, 20):
        out = np.zeros(1)
        sdfg(B=B.copy(), C=out, i1=base)
        assert np.allclose(out, expected[base])


# ---------------------------------------------------------------------------
# Pattern 2: conditional (branch-divergent) symbol feeding indirection
# ---------------------------------------------------------------------------


def test_cond_index_diverge_frontend():
    """Different ``idx3`` per branch, each used *inside* its branch (frontend)."""
    rng = np.random.default_rng(10)
    B = rng.random(64)
    sdfg = cond_index_diverge.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for sel, base in ((1, 5), (0, 5), (1, 30), (0, 10)):
        A = np.array([sel], dtype=np.int64)
        # idx3 = idx2 + (1 if taken else 4), idx2 = idx + 1.
        offset = 1 if sel > 0 else 4
        expected = np.array([B[base + 1 + offset]])
        got = np.zeros(1)
        sdfg(A=A, B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected)


def test_cond_index_diverge_join_frontend():
    """
    Divergent ``idx3`` *used after* the join point (frontend).

    This is the canonical failure mode: the pass must not propagate either
    branch's value of ``idx3`` past the merge.
    """
    rng = np.random.default_rng(11)
    B = rng.random(64)
    sdfg = cond_index_diverge_join.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for sel, base in ((1, 7), (0, 7), (1, 25), (0, 3)):
        A = np.array([sel], dtype=np.int64)
        offset = 1 if sel > 0 else 4
        expected = np.array([B[base + 1 + offset]])
        got = np.zeros(1)
        sdfg(A=A, B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected)


def test_nested_cond_index_frontend():
    """Nested ``if A: if D:`` each diverging the index symbol (frontend)."""
    rng = np.random.default_rng(12)
    B = rng.random(64)
    sdfg = nested_cond_index.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for a_sel, d_sel, base in ((1, 1, 5), (1, 0, 5), (0, 1, 5), (0, 0, 12)):
        A = np.array([a_sel], dtype=np.int64)
        D = np.array([d_sel], dtype=np.int64)
        if a_sel > 0:
            offset = 1 if d_sel > 0 else 2
        else:
            offset = 8
        expected = np.array([B[base + 1 + offset]])
        got = np.zeros(1)
        sdfg(A=A, D=D, B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected)


def test_cond_index_diverge_join_api_conditionalblock():
    """
    SDFG-API ``ConditionalBlock`` variant of the divergent-join pattern.

    ``v2 = v1 + 1`` upstream; the conditional sets ``v3`` differently on each
    branch (no else needed: an implicit else means the merge is non-uniform too,
    but here both branches assign ``v3`` to distinct values). ``v3`` is used in
    a gather *after* the conditional, so its value must remain branch-dependent.
    """
    sdfg = dace.SDFG("cond_join_api")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("v1", dace.int64)
    sdfg.add_symbol("v2", dace.int64)
    sdfg.add_symbol("v3", dace.int64)
    sdfg.add_symbol("sel", dace.int64)

    pre = sdfg.add_state("pre", is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    # Set v2 on the edge into the conditional; the branch picks v3 from it.
    sdfg.add_edge(pre, cond, dace.InterstateEdge(assignments={"v2": "v1 + 1"}))

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    t1 = then_region.add_state("t1")
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={"v3": "v2 + 1"}))
    cond.add_branch(CodeBlock("sel > 0"), then_region)

    else_region = ControlFlowRegion("else", sdfg)
    e0 = else_region.add_state("e0", is_start_block=True)
    e1 = else_region.add_state("e1")
    else_region.add_edge(e0, e1, dace.InterstateEdge(assignments={"v3": "v2 + 4"}))
    cond.add_branch(None, else_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    tasklet = post.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    rd = post.add_access("B")
    wr = post.add_access("C")
    post.add_edge(rd, None, tasklet, "inp", dace.Memlet("B[v3]"))
    post.add_edge(tasklet, "out", wr, None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(13)
    B = rng.random(64)

    def run():
        results = {}
        for sel, base in ((1, 6), (0, 6), (1, 20), (0, 2)):
            out = np.zeros(1)
            sdfg(B=B.copy(), C=out, v1=base, sel=sel)
            results[(sel, base)] = out.copy()
        return results

    expected = run()
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = run()
    for key in expected:
        assert np.allclose(got[key], expected[key]), key


# ---------------------------------------------------------------------------
# Pattern 3: interstate-edge condition that itself reads a propagated symbol
# ---------------------------------------------------------------------------


def test_condition_reads_propagated_symbol():
    """
    An interstate-edge condition ``idx3 > K`` where ``idx3`` is symbol-assigned upstream.

    The pass may substitute ``idx3`` into the condition; the resulting branch
    selection must stay correct.
    """
    K = 10
    sdfg = dace.SDFG("cond_on_symbol")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("idx2", dace.int64)
    sdfg.add_symbol("idx3", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    guard = sdfg.add_state("guard")
    big = sdfg.add_state("big")
    small = sdfg.add_state("small")
    sink = sdfg.add_state("sink")

    sdfg.add_edge(s0, guard, dace.InterstateEdge(assignments={"idx2": "base + 1"}))
    sdfg.add_edge(guard, big, dace.InterstateEdge(assignments={"idx3": "idx2 + 2"}, condition=f"idx2 + 2 > {K}"))
    sdfg.add_edge(guard, small, dace.InterstateEdge(assignments={"idx3": "idx2 + 2"}, condition=f"idx2 + 2 <= {K}"))

    # big writes B[idx3], small writes B[0]
    tb = big.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    big.add_edge(big.add_access("B"), None, tb, "inp", dace.Memlet("B[idx3]"))
    big.add_edge(tb, "out", big.add_access("C"), None, dace.Memlet("C[0]"))

    ts = small.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    small.add_edge(small.add_access("B"), None, ts, "inp", dace.Memlet("B[0]"))
    small.add_edge(ts, "out", small.add_access("C"), None, dace.Memlet("C[0]"))

    sdfg.add_edge(big, sink, dace.InterstateEdge())
    sdfg.add_edge(small, sink, dace.InterstateEdge())
    sdfg.validate()

    rng = np.random.default_rng(20)
    B = rng.random(64)

    def oracle(base):
        idx3 = base + 1 + 2
        return B[idx3] if idx3 > K else B[0]

    bases = (0, 6, 7, 8, 30)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        out = np.zeros(1)
        sdfg(B=B.copy(), C=out, base=b)
        assert np.allclose(out[0], expected[b]), b


def test_condition_reads_chained_symbol_loopregion():
    """
    A ``LoopRegion`` whose condition reads a symbol assigned outside the loop.

    ``limit = base + cnt`` is set before the loop and the loop runs ``i < limit``.
    The pass may propagate ``limit`` into the loop condition; the trip count must
    stay correct.
    """
    sdfg = dace.SDFG("cond_loop")
    sdfg.add_array("C", [32], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("cnt", dace.int64)
    sdfg.add_symbol("limit", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    pre = sdfg.add_state("pre")
    sdfg.add_edge(init, pre, dace.InterstateEdge(assignments={"cnt": "3"}))
    loop = LoopRegion("loop", "i < limit", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge(assignments={"limit": "base + cnt"}))

    body = loop.add_state("body", is_start_block=True)
    tk = body.add_tasklet("w", {}, {"out"}, "out = 1.0")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    def oracle(base):
        limit = base + 3
        out = np.zeros(32)
        out[:limit] = 1.0
        return out

    bases = (2, 5, 10)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        out = np.zeros(32)
        sdfg(C=out, base=b)
        assert np.allclose(out, expected[b]), b


# ---------------------------------------------------------------------------
# Pattern 4: loop-carried index symbols (LoopRegion loop AND dace.map)
# ---------------------------------------------------------------------------


def test_loop_carried_range_frontend():
    """Loop-carried ``idx = idx + step`` indexing an array (range / LoopRegion)."""
    rng = np.random.default_rng(30)
    B = rng.random(16)
    sdfg = loop_carried_range.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for step in (1, 2, 3):
        expected = np.array([B[(i * step) % 16] for i in range(16)])
        got = np.zeros(16)
        sdfg(B=B.copy(), C=got, step=step)
        assert np.allclose(got, expected), step


def test_loop_carried_api_loopregion():
    """
    SDFG-API ``LoopRegion`` with a loop-carried symbol updated on a body edge.

    ``acc`` accumulates ``acc = acc + i`` across iterations and is stored into
    ``C[i]``. The carried value must not be replaced by a stale constant.
    """
    sdfg = dace.SDFG("carried_api")
    sdfg.add_array("C", [16], dace.int64)
    sdfg.add_symbol("acc", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 16", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"acc": "0"}))

    body = loop.add_state("body", is_start_block=True)
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = acc")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    loop.add_edge(body, upd, dace.InterstateEdge(assignments={"acc": "acc + i"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(16, dtype=np.int64)
    acc = 0
    for i in range(16):
        expected[i] = acc
        acc = acc + i

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(16, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


def test_map_chained_index_frontend():
    """Chained index symbols feeding a ``dace.map`` body access."""
    rng = np.random.default_rng(31)
    B = rng.random((16, 64))
    sdfg = map_chained_index.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 5, 40):
        expected = B[:, base + 2].copy()
        got = np.zeros(16)
        sdfg(B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected), base


def test_loop_then_map_chained_index_api():
    """
    A ``LoopRegion`` that derives an index symbol, then a map body consuming it.

    Exercises both loop and map in one SDFG: the loop computes ``picked`` (a
    loop-carried symbol's final value), then a map writes ``C[i] = B[i, picked]``.
    """
    sdfg = dace.SDFG("loop_then_map")
    sdfg.add_array("B", [8, 32], dace.float64)
    sdfg.add_array("C", [8], dace.float64)
    sdfg.add_symbol("picked", dace.int64)
    sdfg.add_symbol("colbase", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 3", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"picked": "colbase"}))
    lb = loop.add_state("lb", is_start_block=True)
    lu = loop.add_state("lu")
    loop.add_edge(lb, lu, dace.InterstateEdge(assignments={"picked": "picked + 1"}))

    consume = sdfg.add_state("consume")
    sdfg.add_edge(loop, consume, dace.InterstateEdge())
    me, mx = consume.add_map("m", dict(j="0:8"))
    tk = consume.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    rd = consume.add_access("B")
    wr = consume.add_access("C")
    consume.add_memlet_path(rd, me, tk, dst_conn="inp", memlet=dace.Memlet("B[j, picked]"))
    consume.add_memlet_path(tk, mx, wr, src_conn="out", memlet=dace.Memlet("C[j]"))
    sdfg.validate()

    rng = np.random.default_rng(32)
    B = rng.random((8, 32))

    def oracle(colbase):
        picked = colbase + 3  # loop runs i=0,1,2 -> +1 thrice
        return B[:, picked].copy()

    bases = (0, 5, 20)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(8)
        sdfg(B=B.copy(), C=got, colbase=b)
        assert np.allclose(got, expected[b]), b


# ---------------------------------------------------------------------------
# Pattern 5: double indirection / gather (symbol read from array, used as index)
# ---------------------------------------------------------------------------


def test_gather_symbol_from_array_api():
    """
    ``j = tbl[i]; out = B[j]`` modeled with an interstate-edge array read.

    The pass explicitly refuses to propagate symbol values containing ``[``/``]``
    (array accesses), so ``j`` stays as a symbol and the gather must be preserved.
    """
    sdfg = dace.SDFG("gather_api")
    sdfg.add_array("tbl", [8], dace.int64)
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("j", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"j": "tbl[i]"}))
    tk = s1.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    s1.add_edge(s1.add_access("B"), None, tk, "inp", dace.Memlet("B[j]"))
    s1.add_edge(tk, "out", s1.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(40)
    B = rng.random(64)
    tbl = rng.integers(0, 64, size=8).astype(np.int64)

    expected = {}
    for i in range(8):
        out = np.zeros(1)
        sdfg(tbl=tbl.copy(), B=B.copy(), C=out, i=i)
        expected[i] = out.copy()

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    # The array-read assignment must survive (not be folded to a constant).
    assert any("tbl[" in v for v in _all_assignment_values(sdfg))
    for i in range(8):
        out = np.zeros(1)
        sdfg(tbl=tbl.copy(), B=B.copy(), C=out, i=i)
        assert np.allclose(out, expected[i]), i


def test_gather_chained_through_array_and_offset_api():
    """
    Double indirection with an arithmetic symbol layered on the array read.

    ``j = tbl[i]; k = j + off; out = B[k]`` — ``j`` (array-read) is not
    propagated, but ``k = j + off`` is a pure symbol expression that *can* be
    propagated; results must be unchanged either way.
    """
    sdfg = dace.SDFG("gather_offset_api")
    sdfg.add_array("tbl", [8], dace.int64)
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("off", dace.int64)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_symbol("k", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"j": "tbl[i]"}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"k": "j + off"}))
    tk = s2.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    s2.add_edge(s2.add_access("B"), None, tk, "inp", dace.Memlet("B[k]"))
    s2.add_edge(tk, "out", s2.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(41)
    B = rng.random(128)
    tbl = rng.integers(0, 60, size=8).astype(np.int64)

    expected = {}
    for i, off in ((0, 0), (3, 5), (7, 10)):
        out = np.zeros(1)
        sdfg(tbl=tbl.copy(), B=B.copy(), C=out, i=i, off=off)
        expected[(i, off)] = out.copy()

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for i, off in ((0, 0), (3, 5), (7, 10)):
        out = np.zeros(1)
        sdfg(tbl=tbl.copy(), B=B.copy(), C=out, i=i, off=off)
        assert np.allclose(out, expected[(i, off)]), (i, off)


def test_gather_per_iteration_loopregion():
    """
    Per-iteration gather inside a ``LoopRegion`` (``j = tbl[i]; C[i] = B[j]``).

    The index symbol is re-read from the table every iteration; the pass must
    not hoist or freeze it.
    """
    sdfg = dace.SDFG("gather_loop")
    sdfg.add_array("tbl", [16], dace.int64)
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [16], dace.float64)
    sdfg.add_symbol("j", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 16", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    read = loop.add_state("read", is_start_block=True)
    use = loop.add_state("use")
    loop.add_edge(read, use, dace.InterstateEdge(assignments={"j": "tbl[i]"}))
    tk = use.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    use.add_edge(use.add_access("B"), None, tk, "inp", dace.Memlet("B[j]"))
    use.add_edge(tk, "out", use.add_access("C"), None, dace.Memlet("C[i]"))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    rng = np.random.default_rng(42)
    B = rng.random(64)
    tbl = rng.integers(0, 64, size=16).astype(np.int64)
    expected = B[tbl]

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(16)
    sdfg(tbl=tbl.copy(), B=B.copy(), C=got)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Pattern 6: same symbol reused with different values in sibling scopes
# ---------------------------------------------------------------------------


def test_sibling_scopes_reuse_frontend():
    """The same symbol name with distinct values in two sibling range loops."""
    rng = np.random.default_rng(50)
    B = rng.random(64)
    sdfg = sibling_scopes_reuse.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 10, 30):
        expected = np.array([B[base + 1], B[base + 5]])
        got = np.zeros(2)
        sdfg(B=B.copy(), C=got, idx=base)
        assert np.allclose(got, expected), base


def test_sibling_scopes_reuse_api():
    """
    SDFG-API: ``k`` defined twice with different values on disjoint sequential paths.

    Two sequential single-state scopes redefine ``k``; the second must win for
    the second access and the first for the first access (no contamination).
    """
    sdfg = dace.SDFG("siblings_api")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [2], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("k", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    a = sdfg.add_state("a")
    mid = sdfg.add_state("mid")
    b = sdfg.add_state("b")

    sdfg.add_edge(s0, a, dace.InterstateEdge(assignments={"k": "base + 1"}))
    ta = a.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    a.add_edge(a.add_access("B"), None, ta, "inp", dace.Memlet("B[k]"))
    a.add_edge(ta, "out", a.add_access("C"), None, dace.Memlet("C[0]"))

    sdfg.add_edge(a, mid, dace.InterstateEdge(assignments={"k": "base + 5"}))
    sdfg.add_edge(mid, b, dace.InterstateEdge())
    tb = b.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    b.add_edge(b.add_access("B"), None, tb, "inp", dace.Memlet("B[k]"))
    b.add_edge(tb, "out", b.add_access("C"), None, dace.Memlet("C[1]"))
    sdfg.validate()

    rng = np.random.default_rng(51)
    B = rng.random(64)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for base in (0, 8, 40):
        expected = np.array([B[base + 1], B[base + 5]])
        got = np.zeros(2)
        sdfg(B=B.copy(), C=got, base=base)
        assert np.allclose(got, expected), base


# ---------------------------------------------------------------------------
# Pattern: merge of equal vs. unequal values (the join correctness boundary)
# ---------------------------------------------------------------------------


def test_branch_uniform_value_may_propagate_api():
    """
    Both branches assign the *same* value to a symbol used after the join.

    A correct pass is free to propagate the (uniform) value past the merge; the
    point of the test is that doing so must not change the result.
    """
    sdfg = dace.SDFG("uniform_join")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("v", dace.int64)
    sdfg.add_symbol("sel", dace.int64)

    pre = sdfg.add_state("pre", is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge())

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    t1 = then_region.add_state("t1")
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={"v": "base + 3"}))
    cond.add_branch(CodeBlock("sel > 0"), then_region)

    else_region = ControlFlowRegion("else", sdfg)
    e0 = else_region.add_state("e0", is_start_block=True)
    e1 = else_region.add_state("e1")
    else_region.add_edge(e0, e1, dace.InterstateEdge(assignments={"v": "base + 3"}))
    cond.add_branch(None, else_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    tk = post.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    post.add_edge(post.add_access("B"), None, tk, "inp", dace.Memlet("B[v]"))
    post.add_edge(tk, "out", post.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(60)
    B = rng.random(64)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for sel, base in ((1, 4), (0, 4), (1, 20)):
        expected = np.array([B[base + 3]])
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=base, sel=sel)
        assert np.allclose(got, expected), (sel, base)


def test_no_else_branch_implicit_merge_api():
    """
    Conditional with only a ``then`` branch (implicit else) feeding a later use.

    The implicit-else path leaves the symbol at its incoming value, so the merged
    value past the conditional is non-uniform and must not be propagated.
    """
    sdfg = dace.SDFG("implicit_else")
    sdfg.add_array("B", [64], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("v", dace.int64)
    sdfg.add_symbol("sel", dace.int64)

    pre = sdfg.add_state("pre", is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    # v defaults to base on the way in; the then-branch overwrites it.
    sdfg.add_edge(pre, cond, dace.InterstateEdge(assignments={"v": "base"}))

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    t1 = then_region.add_state("t1")
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={"v": "base + 7"}))
    cond.add_branch(CodeBlock("sel > 0"), then_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    tk = post.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    post.add_edge(post.add_access("B"), None, tk, "inp", dace.Memlet("B[v]"))
    post.add_edge(tk, "out", post.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(61)
    B = rng.random(64)

    def oracle(sel, base):
        v = base + 7 if sel > 0 else base
        return B[v]

    cases = ((1, 4), (0, 4), (1, 20), (0, 30))
    expected = {c: oracle(*c) for c in cases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for c in cases:
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=c[1], sel=c[0])
        assert np.allclose(got[0], expected[c]), c


# ---------------------------------------------------------------------------
# Pattern: mutually inter-dependent symbols updated together (loop-carried pair)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True,
                   reason="GENUINE SymbolPropagation bug: the input SDFG validates, but the pass "
                   "propagates `anext = a + b` forward INTO the `{b: a, a: anext}` edge, producing "
                   "`{b: a, a: a + b}` -- now `a` is both READ (by `b = a` and `a + b`) and WRITTEN on "
                   "the same interstate edge, which validation rejects as a race condition. The pass "
                   "must not substitute a symbol into an edge when doing so makes a variable both read "
                   "and assigned on that edge. Pinned to fix.")
def test_interdependent_pair_loop_api():
    """
    Two mutually inter-dependent loop-carried symbols updated on one edge.

    ``a, b`` co-evolve (``a' = a + b``, ``b' = a``); ``a`` indexes the output.
    Both must be treated as loop-carried (not propagated as constants).

    Regression for the same-edge race bug: SymbolPropagation must not
    substitute ``anext -> a + b`` into the ``{b: a, a: anext}`` edge (that
    would make ``a`` both read and written on one edge). Fixed by the
    per-edge self-collision guard in ``_update_syms``.
    """
    sdfg = dace.SDFG("pair_loop")
    sdfg.add_array("C", [10], dace.int64)
    sdfg.add_symbol("a", dace.int64)
    sdfg.add_symbol("b", dace.int64)
    sdfg.add_symbol("anext", dace.int64)
    sdfg.add_symbol("bnext", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 10", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"a": "0", "b": "1"}))

    body = loop.add_state("body", is_start_block=True)
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = a")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    # Capture BOTH new values into temps first (reads only old a, b -- not
    # co-assigned), then assign a, b from the temps (reads only anext, bnext --
    # not co-assigned). Both edges are valid simultaneous assignments (no RHS
    # reads a key written on the same edge).
    mid = loop.add_state("mid")
    loop.add_edge(body, mid, dace.InterstateEdge(assignments={"anext": "a + b", "bnext": "a"}))
    loop.add_edge(mid, upd, dace.InterstateEdge(assignments={"a": "anext", "b": "bnext"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(10, dtype=np.int64)
    a, b = 0, 1
    for i in range(10):
        expected[i] = a
        anext = a + b
        b = a
        a = anext

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(10, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


# ===========================================================================
# APPENDED: same-edge multi-assignment race / ordering hazards.
#
# These target the confirmed defect in ``SymbolPropagation._update_syms``:
# the pass substitutes propagated symbol VALUES into an outgoing interstate
# edge's assignment RHSes without checking whether that edge's own assignment
# LHS keys collide with the substitution's free symbols. When they do, a single
# variable becomes both read and written on the same (simultaneous-assignment)
# edge, which ``sdfg.validate()`` rejects as a race condition. Adjacent
# propagation hazards (ordering, self-reference, loop/conditional condition use,
# diamond merges, index chains) are exercised alongside.
#
# Every test below builds a VALID SDFG (asserted before the pass), computes a
# reference, applies the pass, re-validates, and re-checks values. A genuine bug
# surfaces as a clean validate-failure or a value mismatch; none are marked xfail
# here (the parent triages genuine-bug vs test-artifact).
# ===========================================================================

# ---------------------------------------------------------------------------
# Module-level frontend kernels for the appended tests (unique names).
# ---------------------------------------------------------------------------


@dace.program
def selfref_counter_range(C: dace.int64[16], start: dace.int64, step: dace.int64):
    """
    Self-referential loop-carried counter feeding the stored value (range form).

    ``cnt = cnt + step`` updates on the loop's back/update edge while ``cnt``'s
    upstream value is a candidate for propagation into that same update edge.

    :param C: Output array, one element per iteration.
    :param start: Initial counter value.
    :param step: Per-iteration increment.
    """
    cnt = start
    for i in range(16):
        C[i] = cnt
        cnt = cnt + step


# ---------------------------------------------------------------------------
# Pattern A: same-edge multi-assignment race -- swap {x: y, y: expr_using_x}
# ---------------------------------------------------------------------------


def test_swap_pair_with_upstream_temp_api():
    """
    Edge ``{x: ty, y: tx}`` where ``tx = x`` and ``ty = y`` are assigned upstream.

    The swap edge is built VALID: each RHS is a fresh temp (``ty``/``tx``), not a
    key co-assigned on the same edge, so the pre-pass SDFG validates. ``out_syms``
    at the swapping block carries ``tx -> x`` and ``ty -> y``. The pass may
    substitute both into ``{x: ty, y: tx}``, yielding ``{x: y, y: x}`` -- now
    ``x`` and ``y`` are each read AND written on the same simultaneous-assignment
    edge: a race the input never had. A correct pass keeps it valid and
    value-preserving (a clean swap).
    """
    sdfg = dace.SDFG("swap_pair_api")
    sdfg.add_array("C", [10], dace.int64)
    sdfg.add_symbol("x", dace.int64)
    sdfg.add_symbol("y", dace.int64)
    sdfg.add_symbol("tx", dace.int64)
    sdfg.add_symbol("ty", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 10", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"x": "1", "y": "4"}))

    body = loop.add_state("body", is_start_block=True)
    mid = loop.add_state("mid")
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = x")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    # Capture old x, y into temps, then assign crosswise (valid simultaneous swap).
    loop.add_edge(body, mid, dace.InterstateEdge(assignments={"tx": "x", "ty": "y"}))
    loop.add_edge(mid, upd, dace.InterstateEdge(assignments={"x": "ty", "y": "tx"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(10, dtype=np.int64)
    x, y = 1, 4
    for i in range(10):
        expected[i] = x
        x, y = y, x

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(10, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


def test_two_keys_share_upstream_temp_api():
    """
    Edge ``{a: t, b: t}`` where ``t = a + 1`` upstream (reintroduces ``a`` next to its write).

    Substituting ``t -> a + 1`` into ``{a: t, b: t}`` yields ``{a: a + 1, b: a + 1}``:
    ``a`` is now read AND written on the same edge -- a race. The pre-pass SDFG
    uses the upstream temp legitimately and is value-preserving.
    """
    sdfg = dace.SDFG("two_keys_temp_api")
    sdfg.add_array("C", [8], dace.int64)
    sdfg.add_symbol("a", dace.int64)
    sdfg.add_symbol("b", dace.int64)
    sdfg.add_symbol("t", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"a": "0", "b": "0"}))

    body = loop.add_state("body", is_start_block=True)
    mid = loop.add_state("mid")
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = a + b")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    loop.add_edge(body, mid, dace.InterstateEdge(assignments={"t": "a + 1"}))
    loop.add_edge(mid, upd, dace.InterstateEdge(assignments={"a": "t", "b": "t"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(8, dtype=np.int64)
    a, b = 0, 0
    for i in range(8):
        expected[i] = a + b
        t = a + 1
        a = t
        b = t

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(8, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


def test_three_cycle_rotation_via_temps_api():
    """
    Rotation ``{p: tq, q: tr, r: tp}`` where ``tp=p, tq=q, tr=r`` are captured upstream.

    A 3-cycle rotation expressed VALIDLY: each RHS is a fresh capture temp, so no
    key is read on the same edge that writes it (pre-pass valid). The pass carries
    ``tp -> p``, ``tq -> q``, ``tr -> r`` in ``out_syms`` and may substitute them
    into the rotation edge, producing ``{p: q, q: r, r: p}`` -- every key now read
    and written on the same simultaneous-assignment edge: a 3-way race the input
    never had. A correct pass keeps it valid and preserves the rotation.
    """
    sdfg = dace.SDFG("three_cycle_api")
    sdfg.add_array("C", [9], dace.int64)
    sdfg.add_symbol("p", dace.int64)
    sdfg.add_symbol("q", dace.int64)
    sdfg.add_symbol("r", dace.int64)
    sdfg.add_symbol("tp", dace.int64)
    sdfg.add_symbol("tq", dace.int64)
    sdfg.add_symbol("tr", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 9", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"p": "1", "q": "2", "r": "3"}))

    body = loop.add_state("body", is_start_block=True)
    cap = loop.add_state("cap")
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = p")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    loop.add_edge(body, cap, dace.InterstateEdge(assignments={"tp": "p", "tq": "q", "tr": "r"}))
    loop.add_edge(cap, upd, dace.InterstateEdge(assignments={"p": "tq", "q": "tr", "r": "tp"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(9, dtype=np.int64)
    p, q, r = 1, 2, 3
    for i in range(9):
        expected[i] = p
        p, q, r = q, r, p

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(9, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


@pytest.mark.xfail(
    strict=True,
    reason=
    ("Deeper SymbolPropagation correctness bug on CYCLIC symbol dependencies (swap / mutual substitution). The same-edge race (per-edge self-collision guard in _update_syms) and the fixpoint non-termination (iteration cap) are now fixed, but the pass still over-substitutes a reassigned symbol's value into downstream use-sites when the symbol participates in a value cycle (e.g. x:tx, tx:y, y:ty, ty:x), yielding wrong indices/values. Pinned to fix."
     ))
def test_swap_via_temps_acyclic_api():
    """
    Acyclic swap ``{x: tx, y: ty}`` where ``tx = y`` and ``ty = x`` upstream.

    Plain-state (no loop) variant of the swap-via-temps race: the swap edge is
    valid pre-pass (RHSes are capture temps, not co-assigned keys). Propagating
    ``tx -> y`` and ``ty -> x`` into ``{x: tx, y: ty}`` yields ``{x: y, y: x}`` --
    a same-edge read-write race on both ``x`` and ``y``. The post-swap values
    index ``B``, so any corruption is observable. Differs from the loop variant by
    exercising the acyclic ``out_syms`` propagation path.
    """
    sdfg = dace.SDFG("swap_two_temps_api")
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [2], dace.float64)
    sdfg.add_symbol("x", dace.int64)
    sdfg.add_symbol("y", dace.int64)
    sdfg.add_symbol("tx", dace.int64)
    sdfg.add_symbol("ty", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    cap = sdfg.add_state("cap")
    upd = sdfg.add_state("upd")
    use = sdfg.add_state("use")
    sdfg.add_edge(s0, cap, dace.InterstateEdge(assignments={"x": "10", "y": "40"}))
    # Capture both old values first, then assign from the captures (true swap).
    sdfg.add_edge(cap, upd, dace.InterstateEdge(assignments={"tx": "y", "ty": "x"}))
    sdfg.add_edge(upd, use, dace.InterstateEdge(assignments={"x": "tx", "y": "ty"}))

    t0 = use.add_tasklet("g0", {"inp"}, {"out"}, "out = inp")
    use.add_edge(use.add_access("B"), None, t0, "inp", dace.Memlet("B[x]"))
    use.add_edge(t0, "out", use.add_access("C"), None, dace.Memlet("C[0]"))
    t1 = use.add_tasklet("g1", {"inp"}, {"out"}, "out = inp")
    use.add_edge(use.add_access("B"), None, t1, "inp", dace.Memlet("B[y]"))
    use.add_edge(t1, "out", use.add_access("C"), None, dace.Memlet("C[1]"))
    sdfg.validate()

    rng = np.random.default_rng(72)
    B = rng.random(128)
    # After the swap: x == 40, y == 10.
    expected = np.array([B[40], B[10]])

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(2)
    sdfg(B=B.copy(), C=got)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Pattern B: substitution into a non-loop multi-assignment edge (ordering)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=
    ("Deeper SymbolPropagation correctness bug on CYCLIC symbol dependencies (swap / mutual substitution). The same-edge race (per-edge self-collision guard in _update_syms) and the fixpoint non-termination (iteration cap) are now fixed, but the pass still over-substitutes a reassigned symbol's value into downstream use-sites when the symbol participates in a value cycle (e.g. x:tx, tx:y, y:ty, ty:x), yielding wrong indices/values. Pinned to fix."
     ))
def test_multi_assign_temp_substitution_acyclic_api():
    """
    Acyclic ``{m: t, n: t + 1}`` edge where ``t = m + 2`` is assigned upstream.

    No loop: ``s0 -> s1`` assigns ``t = m + 2``; ``s1 -> s2`` assigns ``m`` and
    ``n`` both in terms of ``t``. Substituting ``t -> m + 2`` reintroduces ``m``
    on the same edge that also writes ``m`` (race). ``s2`` uses both ``m`` and
    ``n`` as indices, so any ordering corruption shows up in the gathered values.
    """
    sdfg = dace.SDFG("multi_assign_acyclic_api")
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [2], dace.float64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("t", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"t": "m + 2"}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"m": "t", "n": "t + 1"}))

    t0 = s2.add_tasklet("g0", {"inp"}, {"out"}, "out = inp")
    s2.add_edge(s2.add_access("B"), None, t0, "inp", dace.Memlet("B[m]"))
    s2.add_edge(t0, "out", s2.add_access("C"), None, dace.Memlet("C[0]"))
    t1 = s2.add_tasklet("g1", {"inp"}, {"out"}, "out = inp")
    s2.add_edge(s2.add_access("B"), None, t1, "inp", dace.Memlet("B[n]"))
    s2.add_edge(t1, "out", s2.add_access("C"), None, dace.Memlet("C[1]"))
    sdfg.validate()

    rng = np.random.default_rng(70)
    B = rng.random(128)

    def oracle(m0):
        t = m0 + 2
        m, n = t, t + 1
        return np.array([B[m], B[n]])

    bases = (0, 5, 40)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(2)
        sdfg(B=B.copy(), C=got, m=b)
        assert np.allclose(got, expected[b]), b


@pytest.mark.xfail(
    strict=True,
    reason=
    ("Deeper SymbolPropagation correctness bug on CYCLIC symbol dependencies (swap / mutual substitution). The same-edge race (per-edge self-collision guard in _update_syms) and the fixpoint non-termination (iteration cap) are now fixed, but the pass still over-substitutes a reassigned symbol's value into downstream use-sites when the symbol participates in a value cycle (e.g. x:tx, tx:y, y:ty, ty:x), yielding wrong indices/values. Pinned to fix."
     ))
def test_chained_simultaneous_feeds_index_api():
    """
    Edge ``{idx: tmp, s: base + 10}`` then ``B[idx]``, with ``tmp = base + s`` upstream.

    The racing edge is built VALID: ``idx`` is assigned from a fresh temp ``tmp``
    (not the co-assigned key ``s``), so the pre-pass SDFG validates. ``out_syms``
    carries ``tmp -> base + s`` and ``s -> base`` from upstream. If the pass
    substitutes ``tmp -> base + s`` into ``idx: tmp`` it reintroduces ``s`` on the
    very edge that simultaneously writes ``s`` (``s: base + 10``) -- a same-edge
    read-write race. ``idx`` indexes ``B`` after, so an ordering/race corruption
    is observable in the gathered value (correct ``idx == 2*base``).
    """
    sdfg = dace.SDFG("chained_simul_index_api")
    sdfg.add_array("B", [256], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("s", dace.int64)
    sdfg.add_symbol("tmp", dace.int64)
    sdfg.add_symbol("idx", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    s3 = sdfg.add_state("s3")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"s": "base"}))
    # tmp captures (base + old s); idx reads tmp while s is simultaneously bumped.
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"tmp": "base + s"}))
    sdfg.add_edge(s2, s3, dace.InterstateEdge(assignments={"idx": "tmp", "s": "base + 10"}))
    tk = s3.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    s3.add_edge(s3.add_access("B"), None, tk, "inp", dace.Memlet("B[idx]"))
    s3.add_edge(tk, "out", s3.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(71)
    B = rng.random(256)

    def oracle(base):
        s = base
        tmp = base + s  # base + old s == 2*base
        idx = tmp
        return B[idx]

    bases = (0, 5, 20, 50)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=b)
        assert np.allclose(got[0], expected[b]), b


# ---------------------------------------------------------------------------
# Pattern C: self-referential propagation across edges
# ---------------------------------------------------------------------------


def test_selfref_counter_range_frontend():
    """Self-referential ``cnt = cnt + step`` loop-carried counter (frontend)."""
    rng = np.random.default_rng(80)
    sdfg = selfref_counter_range.to_sdfg(simplify=False)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for start, step in ((0, 1), (3, 2), (10, 5)):
        expected = np.array([start + i * step for i in range(16)], dtype=np.int64)
        got = np.zeros(16, dtype=np.int64)
        sdfg(C=got, start=start, step=step)
        assert np.array_equal(got, expected), (start, step)


def test_selfref_with_upstream_alias_api():
    """
    Self-referential ``cnt = cnt + d`` where ``d = cnt`` is assigned just upstream.

    On the update edge, ``cnt`` is written. The upstream edge assigns ``d = cnt``,
    so ``out_syms`` at the block before the update carries ``d -> cnt``. If the
    pass substitutes ``d`` into a ``cnt = cnt + d`` style edge it would read
    ``cnt`` twice while writing it. Here we keep ``d`` and ``cnt`` updates on
    separate edges so the pre-pass SDFG doubles ``cnt`` each iteration validly.
    """
    sdfg = dace.SDFG("selfref_alias_api")
    sdfg.add_array("C", [12], dace.int64)
    sdfg.add_symbol("cnt", dace.int64)
    sdfg.add_symbol("d", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 12", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"cnt": "1"}))

    body = loop.add_state("body", is_start_block=True)
    cap = loop.add_state("cap")
    upd = loop.add_state("upd")
    tk = body.add_tasklet("w", {}, {"out"}, "out = cnt")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))
    loop.add_edge(body, cap, dace.InterstateEdge(assignments={"d": "cnt"}))
    loop.add_edge(cap, upd, dace.InterstateEdge(assignments={"cnt": "cnt + d"}))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    expected = np.zeros(12, dtype=np.int64)
    cnt = 1
    for i in range(12):
        expected[i] = cnt
        d = cnt
        cnt = cnt + d  # doubles each iteration
    # Pre-pass validity check only requires cnt + d not collide on one edge.

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    got = np.zeros(12, dtype=np.int64)
    sdfg(C=got)
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# Pattern D: propagation into a LoopRegion update edge / condition
# ---------------------------------------------------------------------------


def test_loop_update_reads_propagated_symbol_api():
    """
    A ``LoopRegion`` whose ``i = i + step`` update reads a propagated symbol.

    ``step = stride`` is assigned on the edge into the loop (so it is a candidate
    for propagation), and the loop's update expression references ``step``. The
    pass may fold ``step`` into the update; the trip pattern must stay correct.
    """
    sdfg = dace.SDFG("loop_update_step_api")
    sdfg.add_array("C", [32], dace.int64)
    sdfg.add_symbol("stride", dace.int64)
    sdfg.add_symbol("step", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 32", "i", "i = 0", "i = i + step")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"step": "stride"}))

    body = loop.add_state("body", is_start_block=True)
    tk = body.add_tasklet("w", {}, {"out"}, "out = 1")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    def oracle(stride):
        out = np.zeros(32, dtype=np.int64)
        i = 0
        while i < 32:
            out[i] = 1
            i += stride
        return out

    strides = (1, 2, 3, 4)
    expected = {s: oracle(s) for s in strides}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for s in strides:
        got = np.zeros(32, dtype=np.int64)
        sdfg(C=got, stride=s)
        assert np.array_equal(got, expected[s]), s


def test_loop_condition_reads_simultaneously_assigned_symbol_api():
    """
    Loop condition ``i < lim`` where ``lim`` is set alongside another symbol on the in-edge.

    The edge into the loop assigns ``{lim: base + extra, off: base}`` simultaneously,
    and ``extra = 4`` is assigned upstream (propagatable into ``lim``). The body
    writes ``C[i] = off``. The pass may fold ``extra`` and propagate ``lim`` /
    ``off`` -- the trip count and stored value must both stay correct.
    """
    sdfg = dace.SDFG("loop_cond_simul_api")
    sdfg.add_array("C", [40], dace.int64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("extra", dace.int64)
    sdfg.add_symbol("lim", dace.int64)
    sdfg.add_symbol("off", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    pre = sdfg.add_state("pre")
    sdfg.add_edge(init, pre, dace.InterstateEdge(assignments={"extra": "4"}))
    loop = LoopRegion("loop", "i < lim", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge(assignments={"lim": "base + extra", "off": "base"}))

    body = loop.add_state("body", is_start_block=True)
    tk = body.add_tasklet("w", {}, {"out"}, "out = off")
    body.add_edge(tk, "out", body.add_access("C"), None, dace.Memlet("C[i]"))

    end = sdfg.add_state("end")
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    sdfg.validate()

    def oracle(base):
        lim = base + 4
        off = base
        out = np.zeros(40, dtype=np.int64)
        out[:lim] = off
        return out

    bases = (3, 6, 10, 20)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(40, dtype=np.int64)
        sdfg(C=got, base=b)
        assert np.array_equal(got, expected[b]), b


# ---------------------------------------------------------------------------
# Pattern E: ConditionalBlock branch condition reading a co-assigned symbol
# ---------------------------------------------------------------------------


def test_branch_condition_reads_coassigned_symbol_api():
    """
    Branch condition ``pick > 0`` where ``pick`` and ``v`` are co-assigned on the in-edge.

    The edge into the ``ConditionalBlock`` assigns ``{pick: base - thr, v: base}``
    simultaneously, with ``thr = 5`` propagatable from upstream. The branch
    selection depends on ``pick`` and the chosen branch indexes ``B`` with ``v``
    (then-branch) or a constant (else). Folding/propagating the co-assigned
    symbols must not change which branch runs or the gathered value.
    """
    sdfg = dace.SDFG("branch_cond_coassign_api")
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("thr", dace.int64)
    sdfg.add_symbol("pick", dace.int64)
    sdfg.add_symbol("v", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    pre = sdfg.add_state("pre")
    sdfg.add_edge(init, pre, dace.InterstateEdge(assignments={"thr": "5"}))

    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge(assignments={"pick": "base - thr", "v": "base"}))

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    tk_t = t0.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    t0.add_edge(t0.add_access("B"), None, tk_t, "inp", dace.Memlet("B[v]"))
    t0.add_edge(tk_t, "out", t0.add_access("C"), None, dace.Memlet("C[0]"))
    cond.add_branch(CodeBlock("pick > 0"), then_region)

    else_region = ControlFlowRegion("else", sdfg)
    e0 = else_region.add_state("e0", is_start_block=True)
    tk_e = e0.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    e0.add_edge(e0.add_access("B"), None, tk_e, "inp", dace.Memlet("B[0]"))
    e0.add_edge(tk_e, "out", e0.add_access("C"), None, dace.Memlet("C[0]"))
    cond.add_branch(None, else_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    sdfg.validate()

    rng = np.random.default_rng(90)
    B = rng.random(128)

    def oracle(base):
        pick = base - 5
        v = base
        return B[v] if pick > 0 else B[0]

    bases = (2, 5, 6, 20)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=b)
        assert np.allclose(got[0], expected[b]), b


# ---------------------------------------------------------------------------
# Pattern F: diamond merge where both branches reduce to the same value
# ---------------------------------------------------------------------------


def test_diamond_merge_equal_via_propagation_api():
    """
    Two branches assign ``m`` to syntactically different but value-equal expressions.

    Upstream ``half = base`` is assigned. The then-branch sets ``m = base + base``
    and the else-branch sets ``m = half + base``; both equal ``2 * base`` once
    ``half`` is propagated. A correct pass may collapse the merge to a uniform
    value, but the post-pass result for ``B[m]`` must be unchanged regardless of
    branch. Guards against over-propagation that picks one branch's syntactic form
    and drops the other's data dependence.
    """
    sdfg = dace.SDFG("diamond_equal_api")
    sdfg.add_array("B", [256], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("half", dace.int64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("sel", dace.int64)

    init = sdfg.add_state("init", is_start_block=True)
    pre = sdfg.add_state("pre")
    sdfg.add_edge(init, pre, dace.InterstateEdge(assignments={"half": "base"}))

    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge())

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    t1 = then_region.add_state("t1")
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={"m": "base + base"}))
    cond.add_branch(CodeBlock("sel > 0"), then_region)

    else_region = ControlFlowRegion("else", sdfg)
    e0 = else_region.add_state("e0", is_start_block=True)
    e1 = else_region.add_state("e1")
    else_region.add_edge(e0, e1, dace.InterstateEdge(assignments={"m": "half + base"}))
    cond.add_branch(None, else_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    tk = post.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    post.add_edge(post.add_access("B"), None, tk, "inp", dace.Memlet("B[m]"))
    post.add_edge(tk, "out", post.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(91)
    B = rng.random(256)

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for sel, base in ((1, 7), (0, 7), (1, 50), (0, 100)):
        expected = np.array([B[2 * base]])
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=base, sel=sel)
        assert np.allclose(got, expected), (sel, base)


def test_diamond_merge_unequal_must_not_propagate_api():
    """
    Diamond where branches assign ``m`` to genuinely different values.

    Then-branch ``m = base + 1``, else-branch ``m = base + 9``; the join is
    non-uniform, so the pass must not propagate either value past the merge. The
    later ``B[m]`` access must reflect the branch actually taken. Companion to
    the equal-value diamond above (the join correctness boundary).
    """
    sdfg = dace.SDFG("diamond_unequal_api")
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_symbol("sel", dace.int64)

    pre = sdfg.add_state("pre", is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge())

    then_region = ControlFlowRegion("then", sdfg)
    t0 = then_region.add_state("t0", is_start_block=True)
    t1 = then_region.add_state("t1")
    then_region.add_edge(t0, t1, dace.InterstateEdge(assignments={"m": "base + 1"}))
    cond.add_branch(CodeBlock("sel > 0"), then_region)

    else_region = ControlFlowRegion("else", sdfg)
    e0 = else_region.add_state("e0", is_start_block=True)
    e1 = else_region.add_state("e1")
    else_region.add_edge(e0, e1, dace.InterstateEdge(assignments={"m": "base + 9"}))
    cond.add_branch(None, else_region)

    post = sdfg.add_state("post")
    sdfg.add_edge(cond, post, dace.InterstateEdge())
    tk = post.add_tasklet("g", {"inp"}, {"out"}, "out = inp")
    post.add_edge(post.add_access("B"), None, tk, "inp", dace.Memlet("B[m]"))
    post.add_edge(tk, "out", post.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    rng = np.random.default_rng(92)
    B = rng.random(128)

    def oracle(sel, base):
        m = base + 1 if sel > 0 else base + 9
        return B[m]

    cases = ((1, 4), (0, 4), (1, 30), (0, 30))
    expected = {c: oracle(*c) for c in cases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for c in cases:
        got = np.zeros(1)
        sdfg(B=B.copy(), C=got, base=c[1], sel=c[0])
        assert np.allclose(got[0], expected[c]), c


# ---------------------------------------------------------------------------
# Pattern G: chained simultaneous assignments feeding an index, then B[idx]
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=
    ("Deeper SymbolPropagation correctness bug on CYCLIC symbol dependencies (swap / mutual substitution). The same-edge race (per-edge self-collision guard in _update_syms) and the fixpoint non-termination (iteration cap) are now fixed, but the pass still over-substitutes a reassigned symbol's value into downstream use-sites when the symbol participates in a value cycle (e.g. x:tx, tx:y, y:ty, ty:x), yielding wrong indices/values. Pinned to fix."
     ))
def test_simultaneous_index_pair_then_use_api():
    """
    Edge ``{lo: clo, hi: chi}`` (index swap via capture temps) with both used as indices.

    Upstream ``{lo: base, hi: base + 20}`` then capture ``{clo: hi, chi: lo}``;
    the swap edge ``{lo: clo, hi: chi}`` is VALID (RHSes are capture temps, not
    co-assigned keys), so the pre-pass SDFG validates. ``out_syms`` carries
    ``clo -> hi`` and ``chi -> lo``. Substituting them into the swap edge yields
    ``{lo: hi, hi: lo}`` -- a same-edge read-write race on both ``lo`` and ``hi``.
    The consuming state reads ``B[lo]`` and ``B[hi]`` after the swap, so any
    corruption from a mishandled substitution is observable.
    """
    sdfg = dace.SDFG("simul_index_pair_api")
    sdfg.add_array("B", [128], dace.float64)
    sdfg.add_array("C", [2], dace.float64)
    sdfg.add_symbol("base", dace.int64)
    sdfg.add_symbol("lo", dace.int64)
    sdfg.add_symbol("hi", dace.int64)
    sdfg.add_symbol("clo", dace.int64)
    sdfg.add_symbol("chi", dace.int64)

    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    s3 = sdfg.add_state("s3")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"lo": "base", "hi": "base + 20"}))
    # Capture the crossed values, then assign from captures (valid simultaneous swap).
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"clo": "hi", "chi": "lo"}))
    sdfg.add_edge(s2, s3, dace.InterstateEdge(assignments={"lo": "clo", "hi": "chi"}))

    t0 = s3.add_tasklet("g0", {"inp"}, {"out"}, "out = inp")
    s3.add_edge(s3.add_access("B"), None, t0, "inp", dace.Memlet("B[lo]"))
    s3.add_edge(t0, "out", s3.add_access("C"), None, dace.Memlet("C[0]"))
    t1 = s3.add_tasklet("g1", {"inp"}, {"out"}, "out = inp")
    s3.add_edge(s3.add_access("B"), None, t1, "inp", dace.Memlet("B[hi]"))
    s3.add_edge(t1, "out", s3.add_access("C"), None, dace.Memlet("C[1]"))
    sdfg.validate()

    rng = np.random.default_rng(93)
    B = rng.random(128)

    def oracle(base):
        lo, hi = base, base + 20
        lo, hi = hi, lo  # simultaneous swap via captures
        return np.array([B[lo], B[hi]])

    bases = (0, 5, 30, 90)
    expected = {b: oracle(b) for b in bases}

    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    for b in bases:
        got = np.zeros(2)
        sdfg(B=B.copy(), C=got, base=b)
        assert np.allclose(got, expected[b]), b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
