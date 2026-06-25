# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg import utils as sdutil
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.array_elimination import ArrayElimination


def test_redundant_simple():

    @dace.program
    def tester(A: dace.float64[20], B: dace.float64[20]):
        e = dace.ndarray([20], dace.float64)
        f = dace.ndarray([20], dace.float64)
        g = dace.ndarray([20], dace.float64)
        h = dace.ndarray([20], dace.float64)
        c = A + 1
        d = A + 2
        e[:] = c
        f[:] = d
        g[:] = f
        h[:] = d
        B[:] = g + e

    sdfg = tester.to_sdfg(simplify=False)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(sdfg.arrays) == 4


def test_merge_simple():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)

    state = sdfg.add_state()
    a1 = state.add_read('A')
    a2 = state.add_read('A')
    b1 = state.add_write('B')
    b2 = state.add_write('B')
    t1 = state.add_tasklet('doit1', {'a'}, {'b'}, 'b = a')
    t2 = state.add_tasklet('doit2', {'a'}, {'b'}, 'b = a')
    state.add_edge(a1, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(a2, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t1, 'b', b1, None, dace.Memlet('B[0]'))
    state.add_edge(t2, 'b', b2, None, dace.Memlet('B[1]'))

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(state.data_nodes()) == 2


def test_source_merge_refuses_when_data_is_also_written():
    """Frontend-emitted shape: state has two source AccessNodes (``in_degree
    == 0``) for ``b`` plus one sink AccessNode (``in_degree > 0``) for ``b``.
    The topological "fence" that keeps each source's downstream reader after
    the in-state write is the presence of distinct predecessor-less source
    nodes. Folding the two sources into one removes that fence and codegen
    is free to reorder a read past the write.

    ``ArrayElimination.merge_access_nodes`` must refuse the source-merge in
    this configuration. The three ``b`` nodes must survive.
    """
    from dace.sdfg import nodes

    sdfg = dace.SDFG('source_merge_with_write')
    sdfg.add_array('b', [10], dace.float64)
    sdfg.add_array('out', [10], dace.float64)
    state = sdfg.add_state()

    src_b1 = state.add_read('b')
    sink_b = state.add_write('b')
    t_write = state.add_tasklet('twrite', {'x'}, {'y'}, 'y = x + 1.0')
    state.add_edge(src_b1, None, t_write, 'x', dace.Memlet('b[0]'))
    state.add_edge(t_write, 'y', sink_b, None, dace.Memlet('b[0]'))

    src_b2 = state.add_read('b')
    out_n = state.add_write('out')
    t_read = state.add_tasklet('tread', {'x'}, {'y'}, 'y = x * 2.0')
    state.add_edge(src_b2, None, t_read, 'x', dace.Memlet('b[0]'))
    state.add_edge(t_read, 'y', out_n, None, dace.Memlet('out[0]'))

    sdfg.validate()

    b_nodes_before = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'b']
    assert len(b_nodes_before) == 3

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})

    b_nodes_after = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'b']
    assert len(b_nodes_after) == 3, ('ArrayElimination merged source AccessNodes of a data container that is '
                                     'also written in the same state -- this destroys the topological ordering '
                                     'that keeps the read after the write')


def test_sink_merge_refuses_when_data_is_also_read():
    """Symmetric: state has two sink AccessNodes (``out_degree == 0``) for
    ``a`` plus one source AccessNode (``out_degree > 0``) for ``a``. The
    source's read must stay between the two writes in execution order;
    folding the two sinks removes that ordering.
    """
    from dace.sdfg import nodes

    sdfg = dace.SDFG('sink_merge_with_read')
    sdfg.add_array('a', [10], dace.float64)
    sdfg.add_array('b', [10], dace.float64)
    sdfg.add_array('c', [10], dace.float64)
    state = sdfg.add_state()

    src_b1 = state.add_read('b')
    sink_a1 = state.add_write('a')
    t_w1 = state.add_tasklet('tw1', {'x'}, {'y'}, 'y = x + 1.0')
    state.add_edge(src_b1, None, t_w1, 'x', dace.Memlet('b[0]'))
    state.add_edge(t_w1, 'y', sink_a1, None, dace.Memlet('a[0]'))

    src_a = state.add_read('a')
    sink_c = state.add_write('c')
    t_r = state.add_tasklet('tr', {'x'}, {'y'}, 'y = x * 2.0')
    state.add_edge(src_a, None, t_r, 'x', dace.Memlet('a[0]'))
    state.add_edge(t_r, 'y', sink_c, None, dace.Memlet('c[0]'))

    src_b2 = state.add_read('b')
    sink_a2 = state.add_write('a')
    t_w2 = state.add_tasklet('tw2', {'x'}, {'y'}, 'y = x - 1.0')
    state.add_edge(src_b2, None, t_w2, 'x', dace.Memlet('b[0]'))
    state.add_edge(t_w2, 'y', sink_a2, None, dace.Memlet('a[0]'))

    sdfg.validate()

    a_nodes_before = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'a']
    assert len(a_nodes_before) == 3

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})

    a_nodes_after = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'a']
    assert len(a_nodes_after) == 3, ('ArrayElimination merged sink AccessNodes of a data container that is '
                                     'also read in the same state -- this destroys the topological ordering '
                                     'that keeps the read between the two writes')


def test_source_merge_allowed_when_write_is_in_another_state():
    """Cross-state case: state S1 holds two source AccessNodes for ``b`` and
    no write; a separate state S2 (successor in the control-flow graph)
    writes ``b``. Within S1 the source-merge is SAFE -- the control-flow
    edge from S1 to S2 enforces ordering between the reads and the write.
    The refusal must be intra-state only, not over-eager.
    """
    from dace.sdfg import nodes

    sdfg = dace.SDFG('cross_state_safe_merge')
    sdfg.add_array('b', [10], dace.float64)
    sdfg.add_array('out1', [10], dace.float64)
    sdfg.add_array('out2', [10], dace.float64)

    s1 = sdfg.add_state('read_only')
    src_b1 = s1.add_read('b')
    src_b2 = s1.add_read('b')
    o1 = s1.add_write('out1')
    o2 = s1.add_write('out2')
    t1 = s1.add_tasklet('t1', {'x'}, {'y'}, 'y = x + 1.0')
    t2 = s1.add_tasklet('t2', {'x'}, {'y'}, 'y = x * 2.0')
    s1.add_edge(src_b1, None, t1, 'x', dace.Memlet('b[0]'))
    s1.add_edge(t1, 'y', o1, None, dace.Memlet('out1[0]'))
    s1.add_edge(src_b2, None, t2, 'x', dace.Memlet('b[0]'))
    s1.add_edge(t2, 'y', o2, None, dace.Memlet('out2[0]'))

    s2 = sdfg.add_state('write')
    sink_b = s2.add_write('b')
    src_o1 = s2.add_read('out1')
    t3 = s2.add_tasklet('t3', {'x'}, {'y'}, 'y = x - 1.0')
    s2.add_edge(src_o1, None, t3, 'x', dace.Memlet('out1[0]'))
    s2.add_edge(t3, 'y', sink_b, None, dace.Memlet('b[0]'))

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.validate()

    b_in_s1_before = [n for n in s1.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'b']
    assert len(b_in_s1_before) == 2

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})

    b_in_s1_after = [n for n in s1.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'b']
    assert len(b_in_s1_after) == 1, ('ArrayElimination should merge the two source b AccessNodes in S1 -- '
                                     'the write to b lives in a separate state and the inter-state edge '
                                     'enforces the ordering, so the merge is safe')


def test_source_merge_preserves_carrier_raw_order_on_sibling_transient():
    """Regression for TSVC s254. Source-merge of two ``b`` source AccessNodes
    in the loop body destroys the implicit ordering between the *compute*
    chain (which reads ``b`` then reads the carried scalar ``x``) and the
    *seed-write* chain (which reads ``b`` then writes the next ``x``). After
    the merge, the codegen DFS visits the merged source's out-edges in
    insertion order, schedules the seed-write FIRST, and the subsequent
    compute reads the NEW ``x`` instead of the carried old value.

    The Phase 2.4a non-transient guard catches this when the merged source's
    own container is written in the state (``acc[c]`` in s243's body); it
    misses this case because the read+write pair lives on a DIFFERENT
    transient (``x``) while the merge target is ``b`` (read-only in the
    state).
    """
    import numpy as np
    from tests.corpus.tsvc.tsvc import s254_d_single
    from dace.transformation.passes.canonicalize.pipeline import _build_stages

    sdfg = s254_d_single.to_sdfg(simplify=True)
    # Drive the SDFG to the state right BEFORE the final SimplifyPass enters
    # ArrayElimination -- the agent's bisect proved ArrayElimination is the
    # specific Simplify sub-pass that breaks s254.
    for label, pass_obj in _build_stages():
        if label == 'end':
            break
        if hasattr(pass_obj, 'apply_pass'):
            try:
                pass_obj.apply_pass(sdfg, {})
            except Exception:
                pass
    # Run every Simplify sub-pass EXCEPT ArrayElimination -- proves the SDFG
    # at this point is correct.
    from dace.transformation.passes.simplify import SimplifyPass
    SimplifyPass(skip={'ArrayElimination'}).apply_pass(sdfg, {})
    sdfg.validate()

    n = 32
    rng = np.random.default_rng(254)
    a0 = rng.random(n)
    b = rng.random(n)
    a_exp = a0.copy()
    x = b[n - 1]
    for i in range(n):
        a_exp[i] = (b[i] + x) * 0.5
        x = b[i]

    sa = a0.copy()
    sdfg(a=sa, b=b.copy(), LEN_1D=n)
    assert np.allclose(sa, a_exp), 'baseline (no ArrayElimination) must compute s254 correctly'

    # Now run ArrayElimination in isolation. If it preserves the carrier-RAW
    # ordering on the sibling transient ``x``, the kernel still computes the
    # right values. The pre-fix behaviour is divergence -- the seed-write
    # chain races ahead of the compute.
    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    sdfg.validate()

    sa = a0.copy()
    sdfg(a=sa, b=b.copy(), LEN_1D=n)
    assert np.allclose(sa, a_exp), ('ArrayElimination broke the carrier RAW order: the two ``b`` source '
                                    'AccessNodes were merged, destroying the implicit ordering between '
                                    'the compute chain (reads OLD x) and the seed-write chain '
                                    '(writes NEW x). Fix must extend the Phase 2.4a refusal to also '
                                    'check sibling transients with both source AND sink in the same state.')


if __name__ == '__main__':
    test_redundant_simple()
    test_merge_simple()
    test_source_merge_refuses_when_data_is_also_written()
    test_sink_merge_refuses_when_data_is_also_read()
    test_source_merge_allowed_when_write_is_in_another_state()
