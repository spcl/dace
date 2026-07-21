# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.loop_peeling import LoopPeeling


@dace.program
def tounroll(A: dace.float64[20], B: dace.float64[20]):
    for i in range(5):
        for j in dace.map[0:20]:
            with dace.tasklet:
                a << A[j]
                b_in << B[j]
                b_out >> B[j]
                b_out = b_in + a * i


def regression(A, B):
    result = np.zeros_like(B)
    result[:] = B
    for i in range(5):
        result += A * i
    return result


def test_unroll():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopUnroll)
    # LoopUnroll prepends an empty predecessor state when the unrolled loop is the graph's own
    # start block (avoids an ambiguous start block after unrolling) -- +1 past the 5*2 body nodes.
    assert len(sdfg.nodes()) == 5 * 2 + 1
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


def test_peeling_start():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopPeeling, dict(count=2))
    assert len(sdfg.nodes()) == 3
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


def test_peeling_end():
    sdfg: dace.SDFG = tounroll.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    sdfg.apply_transformations(LoopPeeling, dict(count=2, begin=False))
    assert len(sdfg.nodes()) == 3
    A = np.random.rand(20)
    B = np.random.rand(20)
    reg = regression(A, B)

    # HACK: Workaround to deal with bug in frontend (See PR #161)
    if 'i' in sdfg.symbols:
        del sdfg.symbols['i']

    sdfg(A=A, B=B)
    assert np.allclose(B, reg)


def test_peeling_end_no_loop_symbol_leak():
    """Back-peeling must anchor each peeled iteration on the concrete loop end
    (``end``, ``end - stride``, ...) rather than the loop variable. Otherwise the
    loop-defined iteration symbol stays live in the peeled-after region, which
    blocks downstream LoopToMap. Peel a symbolic-bound loop and assert the loop
    variable does not appear in any peeled-after interstate edge or block."""
    N = dace.symbol('N')

    @dace.program
    def symbolic_loop(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            A[i] = B[i] * 2.0

    sdfg: dace.SDFG = symbolic_loop.to_sdfg(simplify=True)
    loop = next(n for n in sdfg.nodes() if isinstance(n, dace.sdfg.state.LoopRegion))
    loop_var = loop.loop_variable
    # Symbolic-bound loops are rejected by LoopUnroll's constant-size gate, so peel
    # directly with ``verify=False`` (the same path BestEffortLoopPeeling uses), and
    # keep the peeled iterations as distinct regions so the symbol-leak contract is
    # inspectable.
    LoopPeeling().apply_to(sdfg=sdfg,
                           loop=loop,
                           verify=False,
                           options={
                               'count': 2,
                               'begin': False,
                               'inline_iterations': False
                           })

    # ``LoopPeeling`` now emits each peeled iter as either a ``ControlFlowRegion``
    # (multi-state body) or a flat ``SDFGState`` directly in the parent graph
    # (single-state body, the case here). Either shape must hold the
    # no-loop-symbol-leak contract.
    peeled = [
        n for n in sdfg.nodes()
        if isinstance(n, (dace.sdfg.state.ControlFlowRegion, dace.sdfg.state.SDFGState)) and n is not loop
    ]
    assert peeled, 'back-peel must produce peeled-after regions or states'

    def _iter_interstate_edges(block):
        if isinstance(block, dace.sdfg.state.ControlFlowRegion):
            yield from block.all_interstate_edges()

    def _iter_states(block):
        if isinstance(block, dace.sdfg.state.ControlFlowRegion):
            yield from block.all_states()
        elif isinstance(block, dace.sdfg.state.SDFGState):
            yield block

    for region in peeled:
        for edge in _iter_interstate_edges(region):
            assert loop_var not in edge.data.free_symbols, \
                f'peeled region {region.label} leaks loop symbol {loop_var}'
        for state in _iter_states(region):
            for e in state.edges():
                if e.data.data is not None:
                    assert loop_var not in set(map(str, e.data.subset.free_symbols)), \
                        f'peeled subset in {region.label} leaks loop symbol {loop_var}'

    A = np.random.rand(8)
    B = np.random.rand(8)
    ref = B * 2.0
    sdfg(A=A, B=B, N=8)
    assert np.allclose(A, ref)


def test_peeling_single_state_body_emits_flat_states():
    """When the loop body is a SINGLE ``SDFGState``, ``LoopPeeling`` must
    emit each peeled iteration as a flat ``SDFGState`` directly in the
    parent graph (no wrapping ``ControlFlowRegion``). Downstream
    ``StateFusionExtended`` handles fusion between the peeled state and
    the remainder via interstate edges, the same way it handles any other
    adjacent state pair."""
    N = dace.symbol('N')

    @dace.program
    def single_state_body(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            B[i] = A[i] * 2.0

    sdfg = single_state_body.to_sdfg(simplify=True)
    loop = next(n for n in sdfg.nodes() if isinstance(n, dace.sdfg.state.LoopRegion))
    LoopPeeling().apply_to(sdfg=sdfg, loop=loop, verify=False, options={'count': 2, 'begin': True})

    peeled_states = [
        n for n in sdfg.nodes() if isinstance(n, dace.sdfg.state.SDFGState)
        and not isinstance(n, dace.sdfg.state.ControlFlowRegion) and n.label.startswith(loop.label + '_')
    ]
    assert len(peeled_states) == 2, f'expected 2 flat peeled SDFGStates, got {len(peeled_states)}'

    A = np.arange(8, dtype=np.float64) + 0.5
    B = np.zeros(8, dtype=np.float64)
    sdfg(A=A, B=B, N=8)
    assert np.allclose(B, A * 2.0)


def test_peeling_multi_state_body_emits_cfr_with_deepcopied_edges():
    """When the loop body has MULTIPLE states (interstate edges inside the
    body), ``LoopPeeling`` must emit each peeled iteration as a
    ``ControlFlowRegion`` (the body's internal interstate edges live with
    the iter). Cloning uses ``copy.deepcopy`` on each block and an
    ``{old: new}`` map to remap edge endpoints; the original loop body and
    the peeled-iter clones must remain disjoint graph objects."""
    sdfg = dace.SDFG('multi_state_body_peel')
    sdfg.add_symbol('N', dace.int64)
    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('t', [N], dace.float64)

    pre = sdfg.add_state('pre', is_start_block=True)
    loop = dace.sdfg.state.LoopRegion('peelme',
                                      condition_expr='i < N',
                                      loop_var='i',
                                      initialize_expr='i = 0',
                                      update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    body1 = loop.add_state('body1', is_start_block=True)
    body2 = loop.add_state('body2')
    loop.add_edge(body1, body2, dace.InterstateEdge())
    a_r = body1.add_read('A')
    t_w = body1.add_write('t')
    cp = body1.add_tasklet('cp1', {'_in'}, {'_out'}, '_out = _in + 1.0')
    body1.add_edge(a_r, None, cp, '_in', dace.Memlet('A[i]'))
    body1.add_edge(cp, '_out', t_w, None, dace.Memlet('t[i]'))
    t_r = body2.add_read('t')
    b_w = body2.add_write('B')
    cp2 = body2.add_tasklet('cp2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    body2.add_edge(t_r, None, cp2, '_in', dace.Memlet('t[i]'))
    body2.add_edge(cp2, '_out', b_w, None, dace.Memlet('B[i]'))
    sdfg.validate()

    LoopPeeling().apply_to(sdfg=sdfg, loop=loop, verify=False, options={'count': 2, 'begin': True})

    peeled_cfrs = [
        n for n in sdfg.nodes()
        if isinstance(n, dace.sdfg.state.ControlFlowRegion) and n is not loop and n.label.startswith(loop.label + '_')
    ]
    assert len(peeled_cfrs) == 2, f'expected 2 peeled CFRs for multi-state body, got {len(peeled_cfrs)}'
    for cfr in peeled_cfrs:
        body_states = [n for n in cfr.nodes() if isinstance(n, dace.sdfg.state.SDFGState)]
        assert len(body_states) == 2, f'multi-state body must yield 2 states per peeled iter, got {len(body_states)}'
        # And no node identity is shared with the original loop body.
        loop_body_state_ids = {id(b) for b in loop.nodes()}
        for s in body_states:
            assert id(s) not in loop_body_state_ids, 'peeled iter must be a deepcopy, not aliased'

    n = 6
    A = np.arange(n, dtype=np.float64) + 1.0
    B = np.zeros(n, dtype=np.float64)
    sdfg(A=A, B=B, N=n)
    # B[i] = (A[i] + 1) * 2 for each i
    assert np.allclose(B, (A + 1.0) * 2.0)


if __name__ == '__main__':
    test_unroll()
    test_peeling_start()
    test_peeling_end()
    test_peeling_end_no_loop_symbol_leak()
    test_peeling_single_state_body_emits_flat_states()
    test_peeling_multi_state_body_emits_cfr_with_deepcopied_edges()
