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
    assert len(sdfg.nodes()) == 5 * 2
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
                           options={'count': 2, 'begin': False, 'inline_iterations': False})

    # The peeled iterations are the control-flow regions added after the remainder loop.
    peeled = [n for n in sdfg.nodes() if isinstance(n, dace.sdfg.state.ControlFlowRegion) and n is not loop]
    assert peeled, 'back-peel must produce peeled-after regions'
    for region in peeled:
        for edge in region.all_interstate_edges():
            assert loop_var not in edge.data.free_symbols, \
                f'peeled region {region.label} leaks loop symbol {loop_var}'
        for state in region.all_states():
            for e in state.edges():
                if e.data.data is not None:
                    assert loop_var not in set(map(str, e.data.subset.free_symbols)), \
                        f'peeled subset in {region.label} leaks loop symbol {loop_var}'

    A = np.random.rand(8)
    B = np.random.rand(8)
    ref = B * 2.0
    sdfg(A=A, B=B, N=8)
    assert np.allclose(A, ref)


if __name__ == '__main__':
    test_unroll()
    test_peeling_start()
    test_peeling_end()
    test_peeling_end_no_loop_symbol_leak()
