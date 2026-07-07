# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ParallelizeUnderConstraint: guard-and-parallelize a symbolic-stride loop.

A symbolic-stride write is data-parallel iff its stride coefficient is nonzero.
The pass should emit a runtime ``__builtin_trap()`` guard that fires on the
violated assumption (``coeff == 0``) and lift the loop to a plain (WCR-free) Map.
Two write shapes qualify: the in-place read-modify-write TSVC ``s171``
(``a[i*inc] = a[i*inc] + b[i]``) and the plain injective store
(``dst[i*S] = src[i]*scale``, TSVC-2.5 ``ext_strided_store_ssym``). A loop-carried
recurrence (the array read at a *different* subset) is excluded.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np

import dace
from dace import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.parallelize_under_constraint import ParallelizeUnderConstraint

N = dace.symbol('N')
S = dace.symbol('S')


@dace.program
def affine_stride_rmw(a: dace.float64[N], b: dace.float64[N], inc: dace.int64):
    for i in range(N):
        a[i * inc] = a[i * inc] + b[i]


@dace.program
def symbolic_stride_store(src: dace.float64[N], dst: dace.float64[S * N], scale: dace.float64):
    """Plain symbolic-stride store ``dst[i*S] = src[i]*scale`` (TSVC-2.5
    ``ext_strided_store_ssym``): injective -- hence data-parallel -- iff ``S != 0``.
    Not a read-modify-write; ``dst`` is written but never read."""
    for i in range(N):
        dst[i * S] = src[i] * scale


def _trap_tasklets(sdfg):
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and '__builtin_trap' in (n.code.as_string or '')
    ]


def _single_state_stride_loop(read_subset):
    """A one-state ``for i`` loop that writes ``a[S*i]`` and, when ``read_subset``
    is given, reads ``a[read_subset]`` -- the minimal fixture for exercising
    :meth:`ParallelizeUnderConstraint._symbolic_stride_violation` directly.

    ``read_subset=None`` is a plain store, ``'S*i'`` an in-place read-modify-write,
    ``'S*i - 1'`` a loop-carried recurrence.
    """
    sdfg = dace.SDFG('stride_loop')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    loop = LoopRegion('L', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    st = loop.add_state('body', is_start_block=True)
    write = st.add_write('a')
    if read_subset is None:
        src = st.add_read('b')
        tasklet = st.add_tasklet('w', {'r'}, {'o'}, 'o = r')
        st.add_edge(src, None, tasklet, 'r', Memlet('b[i]'))
    else:
        src = st.add_read('a')
        tasklet = st.add_tasklet('w', {'r'}, {'o'}, 'o = r + 1')
        st.add_edge(src, None, tasklet, 'r', Memlet(f'a[{read_subset}]'))
    st.add_edge(tasklet, 'o', write, None, Memlet('a[S*i]'))
    return sdfg, loop


def test_symbolic_stride_emits_trap_guard_and_parallel_map():
    sdfg = affine_stride_rmw.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1, f'expected exactly one runtime trap guard, got {len(traps)}'
    trap = traps[0]
    assert trap.side_effects, 'trap must be side-effecting so DCE does not prune it'
    assert 'inc' in trap.code.as_string, f'trap must check the constraint symbol inc: {trap.code.as_string!r}'

    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), 'loop should be lifted to a Map'
    assert not any(
        isinstance(cf, LoopRegion) and cf.loop_variable
        for cf in sdfg.all_control_flow_regions(recursive=True)), 'no sequential loop should remain'
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), 'the parallel Map must carry no WCR'


def test_value_preserving_under_nonzero_stride():
    # inc=1 keeps every i*inc in bounds (the realistic stride); the parallel Map
    # under the trap guard must reproduce the sequential in-place update.
    n = 64
    inc = 1
    rng = np.random.default_rng(0)
    a0, b = rng.standard_normal(n), rng.standard_normal(n)
    sdfg = affine_stride_rmw.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    got = a0.copy()
    sdfg(a=got, b=b, inc=inc, N=n)
    exp = a0 + b  # a[i*1] = a[i] + b[i] for all i
    assert np.allclose(got, exp), 'value mismatch at inc=1'


def test_symbolic_stride_violation_matches_store_and_rmw_excludes_recurrence():
    """The structural matcher admits an injective symbolic-stride write -- a plain
    store *or* a same-subset read-modify-write -- and refuses a loop-carried
    recurrence (the array read at a subset other than the one written)."""
    inst = ParallelizeUnderConstraint()

    store_sdfg, store_loop = _single_state_stride_loop(read_subset=None)
    assert inst._symbolic_stride_violation(store_loop, store_sdfg) == '(S) == 0', \
        'a plain symbolic-stride store must be guardable on S == 0'

    rmw_sdfg, rmw_loop = _single_state_stride_loop(read_subset='S*i')
    assert inst._symbolic_stride_violation(rmw_loop, rmw_sdfg) == '(S) == 0', \
        'a same-subset read-modify-write (s171) must be guardable on S == 0'

    rec_sdfg, rec_loop = _single_state_stride_loop(read_subset='S*i - 1')
    assert inst._symbolic_stride_violation(rec_loop, rec_sdfg) is None, \
        'a loop-carried recurrence (read at a different subset) must be excluded'


def test_symbolic_stride_store_emits_trap_guard_and_parallel_map():
    sdfg = symbolic_stride_store.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1, f'expected exactly one runtime trap guard, got {len(traps)}'
    trap = traps[0]
    assert trap.side_effects, 'trap must be side-effecting so DCE does not prune it'
    assert 'S' in trap.code.as_string, f'trap must check the stride symbol S: {trap.code.as_string!r}'

    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), \
        'the plain symbolic-stride store should be lifted to a Map'
    assert not any(
        isinstance(cf, LoopRegion) and cf.loop_variable
        for cf in sdfg.all_control_flow_regions(recursive=True)), 'no sequential loop should remain'
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), 'the parallel Map must carry no WCR'


def test_value_preserving_symbolic_stride_store():
    # S=3 keeps every i*S in bounds of dst[S*N]; the guarded parallel Map must
    # reproduce the sequential strided store bit-for-bit.
    n, s = 32, 3
    rng = np.random.default_rng(7)
    src = rng.standard_normal(n)
    scale = float(rng.standard_normal())
    sdfg = symbolic_stride_store.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    dst = np.zeros(s * n)
    sdfg(src=src, dst=dst, scale=scale, N=n, S=s)
    exp = np.zeros(s * n)
    exp[::s] = src * scale
    assert np.allclose(dst, exp), 'value mismatch on the symbolic-stride store'


if __name__ == '__main__':
    test_symbolic_stride_emits_trap_guard_and_parallel_map()
    test_value_preserving_under_nonzero_stride()
    test_symbolic_stride_violation_matches_store_and_rmw_excludes_recurrence()
    test_symbolic_stride_store_emits_trap_guard_and_parallel_map()
    test_value_preserving_symbolic_stride_store()
