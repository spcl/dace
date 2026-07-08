# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ParallelizeUnderConstraint: specialize a symbolic-stride loop into
``if cond: parallel-Map else: sequential-loop``.

A symbolic-stride write is data-parallel iff its stride coefficient is nonzero. The
pass replaces the loop with a two-way conditional: true branch (guard ``coeff != 0``)
= loop lifted to a plain WCR-free Map, else branch = original sequential loop -- a
violating value still computes on the fallback, never a trap. Two write shapes qualify:
in-place RMW TSVC ``s171`` (``a[i*inc] = a[i*inc] + b[i]``) and the injective store
(``dst[i*S] = src[i]*scale``, TSVC-2.5 ``ext_strided_store_ssym``). A loop-carried
recurrence (array read at a *different* subset) is excluded.
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
from dace.sdfg.state import ConditionalBlock, LoopRegion
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
    ``ext_strided_store_ssym``): injective (data-parallel) iff ``S != 0``. Not RMW;
    ``dst`` written, never read."""
    for i in range(N):
        dst[i * S] = src[i] * scale


def _has_map(region):
    return any(isinstance(n, nodes.MapEntry) for n, _ in region.all_nodes_recursive())


def _sequential_loops(region):
    return [
        r for r in region.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable
    ]


def _specialize_conditional(sdfg):
    """The single ``if cond: parallel else: sequential`` conditional this pass emits, as
    ``(condition_string, parallel_region, sequential_region)``. First branch holds a Map;
    else branch (condition ``None``) keeps a sequential loop."""
    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f'expected exactly one specialization conditional, got {len(cbs)}'
    branches = cbs[0].branches
    assert len(branches) == 2, f'expected an if/else (2 branches), got {len(branches)}'
    (cond, par_region), (else_cond, seq_region) = branches
    assert cond is not None and else_cond is None, 'branches must be (cond -> parallel, else -> sequential)'
    return cond.as_string, par_region, seq_region


def _single_state_stride_loop(read_subset):
    """One-state ``for i`` loop writing ``a[S*i]`` and (when ``read_subset`` given)
    reading ``a[read_subset]`` -- minimal fixture for
    :meth:`ParallelizeUnderConstraint._symbolic_stride_condition`.

    ``read_subset=None`` = plain store, ``'S*i'`` = in-place RMW, ``'S*i - 1'`` =
    loop-carried recurrence.
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


def test_symbolic_stride_specializes_if_par_else_seq():
    sdfg = affine_stride_rmw.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    cond, par_region, seq_region = _specialize_conditional(sdfg)
    assert 'inc' in cond and '!= 0' in cond, f'true branch must be guarded by inc != 0: {cond!r}'
    assert _has_map(par_region), 'the inc != 0 branch must be the loop lifted to a Map'
    assert _sequential_loops(seq_region), 'the else branch must keep the original sequential loop'
    # The fallback loop is pinned so no later parallelizer lifts it back to a Map.
    assert all(l.pinned_sequential for l in _sequential_loops(seq_region))
    assert all(e.data.wcr is None for s in par_region.all_states() for e in s.edges()), \
        'the parallel-branch Map must carry no WCR'


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


def test_symbolic_stride_condition_matches_store_and_rmw_excludes_recurrence():
    """The structural matcher admits an injective symbolic-stride write -- a plain
    store *or* a same-subset read-modify-write -- and refuses a loop-carried
    recurrence (the array read at a subset other than the one written)."""
    inst = ParallelizeUnderConstraint()

    store_sdfg, store_loop = _single_state_stride_loop(read_subset=None)
    assert inst._symbolic_stride_condition(store_loop, store_sdfg) == '(S) != 0', \
        'a plain symbolic-stride store is parallel iff S != 0'

    rmw_sdfg, rmw_loop = _single_state_stride_loop(read_subset='S*i')
    assert inst._symbolic_stride_condition(rmw_loop, rmw_sdfg) == '(S) != 0', \
        'a same-subset read-modify-write (s171) is parallel iff S != 0'

    rec_sdfg, rec_loop = _single_state_stride_loop(read_subset='S*i - 1')
    assert inst._symbolic_stride_condition(rec_loop, rec_sdfg) is None, \
        'a loop-carried recurrence (read at a different subset) must be excluded'


def test_symbolic_stride_store_specializes_if_par_else_seq():
    sdfg = symbolic_stride_store.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    cond, par_region, seq_region = _specialize_conditional(sdfg)
    assert 'S' in cond and '!= 0' in cond, f'true branch must be guarded by S != 0: {cond!r}'
    assert _has_map(par_region), 'the S != 0 branch must lift the store to a Map'
    assert _sequential_loops(seq_region), 'the else branch must keep the original sequential loop'
    assert all(l.pinned_sequential for l in _sequential_loops(seq_region))
    assert all(e.data.wcr is None for s in par_region.all_states() for e in s.edges()), \
        'the parallel-branch Map must carry no WCR'


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
    test_symbolic_stride_specializes_if_par_else_seq()
    test_value_preserving_under_nonzero_stride()
    test_symbolic_stride_condition_matches_store_and_rmw_excludes_recurrence()
    test_symbolic_stride_store_specializes_if_par_else_seq()
    test_value_preserving_symbolic_stride_store()
