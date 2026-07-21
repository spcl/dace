# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vectorizer must refuse a map running several array recurrences at once.

The tile widener lane-privatises a single carried array sweep correctly (trmm's
triangular ``B[i]=f(B[k<i])``), but not several distinct sweeps in one body: the
per-lane tiles alias and the lanes race. That is the ADI pattern (one row map
forward/back-sweeping ``p``, ``q``, ``v``). Rather than emit the race, the
vectorizer refuses and leaves the kernel un-tiled and correct.
"""
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.utils.pass_invariants import no_multi_array_carried_sweep_in_map

N = 16


def _row_sweep_loop(inner: dace.SDFG, arr: str, tag: str) -> LoopRegion:
    """Build ``for j in [1, N): arr[i, j] = arr[i, j] + arr[i, j-1]`` as a loop."""
    loop = LoopRegion(tag, f'j < {N}', 'j', 'j = 1', 'j = j + 1', sdfg=inner)
    inner.add_node(loop)
    body = loop.add_state('body', is_start_block=True)
    r = body.add_access(arr)
    w = body.add_access(arr)
    t = body.add_tasklet(f'{tag}_t', {'cur', 'prev'}, {'out'}, 'out = cur + prev')
    body.add_edge(r, None, t, 'cur', dace.Memlet(f'{arr}[i, j]'))
    body.add_edge(r, None, t, 'prev', dace.Memlet(f'{arr}[i, j - 1]'))
    body.add_edge(t, 'out', w, None, dace.Memlet(f'{arr}[i, j]'))
    return loop


def _multi_sweep(arrays):
    """A row map whose body NSDFG sweeps each of ``arrays`` along j in sequence."""
    sdfg = dace.SDFG('multi_sweep')
    for a in arrays:
        sdfg.add_array(a, [N, N], dace.float64)
    st = sdfg.add_state('main', is_start_block=True)

    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    for a in arrays:
        inner.add_array(a, [N, N], dace.float64)
    prev = None
    for k, a in enumerate(arrays):
        loop = _row_sweep_loop(inner, a, f'sweep_{a}')
        if prev is None:
            inner.start_block = inner.node_id(loop)
        else:
            inner.add_edge(prev, loop, dace.InterstateEdge())
        prev = loop

    me, mx = st.add_map('row', dict(i=f'0:{N}'))
    ns = st.add_nested_sdfg(inner, set(arrays), set(arrays), symbol_mapping={'i': 'i'})
    for a in arrays:
        st.add_memlet_path(st.add_access(a), me, ns, dst_conn=a, memlet=dace.Memlet(f'{a}[0:{N}, 0:{N}]'))
        st.add_memlet_path(ns, mx, st.add_access(a), src_conn=a, memlet=dace.Memlet(f'{a}[0:{N}, 0:{N}]'))
    return sdfg


def test_invariant_flags_two_sweeps_not_one():
    """The detector fires on two distinct array sweeps, not on a single one."""
    assert no_multi_array_carried_sweep_in_map(_multi_sweep(['p', 'q'])) is not None
    assert no_multi_array_carried_sweep_in_map(_multi_sweep(['p'])) is None


def test_invariant_flags_three_sweeps():
    """The ADI arity (p, q, v)."""
    assert no_multi_array_carried_sweep_in_map(_multi_sweep(['p', 'q', 'v'])) is not None


def test_real_adi_is_refused_and_correct():
    """End-to-end on the real polybench ADI kernel: canon+vec must refuse it
    (the multi-sweep shape) and leave a numerically correct SDFG.

    Skipped if the corpus harness is unavailable in this environment.
    """
    import copy
    import numpy as np
    pytest = __import__('pytest')
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target
    from dace.transformation.passes.vectorization.config import VectorizeConfig
    from dace.transformation.passes.vectorization.enums import ISA
    from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

    base, checker = mp.CORPORA['poly'][1]('adi')
    sd = copy.deepcopy(base)
    canonicalize(sd, validate=True, validate_all=False, **mp.cpu_params(4))
    assert no_multi_array_carried_sweep_in_map(sd) is not None, 'adi should present the multi-sweep shape'
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sd, {})
    fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
    fin.name = 'adi_refusal_test'
    assert bool(checker(fin)), 'adi must be value-correct after the refusal'


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-v'])
