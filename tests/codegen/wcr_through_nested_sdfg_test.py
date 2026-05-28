# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""WCR codegen must work when the WCR edge sources from a NestedSDFG, not a Tasklet.

Some transformation pipelines (e.g. vectorization's "normalize the map body into a
NestedSDFG") move the per-iteration computation inside a NestedSDFG before the Map
exits. The WCR edge that previously ran ``tasklet -> MapExit -> acc[c]`` now runs
``NSDFG -> MapExit -> acc[c]`` -- same conflict resolution semantics, different
upstream node class. The codegen must still emit the reduction (atomic or scalar)
correctly in that case; if it special-cases ``Tasklet`` as the only WCR source,
the NSDFG path silently degenerates to a plain store and the parallel result is
wrong.

The shape exercised here mirrors what ``AugAssignToWCR`` + a normalising pass
would leave behind: a parallel Map whose body is a small NestedSDFG that writes
to a scalar carry, and the carry value is funneled out via a WCR-sum edge into a
1-element accumulator in the outer SDFG.
"""
import numpy as np
import pytest

import dace


def _build_wcr_nsdfg_sdfg(n: int) -> dace.SDFG:
    """Build ``acc[0] = sum(src[i] for i in range(n))`` as a Map + WCR write whose
    source is a NestedSDFG (one scalar in, one scalar out per Map iteration)."""
    sdfg = dace.SDFG(f'wcr_through_nsdfg_n{n}')
    sdfg.add_array('src', [n], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    state = sdfg.add_state('map')
    src_read = state.add_read('src')
    acc_write = state.add_write('acc')

    # The body NestedSDFG: read ``_in`` (a scalar slice of ``src``) and produce ``_out``
    # (a scalar). The body's only computation is ``_out = _in``; the actual reduction
    # semantics live on the WCR edge from this NSDFG's output to the outer ``acc``.
    nsdfg = dace.SDFG('body')
    nsdfg.add_array('_in', [1], dace.float64)
    nsdfg.add_array('_out', [1], dace.float64)
    nstate = nsdfg.add_state('s0')
    nin = nstate.add_read('_in')
    nout = nstate.add_write('_out')
    nstate.add_nedge(nin, nout, dace.Memlet(data='_in', subset='0', other_subset='0'))

    map_entry, map_exit = state.add_map('outer', {'i': f'0:{n}'})
    nnode = state.add_nested_sdfg(nsdfg, {'_in'}, {'_out'})

    state.add_memlet_path(src_read, map_entry, nnode, dst_conn='_in',
                          memlet=dace.Memlet(data='src', subset='i'))
    # The WCR edges: from the NSDFG output, through MapExit, to the outer ``acc[0]``.
    # ``wcr='lambda a, b: a + b'`` should make the codegen emit a sum reduction --
    # set on BOTH the inner (NSDFG -> MapExit) and outer (MapExit -> AccessNode) edges
    # so propagation and codegen both see the WCR semantics regardless of where the
    # analysis picks the edge up.
    state.add_memlet_path(nnode, map_exit, acc_write, src_conn='_out',
                          memlet=dace.Memlet(data='acc', subset='0',
                                              wcr='lambda a, b: a + b'))

    sdfg.validate()
    return sdfg


@pytest.mark.xfail(
    reason='Codegen bug: WCR sum through a NestedSDFG source is silently dropped; the '
    'result is approximately the last iteration value, not the running sum. The '
    'WCR codegen path special-cases ``Tasklet`` as the upstream node class; when '
    'the upstream is a ``NestedSDFG`` (vectorization-pipeline-style normalised map '
    'body), the reduction is not emitted and the parallel result is wrong. This '
    'test is the regression target -- fixing the codegen should flip it to PASS.',
    strict=True,
)
def test_wcr_through_nested_sdfg_sum_reduction():
    """A WCR-sum edge whose source is a NestedSDFG (one Map iteration per body
    invocation) should accumulate correctly: ``acc[0] += sum(src)``.
    """
    n = 64
    rng = np.random.default_rng(0)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.zeros(1, dtype=np.float64)

    sdfg = _build_wcr_nsdfg_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], src.sum()), (
        f'WCR sum through NSDFG returned {acc[0]}, expected {src.sum()}.')


@pytest.mark.xfail(
    reason='Same WCR-through-NSDFG codegen bug as above; documents that the bug also '
    'breaks the live-in case (``acc[0] = 10.0`` before the Map runs).',
    strict=True,
)
def test_wcr_through_nested_sdfg_with_initial_value():
    """The pre-loop ``acc[0]`` value should be preserved (WCR accumulates *into* it)."""
    n = 32
    rng = np.random.default_rng(1)
    src = rng.uniform(-1.0, 1.0, size=n)
    acc = np.array([10.0])

    sdfg = _build_wcr_nsdfg_sdfg(n)
    sdfg(src=src, acc=acc)
    assert np.isclose(acc[0], 10.0 + src.sum())


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
