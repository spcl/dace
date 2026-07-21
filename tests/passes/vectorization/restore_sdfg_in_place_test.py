# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``restore_sdfg_in_place`` must re-point blocks at EVERY control-flow nesting level.

Rolling a refused vectorization back copies the snapshot's contents onto the caller's SDFG object.
Fixing only the top-level blocks leaves states inside a ``LoopRegion`` / ``ConditionalBlock``
pointing at the throwaway source. Nothing fails immediately -- the source is an equivalent graph --
but ``ControlFlowBlock.__deepcopy__`` keeps ``_sdfg`` only when the owner is already in the copy's
``memo``, so the next ``deepcopy`` silently sets those states' ``sdfg`` to ``None``. Type inference
on the copy then dies with ``'NoneType' object has no attribute 'arrays'`` (polybench lu /
gramschmidt, whose WCR bodies take the refusal path).
"""
import copy

import dace
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.vectorize_multi_dim import restore_sdfg_in_place

N = 8


def _nested_loop_sdfg(name: str) -> dace.SDFG:
    """``for j in [0, N): A[j] = A[j] * 2`` -- a state nested inside a LoopRegion."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    loop = LoopRegion('sweep', f'j < {N}', 'j', 'j = 0', 'j = j + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('scale', {'cur'}, {'out'}, 'out = cur * 2')
    body.add_edge(body.add_access('A'), None, t, 'cur', dace.Memlet('A[j]'))
    body.add_edge(t, 'out', body.add_access('A'), None, dace.Memlet('A[j]'))
    return sdfg


def _states_with_no_owner(sdfg: dace.SDFG):
    return [st.label for sub in sdfg.all_sdfgs_recursive() for st in sub.states() if st.sdfg is None]


def test_nested_states_are_repointed_at_the_target():
    """Every block, at every nesting level, must own-point at ``target`` -- not at ``source``."""
    target = _nested_loop_sdfg('target')
    source = _nested_loop_sdfg('source')
    restore_sdfg_in_place(target, source)

    for sub in target.all_sdfgs_recursive():
        for state in sub.states():
            assert state.sdfg is sub, f'{state.label} points at {state.sdfg} instead of {sub}'
    assert target.name == 'source', 'the restore must adopt the snapshot contents'


def test_restored_sdfg_survives_a_deep_copy():
    """The regression itself: the stale owner only shows up one ``deepcopy`` later."""
    target = _nested_loop_sdfg('target')
    restore_sdfg_in_place(target, _nested_loop_sdfg('source'))

    assert _states_with_no_owner(target) == []
    assert _states_with_no_owner(copy.deepcopy(target)) == []


@pytest.mark.parametrize('name', ['lu', 'gramschmidt'])
def test_refused_kernels_finalize(name):
    """End-to-end: the polybench kernels that take the WCR refusal path finalize and are correct.

    Skipped if the corpus harness is unavailable in this environment.
    """
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    base, checker = mp.CORPORA['poly'][1](name)
    sd = copy.deepcopy(base)
    mp.apply_config(sd, 'canon+vec', mp.cpu_params(4))
    fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
    fin.name = f'{name}_restore_test'
    assert bool(checker(fin)), f'{name} must be value-correct after the refusal'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
