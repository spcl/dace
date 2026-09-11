# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""":class:`~dace.libraries.standard.nodes.find_first.FindFirst` on the host.

The CPU counterpart of :mod:`tests.library.detect_cudatest`: the node is built directly, so what
is pinned here is the library node's own contract and the runtime primitive it calls
(``dace::find_first_index`` in ``dace/runtime/include/dace/detect.h``), independent of any pass
that emits one.
"""
import os
import pathlib

import numpy as np
import pytest

import dace
from dace.transformation.layout.isolation import set_openmp_thread_count
from dace.libraries.standard.nodes import FindFirst
from dace.libraries.standard.nodes.find_first import INDEX_NAME, OUTPUT_CONNECTOR_NAME

N = dace.symbol('N')


def make_find_first_sdfg(name: str, implementation: str) -> dace.SDFG:
    """``out[0] = first i with a[i] < 0``, or ``N`` when the predicate never fires."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [1], dace.int64)
    state = sdfg.add_state('search', is_start_block=True)

    node = FindFirst('search', predicate=f'__r_a[{INDEX_NAME}] < 0.0', begin=0, end=N)
    node.implementation = implementation
    state.add_node(node)
    node.add_in_connector('__r_a')
    state.add_edge(state.add_read('a'), None, node, '__r_a', dace.Memlet('a[0:N]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def detect_header_text() -> str:
    """The runtime header the search lowers to, located from the installed package."""
    return (pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'detect.h').read_text()


def test_find_first_answer_is_a_reduction_not_the_shared_hint():
    """The answer and the cancellation hint must be two different variables.

    Folding them into one shared word makes the update a read-compare-write, which is not atomic
    as a whole: a thread that found a SMALLER index can have its write overwritten by a thread
    that found a larger one, and the search then answers a firing index that is not the first.
    The hint may lose updates (it only prunes); the answer may not, so it is a reduction. This is
    a source assertion because the numeric symptom needs a loaded machine to appear."""
    text = detect_header_text()
    body = text.split('inline long long find_first_index(')[1]
    chunk_pragma = next(ln for ln in body.splitlines() if 'omp parallel for schedule' in ln)
    assert 'reduction(min : best)' in chunk_pragma, (f'the chunk loop must reduce the answer, got {chunk_pragma!r}')
    # The schedule KIND is not pinned: guided and block static are measurably slower on a hit
    # early in a large range, but retuning that choice is a performance decision, not a
    # correctness regression. The line selector above already requires a schedule clause.
    assert 'return best;' in body, 'find_first_index must return the reduction, never the shared hint'
    assert 'return hint;' not in body, 'returning the raced hint loses updates under load'


def restore_omp_threads(previous: str | None) -> None:
    """Put ``OMP_NUM_THREADS`` -- and the loaded runtime's own count -- back as they were."""
    if previous is None:
        os.environ.pop('OMP_NUM_THREADS', None)
    elif previous.isdigit():
        set_openmp_thread_count(int(previous))
    else:
        os.environ['OMP_NUM_THREADS'] = previous


@pytest.mark.parametrize('implementation', ['pure', 'OpenMP'])
def test_find_first_is_exact_when_most_indices_fire(implementation):
    """Numeric guard for the same bug: with a dense firing tail every chunk finds something and
    they all race to publish, so a lost update answers too large an index. Repeated, because a
    race needs room to show."""
    sdfg = make_find_first_sdfg(f'find_first_dense_{implementation.lower()}', implementation)
    csdfg = sdfg.compile()

    previous = os.environ.get('OMP_NUM_THREADS')
    # libgomp caches OMP_NUM_THREADS at initialisation, so the count this process actually runs
    # with is the runtime's, not the environment's. On a team of one the publish race cannot
    # happen at all and every trial below passes for the wrong reason.
    assert set_openmp_thread_count(4), 'the publish race needs a real multi-thread team'
    n = 4096
    try:
        for trial in range(16):
            first = 1 + 7 * trial
            a = np.ones(n)
            a[first:] = -1.0  # every index from ``first`` on fires
            out = np.zeros(1, dtype=np.int64)
            csdfg(a=a, out=out, N=n)
            assert out[0] == first, (f'dense-firing trial {trial}: the search answered {out[0]}, not the first '
                                     f'firing index {first}')
    finally:
        restore_omp_threads(previous)


@pytest.mark.parametrize('implementation', ['pure', 'OpenMP'])
@pytest.mark.parametrize('first', [0, 1, 517, 4095, None])
def test_find_first_answers_every_firing_position(implementation, first):
    """Every firing position, and the no-hit case whose answer is the exclusive end -- the one a
    search that forgot its sentinel gets wrong."""
    sdfg = make_find_first_sdfg(f'find_first_pos_{implementation.lower()}', implementation)
    csdfg = sdfg.compile()

    n = 4096
    a = np.ones(n)
    if first is not None:
        a[first] = -1.0
    out = np.zeros(1, dtype=np.int64)
    csdfg(a=a, out=out, N=n)
    assert out[0] == (n if first is None else first)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
