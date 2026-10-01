# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live tests of iterating on a generated tasklet.

These call a real model provider, so they are marked ``ai`` and excluded from CI. Run them with::

    ANTHROPIC_API_KEY=... ./py314/bin/python -m pytest tests/library/ai/ai_iteration_test.py -m ai -v

Any configured provider works::

    DACE_ai_provider=responses OPENAI_API_KEY=... ...      # OpenAI Responses API
    DACE_ai_provider=manual ...  -s < /dev/null            # relay the prompt by hand

Add ``DACE_debugprint=verbose`` to watch the prompts, answers and probe compilations go by. Each
test makes **two** paid generations: one to produce the tasklet, one to revise it, and prints the
directory holding the conversation when it finishes (run with ``-s`` to see it).

Running them a second time costs nothing: answers are cached by prompt, so a re-run replays the
same conversation offline. Pass ``DACE_ai_cache=false`` to force fresh generations, which is what
you want when judging whether the prompts reliably produce good code rather than whether the
plumbing works.

What is asserted is behavior, not text: that the program still computes the right answer after a
revision, and that the specific thing the feedback asked for is present. The point is that the
second call *continued the conversation* rather than starting over.
"""

import platform

import numpy as np
import pytest

import dace
import dace.libraries.ai as ai
from dace import dtypes, nodes
from dace.libraries.ai.nodes import AINode
from dace.libraries.ai.sysinfo import cpu_has_feature

N = 64
TILE = 8

#: Named in the feedback and asserted afterwards. Distinctive enough that its presence can only
#: come from the revision request having been read.
HELPER = 'dace_ai_demo_helper'

SQUARE_DESCRIPTION = """
Compute _out = _in * _in + 1 for one double-precision element. Keep it simple.
""".strip()

SQUARE_FEEDBACK = f"""
The result is correct, so do not change what it computes.

Restructure it: move the arithmetic into a helper function named exactly `{HELPER}`, defined at
file scope in `code_global` with static linkage, and have the tasklet body call that helper instead
of doing the arithmetic inline.
""".strip()

NAIVE_TILE_DESCRIPTION = f"""
Compute one {TILE}x{TILE} output tile of a single-precision matrix multiplication:

    for i in 0..{TILE}, j in 0..{TILE}:
        _c[i][j] = sum over p in 0..{N} of _a[i][p] * _b[p][j]

Write it as a straightforward scalar triple loop. Do not use intrinsics.
""".strip()

VECTORIZE_FEEDBACK = f"""
This is correct but far too slow: it is a scalar triple loop, and it runs on a CPU with AVX2 and
FMA. Rewrite it as a register-blocked microkernel using AVX2 intrinsics from <immintrin.h>,
accumulating the {TILE}x{TILE} tile in YMM registers across the reduction dimension and storing it
once at the end. Use FMA instructions. Keep the result numerically equivalent.
""".strip()


@pytest.fixture(autouse=True)
def fresh_session(tmp_path, request):
    """
    Gives each run its own session directory, and says where it is.

    A session is keyed by SDFG and node name and is *resumed* when it already exists -- which is
    the point of the feature, and would otherwise leave these tests accumulating rounds in
    ``~/.dace/ai_sessions`` forever. The assertions do not depend on this: they measure the rounds
    each run adds, so they hold even when ``DACE_ai_session_dir`` overrides this fixture, which a
    DaCe environment variable always does (``dace/config.py`` consults the environment first).

    The answer cache is deliberately left alone, so re-running these tests still costs nothing.

    :param tmp_path: The pytest temporary directory.
    :param request: The test being run, used to name the directory.
    """
    directory = tmp_path / 'sessions'
    with dace.config.set_temporary('ai', 'sessions', value=True):
        with dace.config.set_temporary('ai', 'session_dir', value=str(directory)):
            yield
    print(f'\n[{request.node.name}] conversation written to {directory}')


def _square_sdfg() -> dace.SDFG:
    """
    Builds an elementwise map whose body is an :class:`AINode`.

    Deliberately trivial: this test is about the conversation continuing, not about the difficulty
    of the kernel, and it must run anywhere.

    :return: The SDFG.
    """
    sdfg = dace.SDFG('ai_iteration_square')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('elements', {'i': f'0:{N}'})
    node = AINode('square_plus_one', SQUARE_DESCRIPTION, inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_memlet_path(state.add_read('A'), entry, node, dst_conn='_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(node, exit_node, state.add_write('B'), src_conn='_out', memlet=dace.Memlet('B[i]'))
    return sdfg


def _tile_gemm_sdfg() -> dace.SDFG:
    """
    Builds a tiled matrix multiplication whose innermost kernel is an :class:`AINode`.

    :return: The SDFG.
    """
    sdfg = dace.SDFG('ai_iteration_gemm')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dace.float32)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('tiles', {
        'ti': f'0:{N}:{TILE}',
        'tj': f'0:{N}:{TILE}'
    },
                                     schedule=dtypes.ScheduleType.CPU_Multicore)
    node = AINode('gemm_tile', NAIVE_TILE_DESCRIPTION, inputs={'_a', '_b'}, outputs={'_c'})
    state.add_node(node)
    state.add_memlet_path(state.add_read('A'),
                          entry,
                          node,
                          dst_conn='_a',
                          memlet=dace.Memlet(f'A[ti:ti+{TILE}, 0:{N}]'))
    state.add_memlet_path(state.add_read('B'),
                          entry,
                          node,
                          dst_conn='_b',
                          memlet=dace.Memlet(f'B[0:{N}, tj:tj+{TILE}]'))
    state.add_memlet_path(node,
                          exit_node,
                          state.add_write('C'),
                          src_conn='_c',
                          memlet=dace.Memlet(f'C[ti:ti+{TILE}, tj:tj+{TILE}]'))
    return sdfg


def _tasklet(sdfg: dace.SDFG) -> nodes.Tasklet:
    """
    Returns the single tasklet of an expanded SDFG.

    :param sdfg: The SDFG to search.
    :return: The tasklet.
    """
    return next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.Tasklet))


@pytest.mark.ai
def test_feedback_revises_a_working_tasklet():
    """
    The core of iteration: revise code that already works, without breaking it.

    Two generations. The first produces a correct kernel; the second is told to restructure it in a
    specific, checkable way. Both must compute the same thing, and the session must show one
    conversation of two rounds rather than two unrelated questions.
    """
    sdfg = _square_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))

    node.expand(state, 'ai')
    first = _tasklet(sdfg).code.as_string
    # A session is resumed when it already exists, so the round this run starts from depends on
    # what earlier runs left behind. Assert on what *this* run adds.
    baseline = ai.sessions(sdfg)[0].round

    rng = np.random.default_rng(0)
    a = rng.random(N)
    b = np.zeros(N)
    sdfg(A=a, B=b)
    assert np.allclose(b, a * a + 1), 'the first generation did not compute the right thing'

    # --- round 2: same slot, same conversation, new requirement -------------------------------
    ai.refine(sdfg, 'square_plus_one', SQUARE_FEEDBACK)

    tasklet = _tasklet(sdfg)
    assert HELPER in tasklet.code_global.as_string, \
        f'the revision did not define {HELPER} at file scope; the feedback did not reach the model'
    assert HELPER in tasklet.code.as_string, f'the body does not call {HELPER}'
    assert tasklet.code.as_string != first, 'the code did not change at all'

    b2 = np.zeros(N)
    sdfg(A=a, B=b2)
    assert np.allclose(b2, a * a + 1), 'the revision broke a result that was correct before'

    # The slot is one lineage, not two independent generations: refining added exactly one round
    # to the conversation the expansion started.
    slot = ai.sessions(sdfg)[0]
    assert slot.round == baseline + 1, f'expected round {baseline + 1}, got {slot.round}'
    rounds = {r.number: r for r in ai.history(sdfg, 'square_plus_one')}
    assert set(rounds) >= {baseline, baseline + 1}
    assert rounds[baseline].feedback == '', 'the initial generation should carry no feedback'
    assert HELPER in rounds[baseline + 1].feedback, 'the round did not record what was asked of it'
    assert rounds[baseline].outcome == 'expanded' and rounds[baseline + 1].outcome == 'expanded'

    # --- rollback: free, and it must produce working code again -------------------------------
    ai.rollback(sdfg, 'square_plus_one', round=baseline)
    assert _tasklet(sdfg).code.as_string == first, 'rollback did not restore the first version'

    b3 = np.zeros(N)
    sdfg(A=a, B=b3)
    assert np.allclose(b3, a * a + 1), 'the rolled-back version does not run'


@pytest.mark.ai
@pytest.mark.skipif(platform.machine() != 'x86_64', reason='needs an x86-64 host')
@pytest.mark.skipif(not cpu_has_feature('avx2') or not cpu_has_feature('fma'),
                    reason='needs a host CPU with AVX2 and FMA')
def test_a_performance_complaint_gets_a_vectorized_rewrite():
    """
    The motivating case: the kernel is correct but too slow, and the fix is told, not re-derived.

    The first round is asked for a scalar triple loop, so the second round is a genuine rewrite of
    working code rather than a first attempt that happens to be vectorized.
    """
    sdfg = _tile_gemm_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))

    node.expand(state, 'ai')
    assert 'immintrin' not in _tasklet(sdfg).code_global.as_string, \
        'the first round was already vectorized, so this does not test a revision'
    baseline = ai.sessions(sdfg)[0].round

    rng = np.random.default_rng(0)
    a = rng.random((N, N), dtype=np.float32)
    b = rng.random((N, N), dtype=np.float32)
    c = np.zeros((N, N), dtype=np.float32)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4), 'the scalar version is already wrong'

    ai.refine(sdfg, 'gemm_tile', VECTORIZE_FEEDBACK)

    tasklet = _tasklet(sdfg)
    assert 'immintrin' in tasklet.code_global.as_string, 'the include belongs at file scope'
    assert 'immintrin' not in tasklet.code.as_string, 'the include must not be in the body'
    assert '_mm256' in tasklet.code.as_string, 'the revision is not using AVX2 intrinsics'
    # The enclosing map is already parallel; the context says so and the revision must respect it
    assert '#pragma omp parallel' not in tasklet.code.as_string

    c2 = np.zeros((N, N), dtype=np.float32)
    sdfg(A=a, B=b, C=c2)
    assert np.allclose(c2, a @ b, rtol=1e-4, atol=1e-4), 'the vectorized rewrite is wrong'
    assert ai.sessions(sdfg)[0].round == baseline + 1


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'ai', '-v'])
