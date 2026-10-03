# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A symbolic OpenMP chunk must reach the pragma AND still compile and compute.

OpenMP evaluates ``chunk_size`` as an integer expression at run time, so a chunk derived from the
map's own trip count and the team size adapts to the problem instead of baking one machine's
constant into the graph. That only helps if what lands in the pragma is valid C -- a chunk is a
piece of GENERATED SOURCE, and a bad one fails the whole build, not just the schedule. So every case
here compiles for real and checks the result, rather than inspecting the string and stopping.

The measurement that motivated the derived form: a fixed ``dynamic, 1`` was 3.5x at 128 threads and
0.52x at 32 on the same kernel, and 0.03x on a balanced one. A chunk that scales with the team is
the difference between a policy and a constant fitted to one machine.
"""
import numpy as np
import pytest

import dace
from dace import dtypes, symbolic
from dace.sdfg import nodes

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def scale(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


def built(chunk, kind=dtypes.OMPScheduleType.Static):
    sdfg = scale.to_sdfg(simplify=True)
    touched = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                node.map.schedule = dtypes.ScheduleType.CPU_Multicore
                node.map.omp_schedule = kind
                node.map.omp_chunk_size = chunk
                touched += 1
    assert touched, 'no map to schedule'
    return sdfg


def code_of(sdfg):
    return '\n'.join(obj.clean_code for obj in sdfg.generate_code())


def runs_correctly(sdfg, n=257):
    rng = np.random.default_rng(5)
    a = rng.random(n)
    b = np.zeros(n)
    sdfg(a=a, b=b, N=n)
    assert np.allclose(b, a * 2.0), 'the scheduled kernel computed the wrong answer'


def test_a_constant_chunk_still_works():
    sdfg = built(4)
    assert 'schedule(static, 4)' in code_of(sdfg)
    runs_correctly(sdfg)


def test_zero_emits_no_chunk_clause():
    """0 is the sentinel for "no chunk", not a chunk of size zero -- which OpenMP rejects."""
    code = code_of(built(0))
    assert 'schedule(static)' in code
    assert 'schedule(static,' not in code


def test_a_symbolic_chunk_reaches_the_pragma_and_compiles():
    """The whole point: a chunk the compiler cannot evaluate, emitted as an expression."""
    sdfg = built(symbolic.pystr_to_symbolic('N / 8'))
    code = code_of(sdfg)
    assert 'schedule(static,' in code and 'N' in code.split('schedule(static,')[1][:40]
    runs_correctly(sdfg)  # compiles for real -- a malformed expression fails the build, not the test


def test_a_derived_chunk_over_trip_and_team_compiles():
    """The form the policy actually wants: ceil(trip / (P * K)), team read at run time."""
    sdfg = built(symbolic.pystr_to_symbolic('int_ceil(N, 8)'))
    assert 'schedule(static,' in code_of(sdfg)
    runs_correctly(sdfg)


@pytest.mark.parametrize('kind',
                         [dtypes.OMPScheduleType.Static, dtypes.OMPScheduleType.Dynamic, dtypes.OMPScheduleType.Guided])
def test_every_schedule_kind_takes_a_symbolic_chunk(kind):
    sdfg = built(symbolic.pystr_to_symbolic('int_ceil(N, 16)'), kind)
    assert f'schedule({kind.name.lower()},' in code_of(sdfg)
    runs_correctly(sdfg)


@pytest.mark.parametrize('n', [1, 2, 17, 256, 1000])
def test_the_derived_chunk_is_correct_at_every_size(n):
    """A chunk expression that evaluates to 0 or a negative at some N would abort at run time."""
    sdfg = built(symbolic.pystr_to_symbolic('int_ceil(N, 8)'))
    runs_correctly(sdfg, n)


def test_a_symbolic_chunk_survives_serialization():
    """A chunk that does not round-trip would silently become a different schedule on reload."""
    sdfg = built(symbolic.pystr_to_symbolic('int_ceil(N, 8)'))
    reloaded = dace.SDFG.from_json(sdfg.to_json())
    chunks = [
        n.map.omp_chunk_size for state in reloaded.states() for n in state.nodes() if isinstance(n, nodes.MapEntry)
    ]
    assert chunks, 'no map survived the round trip'
    assert all(symbolic.pystr_to_symbolic(str(c)).free_symbols for c in chunks), chunks
    assert 'schedule(static,' in code_of(reloaded)


if __name__ == '__main__':
    test_a_constant_chunk_still_works()
    test_zero_emits_no_chunk_clause()
    test_a_symbolic_chunk_reaches_the_pragma_and_compiles()
    test_a_derived_chunk_over_trip_and_team_compiles()
    test_every_schedule_kind_takes_a_symbolic_chunk(dtypes.OMPScheduleType.Dynamic)
    test_the_derived_chunk_is_correct_at_every_size(17)
    test_a_symbolic_chunk_survives_serialization()
