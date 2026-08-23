# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A WCR accumulator under a SEQUENTIAL map lives in a register, not in memory.

A sequential map has no concurrency, so its conflict resolution needs no atomic -- but the plain
``wcr_fixed`` form still reads and writes the destination every iteration for a location that never
moves. The CPU backend hoists such a target into a local scalar, accumulates there, and stores once
at the map exit: identical arithmetic in identical order, one load and one store instead of one
pair per iteration.

The emitted text IS the product here, so the structural assertions read the generated C++ -- a
numeric check alone passes just as well on the un-hoisted form it is meant to replace.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.codegen import generate_code

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')


def generated_cpu_code(sdfg: dace.SDFG) -> str:
    return '\n'.join(c.clean_code for c in generate_code(sdfg))


def sequential_sum_sdfg(name: str = 'seq_wcr_sum') -> dace.SDFG:
    """``out[0] += x[i]`` over a sequential map -- one accumulator, invariant target."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('acc', {'i': '0:N'}, {'__x': dace.Memlet('x[i]')},
                             '__o = __x', {'__o': dace.Memlet('out[0]', wcr='lambda a, b: a + b')},
                             external_edges=True,
                             schedule=dtypes.ScheduleType.Sequential)
    return sdfg


def test_sequential_wcr_accumulates_in_a_register():
    """One load before the loop, one store after, and the body touching only the scalar."""
    code = generated_cpu_code(sequential_sum_sdfg())
    acc = [ln.strip() for ln in code.splitlines() if '__acc_' in ln]
    assert len(acc) == 3, f'expected declare / accumulate / store-back, got {acc}'
    declare, accumulate, store = acc
    assert declare.startswith('double __acc_') and declare.endswith('= *(out);'), declare
    # The accumulator VARIABLE is named after its target, so test for the dereference, not the
    # name: what must not appear in the body is a read or write of the location itself.
    assert '=' in accumulate and '+' in accumulate and '*(out)' not in accumulate, \
        f'the body must accumulate into the scalar, not into memory: {accumulate}'
    assert store.startswith('*(out) = __acc_'), store
    assert 'reduce_atomic' not in code, 'a sequential map needs no atomic'


def test_sequential_wcr_is_bit_exact_against_the_sequential_oracle():
    """Hoisting must not reassociate: same order, so the sum matches the oracle bit for bit."""
    sdfg = sequential_sum_sdfg('seq_wcr_sum_numeric')
    n = 4096
    x = np.random.default_rng(0).random(n)
    out = np.zeros(1)
    sdfg(x=x, out=out, N=n)

    expected = 0.0
    for value in x:
        expected += value
    assert out[0] == expected, f'{out[0]!r} != {expected!r}'


def test_a_shared_target_under_a_parallel_map_keeps_its_atomic():
    """THIS map being sequential is not enough. Under a parallel outer map the target is shared,
    and a per-thread register accumulator would drop every other thread's contribution."""
    sdfg = dace.SDFG('par_outer_seq_inner')
    sdfg.add_array('x', [M, N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', {'j': '0:M'}, schedule=dtypes.ScheduleType.CPU_Multicore)
    inner_entry, inner_exit = state.add_map('inner', {'i': '0:N'}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('t', {'__x'}, {'__o'}, '__o = __x')
    state.add_memlet_path(state.add_read('x'),
                          outer_entry,
                          inner_entry,
                          tasklet,
                          dst_conn='__x',
                          memlet=dace.Memlet('x[j, i]'))
    state.add_memlet_path(tasklet,
                          inner_exit,
                          outer_exit,
                          state.add_write('out'),
                          src_conn='__o',
                          memlet=dace.Memlet('out[0]', wcr='lambda a, b: a + b'))
    sdfg.validate()

    code = generated_cpu_code(sdfg)
    assert '__acc_' not in code, 'a target shared across threads must not be hoisted'

    m, n = 64, 200
    x = np.random.default_rng(1).random((m, n))
    out = np.zeros(1)
    sdfg(x=x, out=out, M=m, N=n)
    assert np.isclose(out[0], x.sum())


def test_a_target_the_body_also_reads_is_not_hoisted():
    """The register holds the running value until the exit, so a read of the same array inside the
    map would see the stale memory copy. Refusing keeps that read correct."""
    sdfg = dace.SDFG('seq_wcr_readback')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('acc', {'i': '0:N'}, {
        '__x': dace.Memlet('x[i]'),
        '__prev': dace.Memlet('out[0]')
    },
                             '__o = __x * __prev', {'__o': dace.Memlet('out[0]', wcr='lambda a, b: a + b')},
                             external_edges=True,
                             schedule=dtypes.ScheduleType.Sequential)
    sdfg.validate()
    assert '__acc_' not in generated_cpu_code(sdfg), 'an array the body also reads must not be hoisted'


def test_a_scattered_target_is_not_hoisted():
    """A destination that moves with the map parameter is a scatter, not one accumulator."""
    sdfg = dace.SDFG('seq_wcr_scatter')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('acc', {'i': '0:N'}, {'__x': dace.Memlet('x[i]')},
                             '__o = __x', {'__o': dace.Memlet('out[i]', wcr='lambda a, b: a + b')},
                             external_edges=True,
                             schedule=dtypes.ScheduleType.Sequential)
    sdfg.validate()
    assert '__acc_' not in generated_cpu_code(sdfg), 'a param-dependent destination must not be hoisted'


def test_a_wcr_that_arrives_as_a_copy_is_not_hoisted():
    """The register is only ever accumulated into from :meth:`write_and_resolve_expr`, which sees
    Tasklet and NestedSDFG connector writes. A WCR whose source is an AccessNode is a memlet COPY,
    emitted as a ``CopyND::Accumulate`` straight to memory -- hoisting it would declare a register,
    never add to it, and store the pre-loop value back over the real result at the exit."""
    sdfg = dace.SDFG('seq_wcr_copy_source')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('acc', {'i': '0:N'}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('t', {'__x'}, {'__o'}, '__o = __x')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_read('x'), entry, tasklet, dst_conn='__x', memlet=dace.Memlet('x[i]'))
    state.add_edge(tasklet, '__o', tmp, None, dace.Memlet('tmp[0]'))
    state.add_memlet_path(tmp,
                          exit_node,
                          state.add_write('out'),
                          memlet=dace.Memlet('out[0]', wcr='lambda a, b: a + b'))
    sdfg.validate()

    code = generated_cpu_code(sdfg)
    assert '__acc_' not in code, 'a WCR reaching the exit as a copy must not be hoisted'

    n = 512
    x = np.random.default_rng(2).random(n)
    out = np.zeros(1)
    sdfg(x=x, out=out, N=n)
    assert np.isclose(out[0], x.sum()), f'{out[0]!r} != {x.sum()!r}'


def test_a_wcr_a_nested_sdfg_resolves_is_not_hoisted():
    """A NestedSDFG resolves the write conflict INSIDE its own body: codegen outlines that body
    into its own function, which calls ``wcr_fixed::reduce`` through the connector pointer it was
    handed. The caller's register is never added to, so hoisting stores the pre-loop value back
    over the real result -- silently, with a valid SDFG and C++ that compiles.

    Spelled through the frontend rather than hand-built, because whether the body stays outlined
    is the whole point: a nested SDFG holding a single tasklet is inlined during codegen
    preprocessing, which turns the source into a Tasklet and makes hoisting correct again.
    """

    @dace.program
    def rowsums(x: dace.float64[N, K], out: dace.float64[N]):

        @dace.mapscope
        def rows(i: _[0:N]):

            @dace.map
            def cols(k: _[0:K]):
                inp << x[i, k]
                o >> out(1, lambda a, b: a + b)[i]
                o = inp

    sdfg = rowsums.to_sdfg()
    assert '__acc_' not in generated_cpu_code(sdfg), 'a WCR a nested SDFG resolves must not be hoisted'

    n, k = 32, 7
    x = np.random.default_rng(4).random((n, k))
    out = np.zeros(n)
    sdfg(x=x, out=out, N=n, K=k)
    assert np.allclose(out, x.sum(axis=1)), f'{out!r} != {x.sum(axis=1)!r}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
