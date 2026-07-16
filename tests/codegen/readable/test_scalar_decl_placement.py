# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.scalar_decl_placement`` (FEATURE A).

``eager`` (the default) declares a mutable scope-lifetime value scalar as ``T x;`` at the top of its
scope, exactly as the legacy generator does. ``late`` defers that declaration to the line immediately
before the first tasklet that uses the scalar. The knob is experimental_readable-only, so legacy output
is byte-identical for every value, and the default reproduces today's readable output byte-for-byte.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary

from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, generated_code,
                                             run_isolated, use_implementation)

# A[0] + 1 + 2 + 3 = A[0] + 6; pre = A[0] * 10.
EXPECTED_S = 6.0
EXPECTED_PRE_FACTOR = 10.0


def reassign_scalar_sdfg(name):
    """State-scope value scalar ``s`` reassigned three times (a genuinely MUTABLE scalar, not the
    write-once ``const`` form), plus a single-write ``pre`` whose only read is the very last statement.
    Both are plain register scalars whose every access is a direct-child tasklet of the state."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [2], dace.float64)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    sdfg.add_scalar('pre', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    state = sdfg.add_state('main')

    # pre = A[0] * 10 (read only at the end, so its declaration sits far from its use under eager).
    a_pre = state.add_read('A')
    pre = state.add_access('pre')
    t_pre = state.add_tasklet('tp', {'inp'}, {'o'}, 'o = inp * 10.0')
    state.add_edge(a_pre, None, t_pre, 'inp', dace.Memlet('A[0]'))
    state.add_edge(t_pre, 'o', pre, None, dace.Memlet('pre[0]'))

    # s = A[0] + 1; s = s + 2; s = s + 3  (three reassignments -> one mutable variable).
    a = state.add_read('A')
    prev = state.add_access('s')
    t1 = state.add_tasklet('t1', {'inp'}, {'o'}, 'o = inp + 1.0')
    state.add_edge(a, None, t1, 'inp', dace.Memlet('A[0]'))
    state.add_edge(t1, 'o', prev, None, dace.Memlet('s[0]'))
    for k in (2, 3):
        nxt = state.add_access('s')
        t = state.add_tasklet(f't{k}', {'inp'}, {'o'}, f'o = inp + {float(k)}')
        state.add_edge(prev, None, t, 'inp', dace.Memlet('s[0]'))
        state.add_edge(t, 'o', nxt, None, dace.Memlet('s[0]'))
        prev = nxt

    w = state.add_write('out')
    t_out = state.add_tasklet('t4', {'inp'}, {'o'}, 'o = inp')
    state.add_edge(prev, None, t_out, 'inp', dace.Memlet('s[0]'))
    state.add_edge(t_out, 'o', w, None, dace.Memlet('out[0]'))
    t_pre_out = state.add_tasklet('t5', {'inp'}, {'o'}, 'o = inp')
    state.add_edge(pre, None, t_pre_out, 'inp', dace.Memlet('pre[0]'))
    state.add_edge(t_pre_out, 'o', w, None, dace.Memlet('out[1]'))
    sdfg.validate()
    return sdfg


def parallel_map_sdfg(name):
    """A CPU_Multicore (OpenMP parallel) elementwise map -- no state-scope scalar to defer, so the
    knob must leave every byte of its output unchanged."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [16], dace.float64)
    sdfg.add_array('b', [16], dace.float64)
    state = sdfg.add_state('main')
    read, write = state.add_read('a'), state.add_write('b')
    entry, exit_node = state.add_map('m', {'i': '0:16'}, schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet('t', {'inp'}, {'o'}, 'o = inp + 1.0')
    state.add_memlet_path(read, entry, tasklet, dst_conn='inp', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_node, write, src_conn='o', memlet=dace.Memlet('b[i]'))
    sdfg.validate()
    return sdfg


def readable_code(sdfg, placement):
    with use_implementation(EXPERIMENTAL), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_decl_placement', value=placement):
        return generated_code(sdfg)


def body_lines(code):
    """The stripped, non-empty source lines of the ``*_internal`` function body."""
    out, inside = [], False
    for raw in code.splitlines():
        line = raw.split('////')[0].rstrip()
        if 'internal(' in line:
            inside = True
        if inside and line.strip():
            out.append(line.strip())
        if inside and line.strip() == '}' and len(out) > 3:
            break
    return out


def test_late_moves_declaration_to_first_use(require_experimental):
    """Under ``late`` the ``double s;`` declaration is emitted on the line immediately before the first
    statement that uses ``s`` (``s = (A...``); under ``eager`` it is hoisted above unrelated code."""
    eager = body_lines(readable_code(reassign_scalar_sdfg('a_eager'), 'eager'))
    late = body_lines(readable_code(reassign_scalar_sdfg('a_late'), 'late'))

    decl = 'double s;'
    first_use = next(i for i, l in enumerate(late) if l.startswith('s = ('))
    assert decl in late, late
    # late: the declaration sits on exactly the line before the first use.
    assert late[first_use - 1] == decl, late

    # eager: the declaration is present but hoisted -- there is intervening code (the `pre` chain) between
    # it and the first use of `s`, so it is NOT on the immediately-preceding line.
    eager_decl = eager.index(decl)
    eager_use = next(i for i, l in enumerate(eager) if l.startswith('s = ('))
    assert eager_use - eager_decl > 1, eager


def test_default_placement_is_eager(require_experimental):
    """The default readable output equals the explicit ``eager`` output (byte-identical)."""
    with use_implementation(EXPERIMENTAL):
        default = generated_code(reassign_scalar_sdfg('a_def'))
    explicit_eager = readable_code(reassign_scalar_sdfg('a_def'), 'eager')
    assert default == explicit_eager


def test_legacy_byte_identical_across_placement():
    """Legacy ignores the experimental-only key: its output is identical for eager and late."""

    def legacy(placement):
        sdfg = reassign_scalar_sdfg('a_leg')
        with use_implementation(LEGACY), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_decl_placement', value=placement):
            return generated_code(sdfg)

    assert legacy('eager') == legacy('late')


def test_parallel_map_unaffected(require_experimental):
    """A parallel (CPU_Multicore) map has no deferrable state scalar, so eager and late produce
    byte-identical code -- the knob never touches anything inside a map scope."""
    eager = readable_code(parallel_map_sdfg('a_par'), 'eager')
    late = readable_code(parallel_map_sdfg('a_par'), 'late')
    assert '#pragma omp parallel for' in eager
    assert eager == late


@pytest.mark.parametrize('placement', ['eager', 'late'])
def test_compiles_and_runs_bit_identical(require_experimental, placement):
    """Both placements compile and reproduce the legacy result bit-for-bit."""
    A = np.array([2.0], dtype=np.float64)

    def run(impl, decl_placement='eager'):

        def build_and_run():
            with use_implementation(impl), \
                 set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_decl_placement', value=decl_placement):
                csdfg = reassign_scalar_sdfg('a_run').compile()
                out = np.zeros(2, dtype=np.float64)
                csdfg(A=A.copy(), out=out)
                return {'out': out}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    experimental = run(EXPERIMENTAL, placement)
    assert np.allclose(legacy['out'], [A[0] + EXPECTED_S, A[0] * EXPECTED_PRE_FACTOR])
    assert_outputs_equivalent(legacy, experimental, 'cpu', label=f'scalar_decl_placement={placement}')


if __name__ == '__main__':
    test_late_moves_declaration_to_first_use(None)
    test_default_placement_is_eager(None)
    test_legacy_byte_identical_across_placement()
    test_parallel_map_unaffected(None)
    for p in ('eager', 'late'):
        test_compiles_and_runs_bit_identical(None, p)
    print('OK')
