# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.ssa_loop_scalars`` (FEATURE B).

``off`` (the default) leaves a scalar that is reassigned several times as one mutable variable. ``on``
runs the existing ``ScalarFission`` (``PrivatizeScalars``) pass inside the readable-generator pipeline,
versioning each dominating write into its own single-assignment container (``s``, ``s_0``, ``s_1`` ...);
each version is then a write-once value the ``MarkConstInit`` path emits as ``const T s_n = expr;``. The
knob is experimental_readable-only (it gates an SDFG rewrite in the readable block of ``codegen.py``),
so legacy output is byte-identical for every value, and ``off`` reproduces today's readable output.
"""
import re

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary

from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, generated_code,
                                             run_isolated, use_implementation)

EXPECTED_S = 6.0  # A[0] + 1 + 2 + 3.


def reassign_scalar_sdfg(name):
    """State-scope value scalar ``s`` reassigned three times -- one mutable variable under ``off``, three
    single-assignment versions under ``on``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    state = sdfg.add_state('main')
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
    sdfg.validate()
    return sdfg


def readable_code(sdfg, ssa):
    with use_implementation(EXPERIMENTAL), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'ssa_loop_scalars', value=ssa):
        return generated_code(sdfg)


def assignment_targets(code):
    """Every assignment target in the emitted body, so a single-assignment name appears exactly once.

    Matches all three spellings a write to ``s`` can take: a bare ``s = (...)``, the write-once
    ``const double s = (...)`` binding MarkConstInit produces, and the ``double s = (...)`` binding
    the ``fused`` default of ``scalar_init_style`` produces for a mutable scalar's FIRST write. Only
    accepting the first two would silently miss that first write and undercount.
    """
    targets = []
    for raw in code.splitlines():
        line = raw.split('////')[0].strip()
        m = re.match(r'(?:(?:const\s+)?\w+\s+)?(s(?:_\d+)?)\s*=\s*\(', line)
        if m:
            targets.append(m.group(1))
    return targets


def test_ssa_on_creates_single_assignment_versions(require_experimental):
    """``on`` versions ``s`` into several single-assignment names, each written exactly once and each a
    ``const``; ``off`` keeps one mutable ``s`` reassigned three times."""
    off = readable_code(reassign_scalar_sdfg('b_off'), 'off')
    on = readable_code(reassign_scalar_sdfg('b_on'), 'on')

    off_targets = assignment_targets(off)
    assert off_targets == ['s', 's', 's'], off_targets  # one mutable variable, three writes
    assert 'const double s' not in off  # not const: it is reassigned
    # ... and exactly one of those three writes carries the declaration (the `fused` default), so `s`
    # is still ONE variable and not three shadowing ones.
    assert off.count('double s') == 1, off

    on_targets = assignment_targets(on)
    assert len(on_targets) == 3, on_targets
    assert len(set(on_targets)) == 3, on_targets  # three DISTINCT single-assignment names
    for target in on_targets:
        assert on_targets.count(target) == 1  # each written exactly once
        assert f'const double {target} =' in on, on  # ... and emitted as const


def test_default_is_off(require_experimental):
    """The default readable output equals the explicit ``off`` output (byte-identical)."""
    with use_implementation(EXPERIMENTAL):
        default = generated_code(reassign_scalar_sdfg('b_def'))
    assert default == readable_code(reassign_scalar_sdfg('b_def'), 'off')


def test_legacy_byte_identical_across_ssa():
    """Legacy never enters the readable pipeline block: its output is identical for off and on."""

    def legacy(ssa):
        sdfg = reassign_scalar_sdfg('b_leg')
        with use_implementation(LEGACY), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'ssa_loop_scalars', value=ssa):
            return generated_code(sdfg)

    assert legacy('off') == legacy('on')


@pytest.mark.parametrize('ssa', ['off', 'on'])
def test_compiles_and_runs_bit_identical(require_experimental, ssa):
    """Both settings compile and reproduce the legacy result bit-for-bit."""
    A = np.array([2.0], dtype=np.float64)

    def run(impl, ssa_value=None):

        def build_and_run():
            with use_implementation(impl):
                if ssa_value is not None:
                    with set_temporary('compiler', 'cpu', 'codegen_params', 'ssa_loop_scalars', value=ssa_value):
                        csdfg = reassign_scalar_sdfg('b_run').compile()
                else:
                    csdfg = reassign_scalar_sdfg('b_run').compile()
                out = np.zeros(1, dtype=np.float64)
                csdfg(A=A.copy(), out=out)
                return {'out': out}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    experimental = run(EXPERIMENTAL, ssa)
    assert np.allclose(legacy['out'], [A[0] + EXPECTED_S])
    assert_outputs_equivalent(legacy, experimental, 'cpu', label=f'ssa_loop_scalars={ssa}')


def test_ssa_on_with_late_placement_runs(require_experimental):
    """FEATURE A + FEATURE B together compile and stay bit-exact: SSA versioning makes each write a
    const (so the late-placement path finds nothing left to defer), and the result is unchanged."""
    A = np.array([3.0], dtype=np.float64)

    def build_and_run():
        with use_implementation(EXPERIMENTAL), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'ssa_loop_scalars', value='on'), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'decl_placement', value='late'):
            csdfg = reassign_scalar_sdfg('b_mixed').compile()
            out = np.zeros(1, dtype=np.float64)
            csdfg(A=A.copy(), out=out)
            return {'out': out}

    result = run_isolated(build_and_run)
    assert np.allclose(result['out'], [A[0] + EXPECTED_S])


if __name__ == '__main__':
    test_ssa_on_creates_single_assignment_versions(None)
    test_default_is_off(None)
    test_legacy_byte_identical_across_ssa()
    for s in ('off', 'on'):
        test_compiles_and_runs_bit_identical(None, s)
    test_ssa_on_with_late_placement_runs(None)
    print('OK')
