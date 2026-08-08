# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The readable generator must NOT SSA-version reassigned scalars.

Scalar fission (``PrivatizeScalars`` / ``ScalarFission``) used to run inside the readable-generator
pipeline in ``codegen.py`` behind a ``ssa_loop_scalars`` knob. It was wired out: it is an
optimization pass and belongs in the caller's pipeline, ahead of WCR formation. By codegen time an
accumulator chain is already a WCR memlet, and WCR is a read-modify-write -- starting a fresh SSA
version at one drops the running value, which silently miscompiled CloudSC ``full_cpu``.

These tests pin the resulting contract: a scalar reassigned several times stays ONE mutable variable
in the readable output, and the readable result is bit-identical to legacy.
"""
import re

import numpy as np

import dace
from dace import dtypes

from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, generated_code,
                                             run_isolated, use_implementation)

EXPECTED_S = 6.0  # A[0] + 1 + 2 + 3.


def reassign_scalar_sdfg(name):
    """State-scope value scalar ``s`` reassigned three times -- one mutable variable, three writes."""
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


def assignment_targets(code):
    """Every assignment target in the emitted body, so a versioned name would show up as ``s_0``.

    Matches all three spellings a write to ``s`` can take: a bare ``s = (...)``, a ``const double s =
    (...)`` binding from MarkConstInit, and the ``double s = (...)`` binding the ``fused`` default of
    ``scalar_init_style`` produces for a mutable scalar's FIRST write.
    """
    targets = []
    for raw in code.splitlines():
        line = raw.split('////')[0].strip()
        match = re.match(r'(?:(?:const\s+)?\w+\s+)?(s(?:_\d+)?)\s*=\s*\(', line)
        if match:
            targets.append(match.group(1))
    return targets


def test_reassigned_scalar_stays_one_variable(require_experimental):
    """No SSA versioning at codegen: ``s`` is written three times under its own single name."""
    with use_implementation(EXPERIMENTAL):
        code = generated_code(reassign_scalar_sdfg('nofission'))

    targets = assignment_targets(code)
    assert targets == ['s', 's', 's'], targets  # one mutable variable, three writes -- no s_0 / s_1
    assert 'const double s' not in code  # not const: it is reassigned
    assert code.count('double s') == 1, code  # exactly one declaration, not three shadowing ones


def test_readable_matches_legacy_bit_for_bit(require_experimental):
    """The readable result equals legacy's, bit for bit."""
    A = np.array([2.0], dtype=np.float64)

    def run(impl):

        def build_and_run():
            with use_implementation(impl):
                csdfg = reassign_scalar_sdfg('nofission_run').compile()
                out = np.zeros(1, dtype=np.float64)
                csdfg(A=A.copy(), out=out)
                return {'out': out}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    experimental = run(EXPERIMENTAL)
    assert np.allclose(legacy['out'], [A[0] + EXPECTED_S])
    assert_outputs_equivalent(legacy, experimental, 'cpu', label='no scalar fission at codegen')


def test_late_placement_still_runs(require_experimental):
    """The late-placement path is unaffected by the removal and stays numerically exact."""
    A = np.array([3.0], dtype=np.float64)

    def build_and_run():
        with use_implementation(EXPERIMENTAL), \
             dace.config.set_temporary('compiler', 'cpu', 'codegen_params', 'decl_placement', value='late'):
            csdfg = reassign_scalar_sdfg('nofission_late').compile()
            out = np.zeros(1, dtype=np.float64)
            csdfg(A=A.copy(), out=out)
            return {'out': out}

    result = run_isolated(build_and_run)
    assert np.allclose(result['out'], [A[0] + EXPECTED_S])


if __name__ == '__main__':
    test_reassigned_scalar_stays_one_variable(None)
    test_readable_matches_legacy_bit_for_bit(None)
    test_late_placement_still_runs(None)
    print('OK')
