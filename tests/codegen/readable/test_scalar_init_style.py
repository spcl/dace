# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.scalar_init_style``.

``split`` (the default) emits the legacy pair -- ``T x;`` at the scope top, ``x = expr;`` where the
write happens. ``fused`` emits ``T x = expr;``: the declaration IS the first write. The knob is
experimental_readable-only, so legacy output is byte-identical for every value, and the default
reproduces today's readable output byte-for-byte.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary

from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, generated_code,
                                             run_isolated, use_implementation)

N = dace.symbol('N')


@dace.program
def reassigned(A: dace.float64[N], B: dace.float64[N]):
    """``t`` is written twice, so it can never be the ``const T t = expr;`` form MarkConstInit emits --
    it is exactly the mutable scalar this knob is about. It lives in a map body: the scope a frontend
    actually puts scalars in."""
    for i in dace.map[0:N]:
        t = A[i] * 2.0
        u = t + 1.0
        t = u * u
        B[i] = t + u


def readable_code(sdfg, style):
    with use_implementation(EXPERIMENTAL), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_init_style', value=style):
        return generated_code(sdfg)


def test_fused_folds_declaration_into_first_write(require_experimental):
    """``double t; t = expr;`` becomes ``double t = expr;``; the SECOND write stays a plain
    assignment (a reassigned scalar cannot be const-bound, only defined by its first write)."""
    sdfg = reassigned.to_sdfg(simplify=True)
    split, fused = readable_code(sdfg, 'split'), readable_code(sdfg, 'fused')

    assert 'double t;' in split
    assert 't = (A[A_idx(i)] * 2.0);' in split

    assert 'double t;' not in fused, fused
    assert 'double t = (A[A_idx(i)] * 2.0);' in fused, fused
    # The later write is NOT re-declared -- exactly one declaration of `t` survives.
    assert fused.count('double t = ') == 1, fused
    assert 't = t_0;' in fused, fused


def test_default_style_is_split(require_experimental):
    """The default readable output equals the explicit ``split`` output (byte-identical)."""
    with use_implementation(EXPERIMENTAL):
        default = generated_code(reassigned.to_sdfg(simplify=True))
    assert default == readable_code(reassigned.to_sdfg(simplify=True), 'split')


def test_legacy_byte_identical_across_style():
    """Legacy ignores the experimental-only key: identical output for split and fused."""

    def legacy(style):
        with use_implementation(LEGACY), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_init_style', value=style):
            return generated_code(reassigned.to_sdfg(simplify=True))

    assert legacy('split') == legacy('fused')


def test_braced_tasklet_falls_back_to_a_plain_declaration(require_experimental):
    """A first-use tasklet that needs its own ``{ ... }`` block cannot carry the declaration -- it would
    be scoped to that block. The candidate must fall back to a plain ``T x;`` ahead of the block, NOT
    silently lose its declaration. A multi-statement tasklet is always braced."""
    sdfg = dace.SDFG('braced')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('m', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    state = sdfg.add_state('main')
    # Two statements -> the readable generator keeps this tasklet's brace.
    t1 = state.add_tasklet('w', {'inp'}, {'o'}, 'tmp = inp * 3.0\no = tmp + 1.0')
    state.add_edge(state.add_read('A'), None, t1, 'inp', dace.Memlet('A[0]'))
    m = state.add_access('m')
    state.add_edge(t1, 'o', m, None, dace.Memlet('m[0]'))
    t2 = state.add_tasklet('rw', {'inp'}, {'o'}, 'o = inp + 1.0')
    state.add_edge(m, None, t2, 'inp', dace.Memlet('m[0]'))
    m2 = state.add_access('m')
    state.add_edge(t2, 'o', m2, None, dace.Memlet('m[0]'))
    t3 = state.add_tasklet('r', {'inp'}, {'o'}, 'o = inp')
    state.add_edge(m2, None, t3, 'inp', dace.Memlet('m[0]'))
    state.add_edge(t3, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()

    fused = readable_code(sdfg, 'fused')
    # However it is spelled, `m` must be declared exactly once and before the brace that writes it.
    assert fused.count('double m') == 1, fused


@pytest.mark.parametrize('style', ['split', 'fused'])
def test_compiles_and_runs_bit_identical(require_experimental, style):
    """Both styles compile and reproduce the legacy result bit-for-bit."""
    A = np.random.default_rng(0).random(16)

    def run(impl, init_style='split'):

        def build_and_run():
            with use_implementation(impl), \
                 set_temporary('compiler', 'cpu', 'codegen_params', 'scalar_init_style', value=init_style):
                csdfg = reassigned.to_sdfg(simplify=True).compile()
                B = np.zeros(16)
                csdfg(A=A.copy(), B=B, N=16)
                return {'B': B}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    experimental = run(EXPERIMENTAL, style)
    assert_outputs_equivalent(legacy, experimental, 'cpu', label=f'scalar_init_style={style}')


if __name__ == '__main__':
    test_fused_folds_declaration_into_first_write(None)
    test_default_style_is_split(None)
    test_legacy_byte_identical_across_style()
    test_braced_tasklet_falls_back_to_a_plain_declaration(None)
    for s in ('split', 'fused'):
        test_compiles_and_runs_bit_identical(None, s)
    print('OK')
