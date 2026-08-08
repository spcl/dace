# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the experimental readable CPU generator's codegen-style knobs:
    ``compiler.cpu.codegen_params.heap_ptr_restrict`` and ``index_fn_qualifier``. Each defaults to the
    faster/current form and only the experimental generator emits the qualified constructs, so legacy
    output is unaffected by construction. """
import numpy

import dace
from dace.config import set_temporary

N = dace.symbol('N')


@dace.program
def scaled_twice(A: dace.float64[N], B: dace.float64[N]):
    tmp = numpy.empty(N, dace.float64)  # symbolic-size transient -> heap-allocated pointer
    for i in dace.map[0:N]:
        tmp[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        B[i] = tmp[i] + 1.0


def generate(heap_ptr_restrict='restrict', index_fn_qualifier='inline_constexpr'):
    sdfg = scaled_twice.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict', value=heap_ptr_restrict), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier', value=index_fn_qualifier):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_heap_ptr_restrict_default_emits_restrict():
    code = generate(heap_ptr_restrict='restrict')
    # The fused heap declaration of the transient carries __restrict__.
    assert 'double* __restrict__ tmp' in code


def test_heap_ptr_restrict_none_drops_restrict():
    code = generate(heap_ptr_restrict='none')
    assert '__restrict__ tmp' not in code
    assert 'double* tmp' in code


def test_index_fn_qualifier_default_is_constexpr():
    code = generate(index_fn_qualifier='inline_constexpr')
    assert 'static DACE_HDFI constexpr' in code
    assert 'always_inline' not in code


def test_index_fn_qualifier_always_inline():
    code = generate(index_fn_qualifier='always_inline')
    # Every generated _idx helper gains the forced-inline attribute.
    assert '__attribute__((always_inline))' in code
    for line in code.splitlines():
        if '_idx(' in line and 'return' in line and '{' in line:
            assert '__attribute__((always_inline))' in line


def test_legacy_ignores_both_flags():
    """Legacy emits neither a fused restrict declaration nor _idx helpers, so its output is identical
    across every value of these experimental-only keys."""

    def legacy(hpr, ifq):
        sdfg = scaled_twice.to_sdfg(simplify=True)
        with set_temporary('compiler', 'cpu', 'implementation', value='legacy'), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict', value=hpr), \
             set_temporary('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier', value=ifq):
            return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')

    baseline = legacy('restrict', 'inline_constexpr')
    assert baseline == legacy('none', 'inline_constexpr')
    assert baseline == legacy('restrict', 'always_inline')


def test_both_configs_compile_and_run():
    for hpr in ('restrict', 'none'):
        for ifq in ('inline_constexpr', 'always_inline'):
            with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'), \
                 set_temporary('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict', value=hpr), \
                 set_temporary('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier', value=ifq):
                A = numpy.random.default_rng(0).random(40)
                B = numpy.zeros(40)
                scaled_twice(A=A, B=B, N=40)
                assert numpy.allclose(B, A * 2.0 + 1.0)


if __name__ == '__main__':
    test_heap_ptr_restrict_default_emits_restrict()
    test_heap_ptr_restrict_none_drops_restrict()
    test_index_fn_qualifier_default_is_constexpr()
    test_index_fn_qualifier_always_inline()
    test_legacy_ignores_both_flags()
    test_both_configs_compile_and_run()
