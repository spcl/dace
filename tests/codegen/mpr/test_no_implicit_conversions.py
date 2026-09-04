# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR renders no IMPLICIT conversion: the allocation size and a nested body's symbol parameters.

Two narrowings, both invisible until a strict build:

``aligned_alloc``/``malloc`` take a ``size_t`` while an extent is a SIGNED expression, so every
allocation the C dialect emits converted sign implicitly -- four ``-Wsign-conversion`` diagnostics
per render on a kernel with four transients, which is a render that cannot pass a ``-Werror`` gate.

A nested SDFG's symbol parameter is typed from the symbol table, where a loop iterator carries the
int32 DEFAULT symbol type. The body then names it ``int`` while every index and ``_size`` helper
MPR emits beside it is ``int64_t``, so the call NARROWS. In cholesky the narrowed value is a dot
product's trip count -- an extent, not just an index -- which truncates past 2^31.

The shared ``WARNING_FLAGS`` are only ``-Wall -Wextra``, and neither narrowing is diagnosed there:
the conversion flags below are what makes these assertions bite.
"""
import re

import dace
from dace import dtypes, mpr_lowering
from dace.codegen.mpr import render
from dace.codegen.targets.cpu import CPUCodeGen

from tests.codegen.mpr.conftest import compile_standalone

#: What a strict consumer builds with; the shared WARNING_FLAGS do not include these.
CONVERSION_FLAGS = ('-Wall', '-Wextra', '-Wconversion', '-Wsign-conversion', '-Werror')

N = dace.symbol('N')


@dace.program
def two_stage(src: dace.float64[N], dst: dace.float64[N]):
    tmp = src * 2.0
    dst[:] = tmp + 1.0


def test_the_c_allocation_spells_its_size_conversion():
    """The transient's byte count reaches ``aligned_alloc``/``malloc`` through an explicit cast."""
    sdfg = two_stage.to_sdfg(simplify=True)
    sdfg.name = 'mpr_alloc_c'
    code = render(sdfg, language='c').code
    allocations = [line for line in code.splitlines() if 'aligned_alloc(' in line or 'malloc(' in line]
    assert allocations, code
    for line in allocations:
        assert '(size_t)(' in line, line


def test_a_strict_conversion_build_of_the_c_render_is_clean():
    """``-Wconversion -Wsign-conversion -Werror``: the flags the allocation used to fail."""
    sdfg = two_stage.to_sdfg(simplify=True)
    sdfg.name = 'mpr_alloc_strict'
    code = render(sdfg, language='c').code
    compile_standalone(code, 'mpr_alloc_strict', extra_flags=CONVERSION_FLAGS, language='c')


def test_a_nested_symbol_parameter_is_int64_under_mpr():
    """An integer symbol widens, whatever the symbol table says, so no call site narrows."""
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.STANDALONE):
        assert CPUCodeGen.nsdfg_symbol_argument(dtypes.int32, '_loop_it_0') == dtypes.int64.as_arg('_loop_it_0')
        assert CPUCodeGen.nsdfg_symbol_argument(dtypes.int64, 'N') == dtypes.int64.as_arg('N')
        # A non-integer symbol is not this rule's business.
        assert CPUCodeGen.nsdfg_symbol_argument(dtypes.float64, 'f') == dtypes.float64.as_arg('f')


def test_main_signatures_are_left_alone():
    """Outside a standalone dialect the symbol keeps its own type: main's text is not ours to change."""
    assert CPUCodeGen.nsdfg_symbol_argument(dtypes.int32, '_loop_it_0') == dtypes.int32.as_arg('_loop_it_0')
    assert 'int64_t' not in CPUCodeGen.nsdfg_symbol_argument(dtypes.int32, '_loop_it_0')


def test_no_nested_body_signature_narrows_a_symbol():
    """End to end, on the NESTED signatures only.

    The ENTRY point is a separate question: it takes the symbol's declared type, so a program whose
    symbol is the int32 default renders ``int N`` there and every call from it WIDENS. Only a nested
    body's parameter is reached by a narrowing call, and only those are asserted here.
    """
    sdfg = two_stage.to_sdfg(simplify=True)
    sdfg.name = 'mpr_nested_symbols'
    code = render(sdfg, language='c++').code
    nested = [line for line in code.splitlines() if re.match(r'\s*(inline\s+)?void\s+_', line)]
    for line in nested:
        assert not re.search(r'\bint\s+\w+\s*[,)]', line), line


@dace.program
def mixed_store(arr: dace.int64[N], out: dace.int64[N], scale: dace.float64):
    out[:] = arr * scale


def test_a_narrowing_store_spells_its_conversion():
    """``int64 * float64`` computes in double and stores into ``int64_t``: the cast is emitted.

    The frontend chose the narrowing, so this is not about changing the value -- it is about the
    render saying so, and about the file staying clean under ``-Wfloat-conversion``.
    """
    sdfg = mixed_store.to_sdfg(simplify=True)
    sdfg.name = 'mpr_narrowing_store'
    code = render(sdfg, language='c').code
    stores = [line for line in code.splitlines() if 'out[' in line and '=' in line]
    assert stores, code
    assert any('(int64_t)(' in line for line in stores), stores


def test_a_narrowing_store_builds_clean_under_conversion_warnings():
    sdfg = mixed_store.to_sdfg(simplify=True)
    sdfg.name = 'mpr_narrowing_strict'
    code = render(sdfg, language='c').code
    compile_standalone(code, 'mpr_narrowing_strict', extra_flags=CONVERSION_FLAGS, language='c')


def test_a_matched_store_is_not_cast():
    """No cast where the types already agree -- the rule is 'no IMPLICIT conversion', not 'cast all'."""
    sdfg = two_stage.to_sdfg(simplify=True)
    sdfg.name = 'mpr_matched_store'
    code = render(sdfg, language='c').code
    stores = [line for line in code.splitlines() if 'dst[' in line and '=' in line]
    assert stores, code
    for line in stores:
        assert '(double)(' not in line, line


@dace.program
def fused_store(arr: dace.int64[N], out: dace.int64[N], scale: dace.float64, bias: dace.float64):
    tmp = arr * scale
    out[:] = tmp + bias


def test_a_fused_store_that_names_its_data_still_spells_the_conversion():
    """The fused body names its DATA, not its connectors -- inference has to be given both.

    ``tmp + bias`` arrives as ``(tmp[0, 0] + bias[0])``. Typed from ``in_connectors`` alone those
    names are unbound, inference returns ``None``, and the miss reads exactly like "the types agree"
    -- so the narrowing goes back to being implicit on precisely the kernels that fuse.
    """
    sdfg = fused_store.to_sdfg(simplify=True)
    sdfg.name = 'mpr_fused_store'
    code = render(sdfg, language='c').code
    stores = [line for line in code.splitlines() if 'out[' in line and '=' in line]
    assert stores, code
    assert any('(int64_t)(' in line for line in stores), stores


def test_the_cpp_render_casts_with_static_cast():
    """C++ has ``-Wold-style-cast``: the C spelling would trade one diagnostic for another."""
    sdfg = fused_store.to_sdfg(simplify=True)
    sdfg.name = 'mpr_fused_store_cpp'
    code = render(sdfg, language='c++').code
    stores = [line for line in code.splitlines() if 'out[' in line and '=' in line]
    assert stores, code
    assert any('static_cast<int64_t>(' in line for line in stores), stores
    for line in stores:
        assert '(int64_t)(' not in line, line


def test_the_cpp_render_is_clean_under_strict_cast_warnings():
    sdfg = fused_store.to_sdfg(simplify=True)
    sdfg.name = 'mpr_fused_store_strict'
    code = render(sdfg, language='c++').code
    compile_standalone(code,
                       'mpr_fused_store_strict',
                       extra_flags=CONVERSION_FLAGS + ('-Wold-style-cast', ),
                       language='c++')
