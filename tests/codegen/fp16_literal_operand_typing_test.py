# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A bare Python literal beside an fp16 operand, through the readable CPU generator.

``dace::float16`` converts implicitly BOTH to and from the built-in floats, so ``1.0 / x`` with ``x``
an fp16 offers two equally good operators -- the built-in ``arithmetic / arithmetic`` converting
``x`` up, and the class's own converting the literal down -- and the compiler rejects the expression:

    error: more than one operator "/" matches these operands:
                built-in operator "arithmetic / arithmetic"
                function "operator/(const __half &, const __half &)"

The readable CPU generator (``compiler.cpu.implementation = experimental_readable``) is a distinct
path into the C++ printer: ``ReadableKeywordRemover`` inlines the tasklet connectors away and renders
the surviving assignment itself, so the printer only knows the operand dtypes this class hands it --
an inlined array access under its access text, an inlined value scalar under its emitted name. That
half is what these pin, at the source-text level, so it needs neither nvcc nor a GPU.

The host ``dace::float16`` is a struct whose conversions g++ can still rank, so a host compile does
NOT reproduce the ambiguity -- the generated text is the only CPU-side evidence, and the run below
exists to show the cast did not move the numbers rather than to catch the overload.
"""
import re

import numpy as np
import pytest

import dace
from dace.config import set_temporary

#: The whole module is about float16, so it carries the marker the fp16 CI leg selects on.
pytestmark = pytest.mark.fp16

N = 8
LITERAL = '1.0'
HALF = dace.float16.ctype

#: Emitted by the classic generator around every tasklet body; its absence proves the statements
#: asserted on below came from ``ReadableKeywordRemover`` and not from ``unparse_tasklet``.
CLASSIC_MARKER = '// Tasklet code'


def reciprocal_sdfg(name: str, dtype: dace.typeclass) -> dace.SDFG:
    """``b[i] = 1.0 / (a[i] + 1.0)`` over a map, through a scalar transient.

    The two tasklets between them reach every operand the commit under test had to type: ``shift``
    combines a literal with an INLINED ARRAY ACCESS (``ReadableKeywordRemover.visit_Subscript``),
    ``recip`` combines one with an INLINED VALUE SCALAR (``visit_Name``), and both statements are
    rendered by ``visit_Assign`` rather than by ``unparse_tasklet``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dtype)
    sdfg.add_array('b', [N], dtype)
    sdfg.add_scalar('s', dtype, transient=True)
    state = sdfg.add_state('main')
    read, write = state.add_read('a'), state.add_write('b')
    entry, exit_node = state.add_map('m', {'i': '0:%d' % N})
    scalar = state.add_access('s')
    shift = state.add_tasklet('shift', {'inp'}, {'o'}, 'o = inp + 1.0')
    recip = state.add_tasklet('recip', {'v'}, {'out'}, 'out = 1.0 / v')
    state.add_memlet_path(read, entry, shift, dst_conn='inp', memlet=dace.Memlet('a[i]'))
    state.add_edge(shift, 'o', scalar, None, dace.Memlet('s[0]'))
    state.add_edge(scalar, None, recip, 'v', dace.Memlet('s[0]'))
    state.add_memlet_path(recip, exit_node, write, src_conn='out', memlet=dace.Memlet('b[i]'))
    sdfg.validate()
    return sdfg


def readable_code(sdfg: dace.SDFG) -> str:
    """Generated C++ for ``sdfg``, whitespace-normalized, with the readable CPU generator selected.

    ``set_temporary`` and not a raw ``Config.set``: a ``DACE_compiler_cpu_implementation`` variable
    outranks the stored configuration, and the CI images export one -- the context manager pops it
    for the duration, so the selection holds whatever the ambient environment asked for.
    """
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        code = '\n'.join((obj.clean_code or obj.code) for obj in sdfg.generate_code())
    return '\n'.join(' '.join(line.split()) for line in code.splitlines())


def literal_statements(code: str):
    """The emitted statements carrying the ``1.0`` this program's tasklets are written with."""
    lines = [line for line in code.splitlines() if LITERAL in line and not line.startswith('//')]
    assert lines, 'no statement carrying the literal was emitted; the test would prove nothing'
    return lines


def untyped(statements, ctype: str):
    """Those ``statements`` holding a ``1.0`` that is NOT immediately preceded by a ``ctype(`` cast."""
    return [
        stmt for stmt in statements
        if any(not stmt[:match.start()].endswith(ctype + '(') for match in re.finditer(re.escape(LITERAL), stmt))
    ]


def test_readable_generator_types_the_literal_against_the_fp16_operand():
    """Both inlined operand kinds must carry their dtype into the literal beside them.

    The two substrings are the whole point: without the cast the same statements read ``(... + 1.0)``
    and ``(1.0 / s)``, which is the pairing no C++ compiler can resolve for ``__half``.
    """
    code = readable_code(reciprocal_sdfg('fp16_literal_operand_readable', dace.float16))
    assert CLASSIC_MARKER not in code, 'the classic generator ran; this test would not reach ReadableKeywordRemover'
    statements = literal_statements(code)
    assert any('+ %s(%s))' % (HALF, LITERAL) in stmt for stmt in statements), statements
    assert any('(%s(%s) / s)' % (HALF, LITERAL) in stmt for stmt in statements), statements
    assert not untyped(statements, HALF), untyped(statements, HALF)


def test_readable_generator_leaves_an_fp64_literal_bare():
    """``double`` is a built-in: its usual arithmetic conversions are unambiguous, so the same
    program in fp64 must keep printing the literal unchanged rather than growing a cast."""
    code = readable_code(reciprocal_sdfg('fp64_literal_operand_readable', dace.float64))
    assert CLASSIC_MARKER not in code, 'the classic generator ran; this control would compare the wrong output'
    statements = literal_statements(code)
    assert any('+ %s)' % LITERAL in stmt for stmt in statements), statements
    assert any('(%s / s)' % LITERAL in stmt for stmt in statements), statements
    assert untyped(statements, dace.float64.ctype) == statements, statements


def test_readable_fp16_kernel_matches_the_half_precision_oracle():
    """Typing the literal is what NumPy computes here too -- a weak Python scalar takes the array's
    dtype -- so the compiled kernel must reproduce the half-precision oracle exactly, not the
    double-then-narrow one a bare literal would have asked for."""
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        csdfg = reciprocal_sdfg('fp16_literal_operand_run', dace.float16).compile()
        a = np.arange(1, N + 1).astype(np.float16)
        b = np.zeros(N, dtype=np.float16)
        csdfg(a=a, b=b)
    expected = np.float16(1.0) / (a + np.float16(1.0))
    assert np.array_equal(b, expected), 'got %s, expected %s' % (b, expected)


if __name__ == '__main__':
    pytest.main([__file__])
