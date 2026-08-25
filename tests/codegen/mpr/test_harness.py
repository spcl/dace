# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Self-test for the MPR acceptance harness.

The harness in ``conftest.py`` is the gate every later MPR phase is judged by, so it is verified
first and on its own terms -- against hand-written C++ shaped exactly like the output MPR is
specified to produce. Two claims have to hold before any of it is worth trusting:

* the bare-compiler build genuinely REJECTS a DaCe include (a gate that quietly succeeds on a
  leaked header is worse than no gate), and
* the ctypes call reproduces DaCe's own argument order, so a numeric compare against the SDFG is
  measuring the kernel and not the harness.

The C++ below is written by hand, not generated: it is the specification of MPR's output shape --
``extern "C"`` entry point, inlined ``<array>_idx`` helper, ``#pragma omp parallel for``,
exact-width integer types, system headers only.
"""
import ctypes
import textwrap

import numpy as np
import pytest

import dace
from tests.codegen.mpr.conftest import (BANNED_PATTERNS, UNQUALIFIED_RUNTIME_FUNCTIONS, assert_matches,
                                        assert_standalone, build_standalone, call_standalone, compile_diagnostics,
                                        compile_standalone, entry_argtypes, host_compiler)

#: Hand-written stand-in for MPR output: ``B[i] = A[i] * 2 + N`` over a symbolic extent, in the
#: exact shape the emitter is specified to produce.
AXPY_MPR = textwrap.dedent("""
    #include <cstdint>

    static constexpr inline int64_t A_idx(int64_t i) { return i; }
    static constexpr inline int64_t B_idx(int64_t i) { return i; }

    extern "C" void mpr_axpy(double * __restrict__ A, double * __restrict__ B, int32_t N)
    {
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(N); i += 1) {
            B[B_idx(i)] = A[A_idx(i)] * 2.0 + static_cast<double>(N);
        }
    }
    """)

#: The WCR form MPR is allowed to emit: an OpenMP ``reduction(op:var)`` clause, never an atomic
#: through a ``dace::wcr_fixed`` helper.
SUM_MPR = textwrap.dedent("""
    #include <cstdint>

    extern "C" void mpr_sum(double * __restrict__ A, double * __restrict__ out, int32_t N)
    {
        double acc = 0.0;
        #pragma omp parallel for reduction(+:acc)
        for (int64_t i = 0; i < static_cast<int64_t>(N); i += 1) {
            acc += A[i];
        }
        out[0] = acc;
    }
    """)


def axpy_sdfg():
    """SDFG whose arglist defines ``mpr_axpy``'s signature: arrays ``A``, ``B``, then symbol ``N``."""
    N = dace.symbol('N')
    sdfg = dace.SDFG('mpr_axpy')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main')
    read, write = state.add_read('A'), state.add_write('B')
    entry, exit_node = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp * 2.0 + N')
    state.add_memlet_path(read, entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, write, src_conn='out', memlet=dace.Memlet('B[i]'))
    return sdfg


def test_host_compiler_resolves():
    """A supported box has a C++ compiler; the harness asserts rather than skipping without one."""
    compiler = host_compiler()
    assert compiler and compiler.startswith('/'), f'host_compiler() must resolve to an absolute path, got {compiler!r}'


@pytest.mark.parametrize('snippet,expected_meaning', [
    ('#include <dace/dace.h>', 'DaCe runtime header include'),
    ('#include "../../include/hash.h"', 'quoted (relative) include -- MPR may only use system headers'),
    ('x = dace::math::exp(y);', 'DaCe runtime namespace reference'),
    ('static DACE_HDFI constexpr int64_t A_idx(int64_t i) { return i; }', 'DaCe preprocessor macro'),
    ('__dace_init_cuda(__state);', 'DaCe init/exit entry point'),
    ('double *A = __state->__0_A;', 'DaCe state-struct dereference'),
    ('dace::CopyND<double, 1, false, 8>::Copy(src, dst);', 'dace::CopyND copy fallback'),
])
def test_banned_pattern_fires(snippet, expected_meaning):
    """Every banned pattern actually catches the construct it names -- the check is not vacuous."""
    with pytest.raises(AssertionError) as excinfo:
        assert_standalone(snippet, label='probe')
    assert expected_meaning in str(excinfo.value), (f'{snippet!r} was rejected, but not as {expected_meaning!r}: '
                                                    f'{excinfo.value}')


#: Exactly what the symbolic printer emits today for each construct (measured, not guessed), with
#: the unqualified DaCe runtime function it lands on. Each of these needs an MPR mapping.
UNQUALIFIED_LEAKS = [
    ('(int_ceil(x, 3))', 'int_ceil'),  # sympy ceiling(x/3) AND symbolic.int_ceil both land here
    ('(reciprocal((x) * (x)))', 'reciprocal'),  # x**-2
    ('(sign(x))', 'sign'),
    ('(mod(x, y))', 'mod'),
    ('(ROUND(x))', 'ROUND'),
    ('(Abs(x))', 'Abs'),
    ('(Max(x, y))', 'Max'),
    ('(Min(x, y))', 'Min'),
    ('(ITE(x > 0, x, y))', 'ITE'),
    ('(logical_left_shift(x, y))', 'logical_left_shift'),
    ('(logical_right_shift(x, y))', 'logical_right_shift'),
]


@pytest.mark.parametrize('snippet,name', UNQUALIFIED_LEAKS, ids=[name for _, name in UNQUALIFIED_LEAKS])
def test_unqualified_runtime_call_is_rejected(snippet, name):
    """Unqualified DaCe runtime calls are caught by name.

    These carry no ``dace::`` marker -- they are declared at global scope by ``math.h`` /
    ``pyinterop.h`` / ``ITE.h``, which MPR does not include. Without this check a leak would
    surface only as an "undeclared identifier" from the compiler, naming nothing useful.
    """
    with pytest.raises(AssertionError, match=f'unqualified DaCe runtime function {name!r}'):
        assert_standalone(snippet, label='probe')


def test_emitted_definition_is_not_a_leak():
    """MPR emitting its OWN ``int_ceil`` is the fix, not a violation -- the check must allow it."""
    code = ('#include <cstdint>\n'
            'static constexpr inline int64_t int_ceil(int64_t a, int64_t b) { return (a + b - 1) / b; }\n'
            'extern "C" void k(int64_t * out, int32_t n) { out[0] = int_ceil(n, 3); }\n')
    assert_standalone(code)
    assert compile_diagnostics(code, name='mpr_intceil') == ''


def test_qualified_call_is_not_a_leak():
    """A namespace-qualified name sharing a listed spelling is not flagged; only a bare call is.

    Also covers the ambiguous direction: ``abs`` / ``max`` / ``round`` / ``conj`` exist in both
    ``std`` and the DaCe headers, so they are deliberately absent from the list -- flagging them
    would reject correct ``std::``-based output.
    """
    assert_standalone('x = mylib::sign(y) + std::abs(z) + std::max(a, b);')


def test_measured_leaks_are_all_listed():
    """Every construct measured coming out of the symbolic printer is covered by the name list."""
    uncovered = sorted({name for _, name in UNQUALIFIED_LEAKS} - UNQUALIFIED_RUNTIME_FUNCTIONS)
    assert not uncovered, (f'measured unqualified runtime leaks {uncovered} are missing from '
                           'UNQUALIFIED_RUNTIME_FUNCTIONS, so MPR output carrying them would pass the gate')


def test_banned_patterns_are_distinct():
    """Each banned pattern carries its own meaning string, so a failure names one cause."""
    meanings = [meaning for _, meaning in BANNED_PATTERNS]
    assert len(meanings) == len(set(meanings)), f'duplicate banned-pattern meanings: {meanings}'


@pytest.mark.parametrize('code', [AXPY_MPR, SUM_MPR], ids=['axpy', 'sum'])
def test_mpr_shaped_code_is_standalone(code):
    """The specification snippets pass the string gate -- MPR's target shape is reachable."""
    assert_standalone(code)


def test_dace_include_fails_the_bare_build():
    """The build really rejects a leaked DaCe header.

    This is the harness's central claim: the compile runs in a fresh directory with NO ``-I``, so a
    ``#include <dace/dace.h>`` cannot resolve off an inherited include path. If this test ever
    passes the compile, every other build assertion here is worthless.
    """
    leaked = '#include <dace/dace.h>\nextern "C" void mpr_leak() {}\n'
    with pytest.raises(AssertionError) as excinfo:
        compile_standalone(leaked, name='mpr_leak')
    assert 'does not build with a bare host compiler' in str(excinfo.value)
    assert 'dace/dace.h' in str(excinfo.value), 'the compiler diagnostic must name the header it could not find'


def test_entry_argtypes_follow_the_sdfg_arglist():
    """ctypes signature order and kinds mirror ``SDFG.arglist``: pointers first, then scalars by value."""
    sdfg = axpy_sdfg()
    assert list(sdfg.arglist()) == ['A', 'B', 'N'], (f'arglist order changed: {list(sdfg.arglist())}; the MPR entry '
                                                     'point signature is defined by it')
    argtypes = entry_argtypes(sdfg)
    assert argtypes == [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32], argtypes


def test_standalone_call_writes_in_place():
    """A built MPR-shaped kernel runs through ctypes and its writes land in the caller's array."""
    sdfg = axpy_sdfg()
    library = build_standalone(AXPY_MPR, name='mpr_axpy')
    n = 64
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    call_standalone(library, sdfg, {'A': a, 'B': b, 'N': n})
    assert_matches({'B': a * 2.0 + n}, {'B': b}, label='mpr_axpy')


def sum_sdfg():
    """SDFG whose arglist defines ``mpr_sum``'s signature, with a WCR accumulation into ``out``.

    The map body is not decoration: ``SDFG.arglist`` derives its symbol arguments from
    ``used_symbols``, so a symbol no node references never reaches the entry point at all.
    """
    N = dace.symbol('N')
    sdfg = dace.SDFG('mpr_sum')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state('main')
    read, write = state.add_read('A'), state.add_write('out')
    entry, exit_node = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('t', {'inp'}, {'acc'}, 'acc = inp')
    state.add_memlet_path(read, entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet,
                          exit_node,
                          write,
                          src_conn='acc',
                          memlet=dace.Memlet('out[0]', wcr='lambda a, b: a + b'))
    return sdfg


def test_reduction_clause_kernel_runs():
    """The only WCR form MPR may emit -- an OpenMP reduction clause -- builds bare and is correct."""
    sdfg = sum_sdfg()
    assert list(sdfg.arglist()) == ['A', 'out', 'N'], (f'arglist order changed: {list(sdfg.arglist())}; the MPR '
                                                       'entry point signature is defined by it')
    library = build_standalone(SUM_MPR, name='mpr_sum')
    n = 1024
    a = np.random.default_rng(0).random(n)
    out = np.zeros(1, dtype=np.float64)
    call_standalone(library, sdfg, {'A': a, 'out': out, 'N': n})
    assert_matches({'out': np.array([a.sum()])}, {'out': out}, label='mpr_sum')


@pytest.mark.parametrize('code,name', [(AXPY_MPR, 'mpr_axpy'), (SUM_MPR, 'mpr_sum')], ids=['axpy', 'sum'])
def test_mpr_shaped_code_compiles_without_warnings(code, name):
    """Warnings are errors: the specification snippets build clean under ``-Wall -Wextra``."""
    diagnostics = compile_diagnostics(code, name=name)
    assert diagnostics == '', f'{name}: MPR-shaped code produced compiler warnings\n{diagnostics}'


def test_call_rejects_an_argument_the_sdfg_never_uses():
    """An extra name is refused, not dropped.

    A symbol no node references is absent from ``SDFG.arglist``, so passing it would leave the
    kernel reading an uninitialized extent and the test would compare against garbage.
    """
    sdfg = axpy_sdfg()
    library = build_standalone(AXPY_MPR, name='mpr_axpy')
    with pytest.raises(AssertionError, match='not in the SDFG arglist'):
        call_standalone(library, sdfg, {
            'A': np.zeros(4),
            'B': np.zeros(4),
            'N': 4,
            'M': 4,
        })


def test_assert_matches_rejects_a_wrong_answer():
    """The numeric gate fails on a real divergence -- it is not a tautology."""
    with pytest.raises(AssertionError, match='diverges from the SDFG'):
        assert_matches({'x': np.zeros(4)}, {'x': np.ones(4)}, label='probe')
