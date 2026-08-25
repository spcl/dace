# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end MPR rendering in C: SDFG in, self-contained C23 out, same numbers.

Every kernel here is the same shape as its counterpart in ``test_emission.py`` -- the patterns that
decide which printer, which allocation path and which lowering-table entry MPR reaches -- but
rendered with ``language='c'``, built by ``gcc -std=c23 -Wall -Wextra -fopenmp`` in an empty
directory with no ``-I`` at all, and run through the same ctypes harness.

Building with the C DRIVER is the point: ``g++`` would accept most of this file's output as C++ and
hide exactly the constructs the C dialect exists to avoid. What differs from C++ is not cosmetic --
templates, ``constexpr`` functions, aligned ``new``, ``std::max``, references and the type-generic
maths all have no C spelling -- so each kernel also asserts the C form it must have taken.
"""
import ctypes
import math
import re

import numpy as np
import pytest

import dace
from dace import mpr, mpr_lowering
from dace.codegen import mpr as mpr_module
from dace.codegen.mpr import render as render_sdfg
from dace.codegen.targets.experimental_cpu import format_index_helper

from tests.codegen.mpr.conftest import (assert_matches, assert_standalone, build_standalone, call_standalone,
                                        compile_diagnostics, compile_standalone)

N = dace.symbol('N')
M = dace.symbol('M')


def render_c(program, name: str, simplify: bool = True):
    """``(sdfg, code)`` for a ``@dace.program`` rendered as C under its own kernel name.

    The SDFG returned is the PREPARED one MPR rendered: expanding a library node can introduce an
    extent symbol, which makes the two argument lists differ (see ``dace.codegen.mpr.Rendering``).
    """
    sdfg = program.to_sdfg(simplify=simplify)
    sdfg.name = name
    rendering = render_sdfg(sdfg, language='c')
    return rendering.sdfg, rendering.code


def run_c(sdfg, code, arguments, name: str):
    """Build ``code`` as C and invoke it on ``arguments`` (in place, as MPR's entry point writes)."""
    assert_standalone(code, name, language='c')
    call_standalone(build_standalone(code, name, language='c'), sdfg, arguments)


@dace.program
def c_scale_add(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    y[:] = a * x + y


@dace.program
def c_transpose_add(a: dace.float64[M, N], b: dace.float64[N, M]):
    for i, j in dace.map[0:M, 0:N]:
        b[j, i] = a[i, j] + 1.0


@dace.program
def c_total(x: dace.float64[N], out: dace.float64[1]):
    out[0] = np.sum(x)


@dace.program
def c_diffuse(a: dace.float64[N], steps: dace.int32):
    for _ in range(steps):
        a[1:-1] = 0.25 * (a[:-2] + 2.0 * a[1:-1] + a[2:])


@dace.program
def c_mixed_maths(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            xin << x[i]
            yout >> y[i]
            yout = math.sqrt(math.fabs(xin)) + math.exp(-math.fabs(xin))


@dace.program
def c_halves(x: dace.int64[N], y: dace.int64[N // 2]):
    for i in dace.map[0:N // 2]:
        y[i] = x[2 * i] + x[2 * i + 1]


@dace.program
def c_weighted(x: dace.float32[N], k: dace.int32[N], y: dace.float32[N]):
    for i in dace.map[0:N]:
        y[i] = x[i] * dace.float32(k[i])


@dace.program
def c_matmul(a: dace.float64[M, N], b: dace.float64[N, M], c: dace.float64[M, M]):
    c[:] = a @ b


@dace.program
def c_strided(a: dace.float64[M, N], b: dace.float64[N, M]):
    b[:] = np.transpose(a)


@dace.program
def c_prefix(x: dace.float64[N], y: dace.float64[N]):
    y[:] = np.cumsum(x)


@dace.program
def c_clamp(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = min(max(x[i], 0.0), 1.0)


@dace.program
def c_fp32_maths(x: dace.float32[N], y: dace.float32[N]):
    for i in dace.map[0:N]:
        y[i] = math.sqrt(x[i])


def test_entry_point_has_no_language_linkage():
    """``extern "C"`` is a C++ construct; the C entry point is a plain definition with the same ABI."""
    sdfg, code = render_c(c_scale_add, 'mprc_axpy')
    assert 'extern "C"' not in code, 'extern "C" does not exist in C'
    assert re.search(r'^void mprc_axpy\(', code, re.M), 'MPR must export the SDFG under its own name'
    assert '#pragma omp parallel for' in code, 'a data-parallel map must render as an OpenMP loop'
    assert '#define x_idx(' in code, 'C has no constexpr function, so the index helper must be a macro'

    n = 128
    x = np.random.rand(n)
    y = np.random.rand(n)
    expected = 2.5 * x + y
    run_c(sdfg, code, {'x': x, 'y': y, 'N': n, 'a': 2.5}, 'mprc_axpy')
    assert_matches({'y': expected}, {'y': y}, 'mprc_axpy')


def test_two_dimensional_index_macro_carries_the_stride():
    """A 2-D index helper is a macro whose parameters are each parenthesized and cast."""
    sdfg, code = render_c(c_transpose_add, 'mprc_transpose')
    assert '#pragma omp parallel for' in code
    macro = re.search(r'^#define a_idx\(__d0, __d1[^)]*\) (.*)$', code, re.M)
    assert macro is not None, f'expected a 2-index a_idx macro in:\n{code}'
    assert '((int64_t)(__d0))' in macro.group(1) and '((int64_t)(__d1))' in macro.group(1), macro.group(1)

    m, n = 12, 20
    a = np.random.rand(m, n)
    b = np.zeros((n, m))
    run_c(sdfg, code, {'a': a, 'b': b, 'M': m, 'N': n}, 'mprc_transpose')
    assert_matches({'b': a.T + 1.0}, {'b': b}, 'mprc_transpose')


def test_wcr_folds_into_an_openmp_reduction():
    """A WCR accumulator must become a ``reduction`` clause here too, not a serialized atomic."""
    sdfg, code = render_c(c_total, 'mprc_sum')
    assert 'reduction(' in code, 'a WCR accumulation must fold into an OpenMP reduction clause'

    n = 4096
    x = np.random.rand(n)
    out = np.zeros(1)
    run_c(sdfg, code, {'x': x, 'out': out, 'N': n}, 'mprc_sum')
    assert_matches({'out': np.array([x.sum()])}, {'out': out}, 'mprc_sum')


def test_heap_transients_use_aligned_alloc_and_free():
    """C++ aligned ``new[]``/``operator delete[]`` become ``aligned_alloc``/``free``.

    The byte count must be rounded UP to a multiple of the alignment: C11 requires the size to be an
    integral multiple of it, and an un-rounded call is undefined behaviour that happens to work.
    """
    sdfg, code = render_c(c_diffuse, 'mprc_diffuse')
    allocations = re.findall(r'aligned_alloc\(64, \(\(sizeof\((\w+)\) \* \(([^;]*?)\) \+ 63\) / 64\) \* 64\)', code)
    assert allocations, f'this test needs a heap transient, or it asserts nothing:\n{code}'
    assert code.count('free(') == len(allocations), 'every allocation must be released exactly once'
    assert 'is_trivially_destructible' not in code, 'C has no destructors for that assertion to be about'
    assert '#pragma omp parallel for' in code, 'the inner stencil sweep is data-parallel'
    body = code[code.index('void mprc_diffuse('):]
    assert body.index('for (') < body.index('#pragma omp parallel for'), (
        'the sequential time loop must enclose the parallel sweep')

    n = 64
    a = np.random.rand(n)
    expected = a.copy()
    for _ in range(5):
        expected[1:-1] = 0.25 * (expected[:-2] + 2.0 * expected[1:-1] + expected[2:])
    run_c(sdfg, code, {'a': a, 'N': n, 'steps': 5}, 'mprc_diffuse')
    assert_matches({'a': expected}, {'a': a}, 'mprc_diffuse')


def test_maths_reaches_the_generic_dispatch_macros():
    """Maths in a TASKLET body: C has no ``std::sqrt``, so the call goes through ``mpr_sqrt``."""
    sdfg, code = render_c(c_mixed_maths, 'mprc_maths')
    assert 'mpr_sqrt(' in code and 'mpr_exp(' in code, 'maths must be lowered to MPR\'s generic macros'
    assert '#define mpr_sqrt(' in code, 'the macro a call names must be defined in the same unit'

    n = 256
    x = np.random.rand(n) * 4.0 - 2.0
    y = np.zeros(n)
    expected = np.sqrt(np.abs(x)) + np.exp(-np.abs(x))
    run_c(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mprc_maths')
    assert_matches({'y': expected}, {'y': y}, 'mprc_maths')


def test_a_nested_sdfg_function_takes_no_reference_and_agrees_with_cpp():
    """A nested SDFG emitted as a FUNCTION is where a C++ reference parameter would appear.

    ``prepare`` promotes a written signature scalar to a length-1 array before code generation, so
    the written connector arrives as a pointer; a read-only one binds by value in C. Neither leg may
    show a ``&``, and the two must produce identical integers -- an indirection that changed what
    the kernel computes would not show up in the text at all.
    """
    sdfg, code = render_c(c_halves, 'mprc_halves')
    nested = re.search(r'^inline void \w+\(([^)]*)\)', code, re.M)
    assert nested is not None, f'this test needs a nested SDFG function, or it asserts nothing:\n{code}'
    assert '&' not in nested.group(1), f'C has no reference parameters: {nested.group(1)}'

    n = 65
    x = np.arange(n, dtype=np.int64)
    y = np.zeros(n // 2, dtype=np.int64)
    expected = x[0:2 * (n // 2):2] + x[1:2 * (n // 2):2]
    run_c(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mprc_halves')
    assert_matches({'y': expected}, {'y': y}, 'mprc_halves')

    cxx_sdfg = c_halves.to_sdfg(simplify=True)
    cxx_sdfg.name = 'mprc_halves_cxx'
    cxx = render_sdfg(cxx_sdfg)
    y_cxx = np.zeros(n // 2, dtype=np.int64)
    call_standalone(build_standalone(cxx.code, 'mprc_halves_cxx'), cxx.sdfg, {'x': x, 'y': y_cxx, 'N': n})
    assert np.array_equal(y, y_cxx), 'the C and C++ renderings of one SDFG disagree'


def test_numeric_typecast_is_a_cast_expression():
    """``dace.float32(k)`` is a C++ functional cast; C needs the cast-expression form."""
    sdfg, code = render_c(c_weighted, 'mprc_mixed_types')
    assert '((float)(' in code, f'expected a C cast expression in:\n{code}'
    assert not re.search(r'(?<![\w)])float\(', code), 'a functional cast does not parse in C'

    n = 100
    x = np.random.rand(n).astype(np.float32)
    k = np.arange(n, dtype=np.int32)
    y = np.zeros(n, dtype=np.float32)
    run_c(sdfg, code, {'x': x, 'k': k, 'y': y, 'N': n}, 'mprc_mixed_types')
    assert_matches({'y': x * k.astype(np.float32)}, {'y': y}, 'mprc_mixed_types')
    assert y.dtype == np.float32


def test_matrix_multiply_renders_as_loops():
    """A matmul: the one place a library node would otherwise reach for BLAS."""
    sdfg, code = render_c(c_matmul, 'mprc_matmul')
    executable = '\n'.join(line for line in code.splitlines() if not line.strip().startswith('//'))
    for banned in ('cblas', 'dgemm', 'MKL', 'BLAS'):
        assert banned not in executable, f'MPR rendered a BLAS call ({banned})'
    assert '// BLAS gemm' in code, 'the rendering must still say what the loop nest used to be'

    m, n = 24, 16
    a = np.random.rand(m, n)
    b = np.random.rand(n, m)
    c = np.zeros((m, m))
    run_c(sdfg, code, {'a': a, 'b': b, 'c': c, 'M': m, 'N': n}, 'mprc_matmul')
    assert_matches({'c': a @ b}, {'c': c}, 'mprc_matmul')


def test_strided_copy_renders_as_a_loop():
    """A non-contiguous copy is where ``dace::CopyND`` would appear."""
    sdfg, code = render_c(c_strided, 'mprc_strided_copy')
    assert 'for (' in code, 'a strided copy cannot be a single memcpy'

    m, n = 9, 14
    a = np.random.rand(m, n)
    b = np.zeros((n, m))
    run_c(sdfg, code, {'a': a, 'b': b, 'M': m, 'N': n}, 'mprc_strided_copy')
    assert_matches({'b': a.T}, {'b': b}, 'mprc_strided_copy')


def test_scan_keeps_its_parallel_inscan_form():
    """The prefix scan must stay the ``inscan`` form, not degrade to a sequential loop.

    That form IS the parallel scan, and it is the reason the eight scan helpers exist at all -- a
    rendering that quietly serialized every prefix sum would not be a maximal parallel rendering.
    """
    sdfg, code = render_c(c_prefix, 'mprc_scan')
    assert 'scan_incl_sum(' in code, f'this test needs the scan helper, or it asserts nothing:\n{code}'
    assert '#pragma omp simd reduction(inscan, +:acc)' in code, 'the scan must keep its inscan clause'
    assert '#pragma omp scan inclusive(acc)' in code

    n = 512
    x = np.random.rand(n)
    y = np.zeros(n)
    run_c(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mprc_scan')
    assert_matches({'y': np.cumsum(x)}, {'y': y}, 'mprc_scan')


def test_output_builds_without_warnings():
    """``-Wall -Wextra`` clean under the C driver, including every unused typed helper it emits."""
    _, code = render_c(c_clamp, 'mprc_clamp')
    assert 'mpr_max(' in code and 'mpr_min(' in code, 'min/max must come from MPR\'s typed pair'
    diagnostics = compile_diagnostics(code, 'mprc_clamp', language='c')
    assert not diagnostics.strip(), f'MPR C output warns:\n{diagnostics}'


def test_rendering_is_deterministic():
    """The same SDFG renders to the same C text twice; MPR output exists to be read and diffed."""
    first = render_c(c_scale_add, 'mprc_determinism')[1]
    second = render_c(c_scale_add, 'mprc_determinism')[1]
    assert first == second, 'MPR rendered the same SDFG two different ways'


def test_fp32_maths_does_not_double_round():
    """``<math.h>`` is not type-generic: bare ``sqrt`` on a float promotes to double and rounds twice.

    So the fp32 call has to reach ``sqrtf``. Asserted twice over, because either half alone is weak:
    the emitted dispatch must NAME ``sqrtf`` for a ``float``, and the C result must be BIT-identical
    to the C++ rendering of the same SDFG, which calls ``std::sqrt(float)``.
    """
    sdfg, code = render_c(c_fp32_maths, 'mprc_fp32')
    assert 'mpr_sqrt(' in code, f'this test needs an fp32 sqrt call, or it asserts nothing:\n{code}'
    assert 'float: sqrtf' in code, 'a float argument must select sqrtf, not promote to double'

    n = 257
    # Values whose fp32 square root is NOT representable, so a double rounding is observable.
    x = (np.arange(1, n + 1, dtype=np.float32) * np.float32(1.0000001)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    run_c(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mprc_fp32')

    cxx_sdfg = c_fp32_maths.to_sdfg(simplify=True)
    cxx_sdfg.name = 'mprc_fp32_cxx'
    cxx = render_sdfg(cxx_sdfg)
    y_cxx = np.zeros(n, dtype=np.float32)
    call_standalone(build_standalone(cxx.code, 'mprc_fp32_cxx'), cxx.sdfg, {'x': x, 'y': y_cxx, 'N': n})
    assert np.array_equal(y.view(np.uint32), y_cxx.view(
        np.uint32)), ('the C rendering rounded differently from the C++ one, which is what a promotion to double does')


def test_complex_header_macro_is_undefined():
    """``<complex.h>`` defines ``I``, which is a plausible loop-index name in scientific code."""
    _, code = render_c(c_scale_add, 'mprc_undef_i')
    assert '#include <complex.h>' in code, 'this test asserts nothing unless complex.h is included'
    include = code.index('#include <complex.h>')
    undef = code.index('#undef I')
    assert include < undef, 'the macro must be removed AFTER the header that defines it'
    assert undef < code.index('void mprc_undef_i('), 'and before any code that could use the name'


@dace.program
def c_named_i(I: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = I[i] * 2.0


def test_a_container_named_i_still_renders_and_runs():
    """The reason ``#undef I`` exists, exercised end to end rather than asserted on the text."""
    sdfg, code = render_c(c_named_i, 'mprc_named_i')
    assert re.search(r'\bI\b', code), 'this test needs the container name to survive into the output'

    n = 48
    data = np.random.rand(n)
    y = np.zeros(n)
    run_c(sdfg, code, {'I': data, 'y': y, 'N': n}, 'mprc_named_i')
    assert_matches({'y': data * 2.0}, {'y': y}, 'mprc_named_i')


def test_gpu_schedules_are_refused_with_a_reason():
    """MPR renders one host unit in either language."""

    sdfg = c_scale_add.to_sdfg(simplify=True)
    sdfg.name = 'mprc_gpu'
    sdfg.apply_gpu_transformations()
    with pytest.raises(NotImplementedError, match='host translation unit'):
        mpr(sdfg, language='c')


@pytest.mark.parametrize('language', ['c++', 'c'])
def test_language_selects_the_dialect(language):
    """``dace.mpr`` and ``render`` take the same argument, and ``'c++'`` stays the default."""
    sdfg = c_scale_add.to_sdfg(simplify=True)
    sdfg.name = 'mprc_language_%s' % ('cxx' if language == 'c++' else 'c')
    assert mpr(sdfg, language=language) == render_sdfg(sdfg, language=language).code
    if language == 'c++':
        assert mpr(sdfg) == mpr(sdfg, language='c++'), 'the default must not have moved'


def test_an_unknown_language_is_refused():
    """A typo must name what is available rather than silently rendering C++."""
    sdfg = c_scale_add.to_sdfg(simplify=True)
    sdfg.name = 'mprc_bad_language'
    with pytest.raises(ValueError, match=r"\['c', 'c\+\+'\]"):
        mpr(sdfg, language='fortran')


# -- the index macro, probed directly ------------------------------------------------------------
#
# A macro that mis-parses its argument is a WRONG ANSWER, not a compile error, so the two hazards
# are probed by building and RUNNING the generated macro rather than by matching its text.

_MACRO_PROBE = """
#include <stdint.h>
static inline int64_t mpr_probe_max_l(int64_t a, int64_t b) {{ return (a > b) ? a : b; }}
#define probe_max(a, b) _Generic((a) + (b), long: mpr_probe_max_l)(a, b)
{helper}
void probe(int64_t * out) {{ out[0] = {call}; }}
"""


@pytest.mark.parametrize('call,expected', [
    ('probe_idx(2 + 1, 5)', 3 * 10 + 5),
    ('probe_idx(probe_max(3L, 7L), 5)', 7 * 10 + 5),
],
                         ids=['compound-argument', 'call-argument-with-a-comma'])
def test_index_macro_parses_its_arguments(call, expected):
    """``A_idx(i + 1, j)`` must mean what it says, and a comma inside a call is not a hazard."""
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.STANDALONE_C):
        helper = format_index_helper('unused', 'int64_t', 'probe_idx', ['__d0', '__d1'], '((10 * __d0) + __d1)')
    assert helper.startswith('#define probe_idx('), f'the C dialect must emit a macro, not a function: {helper}'

    code = _MACRO_PROBE.format(helper=helper, call=call)
    library = ctypes.CDLL(compile_standalone(code, 'mpr_idx_probe', language='c'))
    out = np.zeros(1, dtype=np.int64)
    library.probe.argtypes = [ctypes.c_void_p]
    library.probe.restype = None
    library.probe(ctypes.c_void_p(out.ctypes.data))
    assert out[0] == expected, f'{call} expanded to a different expression than it reads as'


def test_index_macro_computes_in_int64():
    """The C++ helper returned ``int64_t``; the macro must keep that, or a large extent overflows."""
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.STANDALONE_C):
        helper = format_index_helper('unused', 'int64_t', 'probe_idx', ['__d0', '__d1'], '((10 * __d0) + __d1)')
    code = _MACRO_PROBE.format(helper=helper, call='probe_idx(300000000, 7)')
    library = ctypes.CDLL(compile_standalone(code, 'mpr_idx_width', language='c'))
    out = np.zeros(1, dtype=np.int64)
    library.probe.argtypes = [ctypes.c_void_p]
    library.probe.restype = None
    library.probe(ctypes.c_void_p(out.ctypes.data))
    assert out[0] == 3000000007, 'the index arithmetic wrapped through 32 bits'


# -- the _Generic min/max dispatch ---------------------------------------------------------------

_MINMAX_PROBE = """
#include <stdint.h>
{definitions}
void probe({ctype} * out, {ctype} a, {ctype} b) {{ out[0] = {call}; }}
"""

#: ``(numpy dtype, C type, a, b)``. The int64 pair is the one that matters: both values are exactly
#: representable in double but their DIFFERENCE is not, so a dispatch that widened through double
#: would return the wrong one of the two.
MINMAX_CASES = [
    (np.int32, 'int32_t', np.int32(-2147483648), np.int32(2147483647)),
    (np.int64, 'int64_t', np.int64(2**53 + 1), np.int64(2**53 + 2)),
    (np.float32, 'float', np.float32(1.5), np.float32(-2.25)),
    (np.float64, 'double', 1.5, -2.25),
]


@pytest.mark.parametrize('dtype,ctype,a,b', MINMAX_CASES, ids=[case[1] for case in MINMAX_CASES])
@pytest.mark.parametrize('name', ['mpr_max', 'mpr_min'])
def test_generic_minmax_selects_the_exact_typed_helper(name, dtype, ctype, a, b):
    """``mpr_max`` must pick the helper for the argument's own type, at full width."""
    definitions = '\n'.join(mpr_lowering.definitions_for({name}, mpr_lowering.Dialect.STANDALONE_C))
    code = _MINMAX_PROBE.format(definitions=definitions, ctype=ctype, call='%s(a, b)' % name)
    library = ctypes.CDLL(compile_standalone(code, 'mpr_minmax_%s_%s' % (name, ctype), language='c'))
    out = np.zeros(1, dtype=dtype)
    scalar = np.ctypeslib.as_ctypes_type(dtype)
    library.probe.argtypes = [ctypes.c_void_p, scalar, scalar]
    library.probe.restype = None
    library.probe(ctypes.c_void_p(out.ctypes.data), a, b)
    expected = max(a, b) if name == 'mpr_max' else min(a, b)
    assert out[0] == expected, f'{name} on {ctype} gave {out[0]!r}, expected {expected!r}'


def test_generic_minmax_has_no_default_association():
    """A type outside the closed list must fail to select, not widen silently through ``double``."""
    for name in ('mpr_max', 'mpr_min'):
        definition = mpr_lowering.C_INLINE_DEFINITIONS[name]
        assert 'default:' not in definition, (f'{name} carries a default association, so an unlisted type would be '
                                              'converted instead of rejected')


# -- the C verify() gate -------------------------------------------------------------------------

#: ``(label, a line of C++ the C dialect must never emit)``. Each one BUILDS inside the DaCe tree,
#: so only this gate stands between it and a translation unit the C driver rejects.
LEAKS = [
    ('std', 'x = std::max(a, b);'),
    ('template', 'template <typename T> T f(T x) { return x; }'),
    ('extern-c', 'extern "C" void kernel(double * x)'),
    ('new', 'p = new (std::align_val_t(64)) double[n];'),
    ('delete', '::operator delete[](p, std::align_val_t(64));'),
    ('constexpr-function', 'static constexpr inline int64_t a_idx(int64_t __d0) { return __d0; }'),
    ('consteval-function', 'static consteval inline int64_t a_size() { return 12; }'),
    ('reference-parameter', 'inline void nest(const double&  x, double&  y) {'),
]


@pytest.mark.parametrize('label,line', LEAKS, ids=[label for label, _ in LEAKS])
def test_the_c_gate_rejects_each_cpp_construct(label, line):
    """Every construct the C dialect had to replace is caught by name, on the finished text."""
    with pytest.raises(RuntimeError, match='not self-contained'):
        mpr_module.verify(line + '\n', 'probe', mpr_lowering.Dialect.STANDALONE_C)


#: What the C gate must NOT reject. A gate with a false positive gets disabled, so the two shapes
#: that look like a banned pattern are pinned here: ``constexpr`` on an OBJECT is how MPR emits an
#: SDFG constant in C23, and a bitwise ``&`` between two operands is not a reference declarator.
ACCEPTED = [
    ('constexpr-object', 'constexpr double ztw1 = 1329.31;'),
    ('bitwise-and', 'if (exponent & 1) { result *= base; }'),
    ('address-of-argument', 'nest(&(x[0]), &(y[i]));'),
]


@pytest.mark.parametrize('label,line', ACCEPTED, ids=[label for label, _ in ACCEPTED])
def test_the_c_gate_accepts_what_the_c_dialect_emits(label, line):
    """The gate's own output must pass it."""
    mpr_module.verify(line + '\n', 'probe', mpr_lowering.Dialect.STANDALONE_C)


def test_the_cpp_gate_still_allows_the_cpp_constructs():
    """The C table applies to the C dialect only; C++ output is unchanged."""
    for _, line in LEAKS:
        mpr_module.verify(line + '\n', 'probe', mpr_lowering.Dialect.STANDALONE)


def thread_local_sdfg(name: str) -> dace.SDFG:
    """A parallel map with a ``CPU_ThreadLocal`` transient, which is allocated per OpenMP thread."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [64], dace.float64)
    sdfg.add_array('b', [64], dace.float64)
    sdfg.add_array('scratch', [8],
                   dace.float64,
                   transient=True,
                   storage=dace.StorageType.CPU_ThreadLocal,
                   lifetime=dace.AllocationLifetime.SDFG)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('outer', {'i': '0:64'}, schedule=dace.ScheduleType.CPU_Multicore)
    stage = state.add_tasklet('stage', {'i_'}, {'o'}, 'o = i_ * 2.0')
    reread = state.add_tasklet('reread', {'i_'}, {'o'}, 'o = i_ + 1.0')
    scratch = state.add_access('scratch')
    state.add_memlet_path(state.add_access('a'), entry, stage, dst_conn='i_', memlet=dace.Memlet('a[i]'))
    state.add_edge(stage, 'o', scratch, None, dace.Memlet('scratch[0]'))
    state.add_edge(scratch, None, reread, 'i_', dace.Memlet('scratch[0]'))
    state.add_memlet_path(reread, exit_node, state.add_access('b'), src_conn='o', memlet=dace.Memlet('b[i]'))
    return sdfg


def test_thread_local_storage_allocates_with_aligned_alloc():
    """``CPU_ThreadLocal`` builds its allocation on its own branch, which once bypassed the C form.

    The branch emitted a ``new``-expression directly instead of going through the statement builder
    every other storage uses, so C rendering produced a C++ allocation that the gate caught but
    could not spell. Both dialects are asserted, because the fix must not move the C++ output.
    """
    rendering = render_sdfg(thread_local_sdfg('mpr_c_threadlocal'), validate=False, language='c')
    assert 'aligned_alloc(64,' in rendering.code, 'the thread-local allocation did not reach the C form'
    assert 'free(scratch)' in rendering.code, 'the matching free did not reach the C form'
    assert '#pragma omp threadprivate(scratch)' in rendering.code, 'the storage stopped being thread-local'
    assert_standalone(rendering.code, 'mpr_c_threadlocal', language='c')

    cpp = render_sdfg(thread_local_sdfg('mpr_cpp_threadlocal'), validate=False).code
    assert 'new (std::align_val_t(64))' in cpp, 'the C++ allocation changed shape'

    a = np.random.rand(64)
    b = np.zeros(64)
    call_standalone(build_standalone(rendering.code, 'mpr_c_threadlocal', language='c'), rendering.sdfg, {
        'a': a,
        'b': b
    })
    assert np.allclose(b, a * 2.0 + 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
