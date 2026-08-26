# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end MPR rendering: SDFG in, self-contained C++ out, same numbers.

Every test here follows one shape -- render with :func:`dace.mpr`, assert the text is standalone,
build it with a bare compiler in an empty directory, run it through ctypes, compare against a numpy
reference. What differs between them is the SDFG PATTERN, because that is what decides which
printer, which allocation path and which lowering table entry MPR reaches:

* elementwise maps and multi-dimensional index helpers (the ``<array>_idx`` shape),
* a WCR accumulation, which must fold to an OpenMP ``reduction`` clause rather than an atomic,
* a sequential loop wrapping a parallel map (a ``LoopRegion``, and the loop counter's declaration),
* symbolic extents and integer index arithmetic (``int_floor``, never a truncating ``/``),
* the maths lowering table reached through a TASKLET body, not only through a memlet subset,
* mixed precision, integer types, and a strided/transposed access.

Numeric equality is necessary but not sufficient: a rendering that dropped the ``omp parallel for``
would still produce the right numbers, so each test also asserts the structure that makes the
output an MPR -- the parallel pragma, the reduction clause, the index helper, the entry signature.
"""
import re

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import FindFirst
from dace import mpr
from dace.codegen.mpr import render as render_sdfg
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (insert_assumption_guards,
                                                                                set_symbol_nonnegative_assumptions)

from tests.codegen.mpr.conftest import (assert_matches, assert_standalone, build_standalone, call_standalone,
                                        compile_diagnostics, wcr_sdfg)

N = dace.symbol('N')
M = dace.symbol('M')


def render(program, name: str, simplify: bool = True):
    """``(sdfg, code)`` for a ``@dace.program``, rendered under its own kernel name.

    The name is set on the SDFG because it becomes the entry point's symbol, which
    :func:`call_standalone` looks up -- two tests sharing a name would build two different kernels
    into the same symbol.

    The SDFG returned is the PREPARED one MPR rendered, not the one built here: expanding a library
    node can introduce an extent symbol, which makes the two argument lists differ (see
    :class:`dace.codegen.mpr.Rendering`). Calling with the wrong one drops that argument.
    """
    sdfg = program.to_sdfg(simplify=simplify)
    sdfg.name = name
    rendering = render_sdfg(sdfg)
    return rendering.sdfg, rendering.code


def run(sdfg, code, arguments, name: str):
    """Build ``code`` and invoke it on ``arguments`` (in place, as MPR's entry point writes)."""
    assert_standalone(code, name)
    call_standalone(build_standalone(code, name), sdfg, arguments)


def test_elementwise_map_is_rendered_parallel():
    """The base case: one parallel map, one index helper per array, one entry point."""

    @dace.program
    def scale_add(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
        y[:] = a * x + y

    sdfg, code = render(scale_add, 'mpr_axpy')
    assert 'extern "C" void mpr_axpy(' in code, 'MPR must export the SDFG under its own name'
    assert '#pragma omp parallel for' in code, 'a data-parallel map must render as an OpenMP loop'
    assert 'x_idx(' in code, 'array accesses go through the generated <array>_idx helper'

    n = 128
    x = np.random.rand(n)
    y = np.random.rand(n)
    expected = 2.5 * x + y
    run(sdfg, code, {'x': x, 'y': y, 'N': n, 'a': 2.5}, 'mpr_axpy')
    assert_matches({'y': expected}, {'y': y}, 'mpr_axpy')


def test_two_dimensional_map_indexes_through_a_stride():
    """A 2-D array: the index helper carries the stride, so the access is not just ``p[i]``."""

    @dace.program
    def transpose_add(a: dace.float64[M, N], b: dace.float64[N, M]):
        for i, j in dace.map[0:M, 0:N]:
            b[j, i] = a[i, j] + 1.0

    sdfg, code = render(transpose_add, 'mpr_transpose')
    assert '#pragma omp parallel for' in code
    # The helper takes the two indices, then whichever extents the stride arithmetic needs (a
    # symbolic shape is passed in rather than baked in), so match the leading two and not the arity.
    assert re.search(r'a_idx\(int64_t __d0, int64_t __d1[,)]',
                     code), ('a 2-D array needs a two-index helper; a one-index one would mean the shape was lost')

    m, n = 12, 20
    a = np.random.rand(m, n)
    b = np.zeros((n, m))
    expected = a.T + 1.0
    run(sdfg, code, {'a': a, 'b': b, 'M': m, 'N': n}, 'mpr_transpose')
    assert_matches({'b': expected}, {'b': b}, 'mpr_transpose')


def test_wcr_folds_into_an_openmp_reduction():
    """A WCR accumulator must become a ``reduction`` clause.

    MPR allows exactly the tree-reducible form. The alternative lowering is an atomic through
    ``dace::wcr_fixed``, which is a runtime symbol -- so an atomic fallback here would not merely be
    slow, it would fail :func:`assert_standalone`. Asserting the clause pins WHICH of the two ran.
    """

    @dace.program
    def total(x: dace.float64[N], out: dace.float64[1]):
        out[0] = np.sum(x)

    sdfg, code = render(total, 'mpr_sum')
    assert 'reduction(' in code, 'a WCR accumulation must fold into an OpenMP reduction clause'

    n = 4096
    x = np.random.rand(n)
    out = np.zeros(1)
    run(sdfg, code, {'x': x, 'out': out, 'N': n}, 'mpr_sum')
    assert_matches({'out': np.array([x.sum()])}, {'out': out}, 'mpr_sum')


#: Non-conflicting conflict resolutions, with the statement each must render to and its oracle.
#: ``Sub`` and ``Div`` are not in the table -- no OpenMP clause names them, so they reach MPR as
#: ``Custom``, which is why one entry per SPELLING is not the same as one entry per reduction type.
NON_CONFLICTING_WCR = [
    ('sub', 'lambda p, q: p - q', '*(out + i) = (*(out + i) - (y));', lambda o, x: o - x),
    ('div', 'lambda p, q: p / q', '*(out + i) = (*(out + i) / (y));', lambda o, x: o / x),
    ('exchange', 'lambda p, q: q', '*(out + i) = (y);', lambda o, x: x),
    ('xor', 'lambda p, q: p != q', '*(out + i) = (*(out + i) != (y));', lambda o, x: (o != x).astype(np.float64)),
    ('custom', 'lambda p, q: p * q + 1.0', '*(out + i) = ((*(out + i) * (y)) + 1.0);', lambda o, x: o * x + 1.0),
]


@pytest.mark.parametrize('label, resolution, statement, oracle', NON_CONFLICTING_WCR)
def test_non_conflicting_wcr_renders_without_the_reduction_runtime(label, resolution, statement, oracle):
    """Each resolution becomes a plain assignment, identical in both dialects.

    A conflicting WCR is refused, and a tree-reducible one folds into an ``reduction`` clause
    (:func:`test_wcr_folds_into_an_openmp_reduction`). This is the third case: no conflict at all,
    where the runtime would take ``wcr_custom<T>::reduce`` -- ``*ptr = wcr(*ptr, value)``, no
    critical section. Reproducing that is a plain assignment, so reaching for ``dace::wcr_fixed``
    or a C++ lambda would be both unnecessary and unrenderable.

    The statement is asserted verbatim because the emitted text IS the product here, and the two
    dialects are compared against each other: a C++-only spelling (a lambda) would pass a numeric
    check and still leave the C dialect with nothing to emit.
    """
    rendering = render_sdfg(wcr_sdfg(f'mpr_resolve_{label}', resolution))
    cpp_code = rendering.code
    c_code = render_sdfg(wcr_sdfg(f'mpr_resolve_{label}', resolution), language='c').code
    assert statement in cpp_code, f'{label}: expected {statement!r} in the C++ rendering'
    assert statement in c_code, f'{label}: expected {statement!r} in the C rendering'
    for leaked in ('wcr_fixed', 'wcr_custom'):
        assert leaked not in cpp_code and leaked not in c_code, f'{label}: {leaked} leaked into the output'

    rng = np.random.default_rng(0)
    if label == 'xor':  # a truth value, or the oracle is the constant 1.0 and asserts nothing
        a, out = rng.integers(0, 2, 32).astype(np.float64), rng.integers(0, 2, 32).astype(np.float64)
        assert (a == out).any() and (a != out).any(), 'both xor outcomes have to occur in the inputs'
    else:
        a, out = rng.random(32) + 0.5, rng.random(32) + 0.5  # away from zero, so the division is conditioned
    expected = oracle(out.copy(), a)
    run(rendering.sdfg, cpp_code, {'a': a, 'out': out}, f'mpr_resolve_{label}')
    assert_matches({'out': expected}, {'out': out}, f'mpr_resolve_{label}')


def test_sequential_loop_around_a_parallel_map():
    """A time loop over a parallel body: the loop stays sequential, the body stays parallel.

    This is the shape MPR exists to make visible -- the maximally parallel rendering of a program
    whose outer axis genuinely carries a dependence.
    """

    @dace.program
    def diffuse(a: dace.float64[N], steps: dace.int32):
        for _ in range(steps):
            a[1:-1] = 0.25 * (a[:-2] + 2.0 * a[1:-1] + a[2:])

    sdfg, code = render(diffuse, 'mpr_diffuse')
    assert '#pragma omp parallel for' in code, 'the inner stencil sweep is data-parallel'
    body = code[code.index('extern "C" void mpr_diffuse('):]
    outer = body.index('for (')
    inner = body.index('#pragma omp parallel for')
    assert outer < inner, ('the sequential time loop must enclose the parallel sweep; the reverse order would '
                           'mean the carried dependence was parallelized')

    n = 64
    a = np.random.rand(n)
    expected = a.copy()
    for _ in range(5):
        expected[1:-1] = 0.25 * (expected[:-2] + 2.0 * expected[1:-1] + expected[2:])
    run(sdfg, code, {'a': a, 'N': n, 'steps': 5}, 'mpr_diffuse')
    assert_matches({'a': expected}, {'a': a}, 'mpr_diffuse')


def test_maths_functions_reach_the_standard_library():
    """Maths in a TASKLET body, not a memlet subset -- the other printer into the lowering table."""

    @dace.program
    def mixed_maths(x: dace.float64[N], y: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                xin << x[i]
                yout >> y[i]
                yout = math.sqrt(math.fabs(xin)) + math.exp(-math.fabs(xin))

    sdfg, code = render(mixed_maths, 'mpr_maths')
    assert 'std::sqrt' in code, 'sqrt must be lowered to the standard library, not dace::math::sqrt'
    assert 'std::exp' in code

    n = 256
    x = np.random.rand(n) * 4.0 - 2.0
    y = np.zeros(n)
    expected = np.sqrt(np.abs(x)) + np.exp(-np.abs(x))
    run(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mpr_maths')
    assert_matches({'y': expected}, {'y': y}, 'mpr_maths')


def test_integer_index_arithmetic_stays_integer():
    """A symbolic extent divided by a literal: integer division, never a rational.

    ``N // 2`` reaches C++ through the symbolic printer. Emitting it as a floating division that is
    then truncated would still give the right answer for these inputs and the wrong one for an odd
    ``N``, so the numeric check below uses an ODD extent.
    """

    @dace.program
    def halves(x: dace.int64[N], y: dace.int64[N // 2]):
        for i in dace.map[0:N // 2]:
            y[i] = x[2 * i] + x[2 * i + 1]

    sdfg, code = render(halves, 'mpr_halves')
    assert 'int_floor' not in code, 'int_floor is a DaCe runtime function; MPR lowers it to C++ division'

    n = 65
    x = np.arange(n, dtype=np.int64)
    y = np.zeros(n // 2, dtype=np.int64)
    expected = x[0:2 * (n // 2):2] + x[1:2 * (n // 2):2]
    run(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mpr_halves')
    assert_matches({'y': expected}, {'y': y}, 'mpr_halves')


def test_single_precision_and_integer_types_survive():
    """fp32 stays fp32 and int32 stays int32: a widened accumulator would pass a loose tolerance."""

    @dace.program
    def weighted(x: dace.float32[N], k: dace.int32[N], y: dace.float32[N]):
        for i in dace.map[0:N]:
            y[i] = x[i] * dace.float32(k[i])

    sdfg, code = render(weighted, 'mpr_mixed_types')
    assert 'float ' in code and 'int32_t' in code or 'int ' in code

    n = 100
    x = np.random.rand(n).astype(np.float32)
    k = np.arange(n, dtype=np.int32)
    y = np.zeros(n, dtype=np.float32)
    expected = x * k.astype(np.float32)
    run(sdfg, code, {'x': x, 'k': k, 'y': y, 'N': n}, 'mpr_mixed_types')
    assert_matches({'y': expected}, {'y': y}, 'mpr_mixed_types')
    assert y.dtype == np.float32


def test_matrix_multiply_renders_as_loops():
    """A matmul: the one place a library node would otherwise reach for BLAS.

    MPR has no BLAS to call, so the rendering must be loops. The reference is numpy's, so a wrong
    accumulation order would still pass the fp64 tolerance -- the structural assertion is what
    states that no library call survived.
    """

    @dace.program
    def matmul(a: dace.float64[M, N], b: dace.float64[N, M], c: dace.float64[M, M]):
        c[:] = a @ b

    sdfg, code = render(matmul, 'mpr_matmul')
    # Comments are stripped first: MPR NAMES the library node it rendered away ("// BLAS gemm"),
    # and that line is the opposite of a leaked BLAS call.
    executable = '\n'.join(line for line in code.splitlines() if not line.strip().startswith('//'))
    for banned in ('cblas', 'dgemm', 'MKL', 'BLAS'):
        assert banned not in executable, (f'MPR rendered a BLAS call ({banned}); it must render the loop nest '
                                          'instead')
    assert '// BLAS gemm' in code, 'the rendering must still say what the loop nest used to be'

    m, n = 24, 16
    a = np.random.rand(m, n)
    b = np.random.rand(n, m)
    c = np.zeros((m, m))
    run(sdfg, code, {'a': a, 'b': b, 'c': c, 'M': m, 'N': n}, 'mpr_matmul')
    assert_matches({'c': a @ b}, {'c': c}, 'mpr_matmul')


def test_strided_copy_renders_as_a_loop_not_a_runtime_call():
    """A non-contiguous copy is where ``dace::CopyND`` would appear.

    The copy library node's ``Auto`` implementation picks a memcpy for a contiguous buffer and the
    runtime's N-dimensional strided copy otherwise. The second is a template from a DaCe header, so
    MPR has to have made the choice itself -- and a transposed slice assignment is the shape that
    forces it.
    """

    @dace.program
    def strided(a: dace.float64[M, N], b: dace.float64[N, M]):
        b[:] = np.transpose(a)

    sdfg, code = render(strided, 'mpr_strided_copy')
    assert 'memcpy' not in code or 'for (' in code, 'a strided copy cannot be a single memcpy'

    m, n = 9, 14
    a = np.random.rand(m, n)
    b = np.zeros((n, m))
    run(sdfg, code, {'a': a, 'b': b, 'M': m, 'N': n}, 'mpr_strided_copy')
    assert_matches({'b': a.T}, {'b': b}, 'mpr_strided_copy')


def test_scan_is_inlined_rather_than_called():
    """A prefix sum must render as code, not as a call into the DaCe scan header.

    ``dace/scan.hpp`` is exactly the kind of dependency MPR cannot have; the standard library's
    ``std::inclusive_scan`` is equally unavailable to it, since the CPU expansion reaches it through
    that header's environment. The pure expansion is loops, which is what has to come out.
    """

    @dace.program
    def prefix(x: dace.float64[N], y: dace.float64[N]):
        y[:] = np.cumsum(x)

    sdfg, code = render(prefix, 'mpr_scan')
    assert 'inclusive_scan' not in code and 'partial_sum' not in code, (
        'the scan reached the standard-library algorithm through the DaCe scan header instead of being inlined')

    n = 512
    x = np.random.rand(n)
    y = np.zeros(n)
    run(sdfg, code, {'x': x, 'y': y, 'N': n}, 'mpr_scan')
    assert_matches({'y': np.cumsum(x)}, {'y': y}, 'mpr_scan')


def test_rendering_is_deterministic():
    """The same SDFG renders to the same text twice.

    MPR output is meant to be read and diffed. A rendering that reordered helpers between runs --
    from a set iteration or a dictionary keyed on ``id()`` -- would make every diff noise.
    """

    @dace.program
    def saxpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
        y[:] = a * x + y

    first = render(saxpy, 'mpr_determinism')[1]
    second = render(saxpy, 'mpr_determinism')[1]
    assert first == second, 'MPR rendered the same SDFG two different ways'


def test_output_builds_without_warnings():
    """``-Wall -Wextra`` clean. A warning in copy-pasteable output is a defect in the output."""

    @dace.program
    def clamp(x: dace.float64[N], y: dace.float64[N]):
        for i in dace.map[0:N]:
            y[i] = min(max(x[i], 0.0), 1.0)

    _, code = render(clamp, 'mpr_clamp')
    assert 'mpr_max' in code or 'mpr_min' in code, 'min/max must be the runtime form, not the standard one'
    diagnostics = compile_diagnostics(code, 'mpr_clamp')
    assert not diagnostics.strip(), f'MPR output warns:\n{diagnostics}'


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_minmax_follows_python_on_nan(language):
    """``max``/``min`` keep the EARLIER operand when the comparison is false.

    That is Python's rule, and a ``@dace.program`` is Python, so ``max(x, 0.0)`` keeps a NaN and
    ``max(0.0, x)`` swallows it -- the two orders answer differently, which is what makes this
    worth pinning. The runtime and both MPR dialects have to agree on it; Python is the oracle for
    the runtime, and the compiled SDFG is the oracle for MPR.
    """

    @dace.program
    def clamp_nan(x: dace.float64[4], kept: dace.float64[4], swallowed: dace.float64[4]):
        for i in dace.map[0:4]:
            kept[i] = min(max(x[i], 0.0), 1.0)
            swallowed[i] = max(0.0, x[i])

    sdfg = clamp_nan.to_sdfg(simplify=True)
    sdfg.name = 'mpr_minmax_nan_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)

    x = np.array([np.nan, -1.0, 0.5, 2.0])
    expected_kept = np.array([min(max(v, 0.0), 1.0) for v in x])
    expected_swallowed = np.array([max(0.0, v) for v in x])
    assert np.isnan(expected_kept[0]) and not np.isnan(expected_swallowed[0]), \
        'Python stopped distinguishing the two argument orders, so this asserts nothing'

    kept, swallowed = np.zeros(4), np.zeros(4)
    sdfg(x=x.copy(), kept=kept, swallowed=swallowed)
    assert np.array_equal(kept, expected_kept,
                          equal_nan=True), f'the runtime clamps to {kept}, Python to {expected_kept}'
    assert np.array_equal(swallowed, expected_swallowed, equal_nan=True)

    mpr_kept, mpr_swallowed = np.zeros(4), np.zeros(4)
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {
        'x': x,
        'kept': mpr_kept,
        'swallowed': mpr_swallowed
    })
    assert np.array_equal(mpr_kept, kept, equal_nan=True), f'MPR clamps to {mpr_kept}, the SDFG to {kept}'
    assert np.array_equal(mpr_swallowed, swallowed, equal_nan=True)


def test_persistent_lifetime_is_demoted_not_left_in_a_state():
    """A persistent transient has nowhere to live in MPR, so it must become an SDFG-lifetime one."""

    @dace.program
    def with_persistent(x: dace.float64[N], y: dace.float64[N]):
        y[:] = (x + 1.0) * (x + 2.0)

    sdfg = with_persistent.to_sdfg(simplify=True)
    sdfg.name = 'mpr_persistent'
    persistent = [name for name, desc in sdfg.arrays.items() if desc.transient]
    assert persistent, 'this test needs a transient to mark persistent, or it asserts nothing'
    for name in persistent:
        sdfg.arrays[name].lifetime = dace.dtypes.AllocationLifetime.Persistent

    rendering = render_sdfg(sdfg)
    assert_standalone(rendering.code, 'mpr_persistent')
    assert all(sdfg.arrays[name].lifetime == dace.dtypes.AllocationLifetime.Persistent
               for name in persistent), ("MPR must render a COPY: the caller's SDFG still describes persistent storage")
    assert all(rendering.sdfg.arrays[name].lifetime == dace.dtypes.AllocationLifetime.SDFG
               for name in persistent), ('the rendered copy must hold the demoted lifetime')

    n = 32
    x = np.random.rand(n)
    y = np.zeros(n)
    call_standalone(build_standalone(rendering.code, 'mpr_persistent'), rendering.sdfg, {'x': x, 'y': y, 'N': n})
    assert_matches({'y': (x + 1.0) * (x + 2.0)}, {'y': y}, 'mpr_persistent')


def test_gpu_schedules_are_refused_with_a_reason():
    """MPR renders one host unit. A GPU SDFG must say so, not emit half a program."""

    @dace.program
    def on_gpu(x: dace.float64[N], y: dace.float64[N]):
        y[:] = x * 2.0

    sdfg = on_gpu.to_sdfg(simplify=True)
    sdfg.name = 'mpr_gpu'
    sdfg.apply_gpu_transformations()
    with pytest.raises(NotImplementedError, match='host translation unit'):
        mpr(sdfg)


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_complex_containers_render_in_both_dialects(language):
    """A complex-typed CONTAINER, not just a complex expression inside a tasklet.

    The element type reaches the text through the entry signature and the transient declarations,
    which no expression printer sees, so this is the case a lowering table alone does not cover.
    """

    @dace.program
    def scale(a: dace.complex128[16], b: dace.complex128[16]):
        b[:] = a * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    sdfg.name = 'mpr_complex_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)
    expected = 'double _Complex' if language == 'c' else 'std::complex<double>'
    assert expected in rendering.code, f'the container type is not spelled {expected}'
    assert_standalone(rendering.code, sdfg.name, language=language)

    a = np.random.rand(16) + 1j * np.random.rand(16)
    b = np.zeros(16, dtype=np.complex128)
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {'a': a, 'b': b})
    assert np.allclose(b, a * 2.0)


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_cholesky_renders_as_loops_with_no_library(language):
    """``np.linalg.cholesky`` reaches a library node that only vendor BLAS implements.

    MPR renders it through the pure expansion instead, so the output links against nothing. The
    provenance comment is asserted too: a factorization rendered as anonymous loops is exactly what
    the comment exists to prevent.
    """

    @dace.program
    def factorize(A: dace.float64[8, 8], L: dace.float64[8, 8]):
        L[:] = np.linalg.cholesky(A)

    sdfg = factorize.to_sdfg(simplify=True)
    sdfg.name = 'mpr_cholesky_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)
    assert_standalone(rendering.code, sdfg.name, language=language)
    assert '// Cholesky factorization' in rendering.code, 'the expansion rendered without its provenance comment'

    matrix = np.random.rand(8, 8)
    a = matrix @ matrix.T + 8 * np.eye(8)
    result = np.zeros((8, 8))
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {
        'A': a.copy(),
        'L': result
    })
    assert np.allclose(result, np.linalg.cholesky(a))


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_linear_solve_renders_as_loops_with_no_library(language):
    """``np.linalg.solve`` is the other corpus call whose node only vendor BLAS implemented."""

    @dace.program
    def solve(A: dace.float64[6, 6], B: dace.float64[6, 3], X: dace.float64[6, 3]):
        X[:] = np.linalg.solve(A, B)

    sdfg = solve.to_sdfg(simplify=True)
    sdfg.name = 'mpr_solve_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)
    assert_standalone(rendering.code, sdfg.name, language=language)
    assert '// solve the linear system' in rendering.code, 'the expansion rendered without its provenance comment'

    a = np.random.rand(6, 6) + 6 * np.eye(6)
    b = np.random.rand(6, 3)
    x = np.zeros((6, 3))
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {
        'A': a.copy(),
        'B': b.copy(),
        'X': x
    })
    assert np.allclose(x, np.linalg.solve(a, b))


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_matrix_inverse_renders_as_loops_with_no_library(language):
    """``np.linalg.inv`` is the third corpus call whose node only vendor BLAS implemented."""

    @dace.program
    def invert(A: dace.float64[6, 6], B: dace.float64[6, 6]):
        B[:] = np.linalg.inv(A)

    sdfg = invert.to_sdfg(simplify=True)
    sdfg.name = 'mpr_inv_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)
    assert_standalone(rendering.code, sdfg.name, language=language)

    a = np.random.rand(6, 6) + 6 * np.eye(6)
    result = np.zeros((6, 6))
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {
        'A': a.copy(),
        'B': result
    })
    assert np.allclose(result, np.linalg.inv(a))


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_conditional_expression_renders_as_a_ternary(language):
    """``ITE`` needs no helper in either dialect: C and C++ both have ``?:``."""

    @dace.program
    def identity(out: dace.float64[6, 6]):
        for i, j in dace.map[0:6, 0:6]:
            out[i, j] = 1 if i == j else 0

    sdfg = identity.to_sdfg(simplify=True)
    sdfg.name = 'mpr_ite_' + ('cpp' if language == 'c++' else 'c')
    rendering = render_sdfg(sdfg, language=language)
    assert_standalone(rendering.code, sdfg.name, language=language)
    assert re.search(r'\?\s*\(1\)\s*:\s*\(0\)', rendering.code), 'the conditional did not render as a ternary'

    out = np.zeros((6, 6))
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, {'out': out})
    assert np.allclose(out, np.eye(6))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def find_first_sdfg(name: str, implementation: str):
    """An SDFG whose only node is the search an early-exit loop lifts to.

    Built from the library node rather than from a ``@dace.program`` with a ``break`` so that the
    test pins MPR against the node's own contract: which expansion ran, and what its C++ body says.
    ``EarlyExitToFindIndex`` is what puts this node into a real kernel, and it is tested where it
    lives.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [1], dace.int64)
    state = sdfg.add_state()
    node = FindFirst('ff', predicate='_a[__i] > 0.5', begin=0, end=N)
    node.implementation = implementation
    node.add_in_connector('_a', dace.pointer(dace.float64))
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_a', dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(node, '_out_idx', state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


#: Where the predicate fires, and the answer each position must produce. ``None`` is the no-hit
#: case, whose answer is the exclusive end -- the one a search that forgot its sentinel gets wrong.
#: The span is well past a single chunk, so the parallel expansion really does split it.
FIND_FIRST_SPAN = 100000
FIND_FIRST_HITS = (0, 1, 517, FIND_FIRST_SPAN - 1, None)


def find_first_input(hit):
    """``(a, expected)`` for a search whose predicate fires at ``hit`` and nowhere else."""
    a = np.zeros(FIND_FIRST_SPAN)
    if hit is not None:
        a[hit] = 1.0
    return a, FIND_FIRST_SPAN if hit is None else hit


@pytest.mark.parametrize('implementation', ['pure', 'OpenMP'])
def test_find_first_renders_the_cancelling_search(implementation):
    """A find-first must render as the short-circuiting parallel search, not as a scan of the range.

    ``dace::find_first_index`` lives in ``dace/runtime/include/dace/detect.h``, which MPR cannot
    include, and the value of the construct is that the range past the answer is never read. So the
    structure is asserted as well as the numbers: a rendering that walked the whole range would
    agree on every case below and still not be a find-first.
    """
    sdfg = find_first_sdfg('mpr_find_first_%s' % implementation.lower(), implementation)
    rendering = render_sdfg(sdfg)
    code = rendering.code
    assert 'schedule(dynamic, 1)' in code and 'reduction(min : best)' in code, (
        'the search lost its cancelling parallel form and became a plain scan of the range')
    assert '#pragma omp simd reduction(min : block)' in code, 'the in-chunk scan is no longer vectorized'

    assert_standalone(code, 'mpr_find_first')
    library = build_standalone(code, 'mpr_find_first_%s' % implementation.lower())
    for hit in FIND_FIRST_HITS:
        a, expected = find_first_input(hit)
        out = np.zeros(1, dtype=np.int64)
        call_standalone(library, rendering.sdfg, {'a': a, 'out': out, 'N': FIND_FIRST_SPAN})
        assert out[0] == expected, f'search over a predicate firing at {hit} answered {out[0]}, not {expected}'


def stream_sdfg() -> dace.SDFG:
    """A map writing into a stream, which is the runtime class ``dace::Stream``."""
    sdfg = dace.SDFG('mpr_stream')
    sdfg.add_array('a', [32], dace.float64)
    sdfg.add_stream('s', dace.float64, buffer_size=32, transient=True)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:32'})
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('s'), src_conn='y', memlet=dace.Memlet('s[0]'))
    return sdfg


def consume_sdfg() -> dace.SDFG:
    """A consume scope draining a stream, which is the runtime class ``dace::Consume``.

    Wired by hand rather than through ``add_memlet_path`` because the scope's stream arrives on the
    named ``IN_stream`` connector, which the path helper does not know to bind.
    """
    sdfg = dace.SDFG('mpr_consume')
    sdfg.add_stream('s', dace.float64, buffer_size=32, transient=True)
    sdfg.add_array('out', [32], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_consume('c', ('p', '2'))
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x + 1.0')
    state.add_edge(state.add_read('s'), None, entry, 'IN_stream', dace.Memlet('s[0]'))
    state.add_edge(entry, 'OUT_stream', tasklet, 'x', dace.Memlet('s[0]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('out'), src_conn='y', memlet=dace.Memlet('out[0]'))
    return sdfg


def vector_wcr_sdfg() -> dace.SDFG:
    """A non-conflicting WCR on a VECTOR element type, which is ``dace::vec<T, N>``."""
    sdfg = dace.SDFG('mpr_vector_wcr')
    element = dace.vector(dace.float64, 4)
    sdfg.add_array('a', [8], element)
    sdfg.add_array('out', [8], element)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:8'})
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('out'),
                          src_conn='y',
                          memlet=dace.Memlet(data='out', subset='i', wcr='lambda p, q: p + q'))
    return sdfg


#: The three constructs whose only implementation is a DaCe runtime class, with the phrase each
#: refusal has to name. They are pinned together because they fail for one reason -- a template
#: carrying state (a queue, a quiescence counter, a SIMD element) that MPR cannot inline -- and the
#: point of the test is that each says WHICH construct, rather than failing later on the
#: self-containment assertion, whose message names ``dace::`` and not the container it came from.
RUNTIME_ONLY_CONSTRUCTS = [
    ('stream', stream_sdfg, 'a Stream is the runtime class'),
    ('consume', consume_sdfg, 'a consume scope is driven'),
    ('vector_wcr', vector_wcr_sdfg, 'the vector type is a DaCe runtime template'),
]


@pytest.mark.parametrize('label, builder, reason', RUNTIME_ONLY_CONSTRUCTS)
@pytest.mark.parametrize('language', ('c++', 'c'))
def test_runtime_only_constructs_are_refused_with_a_reason(label, builder, reason, language):
    """Refused in BOTH dialects, and refused by name.

    The dialect is parametrized because these are not a C-only gap: neither spelling can hold a
    lock-free queue, so a C++ rendering that quietly succeeded would be the bug.
    """
    sdfg = builder()
    sdfg.validate()  # the refusal has to be MPR's, not a malformed SDFG the builder wrote
    with pytest.raises(NotImplementedError, match=re.escape(reason)):
        mpr(sdfg, language=language)


def assumption_guard_sdfg(name: str) -> dace.SDFG:
    """A map over a symbolic extent, carrying canonicalization's OWN assumption guard.

    The guard is inserted by the pass rather than written here, because the exact spelling is the
    thing under test: ``insert_assumption_guards`` traps a violated assumption with
    ``if ((N < 0)) {{ std::abort(); }}`` and then DEDUPS its own guards by searching tasklet bodies
    for the literal ``std::abort``, so the spelling cannot be changed at the source. A hand-written
    trap would keep passing after the pass moved on.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x * 2.0')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('out'), src_conn='y', memlet=dace.Memlet('out[i]'))
    set_symbol_nonnegative_assumptions(sdfg)
    insert_assumption_guards(sdfg)
    return sdfg


@pytest.mark.parametrize('language', ('c++', 'c'))
def test_the_assumption_guard_renders_in_both_dialects(language):
    """Canonicalization's trap is a body no printer sees, and each dialect has to spell it.

    The guard tasklet is emitted verbatim, so ``std::abort`` reaches the text without passing
    through the expression printers and without being a ``dace::`` name -- neither lowering lane
    would see it. C++ therefore needs ``<cstdlib>`` in the preamble, which nothing else pulls in,
    and C needs the name itself rewritten: ``std::`` is not a namespace there, it is a syntax
    error. Both legs are built by their own driver in an empty directory, which is what turns a
    missing declaration into a failure rather than an inherited include path.
    """
    sdfg = assumption_guard_sdfg(f'mpr_guard_{"cpp" if language == "c++" else "c"}')
    guards = [
        node.code.as_string for state in sdfg.states() for node in state.nodes()
        if isinstance(node, dace.sdfg.nodes.Tasklet) and 'std::abort' in node.code.as_string
    ]
    assert guards, 'the pass inserted no guard, so this test would assert nothing'

    rendering = render_sdfg(sdfg, language=language)
    assert_standalone(rendering.code, sdfg.name, language=language)
    if language == 'c':
        assert 'std::abort' not in rendering.code, 'std:: is not a namespace in C'
        assert re.search(r'\babort\(\);', rendering.code), 'the trap itself must survive the rename'
        assert '<stdlib.h>' in rendering.code, 'abort is declared in <stdlib.h>'
    else:
        assert 'std::abort();' in rendering.code, 'the C++ dialect keeps the pass spelling'
        assert '<cstdlib>' in rendering.code, 'std::abort is declared in <cstdlib>'

    library = build_standalone(rendering.code, sdfg.name, language=language)
    rng = np.random.default_rng(0)
    a, out = rng.random(64), np.zeros(64)
    call_standalone(library, rendering.sdfg, {'a': a, 'out': out, 'N': 64})
    assert_matches({'out': a * 2.0}, {'out': out}, sdfg.name)
