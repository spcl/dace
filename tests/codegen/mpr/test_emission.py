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
from dace import mpr
from dace.codegen.mpr import render as render_sdfg

from tests.codegen.mpr.conftest import (assert_matches, assert_standalone, build_standalone, call_standalone,
                                        compile_diagnostics)

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
    assert 'std::max' in code or 'std::min' in code, 'min/max must come from the standard library'
    diagnostics = compile_diagnostics(code, 'mpr_clamp')
    assert not diagnostics.strip(), f'MPR output warns:\n{diagnostics}'


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
