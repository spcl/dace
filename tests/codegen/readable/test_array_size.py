# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the experimental (readable) CPU generator's per-array ``<array>_size``
helpers.

Under ``compiler.cpu.implementation = experimental`` a heap array whose
``total_size`` is symbolic or a compound expression is allocated as
``dace::aligned_alloc<T>(<array>_size(...), align)``, where the generated
``<array>_size`` helper returns that size. Constant sizes get a nullary
``T_size()`` helper (``consteval`` under C++20, ``constexpr`` under C++17); compound symbolic sizes get a ``constexpr``
helper over the sorted free symbols (``T_size(M, N)``); a bare single symbol
(``T[N]``) is deliberately left as the plain name (wrapping it is no readability
win).

Each kernel is compiled + run under both the ``legacy`` and ``experimental``
generators on identical (``copy.deepcopy``'d) inputs; the outputs must be
bit-exact, which proves the routed size expression matches the classic
``sym2cpp(total_size)``. A few cases additionally inspect the generated C++ (the
helper definition and the ``aligned_alloc`` call). CPU kernels run in a forked
child (repo rule) via :func:`conftest.run_isolated`.
"""
import copy
import re
from typing import Callable, Dict

import numpy as np

import dace
from dace.config import Config
from dace.dtypes import StorageType
from tests.codegen.readable.conftest import EXPERIMENTAL, LEGACY, run_isolated, use_implementation


# --------------------------------------------------------------------------- #
# SDFG builders: A (external) -> T (CPU_Heap, size under test) -> B (external)
# --------------------------------------------------------------------------- #
def heap_pipeline_1d(name: str, shape, rng: str) -> dace.SDFG:
    """Build ``T[i] = A[i] + 1`` then ``B[i] = T[i] * 2`` over ``rng``.

    ``A`` and ``B`` are external arrays; only ``T`` is a ``CPU_Heap`` transient, so
    its ``total_size`` (from ``shape``) is the only one routed through an
    ``<array>_size`` helper.

    :param name: SDFG name.
    :param shape: The shared 1-D shape/size expression (int or SymPy expression).
    :param rng: The map range string (e.g. ``'0:N*M'``).
    :return: The validated SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [shape], dace.float64)
    sdfg.add_array('B', [shape], dace.float64)
    sdfg.add_transient('T', [shape], dace.float64, storage=StorageType.CPU_Heap)

    write = sdfg.add_state('write')
    read_a, write_t = write.add_read('A'), write.add_write('T')
    entry, exit_node = write.add_map('write_map', dict(i=rng))
    add = write.add_tasklet('add_one', {'a'}, {'o'}, 'o = a + 1.0')
    write.add_memlet_path(read_a, entry, add, dst_conn='a', memlet=dace.Memlet('A[i]'))
    write.add_memlet_path(add, exit_node, write_t, src_conn='o', memlet=dace.Memlet('T[i]'))

    read = sdfg.add_state_after(write, 'read')
    read_t, write_b = read.add_read('T'), read.add_write('B')
    entry2, exit2 = read.add_map('read_map', dict(i=rng))
    mul = read.add_tasklet('mul_two', {'t'}, {'o'}, 'o = t * 2.0')
    read.add_memlet_path(read_t, entry2, mul, dst_conn='t', memlet=dace.Memlet('T[i]'))
    read.add_memlet_path(mul, exit2, write_b, src_conn='o', memlet=dace.Memlet('B[i]'))

    sdfg.validate()
    return sdfg


def nested_same_name_sdfg(name: str) -> dace.SDFG:
    """Build an outer SDFG whose transient ``T`` (size ``N*M``) nests an SDFG whose
    transient is also named ``T`` but sized ``N*N``.

    DaCe qualifies the nested transient's name (``inner_T``), so the two heap
    allocations resolve to distinct ``inner_T_size(N)`` and ``T_size(M, N)`` helpers
    -- no arity/return collision.

    :param name: Outer SDFG name.
    :return: The validated outer SDFG.
    """
    inner_n = dace.symbol('N')
    inner = dace.SDFG('inner')
    inner.add_array('a', [inner_n], dace.float64)
    inner.add_array('b', [inner_n], dace.float64)
    inner.add_transient('T', [inner_n * inner_n], dace.float64, storage=StorageType.CPU_Heap)
    iw = inner.add_state('write')
    ira, iwt = iw.add_read('a'), iw.add_write('T')
    itk = iw.add_tasklet('w', {'ai'}, {'to'}, 'to = ai')
    iw.add_edge(ira, None, itk, 'ai', dace.Memlet('a[0]'))
    iw.add_edge(itk, 'to', iwt, None, dace.Memlet('T[0]'))
    ir = inner.add_state_after(iw, 'read')
    irt, iwb = ir.add_read('T'), ir.add_write('b')
    itk2 = ir.add_tasklet('r', {'ti'}, {'bo'}, 'bo = ti')
    ir.add_edge(irt, None, itk2, 'ti', dace.Memlet('T[0]'))
    ir.add_edge(itk2, 'bo', iwb, None, dace.Memlet('b[0]'))

    n, m = dace.symbol('N'), dace.symbol('M')
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    sdfg.add_transient('T', [n * m], dace.float64, storage=StorageType.CPU_Heap)
    main = sdfg.add_state('main')
    nsdfg = main.add_nested_sdfg(inner, {'a'}, {'b'}, {'N': 'N'})
    main.add_edge(main.add_read('A'), None, nsdfg, 'a', dace.Memlet('A[0:N]'))
    main.add_edge(nsdfg, 'b', main.add_write('B'), None, dace.Memlet('B[0:N]'))

    later = sdfg.add_state_after(main, 'outer_t')
    outer_t = later.add_access('T')
    seed = later.add_tasklet('seed', {}, {'o'}, 'o = 0.0')
    later.add_edge(seed, 'o', outer_t, None, dace.Memlet('T[0]'))

    sdfg.validate()
    return sdfg


# --------------------------------------------------------------------------- #
# Equivalence + codegen-inspection helpers
# --------------------------------------------------------------------------- #
def run_variant(build: Callable[[str], dace.SDFG], name: str, implementation: str,
                base: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Build + compile + run one variant on a deep copy of ``base``; return outputs.

    Runs in a forked child (repo rule) via ``run_isolated``.

    :param build: Callable building the SDFG from a name.
    :param name: SDFG name for this variant.
    :param implementation: ``legacy`` or ``experimental``.
    :param base: Shared inputs (arrays + symbol scalars); deep-copied per variant.
    :return: The ndarray outputs keyed by name.
    """

    def work() -> Dict[str, np.ndarray]:
        sdfg = build(name)
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}

    with use_implementation(implementation):
        return run_isolated(work)


def assert_bit_exact(build: Callable[[str], dace.SDFG], base_name: str, base: Dict[str,
                                                                                   object]) -> Dict[str, np.ndarray]:
    """Run ``build`` under legacy and experimental; assert every output is bit-exact.

    :param build: Callable building the SDFG from a name.
    :param base_name: Base name for the two variants.
    :param base: Shared inputs, deep-copied per variant.
    :return: The experimental output dict.
    """
    legacy = run_variant(build, base_name + '_legacy', LEGACY, base)
    experimental = run_variant(build, base_name + '_experimental', EXPERIMENTAL, base)
    assert set(legacy) == set(experimental)
    for key in legacy:
        assert np.array_equal(
            legacy[key], experimental[key]), ('%s: experimental CPU codegen is not bit-exact vs legacy for output %s' %
                                              (base_name, key))
    return experimental


def experimental_code(build: Callable[[str], dace.SDFG], name: str) -> str:
    """Generated experimental C++ (codegen only, no compile) for inspection.

    :param build: Callable building the SDFG from a name.
    :param name: SDFG name.
    :return: The cleaned generated C++ of the first (host) code object.
    """
    with use_implementation(EXPERIMENTAL):
        return build(name).generate_code()[0].clean_code


def size_helper_definition(code: str, array: str) -> str:
    """The single-line definition of the emitted ``<array>_size`` helper.

    Matches ``<array>_size`` at a word boundary so a helper for ``T`` is not
    confused with one for ``inner_T``.

    :param code: Generated C++.
    :param array: Array base name.
    :return: The stripped definition line.
    """
    pattern = re.compile(r'(?<!\w)%s_size\(' % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if pattern.search(line) and 'return' in line]
    assert lines, 'experimental codegen emitted no %s_size helper' % array
    return lines[0]


def allocation_line(code: str, array: str) -> str:
    """The ``aligned_alloc`` statement for ``array`` (matched at a word boundary).

    :param code: Generated C++.
    :param array: Array base name.
    :return: The stripped allocation line.
    """
    pattern = re.compile(r'(?<!\w)%s = ' % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if 'aligned_alloc' in line and pattern.search(line)]
    assert lines, 'experimental codegen emitted no aligned_alloc for %s' % array
    return lines[0]


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def test_symbolic_size_helper(require_experimental):
    """Symbolic ``T[N*M]`` heap transient -> ``constexpr T_size(long long M, long long N)``."""
    n, m = dace.symbol('N'), dace.symbol('M')
    build = lambda name: heap_pipeline_1d(name, n * m, '0:N*M')
    base = dict(A=np.random.rand(48), B=np.zeros(48), N=6, M=8)

    experimental = assert_bit_exact(build, 'symsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'symsize_inspect')
    definition = size_helper_definition(code, 'T')
    assert 'constexpr' in definition, definition
    assert 'long long M' in definition and 'long long N' in definition, definition
    assert '(M * N)' in definition, definition
    assert 'dace::aligned_alloc<double>(T_size(M, N), 64)' in allocation_line(code, 'T')


def test_ipow_size_helper(require_experimental):
    """``total_size = ipow(N, 2)`` (``T[N*N]``) -> single-symbol ``constexpr T_size(long long N)``."""
    n = dace.symbol('N')
    build = lambda name: heap_pipeline_1d(name, n * n, '0:N*N')
    base = dict(A=np.random.rand(49), B=np.zeros(49), N=7)

    experimental = assert_bit_exact(build, 'ipowsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'ipowsize_inspect')
    definition = size_helper_definition(code, 'T')
    assert 'constexpr' in definition and 'long long N' in definition, definition
    assert 'ipow(N, 2)' in definition, definition
    assert 'dace::aligned_alloc<double>(T_size(N), 64)' in allocation_line(code, 'T')


def test_constant_size_helper(require_experimental):
    """Constant-size ``CPU_Heap`` transient -> nullary ``T_size()``: ``consteval`` under C++20,
    degrading to ``constexpr`` under C++17 (``consteval`` is not a keyword before C++20)."""
    build = lambda name: heap_pipeline_1d(name, 200, '0:200')
    base = dict(A=np.random.rand(200), B=np.zeros(200))

    experimental = assert_bit_exact(build, 'constsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'constsize_inspect')
    definition = size_helper_definition(code, 'T')
    expected_qual = 'consteval' if int(str(Config.get('compiler', 'cpp_standard')).strip()) >= 20 else 'constexpr'
    assert expected_qual in definition, definition
    assert 'T_size()' in definition and 'return 200;' in definition, definition
    assert 'dace::aligned_alloc<double>(T_size(), 64)' in allocation_line(code, 'T')


def test_bare_single_symbol_not_wrapped(require_experimental):
    """A bare single-symbol size ``T[N]`` is NOT wrapped (wrapping ``N`` is no win)."""
    n = dace.symbol('N')
    build = lambda name: heap_pipeline_1d(name, n, '0:N')
    base = dict(A=np.random.rand(64), B=np.zeros(64), N=64)

    experimental = assert_bit_exact(build, 'baresize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'baresize_inspect')
    assert 'T_size' not in code, 'a bare single-symbol size must not be wrapped in a helper'
    assert 'dace::aligned_alloc<double>(N, 64)' in allocation_line(code, 'T')


def test_distinct_size_helpers_across_nested_sdfgs(require_experimental):
    """Same transient name ``T`` at two sizes across a nested SDFG -> distinct helpers."""
    build = nested_same_name_sdfg
    base = dict(A=np.random.rand(5), B=np.zeros(5), N=5, M=3)

    experimental = assert_bit_exact(build, 'nestedsize', base)
    assert np.array_equal(experimental['B'][0], base['A'][0])

    code = experimental_code(build, 'nestedsize_inspect')
    # The inner (N*N -> ipow) and outer (N*M) sizes yield distinct helpers with
    # distinct arities -- no collision.
    inner_def = size_helper_definition(code, 'inner_T')
    outer_def = size_helper_definition(code, 'T')
    assert 'ipow(N, 2)' in inner_def, inner_def
    assert '(M * N)' in outer_def, outer_def
    assert 'inner_T_size(N)' in allocation_line(code, 'inner_T')
    assert 'T_size(M, N)' in allocation_line(code, 'T')


if __name__ == '__main__':
    test_symbolic_size_helper(None)
    test_ipow_size_helper(None)
    test_constant_size_helper(None)
    test_bare_single_symbol_not_wrapped(None)
    test_distinct_size_helpers_across_nested_sdfgs(None)
    print('ok')
