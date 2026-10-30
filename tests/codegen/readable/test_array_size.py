# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the experimental (readable) CPU generator's per-array ``<array>_size``
helpers.

Under ``compiler.cpu.implementation = experimental`` a heap array whose
``total_size`` is symbolic or a compound expression is allocated as
an aligned ``new[]`` over ``<array>_size(...)``, where the generated
``<array>_size`` helper returns that size. Constant sizes get a nullary
``T_size()`` helper (``consteval`` under C++20, ``constexpr`` under C++17); compound symbolic sizes get a ``constexpr``
helper over the sorted free symbols (``T_size(M, N)``); a bare single symbol
(``T[N]``) is deliberately left as the plain name (wrapping it is no readability
win).

Each kernel is compiled + run under both the ``legacy`` and ``experimental``
generators on identical (``copy.deepcopy``'d) inputs; the outputs must be
bit-exact, which proves the routed size expression matches the classic
``sym2cpp(total_size)``. A few cases additionally inspect the generated C++ (the
helper definition and the aligned ``new[]`` call). CPU kernels run in a forked
child (repo rule) via :func:`conftest.run_isolated`.
"""

import copy
import re
from collections.abc import Callable

import numpy as np

import dace
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
    sdfg.add_array("A", [shape], dace.float64)
    sdfg.add_array("B", [shape], dace.float64)
    sdfg.add_transient("T", [shape], dace.float64, storage=StorageType.CPU_Heap)

    write = sdfg.add_state("write")
    read_a, write_t = write.add_read("A"), write.add_write("T")
    entry, exit_node = write.add_map("write_map", dict(i=rng))
    add = write.add_tasklet("add_one", {"a"}, {"o"}, "o = a + 1.0")
    write.add_memlet_path(read_a, entry, add, dst_conn="a", memlet=dace.Memlet("A[i]"))
    write.add_memlet_path(add, exit_node, write_t, src_conn="o", memlet=dace.Memlet("T[i]"))

    read = sdfg.add_state_after(write, "read")
    read_t, write_b = read.add_read("T"), read.add_write("B")
    entry2, exit2 = read.add_map("read_map", dict(i=rng))
    mul = read.add_tasklet("mul_two", {"t"}, {"o"}, "o = t * 2.0")
    read.add_memlet_path(read_t, entry2, mul, dst_conn="t", memlet=dace.Memlet("T[i]"))
    read.add_memlet_path(mul, exit2, write_b, src_conn="o", memlet=dace.Memlet("B[i]"))

    sdfg.validate()
    return sdfg


# --------------------------------------------------------------------------- #
# Equivalence + codegen-inspection helpers
# --------------------------------------------------------------------------- #
def run_variant(
    build: Callable[[str], dace.SDFG], name: str, implementation: str, base: dict[str, object]
) -> dict[str, np.ndarray]:
    """Build + compile + run one variant on a deep copy of ``base``; return outputs.

    Runs in a forked child (repo rule) via ``run_isolated``.

    :param build: Callable building the SDFG from a name.
    :param name: SDFG name for this variant.
    :param implementation: ``legacy`` or ``experimental``.
    :param base: Shared inputs (arrays + symbol scalars); deep-copied per variant.
    :return: The ndarray outputs keyed by name.
    """

    def work() -> dict[str, np.ndarray]:
        sdfg = build(name)
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}

    with use_implementation(implementation):
        return run_isolated(work)


def assert_bit_exact(
    build: Callable[[str], dace.SDFG], base_name: str, base: dict[str, object]
) -> dict[str, np.ndarray]:
    """Run ``build`` under legacy and experimental; assert every output is bit-exact.

    :param build: Callable building the SDFG from a name.
    :param base_name: Base name for the two variants.
    :param base: Shared inputs, deep-copied per variant.
    :return: The experimental output dict.
    """
    legacy = run_variant(build, base_name + "_legacy", LEGACY, base)
    experimental = run_variant(build, base_name + "_experimental", EXPERIMENTAL, base)
    assert set(legacy) == set(experimental)
    for key in legacy:
        assert np.array_equal(legacy[key], experimental[key]), (
            "%s: experimental CPU codegen is not bit-exact vs legacy for output %s" % (base_name, key)
        )
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
    pattern = re.compile(r"(?<!\w)%s_size\(" % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if pattern.search(line) and "return" in line]
    assert lines, "experimental codegen emitted no %s_size helper" % array
    return lines[0]


def allocation_line(code: str, array: str) -> str:
    """The aligned ``new[]`` allocation statement for ``array`` (matched at a word boundary)."""
    pattern = re.compile(r"(?<!\w)%s = " % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if "aligned_new_array" in line and pattern.search(line)]
    assert lines, "experimental codegen emitted no aligned allocation for %s" % array
    return lines[0]


def test_ipow_size_helper(require_experimental):
    """``total_size = ipow(N, 2)`` (``T[N*N]``) -> single-symbol ``constexpr T_size(int64_t N)``."""
    n = dace.symbol("N")
    build = lambda name: heap_pipeline_1d(name, n * n, "0:N*N")
    base = dict(A=np.random.rand(49), B=np.zeros(49), N=7)

    experimental = assert_bit_exact(build, "ipowsize", base)
    assert np.array_equal(experimental["B"], (base["A"] + 1.0) * 2.0)

    code = experimental_code(build, "ipowsize_inspect")
    definition = size_helper_definition(code, "T")
    assert "constexpr" in definition and "int64_t N" in definition, definition
    # RelaxIntegerPowers lowers ``N**2`` to ``ipow(N, 2)``; ``ipow`` is a constexpr runtime helper, so
    # the constexpr size function may call it directly (see dace/runtime/include/dace/math.h).
    assert "ipow(N, 2)" in definition, definition
    assert "dace::aligned_new_array<double>(T_size(N), 64)" in allocation_line(code, "T")


def test_bare_single_symbol_not_wrapped(require_experimental):
    """A bare single-symbol size ``T[N]`` is NOT wrapped (wrapping ``N`` is no win)."""
    n = dace.symbol("N")
    build = lambda name: heap_pipeline_1d(name, n, "0:N")
    base = dict(A=np.random.rand(64), B=np.zeros(64), N=64)

    experimental = assert_bit_exact(build, "baresize", base)
    assert np.array_equal(experimental["B"], (base["A"] + 1.0) * 2.0)

    code = experimental_code(build, "baresize_inspect")
    assert "T_size" not in code, "a bare single-symbol size must not be wrapped in a helper"
    assert "dace::aligned_new_array<double>(N, 64)" in allocation_line(code, "T")


if __name__ == "__main__":
    test_ipow_size_helper(None)
    test_bare_single_symbol_not_wrapped(None)
    print("ok")
