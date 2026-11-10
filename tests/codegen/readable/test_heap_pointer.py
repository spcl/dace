# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The heap pointer of a transient is ``__restrict__`` and the ``<array>_idx`` helpers compute in ``int64_t``."""

import numpy

import dace
from dace.config import set_temporary

N = dace.symbol("N")


@dace.program
def scaled_twice(A: dace.float64[N], B: dace.float64[N]):
    tmp = numpy.empty(N, dace.float64)  # symbolic-size transient -> heap-allocated pointer
    for i in dace.map[0:N]:
        tmp[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        B[i] = tmp[i] + 1.0


def generate():
    sdfg = scaled_twice.to_sdfg(simplify=True)
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")


def test_a_heap_transient_is_declared_restrict():
    assert "double* __restrict__ tmp" in generate()


def test_a_heap_transient_that_may_alias_is_not_restrict():
    sdfg = scaled_twice.to_sdfg(simplify=True)
    sdfg.arrays["tmp"].may_alias = True
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")
    assert "__restrict__ tmp" not in code
    assert "double* tmp" in code


def test_index_helpers_are_int64_constexpr():
    """Every ``<array>_idx`` helper computes in ``int64_t`` and is ``static DACE_HDFI constexpr``."""
    helpers = [line for line in generate().splitlines() if "_idx(" in line and "return" in line and "{" in line]
    assert helpers
    for line in helpers:
        assert line.lstrip().startswith("static DACE_HDFI constexpr int64_t "), line


if __name__ == "__main__":
    test_a_heap_transient_is_declared_restrict()
    test_a_heap_transient_that_may_alias_is_not_restrict()
    test_index_helpers_are_int64_constexpr()
