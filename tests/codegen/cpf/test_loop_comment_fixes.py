# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact a usability review found in the CPF loop comments.

Each test renders a few-line program and checks the comment string of the loop the finding is about."""

import dace
from tests.codegen.cpf.test_specialization_hints import named

N = dace.symbol("N")
T = dace.symbol("T")


def class_lines(code: str) -> list[str]:
    """The class comments of ``code``, without their indentation."""
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith(("// parallel", "// sequential", "// unknown"))
    ]


def rendered(program: dace.program, name: str) -> str:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    return named(sdfg)


def test_a_sequential_loop_over_an_indirect_access_offers_no_wavefront():
    """The running example: ``x[col[j]]`` has no affine skew, so the wavefront alternative was false."""

    @dace.program
    def indirect(x: dace.float64[N], y: dace.float64[N], col: dace.int64[N]):
        for t in range(T):
            for j in dace.map[0:N]:
                y[j] = x[col[j]]
            for j in dace.map[0:N]:
                x[j] = y[j]

    outer = class_lines(rendered(indirect, "cpf_comment_wavefront"))[0]
    assert outer.startswith("// sequential -- proven dependence on x[0:N]"), outer
    assert "wavefront" not in outer and "alt:" not in outer, outer


def test_an_unproven_loop_over_a_few_iterations_is_unknown_not_sequential():
    """CloudSC's species loop: the dependence test could not show the writes distinct."""

    @dace.program
    def species(a: dace.float64[N], order: dace.int64[5]):
        for s in range(5):
            a[order[s]] = a[order[s]] + 1.0

    first = class_lines(rendered(species, "cpf_comment_small_extent"))[0]
    assert first.startswith("// unknown -- small extent (5 iterations), not proven: "), first
    code = rendered(species, "cpf_comment_small_extent_header")
    assert "// cpf: proven: partial; open: s" in code, code


def test_a_conditional_write_to_a_fixed_element_says_every_iteration_writes_it():
    """The write ``a[0]`` does not move with ``i``: an index array without repeats is no way out of it."""

    @dace.program
    def same_element(a: dace.float64[4], b: dace.float64[N]):
        for i in range(N):
            if b[i] > 0:
                a[0] = b[i]

    first = class_lines(rendered(same_element, "cpf_comment_same_element"))[0]
    assert "every iteration writes the same elements of a" in first, first
    assert "private per iteration or the update is a reduction" in first, first
    assert "index array" not in first, first


def test_a_read_of_what_earlier_iterations_write_is_named_as_such():
    """cholesky's ``i`` loop: row ``i`` reads ``A[j, j]``, which an earlier iteration wrote."""

    @dace.program
    def lower(A: dace.float64[N, N]):
        for i in range(N):
            for j in range(i):
                A[i, j] = A[i, j] / A[j, j]

    first = class_lines(rendered(lower, "cpf_comment_lower"))[0]
    assert "an earlier iteration never writes an element this read touches" in first, first
    assert "no iteration reads an element another iteration writes" not in first, first


if __name__ == "__main__":
    test_a_sequential_loop_over_an_indirect_access_offers_no_wavefront()
    test_an_unproven_loop_over_a_few_iterations_is_unknown_not_sequential()
    test_a_conditional_write_to_a_fixed_element_says_every_iteration_writes_it()
    test_a_read_of_what_earlier_iterations_write_is_named_as_such()
