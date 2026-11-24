# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact a usability review found in the CPF loop comments.

Each test renders a few-line program and checks the comment string of the loop the finding is about."""

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.test_specialization_hints import named

N = dace.symbol("N")
M = dace.symbol("M")
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


def any_per_row_code() -> str:
    @dace.program
    def any_per_row(a: dace.float64[N, M], flag: dace.int32[N], unset: dace.int32[N]):
        for i in range(N):
            flag[i] = 0
            unset[i] = 1
            for j in range(M):
                if a[i, j] > 0:
                    flag[i] = 1
                    unset[i] = 0

    sdfg = any_per_row.to_sdfg(simplify=True)
    sdfg.name = "cpf_comment_any_per_row"
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    return named(sdfg)


def test_only_the_loop_that_shares_the_accumulator_is_a_reduction():
    """logical_any_per_row: the ``i`` loop writes ``flag[i]`` and ``unset[i]``, one element per iteration; only the
    inner ``j`` loop reduces into them."""
    outer, inner = class_lines(any_per_row_code())
    assert outer.startswith("// parallel -- independent: "), outer
    assert inner == "// parallel -- reduction: && into unset and || into flag; the other accesses are independent", (
        inner
    )


def test_the_header_lists_every_reduction_of_a_loop():
    header = [line for line in any_per_row_code().splitlines() if line.startswith("//     reductions into")]
    assert header == ["//     reductions into unset (&&), flag (||)"], header


def test_a_blas_zero_fill_is_a_library_loop_not_a_sequential_one():
    """cholesky: the gemv expansion zero-fills its result before the call; the loop has no order to keep."""

    @dace.program
    def lower(A: dace.float64[N, N]):
        for i in range(N):
            A[i, 0:i] = A[i, 0:i] - A[0:i, 0:i] @ A[i, 0:i]

    code = rendered(lower, "cpf_comment_zero_fill")
    assert "// library loop (BLAS gemv zero fill): not a tuning target" in code, code
    assert "implementation loop of BLAS" not in code, code


def test_a_sequential_loop_says_its_parallel_loops_fork_once_per_iteration():
    """The time loop is in order, and the map inside it forks a team on every one of its iterations."""

    @dace.program
    def time_loop(a: dace.float64[N], b: dace.float64[N]):
        for t in range(1, T):
            for i in dace.map[0:N]:
                b[i] = a[i] * 2.0
            for i in dace.map[0:N]:
                a[i] = b[i]

    code = rendered(time_loop, "cpf_comment_forks")
    assert (
        "// its parallel loops i (N iterations), i2 (N iterations) fork a thread team once per iteration of this loop"
        in code
    ), code


def test_a_blas_call_says_it_runs_on_one_thread():
    """cholesky: OpenBLAS is pinned to one thread beside the OpenMP team, so the gemv does not use the team."""

    @dace.program
    def lower(A: dace.float64[N, N]):
        for i in range(N):
            A[i, 0:i] = A[i, 0:i] - A[0:i, 0:i] @ A[i, 0:i]

    code = rendered(lower, "cpf_comment_blas_threads")
    lines = [line.strip() for line in code.splitlines()]
    call = next(n for n, line in enumerate(lines) if line.startswith("cblas_dgemv("))
    assert lines[call - 1].startswith("// 1 thread: "), lines[call - 3 : call + 1]


def test_the_closing_line_does_not_steer_away_from_fork_placement():
    code = any_per_row_code()
    closing = next(line for line in code.splitlines() if line.startswith("// No reordering is left to prove"))
    assert "not from more parallelism" not in closing and "where the threads fork" in closing, closing


if __name__ == "__main__":
    test_a_sequential_loop_over_an_indirect_access_offers_no_wavefront()
    test_an_unproven_loop_over_a_few_iterations_is_unknown_not_sequential()
    test_a_conditional_write_to_a_fixed_element_says_every_iteration_writes_it()
    test_a_read_of_what_earlier_iterations_write_is_named_as_such()
    test_only_the_loop_that_shares_the_accumulator_is_a_reduction()
    test_the_header_lists_every_reduction_of_a_loop()
    test_a_blas_zero_fill_is_a_library_loop_not_a_sequential_one()
    test_a_sequential_loop_says_its_parallel_loops_fork_once_per_iteration()
    test_a_blas_call_says_it_runs_on_one_thread()
    test_the_closing_line_does_not_steer_away_from_fork_placement()
