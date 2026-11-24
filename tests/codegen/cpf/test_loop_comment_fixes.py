# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact a usability review found in the CPF loop comments.

Each test renders a few-line program and checks the comment string of the loop the finding is about."""

import numpy as np

import dace
from dace.sdfg import tasklet_utils
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes import DecideCpuParallelAxes
from tests.codegen.cpf.conftest import render_gpu
from tests.codegen.cpf.test_specialization_hints import named, trapped_sdfg

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
    assert inner.startswith(
        "// parallel -- reduction: && into unset and || into flag; the other accesses are independent"
    ), inner


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


def guarded_stride_code() -> str:
    inc = dace.symbol("inc")

    @dace.program
    def strided(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i * inc] = a[i * inc] + b[i]

    sdfg = strided.to_sdfg(simplify=True)
    sdfg.name = "cpf_comment_guard"
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    return named(sdfg)


def test_a_guarded_loop_names_its_test_once_and_says_what_the_fallback_is():
    """s171: no doubled parentheses around the operands, and the fallback arm says it sums into ``a[0]``."""
    parallel, fallback = class_lines(guarded_stride_code())
    assert parallel.startswith("// parallel -- guarded by if (inc != 0) (one test per call): "), parallel
    assert fallback.startswith(
        "// sequential -- the else arm of if (inc != 0), taken when inc == 0 (the loop then makes a sum into a[0]): "
    ), fallback


def test_the_header_lists_the_entry_abort_preconditions():
    """The program aborts at entry when ``N <= 0``; a reader of the loops needs it said once, above them."""

    @dace.program
    def positive(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[N - 1 - i] = a[i]

    sdfg = positive.to_sdfg(simplify=True)
    sdfg.name = "cpf_comment_preconditions"
    guard = sdfg.add_state_before(sdfg.start_block, "entry_guard", is_start_block=True)
    tasklet_utils.add_abort_guard(guard, "entry_check", "N <= 0")
    code = named(sdfg)
    assert "// Entry preconditions (the unit abort()s when one holds): N <= 0" in code, code


def test_a_loop_comment_does_not_repeat_an_entry_precondition_the_header_lists():
    """The entry aborts when ``nkb <= 0``; the function keeps the name, so its loop needs no "independent only if"."""
    code = named(trapped_sdfg("cpf_comment_repeat", "nkb", "nkb"))
    assert "// Entry preconditions (the unit abort()s when one holds): nkb <= 0" in code, code
    assert "independent only if" not in code, code


def test_a_loop_comment_keeps_a_precondition_a_function_renamed():
    """The same trap reaches a function as ``n``: only there does the loop comment say what its independence rests on."""
    code = named(trapped_sdfg("cpf_comment_renamed", "n", "nkb"))
    assert "; independent only if n > 0 (the program aborts at entry otherwise)" in code, code


def decided(program: dace.program, name: str) -> str:
    """``program`` canonicalized, with CPU specialization's decision on its parallel axes, rendered."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    DecideCpuParallelAxes().apply_pass(sdfg, {})
    return named(sdfg)


def test_a_map_inside_a_parallel_loop_stays_parallel_and_names_the_loop():
    @dace.program
    def nested(a: dace.float64[M, N], b: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                b[i, j] = a[i, j] * 2.0

    outer, inner = class_lines(decided(nested, "cpf_comment_nested"))
    assert outer.startswith("// parallel -- independent: ") and "not parallelized" not in outer, outer
    assert inner.startswith("// parallel -- independent: ") and inner.endswith(
        "; not parallelized: inside parallel loop i"
    ), inner


def test_a_short_map_stays_parallel_and_says_it_is_too_short_to_fork():
    @dace.program
    def short(a: dace.float64[4], b: dace.float64[4]):
        for j in dace.map[0:4]:
            b[j] = a[j] * 2.0

    (only,) = class_lines(decided(short, "cpf_comment_short"))
    assert only.startswith("// parallel -- independent: "), only
    assert only.endswith("; not parallelized: 4 iterations, too short to pay for a fork"), only


def test_a_map_a_gpu_thread_runs_stays_parallel_and_says_so():
    @dace.program
    def nested(a: dace.float64[M, N], b: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                b[i, j] = a[i, j] * 2.0

    rendering = render_gpu(nested, "cpf_comment_gpu_nested", "cuda")
    lines = class_lines(rendering.code) + class_lines(rendering.device_code or "")
    threaded = [line for line in lines if "; not parallelized: inside GPU kernel loop " in line]
    assert threaded and all(line.startswith("// parallel -- independent: ") for line in threaded), lines
    assert not [line for line in lines if line.startswith("// sequential")], lines


def test_a_witness_satisfies_the_declared_preconditions():
    """``a[i] = a[i + K]`` conflicts only for ``K < N``; the program aborts at entry for ``K < N``, so no run it accepts
    has the conflict and the loop is not proven sequential."""
    K = dace.symbol("K")

    @dace.program
    def ahead(a: dace.float64[N + K]):
        for i in range(N):
            a[i] = a[i + K] + 1.0

    sdfg = ahead.to_sdfg(simplify=True)
    sdfg.name = "cpf_comment_declared"
    guard = sdfg.add_state_before(sdfg.start_block, "entry_guard", is_start_block=True)
    tasklet_utils.add_abort_guard(guard, "entry_check", "K < N")
    (first,) = class_lines(named(sdfg))
    assert first.startswith("// unknown -- "), first
    assert "K=1, N=2" not in first, first


def test_a_sequential_loop_names_the_container_that_carries_it_and_how_it_is_written():
    """The running example's time loop proves a dependence on ``y``, but ``x`` carries the values to the next
    iteration, through the copy ``x = y``."""

    @dace.program
    def power_step(x: dace.float64[N], y: dace.float64[N], col: dace.int64[N]):
        for t in range(T):
            for j in dace.map[0:N]:
                y[j] = x[col[j]]
            x[:] = y

    sequential = [line for line in class_lines(rendered(power_step, "cpf_comment_carrier")) if "sequential" in line]
    assert len(sequential) == 1 and sequential[0].endswith(
        "; x (x = y) carries the values from one iteration to the next"
    ), sequential


def test_the_header_lists_program_loops_not_helper_calls_or_lowering_counters():
    """The scan helper is listed on the scans line and the copy helper's ``__i0`` is no loop of the program."""

    @dace.program
    def running_sum(a: dace.float64[N], y: dace.float64[N]):
        y[:] = np.cumsum(a)

    header = [
        line for line in rendered(running_sum, "cpf_comment_header_names").splitlines() if line.startswith("//   ")
    ]
    parallel = [line for line in header if line.startswith("//   parallel: ") and "proven" not in line]
    assert all("(" not in line and "__" not in line for line in parallel), header


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
    test_a_guarded_loop_names_its_test_once_and_says_what_the_fallback_is()
    test_the_header_lists_the_entry_abort_preconditions()
    test_a_loop_comment_does_not_repeat_an_entry_precondition_the_header_lists()
    test_a_loop_comment_keeps_a_precondition_a_function_renamed()
    test_a_map_inside_a_parallel_loop_stays_parallel_and_names_the_loop()
    test_a_short_map_stays_parallel_and_says_it_is_too_short_to_fork()
    test_a_map_a_gpu_thread_runs_stays_parallel_and_says_so()
    test_a_witness_satisfies_the_declared_preconditions()
    test_a_sequential_loop_names_the_container_that_carries_it_and_how_it_is_written()
    test_the_header_lists_program_loops_not_helper_calls_or_lowering_counters()
