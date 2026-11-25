# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact the second usability review found in the CPF loop comments.

Each test renders a few-line program the way the bench does and checks the line the finding is about."""

from unittest import mock

import pytest

import dace
from dace import cpf_lowering
from dace.codegen import cpf, cpf_summary
from dace.codegen.cpf import render
from dace.cpf_lowering import LoopClass, Reason
from dace.transformation.passes.analysis import smt_dependence
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import render_gpu

N = dace.symbol("N")
M = dace.symbol("M")
T = dace.symbol("T")


def bench_c(program: dace.program, name: str) -> str:
    """``program`` as the bench renders it for the CPU: canonicalized, finalized, LLM-friendly C."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    return render(sdfg, language="c", llm_friendly=True).code


def comment_above(code: str, marker: str) -> list[str]:
    """The comment lines right above the first line of ``code`` that contains ``marker``, past pragmas and braces."""
    lines = code.splitlines()
    at = next(n for n, line in enumerate(lines) if marker in line)
    above = []
    while at > 0 and (lines[at - 1].strip().startswith(("//", "#pragma")) or lines[at - 1].strip() == "{"):
        at -= 1
        above.insert(0, lines[at].strip())
    return [line for line in above if line.startswith("//")]


@dace.program
def dot(a: dace.float64[N], b: dace.float64[N], r: dace.float64[1]):
    for i in range(N):
        r[0] += a[i] * b[i]


@dace.program
def spmv(rowptr: dace.int64[N + 1], col: dace.int64[M], val: dace.float64[M], x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        s = 0.0
        for k in range(rowptr[i], rowptr[i + 1]):
            s += val[k] * x[col[k]]
        y[i] = s


def test_a_grid_stride_reduction_kernel_says_reduction_and_grid_stride():
    """Finding 1: the dot kernel said ``independent iterations, one GPU thread each``."""
    device = render_gpu(dot, "cpf_review2_dot", "cuda").device_code
    (kernel,) = comment_above(device, "__global__")
    assert kernel.startswith("// parallel -- reduction: + into priv_r"), kernel
    assert "kernel: grid-stride reduction" in kernel and "one atomic" in kernel, kernel
    assert "independent iterations" not in device and "one GPU thread each" not in device, device
    assert "each thread runs these iterations in order" not in device, device


def test_a_block_per_row_kernel_has_one_line_per_kernel_and_per_thread_block():
    """Finding 2: the SpMV kernel stacked two class lines and said one thread per row."""
    device = render_gpu(spmv, "cpf_review2_spmv", "cuda").device_code
    (kernel,) = comment_above(device, "__global__")
    assert "kernel: one thread block per iteration of i" in kernel, kernel
    (block,) = comment_above(device, "int j = threadIdx.x")
    assert block.startswith("// parallel -- block tile:") and "thread block:" in block, block


def test_a_lane_strided_reduction_says_reduction_and_carries_no_host_pragma():
    """Finding 3: the lane-strided row loop said ``independent`` and ``not parallelized`` over a host ``omp simd``."""
    device = render_gpu(spmv, "cpf_review2_spmv_lanes", "cuda").device_code
    (lane,) = comment_above(device, "k += 256")
    assert lane.startswith("// parallel -- reduction: + into s"), lane
    assert "lane-strided" in lane and "block reduction" in lane and "not parallelized" not in lane, lane
    assert "#pragma omp" not in device, "a device compiler ignores a host OpenMP pragma"


def test_a_kernel_library_loop_names_the_kernel_not_its_launch_bounds():
    """Finding 4: the function name was read off ``__launch_bounds__(512)``."""
    device = render_gpu(dot, "cpf_review2_bounds", "cuda").device_code
    assert "function __launch_bounds__" not in device, device
    assert "library loop (kernel cpf_review2_bounds_" in device, device


def test_a_simd_loop_says_vectorized_not_not_parallelized():
    """Finding 6: a nested loop under ``omp simd`` said ``not parallelized``."""
    code = bench_c(spmv, "cpf_review2_simd")
    (line,) = [line.strip() for line in comment_above(code, "#pragma omp simd") if "parallel --" in line]
    assert line.endswith("; vectorized (omp simd), not thread-parallel: inside parallel loop i"), line


@dace.program
def time_stencil(a: dace.float64[N], b: dace.float64[N]):
    for t in range(T):
        for i in dace.map[1 : N - 1]:
            b[i] = 0.25 * (a[i - 1] + 2 * a[i] + a[i + 1])
        for i in dace.map[1 : N - 1]:
            a[i] = b[i]


def test_a_time_loop_names_the_flow_dependence_alike_on_cpu_and_gpu():
    """Finding 5: the CPU said an iteration reads the ``b`` an earlier one wrote, though each writes ``b`` first."""
    flow = "// sequential -- proven dependence on a[1:N - 1]: an iteration reads what an earlier one wrote"
    cpu = bench_c(time_stencil, "cpf_review2_stencil_cpu")
    gpu = render_gpu(time_stencil, "cpf_review2_stencil_gpu", "cuda").code
    for code in (cpu, gpu):
        (line,) = [line.strip() for line in code.splitlines() if line.strip().startswith("// sequential --")]
        assert line.startswith(flow), line


def test_a_fallback_arm_that_sums_into_one_element_says_it_is_a_reduction():
    """Finding 7: the ``inc == 0`` arm said only that it keeps the order, hiding the reduction."""
    inc = dace.symbol("inc")

    @dace.program
    def strided(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i * inc] = a[i * inc] + b[i]

    (line,) = [
        line.strip() for line in bench_c(strided, "cpf_review2_guard").splitlines() if "// sequential --" in line
    ]
    assert "(the loop then makes a sum into a[0]): a reduction, kept in order here because" in line, line
    assert line.endswith("parallel with reduction(+:a[0]) if a different rounding is acceptable"), line


@dace.program
def scatter(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N]):
    for i in range(N):
        b[idx[i]] = a[i] * 2.0


def test_a_scatter_guard_says_what_its_flag_is_and_names_the_scattered_element():
    """Finding 8: ``if (tmp <= 0)`` did not say what ``tmp`` is, and the fallback wrote ``b[b_slice_0]``."""
    lines = [line.strip() for line in bench_c(scatter, "cpf_review2_scatter").splitlines()]
    guarded = next(line for line in lines if line.startswith("// parallel -- guarded by if (tmp <= 0)"))
    fallback = next(line for line in lines if line.startswith("// sequential -- the else arm of if (tmp <= 0)"))
    for line in (guarded, fallback):
        assert "where tmp is 1 when idx holds a value twice" in line, line
    assert "(the loop then writes b[idx[i]])" in fallback, fallback


def test_a_gpu_fallback_arm_says_it_runs_on_the_host_and_what_it_copies():
    """Finding 8 (GPU): the fallback loop runs on the host between device copies, which its comment did not say."""
    host = render_gpu(scatter, "cpf_review2_scatter_gpu", "cuda").code
    (note,) = [line.strip() for line in host.splitlines() if "// runs on the host" in line]
    assert note == "// runs on the host, after device-to-host copies of idx, a, b; then copies b back to the device"


def test_an_array_section_reduction_says_each_thread_holds_a_copy():
    """Finding 9: ``reduction(+:h[0:M])`` gives every thread M elements, which the comment did not say."""

    @dace.program
    def hist(a: dace.float64[N], idx: dace.int64[N], h: dace.float64[M]):
        for i in range(N):
            h[idx[i]] += a[i]

    code = bench_c(hist, "cpf_review2_hist")
    (line,) = [line for line in comment_above(code, "reduction(+:h[0:M])") if "parallel --" in line]
    assert "each thread holds a private copy of the M elements h[0:M]" in line, line
    assert "alt: omp atomic updates of h" in line, line


def test_a_fork_note_simplifies_its_trip_counts_and_bounds_a_moving_one():
    """Finding 12: ``(N - 1) - 1`` read unsimplified, and a wavefront step's count had no bound."""
    code = """void f(double *a, int64_t N) {
    // sequential -- wavefront of tiles [64x64]: each step reads what the previous step wrote
    for (int64_t i = 0; i < T; i += 1) {
        #pragma omp parallel for
        for (int64_t j = cpf_max_int64(0, ((i - cpf_int_ceil_int64((N - 1), 64)) + 1)); j < (cpf_min_int64(i, (cpf_int_ceil_int64((N - 1), 64) - 1)) + 1); j += 1) {
            a[j] = 0.0;
        }
        #pragma omp parallel for
        for (int64_t k = 1; k < (N - 1); k += 1) {
            a[k] = 1.0;
        }
    }
}
"""
    (note,) = [line for line in cpf_summary.note_forks(code).splitlines() if "fork a thread team" in line]
    assert ", at most ceil((N - 1)/64) iterations)" in note, note
    assert "k (N - 2 iterations)" in note, note


def test_an_inline_fused_scan_names_the_scan_not_the_kernel_function():
    """LLR40 tsvc_2_s323: the fused scan's loops said ``implementation loop of function <the kernel>``."""

    L = dace.symbol("L", dtype=dace.int64, positive=True)

    @dace.program
    def s323(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L], e: dace.float64[L]):
        for i in range(1, L):
            a[i] = b[i - 1] + c[i] * d[i]
            b[i] = a[i] + c[i] * e[i]

    code = bench_c(s323, "cpf_review2_s323")
    labels = [
        line.strip() for line in code.splitlines() if "library loop (" in line or "implementation loop of" in line
    ]
    assert labels and not [line for line in labels if "function cpf_review2_s323" in line], labels
    assert "// parallel -- implementation loop of scan: vectorized (omp simd)" in labels, labels


def test_a_carried_counter_is_named_without_a_stray_space():
    """LLR40 compact_threshold_pack: ``proven dependence : run = run + ...`` had a space before the colon."""
    L = dace.symbol("L", dtype=dace.int64, positive=True)

    @dace.program
    def pack(src: dace.float64[L], weight: dace.float64[L], packed: dace.float64[L], out_count: dace.int64[1]):
        n = 0
        for i in range(L):
            if src[i] > 0.0:
                packed[n] = src[i] * weight[i]
                n = n + 1
        out_count[0] = n

    rendering = render_gpu(pack, "cpf_review2_pack", "cuda")
    code = rendering.code + rendering.device_code
    assert "proven dependence :" not in code, code
    assert "// sequential -- proven dependence: the counter " in code, code


def test_a_promoted_warp_tile_keeps_one_class_line_and_its_kernel_runs_a_block_per_tile():
    """LLR40 wf_triangular: ``warp tile`` stacked a second class line on the wavefront step, and the kernel said one
    GPU thread per tile column."""
    L = dace.symbol("L", dtype=dace.int64, positive=True)

    @dace.program
    def triangular(a: dace.float64[L, L]):
        for i in range(1, L):
            for j in range(i, L):
                a[i, j] = (a[i, j] + a[i - 1, j] + a[i, j - 1]) / 3.0

    device = render_gpu(triangular, "cpf_review2_triangular", "cuda").device_code
    assert "// parallel -- warp tile" not in device, device
    step = next(line for line in device.splitlines() if "// parallel -- one wavefront step" in line)
    assert "; warp tile: spread over the threads of the block" in step, step
    kernel = comment_above(device, "__global__")[0]
    assert "kernel: one thread block per iteration of j" in kernel, kernel


def test_a_launch_per_trip_note_gives_its_grid_in_blocks_of_threads():
    """Finding 13: the per-trip launch note gave ``ceiling(N/256 - 1/128) x 256`` without a unit."""
    code = render_gpu(time_stencil, "cpf_review2_stencil_launch", "cuda").code
    assert "T launches per call, each of ceil((N - 2)/256) blocks of 256 threads" in code, code


def test_a_copy_names_its_source_and_destination():
    """Finding 14: the stencil copy said ``copy: write one buffer into another``."""
    code = bench_c(time_stencil, "cpf_review2_stencil_copy")
    assert "// copy: b[1:N - 1] into a[1:N - 1]" in code, code
    assert "write one buffer into another" not in code, code


@dace.program
def early_exit(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        if a[i] < 0:
            break
        b[i] = a[i]


def test_a_chunked_early_exit_says_its_blocks_are_searched_in_parallel():
    """Finding 15: the chunked early-exit loop said only that iterations after the exit must not run."""
    code = bench_c(early_exit, "cpf_review2_early")
    (line,) = [line.strip() for line in code.splitlines() if "// sequential -- early exit (break)" in line]
    assert "; blocks of " in line and "searched in parallel for the first exit (find-first)" in line, line


@dace.program
def max_reduce(a: dace.float64[N], r: dace.float64[1]):
    for i in range(N):
        r[0] = max(r[0], a[i])


def test_a_reduction_call_names_its_operator_input_axis_and_output():
    """Finding 16: the max-reduce call said ``reduction over the given axes with the given operator``."""
    code = bench_c(max_reduce, "cpf_review2_maxred")
    assert "// reduction: max of a[0:N] over axis 0 into r[0]" in code, code


def test_the_closing_line_names_fallback_arms_and_units_without_loops():
    """Finding 17: the closing line called a guard's fallback arm and a loop-free unit a proven recurrence."""
    inc = dace.symbol("inc")

    @dace.program
    def strided_closing(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i * inc] = a[i * inc] + b[i]

    assert cpf_summary.CLOSING_WITH_FALLBACKS in bench_c(strided_closing, "cpf_review2_closing_guard")
    loop_free = bench_c(max_reduce, "cpf_review2_closing_library")
    assert cpf_summary.CLOSING_NO_LOOPS in loop_free and cpf_summary.CLOSING not in loop_free, loop_free


L = dace.symbol("L", dtype=dace.int64, positive=True)


@dace.program
def diff_skew(a: dace.float64[L, L]):
    for i in range(1, L):
        for j in range(0, L - 1):
            a[i, j] = (a[i, j] + a[i - 1, j] + a[i - 1, j + 1]) / 3.0


def test_a_row_recurrence_read_through_a_map_scalar_is_proven():
    """wf_diff_skew: the search saw only the map's merged read box ``a[i - 1:i + 1, 0:L]`` and left ``i`` unknown."""
    flow = "// sequential -- proven dependence on a[i - 1, 0:L - 1]: an iteration reads what an earlier one wrote"
    cpu = bench_c(diff_skew, "cpf_review2_skew_cpu")
    gpu = render_gpu(diff_skew, "cpf_review2_skew_gpu", "cuda").code
    for code in (cpu, gpu):
        (line,) = [line.strip() for line in code.splitlines() if line.strip().startswith(("// sequential", "// unk"))]
        assert line.startswith(flow), line


@dace.program
def guarded_columns(aa: dace.float64[L, L], bb: dace.float64[L, L]):
    for i in range(L):
        if aa[0, i] > 0.0:
            for j in range(1, L):
                aa[j, i] = aa[j - 1, i] + bb[j, i]


def test_an_unknown_loop_under_a_guard_on_written_data_names_the_guard():
    """s275: the unknown row loop said only that the test could not show the conflict, not that its guard reads aa."""
    cpu = bench_c(guarded_columns, "cpf_review2_s275_cpu")
    gpu = render_gpu(guarded_columns, "cpf_review2_s275_gpu", "cuda").device_code
    for code in (cpu, gpu):
        (line,) = [line.strip() for line in code.splitlines() if line.strip().startswith("// unknown")]
        assert line.startswith("// unknown -- the guard aa[0, "), line
        assert "reads aa, which this loop writes" in line and "(outside the analyzable domain)" in line, line


S = dace.symbol("S")


@dace.program
def modular_rows(a: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(N):
            a[i, j] = a[i - 1, (j * S) % N] + 1.0


@dace.program
def indirect_read(a: dace.float64[N], idx: dace.int64[N]):
    for i in range(N):
        a[i] = a[idx[i]] + 1.0


def unknown_lines(code: str) -> list[str]:
    return [line.strip() for line in code.splitlines() if line.strip().startswith("// unknown")]


def test_an_unknown_loop_names_a_read_it_could_only_bound():
    """An unknown loop whose read the test saw only as the range ``a[i - 1, 0:N]`` said nothing of why."""
    (line,) = unknown_lines(bench_c(modular_rows, "cpf_review2_range_read"))
    assert line.startswith("// unknown -- the test could only bound the read a[i - 1, 0:N]: "), line
    assert "found no two iterations that meet on one element" in line, line


def test_an_unknown_loop_names_an_index_it_assigns():
    """An unknown loop that reads ``a[sym0]``, ``sym0 = idx[i]``, did not say the loop assigns the index."""
    (line,) = unknown_lines(bench_c(indirect_read, "cpf_review2_assigned_index"))
    assert line.startswith("// unknown -- the index sym0 of a[sym0] is assigned in the loop (sym0 = idx[i])"), line


def test_an_unknown_loop_says_when_the_solver_ran_out_of_budget():
    """A witness query z3 gave up on read as if the test had answered it."""
    original = smt_dependence.bounded_solver
    with mock.patch.object(smt_dependence, "bounded_solver", autospec=True, side_effect=lambda rlimit=0: original(1)):
        (line,) = unknown_lines(bench_c(indirect_read, "cpf_review2_budget"))
    assert line.startswith("// unknown -- " + Reason.BUDGET_EXPIRED.value + "; the index sym0"), line


@dace.program
def dot_then_scale(a: dace.float64[N], b: dace.float64[N], x: dace.float64[M], r: dace.float64[1]):
    for i in range(N):
        r[0] += a[i] * b[i]
    for j in range(M):
        x[j] = x[j] * r[0]


def test_the_header_legend_has_one_row_per_reason_the_unit_uses():
    """Finding 10: the legend explained classes only; a unit of a reduction and an independent loop gets two rows."""
    code = bench_c(dot_then_scale, "cpf_review2_legend")
    table = [line for line in code.splitlines() if line.startswith("//   ") and " | " in line]
    assert table[0].split(" | ")[0].strip() == "//   class" and len(table) == 3, table
    assert [row.split(" | ")[:2] for row in table[1:]] == [
        ["//   parallel", "reduction  ".ljust(len(table[1].split(" | ")[1]))],
        ["//   parallel", "independent".ljust(len(table[1].split(" | ")[1]))],
    ], table
    assert cpf.CPF_LINES_LEGEND in code, code


@pytest.mark.parametrize("reason", list(Reason), ids=[reason.name for reason in Reason])
def test_every_reason_is_told_apart_and_has_a_legend_row(reason: Reason):
    """Each Reason member is found in a line that opens with it (or holds it after the element it names) and gets its
    legend row."""
    inner = "proven dependence on a[i]: " if reason in cpf_lowering.INNER_REASONS else ""
    line = f"// {LoopClass.UNKNOWN.value} -- {inner}{reason.value} of the loop"
    assert cpf_lowering.reason_of(line.split(" -- ", 1)[1]) is reason, line
    (row,) = cpf.reason_table(line)[1:]
    assert row.split(" | ")[2].strip() == cpf_lowering.REASON_LEGEND[reason][0], row


@dace.program
def k_shift(a: dace.float64[N + S]):
    for i in range(N):
        a[i] = a[i + S] + 1.0


def test_the_summary_names_a_scans_blocked_code_as_one_scan():
    """Finding 18: the header listed the scan's blocks as loops ``statement``, ``if()``, ``cpf_i``."""
    header = [line for line in bench_c(k_shift, "cpf_review2_scan_names").splitlines() if line.startswith("//   ")]
    (parallel,) = [line for line in header if line.startswith("//   parallel: ") and "proven;" not in line]
    names = parallel.removeprefix("//   parallel: ").split(", ")
    assert "scan" in names and not any(n.startswith(("statement", "if()", "cpf_")) for n in names), parallel
    assert "//     scans: scan" in header, header


if __name__ == "__main__":
    test_a_grid_stride_reduction_kernel_says_reduction_and_grid_stride()
    test_a_block_per_row_kernel_has_one_line_per_kernel_and_per_thread_block()
    test_a_lane_strided_reduction_says_reduction_and_carries_no_host_pragma()
    test_a_kernel_library_loop_names_the_kernel_not_its_launch_bounds()
    test_a_simd_loop_says_vectorized_not_not_parallelized()
    test_a_time_loop_names_the_flow_dependence_alike_on_cpu_and_gpu()
    test_a_fallback_arm_that_sums_into_one_element_says_it_is_a_reduction()
    test_a_scatter_guard_says_what_its_flag_is_and_names_the_scattered_element()
    test_a_gpu_fallback_arm_says_it_runs_on_the_host_and_what_it_copies()
    test_an_array_section_reduction_says_each_thread_holds_a_copy()
    test_a_fork_note_simplifies_its_trip_counts_and_bounds_a_moving_one()
    test_an_inline_fused_scan_names_the_scan_not_the_kernel_function()
    test_a_carried_counter_is_named_without_a_stray_space()
    test_a_promoted_warp_tile_keeps_one_class_line_and_its_kernel_runs_a_block_per_tile()
    test_a_launch_per_trip_note_gives_its_grid_in_blocks_of_threads()
    test_a_copy_names_its_source_and_destination()
    test_a_chunked_early_exit_says_its_blocks_are_searched_in_parallel()
    test_a_reduction_call_names_its_operator_input_axis_and_output()
    test_the_closing_line_names_fallback_arms_and_units_without_loops()
    test_a_row_recurrence_read_through_a_map_scalar_is_proven()
    test_an_unknown_loop_under_a_guard_on_written_data_names_the_guard()
    test_an_unknown_loop_names_a_read_it_could_only_bound()
    test_an_unknown_loop_names_an_index_it_assigns()
    test_an_unknown_loop_says_when_the_solver_ran_out_of_budget()
    test_the_header_legend_has_one_row_per_reason_the_unit_uses()
    for member in Reason:
        test_every_reason_is_told_apart_and_has_a_legend_row(member)
    test_the_summary_names_a_scans_blocked_code_as_one_scan()
