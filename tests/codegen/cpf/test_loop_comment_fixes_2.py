# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact the second usability review found in the CPF loop comments.

Each test renders a few-line program the way the bench does and checks the line the finding is about."""

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import render_gpu

N = dace.symbol("N")
M = dace.symbol("M")


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


if __name__ == "__main__":
    test_a_grid_stride_reduction_kernel_says_reduction_and_grid_stride()
    test_a_block_per_row_kernel_has_one_line_per_kernel_and_per_thread_block()
    test_a_lane_strided_reduction_says_reduction_and_carries_no_host_pragma()
    test_a_kernel_library_loop_names_the_kernel_not_its_launch_bounds()
    test_a_simd_loop_says_vectorized_not_not_parallelized()
