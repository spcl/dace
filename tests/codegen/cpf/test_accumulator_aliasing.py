# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An accumulator the generator holds outside memory (an OpenMP reduction's private copy, a loop's register) covers
every write of the loop to its array, and its declaration stays inside its own map's scope. Each case runs on four
OpenMP threads and is checked against numpy. The register case of a top-level loop is in
``tests/codegen/sequential_wcr_accumulator_test.py``: CPF schedules that loop itself."""

import ctypes

import numpy as np

import dace
from dace.codegen.cpf import render
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)
THREADS = 4


def two_accumulations(name: str, first: str, second: str) -> dace.SDFG:
    """A parallel map over ``j`` whose two tasklets add into ``out`` through the map exit, each its own write:
    ``out[first] += p[j]`` and ``out[second] += q[j]``."""
    sdfg = dace.SDFG(name)
    for array in ("p", "q", "out"):
        sdfg.add_array(array, [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("accumulate", {"j": "0:N"}, schedule=dace.ScheduleType.CPU_Multicore)
    out = state.add_write("out")
    for source, index in (("p", first), ("q", second)):
        tasklet = state.add_tasklet(f"add_{source}", {"v"}, {"o"}, "o = v")
        state.add_memlet_path(state.add_read(source), entry, tasklet, dst_conn="v", memlet=dace.Memlet(f"{source}[j]"))
        sum_into = dace.Memlet(f"out[{index}]", wcr="lambda a, b: a + b")
        state.add_memlet_path(tasklet, exit_, out, src_conn="o", memlet=sum_into)
    return sdfg


@dace.program
def two_row_sums(a: dace.float64[N, M], b: dace.float64[N, M], c: dace.float64[N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:M] @ dace.ScheduleType.Sequential:
            c[i] += a[i, j]
        for j in dace.map[0:M] @ dace.ScheduleType.Sequential:
            c[i] += b[i, j]


@dace.program
def sum_and_scatter(a: dace.float64[N], b: dace.float64[N], s: dace.float64[N]):
    for i in dace.map[0:N]:
        s[0] += b[i]
        s[i] += a[i]


def run(sdfg: dace.SDFG, arguments: dict) -> str:
    """Render ``sdfg`` in C, run it on :data:`THREADS` OpenMP threads and return the rendered code."""
    code = render(sdfg, language="c").code
    library = build_standalone(code, sdfg.name, language="c")
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(THREADS)
    call_standalone(library, sdfg, arguments)
    return code


def test_sibling_loops_hold_one_element_each_in_their_own_scope():
    sdfg = two_row_sums.to_sdfg(simplify=True)
    sdfg.name = "aliasing_two_row_sums"
    rng = np.random.default_rng(4)
    n, m = 9, 5
    a, b, c = rng.random((n, m)), rng.random((n, m)), rng.random(n)
    want = c + a.sum(axis=1) + b.sum(axis=1)
    code = run(sdfg, {"a": a, "b": b, "c": c, "N": n, "M": m})
    np.testing.assert_allclose(c, want, err_msg=code)


def test_a_reduction_clause_covers_every_write_of_its_array():
    sdfg = two_accumulations("aliasing_sum_and_scatter", "0", "j")
    sdfg.openmp_array_reductions = True
    rng = np.random.default_rng(5)
    n = 64
    p, q, out = rng.random(n), rng.random(n), rng.random(n)
    want = out + q
    want[0] += p.sum()
    code = run(sdfg, {"p": p, "q": q, "out": out, "N": n})
    np.testing.assert_allclose(out, want, err_msg=code)


if __name__ == "__main__":
    test_sibling_loops_hold_one_element_each_in_their_own_scope()
    test_a_reduction_clause_covers_every_write_of_its_array()
