# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF forms are race-free and read only what the program reads.

* A scalar written only inside a parallel map is declared inside it, even when a map elsewhere has a parameter
  of the same name (fft_3d: ``j`` beside an inverse FFT's normalization map over ``j``).
* A sequential loop outlined into a thread team exports no exit value nothing reads (trmm, bitonic_sort).
* A loop whose iterations all write one element is parallel: each iteration writes a private copy, and after the
  loop the element takes the value of the last iteration that wrote it (CloudSC's ``ZVQX(NCLDQI)``).
"""

import ctypes
import os
import re

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

N = dace.symbol("N")
M = dace.symbol("M")


def map_local_scalar_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("cpf_map_local_scalar")
    sdfg.add_array("q", [N], dace.int64)
    sdfg.add_array("r", [N], dace.int64)
    sdfg.add_scalar("j", dace.int64, transient=True)
    first = sdfg.add_state("first", is_start_block=True)
    entry, exit_node = first.add_map("fill", {"i": "0:N"}, schedule=dace.ScheduleType.CPU_Multicore)
    produce = first.add_tasklet("produce", {}, {"o"}, "o = i + 1")
    consume = first.add_tasklet("consume", {"a"}, {"o"}, "o = 2 * a")
    scalar = first.add_access("j")
    first.add_nedge(entry, produce, dace.Memlet())
    first.add_edge(produce, "o", scalar, None, dace.Memlet("j[0]"))
    first.add_edge(scalar, None, consume, "a", dace.Memlet("j[0]"))
    first.add_memlet_path(consume, exit_node, first.add_write("q"), src_conn="o", memlet=dace.Memlet("q[i]"))
    # A map over a parameter named like the scalar: its tasklet reads the parameter, not the container.
    second = sdfg.add_state_after(first, "second")
    second.add_mapped_tasklet(
        "index",
        {"j": "0:N"},
        {},
        "o = j",
        {"o": dace.Memlet("r[j]")},
        external_edges=True,
        schedule=dace.ScheduleType.CPU_Multicore,
    )
    return sdfg


def test_a_map_local_scalar_is_declared_inside_its_parallel_loop():
    sdfg = map_local_scalar_sdfg()
    result = render(sdfg, language="c")
    code = result.code
    parallel = code.index("#pragma omp parallel for")
    declaration = re.search(r"\bint64_t j;", code)
    assert declaration is not None and declaration.start() > parallel, code
    library = build_standalone(code, sdfg.name, language="c")
    n = 1 << 16
    got = {"q": np.zeros(n, dtype=np.int64), "r": np.zeros(n, dtype=np.int64)}
    call_standalone(library, result.sdfg, {**got, "N": n})
    assert_matches({"q": 2 * np.arange(1, n + 1), "r": np.arange(n)}, got)


@dace.program
def trmm(alpha: dace.float64, A: dace.float64[M, M], B: dace.float64[M, N]):
    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
    B *= alpha


def test_an_outlined_loop_exports_no_unread_exit_value():
    sdfg = trmm.to_sdfg(simplify=True)
    sdfg.name = "cpf_trmm_exit"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    assert "#pragma omp parallel num_threads" in result.code, result.code
    assert re.search(r"\bsym_out", result.code) is None, result.code
    library = build_standalone(result.code, sdfg.name, language="c")
    rng = np.random.default_rng(3)
    m, n = 64, 2048
    a, b = rng.random((m, m)), rng.random((m, n))
    expected = b.copy()
    for i in range(m):
        expected[i, :] += a[i + 1 :, i] @ expected[i + 1 :, :]
    expected *= 1.5
    call_standalone(library, result.sdfg, {"A": a, "B": b, "alpha": 1.5, "M": m, "N": n})
    assert_matches({"B": expected}, {"B": b})


#: OpenMP team sizes each last-write case runs at.
THREAD_COUNTS = (1, 2, 3, 8)


def last_write_wins_line(slot: str) -> str:
    """The class line and pragma of a parallel loop that copies ``slot`` out after it."""
    text = f"// parallel -- last write wins: {slot} keeps the value of the last iteration that writes it"
    return re.escape(text) + r"\n\s*#pragma omp parallel for"


def rendered_library(program: dace.frontend.python.parser.DaceProgram, name: str) -> tuple[dace.SDFG, str, object]:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    return result.sdfg, result.code, build_standalone(result.code, sdfg.name, language="c")


def run_on_threads(library: object, sdfg: dace.SDFG, arguments: dict, threads: int) -> None:
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(threads)
    call_standalone(library, sdfg, arguments)


@dace.program
def last_write_unconditional(a: dace.float64[N], b: dace.float64[N], x: dace.float64[4], y: dace.float64[1]):
    for i in range(N):
        x[1] = a[i] * 2.0
        b[i] = x[1] + 1.0
    y[0] = x[1] * 3.0


def test_an_element_every_iteration_writes_keeps_the_last_iterations_value():
    sdfg, code, library = rendered_library(last_write_unconditional, "cpf_last_write_unconditional")
    assert re.search(last_write_wins_line("x[1]"), code), code
    n = 1000
    a = np.random.default_rng(7).random(n)
    for threads in THREAD_COUNTS:
        b, x, y = np.zeros(n), np.arange(4.0), np.zeros(1)
        run_on_threads(library, sdfg, {"a": a, "b": b, "x": x, "y": y, "N": n}, threads)
        assert_matches(
            {"b": 2.0 * a + 1.0, "x": np.array([0.0, 2.0 * a[-1], 2.0, 3.0]), "y": 6.0 * a[-1:]},
            {"b": b, "x": x, "y": y},
            f"{threads} threads",
        )


@dace.program
def last_write_conditional(a: dace.float64[N], b: dace.float64[N], x: dace.float64[4], y: dace.float64[1]):
    for i in range(N):
        if a[i] > 0.5:
            x[1] = a[i] * 2.0
            b[i] = x[1] + 1.0
    y[0] = x[1] * 3.0


def test_an_element_some_iterations_write_keeps_the_value_of_the_last_one_that_did():
    sdfg, code, library = rendered_library(last_write_conditional, "cpf_last_write_conditional")
    assert re.search(last_write_wins_line("x[1]"), code), code
    n = 1000
    a = np.random.default_rng(8).random(n)
    # The last iterations skip the write: their own value is not the one the element keeps
    a[-3:] = 0.25
    hit = np.flatnonzero(a > 0.5)[-1]
    for threads in THREAD_COUNTS:
        b, x, y = np.zeros(n), np.arange(4.0), np.zeros(1)
        run_on_threads(library, sdfg, {"a": a, "b": b, "x": x, "y": y, "N": n}, threads)
        want_b = np.where(a > 0.5, 2.0 * a + 1.0, 0.0)
        want_x = np.array([0.0, 2.0 * a[hit], 2.0, 3.0])
        assert_matches({"b": want_b, "x": want_x, "y": 6.0 * a[hit : hit + 1]}, {"b": b, "x": x, "y": y})
    # No iteration writes: the element keeps its value from before the loop
    b, x, y = np.zeros(n), np.arange(4.0), np.zeros(1)
    run_on_threads(library, sdfg, {"a": np.zeros(n), "b": b, "x": x, "y": y, "N": n}, 4)
    assert_matches({"x": np.arange(4.0), "y": np.array([3.0])}, {"x": x, "y": y})


@dace.program
def last_write_invariant_guard(
    a: dace.float64[N], b: dace.float64[N], x: dace.float64[4], f: dace.int64, k: dace.int64
):
    for i in range(N):
        if f != 0 and k == 2:
            x[1] = a[i] * 0.002
        b[i] = a[i] * x[k - 1]


def test_an_element_written_under_a_loop_invariant_guard_and_read_at_a_symbolic_index_is_private():
    """CloudSC's sedimentation loop: ``ZVQX(NCLDQI)`` written for every column when ``LAERICESED`` and the species is
    ice, then read as ``ZVQX(JM)``."""
    sdfg, code, library = rendered_library(last_write_invariant_guard, "cpf_last_write_invariant_guard")
    assert re.search(last_write_wins_line("x[1]"), code), code
    n = 1000
    a = np.random.default_rng(9).random(n)
    for threads in THREAD_COUNTS:
        for f, k in ((1, 2), (0, 2), (1, 3)):
            b, x = np.zeros(n), np.arange(4.0) + 1.0
            run_on_threads(library, sdfg, {"a": a, "b": b, "x": x, "f": f, "k": k, "N": n}, threads)
            want_x = np.arange(4.0) + 1.0
            if f and k == 2:
                want_x[1] = a[-1] * 0.002
                want_b = a * a * 0.002
            else:
                want_b = a * want_x[k - 1]
            assert_matches({"b": want_b, "x": want_x}, {"b": b, "x": x}, f"f={f} k={k} on {threads} threads")


@dace.program
def carried_through_a_varying_guard(a: dace.float64[N], b: dace.float64[N], x: dace.float64[4]):
    for i in range(N):
        if a[i] > 0.5:
            x[1] = a[i]
        b[i] = x[1] * 3.0


def test_an_element_read_where_a_varying_guard_skipped_its_write_keeps_the_carried_value():
    """The read sees the value of an EARLIER iteration where the guard skipped the write: no copy per iteration."""
    sdfg, code, library = rendered_library(carried_through_a_varying_guard, "cpf_carried_varying_guard")
    assert "last write wins" not in code, code
    n = 1000
    a = np.random.default_rng(10).random(n)
    want_b, want_x = np.zeros(n), np.arange(4.0)
    for i in range(n):
        if a[i] > 0.5:
            want_x[1] = a[i]
        want_b[i] = want_x[1] * 3.0
    for threads in THREAD_COUNTS:
        b, x = np.zeros(n), np.arange(4.0)
        run_on_threads(library, sdfg, {"a": a, "b": b, "x": x, "N": n}, threads)
        assert_matches({"b": want_b, "x": want_x}, {"b": b, "x": x}, f"{threads} threads")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_map_local_scalar_is_declared_inside_its_parallel_loop()
    test_an_outlined_loop_exports_no_unread_exit_value()
    test_an_element_every_iteration_writes_keeps_the_last_iterations_value()
    test_an_element_some_iterations_write_keeps_the_value_of_the_last_one_that_did()
    test_an_element_written_under_a_loop_invariant_guard_and_read_at_a_symbolic_index_is_private()
    test_an_element_read_where_a_varying_guard_skipped_its_write_keeps_the_carried_value()
