# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF forms are race-free and read only what the program reads.

* A scalar written only inside a parallel map is declared inside it, even when a map elsewhere has a parameter
  of the same name (fft_3d: ``j`` beside an inverse FFT's normalization map over ``j``).
* A sequential loop outlined into a thread team exports no exit value nothing reads (trmm, bitonic_sort).
"""

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


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_map_local_scalar_is_declared_inside_its_parallel_loop()
    test_an_outlined_loop_exports_no_unread_exit_value()
