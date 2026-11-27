# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A per-row reduction into a scalar stays the row's own value: CSR ``r = A @ x`` with ``b @ r`` keeps one parallel loop.

Exposing a privatized accumulator as a nested SDFG input (a96ca3546d) also exposed a per-iteration transient scalar,
which then read a value from before its row: the row sum became an N-sized buffer filled by a loop of its own, and the
consumer a second loop (bicgstab, qe_addusxx_g: 0.57-0.72x on CPU)."""

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N")
NNZ = dace.symbol("NNZ")


@dace.program
def csr_residual(
    A_data: dace.float64[NNZ],
    A_indices: dace.int64[NNZ],
    A_indptr: dace.int64[N + 1],
    x: dace.float64[N],
    b: dace.float64[N],
    r: dace.float64[N],
    dot: dace.float64[1],
):
    dot[0] = 0.0
    for i in range(N):
        s = 0.0
        for k in range(A_indptr[i], A_indptr[i + 1]):
            s += A_data[k] * x[A_indices[k]]
        dot[0] += b[i] * s
        r[i] = s


def test_a_row_sum_stays_a_scalar_of_its_row():
    sdfg = csr_residual.to_sdfg(simplify=True)
    sdfg.name = "cpf_private_row_reduction"
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    result = render(sdfg, language="c")
    entry = result.code[result.code.index("void cpf_private_row_reduction(") :]
    assert result.workspace_bytes == "0", f"the row sum became a buffer of {result.workspace_bytes} bytes\n{entry}"
    assert entry.count("for (int64_t") == 1, entry
    rng = np.random.default_rng(5)
    n = 50
    indptr = np.concatenate([[0], np.cumsum(rng.integers(0, 6, n))]).astype(np.int64)
    nnz = int(indptr[-1])
    arguments = {
        "A_data": rng.random(nnz),
        "A_indices": rng.integers(0, n, nnz).astype(np.int64),
        "A_indptr": indptr,
        "x": rng.random(n),
        "b": rng.random(n),
        "r": np.zeros(n),
        "dot": np.zeros(1),
        "N": n,
        "NNZ": nnz,
    }
    expected = np.array(
        [
            sum(
                arguments["A_data"][k] * arguments["x"][arguments["A_indices"][k]]
                for k in range(indptr[i], indptr[i + 1])
            )
            for i in range(n)
        ]
    )
    library = build_standalone(result.code, sdfg.name, language="c")
    call_standalone(library, result.sdfg, arguments, result.arguments)
    np.testing.assert_allclose(arguments["r"], expected, rtol=1e-13)
    np.testing.assert_allclose(arguments["dot"][0], arguments["b"] @ expected, rtol=1e-12)


if __name__ == "__main__":
    test_a_row_sum_stays_a_scalar_of_its_row()
