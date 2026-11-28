# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` gates an inner arm's masked write with the outer guard instead of redirecting it.

CloudSC zeroes ``zacust[jl]`` under ``zlfinalsum[jl] < eps`` inside ``jk > ncldtop``. The inner arm became the masked
write ``IT(c, 0)``; the outer gate then redirected it into a fresh scratch, which ``c`` failing left unset, and stored
that scratch under the outer guard.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

N, M, s = (dace.symbol(name) for name in ("N", "M", "s"))


@dace.program
def nested_guards(A: dace.float64[M], B: dace.float64[N], t: dace.int64):
    # ``A[i + s]`` is not provably in range, so each arm's write is masked
    for i in dace.map[0:N]:
        if i > t:
            if B[i] < 0.5:
                A[i + s] = 0.0


def expected(A: np.ndarray, B: np.ndarray, t: int, shift: int) -> np.ndarray:
    want = A.copy()
    for i in range(B.size):
        if i > t and B[i] < 0.5:
            want[i + shift] = 0.0
    return want


def run(sdfg: dace.SDFG) -> None:
    rng = np.random.default_rng(0)
    A, B = rng.random(40), rng.random(32)
    want = expected(A, B, 6, 3)
    sdfg(A=A, B=B, t=6, N=32, M=40, s=3)
    assert np.allclose(A, want)


def test_an_inner_masked_write_takes_the_outer_guard():
    sdfg = nested_guards.to_sdfg(simplify=True)
    BranchNormalization().apply_pass(sdfg, {})
    sdfg.validate()
    gates = [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and "IT(" in node.code.as_string
    ]
    assert len(gates) == 1, [gate.code.as_string for gate in gates]
    run(sdfg)


def test_nested_guards_vectorized_match_numpy():
    sdfg = nested_guards.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()
    run(sdfg)


if __name__ == "__main__":
    test_an_inner_masked_write_takes_the_outer_guard()
    test_nested_guards_vectorized_match_numpy()
