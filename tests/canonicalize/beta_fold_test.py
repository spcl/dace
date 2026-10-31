# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BetaFold``: ``X = X + c * T`` after the contraction that writes ``T`` becomes that contraction with ``beta=1``."""

import numpy as np

import dace
from dace.libraries.blas import nodes as blas
from dace.transformation.passes.canonicalize.beta_fold import BetaFold
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, K, M = 6, 7, 8


def contraction(state: dace.SDFGState, kind: str, a: str, b: str, out: str) -> dace.nodes.LibraryNode:
    if kind == "einsum":
        node = blas.Einsum("contract")
        node.einsum_str = "ij,jk->ik"
        ins, out_conn = ("_ein00", "_ein01"), "_out"
        for conn in ins:
            node.add_in_connector(conn)
        node.add_out_connector(out_conn)
    elif kind == "matmul":
        node, ins, out_conn = blas.MatMul("contract"), ("_a", "_b"), "_c"
    else:
        node, ins, out_conn = blas.Gemv("contract"), ("_A", "_x"), "_y"
    state.add_node(node)
    sdfg = state.sdfg
    for name, conn in zip((a, b), ins):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    state.add_edge(node, out_conn, state.add_write(out), None, dace.Memlet.from_array(out, sdfg.arrays[out]))
    return node


def contraction_then_add(
    name: str, kind: str, code: str = "__x_out = __x + 0.5 * __t", x_index: str = "i, j"
) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    vector = kind == "gemv"
    sdfg.add_array("A", [N, K], dace.float64)
    sdfg.add_array("B", [K] if vector else [K, M], dace.float64)
    out_shape = [N] if vector else [N, M]
    sdfg.add_array("X", [N] if vector else ([N, 2 * M] if x_index != "i, j" else out_shape), dace.float64)
    sdfg.add_array("D", out_shape, dace.float64)
    sdfg.add_transient("T", out_shape, dace.float64)
    state = sdfg.add_state("main")
    node = contraction(state, kind, "A", "B", "T")
    temp = next(e.dst for e in state.out_edges(node))
    index = "i" if vector else "i, j"
    ranges = {"i": f"0:{N}"} if vector else {"i": f"0:{N}", "j": f"0:{M}"}
    state.add_mapped_tasklet(
        "add",
        ranges,
        {"__x": dace.Memlet(f"X[{'i' if vector else x_index}]"), "__t": dace.Memlet(f"T[{index}]")},
        code,
        {"__x_out": dace.Memlet(f"X[{'i' if vector else x_index}]")},
        input_nodes={"T": temp},
        external_edges=True,
    )
    return sdfg


def expected(A: np.ndarray, B: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X + 0.5 * (A @ B)


def operands(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    vector = kind == "gemv"
    return rng.random((N, K)), rng.random((K,) if vector else (K, M)), rng.random((N,) if vector else (N, M))


def assert_folded(sdfg: dace.SDFG) -> None:
    state = sdfg.start_block
    assert "T" not in sdfg.arrays and not any(n.data == "T" for n in state.data_nodes())
    assert not any(isinstance(n, (dace.nodes.MapEntry, dace.nodes.Tasklet)) for n in state.nodes())
    (node,) = [n for n in state.nodes() if isinstance(n, dace.nodes.LibraryNode)]
    assert float(node.beta) == 1.0 and float(node.alpha) == 0.5


def check_fold(kind: str) -> None:
    sdfg = contraction_then_add(f"beta_fold_{kind}", kind)
    assert BetaFold().apply_pass(sdfg, {}) == 1
    assert_folded(sdfg)
    A, B, X = operands(kind)
    reference = expected(A, B, X)
    sdfg(A=A, B=B, X=X, D=np.zeros_like(X))
    assert np.allclose(X, reference)


def test_einsum_folds_the_add():
    check_fold("einsum")


def test_matmul_folds_the_add():
    check_fold("matmul")


def test_gemv_folds_the_add_and_reads_y():
    check_fold("gemv")


def assert_not_folded(sdfg: dace.SDFG) -> None:
    before = sorted(str(n) for n, _ in sdfg.all_nodes_recursive())
    assert BetaFold().apply_pass(sdfg, {}) is None
    assert sorted(str(n) for n, _ in sdfg.all_nodes_recursive()) == before


def test_temporary_read_twice_is_not_folded():
    sdfg = contraction_then_add("beta_fold_read_twice", "matmul")
    state = sdfg.start_block
    temp = next(n for n in state.data_nodes() if n.data == "T")
    state.add_edge(temp, None, state.add_write("D"), None, dace.Memlet("T[0:6, 0:8]"))
    assert_not_folded(sdfg)
    A, B, X = operands("matmul")
    reference = expected(A, B, X)
    D = np.zeros_like(X)
    sdfg(A=A, B=B, X=X, D=D)
    assert np.allclose(X, reference) and np.allclose(D, A @ B)


def test_output_that_is_an_operand_is_not_folded():
    sdfg = dace.SDFG("beta_fold_operand")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("X", [N, N], dace.float64)
    sdfg.add_transient("T", [N, N], dace.float64)
    state = sdfg.add_state("main")
    node = contraction(state, "matmul", "A", "X", "T")
    temp = next(e.dst for e in state.out_edges(node))
    state.add_mapped_tasklet(
        "add",
        {"i": f"0:{N}", "j": f"0:{N}"},
        {"__x": dace.Memlet("X[i, j]"), "__t": dace.Memlet("T[i, j]")},
        "__x_out = __x + __t",
        {"__x_out": dace.Memlet("X[i, j]")},
        input_nodes={"T": temp},
        external_edges=True,
    )
    assert_not_folded(sdfg)
    rng = np.random.default_rng(5)
    A, X = rng.random((N, N)), rng.random((N, N))
    reference = X + A @ X
    sdfg(A=A, X=X)
    assert np.allclose(X, reference)


def test_strided_output_region_is_not_folded():
    sdfg = contraction_then_add("beta_fold_strided", "matmul", x_index="i, 2 * j")
    assert_not_folded(sdfg)
    A, B, _ = operands("matmul")
    X = np.random.default_rng(3).random((N, 2 * M))
    reference = X.copy()
    reference[:, ::2] += 0.5 * (A @ B)
    sdfg(A=A, B=B, X=X, D=np.zeros((N, M)))
    assert np.allclose(X, reference)


def test_non_add_combine_is_not_folded():
    sdfg = contraction_then_add("beta_fold_product", "matmul", code="__x_out = __x * __t")
    assert_not_folded(sdfg)
    A, B, X = operands("matmul")
    reference = X * (A @ B)
    sdfg(A=A, B=B, X=X, D=np.zeros_like(X))
    assert np.allclose(X, reference)


def test_frontend_accumulated_einsum_lowers_to_one_gemm_with_beta_one():
    @dace.program
    def beta_fold_frontend(A: dace.float64[N, K], B: dace.float64[K, M], Q: dace.float64[N, M]):
        Q[:] = Q + np.einsum("ij,jk->ik", A, B)

    sdfg = finalize_for_target(canonicalize(beta_fold_frontend.to_sdfg(simplify=False)), "cpu")
    gemms = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, blas.Gemm)]
    assert len(gemms) == 1 and float(gemms[0].beta) == 1.0
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())
    A, B, Q = operands("matmul")
    reference = Q + A @ B
    sdfg(A=A, B=B, Q=Q)
    assert np.allclose(Q, reference)


if __name__ == "__main__":
    test_einsum_folds_the_add()
    test_matmul_folds_the_add()
    test_gemv_folds_the_add_and_reads_y()
    test_temporary_read_twice_is_not_folded()
    test_output_that_is_an_operand_is_not_folded()
    test_strided_output_region_is_not_folded()
    test_non_add_combine_is_not_folded()
    test_frontend_accumulated_einsum_lowers_to_one_gemm_with_beta_one()
