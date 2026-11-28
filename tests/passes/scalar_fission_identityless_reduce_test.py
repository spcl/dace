# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar fission keeps a library node that folds onto its output on the container its initializer wrote.

A ``Reduce`` with ``identity=None`` folds into the value its output already holds, a read-modify-write
like a WCR edge. Fission read it as a fresh definition and gave it a new name, so the ``s = 0.0``
before TSVC vsumr's sum loop became a dead write to the old name and was removed: on the GPU the sum
then accumulated into an uninitialized device scalar and came back 9x, 10x, ... over repeated calls.
An ``Einsum`` with ``beta`` 1 folds the same way: jfnk_bratu's lifted ``s = s + A[i, :] @ A[i, :]`` lost its
``s = 0.0`` and the norm came back NaN.
"""

import numpy as np

import dace
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.standard import Reduce
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.scalar_fission import ScalarFission

N = dace.symbol("N")


def zero_then_fold(fold_node: dace.nodes.LibraryNode, inputs: dict[str, str]) -> dace.SDFG:
    """``s = 0; s = fold(inputs, into s); out[0] = s``, ``inputs`` mapping each connector to its memlet."""
    sdfg = dace.SDFG(f"zero_then_{type(fold_node).__name__.lower()}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    init = sdfg.add_state("init", is_start_block=True)
    zero = init.add_tasklet("zero", [], ["y"], "y = 0.0")
    init.add_edge(zero, "y", init.add_write("s"), None, dace.Memlet("s[0]"))
    fold = sdfg.add_state_after(init, "fold")
    fold.add_node(fold_node)
    read = fold.add_read("a")
    for conn, memlet in inputs.items():
        fold.add_edge(read, None, fold_node, conn, dace.Memlet(memlet))
    (out_conn,) = fold_node.out_connectors
    fold.add_edge(fold_node, out_conn, fold.add_write("s"), None, dace.Memlet("s[0]"))
    copy = sdfg.add_state_after(fold, "copy")
    tasklet = copy.add_tasklet("copy", ["x"], ["y"], "y = x")
    copy.add_edge(copy.add_read("s"), None, tasklet, "x", dace.Memlet("s[0]"))
    copy.add_edge(tasklet, "y", copy.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def zero_then_reduce() -> dace.SDFG:
    return zero_then_fold(Reduce("sum", "lambda x, y: x + y", axes=None, identity=None), {"_in": "a[0:N]"})


def zero_then_einsum() -> dace.SDFG:
    """``s = s + a @ a`` as one ``Einsum`` with ``beta`` 1."""
    node = Einsum("dot")
    node.einsum_str, node.beta = "k,k->", 1.0
    for conn in ("_ein00", "_ein01"):
        node.add_in_connector(conn)
    node.add_out_connector("_out")
    return zero_then_fold(node, {"_ein00": "a[0:N]", "_ein01": "a[0:N]"})


def fold_outputs(sdfg: dace.SDFG, kind: type) -> list:
    return [state.out_edges(node)[0].dst.data for node, state in sdfg.all_nodes_recursive() if isinstance(node, kind)]


def reduce_outputs(sdfg: dace.SDFG) -> list:
    return fold_outputs(sdfg, Reduce)


def zero_writes(sdfg: dace.SDFG) -> list:
    """Containers a no-input tasklet writes ``0.0`` into."""
    return [
        state.out_edges(node)[0].data.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.Tasklet) and not node.in_connectors and "0.0" in node.code.as_string
    ]


def test_fission_does_not_split_an_identityless_reduce_from_its_initializer():
    sdfg = zero_then_reduce()
    Pipeline([ScalarFission()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert set(reduce_outputs(sdfg)) <= set(zero_writes(sdfg)), (reduce_outputs(sdfg), zero_writes(sdfg))


def test_fission_does_not_split_a_folding_einsum_from_its_initializer():
    sdfg = zero_then_einsum()
    Pipeline([ScalarFission()]).apply_pass(sdfg, {})
    sdfg.validate()
    outputs = fold_outputs(sdfg, Einsum)
    assert set(outputs) <= set(zero_writes(sdfg)), (outputs, zero_writes(sdfg))


@dace.program
def vsumr(a: dace.float64[N], sum_out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        s = s + a[i]
    sum_out[0] = s


def test_a_canonicalized_sum_keeps_the_initializer_its_reduce_accumulates_into():
    sdfg = vsumr.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    outputs = reduce_outputs(sdfg)
    assert outputs, "the loop was expected to become a Reduce"
    assert set(outputs) <= set(zero_writes(sdfg)), (outputs, zero_writes(sdfg))
    a = np.random.default_rng(0).standard_normal(37)
    out = np.zeros(1)
    for _ in range(3):
        sdfg(a=a, sum_out=out, N=37)
        np.testing.assert_allclose(out, [a.sum()], rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    test_fission_does_not_split_an_identityless_reduce_from_its_initializer()
    test_fission_does_not_split_a_folding_einsum_from_its_initializer()
    test_a_canonicalized_sum_keeps_the_initializer_its_reduce_accumulates_into()
