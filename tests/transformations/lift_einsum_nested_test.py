# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries.blas import nodes as blas
from dace.sdfg import propagation
from dace.transformation.dataflow import LiftEinsum


def batch_of_products(name: str, output: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (2, 3, 4), dace.float64)
    sdfg.add_array("B", (5, 4, 6), dace.float64)
    sdfg.add_array("C", (2, 5, 3, 6), dace.float64)
    state = sdfg.add_state()
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    outer_entry, outer_exit = state.add_map("outer", {"o": "0:2"})
    inner_entry, inner_exit = state.add_map("inner", {"f": "0:5", "i": "0:3", "k": "0:4", "j": "0:6"})
    tasklet = state.add_tasklet("product", {"x", "y"}, {"z"}, "z = x * y")
    state.add_memlet_path(a, outer_entry, inner_entry, tasklet, dst_conn="x", memlet=dace.Memlet("A[o, i, k]"))
    state.add_memlet_path(b, outer_entry, inner_entry, tasklet, dst_conn="y", memlet=dace.Memlet("B[f, k, j]"))
    state.add_memlet_path(
        tasklet,
        inner_exit,
        outer_exit,
        c,
        src_conn="z",
        memlet=dace.Memlet(f"C[{output}, f, i, j]", wcr="lambda a, b: a + b"),
    )
    sdfg.fill_scope_connectors()
    propagation.propagate_memlets_sdfg(sdfg)

    return sdfg


def test_contraction_under_an_enclosing_map_lifts_per_slice():
    sdfg = batch_of_products("lift_einsum_nested_slice", "o")
    assert sdfg.apply_transformations(LiftEinsum) == 1
    state = sdfg.start_state
    einsum = next(n for n in state.nodes() if isinstance(n, blas.Einsum))
    assert isinstance(state.entry_node(einsum), dace.nodes.MapEntry)
    assert all(e.data.wcr is None for e in state.edges())

    a, b = np.random.rand(2, 3, 4), np.random.rand(5, 4, 6)
    c = np.zeros((2, 5, 3, 6))
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, np.einsum("oik,fkj->ofij", a, b))


def test_enclosing_map_the_output_does_not_index_stays_a_reduction():
    sdfg = batch_of_products("lift_einsum_nested_shared", "0")
    assert sdfg.apply_transformations(LiftEinsum) == 0


if __name__ == "__main__":
    test_contraction_under_an_enclosing_map_lifts_per_slice()
    test_enclosing_map_the_output_does_not_index_stays_a_reduction()
