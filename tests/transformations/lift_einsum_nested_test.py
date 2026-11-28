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
    assert sdfg.apply_transformations(LiftEinsum, options={"per_slice": True}) == 1
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
    assert sdfg.apply_transformations(LiftEinsum, options={"per_slice": True}) == 0


def test_outer_product_under_an_enclosing_map_stays_a_map():
    """A slice of an outer product has no contracted index: lifted, it is a K=1 GEMM per enclosing iteration."""
    sdfg = dace.SDFG("lift_einsum_nested_outer")
    sdfg.add_array("A", (8, 3), dace.float64)
    sdfg.add_array("B", (8, 4), dace.float64)
    sdfg.add_array("C", (8, 3, 4), dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", {"p": "0:8"})
    inner_entry, inner_exit = state.add_map("inner", {"i": "0:3", "j": "0:4"})
    tasklet = state.add_tasklet("product", {"x", "y"}, {"z"}, "z = x * y")
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    state.add_memlet_path(a, outer_entry, inner_entry, tasklet, dst_conn="x", memlet=dace.Memlet("A[p, i]"))
    state.add_memlet_path(b, outer_entry, inner_entry, tasklet, dst_conn="y", memlet=dace.Memlet("B[p, j]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, c, src_conn="z", memlet=dace.Memlet("C[p, i, j]"))
    sdfg.fill_scope_connectors()
    propagation.propagate_memlets_sdfg(sdfg)
    assert sdfg.apply_transformations(LiftEinsum, options={"per_slice": True}) == 0


def local_outer_product(name: str, enclosed: bool) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (8, 3), dace.float64)
    sdfg.add_array("B", (8, 4), dace.float64)
    sdfg.add_array("C", (8, 3, 4), dace.float64)
    for local, shape in (("x", (3,)), ("y", (4,)), ("z", (3, 4))):
        sdfg.add_array(local, shape, dace.float64, transient=True)
    state = sdfg.add_state()
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    x, y, z = state.add_access("x"), state.add_access("y"), state.add_access("z")
    inner_entry, inner_exit = state.add_map("inner", {"i": "0:3", "j": "0:4"})
    tasklet = state.add_tasklet("product", {"u", "v"}, {"w"}, "w = u * v")
    state.add_memlet_path(x, inner_entry, tasklet, dst_conn="u", memlet=dace.Memlet("x[i]"))
    state.add_memlet_path(y, inner_entry, tasklet, dst_conn="v", memlet=dace.Memlet("y[j]"))
    state.add_memlet_path(tasklet, inner_exit, z, src_conn="w", memlet=dace.Memlet("z[i, j]"))
    p = "p" if enclosed else "0"
    entry, exit_ = state.add_map("outer", {"p": "0:8"}) if enclosed else (None, None)
    for src, dst, sub, n in ((a, x, "0:3", 3), (b, y, "0:4", 4)):
        hops = [h for h in (src, entry, dst) if h is not None]
        state.add_memlet_path(*hops, memlet=dace.Memlet(data=src.data, subset=f"{p}, {sub}", other_subset=f"0:{n}"))
    hops = [h for h in (z, exit_, c) if h is not None]
    state.add_memlet_path(*hops, memlet=dace.Memlet(data="z", subset="0:3, 0:4", other_subset=f"{p}, 0:3, 0:4"))
    sdfg.fill_scope_connectors()
    propagation.propagate_memlets_sdfg(sdfg)
    return sdfg


def test_outer_product_of_local_operands_under_an_enclosing_map_stays_a_map():
    sdfg = local_outer_product("lift_einsum_local_outer_enclosed", enclosed=True)
    assert sdfg.apply_transformations(LiftEinsum, options={"per_slice": True}) == 0


def test_outer_product_of_local_operands_at_top_level_lifts():
    sdfg = local_outer_product("lift_einsum_local_outer_top", enclosed=False)
    assert sdfg.apply_transformations(LiftEinsum, options={"per_slice": True}) == 1


if __name__ == "__main__":
    test_contraction_under_an_enclosing_map_lifts_per_slice()
    test_enclosing_map_the_output_does_not_index_stays_a_reduction()
    test_outer_product_under_an_enclosing_map_stays_a_map()
    test_outer_product_of_local_operands_under_an_enclosing_map_stays_a_map()
    test_outer_product_of_local_operands_at_top_level_lifts()
