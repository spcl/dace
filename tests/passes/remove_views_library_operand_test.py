# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RemoveViews removes a View next to a library node where the node reads the viewed slice in its place
(``LibraryNode.operand_dims``), and keeps it where the node would read other dimensions."""

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.libraries.blas import MatMul
from dace.libraries.linalg import TensorDot
from dace.libraries.standard import Reduce
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.remove_views import RemoveViews

L, M, K, N = 3, 5, 7, 6


def views(sdfg: dace.SDFG) -> list[str]:
    return [
        node.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.AccessNode) and isinstance(node.desc(state.sdfg), dt.View)
    ]


def matmul_into_plane_sdfg(name: str) -> tuple[dace.SDFG, MatMul]:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [K, N], dace.float64)
    sdfg.add_array("X", [L, M, N], dace.float64)
    sdfg.add_view("xv", [M, N], dace.float64, strides=[N, 1])
    state = sdfg.add_state()
    mm = MatMul("mm")
    state.add_node(mm)
    state.add_edge(state.add_read("A"), None, mm, "_a", dace.Memlet(f"A[0:{M}, 0:{K}]"))
    state.add_edge(state.add_read("B"), None, mm, "_b", dace.Memlet(f"B[0:{K}, 0:{N}]"))
    xv = state.add_access("xv")
    state.add_edge(mm, "_c", xv, None, dace.Memlet(f"xv[0:{M}, 0:{N}]"))
    state.add_edge(xv, "views", state.add_write("X"), None, dace.Memlet(f"X[1, 0:{M}, 0:{N}]"))
    return sdfg, mm


def reduce_sdfg(name: str, view_shape, view_strides, viewed: str, axes) -> tuple[dace.SDFG, Reduce]:
    sdfg = dace.SDFG(name)
    sdfg.add_array("P", [L, M, K], dace.float64)
    out_shape = [1] if axes is None or len(axes) == len(view_shape) else [M]
    sdfg.add_array("out", out_shape, dace.float64)
    sdfg.add_view("pv", view_shape, dace.float64, strides=view_strides)
    state = sdfg.add_state()
    red = Reduce("red", "lambda a, b: a + b", axes=axes, identity=0)
    state.add_node(red)
    pv = state.add_access("pv")
    state.add_edge(state.add_read("P"), None, pv, "views", dace.Memlet(viewed))
    state.add_edge(pv, None, red, "_in", dace.Memlet.from_array("pv", sdfg.arrays["pv"]))
    state.add_edge(red, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg, red


def test_matmul_writes_a_plane_without_a_view():
    """A MatMul output View of ``X[1]`` goes; ``_c`` writes the 3-D subset, which MatMul squeezes."""
    sdfg, mm = matmul_into_plane_sdfg("rv_libop_matmul_plane")
    RemoveViews().apply_pass(sdfg, {})
    assert not views(sdfg)
    state = sdfg.start_state
    (out,) = state.out_edges(mm)
    assert out.data.data == "X" and str(out.data.subset) == f"1, 0:{M}, 0:{N}"
    rng = np.random.default_rng(3)
    A, B, X = rng.random((M, K)), rng.random((K, N)), rng.random((L, M, N))
    expected = X.copy()
    expected[1] = A @ B
    sdfg(A=A, B=B, X=X)
    assert np.allclose(X, expected)


def test_reduce_of_a_strided_column_maps_its_axes_through_the_subset():
    """A full Reduce of ``P[1, :, 2]`` reads the subset; every axis reduced stays a full reduction (``axes=None``)."""
    sdfg, red = reduce_sdfg("rv_libop_reduce_column", [M], [K], f"P[1, 0:{M}, 2]", [0])
    RemoveViews().apply_pass(sdfg, {})
    assert not views(sdfg)
    assert red.axes is None
    P, out = np.random.default_rng(4).random((L, M, K)), np.zeros(1)
    sdfg(P=P, out=out)
    assert np.allclose(out[0], P[1, :, 2].sum())


def test_partial_reduce_of_a_plane_renumbers_its_axis():
    """Reducing axis 1 of the View of ``P[1]`` reduces axis 2 of the subset ``P[1, 0:M, 0:K]``."""
    sdfg, red = reduce_sdfg("rv_libop_reduce_plane", [M, K], [K, 1], f"P[1, 0:{M}, 0:{K}]", [1])
    RemoveViews().apply_pass(sdfg, {})
    assert not views(sdfg)
    assert list(red.axes) == [2]
    P, out = np.random.default_rng(5).random((L, M, K)), np.zeros(M)
    sdfg(P=P, out=out)
    assert np.allclose(out, P[1].sum(axis=1))


def test_tensordot_keeps_a_view_that_drops_a_dimension():
    """TensorDot numbers every dimension of its operand: the View of ``B[0]`` drops one, so it stays."""
    sdfg = dace.SDFG("rv_libop_tensordot_refusal")
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [L, K, N], dace.float64)
    sdfg.add_array("C", [M, N], dace.float64)
    sdfg.add_view("bv", [K, N], dace.float64, strides=[N, 1])
    state = sdfg.add_state()
    td = TensorDot("td", left_axes=[1], right_axes=[0])
    state.add_node(td)
    bv = state.add_access("bv")
    state.add_edge(state.add_read("B"), None, bv, "views", dace.Memlet(f"B[0, 0:{K}, 0:{N}]"))
    state.add_edge(state.add_read("A"), None, td, "_left_tensor", dace.Memlet(f"A[0:{M}, 0:{K}]"))
    state.add_edge(bv, None, td, "_right_tensor", dace.Memlet(f"bv[0:{K}, 0:{N}]"))
    state.add_edge(td, "_out_tensor", state.add_write("C"), None, dace.Memlet(f"C[0:{M}, 0:{N}]"))
    RemoveViews().apply_pass(sdfg, {})
    assert views(sdfg) == ["bv"]


def test_matmul_keeps_a_view_with_a_unit_row():
    """A 1 x K View of ``A[2]`` is a matrix; MatMul would squeeze the subset into a vector, so the View stays."""
    sdfg = dace.SDFG("rv_libop_matmul_unit_row")
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [K, N], dace.float64)
    sdfg.add_array("C", [1, N], dace.float64)
    sdfg.add_view("av", [1, K], dace.float64, strides=[K, 1])
    state = sdfg.add_state()
    mm = MatMul("mm")
    state.add_node(mm)
    av = state.add_access("av")
    state.add_edge(state.add_read("A"), None, av, "views", dace.Memlet(f"A[2, 0:{K}]"))
    state.add_edge(av, None, mm, "_a", dace.Memlet(f"av[0:1, 0:{K}]"))
    state.add_edge(state.add_read("B"), None, mm, "_b", dace.Memlet(f"B[0:{K}, 0:{N}]"))
    state.add_edge(mm, "_c", state.add_write("C"), None, dace.Memlet(f"C[0:1, 0:{N}]"))
    RemoveViews().apply_pass(sdfg, {})
    assert views(sdfg) == ["av"]


@dace.program
def matmul_into_plane(A: dace.float64[M, K], B: dace.float64[K, N], X: dace.float64[L, M, N]):
    X[1, :, :] = A @ B


@dace.program
def sum_of_strided_column(P: dace.float64[L, M, 2], out: dace.float64[1]):
    out[0] = np.sum(P[1, :, 1])


def run_canonical(prog, target: str = "cpu") -> None:
    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target=target)
    assert not views(sdfg)
    if target == "gpu":
        from dace.transformation.passes.canonicalize.finalize import offload_to_gpu

        offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target, validate=True)
    rng = np.random.default_rng(6)
    args = {name: rng.random(desc.shape) for name, desc in sdfg.arglist().items()}
    expected = {k: v.copy() for k, v in args.items()}
    prog.f(**expected)
    if target == "gpu":
        import cupy

        given = {k: cupy.asarray(v) for k, v in args.items()}
        sdfg(**given)
        args = {k: cupy.asnumpy(v) for k, v in given.items()}
    else:
        sdfg(**args)
    for name in args:
        assert np.allclose(args[name], expected[name]), name


@pytest.mark.parametrize("prog", [matmul_into_plane, sum_of_strided_column], ids=lambda p: p.name)
def test_the_canonical_form_has_no_view_next_to_the_node(prog):
    """quatrex's ``xl_d[0] = a @ b`` and vexx_k's sum over ``egrp_pairs[eg, :, 1]``: no View after canonicalization."""
    run_canonical(prog)


@pytest.mark.gpu
def test_the_gpu_gemm_writes_a_plane():
    """cuBLAS / hipBLAS GEMM writes the squeezed 3-D output subset with its leading dimension."""
    run_canonical(matmul_into_plane, "gpu")


if __name__ == "__main__":
    test_matmul_writes_a_plane_without_a_view()
    test_reduce_of_a_strided_column_maps_its_axes_through_the_subset()
    test_partial_reduce_of_a_plane_renumbers_its_axis()
    test_tensordot_keeps_a_view_that_drops_a_dimension()
    test_matmul_keeps_a_view_with_a_unit_row()
    test_the_canonical_form_has_no_view_next_to_the_node(matmul_into_plane)
    test_the_canonical_form_has_no_view_next_to_the_node(sum_of_strided_column)
