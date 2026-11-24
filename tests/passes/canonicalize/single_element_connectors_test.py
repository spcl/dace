# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ConvertLengthOneArraysToScalars``: a single-element transient a nested SDFG writes, or reads in a guard, is a
scalar on both sides of the connector (icon_gather's ``arr1``, mandelbrot's ``active``); an argument keeps its array."""

import numpy as np
import pytest

import dace
from dace import data
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.linalg.nodes.inv import Inv
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

N = dace.symbol("N")


def gather_inner() -> dace.SDFG:
    """``x[0, 0] = 2 * v``: the helper writes its single-element output through the connector."""
    inner = dace.SDFG("gather_inner")
    inner.add_scalar("v", dace.float64)
    inner.add_array("x", [1, 1], dace.float64)
    state = inner.add_state("body")
    tasklet = state.add_tasklet("t", {"i"}, {"o"}, "o = 2.0 * i")
    state.add_edge(state.add_read("v"), None, tasklet, "i", dace.Memlet("v"))
    state.add_edge(tasklet, "o", state.add_write("x"), None, dace.Memlet("x[0, 0]"))
    return inner


def guarded_inner() -> dace.SDFG:
    """``y = x + 1`` when ``x[0, 0] > 0.5``, else ``y = -1``: the helper reads its input in a guard."""
    inner = dace.SDFG("guarded_inner")
    inner.add_array("x", [1, 1, 1], dace.float64)
    inner.add_scalar("y", dace.float64)
    pick = ConditionalBlock("pick")
    inner.add_node(pick, is_start_block=True)
    for condition, code in (("x[0, 0, 0] > 0.5", "o = v + 1.0"), ("not (x[0, 0, 0] > 0.5)", "o = -1.0 + 0.0 * v")):
        arm = ControlFlowRegion(f"arm_{len(pick.branches)}", sdfg=inner)
        state = arm.add_state("body", is_start_block=True)
        tasklet = state.add_tasklet("t", {"v"}, {"o"}, code)
        state.add_edge(state.add_read("x"), None, tasklet, "v", dace.Memlet("x[0, 0, 0]"))
        state.add_edge(tasklet, "o", state.add_write("y"), None, dace.Memlet("y"))
        pick.add_branch(CodeBlock(condition), arm)
    return inner


def written_by_a_helper_sdfg() -> dace.SDFG:
    """Per ``i``: the helper writes ``t[0, 0] = 2 * a[i]``, then ``b[i] = t[0, 0] + 1``."""
    sdfg = dace.SDFG("single_element_written_by_a_helper")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_transient("t", [1, 1], dace.float64, lifetime=dace.dtypes.AllocationLifetime.Scope)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    sdfg.add_scalar("ai", dace.float64, transient=True)
    nest = state.add_nested_sdfg(gather_inner(), {"v"}, {"x"})
    t, ai = state.add_access("t"), state.add_access("ai")
    load = state.add_tasklet("load", {"v"}, {"o"}, "o = v")
    add = state.add_tasklet("add", {"w"}, {"o"}, "o = w + 1.0")
    state.add_memlet_path(state.add_read("a"), entry, load, dst_conn="v", memlet=dace.Memlet("a[i]"))
    state.add_edge(load, "o", ai, None, dace.Memlet("ai"))
    state.add_edge(ai, None, nest, "v", dace.Memlet("ai"))
    state.add_edge(nest, "x", t, None, dace.Memlet("t[0, 0]"))
    state.add_edge(t, None, add, "w", dace.Memlet("t[0, 0]"))
    state.add_memlet_path(add, exit_, state.add_write("b"), src_conn="o", memlet=dace.Memlet("b[i]"))
    sdfg.validate()
    return sdfg


def read_in_a_guard_sdfg(transient: bool) -> dace.SDFG:
    """Per ``i``: ``t[0, 0, 0] = 2 * a[i]`` (an argument ``t`` with ``transient=False``), then the guarded helper maps
    ``t`` to ``b[i]``."""
    sdfg = dace.SDFG(f"single_element_read_in_a_guard_{transient}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("t", [1, 1, 1], dace.float64, transient=transient)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", {"i": "0:N"}, schedule=dace.ScheduleType.Sequential)
    double = state.add_tasklet("double", {"v"}, {"w"}, "w = 2.0 * v")
    t = state.add_access("t")
    nest = state.add_nested_sdfg(guarded_inner(), {"x"}, {"y"})
    sdfg.add_scalar("yo", dace.float64, transient=True)
    yo = state.add_access("yo")
    store = state.add_tasklet("store", {"v"}, {"o"}, "o = v")
    state.add_memlet_path(state.add_read("a"), entry, double, dst_conn="v", memlet=dace.Memlet("a[i]"))
    state.add_edge(double, "w", t, None, dace.Memlet("t[0, 0, 0]"))
    state.add_edge(t, None, nest, "x", dace.Memlet("t[0:1, 0:1, 0:1]"))
    state.add_edge(nest, "y", yo, None, dace.Memlet("yo"))
    state.add_edge(yo, None, store, "v", dace.Memlet("yo"))
    state.add_memlet_path(store, exit_, state.add_write("b"), src_conn="o", memlet=dace.Memlet("b[i]"))
    sdfg.validate()
    return sdfg


def nested(sdfg: dace.SDFG) -> dace.SDFG:
    return next(n.sdfg for n in sdfg.start_block.nodes() if isinstance(n, dace.nodes.NestedSDFG))


def test_a_single_element_matrix_a_helper_writes_is_a_scalar_on_both_sides():
    sdfg = written_by_a_helper_sdfg()
    assert ConvertLengthOneArraysToScalars(single_element=True).apply_pass(sdfg, {}) >= {"t"}
    sdfg.validate()
    assert isinstance(sdfg.arrays["t"], data.Scalar)
    assert isinstance(nested(sdfg).arrays["x"], data.Scalar)
    a = np.array([0.5, -1.0, 2.0, 0.0])
    b = np.zeros(4)
    sdfg(a=a, b=b, N=4)
    assert np.allclose(b, [2.0, -1.0, 5.0, 1.0])


def test_a_single_element_cube_a_helper_reads_in_a_guard_is_a_scalar_on_both_sides():
    sdfg = read_in_a_guard_sdfg(transient=True)
    assert ConvertLengthOneArraysToScalars(single_element=True).apply_pass(sdfg, {}) >= {"t"}
    sdfg.validate()
    assert isinstance(sdfg.arrays["t"], data.Scalar)
    assert isinstance(nested(sdfg).arrays["x"], data.Scalar)
    a = np.array([0.1, 0.2, 0.3, 0.9, 0.0, 0.26])
    b = np.zeros(6)
    sdfg(a=a, b=b, N=6)
    assert np.allclose(b, [-1.0, -1.0, 1.6, 2.8, -1.0, 1.52])


def test_a_single_element_argument_keeps_its_array_on_both_sides():
    sdfg = read_in_a_guard_sdfg(transient=False)
    ConvertLengthOneArraysToScalars(single_element=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert type(sdfg.arrays["t"]) is data.Array and tuple(sdfg.arrays["t"].shape) == (1, 1, 1)
    assert type(nested(sdfg).arrays["x"]) is data.Array
    a = np.array([0.3, 0.1])
    b, t = np.zeros(2), np.zeros((1, 1, 1))
    sdfg(a=a, b=b, t=t, N=2)
    assert np.allclose(b, [1.6, -1.0]) and np.allclose(t, 0.2)


def library_results_sdfg() -> dace.SDFG:
    """``s = sum(a)`` through a Reduce, ``d = a . b`` through a Dot, each into a ``(1,)`` transient; ``out = s + d``."""
    sdfg = dace.SDFG("single_element_library_results")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("s", [1], dace.float64)
    sdfg.add_transient("d", [1, 1], dace.float64)
    state = sdfg.add_state("main")
    a, s, d = state.add_read("a"), state.add_access("s"), state.add_access("d")
    reduce = state.add_reduce("lambda x, y: x + y", None, identity=0.0)
    state.add_edge(a, None, reduce, None, dace.Memlet("a[0:N]"))
    state.add_edge(reduce, None, s, None, dace.Memlet("s[0]"))
    dot = Dot("dot")
    dot.implementation = "pure"
    state.add_node(dot)
    state.add_edge(a, None, dot, "_x", dace.Memlet("a[0:N]"))
    state.add_edge(state.add_read("b"), None, dot, "_y", dace.Memlet("b[0:N]"))
    state.add_edge(dot, "_result", d, None, dace.Memlet("d[0, 0]"))
    add = state.add_tasklet("add", {"x", "y"}, {"o"}, "o = x + y")
    state.add_edge(s, None, add, "x", dace.Memlet("s[0]"))
    state.add_edge(d, None, add, "y", dace.Memlet("d[0, 0]"))
    state.add_edge(add, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def test_single_element_results_of_a_reduce_and_a_dot_are_scalars():
    sdfg = library_results_sdfg()
    assert ConvertLengthOneArraysToScalars(single_element=True).apply_pass(sdfg, {}) >= {"s", "d"}
    sdfg.validate()
    assert isinstance(sdfg.arrays["s"], data.Scalar) and isinstance(sdfg.arrays["d"], data.Scalar)
    a = np.array([1.0, 2.0, -0.5, 4.0])
    b = np.array([0.5, -1.0, 2.0, 0.25])
    out = np.zeros(1)
    sdfg(a=a, b=b, out=out, N=4)
    assert np.allclose(out, [5.0])


@dace.program
def invert(a: dace.float64[4, 4], out: dace.float64[4, 4]):
    out[:] = np.linalg.inv(a)


@pytest.mark.lapack
def test_a_host_lapack_status_is_a_scalar():
    sdfg = invert.to_sdfg()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, Inv):
            node.implementation = "OpenBLAS"
    sdfg.expand_library_nodes()
    statuses = [sd.arrays["_info"] for sd in sdfg.all_sdfgs_recursive() if "_info" in sd.arrays]
    assert statuses and all(isinstance(desc, data.Scalar) for desc in statuses)
    a = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.5], [0.0, 0.0, 0.5, 1.0]])
    out = np.zeros((4, 4))
    sdfg(a=a.copy(), out=out)
    assert np.allclose(out @ a, np.eye(4))


if __name__ == "__main__":
    test_a_single_element_matrix_a_helper_writes_is_a_scalar_on_both_sides()
    test_a_single_element_cube_a_helper_reads_in_a_guard_is_a_scalar_on_both_sides()
    test_a_single_element_argument_keeps_its_array_on_both_sides()
    test_single_element_results_of_a_reduce_and_a_dot_are_scalars()
    test_a_host_lapack_status_is_a_scalar()
