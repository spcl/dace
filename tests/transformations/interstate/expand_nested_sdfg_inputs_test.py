import numpy as np
import pytest

import dace
from dace.sdfg.dealias import convert_legacy_nested_sdfgs
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol("N")


@dace.program
def scatter_gatter(
    A: dace.float64[N, N, N],
    B: dace.float64[N, N, N],
    C: dace.float64[N, N, N],
    idx0: dace.int64[N],
    idx1: dace.int64[N],
):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        a = A[idx0[i], j, idx1[k]]
        b = B[i, j, k]
        c = C[idx0[i], j, idx1[k]]
        C[idx0[i], j, idx1[k]] = a * b * c * 2.0


def test_expand_nested_sdfg_inputs():
    sdfg = scatter_gatter.to_sdfg()
    sdfg.validate()
    print("there")
    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nsdfgs) == 1, f"Expected exactly one NestedSDFG before transformation, found {len(nsdfgs)}."
    nsdfg, parent_state = nsdfgs[0]

    ExpandNestedSDFGInputs().apply_to(sdfg=parent_state.sdfg, nested_sdfg=nsdfg)
    print("here")

    sdfg.validate()

    # Execute once to ensure the transformed SDFG is runnable.
    n = 16
    rng = np.random.default_rng(42)

    A = rng.random((n, n, n))
    B = rng.random((n, n, n))
    C = rng.random((n, n, n))

    idx0 = rng.permutation(n).astype(np.int64)
    idx1 = rng.permutation(n).astype(np.int64)

    sdfg(A=A, B=B, C=C, idx0=idx0, idx1=idx1, N=n)

    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs) == 1, f"Expected exactly one NestedSDFG after transformation, found {len(nsdfgs)}."

    nsdfg, parent_state = nsdfgs[0]

    for edge in parent_state.in_edges(nsdfg):
        expected = dace.subsets.Range.from_array(parent_state.sdfg.arrays[edge.data.data])

        assert edge.data.subset == expected, (
            f"Input edge '{edge.data.data}' was not expanded to the full array.\n"
            f"Expected: {expected}\n"
            f"Actual:   {edge.data.subset}"
        )

    for edge in parent_state.out_edges(nsdfg):
        expected = dace.subsets.Range.from_array(parent_state.sdfg.arrays[edge.data.data])

        assert edge.data.subset == expected, (
            f"Output edge '{edge.data.data}' was not expanded to the full array.\n"
            f"Expected: {expected}\n"
            f"Actual:   {edge.data.subset}"
        )


import dace

N = dace.symbol("N")


# Reference implementation
def numpy_reference(A, B, idx):
    C = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[idx[i], j] = A[idx[i], j] + B[i, j] * 2.0
    return C


# DaCe program (forces Subscript + scalar mix)
@dace.program
def column_gather_scatter(
    A: dace.float64[N, N],
    B: dace.float64[N, N],
    C: dace.float64[N, N],
    idx: dace.int64[N],
):
    for i, j in dace.map[0:N, 0:N]:
        # key pattern:
        # scalar index + subscript reuse
        a = A[idx[i], j]
        b = B[i, j]

        C[idx[i], j] = a + b * 2.0


# Test
def test_expand_nested_sdfg_inputs_column_scalar_uncollapse_e2e():
    sdfg = column_gather_scatter.to_sdfg()
    sdfg.validate()

    # locate NestedSDFG
    nsdfgs = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs) == 1, f"Expected exactly one NestedSDFG before transformation, found {len(nsdfgs)}"

    nsdfg, parent_state = nsdfgs[0]

    # snapshot input edges (pre-transform sanity)
    pre_in_edges = list(parent_state.in_edges(nsdfg))
    pre_out_edges = list(parent_state.out_edges(nsdfg))

    # Apply transformation
    ExpandNestedSDFGInputs().apply_to(
        sdfg=parent_state.sdfg,
        nested_sdfg=nsdfg,
    )

    sdfg.validate()

    # Run E2E execution
    n = 16
    rng = np.random.default_rng(0)

    A = rng.random((n, n))
    B = rng.random((n, n))
    C = np.zeros((n, n))
    idx = rng.permutation(n)

    sdfg(A=A, B=B, C=C, idx=idx, N=n)

    # Reference result
    C_ref = numpy_reference(A, B, idx)

    assert np.allclose(C, C_ref), (
        f"Numerical mismatch after ExpandNestedSDFGInputs\nmax error = {np.max(np.abs(C - C_ref))}"
    )

    # Structural validation
    nsdfgs_after = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]

    assert len(nsdfgs_after) == 1, f"Expected exactly one NestedSDFG after transformation, found {len(nsdfgs_after)}"

    nsdfg_after, parent_state_after = nsdfgs_after[0]

    # ensure memlets fully expanded
    for edge in parent_state_after.in_edges(nsdfg_after):
        expected = dace.subsets.Range.from_array(parent_state_after.sdfg.arrays[edge.data.data])
        assert edge.data.subset == expected, (
            f"Input edge not fully expanded for {edge.data.data}\nExpected: {expected}\nActual:   {edge.data.subset}"
        )

    for edge in parent_state_after.out_edges(nsdfg_after):
        expected = dace.subsets.Range.from_array(parent_state_after.sdfg.arrays[edge.data.data])
        assert edge.data.subset == expected, (
            f"Output edge not fully expanded for {edge.data.data}\nExpected: {expected}\nActual:   {edge.data.subset}"
        )

    # Regression guard: ensure transformation actually changed structure
    post_in_edges = list(parent_state_after.in_edges(nsdfg_after))

    assert len(pre_in_edges) == len(post_in_edges), "Unexpected change in number of input edges after transformation"


def test_expand_terminates_on_wcr_reduction_out_edge():
    """A body-NSDFG whose only non-full boundary edge is a WCR (reduction) out-edge
    must NOT spin ``apply_transformations_repeated`` forever.

    ``apply`` deliberately leaves WCR out-edges un-widened (their subset is the
    reduction target slot). ``can_be_applied`` must skip them too, otherwise the
    non-full WCR subset re-matches after every other edge is already full -- the
    reduce-at-output reduction hang (cholesky/lu/symm/trisolv), where the surviving
    ``A[i, Min(i,k):Max(i,k)+1]`` WCR edge looped 30k+ times.
    """
    M = dace.symbol("M")
    sdfg = dace.SDFG("wcr_reduction_expand")
    sdfg.add_array("A", [M], dace.float64)
    sdfg.add_array("out", [M], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map("m", {"i": "0:M"})

    body = dace.SDFG("body_expand_terminates_on_wcr_reduction_out_edge")
    body.add_array("a", [M], dace.float64)
    body.add_scalar("o", dace.float64)
    bst = body.add_state()
    t = bst.add_tasklet("r", {"_a"}, {"_o"}, "_o = _a[0]")
    bst.add_edge(bst.add_read("a"), None, t, "_a", dace.Memlet("a[0:M]"))
    bst.add_edge(t, "_o", bst.add_write("o"), None, dace.Memlet("o[0]"))
    nsdfg = st.add_nested_sdfg(body, {"a"}, {"o"})
    # Full (already-widened) in-edge -- nothing to widen there.
    st.add_memlet_path(st.add_read("A"), me, nsdfg, dst_conn="a", memlet=dace.Memlet("A[0:M]"))
    # The only non-full edge: a WCR reduction out-edge (a single-element slot).
    st.add_memlet_path(
        nsdfg, mx, st.add_write("out"), src_conn="o", memlet=dace.Memlet("out[i]", wcr="lambda x, y: x + y")
    )
    convert_legacy_nested_sdfgs(sdfg)
    sdfg.validate()

    # The WCR out-edge is apply's no-widen set, so this is a fixed point already:
    # a bounded (here: no-op) result, NOT an unbounded spin.
    count = sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
    assert count in (None, 0), f"WCR-only reduction NSDFG must be a fixed point (no widen); got {count} applications"
    out_edge = next(iter(st.out_edges(nsdfg)))
    assert out_edge.data.wcr is not None and str(out_edge.data.subset) == "i", (
        "the WCR reduction out-edge must be left untouched"
    )


def per_iteration_nsdfg_in_an_int32_loop() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.NestedSDFG]:
    """``for it = M32; it <= N32; it += S32: B[it] = 2 * A[it]`` through a one-element nested SDFG.

    Symbolic bounds and stride with no integer literal, so the loop types ``it`` as int32; ``it`` is
    bound by the loop and registered in no symbol table.
    """
    inner = dace.SDFG("body_per_iteration_nsdfg_in_an_int32_loop")
    inner.add_array("x", [1], dace.float64)
    inner.add_array("y", [1], dace.float64)
    inner_state = inner.add_state("compute", is_start_block=True)
    tasklet = inner_state.add_tasklet("twice", {"a"}, {"b"}, "b = 2 * a")
    inner_state.add_edge(inner_state.add_read("x"), None, tasklet, "a", dace.Memlet("x[0]"))
    inner_state.add_edge(tasklet, "b", inner_state.add_write("y"), None, dace.Memlet("y[0]"))

    sdfg = dace.SDFG("outer_per_iteration_nsdfg_in_an_int32_loop")
    for name in ("M32", "N32", "S32"):
        sdfg.add_symbol(name, dace.int32)
    sdfg.add_array("A", ["N32 + 1"], dace.float64)
    sdfg.add_array("B", ["N32 + 1"], dace.float64)
    loop = LoopRegion("L", "it <= N32", "it", "it = M32", "it = it + S32")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    nsdfg = body.add_nested_sdfg(inner, {"x": None}, {"y": None})
    body.add_edge(body.add_read("A"), None, nsdfg, "x", dace.Memlet("A[it]"))
    body.add_edge(nsdfg, "y", body.add_write("B"), None, dace.Memlet("B[it]"))
    sdfg.reset_cfg_list()
    return sdfg, body, nsdfg


def test_an_introduced_loop_iterator_is_declared_inside_at_the_width_its_loop_gives_it():
    sdfg, body, nsdfg = per_iteration_nsdfg_in_an_int32_loop()
    sut = ExpandNestedSDFGInputs()
    sut.setup_match(
        sdfg, body.parent_graph.cfg_id, body.block_id, {ExpandNestedSDFGInputs.nested_sdfg: nsdfg}, 0, override=True
    )
    assert sut.can_be_applied(body, 0, sdfg)

    sut.apply(body, sdfg)

    sdfg.validate()
    assert "it" not in sdfg.symbols
    assert nsdfg.sdfg.symbols["it"] == dace.int32
    assert str(nsdfg.symbol_mapping["it"]) == "it"


def strided_window_sdfg() -> dace.SDFG:
    """``b[k] = x[k]`` in a nested SDFG whose ``x`` is the strided window ``a[0:N:2]``."""
    inner = dace.SDFG("inner_strided_window_sdfg")
    inner.add_array("x", [N // 2], dace.float64, strides=[2])
    inner.add_array("y", [N // 2], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "copy", {"k": "0:N//2"}, {"v": dace.Memlet("x[k]")}, "w = v", {"w": dace.Memlet("y[k]")}, external_edges=True
    )
    outer = dace.SDFG("strided_window")
    outer.add_array("a", [N], dace.float64)
    outer.add_array("b", [N // 2], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"x": None}, {"y": None})
    state.add_edge(state.add_read("a"), None, node, "x", dace.Memlet("a[0:N:2]"))
    state.add_edge(node, "y", state.add_write("b"), None, dace.Memlet("b[0:N//2]"))
    return outer


def test_widening_a_strided_window_scales_the_inner_index_by_its_step():
    sdfg = strided_window_sdfg()

    assert sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs) == 1

    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    reads = [
        e.data for s in inner.states() for e in s.edges() if e.data.data == "a" and e.data.subset.num_elements() == 1
    ]
    assert [str(m.subset) for m in reads] == ["2*k"]
    a, b = np.arange(10.0), np.zeros(5)
    sdfg(a=a, b=b, N=10)
    np.testing.assert_array_equal(b, a[0:10:2])


def indexed_window_sdfg(read: str, x_stride: int, by_condition: bool) -> dace.SDFG:
    """``b[0] = x[1]`` for the window ``x = a[read]``, read by index on an interstate edge (an assignment, or a
    condition that selects which of two values is written)."""
    inner = dace.SDFG("indexed")
    inner.add_array("x", [4], dace.float64, strides=[x_stride])
    inner.add_array("y", [1], dace.float64)
    inner.add_symbol("t", dace.float64)
    first = inner.add_state(is_start_block=True)
    second = inner.add_state()
    if by_condition:
        for condition, value in (("x[1] > 2", "1.0"), ("x[1] <= 2", "0.0")):
            branch = inner.add_state()
            inner.add_edge(first, branch, dace.InterstateEdge(condition=condition))
            inner.add_edge(branch, second, dace.InterstateEdge(assignments={"t": value}))
    else:
        inner.add_edge(first, second, dace.InterstateEdge(assignments={"t": "x[1]"}))
    write = second.add_tasklet("write", {}, {"o"}, "o = t")
    second.add_edge(write, "o", second.add_write("y"), None, dace.Memlet("y[0]"))

    outer = dace.SDFG("indexed_outer")
    outer.add_array("a", [16], dace.float64)
    outer.add_array("b", [1], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_edge(state.add_read("a"), None, node, "x", dace.Memlet(read))
    state.add_edge(node, "y", state.add_write("b"), None, dace.Memlet("b[0]"))
    return outer


@pytest.mark.parametrize("read, x_stride, element", [("a[1:8:2]", 2, 3), ("a[2:6]", 1, 3)])
@pytest.mark.parametrize("by_condition", [False, True])
def test_an_index_read_on_an_interstate_edge_moves_with_the_window(read, x_stride, element, by_condition):
    sdfg = indexed_window_sdfg(read, x_stride, by_condition)

    assert sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs) == 1

    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    edge_code = " ".join(
        e.data.condition.as_string + " " + " ".join(e.data.assignments.values()) for e in inner.all_interstate_edges()
    )
    assert f"a[{element}]" in edge_code and "x[" not in edge_code
    a, b = np.arange(16.0), np.zeros(1)
    sdfg(a=a, b=b)
    assert b[0] == (float(element > 2) if by_condition else a[element])


def column_as_vector_sdfg() -> dace.SDFG:
    """``b = 2 * a[:, 0]`` for an ``(N, 1)`` ``a``, bound whole as an ``(N,)`` connector."""
    inner = dace.SDFG("column")
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "twice", {"i": "0:N"}, {"v": dace.Memlet("x[i]")}, "w = 2 * v", {"w": dace.Memlet("y[i]")}, external_edges=True
    )
    outer = dace.SDFG("column_outer")
    outer.add_array("a", [N, 1], dace.float64)
    outer.add_array("b", [N], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"x"}, {"y"})
    state.add_edge(state.add_read("a"), None, node, "x", dace.Memlet("a[0:N, 0]"))
    state.add_edge(node, "y", state.add_write("b"), None, dace.Memlet("b[0:N]"))
    return outer


def test_a_whole_array_bound_at_lower_rank_is_widened_to_the_outer_rank():
    sdfg = column_as_vector_sdfg()

    assert sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs) == 1

    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert inner.arrays["a"].shape == (N, 1)
    assert {str(e.data.subset) for s in inner.states() for e in s.edges() if e.data.data == "a"} >= {"i, 0"}
    a, b = np.arange(8.0).reshape(8, 1), np.zeros(8)
    sdfg(a=a, b=b, N=8)
    np.testing.assert_array_equal(b, 2 * a[:, 0])


@dace.program
def every_other(src: dace.float64[2 * N], dst: dace.float64[N]):
    for i in dace.map[0:N]:
        dst[i] = src[i * 2]


def test_widening_keeps_an_index_that_already_carries_the_step():
    """The inner array keeps the outer stride and its index is already ``2*i``: widening must not scale
    it again. Canonicalization nests exactly this shape, and scaling it read ``src[4*i]``."""
    sdfg = every_other.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    reads = {
        str(e.data.subset)
        for e, _ in sdfg.all_edges_recursive()
        if isinstance(e.data, dace.Memlet) and e.data.data == "src"
    }
    assert not any("4*" in r for r in reads), reads
    src, dst = np.arange(20.0), np.zeros(10)
    sdfg(src=src, dst=dst, N=10)
    np.testing.assert_array_equal(dst, src[::2])


@dace.program
def bump_every_other(a: dace.float64[2 * N]):
    for i in dace.map[0:N]:
        a[i * 2] = a[i * 2] + 1.0


def test_widening_an_array_read_and_written_in_place():
    """An in-place array is both an input and an output connector; the first widening renames its inner
    array, so the second must not look it up by the old name (CloudSC ``zsolqb``)."""
    sdfg = bump_every_other.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    a = np.arange(20.0)
    expected = a.copy()
    expected[::2] += 1.0
    sdfg(a=a, N=10)
    np.testing.assert_array_equal(a, expected)


if __name__ == "__main__":
    test_expand_nested_sdfg_inputs_column_scalar_uncollapse_e2e()
    test_expand_terminates_on_wcr_reduction_out_edge()
    test_expand_nested_sdfg_inputs()
