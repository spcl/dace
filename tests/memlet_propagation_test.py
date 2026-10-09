# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_subset
from dace.sdfg.state import SymbolResolver


def test_conditional():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    outp = np.zeros((10,))
    conditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_conditional_nested():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    @dace.program
    def nconditional(in1, out):
        conditional(in1, out)

    inp = np.random.rand(10)
    outp = np.zeros((10,))
    nconditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_runtime_conditional():

    @dace.program
    def rconditional(in1, out, mask):
        for i in dace.map[0:10]:
            if mask[i] > 0:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    mask = np.ones((10,))
    mask[0] = 0
    outp = np.zeros((10,))
    rconditional(inp, outp, mask)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_nsdfg_memlet_propagation_with_one_sparse_dimension():
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def sparse(A: dace.float32[M, N], ind: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, ind[i, j]] += 1

    sdfg = sparse.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    # Verify all memlet subsets and volumes in the main state of the program, i.e. around the NSDFG.
    map_state = sdfg.states()[1]
    i = dace.symbol("i")
    j = dace.symbol("j")

    outer_in = map_state.edges()[0].data
    if outer_in.volume != M * N:
        raise RuntimeError("Expected a volume of M*N on the outer input memlet")
    if outer_in.subset[0] != (0, M - 1, 1) or outer_in.subset[1] != (0, N - 1, 1):
        raise RuntimeError("Expected subset of outer in memlet to be [0:M, 0:N], found " + str(outer_in.subset))

    inner_in = map_state.edges()[1].data
    if inner_in.volume != 1:
        raise RuntimeError("Expected a volume of 1 on the inner input memlet")
    if inner_in.subset[0] != (i, i, 1) or inner_in.subset[1] != (j, j, 1):
        raise RuntimeError("Expected subset of inner in memlet to be [i, j], found " + str(inner_in.subset))

    inner_out = map_state.edges()[2].data
    if inner_out.volume != 1:
        raise RuntimeError("Expected a volume of 1 on the inner output memlet")
    if inner_out.subset[0] != (i, i, 1) or inner_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError("Expected subset of inner out memlet to be [i, 0:N], found " + str(inner_out.subset))

    outer_out = map_state.edges()[3].data
    if outer_out.volume != M * N:
        raise RuntimeError("Expected a volume of M*N on the outer output memlet")
    if outer_out.subset[0] != (0, M - 1, 1) or outer_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError("Expected subset of outer out memlet to be [0:M, 0:N], found " + str(outer_out.subset))


def test_nested_conditional_in_loop_in_map():
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def nested_conditional_in_loop_in_map(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(2, N, 1):
                if A[0][0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]

    sdfg = nested_conditional_in_loop_in_map.to_sdfg(simplify=True)
    dace.propagate_memlets_sdfg(sdfg)

    # Verify that the memlet propagation works correctly
    i = dace.symbol("i")
    state = sdfg.source_nodes()[0]
    rnode = state.source_nodes()[0]
    # Input memlets for A should be [0:M, 2:N] (immediately outside of nested SDFG should be [0:i+1, 0:N])
    out_edges = state.out_edges(rnode)
    assert len(out_edges) == 1
    assert out_edges[0].data.subset.ranges == [(0, M - 1, 1), (0, N - 1, 1)]
    nsdfg_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert state.in_edges(nsdfg_node)[0].data.subset.ranges == [(0, i, 1), (0, N - 1, 1)]
    # Output memlets for A should be [0:M, 2:N] (immediately outside of nested SDFG should be [i, 2:N])
    wnode = state.sink_nodes()[0]
    in_edges = state.in_edges(wnode)
    assert len(in_edges) == 1
    assert in_edges[0].data.subset.ranges == [(0, M - 1, 1), (2, N - 1, 1)]
    assert state.out_edges(nsdfg_node)[0].data.subset.ranges == [(i, i, 1), (2, N - 1, 1)]

    N = 20
    M = 20
    a_test = np.zeros((M, N), dtype=np.float64)
    sdfg(a_test, M=M, N=N)
    a_valid = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(2, N, 1):
            a_valid[i, j] = 4.0

    assert np.allclose(a_test, a_valid)


def test_strided_write_keeps_the_multiplier():
    """``C[2 * i]`` covers every second element, not the first ``N``.

    A single-element access has ``re - rb + 1 == 1``, which equals the stride of a unit-stride map
    range, and ``2 * i`` at a zero map begin starts where the map range starts. Both halves of the
    ``i:i+stride`` special case in :class:`~dace.sdfg.propagation.AffineSMemlet` therefore hold for
    an access it was never meant to cover, and returning the map range verbatim drops the
    multiplier -- an under-approximated write set, which is unsound.
    """
    N = dace.symbol("N")

    @dace.program
    def strided_write(A: dace.float64[2 * N], C: dace.float64[2 * N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[2 * i]
                c >> C[2 * i]
                c = a

    sdfg = strided_write.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    state = next(s for s in sdfg.states() if any(isinstance(n, dace.sdfg.nodes.MapExit) for n in s.nodes()))
    out_edge = next(
        e
        for e in state.edges()
        if isinstance(e.src, dace.sdfg.nodes.MapExit)
        and isinstance(e.dst, dace.sdfg.nodes.AccessNode)
        and e.dst.data == "C"
    )
    out = out_edge.data

    assert out.subset.ranges == [(0, 2 * N - 2, 2)], out.subset
    assert out.subset.num_elements() == N, out.subset.num_elements()
    # The written elements must be inside the propagated set; the bug put 2*N-2 outside it.
    # The element is written inside the map, so the map runs
    facts = SymbolResolver().facts_at(state, out_edge.src)
    assert out.subset.covers(dace.subsets.Range([(2 * N - 2, 2 * N - 2, 1)]), facts)


def test_typed_parameter_symbol():
    """A memlet may spell a map parameter with a symbol of another dtype (e.g., a typed loop variable).

    Symbols compare by dtype, so without reconciling the two, the patterns see a parameter-independent
    access. Through a range bounded by scope-local symbols (dynamic map inputs), that access was then
    propagated to the map range itself, leaking the scope-local bounds into the outer memlet.
    """
    N = dace.symbol("N")
    i = dace.symbol("i")
    j = dace.symbol("j", dace.int64)
    b, e = dace.symbol("b"), dace.symbol("e")
    arr = dace.data.Array(dace.float64, [N, N])
    memlet = dace.Memlet(data="A", subset=dace.subsets.Range([(i, i, 1), (j, j, 1)]))

    # Scope-local range: only ``i`` and ``N`` are defined outside, so the dimension over ``j`` is overapproximated
    local = propagate_subset(
        [memlet], arr, ["j"], dace.subsets.Range([(b, e - 1, 1)]), dace.symbolic.Facts.none(), defined_variables={i, N}
    )
    assert local.subset == dace.subsets.Range([(i, i, 1), (0, N - 1, 1)]), local.subset

    # Defined range: the typed ``j`` is still the parameter and is propagated exactly
    defined = propagate_subset(
        [memlet], arr, ["j"], dace.subsets.Range([(0, N - 1, 1)]), dace.symbolic.Facts.none(), defined_variables={i, N}
    )
    assert defined.subset == dace.subsets.Range([(i, i, 1), (0, N - 1, 1)]), defined.subset
    assert "j" not in defined.subset.free_symbols


def test_nested_sdfg_connector_in_mapped_symbols():
    """
    A connector written in the nested SDFG's own symbols is the container it is connected to when the symbol mapping
    restates it as that container, and its memlets propagate in the parent's symbols.
    """
    M = dace.symbol("M")
    outer = dace.SDFG("prop_mapped_connector")
    outer.add_array("A", [M + 1], dace.float64)
    outer.add_array("B", [M + 1], dace.float64)
    state = outer.add_state()
    inner = dace.SDFG("inner")
    inner.add_symbol("N", dace.int64)
    inner.add_array("a", ["N"], dace.float64)
    inner.add_array("b", ["N"], dace.float64)
    inner.add_state().add_mapped_tasklet(
        "cp", {"i": "0:N"}, {"v": dace.Memlet("a[i]")}, "w = v", {"w": dace.Memlet("b[i]")}, external_edges=True
    )
    node = state.add_nested_sdfg(inner, {"a"}, {"b"}, {"N": "M + 1"})
    state.add_edge(state.add_read("A"), None, node, "a", dace.Memlet("A[0:M+1]"))
    state.add_edge(node, "b", state.add_write("B"), None, dace.Memlet("B[0:M+1]"))
    outer.validate()

    propagate_memlets_sdfg(outer)

    for edge in state.all_edges(node):
        assert edge.data.subset == dace.subsets.Range([(0, M, 1)]), edge.data.subset


def test_nested_sdfg_connector_offset():
    """
    A connector keeping an offset of its own (e.g., one-based indices) is the container it is connected to, and its
    memlets propagate in the container's index space.
    """
    outer = dace.SDFG("prop_connector_offset")
    outer.add_array("A", [4], dace.float64)
    outer.add_array("B", [4], dace.float64)
    state = outer.add_state()
    inner = dace.SDFG("inner")
    inner.add_array("a", [4], dace.float64, offset=[-1])
    inner.add_array("b", [4], dace.float64, offset=[-1])
    inner.add_state().add_mapped_tasklet(
        "cp", {"i": "2:4"}, {"v": dace.Memlet("a[i]")}, "w = v", {"w": dace.Memlet("b[i]")}, external_edges=True
    )
    node = state.add_nested_sdfg(inner, {"a"}, {"b"})
    state.add_edge(state.add_read("A"), None, node, "a", dace.Memlet("A[0:4]"))
    state.add_edge(node, "b", state.add_write("B"), None, dace.Memlet("B[0:4]"))
    outer.validate()

    propagate_memlets_sdfg(outer)

    for edge in state.all_edges(node):
        assert edge.data.subset == dace.subsets.Range([(1, 2, 1)]), edge.data.subset


if __name__ == "__main__":
    test_conditional()
    test_conditional_nested()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
    test_nested_conditional_in_loop_in_map()
    test_strided_write_keeps_the_multiplier()
    test_typed_parameter_symbol()
    test_nested_sdfg_connector_in_mapped_symbols()
    test_nested_sdfg_connector_offset()
