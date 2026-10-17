# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests schedule tree input/output memlet computation
"""

import numpy as np
import pytest
import sympy

import dace
from dace import Memlet, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import tree_to_sdfg as t2s
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.propagation import propagate_subset


def _array(*shape):
    return dace.data.Array(dace.float64, list(shape))


def _tasklet(in_memlets, out_memlets):
    """Build a :class:`TaskletNode` from ``{conn: "A[subset]"}`` mappings."""
    node = dace.nodes.Tasklet("t", set(in_memlets), set(out_memlets), "pass")
    return tn.TaskletNode(
        node=node,
        in_memlets={c: Memlet(m) for c, m in in_memlets.items()},
        out_memlets={c: Memlet(m) for c, m in out_memlets.items()},
    )


def _assign(name, expr):
    return tn.AssignNode(name=name, value=CodeBlock(expr), edge=dace.InterstateEdge(assignments={name: expr}))


def _root(child, *, containers, symbols):
    """Wrap a single scope/node as the only child of a fresh root (fixing up parents)."""
    return tn.ScheduleTreeRoot(name="t", children=[child], containers=containers, symbols=symbols)


def _by_data(memlets):
    return {(m.data, str(m.subset), bool(m.dynamic)) for m in memlets}


def _rng(*dims):
    return subsets.Range(
        [
            (
                symbolic.pystr_to_symbolic(str(b)),
                symbolic.pystr_to_symbolic(str(e)),
                symbolic.pystr_to_symbolic(str(s)),
            )
            for b, e, s in dims
        ]
    )



def test_stree_propagation_forloop():
    N = dace.symbol("N")

    @dace.program
    def tester(a: dace.float64[20]):
        for i in range(1, N):
            a[i] = 2
        a[1] = 1

    stree = tester.to_sdfg().as_schedule_tree()
    stree = t2s._insert_state_boundaries_to_tree(stree)

    node_types = list(stree.preorder_traversal())
    assert isinstance(node_types[2], tn.ForScope)
    memlet = dace.Memlet("a[1:N]")
    memlet._is_data_src = False
    assert list(node_types[2].output_memlets()) == [memlet]


@pytest.mark.skip("Suboptimal memlet propagation: https://github.com/spcl/dace/issues/2293")
def test_stree_propagation_symassign():
    # Manually create a schedule tree
    N = dace.symbol("N")
    stree = tn.ScheduleTreeRoot(
        name="tester",
        containers={
            "A": dace.data.Array(dace.float64, [20]),
        },
        symbols={
            "N": N,
        },
        children=[
            tn.MapScope(
                node=dace.nodes.MapEntry(dace.nodes.Map("map", ["i"], dace.subsets.Range([(1, N - 1, 1)]))),
                children=[
                    tn.AssignNode("j", CodeBlock("N + i"), dace.InterstateEdge(assignments=dict(j="N + i"))),
                    tn.TaskletNode(
                        nodes.Tasklet("inner", {}, {"out"}, "out = inp + 2"),
                        {"inp": dace.Memlet("A[j]")},
                        {"out": dace.Memlet("A[j]")},
                    ),
                ],
            ),
        ],
    )
    stree.children[0].parent = stree
    for c in stree.children[0].children:
        c.parent = stree.children[0]

    assert list(stree.children[0].input_memlets()) == [dace.Memlet("A[0:20]", volume=N - 1)]


@pytest.mark.skip("Suboptimal memlet propagation: https://github.com/spcl/dace/issues/2293")
def test_stree_propagation_dynset():
    H = dace.symbol("H")
    nnz = dace.symbol("nnz")
    W = dace.symbol("W")

    @dace.program
    def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W]):
        b = np.zeros([H], dtype=np.float32)

        for i in dace.map[0:H]:
            for j in dace.map[A_row[i] : A_row[i + 1]]:
                b[i] += A_val[j] * x[A_col[j]]

        return b

    sdfg = spmv.to_sdfg()
    stree = sdfg.as_schedule_tree()
    assert len(stree.children) == 2
    assert all(isinstance(c, tn.MapScope) for c in stree.children)
    mapscope = stree.children[1]
    _, _, dynrangemap = mapscope.children
    assert isinstance(dynrangemap, tn.MapScope)

    # Check dynamic range map memlets
    internal_memlets = list(dynrangemap.input_memlets())
    internal_memlet_data = [m.data for m in internal_memlets]
    assert "x" in internal_memlet_data
    assert "A_val" in internal_memlet_data
    assert "A_row" not in internal_memlet_data
    for m in internal_memlets:
        if m.data == "A_val":
            assert m.subset != dace.subsets.Range([(0, nnz - 1, 1)])  # Not propagated

    # Check top-level scope memlets
    external_memlets = list(mapscope.input_memlets())
    assert dace.Memlet("A_row[0:H]") in external_memlets
    assert dace.Memlet("A_row[1:H+1]") in external_memlets
    assert dace.Memlet("x[0:W]", volume=0, dynamic=True) in external_memlets
    assert dace.Memlet("A_val[0:nnz]", volume=0, dynamic=True) in external_memlets
    for m in external_memlets:
        if m.data == "A_val":
            assert m.subset == dace.subsets.Range([(0, nnz - 1, 1)])  # Propagated


def test_stree_ifscope_gathers_children():
    """Fix 1: a scope without a propagation range (an ``if``) still reports its children's memlets."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N)}
    ifs = tn.IfScope(condition=CodeBlock("N > 1"), children=[_tasklet({"a": "A[0]"}, {"c": "C[0]"})])
    _root(ifs, containers=containers, symbols={"N": dace.int64})

    assert _by_data(ifs.input_memlets()) == {("A", "0", False)}
    assert _by_data(ifs.output_memlets()) == {("C", "0", False)}


def test_stree_root_through_if_nonempty():
    """Fix 1: the root propagates memlets of an ``if`` around a map (previously empty)."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N)}
    mapscope = tn.MapScope(
        node=dace.nodes.MapEntry(dace.nodes.Map("m", ["i"], subsets.Range([(0, N - 1, 1)]))),
        children=[_tasklet({"a": "A[i]"}, {"c": "C[i]"})],
    )
    ifs = tn.IfScope(condition=CodeBlock("N > 1"), children=[mapscope])
    root = _root(ifs, containers=containers, symbols={"N": dace.int64})

    assert _by_data(root.input_memlets()) == {("A", "0:N", False)}
    assert _by_data(root.output_memlets()) == {("C", "0:N", False)}


def test_stree_while_widens_loop_locals():
    """Fix 1 + 3: a ``while`` loop with no range widens memlets that use iteration-varying symbols."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N)}
    loop = dace.sdfg.state.LoopRegion("L", condition_expr="i < N")
    scope = tn.WhileScope(
        loop=loop,
        children=[_tasklet({"a": "A[i]"}, {"c": "C[i]"}), _assign("i", "i + 1")],
    )
    _root(scope, containers=containers, symbols={"N": dace.int64, "i": dace.int64})

    # ``i`` changes every iteration (assigned in the body), so reads/writes through it are over-approximated
    # to the full extent, marked dynamic with unknown volume.
    assert _by_data(scope.input_memlets()) == {("A", "0:N", True)}
    assert _by_data(scope.output_memlets()) == {("C", "0:N", True)}
    for m in scope.input_memlets():
        assert m.volume == 0


def test_stree_keep_locals_forwarding():
    """Fix 2: ``keep_locals=True`` must be forwarded by overriding scopes and return the raw union."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N)}
    loop = dace.sdfg.state.LoopRegion("L", condition_expr="i < N")
    scope = tn.WhileScope(
        loop=loop,
        children=[_tasklet({"a": "A[i]"}, {"c": "C[i]"}), _assign("i", "i + 1")],
    )
    _root(scope, containers=containers, symbols={"N": dace.int64, "i": dace.int64})

    # With keep_locals, no widening: the raw, un-propagated memlets are returned.
    assert _by_data(scope.input_memlets(keep_locals=True)) == {("A", "i", False)}
    assert _by_data(scope.output_memlets(keep_locals=True)) == {("C", "i", False)}


def test_stree_loop_body_locals_up_front():
    """Fix 3: a symbol assigned *after* its use in a loop body is still a local for the whole body."""
    N = dace.symbol("N")
    containers = {"B": _array(N), "C": _array(N)}
    loop = dace.sdfg.state.LoopRegion("L", condition_expr="i < N")
    # Read B[k] before k is (re)assigned: k is loop-carried and must be treated as local throughout.
    scope = tn.WhileScope(
        loop=loop,
        children=[_tasklet({"b": "B[k]"}, {"c": "C[i]"}), _assign("k", "k + 1"), _assign("i", "i + 1")],
    )
    _root(scope, containers=containers, symbols={"N": dace.int64, "i": dace.int64, "k": dace.int64})

    assert scope._body_locals() == {"i", "k"}
    assert _by_data(scope.input_memlets()) == {("B", "0:N", True)}
    # Without the up-front computation, B[k] would be reported verbatim (treating k as defined).
    assert _by_data(scope.input_memlets(keep_locals=True)) == {("B", "k", False)}


def test_stree_dyn_scope_copy_local():
    """A ``DynScopeCopyNode`` target is a local for the nodes that follow it in the scope."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N), "B": _array(N), "d": _array(1)}
    dscopy = tn.DynScopeCopyNode(target="d", memlet=Memlet("B[0]"))
    ifs = tn.IfScope(
        condition=CodeBlock("N > 1"),
        children=[dscopy, _tasklet({"a": "A[d]"}, {"c": "C[0]"})],
    )
    _root(ifs, containers=containers, symbols={"N": dace.int64, "d": dace.int64})

    # The read A[d] mentions the dynamic-scope local ``d`` and is widened to the full extent; the dscopy's own
    # read B[0] (a whole-container access would otherwise have ``subset=None``) is reported exactly.
    assert _by_data(ifs.input_memlets()) == {("A", "0:N", True), ("B", "0", False)}


def test_stree_map_scope_disallow_forwarding():
    """Fix 2: a ``MapScope`` overrides ``propagate`` but still forwards ``disallow_propagation``."""
    N = dace.symbol("N")
    containers = {"A": _array(N), "C": _array(N)}
    mapscope = tn.MapScope(
        node=dace.nodes.MapEntry(dace.nodes.Map("m", ["i"], subsets.Range([(0, N - 1, 1)]))),
        children=[_tasklet({"a": "A[i]"}, {"c": "C[i]"})],
    )
    _root(mapscope, containers=containers, symbols={"N": dace.int64})

    # Treat ``i`` as disallowed: the map range is still used as the propagation range, but ``i`` being an
    # extra local must not crash and must be forwarded through (no exception, memlets still produced).
    inputs = mapscope.input_memlets(disallow_propagation={"foo"})
    assert _by_data(inputs) == {("A", "0:N", False)}


def test_propagation_contiguous_tile_exact():
    """Fix 4 (quiz Q10): an un-clipped tiled access propagates to the exact strided image."""
    N, T = (dace.symbol(s) for s in ("N", "T"))
    m = propagate_subset([Memlet("A[i:i+T]")], _array(N), ["i"], _rng((0, N - 1, T)))
    begin, end, step = m.subset.ranges[0]
    assert begin == 0
    assert step == 1
    # Inclusive end is the last visited tile's end, expressed with the same ``ceiling`` form as the map size:
    # T*ceiling(N/T) - 1.
    expected_end = symbolic.pystr_to_symbolic("T*ceiling(N/T) - 1")
    assert symbolic.simplify(end - expected_end) == 0
    # Exact for both a dividing (N=12) and a non-dividing (N=10) extent with T=4: image is 0..11.
    assert end.subs({N: 12, T: 4}) == 11
    assert end.subs({N: 10, T: 4}) == 11
    # The over-run for symbolic N stays undecidable, so validation does not reject it.
    assert (symbolic.simplify(end - N >= 0)) is not sympy.true


def test_propagation_clipped_tile():
    """Fix 4: a clipped tiled access (as emitted by MapTiling for non-dividing sizes) propagates exactly."""
    N, T = (dace.symbol(s) for s in ("N", "T"))
    m = propagate_subset([Memlet("A[i:Min(i+T, N)]")], _array(N), ["i"], _rng((0, N - 1, T)))
    end = m.subset.ranges[0][1]
    # The clipped tile reaches the map's own end on the last iteration, so the exact propagated end is N-1.
    assert symbolic.simplify(end - (N - 1)) == 0
    assert end.subs({N: 12, T: 4}) == 11  # 0..11
    assert end.subs({N: 10, T: 4}) == 9  # 0..9, clipped


def test_propagation_clipped_tile_non_matching():
    """Fix 4: a clipped access whose clip is not the map end keeps the conservative ``Min`` result."""
    N, T = (dace.symbol(s) for s in ("N", "T"))
    m = propagate_subset([Memlet("A[i:Min(i+T, 3)]")], _array(N), ["i"], _rng((0, N - 1, T)))
    # The clip (3) is not the map's end (N-1), so the dedicated case does not fire.
    assert isinstance(m.subset.ranges[0][1], sympy.Min)


def test_propagation_maptiling_clean_outer_memlets():
    """Fix 4: MapTiling with non-dividing tile sizes yields clean (unclipped) outer memlets."""
    from dace.transformation.dataflow import MapTiling

    M, K, Nn = (dace.symbol(s) for s in ("M", "K", "Nn"))

    @dace.program
    def matmul(A: dace.float64[M, K], B: dace.float64[K, Nn], C: dace.float64[M, Nn]):
        for i, j, k in dace.map[0:M, 0:Nn, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda x, y: x + y)[i, j]
                c = a * b

    sdfg = matmul.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(MapTiling, options=dict(tile_sizes=(2, 2, 3))) == 1
    sdfg.validate()
    outer = {}
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, nodes.MapEntry) and n.map.params[0].startswith("tile"):
                for e in state.in_edges(n):
                    if e.data.data:
                        outer[e.data.data] = e.data.subset
    # The outer reads cover the whole arrays with no ``Min`` clipping left in the subset.
    assert str(outer["A"]) == "0:M, 0:K"
    assert str(outer["B"]) == "0:K, 0:Nn"


if __name__ == "__main__":
    test_stree_propagation_forloop()
    test_stree_propagation_symassign()
    test_stree_propagation_dynset()
    test_stree_ifscope_gathers_children()
    test_stree_root_through_if_nonempty()
    test_stree_while_widens_loop_locals()
    test_stree_keep_locals_forwarding()
    test_stree_loop_body_locals_up_front()
    test_stree_dyn_scope_copy_local()
    test_stree_map_scope_disallow_forwarding()
    test_propagation_contiguous_tile_exact()
    test_propagation_clipped_tile()
    test_propagation_clipped_tile_non_matching()
    test_propagation_maptiling_clean_outer_memlets()
