# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements`` and the two things a clone must not lose: ordering edges and View bindings.

An empty memlet transfers nothing and constrains only execution order, so it names no connector; a
connector-name test silently drops it and the happens-before goes with it. A View owns no storage,
so a statement separated from the binding edge is unbound -- the split refuses instead.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.split_statements import SplitStatements


def _two_output_body() -> dace.SDFG:
    inner = dace.SDFG("body")
    for nm in ("a", "b", "c"):
        inner.add_array(nm, [1], dace.float64)
    ist = inner.add_state("ist")
    tb = ist.add_tasklet("tb", {"x"}, {"o"}, "o = x * 2.0")
    tc = ist.add_tasklet("tc", {"x"}, {"o"}, "o = x + 1.0")
    ar = ist.add_access("a")
    ist.add_edge(ar, None, tb, "x", dace.Memlet("a[0]"))
    ist.add_edge(ar, None, tc, "x", dace.Memlet("a[0]"))
    ist.add_edge(tb, "o", ist.add_access("b"), None, dace.Memlet("b[0]"))
    ist.add_edge(tc, "o", ist.add_access("c"), None, dace.Memlet("c[0]"))
    return inner


def _build() -> dace.SDFG:
    """`B = A*2` and `C = A+1` in one NestedSDFG, then `A = 99` in the SAME state.

    The overwriting tasklet is inserted FIRST, so it is the first source the codegen's topological
    walk reaches once the ordering edge is gone.
    """
    sdfg = dace.SDFG("split_ordering")
    for nm in ("A", "B", "C"):
        sdfg.add_array(nm, [1], dace.float64)
    st = sdfg.add_state("main")

    w = st.add_tasklet("w", {}, {"o"}, "o = 99.0")
    st.add_edge(w, "o", st.add_access("A"), None, dace.Memlet("A[0]"))

    nsdfg = st.add_nested_sdfg(_two_output_body(), dict.fromkeys(["a"]), dict.fromkeys(["b", "c"]))
    st.add_edge(st.add_access("A"), None, nsdfg, "a", dace.Memlet("A[0]"))
    st.add_edge(nsdfg, "b", st.add_access("B"), None, dace.Memlet("B[0]"))
    st.add_edge(nsdfg, "c", st.add_access("C"), None, dace.Memlet("C[0]"))

    st.add_nedge(nsdfg, w, dace.Memlet())  # WAR: the body reads A before w overwrites it
    sdfg.validate()
    return sdfg


def _run(sdfg: dace.SDFG) -> dict:
    args = {"A": np.array([3.0]), "B": np.zeros(1), "C": np.zeros(1)}
    sdfg(**args)
    return args


def test_the_ordering_edge_survives_the_split():
    sdfg = _build()
    oracle = _run(copy.deepcopy(sdfg))
    assert (oracle["B"][0], oracle["C"][0]) == (6.0, 4.0)

    assert SplitStatements().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    state = sdfg.states()[0]
    clones = [n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(clones) == 2, [type(n).__name__ for n in state.nodes()]
    ordering = [e for e in state.edges() if e.data.is_empty()]
    assert len(ordering) == 2, f"every clone must inherit the ordering edge, got {len(ordering)}"
    assert {e.src for e in ordering} == set(clones)

    got = _run(sdfg)
    assert got["B"][0] == oracle["B"][0] and got["C"][0] == oracle["C"][0], (got, oracle)


def test_a_body_that_goes_through_a_view_is_not_split():
    """A View has no storage of its own; the split refuses rather than separate it from its binding."""
    inner = dace.SDFG("view_body")
    inner.add_array("a", [2, 2], dace.float64)
    inner.add_array("b", [4], dace.float64)
    inner.add_array("c", [4], dace.float64)
    inner.add_view("av", [4], dace.float64)
    ist = inner.add_state("ist")
    av = ist.add_access("av")
    ist.add_edge(ist.add_access("a"), None, av, "views", dace.Memlet("a[0:2, 0:2]"))
    tb = ist.add_tasklet("tb", {"x"}, {"o"}, "o = x * 2.0")
    tc = ist.add_tasklet("tc", {"x"}, {"o"}, "o = x + 1.0")
    ist.add_edge(av, None, tb, "x", dace.Memlet("av[1]"))
    ist.add_edge(av, None, tc, "x", dace.Memlet("av[2]"))
    ist.add_edge(tb, "o", ist.add_access("b"), None, dace.Memlet("b[0]"))
    ist.add_edge(tc, "o", ist.add_access("c"), None, dace.Memlet("c[0]"))

    sdfg = dace.SDFG("split_view")
    for nm in ("A", "B", "C"):
        sdfg.add_array(nm, [4], dace.float64)
    st = sdfg.add_state("main")
    nsdfg = st.add_nested_sdfg(inner, dict.fromkeys(["a"]), dict.fromkeys(["b", "c"]))
    st.add_edge(st.add_access("A"), None, nsdfg, "a", dace.Memlet("A[0:4]"))
    st.add_edge(nsdfg, "b", st.add_access("B"), None, dace.Memlet("B[0]"))
    st.add_edge(nsdfg, "c", st.add_access("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    assert SplitStatements._independent_output_groups(st, nsdfg) is None
    assert SplitStatements().apply_pass(sdfg, {}) is None
    assert len([n for n in st.nodes() if isinstance(n, dace.nodes.NestedSDFG)]) == 1


if __name__ == "__main__":
    pytest.main([__file__])
