# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ordering edges on a nested SDFG: dropped when they order no conflicting access, rerouted through the
nest's data otherwise, and kept when neither is safe -- after which ``InlineSDFG`` can take the nest."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.prune_and_inline_nested_sdfgs import PruneAndInlineNestedSDFGs
from dace.transformation.passes.canonicalize.route_ordering_through_data import RouteOrderingThroughData

N = dace.symbol("N")


def add_one_nest(state, name, source, target):
    """A nested SDFG computing ``target = source + 1`` element-wise over ``N``."""
    inner = dace.SDFG(name)
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dace.float64)
    inner.add_state("body", is_start_block=True).add_mapped_tasklet(
        "add_one",
        {"i": "0:N"},
        {"v": dace.Memlet("x[i]")},
        "w = v + 1.0",
        {"w": dace.Memlet("y[i]")},
        external_edges=True,
    )
    nest = state.add_nested_sdfg(inner, {"x"}, {"y"}, symbol_mapping={"N": N})
    state.add_edge(state.add_read(source), None, nest, "x", dace.Memlet(f"{source}[0:N]"))
    state.add_edge(nest, "y", state.add_write(target), None, dace.Memlet(f"{target}[0:N]"))
    return nest


def doubling(state, source, target):
    """A mapped ``target = 2 * source``, returning its source access node."""
    read = state.add_read(source)
    state.add_mapped_tasklet(
        "double",
        {"j": "0:N"},
        {"v": dace.Memlet(f"{source}[j]")},
        "w = 2.0 * v",
        {"w": dace.Memlet(f"{target}[j]")},
        external_edges=True,
        input_nodes={source: read},
    )
    return read


def nest_with_ordering_edge(name, conflicting):
    """``b = a + 1`` in a nest, then ``d = 2 * c``, with an ordering edge from the nest to ``c``'s access node.

    With ``conflicting`` the doubling reads AND writes ``a`` (``c`` is ``a``) -- a write after the nest's
    read, so the edge carries a real order; otherwise it touches nothing the nest touches.
    """
    sdfg = dace.SDFG(name)
    for array in ("a", "b", "c", "d"):
        sdfg.add_array(array, [N], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    nest = add_one_nest(state, f"{name}_nest", "a", "b")
    source = "a" if conflicting else "c"
    target = "a" if conflicting else "d"
    read = doubling(state, source, target)
    state.add_nedge(nest, read, dace.Memlet())
    sdfg.validate()
    return sdfg, state


def nests(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]


def test_ordering_edge_guarding_no_conflict_is_dropped_and_the_nest_inlines():
    """The doubling touches only ``c`` and ``d``, so nothing needs the nest before it: the edge goes and the
    nest is inlined."""
    sdfg, state = nest_with_ordering_edge("ordering_no_conflict", conflicting=False)
    assert RouteOrderingThroughData().apply_pass(sdfg, {}) == 1
    assert not any(e.data.is_empty() for e in state.edges() if isinstance(e.src, nodes.NestedSDFG))
    PruneAndInlineNestedSDFGs().apply_pass(sdfg, {})
    assert not nests(sdfg)
    a, b, c, d = np.arange(5.0), np.zeros(5), np.arange(5.0) + 3.0, np.zeros(5)
    sdfg(a=a, b=b, c=c, d=d, N=5)
    assert np.allclose(b, np.arange(5.0) + 1.0) and np.allclose(d, 2.0 * (np.arange(5.0) + 3.0))


def test_ordering_edge_guarding_a_write_after_read_is_kept_in_order():
    """The doubling overwrites ``a``, which the nest reads: whatever happens to the edge, the nest must
    still read ``a`` before the doubling writes it."""
    sdfg, state = nest_with_ordering_edge("ordering_war", conflicting=True)
    RouteOrderingThroughData().apply_pass(sdfg, {})
    nest = nests(sdfg)[0]
    writer = next(n for n in state.nodes() if isinstance(n, nodes.MapExit))
    assert dace.graphlib.has_path(state._nx, nest, writer), "the overwrite of a must stay after the nest's read"
    a, b, c, d = np.arange(5.0), np.zeros(5), np.zeros(5), np.zeros(5)
    sdfg(a=a, b=b, c=c, d=d, N=5)
    assert np.allclose(b, np.arange(5.0) + 1.0) and np.allclose(a, 2.0 * np.arange(5.0))


def test_ordering_edge_that_anchors_an_access_node_in_its_map_is_kept():
    """Inside a map, an access node read from outside data and reached only by the nest's ordering edge
    belongs to the map through that edge alone (velocity_tendencies after the GPU offload). Dropping it
    leaves the node at the top level still feeding a node inside the map, and every walk of the map body
    then counts the body twice."""
    sdfg = dace.SDFG("ordering_anchors_in_map")
    for array in ("x", "y", "c", "d"):
        sdfg.add_array(array, [N], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    inner = dace.SDFG("ordering_anchors_in_map_nest")
    inner.add_array("u", [N], dace.float64)
    inner.add_array("v", [N], dace.float64)
    inner.add_symbol("i", dace.int64)
    body = inner.add_state("body", is_start_block=True)
    plus = body.add_tasklet("plus_one", {"a"}, {"b"}, "b = a + 1.0")
    body.add_edge(body.add_read("u"), None, plus, "a", dace.Memlet("u[i]"))
    body.add_edge(plus, "b", body.add_write("v"), None, dace.Memlet("v[i]"))
    entry, exit_node = state.add_map("lanes", {"i": "0:N"})
    nest = state.add_nested_sdfg(inner, {"u"}, {"v"}, symbol_mapping={"N": N, "i": "i"})
    state.add_memlet_path(state.add_read("x"), entry, nest, dst_conn="u", memlet=dace.Memlet("x[0:N]"))
    state.add_memlet_path(nest, exit_node, state.add_write("y"), src_conn="v", memlet=dace.Memlet("y[0:N]"))
    anchored = state.add_access("c")
    state.add_nedge(nest, anchored, dace.Memlet())
    double = state.add_tasklet("double", {"a"}, {"b"}, "b = 2.0 * a")
    state.add_edge(anchored, None, double, "a", dace.Memlet("c[i]"))
    state.add_memlet_path(double, exit_node, state.add_write("d"), src_conn="b", memlet=dace.Memlet("d[i]"))
    sdfg.validate()
    assert state.scope_dict()[anchored] is entry

    RouteOrderingThroughData().apply_pass(sdfg, {})
    state._clear_scopedict_cache()
    assert state.scope_dict()[anchored] is entry, "the access node left its map"
    sdfg.validate()
    x, y, c, d = np.arange(5.0), np.zeros(5), np.arange(5.0) + 3.0, np.zeros(5)
    sdfg(x=x, y=y, c=c, d=d, N=5)
    assert np.allclose(y, x + 1.0) and np.allclose(d, 2.0 * c)


def test_nest_sharing_a_sink_gets_its_own_access_node_and_inlines():
    """The nest and a later map write the two halves of ONE ``out`` access node, and the copy ``b -> a``
    after the nest's read of ``a`` must follow the nest. Rerouting the order through the shared sink closes a
    cycle through the later map, so the nest gets an ``out`` access node of its own and then inlines."""
    sdfg = dace.SDFG("ordering_shared_sink")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("out", [2 * N], dace.float64)
    state = sdfg.add_state("main", is_start_block=True)
    # ``out[0:N] = a + 1`` in a nest whose outer memlet is the dynamic whole array, as MapToForLoop leaves it.
    inner = dace.SDFG("ordering_shared_sink_nest")
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [2 * N], dace.float64)
    inner.add_state("body", is_start_block=True).add_mapped_tasklet(
        "lower_half",
        {"i": "0:N"},
        {"v": dace.Memlet("x[i]")},
        "w = v + 1.0",
        {"w": dace.Memlet("y[i]")},
        external_edges=True,
    )
    nest = state.add_nested_sdfg(inner, {"x"}, {"y"}, symbol_mapping={"N": N})
    sink = state.add_write("out")
    state.add_edge(state.add_read("a"), None, nest, "x", dace.Memlet("a[0:N]"))
    state.add_edge(nest, "y", sink, None, dace.Memlet("out[0:2*N]", dynamic=True))
    read_b, write_a = state.add_read("b"), state.add_write("a")
    state.add_edge(read_b, None, write_a, None, dace.Memlet(data="b", subset="0:N", other_subset="0:N"))
    state.add_nedge(nest, read_b, dace.Memlet())
    state.add_mapped_tasklet(
        "upper_half",
        {"k": "0:N"},
        {"v": dace.Memlet("a[k]")},
        "w = v",
        {"w": dace.Memlet("out[k + N]")},
        external_edges=True,
        input_nodes={"a": write_a},
        output_nodes={"out": sink},
    )
    sdfg.validate()

    assert RouteOrderingThroughData().apply_pass(sdfg, {}) == 1
    assert not any(e.data.is_empty() for e in state.out_edges(nest))
    PruneAndInlineNestedSDFGs().apply_pass(sdfg, {})
    assert not nests(sdfg)
    sdfg.validate()
    a, b, out = np.arange(5.0), np.arange(5.0) + 10.0, np.zeros(10)
    sdfg(a=a, b=b, out=out, N=5)
    assert np.allclose(out[:5], np.arange(5.0) + 1.0) and np.allclose(out[5:], b) and np.allclose(a, b)


if __name__ == "__main__":
    test_ordering_edge_guarding_no_conflict_is_dropped_and_the_nest_inlines()
    test_ordering_edge_guarding_a_write_after_read_is_kept_in_order()
    test_ordering_edge_that_anchors_an_access_node_in_its_map_is_kept()
    test_nest_sharing_a_sink_gets_its_own_access_node_and_inlines()
