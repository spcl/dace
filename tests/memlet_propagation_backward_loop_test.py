# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import nodes
from dace.sdfg import propagation


def make_backward_loop_nested_sdfg():
    """
    Builds a parent SDFG with one state that reads an array ``A`` (40 elements)
    and feeds it into a nested SDFG. The nested SDFG contains a backward
    ``LoopRegion`` (loop variable initialized to 39, condition ``i >= 0``,
    update ``i = i - 1``) whose body reads the array at index ``i``.
    """
    sdfg = dace.SDFG("memlet_propagation_backward_loop")
    sdfg.add_array("A", [40], dace.int64)
    state = sdfg.add_state()

    nsdfg = dace.SDFG("nested_backward_loop")
    nsdfg.using_explicit_control_flow = True
    nsdfg.add_array("a", [40], dace.int64)

    loop = dace.sdfg.state.LoopRegion(
        "backward_loop",
        condition_expr="i >= 0",
        loop_var="i",
        initialize_expr="i = 39",
        update_expr="i = i - 1",
        sdfg=nsdfg,
    )
    loop_state = loop.add_state("loop_state", is_start_block=True)
    read_a = loop_state.add_read("a")
    tasklet = loop_state.add_tasklet("consume", {"inp"}, {}, "")
    loop_state.add_edge(read_a, None, tasklet, "inp", dace.Memlet("a[i]"))
    nsdfg.add_node(loop, is_start_block=True)

    nsdfg_node = state.add_nested_sdfg(nsdfg, {"a"}, {}, symbol_mapping={})
    state.add_edge(state.add_read("A"), None, nsdfg_node, "a", dace.Memlet("A[0:40]"))

    return sdfg


def test_backward_loop_border_memlet_positive_step():
    """
    Memlet propagation through a backward ``LoopRegion`` must produce a
    positive-step subset on the nested SDFG's border edge.

    The loop reads ``a[i]`` for ``i`` in {39, 38, ..., 0}, so the propagated
    border memlet must describe the full index set {0..39} in the canonical
    forward form ``(0, 39, 1)``. Propagation must not emit the
    equivalent-but-non-canonical negative-step range ``(39, 0, -1)`` (DaCe
    ``Range`` stops are inclusive, so both tuples denote {0..39}): negative-step
    subsets are not handled consistently by downstream machinery such as
    ``Range.covers()``, ``Range.min_element()`` and ``Range.max_element()``.
    """
    sdfg = make_backward_loop_nested_sdfg()

    propagation.propagate_memlets_sdfg(sdfg)

    state = sdfg.nodes()[0]
    nsdfg_nodes = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(nsdfg_nodes) == 1
    in_edges = state.in_edges(nsdfg_nodes[0])
    assert len(in_edges) == 1
    border_memlet = in_edges[0].data
    assert border_memlet.data == "A"
    assert border_memlet.subset.ranges == [(0, 39, 1)], (
        f"Expected the canonical positive-step range (0, 39, 1) on the nested SDFG "
        f"border edge, got {border_memlet.subset.ranges}"
    )


if __name__ == "__main__":
    test_backward_loop_border_memlet_positive_step()
