# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Connectors built by expansions and transformations are ordered (dicts), never taken from a set."""

import warnings

import numpy as np

import dace
from dace.libraries import standard
from dace.transformation.interstate.loop_to_map import LoopToMap

N = 8


def test_reduce_pure_seq_expansion_builds_ordered_tasklet_connectors():
    sdfg = dace.SDFG("reduce_pure_seq_connectors")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state()
    node = standard.Reduce("sum", wcr="lambda a, b: a + b", axes=[1], identity=0)
    node.implementation = "pure-seq"
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, None, dace.Memlet("A[0:8, 0:8]"))
    state.add_edge(node, None, state.add_write("B"), None, dace.Memlet("B[0:8]"))

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Using sets for connectors")
        sdfg.expand_library_nodes()

    A = np.random.rand(N, N)
    B = np.zeros(N)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.sum(axis=1))


def test_loop_to_map_nested_sdfg_connectors_are_sorted():
    sdfg = dace.SDFG("loop_to_map_sorted_connectors")
    names = ["zz", "b", "mm", "a", "qq"]
    for name in names:
        sdfg.add_array(name, [N], dace.float64)
    loop = dace.sdfg.state.LoopRegion("loop", "i < 8", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    for src, dst in zip(names[:2], names[2:4]):
        body.add_mapped_tasklet(
            f"cp_{src}",
            {"k": "0:1"},
            {"v": dace.Memlet(f"{src}[i]")},
            "w = v",
            {"w": dace.Memlet(f"{dst}[i]")},
            external_edges=True,
        )
    body.add_mapped_tasklet(
        "cp_last", {"k": "0:1"}, {"v": dace.Memlet("zz[i]")}, "w = v", {"w": dace.Memlet("qq[i]")}, external_edges=True
    )

    assert sdfg.apply_transformations_repeated(LoopToMap) == 1

    nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert list(nsdfg.in_connectors) == sorted(nsdfg.in_connectors)
    assert list(nsdfg.out_connectors) == sorted(nsdfg.out_connectors)


if __name__ == "__main__":
    test_reduce_pure_seq_expansion_builds_ordered_tasklet_connectors()
    test_loop_to_map_nested_sdfg_connectors_are_sorted()
