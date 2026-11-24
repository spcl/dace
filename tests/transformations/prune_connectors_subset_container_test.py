# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PruneConnectors`` must keep an input connector that only a memlet SUBSET inside the nested SDFG reads.

``B[i] = A[i + off[0]]`` inside a nested SDFG reads ``off`` through a ``Subscript`` in ``A``'s subset. No access
node names ``off`` and ``Memlet.free_symbols`` leaves a ``Subscript``'s container out, so the connector looked
unused and was pruned: the nested SDFG then named ``off`` as a free symbol nothing maps (QE ``vexx_bp_k_gpu``:
``Missing symbols on nested SDFG: ['all_start']`` after canonicalization).
"""

import numpy as np

import dace
from dace import subsets, symbolic
from dace.transformation.dataflow import PruneConnectors

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """A map over ``i`` around a nested SDFG ``B[i] = A[i + off[0]]`` with ``A``, ``off`` and ``B`` connectors."""
    inner = dace.SDFG("gather")
    inner.add_array("A", [2 * N], dace.float64)
    inner.add_array("off", [1], dace.int64)
    inner.add_array("B", [N], dace.float64)
    body = inner.add_state("body")
    tasklet = body.add_tasklet("copy", {"__in"}, {"__out"}, "__out = __in")
    # The memlet string parser cannot hold a subscript in an index; the symbolic one can (as ``Subscript``).
    index = symbolic.pystr_to_symbolic("i + off[0]")
    body.add_edge(
        body.add_read("A"), None, tasklet, "__in", dace.Memlet(data="A", subset=subsets.Range([(index, index, 1)]))
    )
    body.add_edge(tasklet, "__out", body.add_write("B"), None, dace.Memlet("B[i]"))

    sdfg = dace.SDFG("prune_connectors_subset_container")
    sdfg.add_array("A", [2 * N], dace.float64)
    sdfg.add_array("off", [1], dace.int64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", {"i": "0:N"})
    node = state.add_nested_sdfg(inner, {"A", "off"}, {"B"}, symbol_mapping={"N": N, "i": "i"})
    for name in ("A", "off"):
        state.add_memlet_path(
            state.add_read(name), entry, node, dst_conn=name, memlet=dace.Memlet.from_array(name, sdfg.arrays[name])
        )
    state.add_memlet_path(node, exit_, state.add_write("B"), src_conn="B", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_a_connector_a_subset_reads_is_kept():
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(PruneConnectors)
    sdfg.validate()
    node = next(n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert "off" in node.in_connectors, node.in_connectors


def test_a_connector_a_subset_reads_keeps_its_values():
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(PruneConnectors)
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, off=np.array([3], dtype=np.int64), B=b, N=8)
    assert np.array_equal(b, a[3:11])


if __name__ == "__main__":
    test_a_connector_a_subset_reads_is_kept()
    test_a_connector_a_subset_reads_keeps_its_values()
