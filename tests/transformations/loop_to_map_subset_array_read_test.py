# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` must pass into its nested SDFG an array that only a memlet SUBSET of the body reads.

Symbol propagation can fold an interstate assignment that reads an array into the memlets that use the
symbol: QE ``vexx_bp_k_gpu``'s ``jstart = max(jstart_0, (ijt - 1) * jblock + all_start(wegrp))`` ends up as
``rhoc_d[ir, jbnd - Max(jstart_0, ... + all_start[wegrp])]``. ``all_start`` is then read by no access node, so
the lift's read set (access nodes, headers and interstate edges) missed it: the nested SDFG named
``all_start`` as a free symbol nothing mapped, and validation failed with
``Missing symbols on nested SDFG: ['all_start']``.
"""

import numpy as np

import dace
from dace import subsets, symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """``for i: t = A[i + off[0]]; B[i] = 2 * t`` in two states, so the lift nests its body."""
    sdfg = dace.SDFG("loop_to_map_subset_array_read")
    sdfg.add_array("A", [2 * N], dace.float64)
    sdfg.add_array("off", [1], dace.int64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)

    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    load = loop.add_state("load", is_start_block=True)
    # The memlet string parser cannot hold a subscript in an index; the symbolic one can (as ``Subscript``).
    index = symbolic.pystr_to_symbolic("i + off[0]")
    tasklet = load.add_tasklet("load", {"__in"}, {"__out"}, "__out = __in")
    load.add_edge(
        load.add_read("A"), None, tasklet, "__in", dace.Memlet(data="A", subset=subsets.Range([(index, index, 1)]))
    )
    load.add_edge(tasklet, "__out", load.add_write("t"), None, dace.Memlet("t[0]"))
    store = loop.add_state_after(load, "store")
    tasklet = store.add_tasklet("store", {"__in"}, {"__out"}, "__out = 2 * __in")
    store.add_edge(store.add_read("t"), None, tasklet, "__in", dace.Memlet("t[0]"))
    store.add_edge(tasklet, "__out", store.add_write("B"), None, dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_an_array_a_body_subset_reads_is_passed_into_the_nest():
    sdfg = build_sdfg()
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1
    sdfg.validate()
    nested = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert nested and all("off" in n.in_connectors for n in nested), [n.in_connectors for n in nested]


def test_an_array_a_body_subset_reads_keeps_its_values():
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(LoopToMap)
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, off=np.array([3], dtype=np.int64), B=b, N=8)
    assert np.array_equal(b, 2 * a[3:11])


if __name__ == "__main__":
    test_an_array_a_body_subset_reads_is_passed_into_the_nest()
    test_an_array_a_body_subset_reads_keeps_its_values()
