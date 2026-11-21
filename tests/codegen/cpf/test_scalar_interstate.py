# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An interstate edge that reads a Scalar must name the variable, not ``name[0]``.

``name[0]`` is the spelling of element 0 of an array. A Scalar is a plain C variable, so the subscript
does not compile (``invalid types 'int64_t[int]' for array subscript``). cp2k_density_matrix_trs4 hits
this after the second ``subgraph-fission`` cut (``doall235`` / ``state112``): the kernel
``extcall_doall189`` assigns ``sink_slice_4`` from ``__map_fusion_...___sc0_x0_3[0]``.
"""

import dace
from dace.codegen.cpf import render as render_sdfg

N = dace.symbol("N")


def scalar_index_sdfg() -> dace.SDFG:
    """``k = idx[0]`` on the edge into the state that stores ``out[k]``, with ``idx`` a Scalar."""
    sdfg = dace.SDFG("scalar_interstate")
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_scalar("n", dace.int64)
    sdfg.add_scalar("idx", dace.int64, transient=True)
    sdfg.add_array("out", [N], dace.float64)

    first = sdfg.add_state("set")
    tasklet = first.add_tasklet("copy_n", {"i"}, {"o"}, "o = i")
    first.add_edge(first.add_read("n"), None, tasklet, "i", dace.Memlet("n"))
    first.add_edge(tasklet, "o", first.add_write("idx"), None, dace.Memlet("idx[0]"))

    second = sdfg.add_state("store")
    store = second.add_tasklet("one", {}, {"o"}, "o = 1.0")
    second.add_edge(store, "o", second.add_write("out"), None, dace.Memlet("out[k]"))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": "idx[0]"}))
    return sdfg


def test_an_interstate_edge_reads_a_scalar_by_name():
    """``idx[0]`` on the edge is rendered as the variable ``idx``, and the unit compiles."""
    sdfg = scalar_index_sdfg()
    code = render_sdfg(sdfg, check_compiles=True).code
    assert "idx[0]" not in code
    assert "idx[" not in code


if __name__ == "__main__":
    test_an_interstate_edge_reads_a_scalar_by_name()
