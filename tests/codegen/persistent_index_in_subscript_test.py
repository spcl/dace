# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A persistent container read inside another access's index must be spelled as the generated code names it.

Codegen printed the index of ``A[i + off[0]]`` by name, so a persistent ``off`` (held as ``__state->__0_off``) came
out as a bare ``off`` nothing declares. QE ``vexx_bp_k_gpu`` after ``apply_gpu_transformations``: the host-staged
``ofsbeta_host`` inside ``deexx[.. + ofsbeta_host[k]]`` (an interstate condition) and inside a tasklet memlet's index
failed the build with "'ofsbeta_host' was not declared in this scope".
"""

import numpy as np

import dace
from dace import subsets, symbolic

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """``off_p = off`` (persistent); ``k = A[off_p[0]]`` on an edge; then ``B[i] = A[i + off_p[0]] + k``."""
    sdfg = dace.SDFG("persistent_index_in_subscript")
    sdfg.add_array("A", [2 * N], dace.float64)
    sdfg.add_array("off", [2], dace.int64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("off_p", [2], dace.int64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    sdfg.add_symbol("k", dace.float64)
    copy = sdfg.add_state("copy", is_start_block=True)
    copy.add_nedge(copy.add_read("off"), copy.add_write("off_p"), dace.Memlet("off[0:2]"))
    body = sdfg.add_state("body")
    sdfg.add_edge(copy, body, dace.InterstateEdge(assignments={"k": "A[off_p[0]]"}))
    index = symbolic.pystr_to_symbolic("i + off_p[0]")
    body.add_mapped_tasklet(
        "shift",
        {"i": "0:N"},
        {"__in": dace.Memlet(data="A", subset=subsets.Range([(index, index, 1)]))},
        "__out = __in + k",
        {"__out": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def test_a_persistent_index_container_compiles_and_reads():
    sdfg = build_sdfg()
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(8)
    sdfg(A=a, off=np.array([3, 0], dtype=np.int64), B=b, N=8)
    assert np.array_equal(b, a[3:11] + a[3])


if __name__ == "__main__":
    test_a_persistent_index_container_compiles_and_reads()
