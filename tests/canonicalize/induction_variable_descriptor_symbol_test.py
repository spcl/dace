# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InductionVariableSubstitution`` must not inline a derived symbol that a container's shape names.

``m := i + 1`` on the loop's edge is a derived symbol. Inlining it replaces its uses inside the loop and drops the
assignment, but a transient sized ``[m]`` is a descriptor of the SDFG, not a node of the loop: it kept naming ``m``,
which then nothing assigned. QE ``vexx_bp_k_gpu``'s ``jcurr = MIN(jend - jbnd + 1, many_fft)`` sizes the FFT view
``prhoc_d(1:nrxxs*jcurr)``: canonicalization left ``jcurr`` a free symbol the Fortran binding cannot supply.
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """``for i: m = i + 1; T[0:m] = 1; B[i] = sum(T[0:m])`` with ``T`` sized ``[m]``."""
    sdfg = dace.SDFG("induction_variable_descriptor_symbol")
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_array("T", ["m"], dace.float64, transient=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    head = loop.add_state("head", is_start_block=True)
    fill = loop.add_state("fill")
    loop.add_edge(head, fill, dace.InterstateEdge(assignments={"m": "i + 1"}))
    fill.add_mapped_tasklet(
        "ones", {"j": "0:m"}, {}, "__out = 1.0", {"__out": dace.Memlet("T[j]")}, external_edges=True
    )
    total = loop.add_state_after(fill, "total")
    t = total.add_read("T")
    tasklet = total.add_tasklet("count", {"__in"}, {"__out"}, "__out = __in[m - 1] * m")
    total.add_edge(t, None, tasklet, "__in", dace.Memlet("T[0:m]"))
    total.add_edge(tasklet, "__out", total.add_write("B"), None, dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_a_symbol_a_shape_names_stays_assigned():
    sdfg = build_sdfg()
    InductionVariableSubstitution().apply_pass(sdfg, {})
    sdfg.validate()
    assert "m" not in {str(s) for s in sdfg.free_symbols}, sdfg.free_symbols


def test_a_symbol_a_shape_names_keeps_its_values():
    sdfg = build_sdfg()
    InductionVariableSubstitution().apply_pass(sdfg, {})
    b = np.zeros(5)
    sdfg(B=b, N=5)
    assert np.array_equal(b, np.arange(1, 6, dtype=np.float64))


if __name__ == "__main__":
    test_a_symbol_a_shape_names_stays_assigned()
    test_a_symbol_a_shape_names_keeps_its_values()
