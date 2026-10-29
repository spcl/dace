# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` keeps an ``if/else`` whose arms bind a symbol a later block reads (triangle_count).

Split into ``if c`` then ``if not c``, each binding sits under its own conditional, so the nested SDFG reads the
symbol as possibly undefined and the parent fails validation with "Missing symbols on nested SDFG".
"""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


def symbol_bound_in_both_arms() -> dace.SDFG:
    """``if c > 0: n = 3 else: n = 5`` (no data written in either arm), then ``B[0:n] = A[0:n]``, nested."""
    inner = dace.SDFG("arms_bind_n")
    inner.add_array("A", [8], dace.float64)
    inner.add_array("B", [8], dace.float64)
    inner.add_symbol("c", dace.int64)
    inner.add_symbol("n", dace.int64)
    cb = ConditionalBlock("pick_n", sdfg=inner, parent=inner)
    inner.add_node(cb, is_start_block=True)
    for cond, value in (("c > 0", 3), (None, 5)):
        arm = ControlFlowRegion(f"arm_{value}", sdfg=inner, parent=cb)
        first = arm.add_state(f"enter_{value}", is_start_block=True)
        arm.add_edge(first, arm.add_state(f"bound_{value}"), dace.InterstateEdge(assignments={"n": str(value)}))
        cb.add_branch(CodeBlock(cond) if cond is not None else None, arm)
    copy = inner.add_state("copy")
    inner.add_edge(cb, copy, dace.InterstateEdge())
    copy.add_mapped_tasklet(
        "cp", {"j": "0:n"}, {"a": dace.Memlet("A[j]")}, "b = a", {"b": dace.Memlet("B[j]")}, external_edges=True
    )

    sdfg = dace.SDFG("arms_bind_n_outer")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_symbol("c", dace.int64)
    state = sdfg.add_state()
    node = state.add_nested_sdfg(inner, {"A"}, {"B"}, symbol_mapping={"c": "c"})
    state.add_edge(state.add_access("A"), None, node, "A", dace.Memlet("A[0:8]"))
    state.add_edge(node, "B", state.add_access("B"), None, dace.Memlet("B[0:8]"))
    return sdfg


def test_if_else_binding_a_symbol_read_after_stays_valid():
    sdfg = symbol_bound_in_both_arms()
    sdfg.validate()

    BranchNormalization().apply_pass(sdfg, {})

    sdfg.validate()
    a = np.arange(1, 9, dtype=np.float64)
    for c, n in ((1, 3), (0, 5)):
        b = np.zeros(8)
        sdfg(A=a, B=b, c=c)
        assert np.array_equal(b[:n], a[:n]) and not b[n:].any(), (c, b)


if __name__ == "__main__":
    test_if_else_binding_a_symbol_read_after_stays_valid()
    print("ok")
