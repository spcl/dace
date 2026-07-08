# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :func:`lower_reduction_wcr_in_body`.

The transformation resolves the reduction WCR that ``nest_state_subgraph`` duplicates onto
a body edge (``src -[wcr]-> acc``, ``acc`` a non-transient output connector): a tiled body
gets the tile-foldable ``acc = acc <op> src`` form; a step-1 postamble tail keeps the
per-iteration boundary WCR (the in-body copy is simply dropped).
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.lower_reduction_wcr import lower_reduction_wcr_in_body


def _body_with_reduction_wcr(op_lambda: str):
    """A one-state SDFG modelling a nested reduction body: ``t (+)= v`` writing the
    non-transient output ``acc`` via a scalar WCR from an AccessNode source."""
    sdfg = dace.SDFG("red_body")
    sdfg.add_array("v", [1], dace.float32, transient=True)
    sdfg.add_array("acc", [1], dace.float32)  # non-transient == output connector
    state = sdfg.add_state()
    src = state.add_access("v")
    dst = state.add_access("acc")
    state.add_edge(src, None, dst, None, dace.Memlet(data="acc", subset="0", wcr=op_lambda))
    return sdfg, state


def test_tiled_rewrites_to_augassign():
    """``tiled=True`` replaces the WCR edge with an ``__out = __in1 + __in2`` reduce-accum
    tasklet (accumulator read back), leaving no in-body WCR."""
    sdfg, state = _body_with_reduction_wcr("lambda x, y: (x + y)")
    assert lower_reduction_wcr_in_body(sdfg, tiled=True) == 1

    assert not any(e.data is not None and e.data.wcr is not None for e in state.edges()), "no in-body WCR may remain"
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1
    t = tasklets[0]
    assert set(t.in_connectors) == {"__in1", "__in2"} and set(t.out_connectors) == {"__out"}
    # ``_wcr_augassign_body`` parenthesizes the binop; ``_detect_augassign_reduction`` accepts it.
    assert t.code.as_string.strip() == "__out = (__in1 + __in2)"
    # acc is written plain by the tasklet AND read back to feed __in1.
    acc_writes = [e for e in state.edges() if isinstance(e.dst, nodes.AccessNode) and e.dst.data == "acc"]
    assert len(acc_writes) == 1 and acc_writes[0].data.wcr is None
    assert any(isinstance(e.src, nodes.AccessNode) and e.src.data == "acc" and e.dst_conn == "__in1"
               for e in state.edges()), "expected an accumulator read-back into __in1"


def test_tail_strips_wcr():
    """``tiled=False`` (a step-1 postamble tail) drops the in-body WCR without adding a
    tasklet -- the boundary WCR already sums per iteration."""
    sdfg, state = _body_with_reduction_wcr("lambda x, y: (x + y)")
    assert lower_reduction_wcr_in_body(sdfg, tiled=False) == 1
    assert not any(e.data is not None and e.data.wcr is not None for e in state.edges())
    assert not [n for n in state.nodes() if isinstance(n, nodes.Tasklet)], "tail must not add a fold tasklet"


def test_min_max_use_function_form():
    """``min`` / ``max`` WCRs lower to the call-form body ``__out = min(__in1, __in2)``."""
    for fn in ("min", "max"):
        sdfg, state = _body_with_reduction_wcr(f"lambda a, b: {fn}(a, b)")
        assert lower_reduction_wcr_in_body(sdfg, tiled=True) == 1
        t = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)][0]
        assert t.code.as_string.strip() == f"__out = {fn}(__in1, __in2)"


def test_transient_sink_untouched():
    """A WCR into a TRANSIENT sink is not a reduction output -- left alone."""
    sdfg = dace.SDFG("t")
    sdfg.add_array("v", [1], dace.float32, transient=True)
    sdfg.add_array("acc", [1], dace.float32, transient=True)  # transient sink
    state = sdfg.add_state()
    state.add_edge(state.add_access("v"), None, state.add_access("acc"), None,
                   dace.Memlet(data="acc", subset="0", wcr="lambda x, y: (x + y)"))
    assert lower_reduction_wcr_in_body(sdfg, tiled=True) == 0
    assert any(e.data is not None and e.data.wcr is not None for e in state.edges())


def test_nest_reduction_is_idempotent():
    """``NestInnermostMapBodyIntoNSDFG`` run twice on a reduction map is a no-op the second time
    and never re-buries the boundary WCR inside the body NSDFG."""
    from dace.transformation.interstate import LoopToMap
    from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
    from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
    from dace.transformation.passes.vectorization.utils.pass_invariants import no_wcr_inside_nested_sdfgs

    n_sym = dace.symbol("N")

    @dace.program
    def red(A: dace.float32[n_sym], out: dace.float32[1]):
        acc = dace.float32(0.0)
        for i in dace.map[0:n_sym]:
            acc += A[i]
        out[0] = acc

    sdfg = red.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(AugAssignToWCR)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    assert NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {}), "first run must nest"
    assert no_wcr_inside_nested_sdfgs(sdfg) is None, "no loose WCR may remain inside the body NSDFG"
    assert NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {}) is None, \
        "second run must be a no-op (idempotent)"
    assert no_wcr_inside_nested_sdfgs(sdfg) is None, "second run must not re-bury the boundary WCR"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
