# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LowerInterstateConditionalAssignmentsToTasklets`` must REFUSE undemotable symbols.

The pass turns a symbol read by a ``condition_symbol_to_scalar`` tasklet into an fp64
scalar so the condition becomes per-lane DATA. That is only expressible for a symbol the
SDFG itself defines and never evaluates symbolically. Two kinds must be left alone:

* an SDFG ARGUMENT (``N`` -- a shape symbol, so a member of ``free_symbols``): there is no
  definition inside the SDFG to rewrite into a scalar assignment, and
  :func:`~dace.sdfg.utils.demote_symbol_to_scalar` raises on it outright.
* a symbol the GRAPH evaluates (``i`` -- a loop variable that also indexes a memlet):
  demoting it leaves ``b[i]`` subscripting an array with an fp64 container, which the
  generated C++ rejects (``invalid types 'double*[double]'``).

Both are uniform across lanes, so the condition stays valid with them left symbols. The
over-refusal control is ``v``, bound by an interstate-edge assignment from data and read
only by the tasklet: it MUST still be demoted.
"""
import dace
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )

N = dace.symbol("N", nonnegative=True)


def _build_sdfg() -> dace.SDFG:
    """``for i in range(N): v = a[i]; if True: b[i] = v + i + N`` with the arm's tasklet
    carrying the ``condition_symbol_to_scalar`` prefix the pass keys on.

    :returns: the constructed SDFG.
    """
    sdfg = dace.SDFG("conditional_symbol_demotion")
    sdfg.add_array("a", shape=(N, ), dtype=dace.float64)
    sdfg.add_array("b", shape=(N, ), dtype=dace.float64)
    sdfg.add_symbol("v", dace.float64)

    loop = LoopRegion("loop", loop_var="i", initialize_expr="i = 0", condition_expr="i < N", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)

    # loop -> region -> conditional -> arm, the nesting the vectorizer leaves behind and the
    # only one the pass descends through.
    body = ControlFlowRegion("body", sdfg=sdfg)
    loop.add_node(body, is_start_block=True)
    head = body.add_state("head", is_start_block=True)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=body)
    body.add_node(cb)
    # v is bound from data here -- the one symbol that is demotable.
    body.add_edge(head, cb, dace.InterstateEdge(assignments={"v": "a[i]"}))

    arm = ControlFlowRegion("arm", sdfg=sdfg)
    st = arm.add_state("arm_state", is_start_block=True)
    tl = st.add_tasklet("condition_symbol_to_scalar_0", {}, {"_o"}, "_o = v + i + N")
    st.add_edge(tl, "_o", st.add_access("b"), None, dace.Memlet("b[i]"))
    cb.add_branch(CodeBlock("True"), arm)
    return sdfg


def test_argument_and_loop_symbols_are_not_demoted():
    """``N`` (argument) and ``i`` (loop variable / memlet index) survive as symbols; ``v``
    becomes a transient scalar wired into the tasklet."""
    sdfg = _build_sdfg()
    LowerInterstateConditionalAssignmentsToTasklets().apply_pass(sdfg, {})

    assert "N" not in sdfg.arrays, "an SDFG argument must not be demoted to a container"
    assert "N" in sdfg.free_symbols, "N must stay a free (argument) symbol"
    assert "i" not in sdfg.arrays, "a loop variable must not be demoted to a container"
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    assert loop.loop_variable == "i", "the loop variable must be untouched"

    assert "v" in sdfg.arrays, "the data-bound symbol must be demoted"
    demoted = sdfg.arrays["v"]
    assert isinstance(demoted, dace.data.Scalar) and demoted.transient, "the demoted symbol must be a transient Scalar"
    assert "v" not in sdfg.symbols, "the demoted symbol must no longer be a symbol"

    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    conds = [t for t in tasklets if t.label.startswith("condition_symbol_to_scalar")]
    assert len(conds) == 1, f"expected the one conditional-assignment tasklet; got {len(conds)}"
    assert "_in_v" in conds[0].in_connectors, "the demoted scalar must reach the tasklet as an in-connector"
    code = conds[0].code.as_string
    assert "N" in code and "i" in code, f"the refused symbols must stay symbolic in the tasklet: {code}"


def test_demotion_guards_classify_the_three_symbols():
    """The two guards the pass consults, read directly: ``N`` cannot become a transient
    scalar at all, ``i`` carries graph structure, ``v`` is clear on both counts."""
    sdfg = _build_sdfg()
    assert not sdutil.symbol_demotes_to_transient_scalar(sdfg, "N"), "an argument cannot demote to a transient"
    assert sdutil.symbol_carries_graph_structure(sdfg, "i"), "a loop variable indexing a memlet is structural"
    assert sdutil.symbol_demotes_to_transient_scalar(sdfg, "v")
    assert not sdutil.symbol_carries_graph_structure(sdfg, "v")
