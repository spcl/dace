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
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )

N = dace.symbol("N", nonnegative=True)


def _build_sdfg(v_dtype: dace.dtypes.typeclass = dace.float64) -> dace.SDFG:
    """``for i in range(N): v = a[i]; if True: b[i] = v + i + N`` with the arm's tasklet
    carrying the ``condition_symbol_to_scalar`` prefix the pass keys on.

    :param v_dtype: declared dtype of the demotable symbol ``v``.
    :returns: the constructed SDFG.
    """
    sdfg = dace.SDFG("conditional_symbol_demotion")
    sdfg.add_array("a", shape=(N, ), dtype=dace.float64)
    sdfg.add_array("b", shape=(N, ), dtype=dace.float64)
    sdfg.add_symbol("v", v_dtype)

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


@pytest.mark.parametrize("declared", [dace.int64, dace.int32, dace.float32, dace.float64])
def test_demotion_keeps_the_declared_dtype(declared):
    """The demoted scalar carries the symbol's OWN dtype, not a hardcoded fp64.

    The pass used to overwrite ``sdfg.symbols[v]`` with fp64 before demoting -- and that entry
    is exactly what :func:`~dace.sdfg.utils.demote_symbol_to_scalar` reads for the scalar's
    dtype, so every demoted symbol came out a double. An integer accumulator then stopped
    compiling the moment anything shifted or masked it::

        error: invalid operands of types 'double' and 'int' to binary 'operator>>'

    The rest of the graph already read the symbol at its declared dtype, so fp64 was the odd
    one out, not the safe default.
    """
    sdfg = _build_sdfg(v_dtype=declared)
    LowerInterstateConditionalAssignmentsToTasklets().apply_pass(sdfg, {})

    assert "v" in sdfg.arrays, "the data-bound symbol must still be demoted"
    assert sdfg.arrays["v"].dtype == declared, \
        f"demotion changed the dtype: declared {declared}, scalar is {sdfg.arrays['v'].dtype}"


def build_undeclared_arm_bound_sdfg() -> dace.SDFG:
    """A ConditionalBlock arm binds zlcrit on its own interstate edge; zlcrit is declared nowhere (CloudSC shape)."""
    sdfg = dace.SDFG("undeclared_arm_bound_symbol")
    sdfg.add_array("a", shape=(1, ), dtype=dace.float32)

    entry = sdfg.add_state("entry", is_start_block=True)
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())

    arm = ControlFlowRegion("arm", sdfg=sdfg)
    b0 = arm.add_state("b0", is_start_block=True)
    b1 = arm.add_state("b1")
    # zlcrit is bound HERE, on the arm's own interstate edge, from data -- never declared.
    arm.add_edge(b0, b1, dace.InterstateEdge(assignments={"zlcrit": "a[0]"}))
    cb.add_branch(CodeBlock("True"), arm)
    return sdfg


def test_undeclared_arm_bound_symbol_is_demoted_from_the_assignment_type():
    """A symbol bound only on an arm's own edge, absent from sdfg.symbols, must still demote."""
    sdfg = build_undeclared_arm_bound_sdfg()
    assert "zlcrit" not in sdfg.symbols, "zlcrit must start undeclared, the shape this test pins"

    demoted = LowerInterstateConditionalAssignmentsToTasklets().demote_arm_bound_symbols(sdfg)

    assert demoted == 1, f"expected exactly one arm-bound symbol demoted, got {demoted}"
    assert "zlcrit" in sdfg.arrays, "the undeclared arm-bound symbol must be demoted to a scalar"
    scalar = sdfg.arrays["zlcrit"]
    assert isinstance(scalar, dace.data.Scalar) and scalar.transient, \
        "the demoted symbol must be a transient Scalar"
    assert scalar.dtype == dace.float32, \
        f"dtype must come from the assignment (a is float32), got {scalar.dtype}"
    assert "zlcrit" not in sdfg.symbols, "the demoted name must no longer be a symbol"
    sdfg.validate()
