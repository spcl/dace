# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An access node fed only by an empty edge out of a code node stands for that node's side effect and stays."""

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination


def test_a_sink_after_a_side_effect_code_node_is_kept():
    sdfg = dace.SDFG("side_effect_sink")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_scalar("handle", dace.int32, transient=True)
    state = sdfg.add_state()
    effect = state.add_tasklet(
        "declare", {}, {}, 'printf("declared\\n");', language=dace.Language.CPP, side_effects=True
    )
    state.add_edge(effect, None, state.add_access("handle"), None, dace.Memlet())
    Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
    assert effect in state.nodes()
    assert any(n.data == "handle" for n in state.data_nodes())


if __name__ == "__main__":
    test_a_sink_after_a_side_effect_code_node_is_kept()
