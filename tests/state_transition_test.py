# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import re
import numpy as np
import pytest


def test_state_transitions():
    if not dace.config.Config.get_bool('optimizer', 'detect_control_flow'):
        pytest.skip("Control flow not enabled. Skipping test.")

    sdfg = dace.SDFG("Transitions")
    sdfg.add_array("x", (1, ), int)

    start = sdfg.add_state("start")
    left = sdfg.add_state("left")
    right = sdfg.add_state("right")
    end = sdfg.add_state("end")

    left_cond = dace.properties.CodeProperty.from_string("0 < 1", language=dace.dtypes.Language.Python)

    right_cond = dace.properties.CodeProperty.from_string("0 >= 1", language=dace.dtypes.Language.Python)

    sdfg.add_edge(start, left, dace.InterstateEdge(condition=left_cond))
    sdfg.add_edge(start, right, dace.InterstateEdge(condition=right_cond))

    s0 = sdfg.add_state("s0")

    sdfg.add_edge(left, s0, dace.InterstateEdge())
    sdfg.add_edge(right, end, dace.InterstateEdge())

    s1_for_enter = sdfg.add_state("s1_for_enter")
    s1_for_body = sdfg.add_state("s1_for_body")
    x0_for = s1_for_body.add_access("x")
    x1_for = s1_for_body.add_access("x")
    tasklet_for = s1_for_body.add_tasklet("Update_x", {"x_in"}, {"x_out"}, "x_out = x_in + 1")
    s1_for_body.add_edge(x0_for, None, tasklet_for, "x_in", dace.memlet.Memlet.simple(x0_for, "0"))
    s1_for_body.add_edge(tasklet_for, "x_out", x1_for, None, dace.memlet.Memlet.simple(x1_for, "0"))

    s2 = sdfg.add_state("s2")

    for_assignment = dace.InterstateEdge(assignments={"i": 0})
    sdfg.add_edge(s0, s1_for_enter, for_assignment)

    for_entry = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string("i < 16", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s1_for_enter, s1_for_body, for_entry)

    for_continue = dace.InterstateEdge(assignments={"i": "i + 1"})
    sdfg.add_edge(s1_for_body, s1_for_enter, for_continue)

    for_exit = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string("i >= 16", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s1_for_enter, s2, for_exit)

    s3_while_enter = sdfg.add_state("s3_while_enter")
    s3_while_body = sdfg.add_state("s3_while_body")
    x0_while = s3_while_body.add_access("x")
    x1_while = s3_while_body.add_access("x")
    tasklet_while = s3_while_body.add_tasklet("Update_x", {"x_in"}, {"x_out"}, "x_out = x_in * 2; i *= 2")
    s3_while_body.add_edge(x0_while, None, tasklet_while, "x_in", dace.memlet.Memlet.simple(x0_while, "0"))
    s3_while_body.add_edge(tasklet_while, "x_out", x1_while, None, dace.memlet.Memlet.simple(x1_while, "0"))

    s4 = sdfg.add_state("s4")

    while_enter = dace.InterstateEdge()
    sdfg.add_edge(s2, s3_while_enter, while_enter)

    while_entry = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string("i < 128", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s3_while_enter, s3_while_body, while_entry)

    while_continue = dace.InterstateEdge()
    sdfg.add_edge(s3_while_body, s3_while_enter, while_continue)

    while_exit = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string("i >= 128", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s3_while_enter, s4, while_exit)

    s5_then = sdfg.add_state("s5_then")
    s6_else = sdfg.add_state("s6_else")

    x1_else = s6_else.add_access("x")
    tasklet_else = s6_else.add_tasklet("Update_x", {}, {"x_out"}, "x_out = 42")
    s6_else.add_edge(tasklet_else, "x_out", x1_else, None, dace.memlet.Memlet.simple(x1_else, "0"))

    s7_then_then = sdfg.add_state("s7_then_then")

    s8_end = sdfg.add_state("s8_end")

    s9 = sdfg.add_state("s9")

    if_cond = dace.properties.CodeProperty.from_string("i < 512", language=dace.dtypes.Language.Python)
    nested_if_cond = dace.properties.CodeProperty.from_string("i < 256", language=dace.dtypes.Language.Python)

    sdfg.add_edge(s4, s5_then, dace.InterstateEdge(condition=if_cond))
    sdfg.add_edge(s4, s6_else, dace.InterstateEdge(condition=dace.frontend.python.astutils.negate_expr(if_cond)))

    sdfg.add_edge(s5_then, s7_then_then, dace.InterstateEdge(condition=nested_if_cond))
    sdfg.add_edge(s5_then, s8_end,
                  dace.InterstateEdge(condition=dace.frontend.python.astutils.negate_expr(nested_if_cond)))

    sdfg.add_edge(s7_then_then, s8_end, dace.InterstateEdge())

    sdfg.add_edge(s8_end, s9, dace.InterstateEdge())

    sdfg.add_edge(s6_else, s9, dace.InterstateEdge())

    sdfg.add_edge(s9, end, dace.InterstateEdge())

    code = sdfg.generate_code()[0].code

    for_pattern = r"for.*i\s*=\s*0.*i\s*<\s*16"
    assert re.search(for_pattern, code) is not None, "For loop not detected in state transitions"

    while_pattern = r"while.+i\s*<\s*128"
    assert re.search(while_pattern, code) is not None, "While loop not detected in state transitions"

    if_pattern = r"if.+i\s*<\s*512"
    assert re.search(if_pattern, code) is not None, "If not detected in state transitions"

    else_pattern = r"}\s*else\s*{"
    assert re.search(else_pattern, code) is not None, "Else not detected in state transitions"

    x_output = dace.ndarray([1], int)
    x_output[0] = 0
    sdfg(x=x_output)
    x_output = x_output[0]

    assert x_output == 128, f"Expected x = 128, got {x_output}"


def test_state_transition_array():
    """ Toplevel array usage in interstate edge """
    sdfg = dace.SDFG('sta_test')
    sdfg.add_array('inp', [1], dace.float32)
    sdfg.add_array('A', [1], dace.float32)

    s0 = sdfg.add_state()
    s1 = sdfg.add_state()
    s2 = sdfg.add_state()

    # Arrays
    inp = s0.add_access('inp')
    A = s0.add_access('A')
    t = s0.add_tasklet('seta', {'a'}, {'b'}, 'b = a')
    s0.add_edge(inp, None, t, 'a', dace.Memlet.from_array(inp.data, inp.desc(sdfg)))
    s0.add_edge(t, 'b', A, None, dace.Memlet.from_array(A.data, A.desc(sdfg)))

    A = s1.add_access('A')
    t = s1.add_tasklet('geta', {'a'}, {}, 'printf("ok %f\\n", a + 1)')
    s1.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))

    A = s2.add_access('A')
    t = s2.add_tasklet('geta', {'a'}, {}, 'printf("BAD %f\\n", a - 1)')
    s2.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))

    sdfg.add_edge(s0, s1, dace.InterstateEdge('A[0] > 3'))
    sdfg.add_edge(s0, s2, dace.InterstateEdge('A[0] <= 3'))

    input = np.ndarray([1], np.float32)
    input[0] = 10
    output = np.ndarray([1], np.float32)
    output[0] = 10

    sdfg(inp=input, A=output)


if __name__ == "__main__":
    test_state_transitions()
    test_state_transition_array()
