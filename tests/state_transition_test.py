#!/usr/bin/env python

import dace
import re
import sys

if __name__ == "__main__":

    if not dace.config.Config.get_bool('optimizer', 'detect_control_flow'):
        print("Control flow not enabled. Skipping test.")
        sys.exit(0)

    sdfg = dace.SDFG("Transitions")

    start = sdfg.add_state("start")
    left = sdfg.add_state("left")
    right = sdfg.add_state("right")
    end = sdfg.add_state("end")

    left_cond = dace.properties.CodeProperty.from_string(
        "0 < 1", language=dace.dtypes.Language.Python)

    right_cond = dace.properties.CodeProperty.from_string(
        "0 >= 1", language=dace.dtypes.Language.Python)

    sdfg.add_edge(start, left, dace.InterstateEdge(condition=left_cond))
    sdfg.add_edge(start, right, dace.InterstateEdge(condition=right_cond))

    s0 = sdfg.add_state("s0")

    sdfg.add_edge(left, s0, dace.InterstateEdge())
    sdfg.add_edge(right, end, dace.InterstateEdge())

    s1_for_enter = sdfg.add_state("s1_for_enter")
    s1_for_body = sdfg.add_state("s1_for_body")
    x0_for = s1_for_body.add_array("x", (1, ), int)
    x1_for = s1_for_body.add_array("x", (1, ), int)
    tasklet_for = s1_for_body.add_tasklet("Update_x", {"x_in"}, {"x_out"},
                                          "x_out = x_in + 1")
    s1_for_body.add_edge(x0_for, None, tasklet_for, "x_in",
                         dace.memlet.Memlet.simple(x0_for, "0"))
    s1_for_body.add_edge(tasklet_for, "x_out", x1_for, None,
                         dace.memlet.Memlet.simple(x1_for, "0"))

    s2 = sdfg.add_state("s2")

    for_assignment = dace.InterstateEdge(assignments={"i": 0})
    sdfg.add_edge(s0, s1_for_enter, for_assignment)

    for_entry = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string(
            "i < 16", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s1_for_enter, s1_for_body, for_entry)

    for_continue = dace.InterstateEdge(assignments={"i": "i + 1"})
    sdfg.add_edge(s1_for_body, s1_for_enter, for_continue)

    for_exit = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string(
            "i >= 16", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s1_for_enter, s2, for_exit)

    s3_while_enter = sdfg.add_state("s3_while_enter")
    s3_while_body = sdfg.add_state("s3_while_body")
    x0_while = s3_while_body.add_array("x", (1, ), int)
    x1_while = s3_while_body.add_array("x", (1, ), int)
    tasklet_while = s3_while_body.add_tasklet("Update_x", {"x_in"}, {"x_out"},
                                              "x_out = x_in * 2; i *= 2")
    s3_while_body.add_edge(x0_while, None, tasklet_while, "x_in",
                           dace.memlet.Memlet.simple(x0_while, "0"))
    s3_while_body.add_edge(tasklet_while, "x_out", x1_while, None,
                           dace.memlet.Memlet.simple(x1_while, "0"))

    s4 = sdfg.add_state("s4")

    while_enter = dace.InterstateEdge()
    sdfg.add_edge(s2, s3_while_enter, while_enter)

    while_entry = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string(
            "i < 128", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s3_while_enter, s3_while_body, while_entry)

    while_continue = dace.InterstateEdge()
    sdfg.add_edge(s3_while_body, s3_while_enter, while_continue)

    while_exit = dace.InterstateEdge(
        condition=dace.properties.CodeProperty.from_string(
            "i >= 128", language=dace.dtypes.Language.Python))
    sdfg.add_edge(s3_while_enter, s4, while_exit)

    sdfg.draw_to_file("sdfg.dot")

    s5_then = sdfg.add_state("s5_then")
    s6_else = sdfg.add_state("s6_else")

    x1_else = s6_else.add_array("x", (1, ), int)
    tasklet_else = s6_else.add_tasklet("Update_x", {}, {"x_out"}, "x_out = 42")
    s6_else.add_edge(tasklet_else, "x_out", x1_else, None,
                     dace.memlet.Memlet.simple(x1_else, "0"))

    s7_then_then = sdfg.add_state("s7_then_then")

    s8_end = sdfg.add_state("s8_end")

    s9 = sdfg.add_state("s9")

    if_cond = dace.properties.CodeProperty.from_string(
        "i < 512", language=dace.dtypes.Language.Python)
    nested_if_cond = dace.properties.CodeProperty.from_string(
        "i < 256", language=dace.dtypes.Language.Python)

    sdfg.add_edge(s4, s5_then, dace.InterstateEdge(condition=if_cond))
    sdfg.add_edge(
        s4, s6_else,
        dace.InterstateEdge(condition=dace.frontend.python.astutils.
                            negate_expr(if_cond['code_or_block'])))

    sdfg.add_edge(s5_then, s7_then_then,
                  dace.InterstateEdge(condition=nested_if_cond))
    sdfg.add_edge(
        s5_then, s8_end,
        dace.InterstateEdge(condition=dace.frontend.python.astutils.
                            negate_expr(nested_if_cond['code_or_block'])))

    sdfg.add_edge(s7_then_then, s8_end, dace.InterstateEdge())

    sdfg.add_edge(s8_end, s9, dace.InterstateEdge())

    sdfg.add_edge(s6_else, s9, dace.InterstateEdge())

    sdfg.add_edge(s9, end, dace.InterstateEdge())

    code = sdfg.generate_code()[0].code

    for_pattern = "for.*i\s*=\s*0.*i\s*<\s*16"
    if re.search(for_pattern, code) is None:
        raise RuntimeError("For loop not detected in state transitions")

    while_pattern = "while.+i\s*<\s*128"
    if re.search(while_pattern, code) is None:
        raise RuntimeError("While loop not detected in state transitions")

    if_pattern = "if.+i\s*<\s*512"
    if re.search(if_pattern, code) is None:
        raise RuntimeError("If not detected in state transitions")

    else_pattern = "}\s*else\s*{"
    if re.search(else_pattern, code) is None:
        raise RuntimeError("Else not detected in state transitions")

    x_output = dace.ndarray([1], dace.dtypes.int32)
    x_output[0] = 0
    sdfg(x=x_output)
    x_output = x_output[0]

    if x_output != 128:
        raise RuntimeError("Expected x = 128, got {}".format(x_output))
