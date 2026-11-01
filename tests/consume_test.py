# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
import dace as dp


def setup_sdfg() -> dp.SDFG:
    sdfg = dp.SDFG("fib_consume")
    sdfg.add_array("iv", [1], dp.int32)
    sdfg.add_stream("S", dp.int32, transient=True)
    sdfg.add_array("res", [1], dp.float32)

    # Arrays
    state = sdfg.add_state("state")
    initial_value = state.add_access("iv")
    stream = state.add_access("S")
    stream_init = state.add_access("S")
    stream_out = state.add_access("S")
    output = state.add_access("res")

    # Consume and tasklet
    consume_entry, consume_exit = state.add_consume("cons", ("p", "4"))
    tasklet = state.add_tasklet(
        "fibonacci",
        {"s"},
        {"sout", "val"},
        """
if s == 1:
    val = 1
elif s > 1:
    sout = s - 1   # Recurse by pushing smaller values
    sout = s - 2
""",
    )

    # Edges
    state.add_nedge(initial_value, stream_init, dp.Memlet.from_array(stream_init.data, stream_init.desc(sdfg)))
    state.add_edge(stream, None, consume_entry, "IN_stream", dp.Memlet.from_array(stream.data, stream.desc(sdfg)))
    state.add_edge(consume_entry, "OUT_stream", tasklet, "s", dp.Memlet.from_array(stream.data, stream.desc(sdfg)))
    state.add_edge(tasklet, "sout", consume_exit, "IN_S", dp.Memlet.simple(stream_out, "0", num_accesses=-1))
    state.add_edge(consume_exit, "OUT_S", stream_out, None, dp.Memlet.simple(stream_out, "0", num_accesses=-1))
    state.add_edge(
        tasklet, "val", consume_exit, "IN_V", dp.Memlet.simple(output, "0", wcr_str="lambda a,b: a+b", num_accesses=-1)
    )
    state.add_edge(
        consume_exit, "OUT_V", output, None, dp.Memlet.simple(output, "0", wcr_str="lambda a,b: a+b", num_accesses=-1)
    )

    consume_exit.add_in_connector("IN_S")
    consume_exit.add_in_connector("IN_V")
    consume_exit.add_out_connector("OUT_S")
    consume_exit.add_out_connector("OUT_V")

    return sdfg


def fibonacci(v):
    """Computes the Fibonacci sequence at point v."""
    if v == 0:
        return 0
    if v == 1:
        return 1
    return fibonacci(v - 1) + fibonacci(v - 2)


def test_fibonacci_recursion_using_consume():
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = 10
    output[0] = 0
    regression = fibonacci(input[0])

    sdfg = setup_sdfg()
    sdfg(iv=input, res=output)

    diff = (regression - output[0]) ** 2
    assert diff <= 1e-5


@dace.program
def fib_program(iv: dace.int32[1], res: dace.float32[1]):
    """The same recursion written for the Python frontend: the consume body pushes onto ``S``."""
    S = dace.define_stream(dace.int32, 0)

    with dace.tasklet:
        i << iv
        s >> S
        s = i

    @dace.consume(S, 4)
    def scope(elem, p):
        sout >> S(-1)
        val >> res(-1, lambda a, b: a + b)

        if elem == 1:
            val = 1
        elif elem > 1:
            sout = elem - 1
            sout = elem - 2


def test_fibonacci_recursion_using_consume_program():
    """The consume body reads and writes one stream, so both of its connectors name one container."""
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = 10
    output[0] = 0
    regression = fibonacci(input[0])

    fib_program(input, output)

    diff = (regression - output[0]) ** 2
    assert diff <= 1e-5


@dace.program
def fib_loop(iv: dace.int32[1], res: dace.float32[1]):
    """The recursion as a consume loop, which pushes with ``append`` and adds with an augmented assignment."""
    S = dace.define_stream(dace.int32, 0)
    S.append(iv[0])
    for elem, p in dace.consume(S, 4):
        if elem == 1:
            res[0] += 1  # Processing elements add at once: a write-conflict resolution
        elif elem > 1:
            S.append(elem - 1)
            S.append(elem - 2)


def test_fibonacci_recursion_using_consume_loop():
    output = np.zeros([1], np.float32)
    fib_loop(np.array([10], np.int32), output)
    assert output[0] == fibonacci(10)


N = dace.symbol("N")


@dace.program
def doubled_sum(res: dace.int64[1]):
    """Pushes symbolic values from a map, and consumes them in a loop that does not use the processing element."""
    S = dace.define_stream(dace.int64, 0)
    for i in dace.map[0:N]:
        S.append(2 * i + 1)
    for elem, _ in dace.consume(S, 3):
        res[0] += elem


def test_consume_loop_of_elements():
    output = np.zeros([1], np.int64)
    doubled_sum(output, N=100)
    assert output[0] == 100**2


def test_consume_loop_needs_processing_element():
    """A consume loop names both the element and the processing element index."""

    @dace.program
    def elements_only(res: dace.int64[1]):
        S = dace.define_stream(dace.int64, 0)
        S.append(1)
        for elem in dace.consume(S, 2):
            res[0] += elem

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match="for x, pe in"):
        elements_only.to_sdfg()


if __name__ == "__main__":
    test_fibonacci_recursion_using_consume()
    test_fibonacci_recursion_using_consume_program()
    test_fibonacci_recursion_using_consume_loop()
    test_consume_loop_of_elements()
    test_consume_loop_needs_processing_element()
