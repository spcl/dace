# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np


def setup_sdfg() -> dp.SDFG:
    sdfg = dp.SDFG('fib_consume')
    sdfg.add_array('iv', [1], dp.int32)
    sdfg.add_stream('S', dp.int32, transient=True)
    sdfg.add_array('res', [1], dp.float32)

    # Arrays
    state = sdfg.add_state('state')
    initial_value = state.add_access('iv')
    stream = state.add_access('S')
    stream_init = state.add_access('S')
    stream_out = state.add_access('S')
    output = state.add_access('res')

    # Consume and tasklet
    consume_entry, consume_exit = state.add_consume('cons', ('p', '4'))
    tasklet = state.add_tasklet(
        'fibonacci', {'s'}, {'sout', 'val'}, """
if s == 1:
    val = 1
elif s > 1:
    sout = s - 1   # Recurse by pushing smaller values
    sout = s - 2
""")

    # Edges
    state.add_nedge(initial_value, stream_init, dp.Memlet.from_array(stream_init.data, stream_init.desc(sdfg)))
    state.add_edge(stream, None, consume_entry, 'IN_stream', dp.Memlet.from_array(stream.data, stream.desc(sdfg)))
    state.add_edge(consume_entry, 'OUT_stream', tasklet, 's', dp.Memlet.from_array(stream.data, stream.desc(sdfg)))
    state.add_edge(tasklet, 'sout', consume_exit, 'IN_S', dp.Memlet.simple(stream_out, '0', num_accesses=-1))
    state.add_edge(consume_exit, 'OUT_S', stream_out, None, dp.Memlet.simple(stream_out, '0', num_accesses=-1))
    state.add_edge(tasklet, 'val', consume_exit, 'IN_V',
                   dp.Memlet.simple(output, '0', wcr_str='lambda a,b: a+b', num_accesses=-1))
    state.add_edge(consume_exit, 'OUT_V', output, None,
                   dp.Memlet.simple(output, '0', wcr_str='lambda a,b: a+b', num_accesses=-1))

    consume_exit.add_in_connector('IN_S')
    consume_exit.add_in_connector('IN_V')
    consume_exit.add_out_connector('OUT_S')
    consume_exit.add_out_connector('OUT_V')

    return sdfg


def fibonacci(v):
    """ Computes the Fibonacci sequence at point v. """
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

    diff = (regression - output[0])**2
    assert diff <= 1e-5


if __name__ == '__main__':
    test_fibonacci_recursion_using_consume()
