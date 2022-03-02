# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np

sdfg = dp.SDFG('fib_consume')
state = sdfg.add_state('state')

# Arrays
initial_value = state.add_array('iv', [1], dp.int32)
stream = state.add_stream('S', dp.int32, transient=True)
stream_init = state.add_stream('S', dp.int32, transient=True)
stream_out = state.add_stream('S', dp.int32, transient=True)
output = state.add_array('res', [1], dp.float32)

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


def fibonacci(v):
    """ Computes the Fibonacci sequence at point v. """
    if v == 0:
        return 0
    if v == 1:
        return 1
    return fibonacci(v - 1) + fibonacci(v - 2)


def test():
    print('Fibonacci recursion using consume')
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = 10
    output[0] = 0
    regression = fibonacci(input[0])

    sdfg(iv=input, res=output)

    diff = (regression - output[0])**2
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test()
