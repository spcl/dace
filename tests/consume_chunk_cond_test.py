import dace as dp
import numpy as np

sdfg = dp.SDFG('fib_consume_cc')
state = sdfg.add_state('state')
nprocs = 4

# Arrays
initial_value = state.add_array('iv', [1], dp.int32)
stream = state.add_stream('S', dp.int32, transient=True, buffer_size=256)
stream_init = state.add_stream('S', dp.int32, transient=True)
stream_out = state.add_stream('S', dp.int32, transient=True)
output = state.add_array('res', [1], dp.float32)

# Consume and tasklet
consume_entry, consume_exit = state.add_consume('cons', ('p', str(nprocs)),
                                                'res[0] >= 44',
                                                chunksize=2)
tasklet = state.add_tasklet(
    'fibonacci', {'s'}, {'sout', 'val'}, """
for i in range(cons_numelems):
    if s[i] == 1:
        val = 1
    elif s[i] > 1:
        sout = s[i] - 1   # Recurse by pushing smaller values
        sout = s[i] - 2
""")

# Edges
state.add_nedge(initial_value, stream_init,
                dp.Memlet.from_array(stream_init.data, stream_init.desc(sdfg)))
e = state.add_edge(stream, None, consume_entry, 'IN_stream',
                   dp.Memlet.from_array(stream.data, stream.desc(sdfg)))

# FIXME: Due to how memlets and propagation work, force access to stream to
# use an array instead of a scalar.
e.data.allow_oob = True
memlet = dp.Memlet.simple(stream, '0:2')
memlet.allow_oob = True

state.add_edge(consume_entry, 'OUT_stream', tasklet, 's', memlet)
state.add_edge(tasklet, 'sout', consume_exit, 'IN_S',
               dp.Memlet.simple(stream_out, '0', num_accesses=-1))
state.add_edge(consume_exit, 'OUT_S', stream_out, None,
               dp.Memlet.simple(stream_out, '0', num_accesses=-1))
state.add_edge(
    tasklet, 'val', consume_exit, 'IN_V',
    dp.Memlet.simple(output, '0', wcr_str='lambda a,b: a+b', num_accesses=-1))
state.add_edge(
    consume_exit, 'OUT_V', output, None,
    dp.Memlet.simple(output, '0', wcr_str='lambda a,b: a+b', num_accesses=-1))

consume_exit.add_in_connector('IN_S')
consume_exit.add_in_connector('IN_V')
consume_exit.add_out_connector('OUT_S')
consume_exit.add_out_connector('OUT_V')

sdfg.draw_to_file()

if __name__ == '__main__':
    print('Fibonacci recursion using consume (with chunks, custom condition)')
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = 10
    output[0] = 0
    regression = 44

    sdfg(iv=input, res=output)

    diff = output[0] - regression
    print('Difference:', diff)
    # Allowing for race conditions on quiescence condition
    exit(1 if (diff < 0 or diff > nprocs) else 0)
