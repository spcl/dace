# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import InterstateEdge
from dace.sdfg.nodes import MapEntry
from dace.transformation.auto.cloudsc_auto_opt import loop_to_map_outside_first
from dace.memlet import Memlet


def one_loop_test():
    """ Check that one simple loop gets converted """
    sdfg = dace.SDFG('change_strides_test')
    sdfg.add_array('A', [5], dace.float64)
    start = sdfg.add_state('start', is_start_state=True)
    guard = sdfg.add_state('guard', is_start_state=False)
    body = sdfg.add_state('body', is_start_state=False)
    loop_exit = sdfg.add_state('exit', is_start_state=False)

    tasklet = body.add_tasklet('work', {}, {'a'}, 'a = i')
    body.add_memlet_path(tasklet, body.add_write('A'), memlet=Memlet(data='A', subset='i'), src_conn='a')

    sdfg.add_edge(start, guard, InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, body, InterstateEdge(condition='i<5'))
    sdfg.add_edge(guard, loop_exit, InterstateEdge(condition='i>=5'))
    sdfg.add_edge(body, guard, InterstateEdge(assignments={'i': 'i+1'}))

    sdfg.validate()
    assert len(sdfg.states()) > 1
    loop_to_map_outside_first(sdfg)
    sdfg.simplify()
    # Check that a map was created
    assert len(sdfg.states()) == 1
    assert len([n for n in sdfg.start_state.nodes() if isinstance(n, MapEntry)]) == 1


def two_loop_test():
    """ Check that two simple nested loop gets converted """
    sdfg = dace.SDFG('change_strides_test')
    sdfg.add_array('A', [5, 10], dace.float64)
    start1 = sdfg.add_state('start1', is_start_state=True)
    guard1 = sdfg.add_state('guard1', is_start_state=False)
    loop_exit1 = sdfg.add_state('exit1', is_start_state=False)
    guard2 = sdfg.add_state('guard2', is_start_state=False)
    body = sdfg.add_state('body', is_start_state=False)

    tasklet = body.add_tasklet('work', {}, {'a'}, 'a = i*5 + j')
    body.add_memlet_path(tasklet, body.add_write('A'), memlet=Memlet(data='A', subset='i, j'), src_conn='a')

    sdfg.add_edge(start1, guard1, InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard1, guard2, InterstateEdge(condition='i<5', assignments={'j': 0}))
    sdfg.add_edge(guard1, loop_exit1, InterstateEdge(condition='i>=5'))
    sdfg.add_edge(guard2, body, InterstateEdge(condition='j<10'))
    sdfg.add_edge(guard2, guard1, InterstateEdge(condition='j>=10'))
    sdfg.add_edge(body, guard2, InterstateEdge(assignments={'j': 'j+1'}))

    sdfg.validate()
    # Adding a simplify here leads to simplify faling
    sdfg.view()
    assert len(sdfg.states()) > 1
    # Loop to map does not work correctly
    loop_to_map_outside_first(sdfg)
    sdfg.view()
    # sdfg.simplify()
    # Check that a map was created
    assert len(sdfg.states()) == 1
    assert len([n for n in sdfg.start_state.nodes() if isinstance(n, MapEntry)]) == 1
    sdfg.view()


def main():
    one_loop_test()
    two_loop_test()


if __name__ == '__main__':
    main()
