# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test():
    sdfg = dace.SDFG('toplevel_interstate_test')
    _, tmpdesc = sdfg.add_transient('tmp', [1], dace.int32)

    # State that sets tmp
    state = sdfg.add_state()
    tasklet = state.add_tasklet('settmp', {}, {'t'}, 't = 5')
    wtmp = state.add_write('tmp')
    state.add_edge(tasklet, 't', wtmp, None, dace.Memlet.from_array('tmp', tmpdesc))

    # States that uses tmp implicitly (only in interstate edge)
    state2 = sdfg.add_state()
    state2.add_tasklet('sayhi', {}, {}, 'printf("OK\\n")')
    state3 = sdfg.add_state()
    state3.add_tasklet('saybye', {}, {}, 'printf("FAIL\\n")')

    # Conditional edges that use tmp
    sdfg.add_edge(state, state2, dace.InterstateEdge('tmp[0] > 2'))
    sdfg.add_edge(state, state3, dace.InterstateEdge('tmp[0] <= 2'))


if __name__ == "__main__":
    test()
